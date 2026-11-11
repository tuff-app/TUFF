import Foundation
import CoreFoundation
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFModelMetadata

/// Text and JSON-function subsets. These adapters never execute tools, retain
/// responses by ID, or pretend to provide a hosted provider's model features.
enum InferenceAPI: String, Sendable {
    case chat, messages, responses

    static func forPath(_ path: String) -> Self? {
        switch path {
        case "/v1/chat/completions": .chat
        case "/v1/messages": .messages
        case "/v1/responses": .responses
        default: nil
        }
    }

    func request(_ data: Data) throws -> OpenAIChatRequest {
        if self == .chat { return try JSONDecoder().decode(OpenAIChatRequest.self, from: data) }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Self.invalid("request must be an object", nil)
        }
        let translated = try self == .messages ? messagesRequest(root) : responsesRequest(root)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self,
            from: JSONSerialization.data(withJSONObject: translated))
        try validateToolOrdering(request)
        return request
    }

    private func validateToolOrdering(_ request: OpenAIChatRequest) throws {
        var pending: Set<String> = []
        for message in request.messages {
            if message.role == "tool" {
                if let id = message.toolCallID { pending.remove(id) }
                continue
            }
            guard pending.isEmpty else {
                throw Self.invalid("tool outputs must immediately follow their assistant calls before another message", "messages")
            }
            for call in message.toolCalls ?? [] { pending.insert(call.id) }
        }
        guard pending.isEmpty else {
            throw Self.invalid("historical tool calls require their tool outputs", "messages")
        }
    }

    func validateNativeHistory(_ request: OpenAIChatRequest, dialect: ChatDialect) throws {
        guard self != .chat else { return }
        if dialect == .harmony, request.messages.contains(where: { ($0.toolCalls ?? []).count > 1 }) {
            throw Self.invalid("GPT-OSS's native template supports one tool call per assistant turn", "messages")
        }
        guard dialect == .gemma else { return }
        for message in request.messages where !(message.toolCalls ?? []).isEmpty {
            let content: String
            switch message.content {
            case .text(let value): content = value
            case .parts(let parts): content = parts.compactMap(\.text).joined()
            case nil: content = ""
            }
            guard content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw Self.invalid("Gemma's native template cannot preserve mixed assistant text and tool calls; use tool-only assistant turns", "messages")
            }
        }
    }

    static func invalid(_ message: String, _ param: String?) -> ServerRequestError {
        .invalid(message: message, param: param, code: "unsupported_parameter")
    }

    private func keys(_ value: [String: Any], _ allowed: Set<String>, _ path: String = "") throws {
        for key in value.keys.sorted() where !allowed.contains(key) && !(value[key] is NSNull) {
            throw Self.invalid("\(path)\(key) is not supported by TUFF's \(rawValue) adapter", path + key)
        }
    }

    private func boolean(_ value: Any, path: String) throws -> Bool {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
            throw Self.invalid("\(path) must be boolean", path)
        }
        return number.boolValue
    }

    private func text(_ value: Any?, types: Set<String>, path: String) throws -> String {
        if let value = value as? String { return value }
        guard let parts = value as? [[String: Any]] else {
            throw Self.invalid("\(path) must contain text", path)
        }
        return try parts.map { part in
            let isOutput = part["type"] as? String == "output_text" && types.contains("output_text")
            try keys(part, isOutput ? ["type", "text", "annotations", "logprobs"] : ["type", "text"], path + ".")
            for key in isOutput ? ["annotations", "logprobs"] : [] {
                if let value = part[key], !(value is NSNull), (value as? [Any])?.isEmpty != true {
                    throw Self.invalid("nonempty \(key) cannot be replayed", path + "." + key)
                }
            }
            guard let type = part["type"] as? String, types.contains(type), let text = part["text"] as? String else {
                throw Self.invalid("only text content is supported", path)
            }
            return text
        }.joined()
    }

    private func base(_ root: [String: Any]) throws -> [String: Any] {
        guard let model = root["model"] as? String, !model.isEmpty else {
            throw Self.invalid("model is required", "model")
        }
        var output: [String: Any] = ["model": model]
        for key in ["stream", "temperature", "top_p", "prompt_cache_key"] {
            if let value = root[key], !(value is NSNull) { output[key] = value }
        }
        return output
    }

    private func messagesRequest(_ root: [String: Any]) throws -> [String: Any] {
        try keys(root, ["model", "messages", "system", "max_tokens", "stream", "temperature", "top_p", "top_k", "stop_sequences", "tools", "tool_choice", "metadata", "thinking", "prompt_cache_key"])
        guard let maximum = root["max_tokens"] as? NSNumber,
              CFGetTypeID(maximum) != CFBooleanGetTypeID(), maximum.doubleValue.rounded() == maximum.doubleValue,
              maximum.intValue > 0 else { throw Self.invalid("positive max_tokens is required", "max_tokens") }
        if let thinking = root["thinking"] as? [String: Any] {
            try keys(thinking, ["type"], "thinking.")
            guard thinking["type"] as? String == "disabled" else {
                throw Self.invalid("signed Anthropic thinking is not supported", "thinking")
            }
        } else if root["thinking"] != nil && !(root["thinking"] is NSNull) {
            throw Self.invalid("thinking must be disabled", "thinking")
        }
        var output = try base(root)
        output["max_tokens"] = maximum
        output["top_k"] = root["top_k"]
        if let stops = root["stop_sequences"], !(stops is NSNull) {
            guard let values = stops as? [String], values.isEmpty else {
                throw Self.invalid("stop_sequences are not supported by the Messages adapter", "stop_sequences")
            }
        }
        var messages: [[String: Any]] = []
        if let system = root["system"], !(system is NSNull) {
            messages.append(["role": "system", "content": try text(system, types: ["text"], path: "system")])
        }
        guard let input = root["messages"] as? [[String: Any]] else { throw Self.invalid("messages is required", "messages") }
        for message in input {
            try keys(message, ["role", "content"], "messages.")
            guard let role = message["role"] as? String, ["user", "assistant"].contains(role) else {
                throw Self.invalid("message role must be user or assistant", "messages.role")
            }
            if let content = message["content"] as? String {
                messages.append(["role": role, "content": content]); continue
            }
            guard let blocks = message["content"] as? [[String: Any]] else { throw Self.invalid("invalid content", "messages.content") }
            var content = ""
            var calls: [[String: Any]] = []
            var results: [[String: Any]] = []
            var seenText = false
            var seenCall = false
            for block in blocks {
                switch block["type"] as? String {
                case "text":
                    let value = try text([block], types: ["text"], path: "messages.content")
                    if seenCall {
                        guard value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                            throw Self.invalid("text after tool_use cannot be represented exactly", "messages.content")
                        }
                        continue
                    }
                    seenText = true
                    content += value
                case "tool_use" where role == "assistant":
                    seenCall = true
                    try keys(block, ["type", "id", "name", "input"], "tool_use.")
                    guard let id = block["id"] as? String, let name = block["name"] as? String,
                          let arguments = block["input"] as? [String: Any] else { throw Self.invalid("invalid tool_use", "messages") }
                    let json = try JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys])
                    calls.append(["id": id, "type": "function", "function": ["name": name, "arguments": String(decoding: json, as: UTF8.self)]])
                case "tool_result" where role == "user":
                    guard !seenText else { throw Self.invalid("tool_result must precede user text", "messages.content") }
                    try keys(block, ["type", "tool_use_id", "content", "is_error"], "tool_result.")
                    guard let id = block["tool_use_id"] as? String else { throw Self.invalid("tool_use_id is required", "messages") }
                    let isError = try block["is_error"].map { try boolean($0, path: "tool_result.is_error") } ?? false
                    var result = try block["content"].map { try text($0, types: ["text"], path: "tool_result.content") } ?? ""
                    if isError { result = "Tool error: " + result }
                    results.append(["role": "tool", "tool_call_id": id, "content": result])
                default: throw Self.invalid("unsupported content block or role", "messages.content")
                }
            }
            // Results must immediately follow the assistant calls in the native
            // transcript. User text accompanying them comes after the results.
            messages.append(contentsOf: results)
            if !content.isEmpty || !calls.isEmpty || results.isEmpty {
                var value: [String: Any] = ["role": role, "content": content]
                if !calls.isEmpty { value["tool_calls"] = calls }
                messages.append(value)
            }
        }
        output["messages"] = messages
        if let tools = root["tools"] as? [[String: Any]] {
            output["tools"] = try tools.map { tool -> [String: Any] in
                try keys(tool, ["name", "description", "input_schema"], "tools.")
                guard let name = tool["name"] as? String, let schema = tool["input_schema"] as? [String: Any] else { throw Self.invalid("only JSON function tools are supported", "tools") }
                var function: [String: Any] = ["name": name, "parameters": schema]
                function["description"] = tool["description"]
                return ["type": "function", "function": function]
            }
        } else if root["tools"] != nil && !(root["tools"] is NSNull) { throw Self.invalid("tools must be an array", "tools") }
        if let choice = root["tool_choice"] as? [String: Any] {
            try keys(choice, ["type", "disable_parallel_tool_use"], "tool_choice.")
            guard let type = choice["type"] as? String, ["auto", "none"].contains(type) else { throw Self.invalid("only auto or none tool choice is supported", "tool_choice") }
            if let parallel = choice["disable_parallel_tool_use"], try boolean(parallel, path: "tool_choice.disable_parallel_tool_use") {
                throw Self.invalid("disabling parallel tool use is not supported", "tool_choice")
            }
            output["tool_choice"] = type
        } else if root["tool_choice"] != nil && !(root["tool_choice"] is NSNull) { throw Self.invalid("invalid tool_choice", "tool_choice") }
        return output
    }

    private func responsesRequest(_ root: [String: Any]) throws -> [String: Any] {
        try keys(root, ["model", "input", "instructions", "max_output_tokens", "stream", "temperature", "top_p", "tools", "tool_choice", "parallel_tool_calls", "store", "background", "metadata", "include", "text", "reasoning", "prompt_cache_key", "truncation"])
        for key in ["store", "background"] {
            if let value = root[key], !(value is NSNull), try boolean(value, path: key) {
                throw Self.invalid("\(key) must be false; TUFF uses stateless requests", key)
            }
        }
        if let include = root["include"], !(include is NSNull), (include as? [String])?.isEmpty != true { throw Self.invalid("include expansions are not supported", "include") }
        if let truncation = root["truncation"], !(truncation is NSNull), truncation as? String != "disabled" { throw Self.invalid("automatic truncation is not supported", "truncation") }
        if let format = root["text"], !(format is NSNull) {
            guard let object = format as? [String: Any] else { throw Self.invalid("invalid text options", "text") }
            try keys(object, ["format"], "text.")
            if let value = object["format"] {
                guard let value = value as? [String: Any], value.count == 1, value["type"] as? String == "text" else { throw Self.invalid("only plain text output is supported", "text.format") }
            }
        }
        // Checked under the Responses names: once mapped, a bad value was
        // reported against max_completion_tokens, which the client never sent.
        if let limit = root["max_output_tokens"] as? NSNumber, limit.intValue < 1 {
            throw Self.invalid("max_output_tokens must be at least 1", "max_output_tokens")
        }
        if let input = root["input"] {
            let empty = (input as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ?? (input as? [Any])?.isEmpty ?? false
            if empty { throw Self.invalid("input must not be empty", "input") }
        }
        var output = try base(root)
        output["max_completion_tokens"] = root["max_output_tokens"]
        output["tool_choice"] = root["tool_choice"]
        if let parallel = root["parallel_tool_calls"], !(parallel is NSNull) {
            output["parallel_tool_calls"] = try boolean(parallel, path: "parallel_tool_calls")
        }
        if let reasoning = root["reasoning"], !(reasoning is NSNull) {
            guard let value = reasoning as? [String: Any] else { throw Self.invalid("invalid reasoning options", "reasoning") }
            try keys(value, ["effort"], "reasoning.")
            output["reasoning_effort"] = value["effort"]
        }
        var messages: [[String: Any]] = []
        if let instructions = root["instructions"], !(instructions is NSNull) {
            guard let text = instructions as? String else { throw Self.invalid("instructions must be text", "instructions") }
            messages.append(["role": "system", "content": text])
        }
        if let input = root["input"] as? String {
            messages.append(["role": "user", "content": input])
        } else if let items = root["input"] as? [[String: Any]] {
            for item in items {
                switch item["type"] as? String ?? "message" {
                case "message":
                    try keys(item, ["type", "role", "content", "id", "status"], "input.")
                    guard let role = item["role"] as? String, ["system", "developer", "user", "assistant"].contains(role) else { throw Self.invalid("invalid message role", "input.role") }
                    messages.append(["role": role, "content": try text(item["content"], types: ["input_text", "output_text"], path: "input.content")])
                case "function_call":
                    try keys(item, ["type", "id", "status", "call_id", "name", "arguments"], "input.")
                    guard let id = item["call_id"] as? String, let name = item["name"] as? String, let arguments = item["arguments"] as? String else { throw Self.invalid("invalid function_call", "input") }
                    let call: [String: Any] = ["id": id, "type": "function", "function": ["name": name, "arguments": arguments]]
                    if let last = messages.last, last["role"] as? String == "assistant" {
                        let index = messages.count - 1
                        var calls = last["tool_calls"] as? [[String: Any]] ?? []
                        calls.append(call)
                        messages[index]["tool_calls"] = calls
                    } else {
                        messages.append(["role": "assistant", "content": "", "tool_calls": [call]])
                    }
                case "function_call_output":
                    try keys(item, ["type", "id", "call_id", "output"], "input.")
                    guard let id = item["call_id"] as? String else { throw Self.invalid("call_id is required", "input") }
                    messages.append(["role": "tool", "tool_call_id": id, "content": try text(item["output"], types: ["input_text"], path: "input.output")])
                default: throw Self.invalid("only text messages and JSON function calls are supported", "input")
                }
            }
        } else { throw Self.invalid("input must be text or an array of input items", "input") }
        output["messages"] = messages
        if let tools = root["tools"] as? [[String: Any]] {
            output["tools"] = try tools.map { tool -> [String: Any] in
                try keys(tool, ["type", "name", "description", "parameters", "strict"], "tools.")
                guard tool["type"] as? String == "function", let name = tool["name"] as? String,
                      let schema = tool["parameters"] as? [String: Any] else { throw Self.invalid("only JSON function tools are supported", "tools") }
                if let strict = tool["strict"], !(strict is NSNull), try boolean(strict, path: "tools.strict") {
                    throw Self.invalid("strict schema generation is not supported", "tools.strict")
                }
                var function: [String: Any] = ["name": name, "parameters": schema]
                function["description"] = tool["description"]
                return ["type": "function", "function": function]
            }
        } else if root["tools"] != nil && !(root["tools"] is NSNull) { throw Self.invalid("tools must be an array", "tools") }
        return output
    }
}

struct AdapterSSEFrame {
    let event: String
    let object: [String: Any]
}

/// State is accessed serially by the backend's callback. The lock also orders
/// queue-start and terminal/error events against callbacks during cancellation.
final class InferenceAdapterWire: @unchecked Sendable {
    let api: InferenceAPI
    let id: String
    let model: String
    let created: Int
    let dialect: ChatDialect
    private let lock = NSLock()
    private var sequence = 0
    private var output: [[String: Any]] = []
    private var activeText: (index: Int, id: String, text: String)?
    private var sawTool = false
    private var orderingFailure = false
    private var sawSubstantiveText = false
    private var mixedGemmaFailure = false
    private var multipleHarmonyFailure = false

    private var orderingError: OpenAIErrorEnvelope {
        OpenAIErrorEnvelope(message: mixedGemmaFailure
            ? "Gemma's native template cannot preserve mixed assistant text and tool calls"
            : multipleHarmonyFailure
                ? "GPT-OSS's native template supports one tool call per assistant turn"
                : "TUFF adapters cannot represent substantive assistant text after a tool call in the native transcript",
            param: "messages.content", code: "unsupported_content_order")
    }

    init(api: InferenceAPI, id: String, model: String, created: Int, dialect: ChatDialect = .chatml) {
        self.api = api; self.id = id; self.model = model; self.created = created; self.dialect = dialect
    }

    private func frame(_ type: String, _ fields: [String: Any] = [:]) -> AdapterSSEFrame {
        var object = fields
        object["type"] = type
        if api == .responses { object["sequence_number"] = sequence; sequence += 1 }
        return AdapterSSEFrame(event: type, object: object)
    }

    func start(promptTokens: Int = 0) -> [AdapterSSEFrame] {
        lock.lock(); defer { lock.unlock() }
        if api == .messages {
            return [frame("message_start", ["message": message(content: [], usage: OpenAIUsage(promptTokens: promptTokens, completionTokens: 0, totalTokens: promptTokens), reason: nil)])]
        }
        let response = response(output: [], completion: nil)
        return [frame("response.created", ["response": response]), frame("response.in_progress", ["response": response])]
    }

    func event(_ event: ServerInferenceEvent) -> [AdapterSSEFrame] {
        lock.lock(); defer { lock.unlock() }
        guard !orderingFailure else { return [] }
        switch event {
        case .reasoning: return [] // No forged Anthropic signatures or encrypted OpenAI state.
        case .content(let delta):
            guard !delta.isEmpty else { return [] }
            let substantive = !delta.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            if sawTool {
                // Native tool dialects commonly emit a trailing newline. It
                // carries no answer content and must not become a new block
                // that cannot be replayed. Substantive interleaving is refused.
                if substantive {
                    orderingFailure = true
                }
                return []
            }
            if substantive { sawSubstantiveText = true }
            var frames: [AdapterSSEFrame] = []
            if activeText == nil {
                let index = output.count
                let itemID = "msg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
                activeText = (index, itemID, "")
                output.append([:])
                if api == .messages { frames.append(frame("content_block_start", ["index": index, "content_block": ["type": "text", "text": ""]])) }
                else {
                    frames.append(frame("response.output_item.added", ["output_index": index, "item": textItem(id: itemID, text: "", status: "in_progress")]))
                    frames.append(frame("response.content_part.added", ["output_index": index, "item_id": itemID, "content_index": 0, "part": textPart("")]))
                }
            }
            activeText!.text += delta
            let active = activeText!
            if api == .messages { frames.append(frame("content_block_delta", ["index": active.index, "delta": ["type": "text_delta", "text": delta]])) }
            else { frames.append(frame("response.output_text.delta", ["output_index": active.index, "item_id": active.id, "content_index": 0, "delta": delta, "logprobs": []])) }
            return frames
        case .toolCall(let call):
            if dialect == .harmony, sawTool {
                multipleHarmonyFailure = true
                orderingFailure = true
                return []
            }
            if dialect == .gemma, sawSubstantiveText {
                mixedGemmaFailure = true
                orderingFailure = true
                return []
            }
            sawTool = true
            var frames = closeText()
            let index = output.count
            let itemID = "fc_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
            if api == .messages {
                let item = toolBlock(call)
                output.append(item)
                frames += [frame("content_block_start", ["index": index, "content_block": ["type": "tool_use", "id": call.id, "name": call.name, "input": [:]]]),
                    frame("content_block_delta", ["index": index, "delta": ["type": "input_json_delta", "partial_json": call.argumentsJSON]]),
                    frame("content_block_stop", ["index": index])]
            } else {
                let item = functionItem(call, id: itemID, status: "completed")
                output.append(item)
                var initial = item; initial["arguments"] = ""; initial["status"] = "in_progress"
                frames += [frame("response.output_item.added", ["output_index": index, "item": initial]),
                    frame("response.function_call_arguments.delta", ["output_index": index, "item_id": itemID, "delta": call.argumentsJSON]),
                    frame("response.function_call_arguments.done", ["output_index": index, "item_id": itemID, "arguments": call.argumentsJSON]),
                    frame("response.output_item.done", ["output_index": index, "item": item])]
            }
            return frames
        }
    }

    private func closeText() -> [AdapterSSEFrame] {
        guard let active = activeText else { return [] }
        activeText = nil
        if api == .messages {
            output[active.index] = ["type": "text", "text": active.text]
            return [frame("content_block_stop", ["index": active.index])]
        }
        let part = textPart(active.text)
        let item = textItem(id: active.id, text: active.text, status: "completed")
        output[active.index] = item
        return [frame("response.output_text.done", ["output_index": active.index, "item_id": active.id, "content_index": 0, "text": active.text, "logprobs": []]),
                frame("response.content_part.done", ["output_index": active.index, "item_id": active.id, "content_index": 0, "part": part]),
                frame("response.output_item.done", ["output_index": active.index, "item": item])]
    }

    func finish(_ completion: ServerCompletion) -> [AdapterSSEFrame] {
        lock.lock(); defer { lock.unlock() }
        if orderingFailure { return errorFrames(orderingError) }
        var frames = closeText()
        if api == .messages {
            frames += [frame("message_delta", ["delta": ["stop_reason": stopReason(completion), "stop_sequence": NSNull()], "usage": messagesUsage(completion.usage), "tuff_timings_seconds": completion.timingSeconds]), frame("message_stop")]
        } else {
            frames.append(frame(completion.finishReason == "length" ? "response.incomplete" : "response.completed", ["response": response(output: output, completion: completion)]))
        }
        return frames
    }

    func validateCompletion() throws {
        lock.lock(); defer { lock.unlock() }
        if orderingFailure {
            throw ServerRequestError.invalid(message: orderingError.error.message,
                param: orderingError.error.param, code: orderingError.error.code)
        }
    }

    func completed(_ completion: ServerCompletion) throws -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        if orderingFailure {
            throw ServerRequestError.invalid(message: orderingError.error.message,
                param: orderingError.error.param, code: orderingError.error.code)
        }
        var items: [[String: Any]] = []
        if !completion.content.isEmpty {
            items.append(api == .messages ? ["type": "text", "text": completion.content] : textItem(id: "msg_" + id, text: completion.content, status: "completed"))
        }
        for call in completion.toolCalls { items.append(api == .messages ? toolBlock(call) : functionItem(call, id: "fc_" + call.id, status: "completed")) }
        _ = closeText()
        // Preserve event order in JSON just as in SSE. Rebuilding from the
        // aggregate content would move text after a call ahead of that call.
        if !output.isEmpty { items = output }
        if api == .messages {
            var value = message(content: items, usage: completion.usage, reason: stopReason(completion))
            value["tuff_timings_seconds"] = completion.timingSeconds
            return value
        }
        return response(output: items, completion: completion)
    }

    func error(_ envelope: OpenAIErrorEnvelope) -> [AdapterSSEFrame] {
        lock.lock(); defer { lock.unlock() }
        return errorFrames(envelope)
    }

    private func errorFrames(_ envelope: OpenAIErrorEnvelope) -> [AdapterSSEFrame] {
        if api == .messages { return [frame("error", ["error": ["type": envelope.error.type == "server_error" ? "api_error" : envelope.error.type, "message": envelope.error.message]])] }
        return [frame("error", ["code": envelope.error.code, "message": envelope.error.message, "param": envelope.error.param as Any? ?? NSNull()])]
    }

    private func stopReason(_ completion: ServerCompletion) -> String {
        if !completion.toolCalls.isEmpty { return "tool_use" }
        return completion.finishReason == "length" ? "max_tokens" : "end_turn"
    }
    private func messagesUsage(_ usage: OpenAIUsage?) -> [String: Any] {
        if usage?.tokenCountsAvailable == false {
            return ["input_tokens": NSNull(), "output_tokens": NSNull(), "token_counts_available": false]
        }
        return ["input_tokens": usage?.promptTokens ?? 0, "output_tokens": usage?.completionTokens ?? 0]
    }
    private func message(content: [[String: Any]], usage: OpenAIUsage?, reason: String?) -> [String: Any] {
        ["id": id, "type": "message", "role": "assistant", "model": model, "content": content,
         "stop_reason": reason as Any? ?? NSNull(), "stop_sequence": NSNull(), "usage": messagesUsage(usage)]
    }
    private func textPart(_ text: String) -> [String: Any] { ["type": "output_text", "text": text, "annotations": [], "logprobs": []] }
    private func textItem(id: String, text: String, status: String) -> [String: Any] {
        ["type": "message", "id": id, "role": "assistant", "status": status, "content": [textPart(text)]]
    }
    private func toolBlock(_ call: ParsedToolCall) -> [String: Any] {
        ["type": "tool_use", "id": call.id, "name": call.name, "input": (try? JSONSerialization.jsonObject(with: Data(call.argumentsJSON.utf8))) ?? [:]]
    }
    private func functionItem(_ call: ParsedToolCall, id: String, status: String) -> [String: Any] {
        ["type": "function_call", "id": id, "call_id": call.id, "name": call.name, "arguments": call.argumentsJSON, "status": status]
    }
    private func response(output: [[String: Any]], completion: ServerCompletion?) -> [String: Any] {
        var value: [String: Any] = ["id": id, "object": "response", "created_at": created, "model": model,
            "status": completion == nil ? "in_progress" : (completion!.finishReason == "length" ? "incomplete" : "completed"),
            "output": output, "error": NSNull(), "incomplete_details": NSNull(), "usage": NSNull(), "store": false,
            "parallel_tool_calls": true]
        if let completion {
            let usage = completion.usage
            value["usage"] = ["input_tokens": usage.promptTokens, "input_tokens_details": ["cached_tokens": usage.promptTokensDetails.cachedTokens],
                "output_tokens": usage.completionTokens, "output_tokens_details": ["reasoning_tokens": usage.completionTokensDetails?.reasoningTokens ?? 0], "total_tokens": usage.totalTokens]
            if usage.tokenCountsAvailable == false { value["usage"] = NSNull() }
            if completion.finishReason == "length" { value["incomplete_details"] = ["reason": "max_output_tokens"] }
            value["tuff_timings_seconds"] = completion.timingSeconds
        }
        return value
    }
}
