import Foundation
import Testing
@testable import TUFFEngine
@testable import TUFFServerCore

private enum AdapterFixtureFailure: Error { case generic }

private actor AdapterBackend: ServerInferenceBackend {
    private(set) var requests: [ValidatedChatRequest] = []
    let tool: Bool
    let fail: Bool
    let trailing: String
    let genericFailure: Bool
    let reasoningOnly: Bool
    init(tool: Bool = false, fail: Bool = false, trailing: String = "", genericFailure: Bool = false, reasoningOnly: Bool = false) {
        self.tool = tool; self.fail = fail; self.trailing = trailing
        self.genericFailure = genericFailure; self.reasoningOnly = reasoningOnly
    }
    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        requests.append(request)
        if genericFailure { throw AdapterFixtureFailure.generic }
        if fail { throw ServerRequestError.invalid(message: "fixture failure", param: nil, code: "fixture") }
        onEvent(.reasoning("private reasoning"))
        if reasoningOnly {
            return ServerCompletion(content: "", reasoning: "private reasoning", toolCalls: [], finishReason: "length",
                usage: OpenAIUsage(promptTokens: 7, completionTokens: 5, totalTokens: 12, reasoningTokens: 5))
        }
        onEvent(.content("Hello"))
        onEvent(.content(" world"))
        let call = ParsedToolCall(id: "call_1", name: "lookup", arguments: .object(["q": .string("swift")]), argumentsJSON: #"{"q":"swift"}"#)
        if tool { onEvent(.toolCall(call)) }
        if !trailing.isEmpty { onEvent(.content(trailing)) }
        return ServerCompletion(content: "Hello world" + trailing, reasoning: "private reasoning", toolCalls: tool ? [call] : [], finishReason: tool ? "tool_calls" : "stop",
            usage: OpenAIUsage(promptTokens: 7, completionTokens: 5, totalTokens: 12, reasoningTokens: 2))
    }
}

@Suite(.serialized)
struct InferenceAPIAdapterTests {
    private func data(_ object: [String: Any]) throws -> Data { try JSONSerialization.data(withJSONObject: object) }
    private func send(_ path: String, _ object: [String: Any], port: Int) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try data(object)
        let (bytes, response) = try await URLSession.shared.data(for: request)
        return (bytes, try #require(response as? HTTPURLResponse))
    }
    private func messages(stream: Bool = false) -> [String: Any] {
        ["model": "fixture", "max_tokens": 32, "system": "Be concise", "messages": [["role": "user", "content": "Hello"]], "stream": stream]
    }
    private func responses(stream: Bool = false) -> [String: Any] {
        ["model": "fixture", "input": "Hello", "instructions": "Be concise", "max_output_tokens": 32, "store": false, "stream": stream]
    }

    @Test func textEndpointsShareInferenceAndReturnTheirOwnWireFormat() async throws {
        let backend = AdapterBackend()
        let server = TUFFHTTPServer(modelID: "fixture", queueLimit: 2, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            let (a, ar) = try await send("/v1/messages", messages(), port: port)
            let (b, br) = try await send("/v1/responses", responses(), port: port)
            #expect(ar.statusCode == 200 && br.statusCode == 200)
            let am = try #require(JSONSerialization.jsonObject(with: a) as? [String: Any])
            let bm = try #require(JSONSerialization.jsonObject(with: b) as? [String: Any])
            #expect(am["type"] as? String == "message")
            #expect(am["stop_reason"] as? String == "end_turn")
            #expect((am["content"] as? [[String: Any]])?.first?["text"] as? String == "Hello world")
            #expect(bm["object"] as? String == "response")
            #expect(bm["status"] as? String == "completed")
            #expect((bm["usage"] as? [String: Any])?["input_tokens"] as? Int == 7)
            #expect(am["tuff_timings_seconds"] != nil && bm["tuff_timings_seconds"] != nil)
            #expect(!String(decoding: a, as: UTF8.self).contains("private reasoning"))
            #expect(await backend.requests.count == 2)
            try await server.shutdown()
        } catch { try? await server.shutdown(); throw error }
    }

    @Test func streamingTextAndFunctionsHaveOrderedTypedEvents() async throws {
        let server = TUFFHTTPServer(modelID: "fixture", queueLimit: 2, backend: AdapterBackend(tool: true), chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            for (path, payload) in [("/v1/messages", messages(stream: true)), ("/v1/responses", responses(stream: true))] {
                let (bytes, response) = try await send(path, payload, port: port)
                #expect(response.statusCode == 200)
                #expect(response.value(forHTTPHeaderField: "content-type") == "text/event-stream")
                let text = String(decoding: bytes, as: UTF8.self)
                #expect(!text.contains("[DONE]") && !text.contains("private reasoning"))
                let events = try text.components(separatedBy: "\n\n").filter { $0.hasPrefix("event:") }.map { block -> [String: Any] in
                    let line = try #require(block.components(separatedBy: "\n").first { $0.hasPrefix("data: ") })
                    return try #require(JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any])
                }
                let types = events.compactMap { $0["type"] as? String }
                if path == "/v1/messages" {
                    #expect(types.first == "message_start" && types.last == "message_stop")
                    #expect(types.filter { $0 == "content_block_start" }.count == 2)
                    #expect(types.filter { $0 == "content_block_stop" }.count == 2)
                    #expect(text.contains("input_json_delta") && text.contains("tool_use"))
                    let delta = try #require(events.first { $0["type"] as? String == "message_delta" })
                    #expect((delta["usage"] as? [String: Any])?["input_tokens"] as? Int == 7)
                } else {
                    #expect(types.first == "response.created" && types.last == "response.completed")
                    #expect(types.contains("response.function_call_arguments.delta"))
                    #expect(types.contains("response.output_text.done"))
                    #expect(events.compactMap { $0["sequence_number"] as? Int } == Array(events.indices))
                    let final = try #require(events.last?["response"] as? [String: Any])
                    #expect((final["output"] as? [[String: Any]])?.count == 2)
                    #expect((final["usage"] as? [String: Any])?["output_tokens"] as? Int == 5)
                }
            }
            try await server.shutdown()
        } catch { try? await server.shutdown(); throw error }
    }

    @Test func bothToolResultTranscriptsSurviveTranslationAndValidation() async throws {
        let tool: [String: Any] = ["name": "lookup", "input_schema": ["type": "object", "properties": ["q": ["type": "string"]]]]
        let a: [String: Any] = ["model": "fixture", "max_tokens": 32, "tools": [tool], "messages": [
            ["role": "user", "content": "Look up Swift"],
            ["role": "assistant", "content": [["type": "tool_use", "id": "call_1", "name": "lookup", "input": ["q": "swift"]]]],
            ["role": "user", "content": [["type": "tool_result", "tool_use_id": "call_1", "content": "Found it"], ["type": "text", "text": "Summarize"]]]]]
        let b: [String: Any] = ["model": "fixture", "tools": [["type": "function", "name": "lookup", "parameters": tool["input_schema"]!]], "input": [
            ["role": "user", "content": "Look up Swift"],
            ["type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": #"{"q":"swift"}"#],
            ["type": "function_call_output", "call_id": "call_1", "output": "Found it"],
            ["role": "user", "content": [["type": "input_text", "text": "Summarize"]]]]]
        for (api, object) in [(InferenceAPI.messages, a), (.responses, b)] {
            let decoded = try api.request(data(object))
            #expect(decoded.messages.map(\.role) == ["user", "assistant", "tool", "user"])
            #expect(decoded.messages[2].toolCallID == "call_1")
            _ = try OpenAIRequestValidator.validate(decoded, modelID: "fixture", dialect: .gemma)
        }
        let backend = AdapterBackend()
        let server = TUFFHTTPServer(modelID: "fixture", queueLimit: 2, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            for (path, body) in [("/v1/messages", a), ("/v1/responses", b)] {
                let (_, response) = try await send(path, body, port: port)
                #expect(response.statusCode == 200)
            }
            #expect(await backend.requests.count == 2)
            try await server.shutdown()
        } catch { try? await server.shutdown(); throw error }
    }

    @Test func multipleResponsesCallsRemainOneAssistantTurn() throws {
        let body: [String: Any] = ["model": "fixture", "input": [
            ["role": "user", "content": "Look both up"],
            ["role": "assistant", "content": [["type": "output_text", "text": "I will look them up", "annotations": [], "logprobs": []]]],
            ["type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": #"{"q":"swift"}"#],
            ["type": "function_call", "call_id": "call_2", "name": "lookup", "arguments": #"{"q":"metal"}"#],
            ["type": "function_call_output", "call_id": "call_1", "output": "Swift result"],
            ["type": "function_call_output", "call_id": "call_2", "output": "Metal result"],
            ["role": "user", "content": "Summarize"]]]
        let request = try InferenceAPI.responses.request(data(body))
        #expect(request.messages.map(\.role) == ["user", "assistant", "tool", "tool", "user"])
        #expect(request.messages[1].toolCalls?.map(\.id) == ["call_1", "call_2"])
        #expect(request.messages[1].content == .text("I will look them up"))
        #expect(throws: ServerRequestError.self) { try InferenceAPI.responses.validateNativeHistory(request, dialect: .harmony) }
        _ = try OpenAIRequestValidator.validate(request, modelID: "fixture", dialect: .gemma)
    }

    /// Each route's output limit is refused under the name the client sent,
    /// and -1 means "no limit" only on Chat Completions.
    @Test func outputLimitsAreCheckedUnderTheirOwnNames() throws {
        func refusal(_ api: InferenceAPI, _ body: [String: Any]) throws -> (message: String, param: String?) {
            do {
                _ = try api.request(data(body))
            } catch ServerRequestError.invalid(let message, let param, _) {
                return (message, param)
            }
            Issue.record("expected a refusal")
            return ("", nil)
        }
        for value: Any in [1.5, -1, 0, "32", true] {
            var body = responses(); body["max_output_tokens"] = value
            #expect(try refusal(.responses, body).param == "max_output_tokens", "\(value)")
            var message = messages(); message["max_tokens"] = value
            #expect(try refusal(.messages, message).param == "max_tokens", "\(value)")
        }
        var fraction = responses(); fraction["max_output_tokens"] = 1.5
        #expect(try refusal(.responses, fraction).message == "max_output_tokens must be a whole number")
        var whole = responses(); whole["max_output_tokens"] = 64.0
        #expect(try InferenceAPI.responses.request(data(whole)).maxCompletionTokens == 64)
        var absent = responses(); absent["max_output_tokens"] = NSNull()
        #expect(try InferenceAPI.responses.request(data(absent)).maxCompletionTokens == nil)
        var topK = messages(); topK["top_k"] = 2.5
        #expect(try refusal(.messages, topK).param == "top_k")
    }

    @Test func numericFlagsDoNotBecomeBooleans() throws {
        for key in ["store", "background", "parallel_tool_calls"] {
            var body = responses(); body[key] = 0
            #expect(throws: (any Error).self) { try InferenceAPI.responses.request(data(body)) }
        }
        var strict = responses()
        strict["tools"] = [["type": "function", "name": "lookup", "parameters": ["type": "object"], "strict": 0]]
        #expect(throws: (any Error).self) { try InferenceAPI.responses.request(data(strict)) }
        var disabled = messages()
        disabled["tool_choice"] = ["type": "auto", "disable_parallel_tool_use": 0]
        #expect(throws: (any Error).self) { try InferenceAPI.messages.request(data(disabled)) }
        var result = messages()
        result["messages"] = [["role": "user", "content": [["type": "tool_result", "tool_use_id": "call_1", "content": "value", "is_error": 0]]]]
        #expect(throws: (any Error).self) { try InferenceAPI.messages.request(data(result)) }
    }

    @Test func unsupportedFeaturesAreRejectedBeforeInferenceWithCanonicalErrors() async throws {
        let backend = AdapterBackend()
        let server = TUFFHTTPServer(modelID: "fixture", queueLimit: 2, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            var payloads: [(String, [String: Any])] = []
            for (key, value) in [("store", true as Any), ("background", true), ("previous_response_id", "resp_old"), ("include", ["reasoning.encrypted_content"]), ("tools", [["type": "web_search"]])] {
                var body = responses(); body[key] = value; payloads.append(("/v1/responses", body))
            }
            var thinking = messages(); thinking["thinking"] = ["type": "enabled", "budget_tokens": 1000]; payloads.append(("/v1/messages", thinking))
            var image = messages(); image["messages"] = [["role": "user", "content": [["type": "image", "source": ["type": "url", "url": "https://example.com/image.png"]]]]]; payloads.append(("/v1/messages", image))
            for (path, body) in payloads {
                let (bytes, response) = try await send(path, body, port: port)
                #expect(response.statusCode == 400)
                let object = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
                #expect(object["error"] != nil)
                if path == "/v1/messages" { #expect(object["type"] as? String == "error") }
            }
            #expect(await backend.requests.isEmpty)
            try await server.shutdown()
        } catch { try? await server.shutdown(); throw error }
    }

    @Test func streamErrorsTerminateInTheRequestedProtocol() async throws {
        let server = TUFFHTTPServer(modelID: "fixture", queueLimit: 2, backend: AdapterBackend(fail: true))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            for (path, body) in [("/v1/messages", messages(stream: true)), ("/v1/responses", responses(stream: true))] {
                let (bytes, response) = try await send(path, body, port: port)
                #expect(response.statusCode == 200)
                let text = String(decoding: bytes, as: UTF8.self)
                #expect(text.contains("event: error\n"))
                #expect(!text.contains("[DONE]") && !text.contains("response.completed") && !text.contains("message_stop"))
            }
            try await server.shutdown()
        } catch { try? await server.shutdown(); throw error }
    }

    @Test func messagesStreamCanReplayAfterTrailingWhitespaceAndEmptyToolResult() async throws {
        let backend = AdapterBackend(tool: true, trailing: "\n \t")
        let server = TUFFHTTPServer(modelID: "fixture", queueLimit: 2, backend: backend, chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            let (bytes, response) = try await send("/v1/messages", messages(stream: true), port: port)
            #expect(response.statusCode == 200)
            var blocks: [[String: Any]] = []
            var arguments: [Int: String] = [:]
            for line in String(decoding: bytes, as: UTF8.self).components(separatedBy: "\n") where line.hasPrefix("data: ") {
                let event = try #require(JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any])
                if event["type"] as? String == "content_block_start" {
                    blocks.append(try #require(event["content_block"] as? [String: Any]))
                } else if event["type"] as? String == "content_block_delta" {
                    let index = try #require(event["index"] as? Int)
                    let delta = try #require(event["delta"] as? [String: Any])
                    if let text = delta["text"] as? String { blocks[index]["text"] = (blocks[index]["text"] as? String ?? "") + text }
                    if let json = delta["partial_json"] as? String { arguments[index, default: ""] += json }
                }
            }
            for (index, json) in arguments { blocks[index]["input"] = try JSONSerialization.jsonObject(with: Data(json.utf8)) }
            #expect(blocks.count == 2)
            #expect(blocks.last?["type"] as? String == "tool_use")
            var next = messages()
            next["messages"] = [["role": "user", "content": "Hello"], ["role": "assistant", "content": blocks],
                ["role": "user", "content": [["type": "tool_result", "tool_use_id": "call_1"]]]]
            let decoded = try InferenceAPI.messages.request(data(next))
            #expect(decoded.messages.last?.content == .text(""))
            _ = try OpenAIRequestValidator.validate(decoded, modelID: "fixture", dialect: .gemma)
            let (_, continuation) = try await send("/v1/messages", next, port: port)
            #expect(continuation.statusCode == 200)
            let (plainBytes, _) = try await send("/v1/messages", messages(), port: port)
            let plain = try #require(JSONSerialization.jsonObject(with: plainBytes) as? [String: Any])
            #expect((plain["content"] as? [[String: Any]])?.count == 2)
            #expect(await backend.requests.count == 3)
            try await server.shutdown()
        } catch { try? await server.shutdown(); throw error }
    }

    @Test func messagesSubstantivePostToolTextFailsInsteadOfReturningUnreplayableHistory() async throws {
        let server = TUFFHTTPServer(modelID: "fixture", queueLimit: 2, backend: AdapterBackend(tool: true, trailing: "This follows the tool"), chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            let (json, response) = try await send("/v1/messages", messages(), port: port)
            #expect(response.statusCode == 400)
            #expect(String(decoding: json, as: UTF8.self).contains("substantive assistant text after a tool call"))
            let (stream, streamingResponse) = try await send("/v1/messages", messages(stream: true), port: port)
            #expect(streamingResponse.statusCode == 200)
            let text = String(decoding: stream, as: UTF8.self)
            #expect(text.contains("event: error\n") && !text.contains("event: message_stop\n"))
            #expect(!text.contains("This follows the tool"))
            try await server.shutdown()
        } catch { try? await server.shutdown(); throw error }
    }

    @Test func hiddenReasoningDoesNotBecomeAdapterFirstVisibleEvent() async throws {
        let server = TUFFHTTPServer(modelID: "fixture", queueLimit: 2, backend: AdapterBackend(reasoningOnly: true))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            for (path, body, visible) in [("/v1/messages", messages(), false), ("/v1/responses", responses(), false),
                ("/v1/chat/completions", ["model": "fixture", "messages": [["role": "user", "content": "Hello"]]], true)] {
                let (bytes, response) = try await send(path, body, port: port)
                #expect(response.statusCode == 200)
                let object = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
                let timings = try #require(object["tuff_timings_seconds"] as? [String: Any])
                #expect((timings["generation_to_first_event"] != nil) == visible)
                #expect((timings["time_to_first_event"] != nil) == visible)
            }
            try await server.shutdown()
        } catch { try? await server.shutdown(); throw error }
    }

    @Test func messagesGenericStreamFailureUsesCanonicalAPIError() async throws {
        let server = TUFFHTTPServer(modelID: "fixture", queueLimit: 2, backend: AdapterBackend(genericFailure: true))
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            for stream in [false, true] {
                let (bytes, response) = try await send("/v1/messages", messages(stream: stream), port: port)
                #expect(response.statusCode == (stream ? 200 : 500))
                let text = String(decoding: bytes, as: UTF8.self)
                #expect(text.contains("api_error") && !text.contains("server_error"))
                if stream { #expect(text.contains("event: error\n")) }
            }
            try await server.shutdown()
        } catch { try? await server.shutdown(); throw error }
    }

    @Test func responsesPostToolTextIsRefusedBeforeUnreplayableOutputOrHistory() throws {
        let call = ParsedToolCall(id: "call_1", name: "lookup", arguments: .object(["q": .string("swift")]), argumentsJSON: #"{"q":"swift"}"#)
        let completion = ServerCompletion(content: "BeforeAfter", toolCalls: [call], finishReason: "tool_calls",
            usage: OpenAIUsage(promptTokens: 7, completionTokens: 5, totalTokens: 12))
        let wire = InferenceAdapterWire(api: .responses, id: "resp_fixture", model: "fixture", created: 0)
        for event in [ServerInferenceEvent.content("Before"), .toolCall(call)] { _ = wire.event(event) }
        #expect(wire.event(.content("\n \t")).isEmpty)
        try wire.validateCompletion()
        #expect(wire.event(.content("After")).isEmpty)
        #expect(throws: ServerRequestError.self) { try wire.validateCompletion() }
        #expect(throws: ServerRequestError.self) { try wire.completed(completion) }
        #expect(wire.finish(completion).last?.event == "error")
        let history: [[String: Any]] = [["role": "user", "content": "Hello"],
            ["type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"],
            ["role": "assistant", "content": "After"],
            ["type": "function_call_output", "call_id": "call_1", "output": "Result"]]
        #expect(throws: ServerRequestError.self) { try InferenceAPI.responses.request(data(["model": "fixture", "input": history])) }
        let messagesHistory: [[String: Any]] = [["role": "user", "content": "Hello"],
            ["role": "assistant", "content": [["type": "tool_use", "id": "call_1", "name": "lookup", "input": [:]]]],
            ["role": "assistant", "content": "After"],
            ["role": "user", "content": [["type": "tool_result", "tool_use_id": "call_1", "content": "Result"]]]]
        #expect(throws: ServerRequestError.self) {
            try InferenceAPI.messages.request(data(["model": "fixture", "max_tokens": 32, "messages": messagesHistory]))
        }
    }

    @Test func adapterToolOnlyHistoryRendersWithHarmonyAndKeepsTheResult() throws {
        let history: [[String: Any]] = [["role": "user", "content": "Hello"],
            ["type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"],
            ["type": "function_call_output", "call_id": "call_1", "output": "UNIQUE_RESULT_8491"]]
        let decoded = try InferenceAPI.responses.request(data(["model": "fixture", "input": history]))
        let validated = try OpenAIRequestValidator.validate(decoded, modelID: "fixture", dialect: .harmony)
        let rendered = try HarmonyPromptRenderer().render(messages: validated.messages, currentDate: "2026-10-06")
        #expect(rendered.contains("functions.lookup"))
        #expect(rendered.contains("UNIQUE_RESULT_8491"))
        let call = ParsedToolCall(id: "call_1", name: "lookup", arguments: .object([:]), argumentsJSON: "{}")
        let wire = InferenceAdapterWire(api: .responses, id: "resp_fixture", model: "fixture", created: 0, dialect: .harmony)
        _ = wire.event(.toolCall(call))
        _ = wire.event(.toolCall(call))
        #expect(throws: ServerRequestError.self) { try wire.validateCompletion() }
        var unsafe = validated.messages
        unsafe.insert(.init(role: .assistant, content: "After"), at: 2)
        #expect(throws: GFTokenizerError.self) {
            try HarmonyPromptRenderer().render(messages: unsafe, currentDate: "2026-10-06")
        }
    }

    @Test func gemmaMixedTextAndToolsAreRefusedAtAdmissionAndGeneration() async throws {
        let history: [[String: Any]] = [["role": "user", "content": "Hello"],
            ["role": "assistant", "content": "Before"],
            ["type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"],
            ["type": "function_call_output", "call_id": "call_1", "output": "Result"]]
        let payload: [String: Any] = ["model": "fixture", "input": history]
        let decoded = try InferenceAPI.responses.request(data(payload))
        #expect(throws: ServerRequestError.self) { try InferenceAPI.responses.validateNativeHistory(decoded, dialect: .gemma) }
        try InferenceAPI.responses.validateNativeHistory(decoded, dialect: .chatml)
        let backend = AdapterBackend(tool: true)
        let server = TUFFHTTPServer(modelID: "fixture", queueLimit: 2, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            let (_, rejected) = try await send("/v1/responses", payload, port: port)
            #expect(rejected.statusCode == 400)
            #expect(await backend.requests.isEmpty)
            for (path, body) in [("/v1/messages", messages()), ("/v1/responses", responses())] {
                let (bytes, response) = try await send(path, body, port: port)
                #expect(response.statusCode == 400)
                #expect(String(decoding: bytes, as: UTF8.self).contains("mixed assistant text and tool calls"))
            }
            try await server.shutdown()
        } catch { try? await server.shutdown(); throw error }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TUFF_TEST_GEMMA_TOKENIZER"] != nil))
    func installedGemmaTemplateKeepsAdmittedToolOnlyResult() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["TUFF_TEST_GEMMA_TOKENIZER"])
        let tokenizer = try await GFTokenizer.load(from: URL(fileURLWithPath: path))
        let payload: [String: Any] = ["model": "fixture", "input": [
            ["role": "user", "content": "Look up Swift"],
            ["type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"],
            ["type": "function_call_output", "call_id": "call_1", "output": "UNIQUE_RESULT_8491"],
            ["role": "user", "content": "Summarize"]]]
        let decoded = try InferenceAPI.responses.request(data(payload))
        try InferenceAPI.responses.validateNativeHistory(decoded, dialect: .gemma)
        let validated = try OpenAIRequestValidator.validate(decoded, modelID: "fixture", dialect: .gemma)
        let ids = try tokenizer.encodeToolChat(messages: validated.messages, tools: [])
        let rendered = tokenizer.decode(ids, skipSpecialTokens: false)
        #expect(rendered.contains("UNIQUE_RESULT_8491"))
        #expect(rendered.contains("lookup"))
        var unsafe = validated.messages
        unsafe.insert(.init(role: .assistant, content: "AFTER_UNSAFE_8491"), at: 2)
        let unsafeIDs = try tokenizer.encodeToolChat(messages: unsafe, tools: [])
        let unsafeRendered = tokenizer.decode(unsafeIDs, skipSpecialTokens: false)
        #expect(!unsafeRendered.contains("UNIQUE_RESULT_8491"))
    }

}
