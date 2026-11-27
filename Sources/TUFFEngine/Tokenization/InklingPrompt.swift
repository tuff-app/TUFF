import Foundation
import TUFFInferenceTypes

/// Inkling 2's chat format, as the publisher's `chat_template.jinja` writes it.
///
/// A conversation is a run of typed messages. Each opens with a role token,
/// then a content token that says what the body is, and closes with
/// `<|end_message|>`:
///
///     <|message_user|><|content_text|>Hello<|end_message|>
///     <|message_model|><|content_thinking|>…<|end_message|>
///     <|message_model|><|content_text|>Hi.<|end_message|><|content_model_end_sampling|>
///
/// An assistant turn can hold several messages (thinking, text, tool calls)
/// and ends with `<|content_model_end_sampling|>`, which is also where the
/// model stops. There is no generation prompt: the model opens its own turn
/// with `<|message_model|>`.
///
/// How hard to think is a number in a system message placed before the first
/// message that is not a system message. Tools are declared first of all, as
/// compact JSON with sorted keys.
enum InklingPrompt {
    static let userMark = "<|message_user|>"
    static let modelMark = "<|message_model|>"
    static let systemMark = "<|message_system|>"
    static let toolMark = "<|message_tool|>"
    static let textMark = "<|content_text|>"
    static let thinkingMark = "<|content_thinking|>"
    static let toolCallMark = "<|content_invoke_tool_json|>"
    static let endMessageMark = "<|end_message|>"
    static let endTurnMark = "<|content_model_end_sampling|>"

    /// The effort the template writes for each reasoning setting. Off is the
    /// template's "none"; on is the value it uses when nothing is asked for.
    static func effort(_ reasoning: ChatReasoning, level: ModelReasoningEffort? = nil) -> String {
        guard reasoning == .on else { return "0" }
        switch level ?? .high {
        case .low: return "0.2"
        case .medium: return "0.7"
        case .high: return "0.9"
        }
    }

    static func effortMessage(_ reasoning: ChatReasoning, level: ModelReasoningEffort?) -> String {
        systemMark + textMark + "Thinking effort level: " + effort(reasoning, level: level) + endMessageMark
    }

    /// Text from a user, a tool or a system prompt must not be able to close
    /// its own message or open another. The tokenizer turns any of these
    /// markers into its control token wherever it appears, so they are
    /// written out as text instead.
    static func safe(_ text: String) -> String {
        guard text.contains("<|") else { return text.replacingOccurrences(of: "\0", with: "") }
        var value = text.replacingOccurrences(of: "\0", with: "")
        value = value.replacingOccurrences(of: markerPattern, with: "&lt;|$1|&gt;", options: .regularExpression)
        return value
    }

    private static let markerPattern =
        "<\\|((?:message|content)_[a-z_]+|end_message|begin_of_text|endoftext|unused(?:_[0-9]+)?)\\|>"

    /// The same for serialized JSON: a tool's description, a schema or the
    /// arguments of an earlier call. A marker can only sit inside a JSON
    /// string, where its angle brackets are written as escapes, so the JSON
    /// still means what it meant.
    static func safeJSON(_ json: String) -> String {
        guard json.contains("<|") else { return json }
        return json.replacingOccurrences(of: markerPattern, with: "\\\\u003c|$1|\\\\u003e",
                                         options: .regularExpression)
    }

    static func userMessage(_ content: String) -> String {
        userMark + textMark + safe(content) + endMessageMark
    }

    static func render(messages: [InferenceChatMessage], tools: [InferenceFunctionDefinition],
                       reasoning: ChatReasoning, reasoningEffort: ModelReasoningEffort? = nil) throws -> String {
        var text = ""
        if !tools.isEmpty {
            let specs: [JSONValue] = tools.map { tool in
                .object(["description": .string(tool.description), "name": .string(tool.name),
                         "parameters": tool.parameters, "type": .string("function")])
            }
            text += systemMark + "tool_declare<|content_xml|>"
                + safeJSON(try JSONValue.array(specs).encoded()) + endMessageMark
        }
        var effortWritten = false
        for (index, message) in messages.enumerated() {
            let isSystem = message.role == .system || message.role == .developer
            if !effortWritten && !isSystem {
                text += effortMessage(reasoning, level: reasoningEffort)
                effortWritten = true
            }
            switch message.role {
            case .system, .developer:
                text += systemMark + textMark + safe(message.content ?? "") + endMessageMark
            case .user:
                text += userMessage(message.content ?? "")
            case .tool:
                // The header names the tool the result belongs to.
                let name = message.name ?? toolName(forCall: message.toolCallID, before: index, in: messages) ?? ""
                text += toolMark + safe(name) + textMark + safe(message.content ?? "") + endMessageMark
            case .assistant:
                // The model's own words are escaped too: an answer that
                // quotes a marker must not become one when it is read back.
                if let thinking = message.thinking, !thinking.isEmpty {
                    text += modelMark + thinkingMark + safe(thinking) + endMessageMark
                }
                let content = safe(message.content ?? "")
                if !content.isEmpty || (message.toolCalls.isEmpty && (message.thinking ?? "").isEmpty) {
                    text += modelMark + textMark + content + endMessageMark
                }
                for call in message.toolCalls {
                    guard case .object(let arguments) = call.arguments else {
                        throw GFTokenizerError.invalidChatTemplate("Inkling tool arguments must be an object")
                    }
                    let fields = try arguments.keys.sorted().map { key in
                        try JSONValue.string(key).encoded() + ":" + (try arguments[key]!.encoded())
                    }
                    text += modelMark + safe(call.name) + toolCallMark
                        + safeJSON("{\"name\":" + (try JSONValue.string(call.name).encoded())
                            + ",\"args\":{" + fields.joined(separator: ",") + "}}") + endMessageMark
                }
                text += endTurnMark
            }
        }
        if !effortWritten { text += effortMessage(reasoning, level: reasoningEffort) }
        return text
    }

    private static func toolName(forCall id: String?, before index: Int,
                                 in messages: [InferenceChatMessage]) -> String? {
        guard let id else { return nil }
        for message in messages[..<index].reversed() where message.role == .assistant {
            if let call = message.toolCalls.first(where: { $0.id == id }) { return call.name }
        }
        return nil
    }
}

/// Parses the body of an `<|content_invoke_tool_json|>` message:
/// `{"name":"web_search","args":{"query":"…"}}`.
struct InklingToolCallParser {
    func parse(_ text: String, allowedTools: Set<String>, id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= QwenToolCallParser.maximumBytes else { throw ToolCallParserError.oversized }
        guard let value = try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)),
              case .object(let fields) = value,
              case .string(let name)? = fields["name"],
              name.range(of: "^[A-Za-z0-9_.-]{1,64}$", options: .regularExpression) != nil else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else { throw ToolCallParserError.unknownTool(name) }
        // A call with no arguments may leave `args` out.
        let arguments = fields["args"] ?? fields["arguments"] ?? .object([:])
        guard case .object = arguments,
              Set(fields.keys).isSubset(of: ["name", "args", "arguments"]) else {
            throw ToolCallParserError.malformed
        }
        return ParsedToolCall(id: id, name: name, arguments: arguments,
                              argumentsJSON: try arguments.encoded())
    }
}

/// The tokens that structure an Inkling 2 assistant turn.
public struct InklingSpecialTokenIDs: Sendable, Equatable {
    public let model: Int32
    public let text: Int32
    public let thinking: Int32
    public let toolCall: Int32
    public let endMessage: Int32
    public let endTurn: Int32
    /// Content types TUFF has no special handling for; their bodies are shown
    /// as text.
    public let otherContent: Set<Int32>

    public var structuralMarkerIDs: Set<Int32> {
        otherContent.union([model, text, thinking, toolCall, endMessage, endTurn])
    }
}

/// Reads an Inkling 2 assistant turn as it streams: thinking messages become
/// reasoning, text messages become content, and each tool message is parsed
/// when it closes. Whatever the model writes between `<|message_model|>` and
/// the content token is the message's name, which is not shown.
final class InklingAssistantDecoder {
    private enum Phase { case idle, header, text, thinking, tool }

    private let tokenizer: GFTokenizer
    private let ids: InklingSpecialTokenIDs
    private let allowedTools: Set<String>
    private let idGenerator: @Sendable () -> String
    private var phase = Phase.idle
    private var toolTokens: [Int32] = []
    private var failed = false
    private(set) var emittedCalls = 0

    init(tokenizer: GFTokenizer, ids: InklingSpecialTokenIDs, allowedTools: Set<String>,
         idGenerator: @escaping @Sendable () -> String) {
        self.tokenizer = tokenizer
        self.ids = ids
        self.allowedTools = allowedTools
        self.idGenerator = idGenerator
    }

    var isInReasoning: Bool { phase == .thinking }
    var hasToolCalls: Bool { emittedCalls > 0 }

    private func route(_ delta: String) -> [StructuredAssistantEvent] {
        guard !delta.isEmpty else { return [] }
        switch phase {
        case .thinking: return [.thinking(delta)]
        case .text, .idle: return [.content(delta)]
        case .header, .tool: return []
        }
    }

    func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        if phase == .tool, tokenID != ids.endMessage, tokenID != ids.model, tokenID != ids.endTurn {
            toolTokens.append(tokenID)
            guard toolTokens.count * MemoryLayout<Int32>.size <= QwenToolCallParser.maximumBytes else {
                failed = true
                throw ToolCallParserError.oversized
            }
            return []
        }
        // Text the detokenizer held back belongs to the message that was open
        // before this token.
        var events = route(delta)
        switch tokenID {
        case ids.model:
            // A message left open is closed by the next one starting.
            events += try closeMessage()
            phase = .header
        case ids.text:
            phase = .text
        case ids.thinking:
            phase = .thinking
        case ids.toolCall:
            phase = .tool
            toolTokens = []
        case ids.endMessage, ids.endTurn:
            events += try closeMessage()
            phase = .idle
        default:
            if ids.otherContent.contains(tokenID) { phase = .text }
        }
        return events
    }

    private func closeMessage() throws -> [StructuredAssistantEvent] {
        guard phase == .tool else { return [] }
        let text = tokenizer.decode(toolTokens, skipSpecialTokens: false)
        toolTokens = []
        do {
            let call = try InklingToolCallParser().parse(text, allowedTools: allowedTools, id: idGenerator())
            emittedCalls += 1
            return [.toolCall(call)]
        } catch {
            failed = true
            throw error
        }
    }

    func consumeTail(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        return route(text)
    }

    func finish() throws {
        guard !failed, phase != .tool else { throw ToolCallParserError.malformed }
    }
}
