import Foundation

public enum StructuredAssistantEvent: Equatable, Sendable {
    case content(String)
    case thinking(String)
    case toolCall(ParsedToolCall)
}

public final class StructuredAssistantDecoder: @unchecked Sendable {
    private enum Channel {
        case thought
        case visible
        case label
    }

    private let tokenizer: GFTokenizer
    private let allowedTools: Set<String>
    private let parameterSchemas: [String: JSONValue]
    private let idGenerator: @Sendable () -> String
    private let harmonyDecoder: HarmonyAssistantDecoder?
    private var channel: Channel = .visible
    private var label = ""
    private var toolTokens: [Int32]?
    private var emittedCalls = 0
    private var failed = false

    /// `promptOpensThinking` says the prompt already contains the token that
    /// opens the thought channel, so the model will never emit it.
    ///
    /// This is the ChatML case with reasoning on: the generation prompt ends
    /// `<|im_start|>assistant\n<think>\n`, exactly as the upstream Jinja
    /// template writes it. The decoder started in `.visible`, saw no
    /// `<think>` — it was in the prompt, not the output — and streamed the
    /// model's entire reasoning as visible content until `</think>` arrived and
    /// switched it to a channel it was already in. Qwen3.6 answers therefore
    /// began with several hundred words of "Thinking Process: 1. Analyze the
    /// Request…" that belonged in the collapsed disclosure.
    ///
    /// Gemma names its channels inline and Harmony has its own decoder, so
    /// neither was affected, and neither passes this.
    public init(tokenizer: GFTokenizer,
                allowedTools: Set<String>,
                promptOpensThinking: Bool = false,
                parameterSchemas: [String: JSONValue] = [:],
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.tokenizer = tokenizer
        self.allowedTools = allowedTools
        self.parameterSchemas = parameterSchemas
        self.idGenerator = idGenerator
        self.harmonyDecoder = tokenizer.harmonyTokenIDs.map {
            HarmonyAssistantDecoder(
                tokens: $0,
                allowedTools: allowedTools,
                idGenerator: idGenerator)
        }
        if promptOpensThinking,
           tokenizer.dialect == .chatml || tokenizer.dialect == .minimax {
            channel = .thought
        }
    }

    /// Whether a prompt rendered for this tokenizer leaves the thought channel
    /// open for the model to continue in.
    ///
    /// Kept beside the decoder that has to honour it rather than at the call
    /// sites, so a third caller cannot construct the decoder without it.
    public static func promptOpensThinking(
        tokenizer: GFTokenizer,
        reasoning: ChatReasoning
    ) -> Bool {
        promptOpensThinking(dialect: tokenizer.dialect, reasoning: reasoning)
    }

    static func promptOpensThinking(
        dialect: ChatDialect,
        reasoning: ChatReasoning
    ) -> Bool {
        switch dialect {
        case .chatml:
            reasoning == .on
        case .minimax:
            true
        case .gemma, .harmony, .flint:
            false
        }
    }

    /// Generated tokens the decoder routed to reasoning: tokens consumed
    /// inside a thought or analysis channel, including the markers that open
    /// and close it. Reported as `reasoning_tokens`.
    public private(set) var reasoningTokenCount = 0

    /// Whether text arriving now belongs to reasoning.
    public var isInReasoning: Bool {
        if let harmonyDecoder { return harmonyDecoder.isInAnalysis }
        return channel == .thought
    }

    public func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        let wasReasoning = isInReasoning
        let events = try consumeToken(tokenID: tokenID, delta: delta)
        if wasReasoning || isInReasoning
            || events.contains(where: { if case .thinking = $0 { true } else { false } }) {
            reasoningTokenCount += 1
        }
        return events
    }

    private func consumeToken(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        if let harmonyDecoder {
            return try harmonyDecoder.consume(tokenID: tokenID, delta: delta)
        } else if tokenizer.dialect == .chatml || tokenizer.dialect == .minimax || tokenizer.dialect == .flint {
            return try consumeTaggedDialect(tokenID: tokenID, delta: delta)
        }

        // A non-empty delta on a control token is text the detokenizer held
        // back from BEFORE the token (a skipped special contributes nothing of
        // its own), so it belongs to the channel state in effect now — route
        // it before the token changes that state. Inside a tool call the held
        // bytes are part of the payload, which is re-decoded from its IDs at
        // toolCallEnd, so nothing is lost by not routing there.
        let isControl = tokenID == tokenizer.channelStartID
            || tokenID == tokenizer.channelEndID
            || tokenID == tokenizer.toolCallStartID
            || tokenID == tokenizer.toolCallEndID
            || tokenID == tokenizer.toolResponseID
            || tokenID == tokenizer.toolResponseEndID
        var events: [StructuredAssistantEvent] = []
        if isControl, !delta.isEmpty, toolTokens == nil {
            events = routeText(delta)
        }

        if tokenID == tokenizer.channelStartID {
            label = ""
            channel = .label
            return events
        }
        if tokenID == tokenizer.channelEndID {
            channel = .visible
            return events
        }
        if tokenID == tokenizer.toolCallStartID {
            guard toolTokens == nil else {
                failed = true
                throw ToolCallParserError.malformed
            }
            toolTokens = []
            return events
        }
        if tokenID == tokenizer.toolCallEndID {
            guard let tokens = toolTokens else {
                failed = true
                throw ToolCallParserError.malformed
            }
            toolTokens = nil
            let text = tokenizer.decode(tokens, skipSpecialTokens: false)
            do {
                let call = try GemmaToolCallParser().parse(
                    text, allowedTools: allowedTools, id: idGenerator())
                emittedCalls += 1
                return events + [.toolCall(call)]
            } catch {
                failed = true
                throw error
            }
        }
        if tokenID == tokenizer.toolResponseID || tokenID == tokenizer.toolResponseEndID {
            guard emittedCalls > 0, toolTokens == nil else {
                failed = true
                throw ToolCallParserError.malformed
            }
            return events
        }
        if var tokens = toolTokens {
            tokens.append(tokenID)
            guard tokens.count * MemoryLayout<Int32>.size <= GemmaToolCallParser.maximumBytes else {
                failed = true
                throw ToolCallParserError.oversized
            }
            toolTokens = tokens
            return []
        }
        return routeText(delta)
    }

    /// Route text flushed at a stop boundary through the current channel
    /// state. The flush tail is not tied to a token ID, so it cannot go
    /// through `consume`; without this a generation cut off inside the
    /// thought channel would leak its held-back bytes into visible content.
    public func consumeTail(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw GemmaToolCallParserError.malformed }
        if let harmonyDecoder {
            return try harmonyDecoder.consumeTail(text)
        }
        guard toolTokens == nil, !text.isEmpty else { return [] }
        return routeText(text)
    }

    private func routeText(_ delta: String) -> [StructuredAssistantEvent] {
        switch channel {
        case .thought:
            return delta.isEmpty ? [] : [.thinking(delta)]
        case .visible:
            return delta.isEmpty ? [] : [.content(delta)]
        case .label:
            label += delta
            guard let newline = label.firstIndex(of: "\n") else { return [] }
            let name = label[..<newline].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let contentStart = label.index(after: newline)
            let content = String(label[contentStart...])
            channel = name == "final" || name == "answer" ? .visible : .thought
            label = ""
            return routeText(content)
        }
    }

    /// Qwen and MiniMax use thought and tool wrapper tokens, with different
    /// native grammars inside their tool wrappers.
    private func consumeTaggedDialect(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        if tokenID == tokenizer.toolCallStartID {
            guard toolTokens == nil else {
                failed = true
                throw ToolCallParserError.malformed
            }
            toolTokens = []
            return routeText(delta)
        }
        if tokenID == tokenizer.toolCallEndID {
            guard let tokens = toolTokens else {
                failed = true
                throw ToolCallParserError.malformed
            }
            toolTokens = nil
            let text = tokenizer.decode(tokens, skipSpecialTokens: false)
            do {
                let calls: [ParsedToolCall]
                if tokenizer.dialect == .minimax {
                    calls = try MiniMaxToolCallParser().parse(text, allowedTools: allowedTools,
                        parameterSchemas: parameterSchemas, idGenerator: idGenerator)
                } else if tokenizer.dialect == .flint {
                    calls = [try FlintToolCallParser().parse(text, allowedTools: allowedTools, id: idGenerator())]
                } else {
                    calls = [try QwenToolCallParser().parse(
                        text, allowedTools: allowedTools, id: idGenerator())]
                }
                emittedCalls += calls.count
                return calls.map(StructuredAssistantEvent.toolCall)
            } catch {
                failed = true
                throw error
            }
        }
        if var tokens = toolTokens {
            tokens.append(tokenID)
            guard tokens.count * MemoryLayout<Int32>.size <= QwenToolCallParser.maximumBytes else {
                failed = true
                throw ToolCallParserError.oversized
            }
            toolTokens = tokens
            return []
        }
        if tokenID == tokenizer.thinkStartID {
            let events = routeText(delta)
            channel = .thought
            return events
        }
        if tokenID == tokenizer.thinkEndID {
            let events = routeText(delta)
            channel = .visible
            return events
        }
        if channel == .thought {
            return delta.isEmpty ? [] : [.thinking(delta)]
        }
        return delta.isEmpty ? [] : [.content(delta)]
    }

    public func finish() throws {
        if let harmonyDecoder {
            try harmonyDecoder.finish()
            return
        }
        guard !failed, toolTokens == nil else {
            throw ToolCallParserError.malformed
        }
    }

    public var hasToolCalls: Bool {
        harmonyDecoder?.hasToolCalls ?? (emittedCalls > 0)
    }
}
