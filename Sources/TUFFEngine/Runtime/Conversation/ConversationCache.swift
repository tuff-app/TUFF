import Foundation

/// Everything outside the conversation that decides what cached tokens mean.
/// Two requests share reusable state only when every field is equal.
public struct ConversationCacheDomain: Sendable, Equatable, Hashable {
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let runtimeProfileHash: String
    public let maximumContext: Int
    public let kvStorage: String
    public let fp16RingEnabled: Bool
    public let templateSHA256: String

    public init(modelID: String, sourceSnapshotHash: String?, runtimeProfileHash: String,
                maximumContext: Int, kvStorage: String, fp16RingEnabled: Bool,
                templateSHA256: String) {
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.runtimeProfileHash = runtimeProfileHash
        self.maximumContext = maximumContext
        self.kvStorage = kvStorage
        self.fp16RingEnabled = fp16RingEnabled
        self.templateSHA256 = templateSHA256
    }
}

/// The conversation a request asks the model to continue, in the structured
/// form the chat template renders: assistant tool calls, tool results and
/// reasoning stay separate messages and fields rather than flattened text.
public struct ConversationTranscript: Sendable, Equatable {
    public var messages: [GFTokenizer.Message]
    /// Content SHA-256 of each message's images, aligned with `messages`.
    /// Nil means a request carried images without identity; such a request
    /// neither publishes nor matches, because placeholder tokens alone cannot
    /// tell two images apart.
    public var imageIdentities: [[String]]?
    public var tools: [GFTokenizer.FunctionDefinition]
    public var reasoning: ChatReasoning
    public var reasoningEffort: ModelReasoningEffort?
    public var harmonyCurrentDate: String?
    public var preserveThinking: Bool

    public init(messages: [GFTokenizer.Message],
                imageIdentities: [[String]]? = nil,
                tools: [GFTokenizer.FunctionDefinition] = [],
                reasoning: ChatReasoning = .off,
                reasoningEffort: ModelReasoningEffort? = nil,
                harmonyCurrentDate: String? = nil,
                preserveThinking: Bool = false) {
        self.messages = messages
        self.imageIdentities = imageIdentities
            ?? Array(repeating: [], count: messages.count)
        self.tools = tools
        self.reasoning = reasoning
        self.reasoningEffort = reasoningEffort
        self.harmonyCurrentDate = harmonyCurrentDate
        self.preserveThinking = preserveThinking
    }

    /// For a request whose images arrived without identity.
    public static func withoutImageIdentity(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = [],
        reasoning: ChatReasoning = .off,
        reasoningEffort: ModelReasoningEffort? = nil,
        harmonyCurrentDate: String? = nil,
        preserveThinking: Bool = false
    ) -> ConversationTranscript {
        var transcript = ConversationTranscript(
            messages: messages, tools: tools, reasoning: reasoning,
            reasoningEffort: reasoningEffort, harmonyCurrentDate: harmonyCurrentDate,
            preserveThinking: preserveThinking)
        transcript.imageIdentities = nil
        return transcript
    }

    var hasImages: Bool {
        guard let imageIdentities else { return true }
        return !imageIdentities.allSatisfy(\.isEmpty)
    }
}

public struct CachedAssistantTurn: Sendable, Equatable {
    public let message: GFTokenizer.Message
    public let rawStopReason: StopReason

    public init(message: GFTokenizer.Message, rawStopReason: StopReason) {
        self.message = message
        self.rawStopReason = rawStopReason
    }
}

/// What one completed request leaves behind: the transcript it rendered, the
/// assistant turn it generated, and the exact token sequence its KV holds.
public struct ConversationCacheEntry: Sendable, Equatable {
    public let domain: ConversationCacheDomain
    public let transcript: ConversationTranscript
    public let assistantTurn: CachedAssistantTurn
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
    public let kvPosition: Int
    /// The caller's name for this conversation, when it gave one. It only
    /// orders the search and decides which entry a new one replaces; a match
    /// is always established from tokens and messages.
    public let conversationKey: String?
    /// Points inside this KV a runner can return to, shortest first: where
    /// the system prompt and tools end, and where the current user turn
    /// began. A later request whose fresh render starts with exactly one of
    /// these token runs resumes there instead of from scratch, even when the
    /// template rewrites what came after it or it is another conversation.
    public var prefixCheckpoints: [ConversationPrefixCheckpoint] = []

    public var inputMessages: [GFTokenizer.Message] { transcript.messages }
    public var tools: [GFTokenizer.FunctionDefinition] { transcript.tools }
}

/// A runner checkpoint and the KV tokens before it.
public struct ConversationPrefixCheckpoint: Sendable, Equatable {
    public let tokenIDs: [Int32]
    public let snapshot: RunnerStateSnapshot

    public init?(tokenIDs: [Int32], snapshot: RunnerStateSnapshot) {
        guard !tokenIDs.isEmpty, snapshot.position == tokenIDs.count else { return nil }
        self.tokenIDs = tokenIDs
        self.snapshot = snapshot
    }

    public var position: Int { tokenIDs.count }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.tokenIDs == rhs.tokenIDs && lhs.snapshot === rhs.snapshot
    }
}

/// Why a prefix could not be reused. A bridge that fails to render is a
/// defect worth seeing; a history that simply diverged is not.
public enum ConversationCacheMissReason: String, Sendable, Equatable {
    case noEntry = "no-entry"
    case unusableEntry = "unusable-entry"
    case historyDiverged = "history-diverged"
    case unsupportedContinuation = "unsupported-continuation"
    case boundaryMismatch = "boundary-mismatch"
    case bridgeRenderFailed = "bridge-render-failed"
    case missingImageIdentity = "missing-image-identity"
    case imagesDiverged = "images-diverged"
}

public enum ConversationCacheMatch: Sendable, Equatable {
    case miss(ConversationCacheMissReason)
    case hit(effectivePromptIDs: [Int32], cachedPromptTokens: Int)
    /// History and images match, but the new turn carries its own image, so
    /// the caller renders just that turn as a continuation of the cached
    /// tokens. History is never re-rendered: the KV holds what the model
    /// generated, and a fresh render re-tokenizes that text.
    case renderThenResume(cachedPromptTokens: Int)

    public static let miss: ConversationCacheMatch = .miss(.noEntry)

    public var missReason: ConversationCacheMissReason? {
        if case .miss(let reason) = self { return reason }
        return nil
    }

    public var isHit: Bool { missReason == nil }
}

/// Either spelling enables logging, preserving the legacy presence-based switch.
func conversationCacheLoggingEnabled(environment: [String: String]) -> Bool {
    environment["TUFF_LOG_CACHE"] != nil || environment["TFF_LOG_CACHE"] != nil
}

/// Names the condition behind a miss when `TUFF_LOG_CACHE` is set. A miss
/// never breaks a response; it silently pays full prefill, which is why it
/// goes unnoticed without this.
func logConversationCacheMiss(_ reason: @autoclosure () -> String) {
    guard conversationCacheLoggingEnabled(environment: ProcessInfo.processInfo.environment) else { return }
    FileHandle.standardError.write(Data("[cache-miss] \(reason())\n".utf8))
}

public enum ConversationCache {
    /// The entry a completed request publishes, or nil when its KV cannot be
    /// continued exactly: a sequence that did not end at a clean boundary, a
    /// stop string that cut visible text, or images without identity.
    public static func entry(
        domain: ConversationCacheDomain,
        transcript: ConversationTranscript,
        content: String,
        thinking: String? = nil,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool = false,
        conversationKey: String? = nil,
        prefixCheckpoints: [ConversationPrefixCheckpoint] = []
    ) -> ConversationCacheEntry? {
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              !stopStringFiltered,
              result.reason == .endOfTurn
                || result.reason == .toolCalls
                || result.reason == .maxTokens
                // Harmony's `<|return|>` ends a turn as an EOS. Only a
                // checkpoint can continue past it, and it never reads the
                // generated tokens, so such an entry is kept only with one.
                || (result.reason == .eos && !prefixCheckpoints.isEmpty) else {
            // A rejected publish leaves nothing for the next request to match,
            // so a single event reads as a permanently dead cache.
            logConversationCacheMiss(
                "publish rejected: kvPosition=\(result.kvPosition) "
                + "kvBacked=\(result.kvBackedTokenIDs.count) "
                + "boundary=\(result.uncommittedBoundaryTokenIDs.count) "
                + "stopStringFiltered=\(stopStringFiltered) "
                + "reason=\(result.reason)")
            return nil
        }
        guard transcript.imageIdentities != nil else { return nil }
        let historicalCalls = calls.map {
            GFTokenizer.HistoricalToolCall(id: $0.id, name: $0.name, arguments: $0.arguments)
        }
        let trimmedThinking = thinking.flatMap { $0.isEmpty ? nil : $0 }
        let assistant = GFTokenizer.Message(
            role: .assistant,
            content: content.isEmpty && !calls.isEmpty ? nil : content,
            thinking: trimmedThinking,
            toolCalls: historicalCalls)
        return ConversationCacheEntry(
            domain: domain,
            transcript: transcript,
            assistantTurn: CachedAssistantTurn(message: assistant, rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition,
            conversationKey: conversationKey,
            prefixCheckpoints: validCheckpoints(prefixCheckpoints, result: result))
    }

    /// The checkpoints that lie inside the KV this result describes, one per
    /// position, shortest first.
    static func validCheckpoints(_ checkpoints: [ConversationPrefixCheckpoint],
                                 result: RawDecodeResult) -> [ConversationPrefixCheckpoint] {
        var byPosition: [Int: ConversationPrefixCheckpoint] = [:]
        for checkpoint in checkpoints
        where checkpoint.position < result.kvPosition
            && result.kvBackedTokenIDs.starts(with: checkpoint.tokenIDs) {
            byPosition[checkpoint.position] = byPosition[checkpoint.position] ?? checkpoint
        }
        return byPosition.values.sorted { $0.position < $1.position }
    }

    /// Whether `transcript` continues `entry`, and with which prompt.
    ///
    /// `renderedPromptIDs` is nil for multimodal requests, where rendered ids
    /// cannot establish identity: image placeholders are the same token
    /// whichever image they stand for.
    public static func match(
        entry: ConversationCacheEntry?,
        domain: ConversationCacheDomain,
        transcript: ConversationTranscript,
        renderedPromptIDs: [Int32]?,
        tokenizer: GFTokenizer,
        modelVariant: ModelVariant? = nil,
        allowsTextBridge: Bool = true
    ) -> ConversationCacheMatch {
        let continuation = matchContinuation(
            entry: entry, domain: domain, transcript: transcript,
            renderedPromptIDs: renderedPromptIDs, tokenizer: tokenizer,
            modelVariant: modelVariant, allowsTextBridge: allowsTextBridge)
        guard !continuation.isHit,
              let checkpointHit = matchCheckpoint(
                entry: entry, domain: domain, transcript: transcript,
                renderedPromptIDs: renderedPromptIDs) else {
            return continuation
        }
        return checkpointHit
    }

    /// Resumes from the entry's checkpoint when the fresh render starts with
    /// exactly the tokens before it. This is the only reuse that shortens
    /// the KV, so it is the hit whose cached count is below `kvPosition`.
    /// Harmony needs it after a final answer: the next render drops that
    /// turn's reasoning, so the generated tokens are never a prefix of it,
    /// but everything up to where the turn began still is.
    static func matchCheckpoint(
        entry: ConversationCacheEntry?,
        domain: ConversationCacheDomain,
        transcript: ConversationTranscript,
        renderedPromptIDs: [Int32]?,
        maximumPosition: Int = .max
    ) -> ConversationCacheMatch? {
        guard let entry, let renderedPromptIDs,
              entry.domain == domain,
              ConversationCacheIdentity.tools(entry.transcript.tools, transcript.tools),
              entry.transcript.reasoning == transcript.reasoning,
              entry.transcript.reasoningEffort == transcript.reasoningEffort,
              entry.transcript.harmonyCurrentDate == transcript.harmonyCurrentDate,
              entry.transcript.preserveThinking == transcript.preserveThinking,
              let requestIdentities = transcript.imageIdentities,
              requestIdentities.allSatisfy(\.isEmpty),
              entry.transcript.imageIdentities?.allSatisfy(\.isEmpty) == true,
              let checkpoint = entry.prefixCheckpoints.last(where: {
                  $0.position < entry.kvPosition && $0.position <= maximumPosition
                      && renderedPromptIDs.count > $0.position
                      && renderedPromptIDs.starts(with: $0.tokenIDs)
              }) else {
            return nil
        }
        return .hit(effectivePromptIDs: renderedPromptIDs,
                    cachedPromptTokens: checkpoint.position)
    }

    static func matchContinuation(
        entry: ConversationCacheEntry?,
        domain: ConversationCacheDomain,
        transcript: ConversationTranscript,
        renderedPromptIDs: [Int32]?,
        tokenizer: GFTokenizer,
        modelVariant: ModelVariant?,
        allowsTextBridge: Bool
    ) -> ConversationCacheMatch {
        guard let entry,
              entry.domain == domain,
              ConversationCacheIdentity.tools(entry.transcript.tools, transcript.tools),
              entry.transcript.reasoning == transcript.reasoning,
              entry.transcript.reasoningEffort == transcript.reasoningEffort,
              entry.transcript.harmonyCurrentDate == transcript.harmonyCurrentDate,
              entry.transcript.preserveThinking == transcript.preserveThinking,
              entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1 else {
            logConversationCacheMiss(entryMissReason(entry: entry, domain: domain,
                                                     transcript: transcript))
            return entry == nil ? .miss(.noEntry) : .miss(.unusableEntry)
        }
        guard let requestIdentities = transcript.imageIdentities else {
            return .miss(.missingImageIdentity)
        }
        guard let entryIdentities = entry.transcript.imageIdentities else {
            return .miss(.missingImageIdentity)
        }

        // Rendered ids cannot identify an image, so the cheap prefix path is
        // refused whenever either side carries one.
        if let renderedPromptIDs,
           requestIdentities.allSatisfy(\.isEmpty),
           entryIdentities.allSatisfy(\.isEmpty),
           renderedPromptIDs.count > entry.kvPosition,
           renderedPromptIDs.prefix(entry.kvPosition).elementsEqual(entry.kvBackedTokenIDs) {
            return .hit(effectivePromptIDs: renderedPromptIDs,
                        cachedPromptTokens: entry.kvPosition)
        }

        let inputCount = entry.transcript.messages.count
        let messages = transcript.messages
        guard messages.count > inputCount + 1,
              ConversationCacheIdentity.messages(messages.prefix(inputCount), entry.transcript.messages),
              assistantMatches(messages[inputCount], entry.assistantTurn.message) else {
            logConversationCacheMiss(historyMissReason(entry: entry, transcript: transcript))
            return .miss(.historyDiverged)
        }
        guard requestIdentities.count == messages.count,
              requestIdentities.prefix(inputCount).elementsEqual(entryIdentities),
              requestIdentities[inputCount].isEmpty else {
            return .miss(.imagesDiverged)
        }
        // The text bridges cannot render an image, so a continuation carrying
        // one resumes by rendering just that turn, and only after a clean end
        // of turn: a `.maxTokens` entry holds a generated token outside the KV
        // that the bridge does not replay, and a `.toolCalls` boundary is a
        // marker the multimodal bridge does not emit.
        if !requestIdentities.dropFirst(inputCount).allSatisfy(\.isEmpty) {
            guard allowsTextBridge,
                  entry.assistantTurn.rawStopReason == .endOfTurn else {
                return .miss(.unsupportedContinuation)
            }
            return .renderThenResume(cachedPromptTokens: entry.kvPosition)
        }
        let continuation = Array(messages.dropFirst(inputCount + 1))
        if entry.assistantTurn.message.toolCalls.isEmpty {
            // A caller may require an exact rendered prefix for ordinary
            // follow-ups, for example when the cached tokens hold reasoning a
            // fresh render would leave out.
            guard allowsTextBridge else { return .miss(.unsupportedContinuation) }
            return matchTextContinuation(entry: entry, continuation: continuation,
                                         tokenizer: tokenizer, modelVariant: modelVariant,
                                         reasoning: transcript.reasoning)
        }
        return matchToolContinuation(entry: entry, transcript: transcript,
                                     continuation: continuation, tokenizer: tokenizer)
    }

    /// Which part of the entry guard failed. Kept beside the guard so a new
    /// condition added to one is noticed in the other by its test.
    public static func entryMissReason(
        entry: ConversationCacheEntry?,
        domain: ConversationCacheDomain,
        transcript: ConversationTranscript
    ) -> String {
        guard let entry else {
            return "no entry stored (cold start, or the previous publish was rejected)"
        }
        if entry.domain != domain {
            return "domain changed (model, runtime profile, context, or template)"
        }
        if !ConversationCacheIdentity.tools(entry.transcript.tools, transcript.tools) { return "tool set changed" }
        if entry.transcript.reasoning != transcript.reasoning { return "reasoning mode changed" }
        if entry.transcript.reasoningEffort != transcript.reasoningEffort {
            return "reasoning effort changed"
        }
        if entry.transcript.harmonyCurrentDate != transcript.harmonyCurrentDate {
            return "Harmony calendar date changed"
        }
        if entry.transcript.preserveThinking != transcript.preserveThinking {
            return "preserved-thinking setting changed"
        }
        return "entry inconsistent: kvPosition=\(entry.kvPosition) "
            + "kvBacked=\(entry.kvBackedTokenIDs.count) "
            + "boundary=\(entry.uncommittedBoundaryTokenIDs.count)"
    }

    public static func historyMissReason(
        entry: ConversationCacheEntry,
        transcript: ConversationTranscript
    ) -> String {
        let inputCount = entry.transcript.messages.count
        if transcript.messages.count <= inputCount + 1 {
            return "history did not extend: \(transcript.messages.count) messages arrived, "
                + "entry holds \(inputCount) plus the assistant turn"
        }
        if !ConversationCacheIdentity.messages(transcript.messages.prefix(inputCount), entry.transcript.messages) {
            return "client rewrote the preceding \(inputCount) messages"
        }
        return "assistant turn in the history differs from the one generated "
            + "(\(entry.assistantTurn.message.toolCalls.count) tool calls cached)"
    }

    /// The incoming assistant turn must be the one the model generated. An
    /// absent and an empty content are the same rendered text.
    static func assistantMatches(_ incoming: GFTokenizer.Message,
                                 _ cached: GFTokenizer.Message) -> Bool {
        guard incoming.role == .assistant, cached.role == .assistant,
              ConversationCacheIdentity.calls(incoming.toolCalls, cached.toolCalls),
              ConversationCacheIdentity.text(incoming.toolCallID, cached.toolCallID),
              ConversationCacheIdentity.text(incoming.name, cached.name),
              ConversationCacheIdentity.text(incoming.content ?? "", cached.content ?? "") else {
            return false
        }
        // A client that drops the reasoning it was never shown still sent
        // the turn the model produced; one that sends different reasoning
        // did not.
        if let thinking = incoming.thinking, !thinking.isEmpty {
            return ConversationCacheIdentity.text(thinking, cached.thinking)
        }
        return true
    }

    private static func matchTextContinuation(
        entry: ConversationCacheEntry,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer,
        modelVariant: ModelVariant?,
        reasoning: ChatReasoning
    ) -> ConversationCacheMatch {
        guard continuation.count == 1,
              continuation[0].role == .user,
              let content = continuation[0].content,
              continuation[0].toolCalls.isEmpty,
              continuation[0].toolCallID == nil,
              entry.assistantTurn.rawStopReason == .endOfTurn
                || entry.assistantTurn.rawStopReason == .maxTokens else {
            logConversationCacheMiss(
                "text continuation did not match: \(continuation.count) messages, "
                + "stop=\(entry.assistantTurn.rawStopReason)")
            return .miss(.unsupportedContinuation)
        }
        var bridge = tokenizer.encodeTextContinuation(
            userContent: content, modelVariant: modelVariant, reasoning: reasoning)
        if entry.assistantTurn.rawStopReason == .maxTokens {
            bridge = entry.uncommittedBoundaryTokenIDs + bridge
        } else if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
            logConversationCacheMiss("text bridge does not start at the KV boundary")
            return .miss(.boundaryMismatch)
        }
        return .hit(effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
                    cachedPromptTokens: entry.kvPosition)
    }

    private static func matchToolContinuation(
        entry: ConversationCacheEntry,
        transcript: ConversationTranscript,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer
    ) -> ConversationCacheMatch {
        let calls = entry.assistantTurn.message.toolCalls
        guard entry.assistantTurn.rawStopReason == .toolCalls
                || entry.assistantTurn.rawStopReason == .endOfTurn,
              continuation.count == calls.count,
              zip(continuation, calls).allSatisfy({ message, call in
                  message.role == .tool
                    && message.toolCallID == call.id
                    && (message.name == nil || message.name == call.name)
                    && message.content != nil
                    && message.toolCalls.isEmpty
              }) else {
            logConversationCacheMiss(
                "tool-result continuation did not match: \(continuation.count) results "
                + "for \(calls.count) calls, stop=\(entry.assistantTurn.rawStopReason)")
            return .miss(.unsupportedContinuation)
        }
        let bridge: [Int32]
        do {
            bridge = try tokenizer.encodeToolResultContinuation(
                cachedMessages: entry.transcript.messages,
                assistant: entry.assistantTurn.message,
                incomingMessages: transcript.messages,
                tools: transcript.tools,
                reasoning: transcript.reasoning,
                preserveThinking: transcript.preserveThinking,
                reasoningEffort: transcript.reasoningEffort,
                harmonyCurrentDate: transcript.harmonyCurrentDate)
        } catch {
            logConversationCacheMiss("tool-result bridge failed to encode: \(error)")
            return .miss(.bridgeRenderFailed)
        }
        guard bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            logConversationCacheMiss(
                "tool-result bridge does not start at the KV boundary: "
                + "\(String(describing: bridge.first)) vs "
                + "\(String(describing: entry.uncommittedBoundaryTokenIDs.first))")
            return .miss(.boundaryMismatch)
        }
        return .hit(effectivePromptIDs: entry.kvBackedTokenIDs + bridge,
                    cachedPromptTokens: entry.kvPosition)
    }
}
