import Foundation
import TUFFDecodeProtocol
import TUFFInferenceTypes

/// `ValidatedChatRequest` ↔ the serving host's decode protocol.
///
/// The decode path renders conversations from a structured form: an optional
/// system prompt, completed turns (user message, tool rounds, assistant
/// reply), and the current message with its tool rounds. OpenAI messages map
/// onto that form directly when they alternate the way a real conversation
/// does. A message list the form cannot express is refused with
/// `unsupported_value` rather than reshaped into something the model never
/// saw.
enum HostServingTranslation {
    /// Whether the helper needs a real logits head. It is part of the loaded
    /// model's identity, so a request must be prepared with the same answer.
    static func forceLogitsHead(temperature: Float, repetitionPenalty: Float) -> Bool {
        !(temperature == 0 && repetitionPenalty == 1)
    }

    /// One staged image: where its bytes are, and what they hash to.
    struct ImageSource: Equatable {
        let id: UUID
        let fileURL: URL
        let sha256: String
    }

    /// The request's staged images in message order. `imageIdentities` lists
    /// each message's content hashes in the order its images appear.
    static func imageSources(_ request: ValidatedChatRequest) -> [ImageSource] {
        guard let messages = request.multimodalMessages else { return [] }
        var sources: [ImageSource] = []
        for (index, message) in messages.enumerated() {
            let hashes = index < request.imageIdentities.count ? request.imageIdentities[index] : []
            var position = 0
            for part in message.content {
                guard case .image(let id) = part, let url = request.imageFiles[id] else { continue }
                sources.append(ImageSource(
                    id: id, fileURL: url, sha256: position < hashes.count ? hashes[position] : ""))
                position += 1
            }
        }
        return sources
    }

    private static func unsupported(_ message: String) -> ServerRequestError {
        .invalid(message: message, param: "messages", code: "unsupported_value")
    }

    private struct OpenRound {
        var thinking: String?
        var content: String
        var calls: [InferenceToolCall]
        var results: [String: DecodeToolResult] = [:]
    }

    private struct OpenTurn {
        var prompt: String
        var images: [DecodeImageAttachment]
        var rounds: [DecodeToolRound] = []
        var open: OpenRound?
    }

    /// - Parameter images: The host-readable attachment for each staged image.
    static func decodeRequest(
        _ request: ValidatedChatRequest, generationID: UUID,
        maxContextTokens: Int, runtimeOptions: DecodeRuntimeOptions,
        images: [UUID: DecodeImageAttachment]
    ) throws -> DecodeGenerationRequest {
        let messages = request.messages
        var index = 0
        var instructions: [String] = []
        // System and developer guidance becomes the single system prompt.
        while index < messages.count,
              messages[index].role == .system || messages[index].role == .developer {
            if let text = messages[index].content, !text.isEmpty { instructions.append(text) }
            index += 1
        }

        func attachments(at position: Int) -> [DecodeImageAttachment] {
            guard let multimodal = request.multimodalMessages, position < multimodal.count else { return [] }
            return multimodal[position].content.compactMap { part in
                if case .image(let id) = part { return images[id] }
                return nil
            }
        }

        var turns: [DecodeChatTurn] = []
        var current: OpenTurn?
        var prefix: String?
        while index < messages.count {
            let message = messages[index]
            let isLast = index == messages.count - 1
            switch message.role {
            case .system, .developer:
                throw unsupported("system or developer guidance must precede the conversation")
            case .user:
                guard current == nil else {
                    throw unsupported(
                        "a user message must be answered by the assistant before the next user message; "
                            + "consecutive user messages, or a user message directly after tool results, "
                            + "are not supported")
                }
                current = OpenTurn(prompt: message.content ?? "", images: attachments(at: index))
            case .assistant:
                guard var turn = current else {
                    throw unsupported("the conversation must begin with a user message")
                }
                guard turn.open == nil else {
                    throw unsupported("an assistant message with tool calls must be followed by a tool result for each call")
                }
                if !message.toolCalls.isEmpty {
                    turn.open = OpenRound(thinking: message.thinking, content: message.content ?? "",
                                          calls: message.toolCalls)
                    current = turn
                } else if isLast {
                    // A trailing assistant message is text the model continues.
                    guard message.thinking?.isEmpty ?? true else {
                        throw unsupported("reasoning_content on a trailing assistant message is not supported")
                    }
                    prefix = message.content ?? ""
                } else {
                    turns.append(DecodeChatTurn(
                        prompt: turn.prompt, response: message.content ?? "",
                        thinking: message.thinking,
                        images: turn.images.isEmpty ? nil : turn.images,
                        toolRounds: turn.rounds.isEmpty ? nil : turn.rounds))
                    current = nil
                }
            case .tool:
                guard var turn = current, var round = turn.open,
                      let id = message.toolCallID,
                      let call = round.calls.first(where: { $0.id == id }),
                      round.results[id] == nil else {
                    throw unsupported("tool result must follow the assistant message that made the call")
                }
                round.results[id] = DecodeToolResult(
                    callID: id, name: call.name, content: message.content ?? "")
                if round.results.count == round.calls.count {
                    turn.rounds.append(DecodeToolRound(
                        thinking: round.thinking, content: round.content, calls: round.calls,
                        results: round.calls.compactMap { round.results[$0.id] }))
                    turn.open = nil
                } else {
                    turn.open = round
                }
                current = turn
            }
            index += 1
        }
        guard let final = current else {
            throw unsupported("the conversation must end with a user message or tool results")
        }
        guard final.open == nil else {
            throw unsupported("an assistant message with tool calls must be followed by a tool result for each call")
        }

        let config = request.generationConfig
        return DecodeGenerationRequest(
            prompt: final.prompt,
            systemPrompt: instructions.isEmpty ? nil : instructions.joined(separator: "\n\n"),
            assistantPrefix: (prefix?.isEmpty ?? true) ? nil : prefix,
            history: turns,
            imageAttachments: final.images.isEmpty ? nil : final.images,
            maxNewTokens: request.maximumCompletionTokens,
            maxContextTokens: maxContextTokens,
            reasoning: DecodeChatReasoning(rawValue: request.reasoning.rawValue) ?? .off,
            reasoningEffort: request.reasoningEffort.flatMap {
                DecodeGPTOSSReasoningEffort(rawValue: $0.rawValue)
            },
            preserveThinking: request.preserveThinking,
            temperature: config.temperature,
            topK: config.topK,
            topP: config.topP,
            repetitionPenalty: config.repetitionPenalty,
            seed: config.seed,
            runtimeOptions: runtimeOptions,
            generationID: generationID,
            tools: request.tools.isEmpty ? nil : request.tools,
            currentRounds: final.rounds.isEmpty ? nil : final.rounds,
            conversationKey: request.promptCacheKey,
            // The API refuses an over-long conversation; it never silently
            // drops its oldest turns to make it fit.
            preservesFullHistory: true)
    }

    /// The error a failed generation maps to.
    static func requestError(message: String, kind: String?) -> any Error {
        if message.contains("-token context") || message.contains("exceeds this Auto context") {
            return ServerRequestError.invalid(
                message: message, param: "messages", code: "context_length_exceeded")
        }
        if message.contains(ServingMemoryPressure.phrase)
            || ServingMemoryPressure.isExclusiveBusy(message) {
            return ServerRequestError.modelMemoryBusy(message)
        }
        if message.contains("Load the model before generating")
            || message.contains("Reload the model before generating")
            || message.contains("Model is unavailable or session is already active") {
            return ServerRequestError.modelBusy(
                "The shared model changed while the request was starting. Retry the request.")
        }
        if kind == "malformedToolCall" {
            return HostServingError("The model produced a tool call that could not be read: \(message)")
        }
        return HostServingError(message)
    }
}

/// Holds back the tail of streamed text that could still become a stop
/// string, so a stop sequence split across chunks is found and never emitted.
struct HostServingStopFilter {
    private let stops: [String]
    private let holdScalars: Int
    private var buffer = ""
    private(set) var matched = false

    init(stops: [String]) {
        self.stops = stops.filter { !$0.isEmpty }
        holdScalars = max(0, (self.stops.map { $0.unicodeScalars.count }.max() ?? 1) - 1)
    }

    var isActive: Bool { !stops.isEmpty }

    /// Text that is now safe to emit.
    mutating func push(_ text: String) -> String {
        guard isActive else { return text }
        guard !matched else { return "" }
        buffer += text
        var earliest: Range<String.Index>?
        for stop in stops {
            if let range = buffer.range(of: stop, options: .literal),
               earliest.map({ range.lowerBound < $0.lowerBound }) ?? true {
                earliest = range
            }
        }
        if let earliest {
            matched = true
            let output = String(buffer[..<earliest.lowerBound])
            buffer = ""
            return output
        }
        let count = buffer.unicodeScalars.count
        guard count > holdScalars else { return "" }
        let split = buffer.unicodeScalars.index(buffer.unicodeScalars.startIndex,
                                                offsetBy: count - holdScalars)
        let output = String(buffer.unicodeScalars[..<split])
        buffer = String(buffer.unicodeScalars[split...])
        return output
    }

    /// What was still held when the stream ended without a stop.
    mutating func flush() -> String {
        defer { buffer = "" }
        return matched ? "" : buffer
    }
}

/// Folds one generation's host events into API events and the completion.
struct HostServingEventTranslator {
    enum Terminal {
        case completed(ServerCompletion)
        /// The host cancelled the generation and no stop string asked for it.
        case cancelled
        case failed(any Error)
    }

    struct Output {
        var events: [ServerInferenceEvent] = []
        /// A stop string matched: the generation should be cancelled now.
        var requestStop = false
        var terminal: Terminal?
    }

    private var stop: HostServingStopFilter
    private var content = ""
    private var reasoning = ""

    init(stops: [String]) { stop = HostServingStopFilter(stops: stops) }

    mutating func consume(_ event: DecodeServiceEvent) -> Output {
        var output = Output()
        switch event.kind {
        case .snapshot:
            if let thinking = event.thinkingDelta, !thinking.isEmpty {
                reasoning += thinking
                output.events.append(.reasoning(thinking))
            }
            if !event.textDelta.isEmpty {
                let wasMatched = stop.matched
                let text = stop.push(event.textDelta)
                if !text.isEmpty {
                    content += text
                    output.events.append(.content(text))
                }
                if stop.matched, !wasMatched { output.requestStop = true }
            }
        case .finished:
            let tail = stop.flush()
            if !tail.isEmpty {
                content += tail
                output.events.append(.content(tail))
            }
            let calls = (event.toolCalls ?? []).map { call in
                ParsedToolCall(id: call.id, name: call.name, arguments: call.arguments,
                               argumentsJSON: (try? call.arguments.encoded()) ?? "{}")
            }
            for call in calls { output.events.append(.toolCall(call)) }
            let reason: String
            if !calls.isEmpty { reason = "tool_calls" }
            else if stop.matched { reason = "stop" }
            else { reason = event.stopReason == "maxTokens" ? "length" : "stop" }
            output.terminal = .completed(completion(event, calls: calls, reason: reason))
        case .cancelled:
            if stop.matched {
                output.terminal = .completed(completion(event, calls: [], reason: "stop"))
            } else {
                output.terminal = .cancelled
            }
        case .failed:
            output.terminal = .failed(HostServingTranslation.requestError(
                message: event.error ?? "generation failed", kind: event.errorKind))
        default:
            break
        }
        return output
    }

    private func completion(_ event: DecodeServiceEvent, calls: [ParsedToolCall],
                            reason: String) -> ServerCompletion {
        let prompt = event.promptTokenCount ?? 0
        var timings: [String: Double] = [:]
        if let seconds = event.prefillSeconds { timings["host_prefill"] = seconds }
        if let seconds = event.timeToFirstTokenSeconds { timings["host_first_token"] = seconds }
        if event.decodeSeconds > 0 { timings["host_decode"] = event.decodeSeconds }
        // The decode path reports no reasoning-token split, so the usage
        // omits `completion_tokens_details` rather than estimating it.
        return ServerCompletion(
            content: content, reasoning: reasoning, toolCalls: calls, finishReason: reason,
            usage: OpenAIUsage(promptTokens: prompt, completionTokens: event.tokenCount,
                               totalTokens: prompt + event.tokenCount,
                               cachedTokens: event.cachedPromptTokens ?? 0),
            timingSeconds: timings)
    }
}
