import Foundation
import TUFFInferenceTypes
import TUFFDecodeProtocol

/// The app-request ↔ decode-event translation shared by every transport that
/// reaches a decode-service helper: the direct helper connection and the
/// serving host's private socket.
public enum DecodeGenerationTranslation {
    public static func decodeRequest(_ request: AppGenerationRequest,
                                     generationID: UUID) -> DecodeGenerationRequest {
        func image(_ attachment: AppImageAttachment) -> DecodeImageAttachment {
            DecodeImageAttachment(
                id: attachment.id, path: attachment.fileURL.path,
                displayName: attachment.displayName,
                encodedBytes: attachment.encodedBytes, sha256: attachment.sha256)
        }
        var command = DecodeGenerationRequest(
            prompt: request.prompt,
            systemPrompt: request.systemPrompt.isEmpty ? nil : request.systemPrompt,
            assistantPrefix: request.assistantPrefix.isEmpty ? nil : request.assistantPrefix,
            history: request.history.map { turn in
                DecodeChatTurn(
                    prompt: turn.prompt,
                    response: turn.response,
                    thinking: turn.thinking,
                    images: turn.images.isEmpty ? nil : turn.images.map(image),
                    toolRounds: turn.toolRounds.isEmpty
                        ? nil : turn.toolRounds.map(DecodeServiceInferenceClient.decodeToolRound))
            },
            imageAttachments: request.imageAttachments.map(image),
            maxNewTokens: request.maxNewTokens,
            maxContextTokens: request.maxContextTokens,
            reasoning: DecodeChatReasoning(rawValue: request.reasoning.rawValue) ?? .off,
            reasoningEffort: request.reasoningEffort.flatMap {
                DecodeGPTOSSReasoningEffort(rawValue: $0.rawValue)
            },
            preserveThinking: request.preserveThinking,
            temperature: request.temperature,
            topK: request.topK,
            topP: request.topP,
            repetitionPenalty: request.repetitionPenalty,
            seed: request.seed,
            runtimeOptions: DecodeServiceInferenceClient.decodeRuntimeOptions(request.runtimeOptions),
            generationID: generationID)
        command.tools = request.tools.isEmpty ? nil : request.tools
        command.currentRounds = request.currentRounds.isEmpty
            ? nil : request.currentRounds.map(DecodeServiceInferenceClient.decodeToolRound)
        command.conversationKey = request.conversationKey
        command.preservesFullHistory = request.preservesFullHistory
        return command
    }
}

/// Converts one generation's decode-service events into app events. Holds the
/// sequence check and the 0.5 s visible-text throttle that used to live inline
/// in the direct client's reader loop.
public struct DecodeGenerationEventTranslator {
    public enum Terminal {
        case finished
        case cancelled
        case failed(AppInferenceError)
    }

    public struct Output {
        public var events: [AppInferenceEvent] = []
        public var terminal: Terminal?
    }

    private let options: AppRuntimeOptions
    private let mailbox: GenerationTranscriptMailbox?
    private var expectedSequence: UInt64 = 1
    private var lastMetricYield = Date.distantPast
    private var hasYieldedVisibleText = false
    private var pendingVisibleText = ""

    public init(options: AppRuntimeOptions, mailbox: GenerationTranscriptMailbox?) {
        self.options = options
        self.mailbox = mailbox
    }

    /// Throws only when the event sequence is inconsistent.
    public mutating func consume(_ event: DecodeServiceEvent) throws -> Output {
        var output = Output()
        if event.kind == .memory {
            output.events.append(.memorySample)
            return output
        }
        if event.kind == .prefill || event.kind == .snapshot {
            guard event.sequence == expectedSequence else {
                throw AppInferenceError.unknown(
                    "decode service event sequence changed from \(expectedSequence) to \(event.sequence)")
            }
            expectedSequence &+= 1
        }
        if event.kind == .prefill, let done = event.prefillDone, let total = event.prefillTotal {
            output.events.append(.prefillProgress(done: done, total: total))
            return output
        }
        if event.kind == .snapshot {
            if let thinking = event.thinkingDelta, !thinking.isEmpty {
                output.events.append(.thinking(AppTokenEvent(
                    index: max(0, event.tokenCount - 1), textDelta: thinking,
                    elapsedDecodeSeconds: event.decodeSeconds)))
            }
            mailbox?.append(event.textDelta)
            if event.textDelta.isEmpty { return output }
            pendingVisibleText += event.textDelta
            let now = Date()
            let beginsVisibleText = !hasYieldedVisibleText
                && event.textDelta.contains { !$0.isWhitespace }
            if beginsVisibleText || now.timeIntervalSince(lastMetricYield) >= 0.5 {
                lastMetricYield = now
                hasYieldedVisibleText = hasYieldedVisibleText || beginsVisibleText
                output.events.append(.token(AppTokenEvent(
                    index: max(0, event.tokenCount - 1), textDelta: pendingVisibleText,
                    elapsedDecodeSeconds: event.decodeSeconds)))
                pendingVisibleText = ""
            }
            return output
        }

        if !pendingVisibleText.isEmpty {
            output.events.append(.token(AppTokenEvent(
                index: max(0, event.tokenCount - 1), textDelta: pendingVisibleText,
                elapsedDecodeSeconds: event.decodeSeconds)))
            pendingVisibleText = ""
        }
        let diagnostics = DecodeServiceInferenceClient.diagnostics(event, options: options)
        switch event.kind {
        case .finished:
            if let calls = event.toolCalls, !calls.isEmpty {
                output.events.append(.toolCalls(calls.map {
                    AppToolCall(id: $0.id, name: $0.name, arguments: $0.arguments)
                }))
            }
            output.events.append(.finished(diagnostics))
            output.terminal = .finished
        case .cancelled:
            output.events.append(.cancelled(diagnostics))
            output.terminal = .cancelled
        case .failed:
            let error = event.errorKind == "malformedToolCall"
                ? AppInferenceError.malformedToolCall(event.error ?? "")
                : AppInferenceError.unknown(event.error ?? "decode service failed")
            output.events.append(.failed(error, partial: diagnostics))
            output.terminal = .failed(error)
        default:
            break
        }
        return output
    }
}
