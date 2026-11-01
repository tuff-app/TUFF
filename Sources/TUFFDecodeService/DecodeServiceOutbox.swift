import Foundation
import TUFFEngine
import TUFFAppCore
import TUFFDecodeProtocol

final class DecodeServiceOutbox: @unchecked Sendable {
    private struct PrefillProgress {
        var done: Int
        var total: Int
    }

    private struct State {
        var pendingText = ""
        var pendingThinking = ""
        var pendingBytes = 0
        var toolCallBytes = 0
        var latestPrefill: PrefillProgress?
        var latestToken: AppTokenEvent?
        var terminal: DecodeServiceEvent?
        var terminalCommitted = false
        var toolCalls: [GFTokenizer.HistoricalToolCall] = []
        var finished = false
        var sequence: UInt64 = 0
    }

    private let condition = NSCondition()
    private var state = State()
    private let generationID: UUID
    private let memorySampler = AppMemorySampler()
    private let maximumPendingBytes: Int
    private let onOverflow: @Sendable () -> Void

    /// Bytes of image tower currently held mapped, or nil when there is no
    /// vision runtime. Sampled per event so Keep Ready is visible while a run
    /// is happening, not only in its final diagnostics.
    private let towerBytes: @Sendable () -> UInt64?

    init(generationID: UUID,
         towerBytes: @escaping @Sendable () -> UInt64? = { nil },
         maximumPendingBytes: Int = 4 * 1_024 * 1_024,
         onOverflow: @escaping @Sendable () -> Void = {}) {
        self.generationID = generationID
        self.towerBytes = towerBytes
        self.maximumPendingBytes = max(1, maximumPendingBytes)
        self.onOverflow = onOverflow
        memorySampler.resetPeak()
    }

    func publish(_ event: AppInferenceEvent) {
        condition.lock()
        var overflow = false
        switch event {
        case .memorySample:
            // The writer samples on its own schedule; an inbound reading is the
            // runtime's, and there is nothing to queue.
            break
        case .prefillProgress(let done, let total):
            state.latestPrefill = PrefillProgress(done: done, total: total)
            condition.signal()
        case .token(let token):
            let bytes = token.textDelta.utf8.count
            if !state.terminalCommitted && bytes <= maximumPendingBytes - state.pendingBytes - state.toolCallBytes {
                state.pendingText += token.textDelta
                state.pendingBytes += bytes
                state.latestToken = token
            } else if !state.terminalCommitted { overflow = true }
        case .thinking(let token):
            let bytes = token.textDelta.utf8.count
            if !state.terminalCommitted && bytes <= maximumPendingBytes - state.pendingBytes - state.toolCallBytes {
                state.pendingThinking += token.textDelta
                state.pendingBytes += bytes
                state.latestToken = token
            } else if !state.terminalCommitted { overflow = true }
        case .toolCalls(let calls):
            let converted = calls.map(\.historical)
            if !state.terminalCommitted,
               let bytes = try? JSONEncoder().encode(converted).count,
               bytes <= maximumPendingBytes - state.pendingBytes {
                state.toolCalls = converted
                state.toolCallBytes = bytes
            } else if !state.terminalCommitted { overflow = true }
        case .finished(let diagnostics):
            if !state.terminalCommitted {
                state.terminal = terminal(.finished, diagnostics: diagnostics)
                state.terminalCommitted = true
            }
        case .cancelled(let diagnostics):
            if !state.terminalCommitted {
                state.terminal = terminal(.cancelled, diagnostics: diagnostics)
                state.terminalCommitted = true
            }
        case .failed(let error, let diagnostics):
            if !state.terminalCommitted {
                state.terminal = terminal(
                    .failed, diagnostics: diagnostics, error: error.userMessage)
                if case .malformedToolCall = error {
                    state.terminal?.errorKind = "malformedToolCall"
                }
                state.terminalCommitted = true
            }
        }
        if overflow {
            state.terminal = DecodeServiceEvent(kind: .failed, generationID: generationID,
                error: "output consumer is too slow")
            state.terminalCommitted = true
        }
        if state.terminal != nil { condition.signal() }
        condition.unlock()
        if overflow { onOverflow() }
    }

    func finish(error: Error? = nil) {
        condition.lock()
        if !state.terminalCommitted, let error {
            state.terminal = DecodeServiceEvent(
                kind: .failed, generationID: generationID, error: "\(error)")
            state.terminalCommitted = true
        }
        state.finished = true
        condition.broadcast()
        condition.unlock()
    }

    func runWriter(to handle: FileHandle, frameWriter: DecodeServiceFrameWriter? = nil) throws {
        func write(_ event: DecodeServiceEvent) throws {
            let frame = try DecodeFrameCodec.encode(event)
            if let frameWriter { try frameWriter.write(frame) }
            else { try handle.write(contentsOf: frame) }
        }
        while true {
            condition.lock()
            if state.terminal == nil, !state.finished {
                _ = condition.wait(until: Date().addingTimeInterval(0.1))
            }
            let prefill = state.latestPrefill
            let text = state.pendingText
            let thinking = state.pendingThinking
            let token = state.latestToken
            let terminal = state.terminal
            let done = state.finished
            state.latestPrefill = nil
            state.pendingText = ""
            state.pendingThinking = ""
            state.pendingBytes = 0
            state.latestToken = nil
            state.terminal = nil
            var prefillSequence: UInt64?
            if prefill != nil {
                state.sequence &+= 1
                prefillSequence = state.sequence
            }
            var tokenSequence: UInt64?
            if !text.isEmpty || !thinking.isEmpty || token != nil {
                state.sequence &+= 1
                tokenSequence = state.sequence
            }
            condition.unlock()

            _ = memorySampler.sample()

            if prefill == nil, text.isEmpty, thinking.isEmpty,
               token == nil, terminal == nil, !done {
                let snapshot = DecodeServiceEvent(
                    kind: .memory, generationID: generationID,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes())
                try write(snapshot)
                continue
            }
            if let prefill, let prefillSequence {
                let snapshot = DecodeServiceEvent(
                    kind: .prefill, generationID: generationID,
                    sequence: prefillSequence,
                    prefillDone: prefill.done, prefillTotal: prefill.total,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes())
                try write(snapshot)
            }
            if !text.isEmpty || !thinking.isEmpty || token != nil {
                let elapsed = token?.elapsedDecodeSeconds ?? 0
                let count = (token?.index ?? -1) + 1
                let snapshot = DecodeServiceEvent(
                    kind: .snapshot, generationID: generationID,
                    sequence: tokenSequence ?? 0, textDelta: text,
                    thinkingDelta: thinking.isEmpty ? nil : thinking,
                    tokenCount: count,
                    decodeSeconds: elapsed,
                    tokensPerSecond: elapsed > 0 ? Double(count) / elapsed : 0,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes())
                try write(snapshot)
            }
            if var terminal {
                if terminal.kind == .finished {
                    condition.lock()
                    let calls = state.toolCalls
                    condition.unlock()
                    terminal.toolCalls = calls.isEmpty ? nil : calls
                }
                try write(terminal)
            }
            if terminal != nil { return }
            if done { return }
        }
    }

    private func terminal(_ kind: DecodeServiceEventKind,
                          diagnostics: AppDiagnostics?,
                          error: String? = nil) -> DecodeServiceEvent {
        var event = DecodeServiceEvent(
            kind: kind, generationID: generationID,
            tokenCount: diagnostics?.generatedTokens ?? 0,
            promptTokenCount: diagnostics?.promptTokenCount,
            prefillSeconds: diagnostics?.prefillSeconds,
            timeToFirstTokenSeconds: diagnostics?.timeToFirstTokenSeconds,
            decodeSeconds: diagnostics?.decodeSeconds ?? 0,
            tokensPerSecond: diagnostics?.tokensPerSecond ?? 0,
            stopReason: diagnostics?.stopReason.rawValue,
            error: error,
            currentMemoryBytes: memorySampler.sample(),
            peakMemoryBytes: memorySampler.peakBytes,
            visionTowerMappedBytes: diagnostics?.visionTowerMappedBytes,
            prefill: diagnostics?.prefill.map(Self.prefillDiagnostics),
            runner: diagnostics?.runner.map(Self.runnerDiagnostics),
            droppedTurns: diagnostics?.droppedTurns)
        event.cachedPromptTokens = diagnostics?.cachedPromptTokens
        event.conversationCacheSource = diagnostics?.conversationCacheSource
        event.retainedConversations = diagnostics?.retainedConversations
        event.retainedConversationBytes = diagnostics?.retainedConversationBytes
        return event
    }

    private static func prefillDiagnostics(_ value: PrefillExecutionDiagnostics)
        -> DecodePrefillDiagnostics {
        DecodePrefillDiagnostics(
            requestedMode: value.requestedMode.rawValue,
            executedMode: value.executedMode.rawValue,
            kvStorageMode: value.kvStorageMode?.rawValue,
            chunkCompleteness: value.chunkCompleteness.rawValue,
            unsupportedReason: value.unsupportedReason)
    }

    private static func runnerDiagnostics(_ value: AppRunnerDiagnostics)
        -> DecodeRunnerDiagnostics {
        DecodeRunnerDiagnostics(
            cb1MillisecondsPerToken: value.cb1MillisecondsPerToken,
            ioMillisecondsPerToken: value.ioMillisecondsPerToken,
            cb2MillisecondsPerToken: value.cb2MillisecondsPerToken,
            headMillisecondsPerToken: value.headMillisecondsPerToken,
            rdadviseMillisecondsPerToken: value.rdadviseMillisecondsPerToken,
            rdadviseCallsPerToken: value.rdadviseCallsPerToken,
            rdadviseMegabytesPerToken: value.rdadviseMegabytesPerToken,
            rdadviseSkippedPerToken: value.rdadviseSkippedPerToken,
            rdadviseFailures: value.rdadviseFailures,
            expertReads: value.expertReads,
            exposedPrefetchWaitMillisecondsPerToken: value.exposedPrefetchWaitMillisecondsPerToken)
    }
}
