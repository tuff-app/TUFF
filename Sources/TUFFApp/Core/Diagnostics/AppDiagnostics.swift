import Foundation
import TUFFInferenceTypes
import TUFFRuntimeDescriptions

public enum AppStopReason: String, Equatable, Sendable {
    case maxTokens
    case cancelled
    case failed
    case eos
    case endOfTurn
    case stopString
    case toolCalls

    /// How the end of a reply reads in the status line. A finished answer is
    /// just "Done"; the raw case names leaked words like "endOfTurn".
    public var statusLabel: String {
        switch self {
        case .eos, .endOfTurn, .stopString, .toolCalls: "Done"
        case .maxTokens: "Stopped at the length limit"
        case .cancelled: "Stopped"
        case .failed: "Failed"
        }
    }
}

public struct AppRunnerDiagnostics: Equatable, Sendable {
    public var cb1MillisecondsPerToken: Double
    public var ioMillisecondsPerToken: Double
    public var cb2MillisecondsPerToken: Double
    public var headMillisecondsPerToken: Double
    public var rdadviseMillisecondsPerToken: Double
    public var rdadviseCallsPerToken: Double
    public var rdadviseMegabytesPerToken: Double
    public var rdadviseSkippedPerToken: Double
    /// Decode-only logical reads; CPU/GPU durations may overlap.
    public var expertReads: ExpertReadMetrics?
    public var exposedPrefetchWaitMillisecondsPerToken: Double?
    public var rdadviseFailures: UInt64

    public init(cb1MillisecondsPerToken: Double = 0,
                ioMillisecondsPerToken: Double = 0,
                cb2MillisecondsPerToken: Double = 0,
                headMillisecondsPerToken: Double = 0,
                rdadviseMillisecondsPerToken: Double = 0,
                rdadviseCallsPerToken: Double = 0,
                rdadviseMegabytesPerToken: Double = 0,
                rdadviseSkippedPerToken: Double = 0,
                rdadviseFailures: UInt64 = 0,
                expertReads: ExpertReadMetrics? = nil,
                exposedPrefetchWaitMillisecondsPerToken: Double? = nil) {
        self.cb1MillisecondsPerToken = cb1MillisecondsPerToken
        self.ioMillisecondsPerToken = ioMillisecondsPerToken
        self.cb2MillisecondsPerToken = cb2MillisecondsPerToken
        self.headMillisecondsPerToken = headMillisecondsPerToken
        self.rdadviseMillisecondsPerToken = rdadviseMillisecondsPerToken
        self.rdadviseCallsPerToken = rdadviseCallsPerToken
        self.rdadviseMegabytesPerToken = rdadviseMegabytesPerToken
        self.rdadviseSkippedPerToken = rdadviseSkippedPerToken
        self.expertReads = expertReads
        self.exposedPrefetchWaitMillisecondsPerToken = exposedPrefetchWaitMillisecondsPerToken
        self.rdadviseFailures = rdadviseFailures
    }
}

public struct AppDiagnostics: Equatable, Sendable {
    public var generatedTokens: Int
    public var stopReason: AppStopReason
    public var promptTokenCount: Int?
    public var prefillSeconds: Double?
    public var timeToFirstTokenSeconds: Double?
    public var decodeSeconds: Double
    public var tokensPerSecond: Double
    public var peakMemoryBytes: UInt64?
    public var visionTowerMappedBytes: UInt64?
    public var runtimeOptions: AppRuntimeOptions
    public var prefill: PrefillExecutionDiagnostics?
    public var runner: AppRunnerDiagnostics?
    /// Turns dropped from the start of the conversation to make the prompt fit.
    ///
    /// The renderer has always computed this and nothing ever read it, so a long
    /// chat silently forgot its beginning. Carried here so the app can say so.
    public var droppedTurns: Int = 0
    /// Prompt tokens reused from retained state rather than prefilled.
    public var cachedPromptTokens: Int?
    /// Where the reused state came from: the conversation the runner already
    /// held, a retained snapshot, or none.
    public var conversationCacheSource: String?
    /// Retained conversations and their bytes after this request.
    public var retainedConversations: Int?
    public var retainedConversationBytes: Int?
    /// Tool rounds and tool time for the answer this diagnostic finished.
    public var toolRounds: Int = 0
    public var toolSeconds: Double?

    public var requestStartTimeToFirstTokenSeconds: Double? {
        guard let prefillSeconds, let timeToFirstTokenSeconds else { return nil }
        return prefillSeconds + timeToFirstTokenSeconds
    }

    public var prefillTokensPerSecond: Double? {
        guard let promptTokenCount,
              let prefillSeconds,
              prefillSeconds.isFinite,
              prefillSeconds > 0 else {
            return nil
        }
        let computedTokens = promptTokenCount - (cachedPromptTokens ?? 0)
        guard computedTokens > 0 else { return nil }
        let rate = Double(computedTokens) / prefillSeconds
        return rate.isFinite ? rate : nil
    }

    public init(generatedTokens: Int,
                stopReason: AppStopReason,
                promptTokenCount: Int? = nil,
                prefillSeconds: Double? = nil,
                timeToFirstTokenSeconds: Double?,
                decodeSeconds: Double,
                tokensPerSecond: Double,
                peakMemoryBytes: UInt64?,
                visionTowerMappedBytes: UInt64? = nil,
                runtimeOptions: AppRuntimeOptions,
                prefill: PrefillExecutionDiagnostics? = nil,
                runner: AppRunnerDiagnostics? = nil,
                droppedTurns: Int = 0) {
        self.generatedTokens = generatedTokens
        self.stopReason = stopReason
        self.promptTokenCount = promptTokenCount
        self.prefillSeconds = prefillSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.decodeSeconds = decodeSeconds
        self.tokensPerSecond = tokensPerSecond
        self.peakMemoryBytes = peakMemoryBytes
        self.visionTowerMappedBytes = visionTowerMappedBytes
        self.runtimeOptions = runtimeOptions
        self.prefill = prefill
        self.runner = runner
        self.droppedTurns = droppedTurns
    }
}

public struct AppTokenEvent: Equatable, Sendable {
    public var index: Int
    public var textDelta: String
    public var elapsedDecodeSeconds: Double

    public init(index: Int, textDelta: String, elapsedDecodeSeconds: Double) {
        self.index = index
        self.textDelta = textDelta
        self.elapsedDecodeSeconds = elapsedDecodeSeconds
    }
}

public enum AppInferenceEvent: Equatable, Sendable {
    case prefillProgress(done: Int, total: Int)
    case memorySample
    case token(AppTokenEvent)
    case thinking(AppTokenEvent)
    /// Calls the model made, emitted before `.finished`. The calls are
    /// complete and parsed; nothing has been executed.
    case toolCalls([AppToolCall])
    case finished(AppDiagnostics)
    case cancelled(AppDiagnostics)
    case failed(AppInferenceError, partial: AppDiagnostics?)
}
