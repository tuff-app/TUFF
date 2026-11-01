import Foundation
import TUFFInferenceTypes

public struct DecodeRuntimeOptions: Codable, Sendable, Equatable {
    public var expertCacheSlots: Int
    public var expertCachePolicy: String
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int
    public var rdadvisePolicy: String
    public var modelVerification: String
    public var visionResidencyPolicy: String?
    /// Nil selects full residency; an empty list streams every layer.
    public var denseResidentLayers: [Int]?
    public var denseWeightBudgetBytes: UInt64?
    public var retainedConversationBytes: Int?

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: String = "lfu",
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                rdadvisePolicy: String = "off",
                modelVerification: String = "full-sha256",
                visionResidencyPolicy: String? = nil,
                denseResidentLayers: [Int]? = nil,
                denseWeightBudgetBytes: UInt64? = nil, retainedConversationBytes: Int? = nil) {
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillEnabled = prefillEnabled
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
        self.modelVerification = modelVerification
        self.visionResidencyPolicy = visionResidencyPolicy
        self.denseResidentLayers = denseResidentLayers
        self.denseWeightBudgetBytes = denseWeightBudgetBytes
        self.retainedConversationBytes = retainedConversationBytes
    }
}

public struct DecodeImageAttachment: Codable, Sendable, Equatable {
    public var id: UUID
    public var path: String
    public var displayName: String
    public var encodedBytes: Int
    public var sha256: String

    public init(id: UUID, path: String, displayName: String,
                encodedBytes: Int, sha256: String) {
        self.id = id
        self.path = path
        self.displayName = displayName
        self.encodedBytes = encodedBytes
        self.sha256 = sha256
    }
}

public struct DecodeLoadRequest: Codable, Sendable {
    public var modelPath: String
    public var maxContextTokens: Int
    public var runtimeOptions: DecodeRuntimeOptions
    public var forceLogitsHead: Bool
    public var requestID: UUID

    public init(modelPath: String, maxContextTokens: Int,
                runtimeOptions: DecodeRuntimeOptions = DecodeRuntimeOptions(),
                forceLogitsHead: Bool = false,
                requestID: UUID = UUID()) {
        self.modelPath = modelPath
        self.maxContextTokens = maxContextTokens
        self.runtimeOptions = runtimeOptions
        self.forceLogitsHead = forceLogitsHead
        self.requestID = requestID
    }
}

/// A tool result as the model reads it.
public struct DecodeToolResult: Codable, Sendable, Equatable {
    public var callID: String
    public var name: String
    public var content: String

    public init(callID: String, name: String, content: String) {
        self.callID = callID
        self.name = name
        self.content = content
    }
}

/// One generation that ended in tool calls, and the results that answered
/// them. Rendered as the model's native tool turns, never as user text.
public struct DecodeToolRound: Codable, Sendable, Equatable {
    public var thinking: String?
    public var content: String
    public var calls: [InferenceToolCall]
    public var results: [DecodeToolResult]

    public init(thinking: String? = nil, content: String = "",
                calls: [InferenceToolCall], results: [DecodeToolResult]) {
        self.thinking = thinking
        self.content = content
        self.calls = calls
        self.results = results
    }
}

/// One completed exchange carried across the IPC boundary as conversation
/// history. Mirrors `AppChatTurn`, which lives in the app module the service
/// cannot import.
public struct DecodeChatTurn: Codable, Sendable, Equatable {
    public var prompt: String
    public var response: String
    public var thinking: String?
    /// Images this turn was sent with. Optional on the wire, so an older client
    /// that never sends them and a newer service still agree; absent means the
    /// turn contributes text only, which is what every v2 client sent.
    public var images: [DecodeImageAttachment]?
    /// Tool rounds before `response`. Absent from clients before 8.0.
    public var toolRounds: [DecodeToolRound]?

    public init(prompt: String, response: String, thinking: String? = nil,
                images: [DecodeImageAttachment]? = nil,
                toolRounds: [DecodeToolRound]? = nil) {
        self.prompt = prompt
        self.response = response
        self.thinking = thinking
        self.images = images
        self.toolRounds = toolRounds
    }
}

public enum DecodeChatReasoning: String, Codable, Sendable, Equatable {
    case off
    case on
}

public enum DecodeGPTOSSReasoningEffort: String, Codable, Sendable, Equatable {
    case low
    case medium
    case high
}

public struct DecodeGenerationRequest: Codable, Sendable {
    public var prompt: String
    /// Prior turns, oldest first. Decoded as empty when absent, so an older
    /// client and a newer service still agree.
    public var history: [DecodeChatTurn]
    /// The system prompt, rendered ahead of the conversation. Absent on a
    /// request from a client that predates the field.
    public var systemPrompt: String?
    /// Text the model continues rather than a fresh reply. Absent on a request
    /// from a client that predates the field.
    public var assistantPrefix: String?
    public var imageAttachments: [DecodeImageAttachment]?
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    /// Decoded as off for compatibility with v1 app clients.
    public var reasoning: DecodeChatReasoning
    public var reasoningEffort: DecodeGPTOSSReasoningEffort?
    public var preserveThinking: Bool
    public var temperature: Float
    /// Carried explicitly, and optional because nil means "no cut". Leaving
    /// them off the wire did not fall back to the sender's settings: the
    /// service rebuilt the request from its own initializer defaults, so
    /// turning Top-K off, or setting any value other than 64 / 0.95, was
    /// silently ignored on the only client the app ships with.
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var runtimeOptions: DecodeRuntimeOptions
    public var generationID: UUID
    /// Declared tools, rounds already run for this message, and the chat it
    /// belongs to. All absent from clients before 8.0.
    public var tools: [InferenceFunctionDefinition]?
    public var currentRounds: [DecodeToolRound]?
    public var conversationKey: String?
    public var preservesFullHistory: Bool

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        prompt = try container.decode(String.self, forKey: .prompt)
        systemPrompt = try container.decodeIfPresent(
            String.self, forKey: .systemPrompt)
        assistantPrefix = try container.decodeIfPresent(
            String.self, forKey: .assistantPrefix)
        history = try container.decodeIfPresent(
            [DecodeChatTurn].self, forKey: .history) ?? []
        imageAttachments = try container.decodeIfPresent(
            [DecodeImageAttachment].self, forKey: .imageAttachments)
        maxNewTokens = try container.decode(Int.self, forKey: .maxNewTokens)
        maxContextTokens = try container.decode(Int.self, forKey: .maxContextTokens)
        reasoning = try container.decodeIfPresent(
            DecodeChatReasoning.self, forKey: .reasoning) ?? .off
        reasoningEffort = try container.decodeIfPresent(
            DecodeGPTOSSReasoningEffort.self, forKey: .reasoningEffort)
        preserveThinking = try container.decodeIfPresent(
            Bool.self, forKey: .preserveThinking) ?? false
        temperature = try container.decode(Float.self, forKey: .temperature)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        repetitionPenalty = try container.decode(Float.self, forKey: .repetitionPenalty)
        seed = try container.decodeIfPresent(UInt64.self, forKey: .seed)
        runtimeOptions = try container.decode(
            DecodeRuntimeOptions.self, forKey: .runtimeOptions)
        generationID = try container.decode(UUID.self, forKey: .generationID)
        tools = try container.decodeIfPresent(
            [InferenceFunctionDefinition].self, forKey: .tools)
        currentRounds = try container.decodeIfPresent(
            [DecodeToolRound].self, forKey: .currentRounds)
        conversationKey = try container.decodeIfPresent(String.self, forKey: .conversationKey)
        preservesFullHistory = try container.decodeIfPresent(Bool.self, forKey: .preservesFullHistory) ?? false
    }

    public init(prompt: String, systemPrompt: String? = nil,
                assistantPrefix: String? = nil,
                history: [DecodeChatTurn] = [],
                imageAttachments: [DecodeImageAttachment]? = nil,
                maxNewTokens: Int, maxContextTokens: Int,
                reasoning: DecodeChatReasoning = .off,
                reasoningEffort: DecodeGPTOSSReasoningEffort? = nil,
                preserveThinking: Bool = false,
                temperature: Float, topK: Int? = nil, topP: Float? = nil,
                repetitionPenalty: Float = 1,
                seed: UInt64? = nil,
                runtimeOptions: DecodeRuntimeOptions = DecodeRuntimeOptions(),
                generationID: UUID = UUID(),
                tools: [InferenceFunctionDefinition]? = nil,
                currentRounds: [DecodeToolRound]? = nil,
                conversationKey: String? = nil, preservesFullHistory: Bool = false) {
        self.prompt = prompt
        self.systemPrompt = systemPrompt
        self.assistantPrefix = assistantPrefix
        self.history = history
        self.imageAttachments = imageAttachments
        self.maxNewTokens = maxNewTokens
        self.maxContextTokens = maxContextTokens
        self.reasoning = reasoning
        self.reasoningEffort = reasoningEffort
        self.preserveThinking = preserveThinking
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.runtimeOptions = runtimeOptions
        self.generationID = generationID
        self.tools = tools
        self.currentRounds = currentRounds
        self.conversationKey = conversationKey
        self.preservesFullHistory = preservesFullHistory
    }
}

public enum DecodeServiceCommand: Codable, Sendable {
    case hello(DecodeEngineHello)
    case load(DecodeLoadRequest)
    case generate(DecodeGenerationRequest)
    /// Multiplexed sessions share one loaded model but retain independent state.
    case generateSession(sessionID: UUID, request: DecodeGenerationRequest)
    case cancelRequest(UUID)
    case endSession(sessionID: UUID, requestID: UUID)
    case cancelWarmup
    case cancel
    case unload(UUID)
    case shutdown
}

public enum DecodeServiceEventKind: String, Codable, Sendable {
    case hello
    case loading
    case ready
    case prefill
    case snapshot
    /// Carries a live memory reading while image encoding or another silent
    /// phase has not produced progress or tokens yet.
    case memory
    case finished
    case cancelled
    case failed
    case unloaded
}

public struct DecodeRunnerDiagnostics: Codable, Sendable, Equatable {
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

    public init(cb1MillisecondsPerToken: Double,
                ioMillisecondsPerToken: Double,
                cb2MillisecondsPerToken: Double,
                headMillisecondsPerToken: Double,
                rdadviseMillisecondsPerToken: Double,
                rdadviseCallsPerToken: Double,
                rdadviseMegabytesPerToken: Double,
                rdadviseSkippedPerToken: Double,
                rdadviseFailures: UInt64,
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

public struct DecodePrefillDiagnostics: Codable, Sendable, Equatable {
    public var requestedMode: String
    public var executedMode: String
    public var kvStorageMode: String?
    public var chunkCompleteness: String
    public var unsupportedReason: String?

    public init(requestedMode: String, executedMode: String,
                kvStorageMode: String?, chunkCompleteness: String,
                unsupportedReason: String?) {
        self.requestedMode = requestedMode
        self.executedMode = executedMode
        self.kvStorageMode = kvStorageMode
        self.chunkCompleteness = chunkCompleteness
        self.unsupportedReason = unsupportedReason
    }
}

public struct DecodeServiceEvent: Codable, Sendable {
    public var capabilities: DecodeEngineCapabilities?
    public var kind: DecodeServiceEventKind
    public var generationID: UUID
    public var sequence: UInt64
    public var textDelta: String
    public var thinkingDelta: String?
    public var tokenCount: Int
    public var promptTokenCount: Int?
    public var prefillDone: Int?
    public var prefillTotal: Int?
    public var prefillSeconds: Double?
    public var timeToFirstTokenSeconds: Double?
    public var decodeSeconds: Double
    public var tokensPerSecond: Double
    public var stopReason: String?
    public var error: String?
    public var currentMemoryBytes: UInt64?
    public var peakMemoryBytes: UInt64?
    /// Bytes of image tower the inference process holds mapped, or nil when
    /// it has no vision runtime.
    public var visionTowerMappedBytes: UInt64?
    public var prefill: DecodePrefillDiagnostics?
    public var runner: DecodeRunnerDiagnostics?
    /// Turns the renderer dropped to make the prompt fit. Absent on events from
    /// a service that predates the field.
    public var droppedTurns: Int?
    /// Calls the model made, on a finished event. Parsed and validated
    /// against the declared tool names; nothing has run.
    public var toolCalls: [InferenceToolCall]?
    /// A machine-readable failure, so the app can tell a malformed tool call
    /// from other errors.
    public var errorKind: String?
    public var cachedPromptTokens: Int?
    public var conversationCacheSource: String?
    public var retainedConversations: Int?
    public var retainedConversationBytes: Int?

    public init(kind: DecodeServiceEventKind, generationID: UUID,
                sequence: UInt64 = 0, textDelta: String = "",
                thinkingDelta: String? = nil,
                tokenCount: Int = 0, promptTokenCount: Int? = nil,
                prefillDone: Int? = nil, prefillTotal: Int? = nil,
                prefillSeconds: Double? = nil,
                timeToFirstTokenSeconds: Double? = nil,
                decodeSeconds: Double = 0, tokensPerSecond: Double = 0,
                stopReason: String? = nil, error: String? = nil,
                currentMemoryBytes: UInt64? = nil, peakMemoryBytes: UInt64? = nil,
                visionTowerMappedBytes: UInt64? = nil,
                prefill: DecodePrefillDiagnostics? = nil,
                runner: DecodeRunnerDiagnostics? = nil,
                droppedTurns: Int? = nil) {
        self.kind = kind
        self.generationID = generationID
        self.sequence = sequence
        self.textDelta = textDelta
        self.thinkingDelta = thinkingDelta
        self.tokenCount = tokenCount
        self.promptTokenCount = promptTokenCount
        self.prefillDone = prefillDone
        self.prefillTotal = prefillTotal
        self.prefillSeconds = prefillSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.decodeSeconds = decodeSeconds
        self.tokensPerSecond = tokensPerSecond
        self.stopReason = stopReason
        self.error = error
        self.currentMemoryBytes = currentMemoryBytes
        self.peakMemoryBytes = peakMemoryBytes
        self.visionTowerMappedBytes = visionTowerMappedBytes
        self.prefill = prefill
        self.runner = runner
        self.droppedTurns = droppedTurns
    }
}

public enum DecodeFrameCodec {
    public static let maximumPayloadBytes = 4 * 1_024 * 1_024

    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        let payload = try JSONEncoder().encode(value)
        guard payload.count <= maximumPayloadBytes else { throw DecodeFrameError.oversized }
        var length = UInt32(payload.count).littleEndian
        var frame = withUnsafeBytes(of: &length) { Data($0) }
        frame.append(payload)
        return frame
    }

    public static func read<T: Decodable>(_ type: T.Type, from handle: FileHandle) throws -> T {
        let header = try readExactly(4, from: handle)
        let count = header.withUnsafeBytes { raw -> UInt32 in
            raw.loadUnaligned(as: UInt32.self).littleEndian
        }
        guard count <= maximumPayloadBytes else { throw DecodeFrameError.oversized }
        let payload = try readExactly(Int(count), from: handle)
        return try JSONDecoder().decode(type, from: payload)
    }

    private static func readExactly(_ count: Int, from handle: FileHandle) throws -> Data {
        var result = Data()
        result.reserveCapacity(count)
        while result.count < count {
            guard let chunk = try handle.read(upToCount: count - result.count), !chunk.isEmpty else {
                throw DecodeFrameError.unexpectedEOF
            }
            result.append(chunk)
        }
        return result
    }
}

public enum DecodeFrameError: Error {
    case oversized
    case unexpectedEOF
}
