import Foundation

/// Wire types shared by the GUI, the per-user serving host and API adapters.
/// This protocol is intentionally below App and Server so both can attach to
/// one inference owner without a target dependency cycle.
public enum ServingProtocol {
    public static let version = 1
    /// The handshake refusal an idle host sends once it has committed to
    /// exiting. It is not a failure of the client: the client relaunches the
    /// host and connects again.
    public static let hostRetiringReason = "host is retiring after idle exit"
}

/// What an attached client is. The host derives the public-API status from
/// live `api` attachments rather than from a value fixed at host start.
public enum ServingClientRole: String, Codable, Sendable, Equatable {
    case gui
    case api
    case tool
}

public struct ServingClientHello: Codable, Sendable, Equatable {
    public let protocolVersion: Int
    public let profileIdentity: String
    public let clientID: UUID
    public let token: String
    public let role: ServingClientRole

    public init(protocolVersion: Int = ServingProtocol.version,
                profileIdentity: String, clientID: UUID, token: String,
                role: ServingClientRole = .tool) {
        self.protocolVersion = protocolVersion
        self.profileIdentity = profileIdentity
        self.clientID = clientID
        self.token = token
        self.role = role
    }
}

public struct ServingHostHello: Codable, Sendable, Equatable {
    public let protocolVersion: Int
    public let profileIdentity: String
    public let hostID: UUID
    public let buildIdentity: String

    public init(protocolVersion: Int = ServingProtocol.version,
                profileIdentity: String, hostID: UUID, buildIdentity: String) {
        self.protocolVersion = protocolVersion
        self.profileIdentity = profileIdentity
        self.hostID = hostID
        self.buildIdentity = buildIdentity
    }
}

/// A model the host currently holds and the configuration it was loaded with.
public struct ServingLoadedModel: Codable, Sendable, Equatable {
    public let modelID: String
    public let maxContextTokens: Int
    public let runtimeOptions: DecodeRuntimeOptions
    public let forceLogitsHead: Bool

    public init(modelID: String, maxContextTokens: Int,
                runtimeOptions: DecodeRuntimeOptions, forceLogitsHead: Bool) {
        self.modelID = modelID
        self.maxContextTokens = maxContextTokens
        self.runtimeOptions = runtimeOptions
        self.forceLogitsHead = forceLogitsHead
    }
}

/// Live host state. Answered on demand so a surface never reads a value that
/// was true only when its connection opened.
public struct ServingHostStatus: Codable, Sendable, Equatable {
    public let hostID: UUID
    public let buildIdentity: String
    /// An API adapter is attached right now. Says nothing about whether its
    /// TCP listener is open; only the adapter owns that.
    public let apiAdapterAttached: Bool
    public let connectedClients: Int
    public let loadedModelIDs: [String]
    public let loadedModels: [ServingLoadedModel]
    public let activeRequests: Int
    public let openSessions: Int

    public init(hostID: UUID, buildIdentity: String, apiAdapterAttached: Bool,
                connectedClients: Int, loadedModelIDs: [String],
                loadedModels: [ServingLoadedModel] = [],
                activeRequests: Int, openSessions: Int) {
        self.hostID = hostID
        self.buildIdentity = buildIdentity
        self.apiAdapterAttached = apiAdapterAttached
        self.connectedClients = connectedClients
        self.loadedModelIDs = loadedModelIDs
        self.loadedModels = loadedModels
        self.activeRequests = activeRequests
        self.openSessions = openSessions
    }

    private enum CodingKeys: String, CodingKey {
        case hostID, buildIdentity, apiAdapterAttached, connectedClients, loadedModelIDs
        case loadedModels, activeRequests, openSessions
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hostID = try container.decode(UUID.self, forKey: .hostID)
        buildIdentity = try container.decode(String.self, forKey: .buildIdentity)
        apiAdapterAttached = try container.decode(Bool.self, forKey: .apiAdapterAttached)
        connectedClients = try container.decode(Int.self, forKey: .connectedClients)
        loadedModelIDs = try container.decode([String].self, forKey: .loadedModelIDs)
        loadedModels = try container.decodeIfPresent([ServingLoadedModel].self, forKey: .loadedModels) ?? []
        activeRequests = try container.decode(Int.self, forKey: .activeRequests)
        openSessions = try container.decode(Int.self, forKey: .openSessions)
    }
}

/// Session and request identity for one inference operation. `requestID` must
/// equal `payload.generationID`; retaining both makes routing explicit while
/// preserving the existing decode-service correlation ID.
public struct ServingGenerationRequest: Codable, Sendable {
    public let modelID: String
    public let sessionID: UUID
    public let requestID: UUID
    public let payload: DecodeGenerationRequest

    public init(modelID: String, sessionID: UUID, requestID: UUID,
                payload: DecodeGenerationRequest) throws {
        self.modelID = modelID
        self.sessionID = sessionID
        self.requestID = requestID
        self.payload = payload
        try validate()
    }

    public func validate() throws {
        guard !modelID.isEmpty, modelID.utf8.count <= 256 else {
            throw ServingProtocolError.invalidRequest("modelID must be 1–256 UTF-8 bytes")
        }
        guard requestID == payload.generationID else {
            throw ServingProtocolError.invalidRequest(
                "requestID must match payload.generationID")
        }
    }

    private enum CodingKeys: String, CodingKey { case modelID, sessionID, requestID, payload }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelID = try container.decode(String.self, forKey: .modelID)
        sessionID = try container.decode(UUID.self, forKey: .sessionID)
        requestID = try container.decode(UUID.self, forKey: .requestID)
        payload = try container.decode(DecodeGenerationRequest.self, forKey: .payload)
        try validate()
    }
}

public struct ServingModelPreparation: Codable, Sendable, Equatable {
    public let operationID: UUID
    public let modelID: String
    public let maxContextTokens: Int
    public let runtimeOptions: DecodeRuntimeOptions
    public let forceLogitsHead: Bool

    public init(operationID: UUID = UUID(), modelID: String,
                maxContextTokens: Int, runtimeOptions: DecodeRuntimeOptions,
                forceLogitsHead: Bool) throws {
        self.operationID = operationID
        self.modelID = modelID
        self.maxContextTokens = maxContextTokens
        self.runtimeOptions = runtimeOptions
        self.forceLogitsHead = forceLogitsHead
        try validate()
    }

    public func validate() throws {
        guard !modelID.isEmpty, modelID.utf8.count <= 256 else {
            throw ServingProtocolError.invalidRequest("invalid modelID")
        }
        guard (1...262_144).contains(maxContextTokens) else {
            throw ServingProtocolError.invalidRequest("invalid context size")
        }
        let options = runtimeOptions
        guard [4, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512]
                .contains(options.expertCacheSlots),
              [32, 64, 128, 256, 512, 1_024, 2_048]
                .contains(options.prefillChunkTokens),
              ["lfu", "lru"].contains(options.expertCachePolicy),
              ["off", "default", "bounded", "adaptive"].contains(options.rdadvisePolicy),
              ["full-sha256", "trusted-install"].contains(options.modelVerification),
              options.visionResidencyPolicy.map({ ["on-demand", "keep-ready"].contains($0) }) ?? true,
              options.retainedConversationBytes.map({ (0...(1 << 30)).contains($0) }) ?? true,
              options.denseWeightBudgetBytes.map({ (1...(1 << 50)).contains($0) }) ?? true,
              options.denseResidentLayers.map({
                  $0.count <= 1_024 && $0.allSatisfy { (0..<1_024).contains($0) }
                      && Set($0).count == $0.count
              }) ?? true,
              !options.prefillEnabled || options.expertCacheSlots >= 16 else {
            throw ServingProtocolError.invalidRequest("invalid runtime options")
        }
    }

    private enum CodingKeys: String, CodingKey {
        case operationID, modelID, maxContextTokens, runtimeOptions, forceLogitsHead
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        operationID = try container.decode(UUID.self, forKey: .operationID)
        modelID = try container.decode(String.self, forKey: .modelID)
        maxContextTokens = try container.decode(Int.self, forKey: .maxContextTokens)
        runtimeOptions = try container.decode(DecodeRuntimeOptions.self, forKey: .runtimeOptions)
        forceLogitsHead = try container.decode(Bool.self, forKey: .forceLogitsHead)
        try validate()
    }
}

public enum ServingLoadPhase: String, Codable, Sendable, Equatable {
    case preparingEngine
    case validatingDirectory
    case tokenizer
    case verifyingWeights
    case preparingRunner
    case mappingImageTower
}

public enum ServingLoadState: Codable, Sendable, Equatable {
    case loading(ServingLoadPhase)
    case cancelling
    case unloading
    case ready(loadSeconds: Double)
    case cancelled
    case unloaded
    case failed(String)
}

public struct ServingRequestIdentity: Codable, Sendable, Equatable, Hashable {
    public let clientID: UUID
    public let modelID: String
    public let sessionID: UUID
    public let requestID: UUID

    public init(clientID: UUID, modelID: String, sessionID: UUID, requestID: UUID) {
        self.clientID = clientID
        self.modelID = modelID
        self.sessionID = sessionID
        self.requestID = requestID
    }

    public init(clientID: UUID, request: ServingGenerationRequest) {
        self.init(clientID: clientID, modelID: request.modelID, sessionID: request.sessionID,
                  requestID: request.requestID)
    }
}

public enum ServingCommand: Codable, Sendable {
    case prepare(ServingModelPreparation)
    case cancelPreparation(operationID: UUID)
    case unloadModel(operationID: UUID, modelID: String)
    case start(ServingGenerationRequest)
    case cancel(sessionID: UUID, requestID: UUID)
    case endSession(sessionID: UUID)
    case hostStatus(operationID: UUID)

    public func validate() throws {
        switch self {
        case .prepare(let preparation): try preparation.validate()
        case .cancelPreparation, .endSession, .hostStatus: break
        case .unloadModel(_, let modelID):
            guard !modelID.isEmpty, modelID.utf8.count <= 256 else {
                throw ServingProtocolError.invalidRequest("invalid modelID")
            }
        case .start(let request): try request.validate()
        case .cancel: break
        }
    }
}

extension ServingCommand {
    /// The event that answers this command with a failure. Every refusal is
    /// keyed to the one operation that caused it, so it can never fail other
    /// streams sharing the connection. Commands with no response stream
    /// (`cancel`, `cancelPreparation`, `hostStatus`) have no refusal event.
    public func refusal(clientID: UUID, message: String) -> ServingEvent? {
        switch self {
        case .prepare(let preparation):
            .lifecycle(operationID: preparation.operationID, modelID: preparation.modelID,
                       state: .failed(message))
        case .unloadModel(let operationID, let modelID):
            .lifecycle(operationID: operationID, modelID: modelID, state: .failed(message))
        case .start(let request):
            .failed(identity: ServingRequestIdentity(clientID: clientID, request: request),
                    message: message)
        case .endSession(let sessionID):
            .sessionEndFailed(sessionID: sessionID, message: message)
        case .cancelPreparation, .cancel, .hostStatus:
            nil
        }
    }
}

public enum ServingQueueState: String, Codable, Sendable, Equatable {
    case queued
    case admitted
}

/// Progress events are multiplexed by both chat and request. The nested
/// decode event retains existing diagnostics, token sequence and terminal
/// details; identity must agree with its generationID.
public enum ServingEvent: Codable, Sendable {
    case lifecycle(operationID: UUID, modelID: String, state: ServingLoadState)
    case queue(identity: ServingRequestIdentity, state: ServingQueueState,
               position: Int?)
    case generation(identity: ServingRequestIdentity, event: DecodeServiceEvent)
    case failed(identity: ServingRequestIdentity?, message: String)
    case sessionEnded(sessionID: UUID)
    /// Cleanup did not finish. The session keeps its reservation and the
    /// caller may retry.
    case sessionEndFailed(sessionID: UUID, message: String)
    case hostStatus(operationID: UUID, status: ServingHostStatus)

    public func validate() throws {
        if case .generation(let identity, let event) = self {
            guard identity.requestID == event.generationID else {
                throw ServingProtocolError.invalidEvent(
                    "requestID must match event.generationID")
            }
        }
    }
}

/// The first frame on a private socket is a hello. Only after authentication
/// and profile/version validation may command or event frames be exchanged.
public enum ServingIPCMessage: Codable, Sendable {
    case clientHello(ServingClientHello)
    case hostHello(ServingHostHello)
    case command(ServingCommand)
    case event(ServingEvent)
    case rejected(String)
}

/// The one phrase that identifies "another resident model holds the memory"
/// failures once they have crossed the helper boundary as text.
public enum ServingMemoryPressure {
    public static let phrase = "holds model memory"
    /// Image work reserves the whole budget; the helper reports this phrase
    /// when other leases are still held.
    public static let exclusivePhrase = "needs exclusive model memory"

    public static func isExclusiveBusy(_ message: String) -> Bool {
        message.contains(exclusivePhrase)
    }

    public static func isMemoryBusy(_ error: any Error) -> Bool {
        "\(error)".contains(phrase) || error.localizedDescription.contains(phrase)
    }
}

public enum ServingProtocolError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalidRequest(String)
    case invalidEvent(String)
    case unsupportedVersion(Int)
    case profileMismatch
    case authenticationFailed
    case peerMismatch
    case frameTooLarge

    public var description: String {
        switch self {
        case .invalidRequest(let message): "Invalid serving request: \(message)"
        case .invalidEvent(let message): "Invalid serving event: \(message)"
        case .unsupportedVersion(let version): "Unsupported serving protocol version \(version)"
        case .profileMismatch: "Serving host belongs to a different TUFF profile"
        case .authenticationFailed: "Serving host authentication failed"
        case .peerMismatch: "Serving socket peer belongs to a different user"
        case .frameTooLarge: "Serving IPC frame exceeds the size limit"
        }
    }
}
