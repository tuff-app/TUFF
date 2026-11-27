import Foundation
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFModelMetadata

/// A model a request was routed to.
public struct ServerRoutedModel: Equatable, Sendable {
    /// The canonical API identifier echoed in responses.
    public let id: String
    /// What the client sent. Validation compares the request against this, so
    /// an alias such as `default` is accepted where the router allows it.
    public let requestedID: String
    public let dialect: ChatDialect
    /// `ready`, `missing`, `disabled` or `unsupported`. For a model that is not
    /// loaded yet this is the best pre-load answer; the loaded session's own
    /// capability is checked again before generation.
    public let visionCapability: String

    public init(id: String, requestedID: String, dialect: ChatDialect, visionCapability: String) {
        self.id = id
        self.requestedID = requestedID
        self.dialect = dialect
        self.visionCapability = visionCapability
    }
}

/// Everything the HTTP layer needs from whatever owns the models: which ones
/// exist, how a request's `model` resolves, and a scheduled slot to run in.
public protocol ServerModelProvider: Sendable {
    /// Vision capability the body parser stages images against, before the
    /// request's model is known. A router answers `ready` when any of its
    /// models could take images; the routed model is checked afterwards.
    var parserVisionCapability: String { get }
    func modelList() -> OpenAIModelList
    func health() async -> ServerHealth
    func route(_ requestedModel: String) throws -> ServerRoutedModel
    func run<Prepared: Sendable, T: Sendable>(
        _ model: ServerRoutedModel,
        onQueued: @escaping @Sendable () -> Void,
        prepare: @escaping @Sendable (any ServerInferenceBackend) async throws -> Prepared,
        operation: @escaping @Sendable (any ServerInferenceBackend, Prepared) async throws -> T
    ) async throws -> T
    var queuedCount: Int { get async }
    var isActive: Bool { get async }
    func shutdown() async
    /// The returned closure receives the finished completion so the provider
    /// can learn from the run.
    func completionRecorder(for model: ServerRoutedModel, request: ValidatedChatRequest)
        -> (@Sendable (ServerCompletion) -> Void)?
}

extension ServerModelProvider {
    public func completionRecorder(for model: ServerRoutedModel, request: ValidatedChatRequest)
        -> (@Sendable (ServerCompletion) -> Void)? { nil }
}

/// What `GET /health` reports. A client can tell a running server from a
/// loaded model here, and so avoid triggering an expensive load, or a second
/// heavyweight engine, by accident.
public struct ServerHealth: Encodable, Equatable, Sendable {
    public var status = "ok"
    public var mode: String?
    public var vision: String
    /// The API ID of the model in memory, or nil when none is loaded.
    public var loadedModel: String?
    public var activeRequests: Int
    public var queuedRequests: Int

    public init(mode: String? = nil, vision: String, loadedModel: String?,
                activeRequests: Int, queuedRequests: Int) {
        self.mode = mode
        self.vision = vision
        self.loadedModel = loadedModel
        self.activeRequests = activeRequests
        self.queuedRequests = queuedRequests
    }

    enum CodingKeys: String, CodingKey {
        case status, mode, vision
        case loadedModel = "loaded_model"
        case activeRequests = "active_requests"
        case queuedRequests = "queued_requests"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(status, forKey: .status)
        try container.encodeIfPresent(mode, forKey: .mode)
        try container.encode(vision, forKey: .vision)
        // Present as null rather than missing, so "nothing loaded" is explicit.
        try container.encode(loadedModel, forKey: .loadedModel)
        try container.encode(activeRequests, forKey: .activeRequests)
        try container.encode(queuedRequests, forKey: .queuedRequests)
    }
}
