import CryptoKit
import Foundation
import TUFFModelCatalog
import TUFFInferenceTypes

public enum ServerInferenceEvent: Codable, Equatable, Sendable {
    case content(String)
    /// Text from the model's reasoning channel. Never mixed into content or
    /// tool arguments.
    case reasoning(String)
    case toolCall(ParsedToolCall)
}

public struct ServerCompletion: Codable, Equatable, Sendable {
    public let content: String
    /// The model's reasoning, separate from the answer. Empty when the model
    /// did not reason.
    public let reasoning: String
    public let toolCalls: [ParsedToolCall]
    public let finishReason: String
    public let usage: OpenAIUsage
    public var timingSeconds: [String: Double]

    public init(content: String,
                reasoning: String = "",
                toolCalls: [ParsedToolCall],
                finishReason: String,
                usage: OpenAIUsage,
                timingSeconds: [String: Double] = [:]) {
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
        self.timingSeconds = timingSeconds
    }
}

public struct ServerPreparedRequest: Sendable {
    public let request: ValidatedChatRequest
    package let promptIDs: [Int32]?
    package let timingSeconds: [String: Double]

    package let remotePreparationID: UUID?
    package let remotePromptTokenCount: Int?
    public var promptTokenCount: Int? { remotePromptTokenCount ?? promptIDs?.count }
    /// Whether generation events reach the client as they arrive: a stream,
    /// or an API adapter that keeps state from them. When false, a failed
    /// attempt's events were only counted for timings, so it can be retried.
    package var eventsReachClient = true
    /// Called when the backend discards an attempt whose events never reached
    /// the client and starts another.
    package var onDiscardedAttempt: (@Sendable () -> Void)?

    package init(request: ValidatedChatRequest, promptIDs: [Int32]? = nil,
         timingSeconds: [String: Double] = [:],
         remotePreparationID: UUID? = nil, remotePromptTokenCount: Int? = nil) {
        self.remotePreparationID = remotePreparationID
        self.remotePromptTokenCount = remotePromptTokenCount
        self.request = request
        self.promptIDs = promptIDs
        self.timingSeconds = timingSeconds
    }
}

public protocol ServerInferenceBackend: Sendable {
    var visionCapability: String { get }
    func unload() async
    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest
    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
    func generate(_ prepared: ServerPreparedRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
}

public struct ServerCoordinatorActivity: Equatable, Sendable {
    public let activeRequests: Int
    public let queuedRequests: Int

    public init(activeRequests: Int, queuedRequests: Int) {
        self.activeRequests = activeRequests
        self.queuedRequests = queuedRequests
    }
}

public extension ServerInferenceBackend {
    var visionCapability: String { "missing" }
    func unload() async {}
    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        ServerPreparedRequest(request: request)
    }

    func generate(
        _ prepared: ServerPreparedRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        try await generate(prepared.request, onEvent: onEvent)
    }
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    private var admittedCount = 0
    private var active = false
    private var waiters: [Waiter] = []
    private var shuttingDown = false
    private let onActivity: @Sendable (ServerCoordinatorActivity) -> Void

    public init(
        queueLimit: Int,
        onActivity: @escaping @Sendable (ServerCoordinatorActivity) -> Void = { _ in }
    ) {
        self.queueLimit = queueLimit
        self.onActivity = onActivity
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await runPreparing(
            onQueued: onQueued,
            prepare: { () },
            operation: { _ in try await operation() })
    }

    func runPreparing<Prepared: Sendable, T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        prepare: @escaping @Sendable () async throws -> Prepared,
        operation: @escaping @Sendable (Prepared) async throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        guard admittedCount <= queueLimit else { throw ServerRequestError.queueFull }
        admittedCount += 1
        defer { admittedCount -= 1 }

        let prepared = try await prepare()
        try Task.checkCancellation()
        try await acquire(onQueued: onQueued)
        defer { release() }
        return try await operation(prepared)
    }

    private func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if !active {
            active = true
            publishActivity()
            return
        }
        guard waiters.count < queueLimit else { throw ServerRequestError.queueFull }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
                publishActivity()
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
        publishActivity()
    }

    private func release() {
        if waiters.isEmpty {
            active = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
        publishActivity()
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
        publishActivity()
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { active }

    private func publishActivity() {
        onActivity(ServerCoordinatorActivity(
            activeRequests: active ? 1 : 0,
            queuedRequests: waiters.count))
    }
}
