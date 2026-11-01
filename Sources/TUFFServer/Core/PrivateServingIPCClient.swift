import Foundation
import Synchronization
import TUFFDecodeProtocol

/// A single-reader, multiplexed client for the private serving socket.
/// Concurrent generations and lifecycle requests are routed by their typed
/// operation IDs instead of competing to read the same file descriptor.
public final class PrivateServingIPCClient: @unchecked Sendable {
    private enum Key: Hashable, Sendable {
        case request(UUID)
        case operation(UUID)
        case session(UUID)
    }

    private struct Registration: Sendable {
        let continuation: AsyncThrowingStream<ServingEvent, any Error>.Continuation
        /// What stops the host-side work if this stream is failed by the
        /// client rather than by a terminal event.
        let cancel: ServingCommand?
    }

    private struct State: Sendable {
        var continuations: [Key: Registration] = [:]
        var closed = false
    }

    /// Events buffered per stream before its consumer counts as stalled. A
    /// stalled stream is failed and cancelled instead of growing without
    /// bound or blocking the reader the other streams share.
    public static let streamCapacity = 4_096

    private let connection: PrivateServingIPCConnection
    private let state = Mutex(State())

    public var host: ServingHostHello { connection.host }
    public var isClosed: Bool { state.withLock { $0.closed } }

    public init(connection: PrivateServingIPCConnection) {
        self.connection = connection
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.readLoop()
        }
    }

    /// Receives progress and the terminal result for one command.
    public func events(for command: ServingCommand)
        -> AsyncThrowingStream<ServingEvent, any Error> {
        guard let key = Self.key(for: command) else {
            return failedStream(PrivateServingIPCClientError.noResponseKey)
        }
        return AsyncThrowingStream(bufferingPolicy: .bufferingOldest(Self.streamCapacity)) { [weak self] continuation in
            guard let self else {
                continuation.finish(throwing: PrivateServingIPCClientError.closed)
                return
            }
            let registered = state.withLock { current -> Bool in
                guard !current.closed, current.continuations[key] == nil else { return false }
                current.continuations[key] = Registration(
                    continuation: continuation, cancel: Self.cancellation(for: command))
                return true
            }
            guard registered else {
                continuation.finish(throwing: PrivateServingIPCClientError.requestAlreadyActive)
                return
            }
            continuation.onTermination = { [weak self] termination in
                guard let self else { return }
                // Only a still-registered key has no terminal event yet.
                guard self.remove(key) else { return }
                guard case .cancelled = termination else { return }
                switch command {
                case .start(let request):
                    try? self.connection.send(.cancel(
                        sessionID: request.sessionID, requestID: request.requestID))
                case .prepare(let preparation):
                    try? self.connection.send(.cancelPreparation(
                        operationID: preparation.operationID))
                default: break
                }
            }
            do { try connection.send(command) }
            catch {
                remove(key)
                continuation.finish(throwing: error)
            }
        }
    }

    /// Sends a one-way cancellation. The generation stream remains subscribed
    /// until its producer acknowledges cancellation with a terminal event.
    public func send(_ command: ServingCommand) throws {
        try command.validate()
        try connection.send(command)
    }

    public func close() {
        let continuations = state.withLock { current -> [AsyncThrowingStream<ServingEvent, any Error>.Continuation] in
            guard !current.closed else { return [] }
            current.closed = true
            let values = current.continuations.values.map(\.continuation)
            current.continuations.removeAll()
            return values
        }
        connection.close()
        for continuation in continuations {
            continuation.finish(throwing: PrivateServingIPCClientError.closed)
        }
    }

    private func readLoop() {
        do {
            while !state.withLock({ $0.closed }) {
                route(try connection.receive())
            }
        } catch {
            failAll(error)
        }
    }

    private func route(_ event: ServingEvent) {
        if case .failed(nil, let message) = event {
            failPending(afterHostFault: message)
            return
        }
        let routed: [(AsyncThrowingStream<ServingEvent, any Error>.Continuation, Bool)] = state.withLock { current in
            guard !current.closed else { return [] }
            switch event {
            case .lifecycle(let operationID, _, let phase):
                let key = Key.operation(operationID)
                guard let continuation = current.continuations[key]?.continuation else { return [] }
                let terminal: Bool
                switch phase {
                case .ready, .cancelled, .unloaded, .failed: terminal = true
                case .loading, .cancelling, .unloading: terminal = false
                }
                if terminal { current.continuations.removeValue(forKey: key) }
                return [(continuation, terminal)]
            case .queue(let identity, _, _), .generation(let identity, _):
                let key = Key.request(identity.requestID)
                guard let continuation = current.continuations[key]?.continuation else { return [] }
                let terminal: Bool
                if case .generation(_, let generation) = event {
                    terminal = [.finished, .failed, .cancelled].contains(generation.kind)
                } else { terminal = false }
                if terminal { current.continuations.removeValue(forKey: key) }
                return [(continuation, terminal)]
            case .failed(let identity?, _):
                let key = Key.request(identity.requestID)
                guard let continuation = current.continuations.removeValue(forKey: key)?.continuation else { return [] }
                return [(continuation, true)]
            case .sessionEnded(let sessionID), .sessionEndFailed(let sessionID, _):
                let key = Key.session(sessionID)
                guard let continuation = current.continuations.removeValue(forKey: key)?.continuation else { return [] }
                return [(continuation, true)]
            case .hostStatus(let operationID, _):
                let key = Key.operation(operationID)
                guard let continuation = current.continuations.removeValue(forKey: key)?.continuation else { return [] }
                return [(continuation, true)]
            case .failed(nil, _):
                return []
            }
        }
        for (continuation, terminal) in routed {
            if case .dropped = continuation.yield(event) {
                abandonStalledStream(continuation, event: event)
                continue
            }
            if terminal {
                if case .failed(_, let message) = event {
                    continuation.finish(throwing: PrivateServingIPCClientError.remote(message))
                } else if case .sessionEndFailed(_, let message) = event {
                    continuation.finish(throwing: PrivateServingIPCClientError.remote(message))
                } else if case .lifecycle(_, _, .failed(let message)) = event {
                    continuation.finish(throwing: PrivateServingIPCClientError.remote(message))
                } else {
                    continuation.finish()
                }
            }
        }
    }

    /// An unkeyed failure is a connection-level fault: no pending operation
    /// will finish. Each ends with its own typed error, and the host is told
    /// to stop the work it can no longer deliver.
    private func failPending(afterHostFault message: String) {
        let registrations = state.withLock { current -> [Registration] in
            guard !current.closed else { return [] }
            let values = Array(current.continuations.values)
            current.continuations.removeAll()
            return values
        }
        for registration in registrations {
            if let cancel = registration.cancel { try? connection.send(cancel) }
            registration.continuation.finish(
                throwing: PrivateServingIPCClientError.hostFault(message))
        }
    }

    private func failAll(_ error: any Error) {
        let continuations = state.withLock { current -> [AsyncThrowingStream<ServingEvent, any Error>.Continuation] in
            guard !current.closed else { return [] }
            current.closed = true
            let values = current.continuations.values.map(\.continuation)
            current.continuations.removeAll()
            return values
        }
        connection.close()
        for continuation in continuations { continuation.finish(throwing: error) }
    }

    /// Removes the registration and reports whether it was still present.
    @discardableResult
    private func remove(_ key: Key) -> Bool {
        state.withLock { $0.continuations.removeValue(forKey: key) != nil }
    }

    /// The consumer stopped draining. Fail the stream and ask the host to stop
    /// the generation it can no longer deliver.
    private func abandonStalledStream(
        _ continuation: AsyncThrowingStream<ServingEvent, any Error>.Continuation,
        event: ServingEvent
    ) {
        let identity: ServingRequestIdentity?
        switch event {
        case .queue(let value, _, _), .generation(let value, _): identity = value
        case .failed(let value, _): identity = value
        default: identity = nil
        }
        if let identity {
            _ = remove(.request(identity.requestID))
            try? connection.send(.cancel(sessionID: identity.sessionID, requestID: identity.requestID))
        }
        continuation.finish(throwing: PrivateServingIPCClientError.slowConsumer)
    }

    private static func key(for command: ServingCommand) -> Key? {
        switch command {
        case .prepare(let request): .operation(request.operationID)
        case .unloadModel(let operationID, _): .operation(operationID)
        case .start(let request): .request(request.requestID)
        case .endSession(let sessionID): .session(sessionID)
        case .hostStatus(let operationID): .operation(operationID)
        case .cancelPreparation, .cancel: nil
        }
    }

    /// The command that stops the host-side work `command` started.
    private static func cancellation(for command: ServingCommand) -> ServingCommand? {
        switch command {
        case .start(let request): .cancel(sessionID: request.sessionID, requestID: request.requestID)
        case .prepare(let preparation): .cancelPreparation(operationID: preparation.operationID)
        default: nil
        }
    }

    private func failedStream(_ error: any Error)
        -> AsyncThrowingStream<ServingEvent, any Error> {
        AsyncThrowingStream { $0.finish(throwing: error) }
    }
}

public enum PrivateServingIPCClientError: Error, Sendable, Equatable, CustomStringConvertible {
    case closed
    case noResponseKey
    case requestAlreadyActive
    case slowConsumer
    case remote(String)
    /// The host reported a connection-level fault that no single operation
    /// caused. Every pending operation ends with it and is cancelled host-side.
    case hostFault(String)

    public var description: String {
        switch self {
        case .closed: "The TUFF serving connection closed"
        case .noResponseKey: "This TUFF serving command has no response stream"
        case .requestAlreadyActive: "This TUFF serving operation is already active"
        case .slowConsumer: "The app stopped reading this generation, so the serving host cancelled it"
        case .remote(let message): message
        case .hostFault(let message): message
        }
    }
}
