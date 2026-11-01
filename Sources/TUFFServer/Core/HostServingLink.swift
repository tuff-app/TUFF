import Foundation
import Synchronization
import TUFFDecodeProtocol

/// A failure that is neither the caller's mistake nor a typed server status:
/// the host reported a generation or session failure. Surfaces as HTTP 500
/// with the message in the server log.
struct HostServingError: Error, CustomStringConvertible, Sendable {
    let description: String
    init(_ description: String) { self.description = description }
}

/// The API adapter's attachment to the per-profile serving host.
///
/// The host owns every model helper; this link only holds a private socket.
/// The connection is established lazily and re-established on the next
/// operation after it dies (the host may idle-exit or be restarted), so a host
/// restart costs the in-flight request only.
public final class HostServingLink: Sendable {
    public typealias Connector = @Sendable () throws -> PrivateServingIPCConnection

    private let launching: Connector
    private let existing: Connector
    private struct AttachmentState {
        var client: PrivateServingIPCClient?
        var pending: (id: UUID, task: Task<PrivateServingIPCClient, any Error>)?
    }
    private let state = Mutex(AttachmentState())

    /// - Parameters:
    ///   - launching: Connects, starting the host process when none answers.
    ///   - existing: Connects only to a running host. Status queries use it so
    ///     that asking `/health` never starts a host.
    public init(launching: @escaping Connector, existing: @escaping Connector) {
        self.launching = launching
        self.existing = existing
        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
    }

    /// Each adapter instance has its own identity, including when foreground
    /// and background API adapters run at the same time.
    public convenience init(profile: ServingHostProfile) {
        let clientID = UUID()
        self.init(
            launching: {
                try ServingHostRuntime.connectOrLaunch(
                    profile: profile, clientID: clientID, role: .api).connection
            },
            existing: {
                try PrivateServingIPCConnection.connect(
                    profile: profile, clientID: clientID,
                    role: .api, expectedBuildIdentity: ServingHostRuntime.buildIdentity)
            })
    }

    public var isAttached: Bool { state.withLock { $0.client.map { !$0.isClosed } ?? false } }

    /// Drops the current connection. Open sessions end with it; the host
    /// releases connection-owned state.
    public func close() {
        let old = state.withLock { current -> PrivateServingIPCClient? in
            defer { current.client = nil; current.pending = nil }
            return current.client
        }
        old?.close()
    }

    // MARK: - Connection

    func attach(launch: Bool = true) async throws -> PrivateServingIPCClient {
        let pending = state.withLock { current -> (UUID, Task<PrivateServingIPCClient, any Error>) in
            if let client = current.client, !client.isClosed {
                return (UUID(), Task { client })
            }
            if let pending = current.pending { return (pending.id, pending.task) }
            let id = UUID()
            let connector = launch ? launching : existing
            let task = Task.detached(priority: .userInitiated) {
                PrivateServingIPCClient(connection: try connector())
            }
            current.pending = (id, task)
            return (id, task)
        }
        do {
            let fresh = try await pending.1.value
            let accepted = state.withLock { current -> Bool in
                if current.client === fresh { return true }
                guard current.pending?.id == pending.0 else { return false }
                current.pending = nil
                current.client = fresh
                return true
            }
            guard accepted else {
                fresh.close()
                throw PrivateServingIPCClientError.closed
            }
            return fresh
        } catch {
            state.withLock { current in
                if current.pending?.id == pending.0 { current.pending = nil }
            }
            throw ServerRequestError.hostUnavailable(
                "The TUFF inference host could not be reached: \(error)")
        }
    }

    /// Runs `body` on the live connection, once more on a fresh one when the
    /// connection was lost before the operation could complete.
    func withConnection<T: Sendable>(launch: Bool = true,
        _ body: (PrivateServingIPCClient) async throws -> T) async throws -> T {
        var attempt = 0
        while true {
            let current = try await attach(launch: launch)
            do { return try await body(current) }
            catch where Self.isConnectionLoss(error) && attempt == 0 {
                attempt += 1
                current.close()
                try Task.checkCancellation()
            } catch {
                throw Self.map(error)
            }
        }
    }

    static func isConnectionLoss(_ error: any Error) -> Bool {
        if case PrivateServingIPCClientError.closed = error { return true }
        return error is PrivateServingIPCError
    }

    /// The typed request error for a failure on the private socket.
    static func map(_ error: any Error) -> any Error {
        switch error {
        case is CancellationError, is ServerRequestError: return error
        case PrivateServingIPCClientError.remote(let message):
            if message.contains(ServingMemoryPressure.phrase) {
                return ServerRequestError.modelMemoryBusy(message)
            }
            if message.contains("Unload the model before changing") {
                return ServerRequestError.modelBusy(
                    "The model is loaded by another TUFF client with a configuration this request cannot use.")
            }
            return HostServingError(message)
        case let failure as PrivateServingIPCClientError:
            return ServerRequestError.hostUnavailable("\(failure)")
        case let failure as PrivateServingIPCError:
            return ServerRequestError.hostUnavailable("\(failure)")
        default:
            return error
        }
    }

    // MARK: - Operations

    /// Live host state. With `launch: false`, nil when no host is running.
    func status(launch: Bool) async throws -> ServingHostStatus? {
        do {
            return try await withConnection(launch: launch) { client in
                for try await event in client.events(for: .hostStatus(operationID: UUID())) {
                    if case .hostStatus(_, let status) = event { return status }
                }
                try Task.checkCancellation()
                throw ServerRequestError.hostUnavailable(
                    "The TUFF inference host did not answer its status request.")
            }
        } catch ServerRequestError.hostUnavailable where !launch {
            return nil
        }
    }

    /// Makes `preparation`'s model resident in the host (a no-op when the host
    /// already holds it in a compatible configuration).
    func prepare(_ preparation: ServingModelPreparation) async throws {
        try await withConnection { client in
            for try await event in client.events(for: .prepare(preparation)) {
                if case .lifecycle(_, _, .cancelled) = event { throw CancellationError() }
            }
            try Task.checkCancellation()
        }
    }

    /// Asks the host to release `modelID`. False when the host refuses because
    /// the model is not loaded or has active work.
    func unload(modelID: String) async throws -> Bool {
        do {
            try await withConnection { client in
                for try await _ in client.events(for: .unloadModel(operationID: UUID(), modelID: modelID)) {}
                try Task.checkCancellation()
            }
            return true
        } catch let error as HostServingError where error.description.contains("not loaded or has active") {
            return false
        }
    }

    /// Returns only after the host has released the session's state.
    func endSession(_ sessionID: UUID) async throws {
        try await withConnection { client in
            for try await _ in client.events(for: .endSession(sessionID: sessionID)) {}
            try Task.checkCancellation()
        }
    }

    /// One-way cancellation; the stream finishes when the host acknowledges.
    func cancel(sessionID: UUID, requestID: UUID) {
        guard let current = state.withLock({ $0.client }), !current.isClosed else { return }
        try? current.send(.cancel(sessionID: sessionID, requestID: requestID))
    }
}
