import Foundation
import Synchronization
import TUFFDecodeProtocol

/// One generation running in the serving host, consumed by exactly one reader.
///
/// A pump task owns the host stream and is never cancelled with the reader:
/// when the HTTP client disconnects, the reader stops and the host is asked to
/// cancel, but the pump keeps reading until the host's terminal event. Only
/// then is the request's session and staged state released, so nothing is
/// freed while the helper may still be using it.
actor HostServingGeneration {
    nonisolated let sessionID: UUID
    nonisolated let requestID: UUID
    private nonisolated let link: HostServingLink
    private nonisolated let pumpDone = HostServingFlag()
    private var queue: [DecodeServiceEvent] = []
    private var overflowed = false
    private var streamEnded = false
    private var streamFailure: (any Error)?
    private var waiter: (id: UUID, continuation: CheckedContinuation<DecodeServiceEvent?, any Error>)?
    private(set) var sawTerminal = false

    init(link: HostServingLink, client: PrivateServingIPCClient, request: ServingGenerationRequest) {
        self.link = link
        sessionID = request.sessionID
        requestID = request.requestID
        let hostStream = client.events(for: .start(request))
        let done = pumpDone
        Task { [self] in
            do {
                for try await event in hostStream {
                    if case .generation(_, let decoded) = event { await push(decoded) }
                }
                await endStream(nil)
            } catch {
                await endStream(error)
            }
            done.set()
        }
    }

    private func push(_ event: DecodeServiceEvent) {
        if overflowed {
            cancel()
            return
        }
        if let waiter {
            self.waiter = nil
            waiter.continuation.resume(returning: event)
        } else {
            guard queue.count < PrivateServingIPCClient.streamCapacity else {
                queue.removeAll()
                overflowed = true
                streamFailure = PrivateServingIPCClientError.slowConsumer
                cancel()
                return
            }
            queue.append(event)
        }
    }

    private func endStream(_ error: (any Error)?) {
        streamEnded = true
        if !overflowed { streamFailure = error }
        guard let waiter, queue.isEmpty else { return }
        self.waiter = nil
        if let streamFailure { waiter.continuation.resume(throwing: Self.mapStreamFailure(streamFailure)) }
        else { waiter.continuation.resume(returning: nil) }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let waiter, waiter.id == id else { return }
        self.waiter = nil
        waiter.continuation.resume(returning: nil)
    }

    /// Next host event, including any `awaitStart` read ahead. Nil when the
    /// stream ended or the reader was cancelled.
    func next() async throws -> DecodeServiceEvent? {
        if overflowed { throw PrivateServingIPCClientError.slowConsumer }
        if !queue.isEmpty { return queue.removeFirst() }
        if streamEnded {
            if let streamFailure { throw Self.mapStreamFailure(streamFailure) }
            return nil
        }
        let id = UUID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiter = (id, continuation)
                if Task.isCancelled { cancelWaiter(id) }
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    /// Waits until the host has accepted the request: its first progress,
    /// text or terminal event. A rejection (context overflow, a bad image)
    /// arrives as the terminal event before any of that and is thrown here, so
    /// the HTTP layer can answer it with a real status instead of a stream
    /// that opened and then failed.
    func awaitStart() async throws {
        while true {
            guard let event = try await next() else {
                try Task.checkCancellation()
                throw ServerRequestError.hostUnavailable(
                    "The TUFF inference host connection ended before the generation started.")
            }
            switch event.kind {
            case .memory, .hello, .loading, .ready, .unloaded:
                continue
            case .failed:
                sawTerminal = true
                throw HostServingTranslation.requestError(
                    message: event.error ?? "generation failed", kind: event.errorKind)
            case .finished, .cancelled:
                sawTerminal = true
                queue.insert(event, at: 0)
                return
            case .prefill, .snapshot:
                queue.insert(event, at: 0)
                return
            }
        }
    }

    func markTerminal() { sawTerminal = true }

    /// Asks the host to stop. Safe to call repeatedly and from any context.
    nonisolated func cancel() {
        link.cancel(sessionID: sessionID, requestID: requestID)
    }

    /// Returns once the host has ended the stream. With `cancel`, asks the
    /// host to stop first. The start frame was written when this generation
    /// was created and the host handles one connection's commands in order,
    /// so this single cancel can never overtake the request's registration.
    nonisolated func drain(cancel shouldCancel: Bool) async {
        // Cleanup normally runs after the request's task was cancelled; the
        // wait must not be cut short by that.
        await Task.detached { [self] in await drainShielded(cancel: shouldCancel) }.value
    }

    private nonisolated func drainShielded(cancel shouldCancel: Bool) async {
        if shouldCancel, !pumpDone.isSet { cancel() }
        while !pumpDone.isSet {
            try? await Task.sleep(for: .milliseconds(10))
        }
    }

    private static func mapStreamFailure(_ error: any Error) -> any Error {
        if case PrivateServingIPCClientError.remote(let message) = error {
            return HostServingTranslation.requestError(message: message, kind: nil)
        }
        return HostServingLink.map(error)
    }
}

final class HostServingFlag: Sendable {
    private let value = Mutex(false)
    var isSet: Bool { value.withLock { $0 } }
    func set() { value.withLock { $0 = true } }
}
