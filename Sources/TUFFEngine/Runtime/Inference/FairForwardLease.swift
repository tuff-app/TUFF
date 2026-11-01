import Foundation

/// A FIFO lease for mutable resources shared by runners of the same model.
/// Hold it only for a forward operation, including its GPU completion; KV and
/// scratch remain private to each runner. It must never cover a generation.
public actor FairForwardLease {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, any Error>
    }

    private var owner: UUID?
    private var waiters: [Waiter] = []
    var waitingCount: Int { waiters.count }

    public init() {}

    public func withLease<T: Sendable>(
        _ operation: @Sendable () async throws -> T
    ) async throws -> T {
        let id = UUID()
        try await acquire(id)
        do {
            try Task.checkCancellation()
            let result = try await operation()
            release(id)
            return result
        } catch {
            release(id)
            throw error
        }
    }

    private func acquire(_ id: UUID) async throws {
        try Task.checkCancellation()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                if owner == nil {
                    owner = id
                    continuation.resume()
                } else {
                    waiters.append(Waiter(id: id, continuation: continuation))
                }
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release(_ id: UUID) {
        precondition(owner == id, "Only the current forward operation may release its lease")
        if waiters.isEmpty {
            owner = nil
        } else {
            let next = waiters.removeFirst()
            owner = next.id
            next.continuation.resume()
        }
    }
}
