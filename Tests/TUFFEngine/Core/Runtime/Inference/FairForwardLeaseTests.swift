import Foundation
import Testing
@testable import TUFFEngine

private actor LeaseTestGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var opened = false

    func wait() async {
        if opened { return }
        await withCheckedContinuation { continuation = $0 }
    }

    func open() {
        opened = true
        continuation?.resume()
        continuation = nil
    }
}

private actor LeaseTestEvents {
    private(set) var values: [Int] = []
    func append(_ value: Int) { values.append(value) }
}

@Suite struct FairForwardLeaseTests {
    @Test func cancelledWaiterDoesNotBlockNextForward() async throws {
        let lease = FairForwardLease()
        let gate = LeaseTestGate()
        let entered = LeaseTestGate()
        let events = LeaseTestEvents()
        let first = Task {
            try await lease.withLease {
                await events.append(1)
                await entered.open()
                await gate.wait()
                await events.append(2)
            }
        }
        await entered.wait()
        let cancelled = Task {
            try await lease.withLease { await events.append(99) }
        }
        while await lease.waitingCount < 1 { await Task.yield() }
        let next = Task {
            try await lease.withLease { await events.append(3) }
        }
        while await lease.waitingCount < 2 { await Task.yield() }
        cancelled.cancel()
        do {
            try await cancelled.value
            Issue.record("A cancelled queued forward acquired the lease")
        } catch is CancellationError {
        }
        await gate.open()
        try await first.value
        try await next.value
        #expect(await events.values == [1, 2, 3])
    }
}
