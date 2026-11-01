import Foundation
import Synchronization

/// Keeps lifecycle transitions exclusive with new work and prevents the
/// legacy single-runner path from overlapping shared-model peer sessions.
final class InferenceAdmission: Sendable {
    private struct State: Sendable {
        var lifecycle = false
        var legacy = false
        var sharedCount = 0
        var lifecycleWaiters: [CheckedContinuation<Void, Never>] = []
    }
    private let state = Mutex(State())

    func beginLifecycle() async {
        await withCheckedContinuation { continuation in
            let acquired = state.withLock { state -> Bool in
                if state.lifecycle {
                    state.lifecycleWaiters.append(continuation)
                    return false
                }
                state.lifecycle = true
                return true
            }
            if acquired { continuation.resume() }
        }
    }

    func endLifecycle() {
        let next = state.withLock { state -> CheckedContinuation<Void, Never>? in
            guard !state.lifecycleWaiters.isEmpty else {
                state.lifecycle = false
                return nil
            }
            return state.lifecycleWaiters.removeFirst()
        }
        next?.resume()
    }

    func beginLegacy() -> Bool {
        state.withLock { state in
            guard !state.lifecycle, !state.legacy, state.sharedCount == 0 else { return false }
            state.legacy = true
            return true
        }
    }

    func endLegacy() { state.withLock { $0.legacy = false } }

    func beginShared() -> Bool {
        state.withLock { state in
            guard !state.lifecycle, !state.legacy else { return false }
            state.sharedCount += 1
            return true
        }
    }

    func endShared() { state.withLock { $0.sharedCount -= 1 } }
}
