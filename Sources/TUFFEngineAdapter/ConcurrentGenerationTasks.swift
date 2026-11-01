import Foundation
import Synchronization

/// Tracks independent producers until their cleanup has finished. A cancel
/// targets one request; a session may have only one active request at a time.
final class ConcurrentGenerationTasks: Sendable {
    private struct Entry: Sendable {
        let sessionID: UUID
        var task: Task<Void, Never>?
        var cancelled = false
    }
    private struct State: Sendable {
        var entries: [UUID: Entry] = [:]
        var idleWaiters: [CheckedContinuation<Void, Never>] = []
        var sessionWaiters: [UUID: [CheckedContinuation<Void, Never>]] = [:]
        var requestWaiters: [UUID: [CheckedContinuation<Void, Never>]] = [:]
        var closingSessions: Set<UUID> = []
    }
    private let state = Mutex(State())

    func reserve(requestID: UUID, sessionID: UUID) -> Bool {
        state.withLock { state in
            guard !state.closingSessions.contains(sessionID), state.entries[requestID] == nil,
                  !state.entries.values.contains(where: { $0.sessionID == sessionID }) else { return false }
            state.entries[requestID] = Entry(sessionID: sessionID)
            return true
        }
    }

    func attach(_ task: Task<Void, Never>, requestID: UUID) {
        let cancel = state.withLock { state -> Bool in
            guard state.entries[requestID] != nil else { return true }
            state.entries[requestID]?.task = task
            return state.entries[requestID]?.cancelled == true
        }
        if cancel { task.cancel() }
    }

    func cancel(requestID: UUID) {
        let task = state.withLock { state -> Task<Void, Never>? in
            guard state.entries[requestID] != nil else { return nil }
            state.entries[requestID]?.cancelled = true
            return state.entries[requestID]?.task
        }
        task?.cancel()
    }

    func cancelAll() {
        let tasks = state.withLock { state -> [Task<Void, Never>] in
            for id in state.entries.keys { state.entries[id]?.cancelled = true }
            return state.entries.values.compactMap(\.task)
        }
        tasks.forEach { $0.cancel() }
    }

    func clear(requestID: UUID) {
        let waiters = state.withLock { state -> [CheckedContinuation<Void, Never>] in
            guard let entry = state.entries.removeValue(forKey: requestID) else { return [] }
            var waiters = state.sessionWaiters.removeValue(forKey: entry.sessionID) ?? []
            waiters.append(contentsOf: state.requestWaiters.removeValue(forKey: requestID) ?? [])
            if state.entries.isEmpty {
                waiters.append(contentsOf: state.idleWaiters)
                state.idleWaiters.removeAll()
            }
            return waiters
        }
        waiters.forEach { $0.resume() }
    }

    func beginEnding(_ sessionID: UUID) -> Bool {
        let result = state.withLock { state -> (Bool, Task<Void, Never>?) in
            guard state.closingSessions.insert(sessionID).inserted else { return (false, nil) }
            guard let id = state.entries.first(where: { $0.value.sessionID == sessionID })?.key else {
                return (true, nil)
            }
            state.entries[id]?.cancelled = true
            return (true, state.entries[id]?.task)
        }
        result.1?.cancel()
        return result.0
    }

    func waitUntilSessionIdle(_ sessionID: UUID) async {
        await withCheckedContinuation { continuation in
            let idle = state.withLock { state -> Bool in
                guard state.entries.values.contains(where: { $0.sessionID == sessionID }) else { return true }
                state.sessionWaiters[sessionID, default: []].append(continuation)
                return false
            }
            if idle { continuation.resume() }
        }
    }

    func waitUntilRequestIdle(_ requestID: UUID) async {
        await withCheckedContinuation { continuation in
            let idle = state.withLock { state -> Bool in
                guard state.entries[requestID] != nil else { return true }
                state.requestWaiters[requestID, default: []].append(continuation)
                return false
            }
            if idle { continuation.resume() }
        }
    }

    func finishEnding(_ sessionID: UUID) {
        _ = state.withLock { $0.closingSessions.remove(sessionID) }
    }

    func waitUntilIdle() async {
        await withCheckedContinuation { continuation in
            let idle = state.withLock { state -> Bool in
                if state.entries.isEmpty { return true }
                state.idleWaiters.append(continuation)
                return false
            }
            if idle { continuation.resume() }
        }
    }
}
