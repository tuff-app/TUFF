import Foundation
import Synchronization
import Testing
import TUFFDecodeProtocol
@testable import TUFFServerCore

@Suite(.timeLimit(.minutes(1))) struct ServingModelCoordinatorTests {
    private final class FakeHandle: ServingModelHandle, @unchecked Sendable {
        typealias Continuation = AsyncThrowingStream<DecodeServiceEvent, Error>.Continuation
        private struct State: Sendable {
            var streams: [UUID: Continuation] = [:]
            var cancelled: [UUID] = []
            var warmupCancels = 0
            var unloads = 0
            var endSessionFailures = 0
            var endSessionCalls = 0
            var holdEndSession = false
            var prepareErrors: [any Error] = []
            var prepares = 0
            var holdPrepare = false
            var startFailures: [String] = []
            var starts = 0
            var loseTransport = false
        }
        private let state = Mutex(State())
        var streamIDs: [UUID] { state.withLock { Array($0.streams.keys) } }
        var cancelled: [UUID] { state.withLock { $0.cancelled } }
        var warmupCancels: Int { state.withLock { $0.warmupCancels } }
        var unloads: Int { state.withLock { $0.unloads } }
        var endSessionCalls: Int { state.withLock { $0.endSessionCalls } }
        var prepares: Int { state.withLock { $0.prepares } }
        func failNextEndSessions(_ count: Int) { state.withLock { $0.endSessionFailures = count } }
        var starts: Int { state.withLock { $0.starts } }
        func failNextStarts(_ messages: [String]) { state.withLock { $0.startFailures = messages } }
        func holdPrepares(_ hold: Bool) { state.withLock { $0.holdPrepare = hold } }
        func holdEndSessions(_ hold: Bool) { state.withLock { $0.holdEndSession = hold } }
        func failTransport() { state.withLock { $0.loseTransport = true } }
        func failNextPrepares(_ errors: [any Error]) { state.withLock { $0.prepareErrors = errors } }
        func prepare(_ request: ServingModelPreparation,
                     progress: @escaping @Sendable (ServingLoadState) -> Void) async throws {
            progress(.loading(.preparingRunner))
            state.withLock { $0.prepares += 1 }
            while state.withLock({ $0.holdPrepare }) {
                try await Task.sleep(for: .milliseconds(2))
            }
            let failure = state.withLock { state -> (any Error)? in
                state.prepareErrors.isEmpty ? nil : state.prepareErrors.removeFirst()
            }
            if let failure { throw failure }
        }
        func start(_ request: DecodeGenerationRequest, sessionID: UUID)
            -> AsyncThrowingStream<DecodeServiceEvent, Error> {
            let failure = state.withLock { state -> String? in
                state.starts += 1
                return state.startFailures.isEmpty ? nil : state.startFailures.removeFirst()
            }
            return AsyncThrowingStream { continuation in
                if state.withLock({ $0.loseTransport }) {
                    continuation.finish(throwing: ServingHelperTransportLost("helper exited"))
                    return
                }
                if let failure {
                    var event = DecodeServiceEvent(kind: .failed, generationID: request.generationID)
                    event.error = failure
                    continuation.yield(event)
                    continuation.finish()
                    return
                }
                state.withLock { $0.streams[request.generationID] = continuation }
            }
        }
        func cancel(requestID: UUID) {
            let continuation = state.withLock { state -> Continuation? in
                state.cancelled.append(requestID)
                return state.streams.removeValue(forKey: requestID)
            }
            continuation?.yield(DecodeServiceEvent(kind: .cancelled, generationID: requestID))
            continuation?.finish()
        }
        func cancelWarmup() { state.withLock { $0.warmupCancels += 1 } }
        struct CleanupFailure: Error {}
        func endSession(_ sessionID: UUID, requestID: UUID) async throws {
            let fail = state.withLock { state -> Bool in
                state.endSessionCalls += 1
                guard state.endSessionFailures > 0 else { return false }
                state.endSessionFailures -= 1
                return true
            }
            while state.withLock({ $0.holdEndSession }) {
                try await Task.sleep(for: .milliseconds(2))
            }
            if fail { throw CleanupFailure() }
        }
        func unload() async { state.withLock { $0.unloads += 1 } }
        func finish(_ requestID: UUID) {
            let continuation = state.withLock { $0.streams.removeValue(forKey: requestID) }
            continuation?.yield(DecodeServiceEvent(kind: .finished, generationID: requestID))
            continuation?.finish()
        }
    }
    private final class Events: Sendable {
        private let values = Mutex<[ServingEvent]>([])
        func append(_ event: ServingEvent) { values.withLock { $0.append(event) } }
        func count(_ predicate: (ServingEvent) -> Bool) -> Int { values.withLock { $0.filter(predicate).count } }
    }
    private func eventually(_ predicate: @escaping @Sendable () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(5)
        while !predicate(), Date() < deadline { try? await Task.sleep(for: .milliseconds(5)) }
        return predicate()
    }

    @Test func isolatesClientsAndCancelsEveryModelWarmup() async throws {
        let first = FakeHandle(), second = FakeHandle(), events = Events()
        let coordinator = ServingModelCoordinator { id in id == "one" ? first : second }
        let clientA = UUID(), clientB = UUID(), session = UUID(), requestID = UUID()
        let prepA = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        let prepB = try ServingModelPreparation(modelID: "two", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        await coordinator.handle(clientID: clientA, command: .prepare(prepA), emit: events.append)
        await coordinator.handle(clientID: clientA, command: .prepare(prepB), emit: events.append)
        #expect(await eventually {
            events.count { if case .lifecycle(_, _, .ready) = $0 { return true }; return false } == 2
        })
        let payload = DecodeGenerationRequest(prompt: "a", maxNewTokens: 4, maxContextTokens: 1024,
                                              temperature: 0, generationID: requestID)
        let request = try ServingGenerationRequest(modelID: "one", sessionID: session,
                                                    requestID: requestID, payload: payload)
        let firstBefore = first.warmupCancels
        let secondBefore = second.warmupCancels
        let runA = Task { await coordinator.handle(clientID: clientA, command: .start(request), emit: events.append) }
        let runB = Task { await coordinator.handle(clientID: clientB, command: .start(request), emit: events.append) }
        #expect(await eventually { first.streamIDs.count == 2 })
        #expect(first.warmupCancels == firstBefore + 2)
        #expect(second.warmupCancels == secondBefore + 2)
        await coordinator.handle(clientID: clientA, command: .cancel(sessionID: session, requestID: requestID),
                                 emit: events.append)
        #expect(await eventually { first.cancelled.count == 1 })
        let remaining = first.streamIDs
        #expect(remaining.count == 1)
        if let id = remaining.first { first.finish(id) }
        await runA.value
        await runB.value
        #expect(await eventually {
            events.count {
                if case .generation(_, let event) = $0 {
                    return [.finished, .cancelled].contains(event.kind) && event.generationID == requestID
                }
                return false
            } == 2
        })
    }

    @Test func configurationCapsAndIdleEviction() async throws {
        let fake = FakeHandle(), events = Events(), client = UUID()
        let coordinator = ServingModelCoordinator(maxModels: 1, maxSessions: 1,
                                                   maxActiveRequests: 2) { _ in fake }
        let first = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        let different = try ServingModelPreparation(modelID: "one", maxContextTokens: 2048,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        let other = try ServingModelPreparation(modelID: "two", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        await coordinator.handle(clientID: client, command: .prepare(first), emit: events.append)
        await coordinator.handle(clientID: client, command: .prepare(different), emit: events.append)
        await coordinator.handle(clientID: client, command: .prepare(other), emit: events.append)
        #expect(events.count { if case .lifecycle(_, _, .failed) = $0 { return true }; return false } == 2)
        let requestID = UUID(), session = UUID()
        let payload = DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                              temperature: 0, generationID: requestID)
        let request = try ServingGenerationRequest(modelID: "one", sessionID: session,
                                                    requestID: requestID, payload: payload)
        let running = Task { await coordinator.handle(clientID: client, command: .start(request), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        #expect(await coordinator.evictIdleModels().isEmpty)
        await coordinator.handle(clientID: client, command: .endSession(sessionID: session), emit: events.append)
        await running.value
        #expect(await coordinator.evictIdleModels() == ["one"])
        #expect(fake.unloads == 1)
        await coordinator.handle(clientID: client, command: .prepare(other), emit: events.append)
        await coordinator.shutdown()
        #expect(fake.unloads == 2)
    }

    @Test func differentModelsProgressTogetherAndShutdownDrains() async throws {
        let first = FakeHandle(), second = FakeHandle(), events = Events(), client = UUID()
        let coordinator = ServingModelCoordinator { id in id == "one" ? first : second }
        for name in ["one", "two"] {
            let preparation = try ServingModelPreparation(modelID: name, maxContextTokens: 1024,
                runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
            await coordinator.handle(clientID: client, command: .prepare(preparation), emit: events.append)
        }
        func request(_ model: String) throws -> ServingGenerationRequest {
            let id = UUID()
            return try ServingGenerationRequest(modelID: model, sessionID: UUID(), requestID: id,
                payload: DecodeGenerationRequest(prompt: model, maxNewTokens: 4, maxContextTokens: 1024,
                                                 temperature: 0, generationID: id))
        }
        let a = try request("one"), b = try request("two")
        let runA = Task { await coordinator.handle(clientID: client, command: .start(a), emit: events.append) }
        let runB = Task { await coordinator.handle(clientID: client, command: .start(b), emit: events.append) }
        #expect(await eventually { first.streamIDs.count == 1 && second.streamIDs.count == 1 })
        #expect(await coordinator.evictIdleModels().isEmpty)
        await coordinator.shutdown()
        await runA.value
        await runB.value
        #expect(first.cancelled.count == 1 && second.cancelled.count == 1)
        #expect(first.unloads == 1 && second.unloads == 1)
        #expect(events.count {
            if case .generation(_, let event) = $0 { return event.kind == .cancelled }
            return false
        } == 2)
    }

    @Test func terminalAllowsImmediateNextTurnOnSameSession() async throws {
        let fake = FakeHandle(), events = Events(), client = UUID(), session = UUID()
        let coordinator = ServingModelCoordinator { _ in fake }
        let preparation = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        await coordinator.handle(clientID: client, command: .prepare(preparation), emit: events.append)
        func makeRequest() throws -> ServingGenerationRequest {
            let id = UUID()
            return try ServingGenerationRequest(modelID: "one", sessionID: session, requestID: id,
                payload: DecodeGenerationRequest(prompt: "hello", maxNewTokens: 4,
                                                 maxContextTokens: 1024, temperature: 0, generationID: id))
        }
        let first = try makeRequest()
        let runFirst = Task { await coordinator.handle(clientID: client, command: .start(first), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        if let id = fake.streamIDs.first { fake.finish(id) }
        #expect(await eventually {
            events.count { if case .generation(let identity, let event) = $0 {
                return identity.requestID == first.requestID && event.kind == .finished
            }; return false } == 1
        })
        let next = try makeRequest()
        let runNext = Task { await coordinator.handle(clientID: client, command: .start(next), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        if let id = fake.streamIDs.first { fake.finish(id) }
        await runFirst.value
        await runNext.value
        #expect(events.count { if case .generation(let identity, let event) = $0 {
            return identity.requestID == next.requestID && event.kind == .finished
        }; return false } == 1)
    }

    @Test func failedSessionCleanupKeepsOwnershipAndAcknowledgesOnlyRealCleanup() async throws {
        let fake = FakeHandle(), events = Events(), client = UUID(), session = UUID()
        let coordinator = ServingModelCoordinator { _ in fake }
        let preparation = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        await coordinator.handle(clientID: client, command: .prepare(preparation), emit: events.append)
        let id = UUID()
        let request = try ServingGenerationRequest(modelID: "one", sessionID: session, requestID: id,
            payload: DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                             temperature: 0, generationID: id))
        let run = Task { await coordinator.handle(clientID: client, command: .start(request), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        fake.finish(fake.streamIDs[0])
        await run.value
        #expect(await coordinator.snapshot().openSessions == 1)

        fake.failNextEndSessions(1)
        await coordinator.handle(clientID: client, command: .endSession(sessionID: session), emit: events.append)
        #expect(events.count { if case .sessionEndFailed = $0 { return true }; return false } == 1)
        #expect(events.count { if case .sessionEnded = $0 { return true }; return false } == 0)
        // Ownership and reservation survive the failed cleanup, and the
        // model cannot be evicted out from under the unreleased session.
        #expect(await coordinator.snapshot().openSessions == 1)

        await coordinator.handle(clientID: client, command: .endSession(sessionID: session), emit: events.append)
        #expect(events.count { if case .sessionEnded = $0 { return true }; return false } == 1)
        #expect(await coordinator.snapshot().openSessions == 0)
        #expect(fake.endSessionCalls == 2)
    }

    @Test func concurrentEndSessionsShareOneCleanup() async throws {
        let fake = FakeHandle(), events = Events(), client = UUID(), session = UUID()
        let coordinator = ServingModelCoordinator { _ in fake }
        let preparation = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        await coordinator.handle(clientID: client, command: .prepare(preparation), emit: events.append)
        let id = UUID()
        let request = try ServingGenerationRequest(modelID: "one", sessionID: session, requestID: id,
            payload: DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                             temperature: 0, generationID: id))
        let run = Task { await coordinator.handle(clientID: client, command: .start(request), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        // Ending the session cancels its running request first; both closes
        // wait for that drain and the helper is asked to clean up once.
        async let first: Void = coordinator.handle(clientID: client, command: .endSession(sessionID: session), emit: events.append)
        async let second: Void = coordinator.handle(clientID: client, command: .endSession(sessionID: session), emit: events.append)
        _ = await (first, second)
        await run.value
        #expect(fake.endSessionCalls == 1)
        #expect(events.count { if case .sessionEnded = $0 { return true }; return false } == 2)
    }

    @Test func memoryPressureEvictsIdleModelsAndRetriesButNeverActiveOnes() async throws {
        struct Busy: Error, CustomStringConvertible {
            var description: String { "Another TUFF process \(ServingMemoryPressure.phrase). Unload its model." }
        }
        let one = FakeHandle(), two = FakeHandle(), events = Events(), client = UUID()
        let coordinator = ServingModelCoordinator { id in id == "one" ? one : two }
        func prepare(_ model: String) throws -> ServingModelPreparation {
            try ServingModelPreparation(modelID: model, maxContextTokens: 1024,
                runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        }
        await coordinator.handle(clientID: client, command: .prepare(try prepare("one")), emit: events.append)
        two.failNextPrepares([Busy()])
        await coordinator.handle(clientID: client, command: .prepare(try prepare("two")), emit: events.append)
        #expect(one.unloads == 1)      // idle model yielded
        #expect(two.prepares == 2)     // and the load was retried once
        #expect(await coordinator.snapshot().loadedModelIDs == ["two"])

        // Now "two" is busy: a third model's memory failure cannot evict it.
        let three = FakeHandle()
        let coordinator2 = ServingModelCoordinator { id in id == "one" ? one : (id == "two" ? two : three) }
        await coordinator2.handle(clientID: client, command: .prepare(try prepare("one")), emit: events.append)
        let id = UUID()
        let request = try ServingGenerationRequest(modelID: "one", sessionID: UUID(), requestID: id,
            payload: DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                             temperature: 0, generationID: id))
        let run = Task { await coordinator2.handle(clientID: client, command: .start(request), emit: events.append) }
        #expect(await eventually { one.streamIDs.count == 1 })
        three.failNextPrepares([Busy()])
        let before = events.count { if case .lifecycle(_, _, .failed) = $0 { return true }; return false }
        await coordinator2.handle(clientID: client, command: .prepare(try prepare("three")), emit: events.append)
        #expect(events.count { if case .lifecycle(_, _, .failed) = $0 { return true }; return false } == before + 1)
        #expect(three.prepares == 1)
        #expect(await coordinator2.snapshot().loadedModelIDs == ["one"])
        await coordinator2.shutdown()
        await run.value
        await coordinator.shutdown()
    }

    @Test func overlappingLoadsNeverLeaveAWarmupRunningBesideForegroundWork() async throws {
        let one = FakeHandle(), two = FakeHandle(), events = Events(), client = UUID()
        let coordinator = ServingModelCoordinator { id in id == "one" ? one : two }
        func prepare(_ model: String) throws -> ServingModelPreparation {
            try ServingModelPreparation(modelID: model, maxContextTokens: 1024,
                runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        }
        one.holdPrepares(true); two.holdPrepares(true)
        let loadOne = Task { await coordinator.handle(clientID: client, command: .prepare(try prepare("one")), emit: events.append) }
        let loadTwo = Task { await coordinator.handle(clientID: client, command: .prepare(try prepare("two")), emit: events.append) }
        #expect(await eventually { one.prepares == 1 && two.prepares == 1 })
        // "one" finishes while "two" is still loading: its warmup must yield.
        one.holdPrepares(false)
        try await loadOne.value
        #expect(one.warmupCancels >= 1)
        let beforeTwo = two.warmupCancels
        two.holdPrepares(false)
        try await loadTwo.value
        // Nothing else is loading or running, so the last model may warm.
        #expect(two.warmupCancels == beforeTwo)

        // Foreground work on either model stops every model's warmup.
        let id = UUID()
        let request = try ServingGenerationRequest(modelID: "one", sessionID: UUID(), requestID: id,
            payload: DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                             temperature: 0, generationID: id))
        let beforeStart = (one.warmupCancels, two.warmupCancels)
        let run = Task { await coordinator.handle(clientID: client, command: .start(request), emit: events.append) }
        #expect(await eventually { one.streamIDs.count == 1 })
        #expect(one.warmupCancels == beforeStart.0 + 1 && two.warmupCancels == beforeStart.1 + 1)
        one.finish(one.streamIDs[0])
        await run.value

        await coordinator.shutdown()
    }

    @Test func imageWorkReclaimsOnlyIdleSessionsBeforeFailing() async throws {
        let one = FakeHandle(), two = FakeHandle(), events = Events(), client = UUID()
        let coordinator = ServingModelCoordinator { id in id == "one" ? one : two }
        func prepare(_ model: String) throws -> ServingModelPreparation {
            try ServingModelPreparation(modelID: model, maxContextTokens: 1024,
                runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        }
        await coordinator.handle(clientID: client, command: .prepare(try prepare("one")), emit: events.append)
        await coordinator.handle(clientID: client, command: .prepare(try prepare("two")), emit: events.append)
        func request(model: String, session: UUID, image: Bool) throws -> ServingGenerationRequest {
            let id = UUID()
            var payload = DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                                  temperature: 0, generationID: id)
            if image {
                payload.imageAttachments = [DecodeImageAttachment(
                    id: UUID(), path: "/tmp/x.png", displayName: "x.png", encodedBytes: 1, sha256: "0")]
            }
            return try ServingGenerationRequest(modelID: model, sessionID: session, requestID: id, payload: payload)
        }
        // An idle chat on each model keeps a session (and its lease) warm.
        let idleSession = UUID()
        for (model, session, handle) in [("two", idleSession, two), ("one", UUID(), one)] {
            let warm = try request(model: model, session: session, image: false)
            let task = Task { await coordinator.handle(clientID: client, command: .start(warm), emit: events.append) }
            #expect(await eventually { handle.streamIDs.count == 1 })
            handle.finish(handle.streamIDs[0])
            await task.value
        }
        #expect(await coordinator.snapshot().openSessions == 2)

        // Image admission releases idle sessions and the idle other model,
        // then retries without unloading the model serving the image.
        one.failNextStarts(["Image processing \(ServingMemoryPressure.exclusivePhrase). Finish or unload other sessions and models first."])
        let image = try request(model: "one", session: UUID(), image: true)
        let run = Task { await coordinator.handle(clientID: client, command: .start(image), emit: events.append) }
        #expect(await eventually { one.streamIDs.count == 1 })
        #expect(one.starts == 3, "the image request was retried exactly once")
        #expect(two.endSessionCalls == 1 && one.endSessionCalls == 1, "both idle sessions were closed")
        #expect(two.unloads == 1, "the idle other model yields its resident lease")
        #expect(await coordinator.snapshot().loadedModelIDs == ["one"])
        one.finish(one.streamIDs[0])
        await run.value
        // No failure was surfaced to the client.
        #expect(events.count { if case .generation(let id, let event) = $0 {
            return id.requestID == image.requestID && event.kind == .failed }; return false } == 0)
        await coordinator.shutdown()
    }
    @Test func concurrentSameModelPreparationsWaitAndCheckCompatibility() async throws {
        let fake = FakeHandle(), events = Events(), client = UUID()
        let coordinator = ServingModelCoordinator { _ in fake }
        func prep(_ context: Int) throws -> ServingModelPreparation {
            try ServingModelPreparation(modelID: "one", maxContextTokens: context,
                runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        }
        fake.holdPrepares(true)
        let first = Task { await coordinator.handle(clientID: client, command: .prepare(try prep(1024)), emit: events.append) }
        #expect(await eventually { fake.prepares == 1 })
        let second = Task { await coordinator.handle(clientID: UUID(), command: .prepare(try prep(1024)), emit: events.append) }
        let incompatible = Task { await coordinator.handle(clientID: UUID(), command: .prepare(try prep(2048)), emit: events.append) }
        #expect(await eventually {
            events.count { if case .lifecycle(_, _, .loading) = $0 { return true }; return false } == 3
        })
        fake.holdPrepares(false)
        try await first.value
        try await second.value
        try await incompatible.value
        #expect(fake.prepares == 1)
        #expect(events.count { if case .lifecycle(_, _, .ready) = $0 { return true }; return false } == 2)
        #expect(events.count { if case .lifecycle(_, _, .failed) = $0 { return true }; return false } == 1)
    }

    @Test func lostHelperDropsSessionsAndNextPrepareCreatesNewHandle() async throws {
        let dead = FakeHandle(), replacement = FakeHandle(), events = Events(), client = UUID()
        let resolutions = Mutex(0)
        let coordinator = ServingModelCoordinator { _ in
            resolutions.withLock { count in count += 1; return count == 1 ? dead : replacement }
        }
        let preparation = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        await coordinator.handle(clientID: client, command: .prepare(preparation), emit: events.append)
        dead.failTransport()
        let id = UUID()
        let request = try ServingGenerationRequest(modelID: "one", sessionID: UUID(), requestID: id,
            payload: DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                             temperature: 0, generationID: id))
        await coordinator.handle(clientID: client, command: .start(request), emit: events.append)
        #expect(await coordinator.snapshot().loadedModelIDs.isEmpty)
        #expect(await coordinator.snapshot().openSessions == 0)
        await coordinator.handle(clientID: client, command: .prepare(preparation), emit: events.append)
        #expect(replacement.prepares == 1)
        #expect(await coordinator.snapshot().loadedModelIDs == ["one"])
    }

    @Test func textMemoryFailureReclaimsIdleSessionAndRetries() async throws {
        let fake = FakeHandle(), events = Events(), client = UUID()
        let coordinator = ServingModelCoordinator { _ in fake }
        let preparation = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        await coordinator.handle(clientID: client, command: .prepare(preparation), emit: events.append)
        func request() throws -> ServingGenerationRequest {
            let id = UUID()
            return try ServingGenerationRequest(modelID: "one", sessionID: UUID(), requestID: id,
                payload: DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                                 temperature: 0, generationID: id))
        }
        let idle = try request()
        let first = Task { await coordinator.handle(clientID: client, command: .start(idle), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        fake.finish(fake.streamIDs[0])
        await first.value
        fake.failNextStarts(["Another process \(ServingMemoryPressure.phrase)"])
        let next = try request()
        let second = Task { await coordinator.handle(clientID: client, command: .start(next), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        #expect(fake.endSessionCalls == 1)
        #expect(fake.starts == 3)
        fake.finish(fake.streamIDs[0])
        await second.value
        #expect(await coordinator.snapshot().openSessions == 1)
    }

    @Test func shutdownRefusesOnlyTheOperationThatCausedIt() async throws {
        let fake = FakeHandle(), events = Events(), client = UUID()
        let coordinator = ServingModelCoordinator { _ in fake }
        await coordinator.shutdown()
        let preparation = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        let requestID = UUID(), session = UUID()
        let request = try ServingGenerationRequest(modelID: "one", sessionID: session, requestID: requestID,
            payload: DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                             temperature: 0, generationID: requestID))
        await coordinator.handle(clientID: client, command: .prepare(preparation), emit: events.append)
        await coordinator.handle(clientID: client, command: .start(request), emit: events.append)
        await coordinator.handle(clientID: client, command: .endSession(sessionID: session), emit: events.append)
        #expect(events.count {
            if case .lifecycle(let operation, _, .failed) = $0 { return operation == preparation.operationID }
            return false
        } == 1)
        #expect(events.count {
            if case .failed(let identity?, _) = $0 { return identity.requestID == requestID }
            return false
        } == 1)
        #expect(events.count {
            if case .sessionEndFailed(let id, _) = $0 { return id == session }
            return false
        } == 1)
        #expect(events.count { if case .failed(nil, _) = $0 { return true }; return false } == 0)
    }

    @Test func cleanupFromOldConnectionCannotTouchReconnectedSession() async throws {
        let fake = FakeHandle(), events = Events(), client = UUID(), session = UUID()
        let coordinator = ServingModelCoordinator { _ in fake }
        let old = ServingCommandContext(clientID: client, connectionID: UUID())
        let fresh = ServingCommandContext(clientID: client, connectionID: UUID())
        let preparation = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        await coordinator.handle(context: old, command: .prepare(preparation), emit: events.append)
        func request() throws -> ServingGenerationRequest {
            let id = UUID()
            return try ServingGenerationRequest(modelID: "one", sessionID: session, requestID: id,
                payload: DecodeGenerationRequest(prompt: "x", maxNewTokens: 4,
                    maxContextTokens: 1024, temperature: 0, generationID: id))
        }
        let first = try request(), second = try request()
        let firstRun = Task { await coordinator.handle(context: old, command: .start(first), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        let oldHelperID = try #require(fake.streamIDs.first)
        let secondRun = Task { await coordinator.handle(context: fresh, command: .start(second), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 2 })
        let newHelperID = try #require(fake.streamIDs.first(where: { $0 != oldHelperID }))
        // This is exactly what the disconnected connection's cleanup sends.
        await coordinator.handle(context: old, command: .cancel(sessionID: session, requestID: first.requestID),
                                 emit: events.append)
        await coordinator.handle(context: old, command: .endSession(sessionID: session), emit: events.append)
        await firstRun.value
        #expect(fake.cancelled.contains(oldHelperID))
        #expect(!fake.cancelled.contains(newHelperID))
        #expect(fake.streamIDs == [newHelperID])
        #expect(await coordinator.snapshot().openSessions == 1)
        fake.finish(newHelperID)
        await secondRun.value
        #expect(events.count {
            if case .generation(let id, let event) = $0 {
                return id.requestID == second.requestID && event.kind == .finished
            }
            return false
        } == 1)
        await coordinator.shutdown()
    }

    @Test func reclaimNeverCancelsActiveTurnsAndNewTurnWaitsForIdleClose() async throws {
        let fake = FakeHandle(), events = Events(), client = UUID()
        let coordinator = ServingModelCoordinator { _ in fake }
        let preparation = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        await coordinator.handle(clientID: client, command: .prepare(preparation), emit: events.append)
        func request(session: UUID, image: Bool = false) throws -> ServingGenerationRequest {
            let id = UUID()
            var payload = DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                                  temperature: 0, generationID: id)
            if image {
                payload.imageAttachments = [DecodeImageAttachment(id: UUID(), path: "/tmp/x.png",
                    displayName: "x.png", encodedBytes: 1, sha256: "0")]
            }
            return try ServingGenerationRequest(modelID: "one", sessionID: session, requestID: id, payload: payload)
        }
        let idleSession = UUID()
        let warm = try request(session: idleSession)
        let warming = Task { await coordinator.handle(clientID: client, command: .start(warm), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        fake.finish(try #require(fake.streamIDs.first))
        await warming.value
        let active = try request(session: UUID())
        let activeRun = Task { await coordinator.handle(clientID: client, command: .start(active), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        let activeHelperID = try #require(fake.streamIDs.first)
        fake.holdEndSessions(true)
        fake.failNextStarts(["Image processing \(ServingMemoryPressure.exclusivePhrase)"])
        let image = try request(session: UUID(), image: true)
        let imageRun = Task { await coordinator.handle(clientID: client, command: .start(image), emit: events.append) }
        #expect(await eventually { fake.endSessionCalls == 1 })
        let next = try request(session: idleSession)
        let nextRun = Task { await coordinator.handle(clientID: client, command: .start(next), emit: events.append) }
        // The close is held, so neither a cancellation nor an unavailable
        // failure is permitted for the unrelated active or next turn.
        #expect(fake.cancelled.isEmpty)
        #expect(events.count {
            if case .failed(let id?, _) = $0 { return id.requestID == next.requestID }
            return false
        } == 0)
        fake.holdEndSessions(false)
        #expect(await eventually { fake.streamIDs.count == 3 })
        #expect(!fake.cancelled.contains(activeHelperID))
        for id in fake.streamIDs { fake.finish(id) }
        await activeRun.value; await imageRun.value; await nextRun.value
        #expect(events.count {
            if case .generation(let id, let event) = $0 {
                return id.requestID == next.requestID && event.kind == .finished
            }
            return false
        } == 1)
        await coordinator.shutdown()
    }

    @Test func imageMemoryBusyWithNoIdleSessionFailsOnlyImage() async throws {
        let fake = FakeHandle(), events = Events(), client = UUID()
        let coordinator = ServingModelCoordinator { _ in fake }
        let preparation = try ServingModelPreparation(modelID: "one", maxContextTokens: 1024,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        await coordinator.handle(clientID: client, command: .prepare(preparation), emit: events.append)
        func request(image: Bool) throws -> ServingGenerationRequest {
            let id = UUID()
            var payload = DecodeGenerationRequest(prompt: "x", maxNewTokens: 4, maxContextTokens: 1024,
                                                  temperature: 0, generationID: id)
            if image {
                payload.imageAttachments = [DecodeImageAttachment(id: UUID(), path: "/tmp/x.png",
                    displayName: "x.png", encodedBytes: 1, sha256: "0")]
            }
            return try ServingGenerationRequest(modelID: "one", sessionID: UUID(), requestID: id, payload: payload)
        }
        let text = try request(image: false)
        let textRun = Task { await coordinator.handle(clientID: client, command: .start(text), emit: events.append) }
        #expect(await eventually { fake.streamIDs.count == 1 })
        let textHelperID = try #require(fake.streamIDs.first)
        fake.failNextStarts(["Image processing \(ServingMemoryPressure.exclusivePhrase)"])
        let image = try request(image: true)
        await coordinator.handle(clientID: client, command: .start(image), emit: events.append)
        #expect(fake.cancelled.isEmpty)
        #expect(fake.endSessionCalls == 0)
        #expect(fake.streamIDs == [textHelperID])
        #expect(events.count {
            if case .generation(let id, let event) = $0 {
                return id.requestID == image.requestID && event.kind == .failed
                    && (event.error ?? "").contains(ServingMemoryPressure.exclusivePhrase)
            }
            return false
        } == 1)
        fake.finish(textHelperID)
        await textRun.value
        await coordinator.shutdown()
    }
}
