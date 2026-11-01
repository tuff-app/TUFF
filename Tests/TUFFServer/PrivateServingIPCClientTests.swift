import Foundation
import Synchronization
import Testing
import TUFFDecodeProtocol
@testable import TUFFServerCore

/// Drives the real socket server and the multiplexing client with a scripted
/// command handler, so routing, cancellation and failure paths are exercised
/// without a model.
private final class ScriptState: Sendable {
    let cancelled = Mutex(Set<UUID>())
    let endAttempts = Mutex(0)
    /// Requests whose handler should stay open until cancelled.
    let holdOpen = Mutex(Set<UUID>())
    let floodRequests = Mutex(Set<UUID>())
    let failUncorrelated = Mutex(false)
}

private final class ScriptedHost: @unchecked Sendable {
    let root: URL
    let profile: ServingHostProfile
    let server: PrivateServingIPCServer
    let script = ScriptState()

    init() throws {
        root = try servingTestRoot("client")
        profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.tuff.client")
        let script = self.script
        server = PrivateServingIPCServer(
            profile: profile, token: try profile.token(), buildIdentity: "test"
        ) { context, command, emit in
            let clientID = context.clientID
            switch command {
            case .start(let request):
                let identity = ServingRequestIdentity(clientID: clientID, request: request)
                context.markAdmitted()
                if script.failUncorrelated.withLock({ $0 }) {
                    emit(.failed(identity: nil, message: "host refuses work"))
                    return
                }
                if script.floodRequests.withLock({ $0.contains(request.requestID) }) {
                    var sequence: UInt64 = 1
                    while !script.cancelled.withLock({ $0.contains(request.requestID) }), sequence < 200_000 {
                        var event = DecodeServiceEvent(kind: .snapshot, generationID: request.requestID)
                        event.sequence = sequence
                        emit(.generation(identity: identity, event: event))
                        sequence += 1
                        if sequence % 256 == 0 { try? await Task.sleep(for: .milliseconds(1)) }
                    }
                    emit(.generation(identity: identity, event: DecodeServiceEvent(
                        kind: .cancelled, generationID: request.requestID)))
                    return
                }
                if script.holdOpen.withLock({ $0.contains(request.requestID) }) {
                    while !script.cancelled.withLock({ $0.contains(request.requestID) }) {
                        try? await Task.sleep(for: .milliseconds(5))
                    }
                    emit(.generation(identity: identity, event: DecodeServiceEvent(
                        kind: .cancelled, generationID: request.requestID)))
                    return
                }
                emit(.generation(identity: identity, event: DecodeServiceEvent(
                    kind: .finished, generationID: request.requestID)))
            case .cancel(_, let requestID):
                _ = script.cancelled.withLock { $0.insert(requestID) }
            case .endSession(let sessionID):
                let attempt = script.endAttempts.withLock { value -> Int in value += 1; return value }
                emit(attempt == 1
                    ? .sessionEndFailed(sessionID: sessionID, message: "cleanup failed")
                    : .sessionEnded(sessionID: sessionID))
            default: break
            }
        }
        try server.start()
    }

    func connect(_ name: String = "gui") throws -> PrivateServingIPCClient {
        PrivateServingIPCClient(connection: try PrivateServingIPCConnection.connect(
            profile: profile, clientID: try profile.clientID(for: name)))
    }

    func request(session: UUID = UUID(), hold: Bool = false, flood: Bool = false) throws
        -> ServingGenerationRequest {
        let id = UUID()
        if hold { _ = script.holdOpen.withLock { $0.insert(id) } }
        if flood { _ = script.floodRequests.withLock { $0.insert(id) } }
        return try ServingGenerationRequest(
            modelID: "m", sessionID: session, requestID: id,
            payload: DecodeGenerationRequest(prompt: "p", maxNewTokens: 1, maxContextTokens: 64,
                                             temperature: 0, generationID: id))
    }

    func finish() { server.shutdown(); try? FileManager.default.removeItem(at: root) }
}

private func drain(_ stream: AsyncThrowingStream<ServingEvent, any Error>) async throws -> [ServingEvent] {
    var events: [ServingEvent] = []
    for try await event in stream { events.append(event) }
    return events
}

private func eventually(_ condition: () -> Bool) async -> Bool {
    for _ in 0..<500 {
        if condition() { return true }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return condition()
}

@Suite(.serialized) struct PrivateServingIPCClientTests {
    @Test func sequentialTurnsOnOneSessionAreAdmittedImmediately() async throws {
        let host = try ScriptedHost(); defer { host.finish() }
        let client = try host.connect(); defer { client.close() }
        let session = UUID()
        for _ in 0..<5 {
            let events = try await drain(client.events(for: .start(try host.request(session: session))))
            guard case .generation(_, let event)? = events.last else {
                Issue.record("terminal event expected"); return
            }
            #expect(event.kind == .finished)
        }
    }

    @Test func cancellingOneStreamCancelsOnlyThatRequest() async throws {
        let host = try ScriptedHost(); defer { host.finish() }
        let client = try host.connect(); defer { client.close() }
        let held = try host.request(hold: true)
        let other = try host.request(hold: true)
        let heldTask = Task { try await drain(client.events(for: .start(held))) }
        let otherTask = Task { try await drain(client.events(for: .start(other))) }
        try await Task.sleep(for: .milliseconds(200))
        heldTask.cancel()
        #expect(await eventually { host.script.cancelled.withLock { $0.contains(held.requestID) } })
        #expect(!host.script.cancelled.withLock { $0.contains(other.requestID) })
        // The other stream is unaffected until it is cancelled itself.
        otherTask.cancel()
        #expect(await eventually { host.script.cancelled.withLock { $0.contains(other.requestID) } })
        _ = await heldTask.result; _ = await otherTask.result
    }

    @Test func aStalledConsumerIsFailedAndItsGenerationCancelled() async throws {
        let host = try ScriptedHost(); defer { host.finish() }
        let client = try host.connect(); defer { client.close() }
        let flood = try host.request(flood: true)
        let stream = client.events(for: .start(flood))
        // Nobody reads while the host floods past the per-stream bound.
        #expect(await eventually { host.script.cancelled.withLock { $0.contains(flood.requestID) } })
        var received = 0
        do {
            for try await _ in stream { received += 1 }
            Issue.record("a stalled stream must end with an error")
        } catch PrivateServingIPCClientError.slowConsumer {
            #expect(received <= PrivateServingIPCClient.streamCapacity)
        }
        // The connection and its other streams keep working.
        let events = try await drain(client.events(for: .start(try host.request())))
        #expect(!events.isEmpty)
    }

    @Test func hostLossFailsEveryPendingStream() async throws {
        let host = try ScriptedHost()
        defer { try? FileManager.default.removeItem(at: host.root) }
        let client = try host.connect(); defer { client.close() }
        let first = try host.request(hold: true), second = try host.request(hold: true)
        let a = Task { try await drain(client.events(for: .start(first))) }
        let b = Task { try await drain(client.events(for: .start(second))) }
        try await Task.sleep(for: .milliseconds(200))
        host.server.shutdown()
        for task in [a, b] {
            switch await task.result {
            case .success: Issue.record("stream should fail when the host disappears")
            case .failure: break
            }
        }
        #expect(await eventually { client.isClosed })
        // A new stream on the dead connection fails immediately.
        do { _ = try await drain(client.events(for: .start(try host.request()))); Issue.record("should fail") }
        catch {}
    }

    @Test func aConnectionLevelHostFailureEndsEachPendingOperationTypedAndCancelsItHostSide() async throws {
        let host = try ScriptedHost(); defer { host.finish() }
        let client = try host.connect(); defer { client.close() }
        let held = try host.request(hold: true)
        let pending = Task { try await drain(client.events(for: .start(held))) }
        try await Task.sleep(for: .milliseconds(150))
        host.script.failUncorrelated.withLock { $0 = true }
        let refused = try host.request()
        do { _ = try await drain(client.events(for: .start(refused))) } catch {}
        switch await pending.result {
        case .success: Issue.record("pending stream should fail")
        case .failure(let error):
            #expect(error as? PrivateServingIPCClientError == .hostFault("host refuses work"))
        }
        // The host was told to stop the work it can no longer deliver.
        #expect(await eventually { host.script.cancelled.withLock { $0.contains(held.requestID) } })
        // The connection itself survives for later operations.
        host.script.failUncorrelated.withLock { $0 = false }
        let events = try await drain(client.events(for: .start(try host.request())))
        #expect(!events.isEmpty)
    }

    @Test func endSessionFailureIsReportedAndRetryable() async throws {
        let host = try ScriptedHost(); defer { host.finish() }
        let client = try host.connect(); defer { client.close() }
        let session = UUID()
        _ = try await drain(client.events(for: .start(try host.request(session: session))))
        do {
            _ = try await drain(client.events(for: .endSession(sessionID: session)))
            Issue.record("the first cleanup is scripted to fail")
        } catch PrivateServingIPCClientError.remote(let message) {
            #expect(message == "cleanup failed")
        }
        // The session was not released, so the same close can be retried.
        let retry = try await drain(client.events(for: .endSession(sessionID: session)))
        guard case .sessionEnded(let ended)? = retry.last else { Issue.record("expected sessionEnded"); return }
        #expect(ended == session)
    }

    @Test func aRequestIDCannotBeReusedWhileItsStreamIsOpen() async throws {
        let host = try ScriptedHost(); defer { host.finish() }
        let client = try host.connect(); defer { client.close() }
        let held = try host.request(hold: true)
        let first = Task { try await drain(client.events(for: .start(held))) }
        try await Task.sleep(for: .milliseconds(150))
        do { _ = try await drain(client.events(for: .start(held))); Issue.record("duplicate must fail") }
        catch PrivateServingIPCClientError.requestAlreadyActive {}
        first.cancel()
        _ = await first.result
    }
}
