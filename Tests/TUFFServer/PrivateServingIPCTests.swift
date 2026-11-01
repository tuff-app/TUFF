import Foundation
import Synchronization
import Testing
import TUFFDecodeProtocol
@testable import TUFFServerCore

func servingTestRoot(_ name: String) throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("tuff-serving-\(name)-\(UUID().uuidString.prefix(8))", isDirectory: true)
    try FileManager.default.createDirectory(
        at: root, withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700])
    return root
}

@Suite(.serialized) struct PrivateServingIPCTests {
    /// The bound only turns a hang into a failure; the outcome never depends on it.
    private func offPoolWait(_ semaphore: DispatchSemaphore, seconds: Int = 60) async -> Bool {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                continuation.resume(returning: semaphore.wait(timeout: .now() + .seconds(seconds)) == .success)
            }
        }
    }

    @Test func terminalReleasesSessionBeforePublication() async throws {
        let root = try servingTestRoot("terminal-order")
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.terminal-order")
        let firstID = UUID(), secondID = UUID(), sessionID = UUID()
        let publicationBlocked = DispatchSemaphore(value: 0)
        let releasePublication = DispatchSemaphore(value: 0)
        let secondAdmitted = DispatchSemaphore(value: 0)
        let server = PrivateServingIPCServer(profile: profile, token: try profile.token(),
                                             buildIdentity: "test") { context, command, emit in
            let clientID = context.clientID
            guard case .start(let request) = command else { return }
            context.markAdmitted()
            if request.requestID == secondID { secondAdmitted.signal() }
            // `beforeEventSend` blocks the publishing thread by design. Publish
            // from a dispatch thread so no cooperative-pool thread is held.
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                DispatchQueue.global().async {
                    emit(.generation(identity: ServingRequestIdentity(clientID: clientID, request: request),
                                     event: DecodeServiceEvent(kind: .finished, generationID: request.requestID)))
                    continuation.resume()
                }
            }
        }
        server.beforeEventSend = { event in
            if case .generation(let identity, _) = event, identity.requestID == firstID {
                publicationBlocked.signal()
                releasePublication.wait()
            }
        }
        try server.start()
        defer { releasePublication.signal(); server.shutdown() }
        let connection = try PrivateServingIPCConnection.connect(profile: profile, clientID: UUID())
        defer { connection.close() }
        func request(_ id: UUID) throws -> ServingGenerationRequest {
            try ServingGenerationRequest(
                modelID: "test-model", sessionID: sessionID, requestID: id,
                payload: DecodeGenerationRequest(prompt: "hello", maxNewTokens: 8,
                    maxContextTokens: 128, temperature: 0.2, generationID: id))
        }
        try connection.send(.start(request(firstID)))
        #expect(await offPoolWait(publicationBlocked))
        try connection.send(.start(request(secondID)))
        #expect(await offPoolWait(secondAdmitted))
        releasePublication.signal()
    }

    @Test func aServerDirectoryLeftBy9xIsTightenedNotRefused() throws {
        let root = try servingTestRoot("upgrade")
        defer { try? FileManager.default.removeItem(at: root) }
        let server = root.appendingPathComponent("TUFF/Server", isDirectory: true)
        try FileManager.default.createDirectory(at: server, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o755])
        try Data("{}".utf8).write(to: server.appendingPathComponent("background-server.json"))
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.upgrade")
        _ = try profile.token()
        let mode = try FileManager.default.attributesOfItem(atPath: server.path)[.posixPermissions] as? Int
        #expect(mode == 0o700)
        #expect(FileManager.default.fileExists(atPath: server.appendingPathComponent("background-server.json").path))
    }

    @Test func aSymlinkedServerDirectoryIsStillRefused() throws {
        let root = try servingTestRoot("symlink")
        defer { try? FileManager.default.removeItem(at: root) }
        let elsewhere = root.appendingPathComponent("elsewhere", isDirectory: true)
        try FileManager.default.createDirectory(at: elsewhere, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o755])
        try FileManager.default.createDirectory(at: root.appendingPathComponent("TUFF"),
                                                withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("TUFF/Server"),
                                                   withDestinationURL: elsewhere)
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.symlink")
        #expect(throws: ServingHostProfileError.self) { try profile.prepare() }
        let mode = try FileManager.default.attributesOfItem(atPath: elsewhere.path)[.posixPermissions] as? Int
        #expect(mode == 0o755)
    }

    @Test func privateSocketTokenIsSeparateFromHTTPControlToken() throws {
        let root = try servingTestRoot("tokens")
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.tokens")
        let control = profile.serverDirectoryURL.appendingPathComponent("control-token")
        let socketToken = try profile.token()
        try Data("http-token".utf8).write(to: control)
        #expect(profile.tokenURL != control)
        #expect(try profile.token() == socketToken)
        #expect(try String(contentsOf: control, encoding: .utf8) == "http-token")
    }

    @Test func duplicateIdentityAndBuildMismatchFailWithoutLaunching() async throws {
        let root = try servingTestRoot("terminal")
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.terminal")
        let owner = try ServingHostRuntime.startOwner(
            profile: profile, buildIdentity: "old", installed: ServerInstalledModels(modelsRoot: root))
        let first = try PrivateServingIPCConnection.connect(
            profile: profile, clientID: try profile.clientID(for: "duplicate"), expectedBuildIdentity: "old")
        defer { first.close() }
        #expect(throws: PrivateServingIPCError.rejected("client identity already has an active connection")) {
            _ = try ServingHostRuntime.connectOrLaunch(
                profile: profile, clientID: try profile.clientID(for: "duplicate"), role: .api, buildIdentity: "old",
                hostExecutable: URL(fileURLWithPath: "/nonexistent"))
        }
        #expect(throws: PrivateServingIPCError.incompatibleBuild(host: "old")) {
            _ = try ServingHostRuntime.connectOrLaunch(
                profile: profile, clientID: UUID(), role: .api, buildIdentity: "new",
                hostExecutable: URL(fileURLWithPath: "/nonexistent"))
        }
        await owner.shutdown()
    }

    @Test func concurrentLinkAttachmentsShareOneConnection() async throws {
        let root = try servingTestRoot("singleflight")
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.singleflight")
        let owner = try ServingHostRuntime.startOwner(
            profile: profile, buildIdentity: "test", installed: ServerInstalledModels(modelsRoot: root))
        let calls = Mutex(0)
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let connector: HostServingLink.Connector = {
            calls.withLock { $0 += 1 }
            entered.signal()
            release.wait()
            return try PrivateServingIPCConnection.connect(
                profile: profile, clientID: try profile.clientID(for: "singleflight"),
                expectedBuildIdentity: "test")
        }
        let link = HostServingLink(launching: connector, existing: connector)
        let first = Task { try await link.attach() }
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                entered.wait()
                continuation.resume()
            }
        }
        let second = Task { try await link.attach() }
        release.signal()
        let a = try await first.value
        let b = try await second.value
        #expect(a === b)
        #expect(calls.withLock { $0 } == 1)
        link.close()
        await owner.shutdown()
    }

    @Test func oneOwnerPerProfileAndStatusTracksLiveAPIAttachment() async throws {
        let root = try servingTestRoot("owner"); defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(
            applicationSupportURL: root, bundleIdentifier: "test.tuff.runtime")
        let installed = ServerInstalledModels(
            modelsRoot: root.appendingPathComponent("TUFF/Models", isDirectory: true))
        let owner = try ServingHostRuntime.startOwner(
            profile: profile, buildIdentity: "test", installed: installed)
        // A racing second host loses at the profile lease instead of
        // creating a second model pool.
        #expect(throws: PrivateServingIPCError.hostAlreadyRunning) {
            _ = try ServingHostRuntime.startOwner(
                profile: profile, buildIdentity: "test", installed: installed)
        }
        let gui = try PrivateServingIPCConnection.connect(
            profile: profile, clientID: try profile.clientID(for: "gui"), role: .gui,
            expectedBuildIdentity: "test")
        defer { gui.close() }
        let guiClient = PrivateServingIPCClient(connection: gui)
        func status() async throws -> ServingHostStatus {
            for try await event in guiClient.events(for: .hostStatus(operationID: UUID())) {
                if case .hostStatus(_, let value) = event { return value }
            }
            throw PrivateServingIPCClientError.closed
        }
        #expect(try await status().apiAdapterAttached == false)

        let api = try PrivateServingIPCConnection.connect(
            profile: profile, clientID: try profile.clientID(for: "api"), role: .api,
            expectedBuildIdentity: "test")
        let attached = try await status()
        #expect(attached.apiAdapterAttached)
        #expect(attached.hostID == api.host.hostID)
        #expect(attached.connectedClients == 2)
        api.close()
        // The adapter's detachment is reflected without reconnecting the GUI.
        var detached = true
        for _ in 0..<100 {
            detached = try await status().apiAdapterAttached
            if !detached { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!detached)
        guiClient.close()
        await owner.shutdown()
    }

    @Test func mismatchedBuildIsNotAttachedTo() async throws {
        let root = try servingTestRoot("build"); defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(
            applicationSupportURL: root, bundleIdentifier: "test.tuff.build")
        let owner = try ServingHostRuntime.startOwner(
            profile: profile, buildIdentity: "tuff-old",
            installed: ServerInstalledModels(modelsRoot: root))
        #expect(throws: PrivateServingIPCError.incompatibleBuild(host: "tuff-old")) {
            _ = try PrivateServingIPCConnection.connect(
                profile: profile, clientID: try profile.clientID(), expectedBuildIdentity: "tuff-new")
        }
        await owner.shutdown()
    }

    @Test func olderHostIsIdentifiedForTakeoverAndNewerIsNot() async throws {
        let root = try servingTestRoot("older"); defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.tuff.older")
        let owner = try ServingHostRuntime.startOwner(
            profile: profile, buildIdentity: "tuff-10.0.1",
            installed: ServerInstalledModels(modelsRoot: root))
        // The host runs in this process, so its socket peer is this process.
        #expect(throws: PrivateServingIPCError.olderBuild(host: "tuff-10.0.1", pid: getpid())) {
            _ = try PrivateServingIPCConnection.connect(
                profile: profile, clientID: try profile.clientID(), expectedBuildIdentity: "tuff-10.0.2")
        }
        #expect(throws: PrivateServingIPCError.incompatibleBuild(host: "tuff-10.0.1")) {
            _ = try PrivateServingIPCConnection.connect(
                profile: profile, clientID: try profile.clientID(), expectedBuildIdentity: "tuff-10.0.0")
        }
        await owner.shutdown()
    }

    @Test func buildVersionsCompareNumerically() {
        #expect(ServingHostTakeover.isOlder("tuff-9.6.0", than: "tuff-10.0.2"))
        #expect(ServingHostTakeover.isOlder("tuff-10.0.1", than: "tuff-10.0.10"))
        #expect(ServingHostTakeover.isOlder("tuff-10.0.9", than: "tuff-10.1.0"))
        #expect(!ServingHostTakeover.isOlder("tuff-10.0.2", than: "tuff-10.0.2"))
        #expect(!ServingHostTakeover.isOlder("tuff-10.1.0", than: "tuff-10.0.2"))
        #expect(!ServingHostTakeover.isOlder("old", than: "tuff-10.0.2"))
        #expect(!ServingHostTakeover.isOlder("tuff-10.0", than: "tuff-10.0.2"))
        #expect(!ServingHostTakeover.isOlder("tuff-10.0.1", than: "test"))
    }

    @Test func takeoverOnlySignalsATUFFServer() throws {
        let sleeper = Process()
        sleeper.executableURL = URL(fileURLWithPath: "/bin/sleep")
        sleeper.arguments = ["30"]
        try sleeper.run()
        defer { sleeper.terminate(); sleeper.waitUntilExit() }
        #expect(!ServingHostTakeover.retire(pid: sleeper.processIdentifier))
        #expect(!ServingHostTakeover.retire(pid: getpid()))
        #expect(sleeper.isRunning)
        #expect(ServingHostTakeover.arguments(of: sleeper.processIdentifier) == ["/bin/sleep", "30"])
    }

    @Test func profilesForDifferentRootsOrBundlesNeverShareASocket() throws {
        let one = try servingTestRoot("a"), two = try servingTestRoot("b")
        defer { try? FileManager.default.removeItem(at: one); try? FileManager.default.removeItem(at: two) }
        let production = try ServingHostProfile(applicationSupportURL: one, bundleIdentifier: "com.rexmhall09.TUFF")
        let preview = try ServingHostProfile(applicationSupportURL: one, bundleIdentifier: "com.rexmhall09.TUFF.local10")
        let moved = try ServingHostProfile(applicationSupportURL: two, bundleIdentifier: "com.rexmhall09.TUFF")
        #expect(Set([production.socketURL, preview.socketURL, moved.socketURL]).count == 3)
        #expect(production.lockURL != moved.lockURL)
        #expect(Set([production.profileIdentity, preview.profileIdentity, moved.profileIdentity]).count == 3)
        // One support root is claimed by one bundle identity; another bundle
        // pointed at the same root fails closed instead of sharing its host.
        try production.prepare()
        #expect(throws: ServingHostProfileError.profileIdentityMismatch) { try preview.prepare() }
    }

    @Test func duplicateClientConnectionAndStaleCommandsAreHandledWithoutDisturbingStreams() throws {
        let root = try servingTestRoot("ipc"); defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(
            applicationSupportURL: root, bundleIdentifier: "test.tuff.serving")
        let endSessionAttempts = Mutex(0)
        let server = PrivateServingIPCServer(
            profile: profile, token: try profile.token(),
            buildIdentity: "test") { context, command, emit in
                let clientID = context.clientID
                switch command {
                case .start(let request):
                    emit(.generation(
                        identity: ServingRequestIdentity(clientID: clientID, request: request),
                        event: DecodeServiceEvent(kind: .finished, generationID: request.requestID)))
                case .endSession(let sessionID):
                    let attempt = endSessionAttempts.withLock { value -> Int in
                        value += 1
                        return value
                    }
                    if attempt == 1 {
                        emit(.sessionEndFailed(sessionID: sessionID, message: "cleanup failed"))
                    } else {
                        emit(.sessionEnded(sessionID: sessionID))
                    }
                default: break
                }
            }
        try server.start()
        defer { server.shutdown() }

        let clientID = try profile.clientID()
        let cliID = try profile.clientID(for: "cli")
        #expect(cliID != clientID)
        #expect(try profile.clientID(for: "cli") == cliID)
        let first = try PrivateServingIPCConnection.connect(profile: profile, clientID: clientID)
        defer { first.close() }
        #expect(first.host.profileIdentity == profile.profileIdentity)

        #expect(throws: PrivateServingIPCError.self) {
            _ = try PrivateServingIPCConnection.connect(profile: profile, clientID: clientID)
        }

        // A stale cancel is ignored silently; a session never opened here is
        // already closed. Neither produces an uncorrelated failure.
        try first.send(.cancel(sessionID: UUID(), requestID: UUID()))
        let unopened = UUID()
        try first.send(.endSession(sessionID: unopened))
        guard case .sessionEnded(let closed) = try first.receive() else {
            Issue.record("a session this connection never opened is already closed")
            return
        }
        #expect(closed == unopened)

        let sessionID = UUID()
        for _ in 0..<2 {
            let requestID = UUID()
            let payload = DecodeGenerationRequest(
                prompt: "hello", maxNewTokens: 8, maxContextTokens: 128,
                temperature: 0.2, generationID: requestID)
            let request = try ServingGenerationRequest(
                modelID: "test-model", sessionID: sessionID, requestID: requestID,
                payload: payload)
            try first.send(.start(request))
            guard case .generation(let identity, let event) = try first.receive() else {
                Issue.record("generation request should receive a routed terminal event")
                return
            }
            #expect(identity.sessionID == sessionID)
            #expect(identity.requestID == requestID)
            #expect(event.kind == .finished)
        }
        try first.send(.endSession(sessionID: sessionID))
        guard case .sessionEndFailed(let failedSession, let cleanupMessage) = try first.receive() else {
            Issue.record("failed session cleanup should leave the session owned for retry")
            return
        }
        #expect(failedSession == sessionID)
        #expect(cleanupMessage == "cleanup failed")
        try first.send(.endSession(sessionID: sessionID))
        guard case .sessionEnded(let endedSessionID) = try first.receive() else {
            Issue.record("a failed cleanup must remain retryable on the same connection")
            return
        }
        #expect(endedSessionID == sessionID)
    }

    @Test func commandsReachTheHandlerInTheOrderThePeerSentThem() async throws {
        let root = try servingTestRoot("order"); defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.order")
        let seen = Mutex<[String]>([])
        let cancelled = DispatchSemaphore(value: 0)
        let server = PrivateServingIPCServer(
            profile: profile, token: try profile.token(), buildIdentity: "test") { context, command, emit in
                let clientID = context.clientID
                switch command {
                case .start(let request):
                    // Slow admission: an unordered dispatcher would run the
                    // cancel while this start is still being admitted.
                    try? await Task.sleep(for: .milliseconds(150))
                    seen.withLock { $0.append("start") }
                    context.markAdmitted()
                    await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                        DispatchQueue.global().async { cancelled.wait(); continuation.resume() }
                    }
                    emit(.generation(identity: ServingRequestIdentity(clientID: clientID, request: request),
                                     event: DecodeServiceEvent(kind: .cancelled, generationID: request.requestID)))
                case .cancel:
                    seen.withLock { $0.append("cancel") }
                    cancelled.signal()
                default: break
                }
            }
        try server.start()
        defer { cancelled.signal(); server.shutdown() }
        let connection = try PrivateServingIPCConnection.connect(profile: profile, clientID: UUID())
        defer { connection.close() }
        let sessionID = UUID(), requestID = UUID()
        let request = try ServingGenerationRequest(
            modelID: "test-model", sessionID: sessionID, requestID: requestID,
            payload: DecodeGenerationRequest(prompt: "hello", maxNewTokens: 8, maxContextTokens: 128,
                                             temperature: 0.2, generationID: requestID))
        try connection.send(.start(request))
        try connection.send(.cancel(sessionID: sessionID, requestID: requestID))
        guard case .generation(_, let event) = try connection.receive() else {
            Issue.record("the cancelled request should receive its terminal event")
            return
        }
        #expect(event.kind == .cancelled)
        #expect(seen.withLock { $0 } == ["start", "cancel"])
    }

    @Test func idleExitIsAtomicWithConnectionAdmission() async throws {
        let root = try servingTestRoot("retire"); defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.retire")
        let server = PrivateServingIPCServer(
            profile: profile, token: try profile.token(), buildIdentity: "test") { _, _, _ in }
        try server.start()
        defer { server.shutdown() }
        let first = try PrivateServingIPCConnection.connect(profile: profile, clientID: UUID())
        // An attached client always blocks the exit.
        #expect(!server.retireIfIdle())
        first.close()
        var committed = false
        for _ in 0..<500 {
            if server.retireIfIdle() { committed = true; break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(committed)
        // After the commit a connecting client is refused with the typed
        // retiring error, so it relaunches a fresh host.
        #expect(throws: PrivateServingIPCError.hostRetiring) {
            _ = try PrivateServingIPCConnection.connect(profile: profile, clientID: UUID())
        }
        #expect(!server.retireIfIdle())
    }
}
