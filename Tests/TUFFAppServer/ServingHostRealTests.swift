import Darwin
import Foundation
import Testing
import TUFFAppCore
import TUFFDecodeProtocol
import TUFFServerCore
@testable import TUFFAppServer

/// Opt-in real-model evidence for the shared serving pool. Runs the actual
/// `TUFFServer --private-host` process, the GUI client and an API-role client
/// against real helpers. Guard-returns (a fast "pass" that exercised nothing)
/// unless every variable is present:
///
/// - `TUFF_SERVING_REAL_SUPPORT`: an Application Support root holding
///   `TUFF/Models/gemma4-e2b.gturbo` and `TUFF/Models/bonsai-2-2b.gturbo`
///   (installed layout, each with `verified-install.json`).
/// - `TUFF_SERVING_HOST_EXECUTABLE`: the built `TUFFServer`.
/// - `TUFF_SERVING_HELPER_EXECUTABLE`: the built `TUFFDecodeService` helper.
private final class PIDList: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [pid_t] = []
    func add(_ pid: pid_t) { lock.withLock { values.append(pid) } }
    var all: [pid_t] { lock.withLock { values } }
}

@Suite(.serialized) struct ServingHostRealTests {
    private struct Fixture {
        let support: URL
        let host: URL
        let helper: URL
    }

    private static func fixture() -> Fixture? {
        let environment = ProcessInfo.processInfo.environment
        guard let support = environment["TUFF_SERVING_REAL_SUPPORT"],
              let host = environment["TUFF_SERVING_HOST_EXECUTABLE"],
              let helper = environment["TUFF_SERVING_HELPER_EXECUTABLE"] else { return nil }
        return Fixture(support: URL(fileURLWithPath: support, isDirectory: true),
                       host: URL(fileURLWithPath: host), helper: URL(fileURLWithPath: helper))
    }

    private func log(_ line: String) {
        print("[serving-real] \(line)")
        fflush(stdout)
    }

    /// Greedy single-turn text through a scoped chat session of the GUI client.
    private func guiText(_ client: ServingHostInferenceClient, directory: URL, session: UUID,
                         prompt: String, options: AppRuntimeOptions, maxNewTokens: Int = 8,
                         onFirstToken: (() -> Void)? = nil) async throws -> String {
        let request = AppGenerationRequest(
            modelDirectory: directory, prompt: prompt, maxNewTokens: maxNewTokens, maxContextTokens: 1024,
            temperature: 0, topK: nil, topP: nil, runtimeOptions: options)
        var text = ""
        var finished = false
        for try await event in client.scopedSession(session).generate(request) {
            switch event {
            case .token(let token):
                if text.isEmpty { onFirstToken?() }
                text += token.textDelta
            case .finished: finished = true
            case .failed(let error, _): Issue.record("generation failed: \(error)")
            default: break
            }
        }
        #expect(finished)
        return text
    }

    private func apiText(_ ipc: PrivateServingIPCClient, modelID: String, session: UUID,
                         prompt: String, options: AppRuntimeOptions, maxNewTokens: Int = 8,
                         onFirstToken: (() -> Void)? = nil) async throws -> String {
        let requestID = UUID()
        let payload = DecodeGenerationTranslation.decodeRequest(
            AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/unused"), prompt: prompt,
                                 maxNewTokens: maxNewTokens, maxContextTokens: 1024, temperature: 0,
                                 topK: nil, topP: nil, runtimeOptions: options),
            generationID: requestID)
        let request = try ServingGenerationRequest(
            modelID: modelID, sessionID: session, requestID: requestID, payload: payload)
        var text = ""
        var finished = false
        for try await event in ipc.events(for: .start(request)) {
            if case .generation(_, let decode) = event {
                if !decode.textDelta.isEmpty, text.isEmpty { onFirstToken?() }
                text += decode.textDelta
                if decode.kind == .finished { finished = true }
            }
        }
        #expect(finished)
        return text
    }

    private func helperChildren(of pid: pid_t) -> Int {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        process.arguments = ["-P", "\(pid)"]
        let pipe = Pipe()
        process.standardOutput = pipe
        try? process.run()
        process.waitUntilExit()
        let text = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return text.split(whereSeparator: \.isNewline).count
    }

    /// `TUFF_SERVING_REAL_VISION_SUPPORT` is an isolated Application Support
    /// root whose TUFF/Models contains the verified E2B text and vision packs.
    /// This is separate from the two-model fixture above.
    @Test(.timeLimit(.minutes(20)))
    func secondLiveSessionCanUseInstalledVisionCompanion() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let support = environment["TUFF_SERVING_REAL_VISION_SUPPORT"],
              let host = environment["TUFF_SERVING_HOST_EXECUTABLE"],
              let helper = environment["TUFF_SERVING_HELPER_EXECUTABLE"] else { return }
        let root = URL(fileURLWithPath: support, isDirectory: true)
        setenv("TUFF_APPLICATION_SUPPORT_ROOT", root.path, 1)
        setenv("TUFF_LOCAL_BUILD", "1", 1)
        setenv("TUFF_SERVING_HELPER_EXECUTABLE", helper, 1)
        let bundle = "test.tuff.real-vision-peer"
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: bundle)
        let models = root.appendingPathComponent("TUFF/Models", isDirectory: true)
        let gemma = models.appendingPathComponent("gemma4-e2b.gturbo", isDirectory: true)
        let launched = PIDList()
        let gui = ServingHostInferenceClient(modelsRoot: models) {
            let attachment = try ServingHostRuntime.connectOrLaunch(
                profile: profile, clientID: UUID(), role: .gui,
                hostExecutable: URL(fileURLWithPath: host))
            if let pid = attachment.launchedHostPID { launched.add(pid) }
            return attachment.connection
        }
        defer { for pid in launched.all { kill(pid, SIGTERM) } }
        var options = AppRuntimeOptions()
        options.modelVerification = .fullSha256
        try await gui.ensureLoaded(modelDirectory: gemma, maxContextTokens: 1024, options: options,
                                   forceLogitsHead: false) { _ in }
        let owner = UUID()
        let text = try await guiText(gui, directory: gemma, session: owner,
                                     prompt: "What is 2 plus 2?", options: options)
        #expect(!text.isEmpty)

        let image = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("TUFFEngine/Core/Runtime/Vision/Fixtures/images/natural-square.png")
        let store = AppImageAttachmentStore()
        let attachment = try store.stage(image)
        defer { store.remove(attachment) }
        let peer = UUID()
        let request = AppGenerationRequest(
            modelDirectory: gemma, prompt: "Describe this image briefly.",
            imageAttachments: [attachment], maxNewTokens: 8, maxContextTokens: 1024,
            temperature: 0, topK: nil, topP: nil, runtimeOptions: options)
        var finished = false
        var answer = ""
        for try await event in gui.scopedSession(peer).generate(request) {
            switch event {
            case .token(let token): answer += token.textDelta
            case .finished: finished = true
            case .failed(let error, _): Issue.record("peer image generation failed: \(error)")
            default: break
            }
        }
        #expect(finished)
        #expect(!answer.isEmpty)
        #expect(try await gui.hostStatus().openSessions == 2,
                "the owner must remain live; otherwise this did not exercise a peer")
        try await gui.endSession(peer, requestID: UUID())
        try await gui.endSession(owner, requestID: UUID())
    }

    @Test(.timeLimit(.minutes(20)))
    func commonHostSharesOneHandlePerModelAcrossGUIAndAPIAndSurvivesOwnerExit() async throws {
        guard let fixture = Self.fixture() else { return }
        setenv("TUFF_APPLICATION_SUPPORT_ROOT", fixture.support.path, 1)
        setenv("TUFF_LOCAL_BUILD", "1", 1)
        setenv("TUFF_SERVING_HELPER_EXECUTABLE", fixture.helper.path, 1)
        let bundle = "test.tuff.real-host"
        let profile = try ServingHostProfile(applicationSupportURL: fixture.support, bundleIdentifier: bundle)
        let models = fixture.support.appendingPathComponent("TUFF/Models", isDirectory: true)
        let gemma = models.appendingPathComponent("gemma4-e2b.gturbo", isDirectory: true)
        let bonsai = models.appendingPathComponent("bonsai-2-2b.gturbo", isDirectory: true)
        var options = AppRuntimeOptions()
        options.modelVerification = .fullSha256

        let launched = PIDList()
        @Sendable func connect(_ name: String, _ role: ServingClientRole) throws -> ServingHostRuntime.Attachment {
            let attachment = try ServingHostRuntime.connectOrLaunch(
                profile: profile, clientID: try profile.clientID(for: name), role: role,
                hostExecutable: fixture.host)
            if let pid = attachment.launchedHostPID { launched.add(pid) }
            return attachment
        }
        defer { for pid in launched.all { kill(pid, SIGTERM) } }

        let gui = ServingHostInferenceClient(modelsRoot: models) {
            try connect("gui", .gui).connection
        }
        // GUI loads Gemma E2B; the host is launched by this first attach.
        let loadStarted = Date()
        try await gui.ensureLoaded(modelDirectory: gemma, maxContextTokens: 1024, options: options,
                                   forceLogitsHead: false) { _ in }
        log("gui loaded gemma4-e2b in \(String(format: "%.1f", Date().timeIntervalSince(loadStarted))) s")
        let guiStatus = try await gui.hostStatus()
        #expect(guiStatus.loadedModelIDs == ["gemma-4-e2b-it"])
        let hostID = guiStatus.hostID

        // Immediate next turns on one chat session never see generationInFlight.
        let chat = UUID()
        var guiAnswer = ""
        for turn in 0..<4 {
            let text = try await guiText(gui, directory: gemma, session: chat,
                                         prompt: "What is 2 plus 2?", options: options)
            if turn == 0 { guiAnswer = text }
            #expect(!text.isEmpty)
        }
        log("gui greedy text: \(guiAnswer.debugDescription)")

        // API path: a separate attachment prepares the same model with the
        // same configuration; the host answers ready immediately (no second
        // helper) and the greedy output matches the GUI's.
        let api = try connect("api", .api)
        let apiIPC = PrivateServingIPCClient(connection: api.connection)
        defer { apiIPC.close() }
        let preparation = try ServingModelPreparation(
            modelID: "gemma-4-e2b-it", maxContextTokens: 1024,
            runtimeOptions: DecodeServiceInferenceClient.decodeRuntimeOptions(options),
            forceLogitsHead: false)
        let prepareStarted = Date()
        for try await _ in apiIPC.events(for: .prepare(preparation)) {}
        let prepareSeconds = Date().timeIntervalSince(prepareStarted)
        log("api prepare of the resident model took \(String(format: "%.2f", prepareSeconds)) s")
        #expect(prepareSeconds < 5)
        let apiSession = UUID()
        let apiAnswer = try await apiText(apiIPC, modelID: "gemma-4-e2b-it", session: apiSession,
                                          prompt: "What is 2 plus 2?", options: options)
        log("api greedy text: \(apiAnswer.debugDescription)")
        #expect(apiAnswer == guiAnswer)
        var status = try await gui.hostStatus()
        #expect(status.hostID == hostID)
        #expect(status.apiAdapterAttached)
        #expect(status.loadedModelIDs == ["gemma-4-e2b-it"])
        let hostPID = try #require(launched.all.first)
        #expect(hostPID != 35872, "the isolated fixture must not attach to production")
        #expect(helperChildren(of: hostPID) == 1, "one helper process serves both surfaces")

        // Closing the other surface during an active stream must not take the
        // independent host, or that stream, down with it.
        let departingGUI = try connect("departing-gui", .gui)
        let apiAfterGUIExit = try await apiText(
            apiIPC, modelID: "gemma-4-e2b-it", session: apiSession,
            prompt: "What is 2 plus 2?", options: options,
            onFirstToken: { departingGUI.connection.close() })
        #expect(apiAfterGUIExit == guiAnswer)
        let departingAPI = try connect("departing-api", .api)
        let guiAfterAPIExit = try await guiText(
            gui, directory: gemma, session: chat, prompt: "What is 2 plus 2?",
            options: options, onFirstToken: { departingAPI.connection.close() })
        #expect(guiAfterAPIExit == guiAnswer)

        // A second, different model resident at once, driven from the GUI
        // while the API streams on the first.
        try await gui.ensureLoaded(modelDirectory: bonsai, maxContextTokens: 1024, options: options,
                                   forceLogitsHead: false) { _ in }
        status = try await gui.hostStatus()
        #expect(Set(status.loadedModelIDs) == ["gemma-4-e2b-it", "bonsai-2-2b"])
        #expect(helperChildren(of: hostPID) == 2)
        async let fromAPI = apiText(apiIPC, modelID: "gemma-4-e2b-it", session: apiSession,
                                    prompt: "What is 2 plus 2?", options: options)
        async let fromGUI = guiText(gui, directory: bonsai, session: UUID(),
                                    prompt: "What is 3 plus 4?", options: options)
        let (a, b) = try await (fromAPI, fromGUI)
        log("concurrent api(gemma)=\(a.debugDescription) gui(bonsai)=\(b.debugDescription)")
        #expect(a == guiAnswer)
        #expect(!b.isEmpty)


        // Owner exit: the host dies; the GUI reconnects, relaunches it and
        // re-prepares transparently on its next turn.
        kill(hostPID, SIGTERM)
        for _ in 0..<400 where kill(hostPID, 0) == 0 { try await Task.sleep(for: .milliseconds(25)) }
        let probe = try? await gui.hostStatus()
        #expect(apiIPC.isClosed || probe != nil)
        let afterExit = try await guiText(gui, directory: gemma, session: UUID(),
                                          prompt: "What is 2 plus 2?", options: options)
        log("after host restart gui text: \(afterExit.debugDescription)")
        #expect(afterExit == guiAnswer)
        let restarted = try await gui.hostStatus()
        #expect(restarted.hostID != hostID)

        try await gui.endSession(chat, requestID: UUID())
        log("complete")
    }

    /// A client process the test can kill the way quitting the GUI or stopping
    /// the HTTP adapter would: `TUFFServingProbe` is built beside `TUFFServer`.
    private final class ProbeProcess: @unchecked Sendable {
        private let process = Process()
        private var lines: AsyncLineSequence<FileHandle.AsyncBytes>.AsyncIterator
        private(set) var output: [String] = []

        init(executable: URL, arguments: [String]) throws {
            let pipe = Pipe()
            process.executableURL = executable
            process.arguments = arguments
            process.standardOutput = pipe
            process.standardError = pipe
            lines = pipe.fileHandleForReading.bytes.lines.makeAsyncIterator()
            try process.run()
        }

        var pid: pid_t { process.processIdentifier }

        /// Reads until a line starting with `marker`; any `ERROR` line or the
        /// end of output before it is a failure.
        func wait(for marker: String) async throws {
            while let line = try await lines.next() {
                output.append(line)
                if line.hasPrefix(marker) { return }
                if line.hasPrefix("ERROR") { throw ProbeError.reported(line) }
            }
            throw ProbeError.endedBefore(marker, output)
        }

        func killAndReap() async -> Bool {
            kill(process.processIdentifier, SIGKILL)
            for _ in 0..<400 where process.isRunning { try? await Task.sleep(for: .milliseconds(25)) }
            return !process.isRunning && process.terminationReason == .uncaughtSignal
                && process.terminationStatus == SIGKILL
        }

        enum ProbeError: Error { case reported(String), endedBefore(String, [String]) }
    }

    /// An actual quit (SIGKILL) of a GUI-role process while an API request is
    /// generating must not interrupt the API work, and the same for a killed
    /// API-role process while a GUI stream runs. Both surfaces are real
    /// processes sharing one host and one Gemma E2B helper.
    @Test(.timeLimit(.minutes(20)))
    func killingOneSurfaceProcessNeverInterruptsTheOthersGeneration() async throws {
        guard let fixture = Self.fixture() else { return }
        setenv("TUFF_APPLICATION_SUPPORT_ROOT", fixture.support.path, 1)
        setenv("TUFF_LOCAL_BUILD", "1", 1)
        setenv("TUFF_SERVING_HELPER_EXECUTABLE", fixture.helper.path, 1)
        let bundle = "test.tuff.real-host"
        let profile = try ServingHostProfile(applicationSupportURL: fixture.support, bundleIdentifier: bundle)
        let models = fixture.support.appendingPathComponent("TUFF/Models", isDirectory: true)
        let gemma = models.appendingPathComponent("gemma4-e2b.gturbo", isDirectory: true)
        let probeExecutable = fixture.host.deletingLastPathComponent()
            .appendingPathComponent("TUFFServingProbe")
        try #require(FileManager.default.isExecutableFile(atPath: probeExecutable.path),
                     "build TUFFServingProbe beside TUFFServer")
        var options = AppRuntimeOptions()
        options.modelVerification = .fullSha256
        let prompt = "Count from 1 to 300 separated by spaces."
        func probeArguments(_ role: String) -> [String] {
            ["\(role)", "--support", fixture.support.path, "--bundle", bundle,
             "--host", fixture.host.path, "--model", "gemma4-e2b.gturbo",
             "--model-id", "gemma-4-e2b-it", "--prompt", prompt, "--tokens", "256"]
        }

        // The previous test's host was signalled in its `defer`; wait until it
        // no longer answers so this test always starts its own host.
        for _ in 0..<1200 {
            guard let stale = try? PrivateServingIPCConnection.connect(
                profile: profile, clientID: UUID()) else { break }
            stale.close()
            try await Task.sleep(for: .milliseconds(25))
        }
        let launched = PIDList()
        @Sendable func connect(_ name: String, _ role: ServingClientRole) throws -> ServingHostRuntime.Attachment {
            let attachment = try ServingHostRuntime.connectOrLaunch(
                profile: profile, clientID: try profile.clientID(for: name), role: role,
                hostExecutable: fixture.host)
            if let pid = attachment.launchedHostPID { launched.add(pid) }
            return attachment
        }
        defer { for pid in launched.all { kill(pid, SIGTERM) } }
        let gui = ServingHostInferenceClient(modelsRoot: models) { try connect("proc-gui", .gui).connection }
        try await gui.ensureLoaded(modelDirectory: gemma, maxContextTokens: 1024, options: options,
                                   forceLogitsHead: false) { _ in }
        let hostID = try await gui.hostStatus().hostID
        let apiIPC = PrivateServingIPCClient(connection: try connect("proc-api", .api).connection)
        defer { apiIPC.close() }
        let preparation = try ServingModelPreparation(
            modelID: "gemma-4-e2b-it", maxContextTokens: 1024,
            runtimeOptions: DecodeServiceInferenceClient.decodeRuntimeOptions(options),
            forceLogitsHead: false)
        for try await _ in apiIPC.events(for: .prepare(preparation)) {}
        let hostPID = try #require(launched.all.first)
        #expect(hostPID != 35872, "the isolated fixture must not attach to production")

        // GUI process quits mid-generation; the API stream keeps producing.
        let guiProcess = try ProbeProcess(executable: probeExecutable, arguments: probeArguments("gui"))
        try await guiProcess.wait(for: "FIRST_TOKEN")
        let apiTextResult = try await apiText(
            apiIPC, modelID: "gemma-4-e2b-it", session: UUID(), prompt: "What is 2 plus 2?",
            options: options, maxNewTokens: 48,
            onFirstToken: { kill(guiProcess.pid, SIGKILL) })
        #expect(!apiTextResult.isEmpty)
        #expect(await guiProcess.killAndReap(), "the GUI-role process must have died by SIGKILL, not finished")
        #expect(!guiProcess.output.contains { $0.hasPrefix("DONE") }, "the killed GUI work must not have completed first")

        // API process stops mid-generation; the GUI stream keeps producing.
        let apiProcess = try ProbeProcess(executable: probeExecutable, arguments: probeArguments("api"))
        try await apiProcess.wait(for: "FIRST_TOKEN")
        let guiAnswer = try await guiText(
            gui, directory: gemma, session: UUID(), prompt: "What is 2 plus 2?", options: options,
            maxNewTokens: 48,
            onFirstToken: { kill(apiProcess.pid, SIGKILL) })
        #expect(!guiAnswer.isEmpty)
        #expect(await apiProcess.killAndReap(), "the API-role process must have died by SIGKILL, not finished")
        #expect(!apiProcess.output.contains { $0.hasPrefix("DONE") }, "the killed API work must not have completed first")

        // Same host, same helper, and the killed processes' sessions are gone.
        let after = try await gui.hostStatus()
        #expect(after.hostID == hostID)
        #expect(helperChildren(of: hostPID) == 1)
        log("process-quit: api survived gui kill (\(apiTextResult.count) chars), gui survived api kill (\(guiAnswer.count) chars)")
    }
}
