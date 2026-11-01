import TUFFRuntimeManagement
import TUFFModelMetadata
import Darwin
import Foundation
import Synchronization
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFDecodeProtocol

public final class DecodeServiceInferenceClient: AppModelLifecycleClient,
    AppSessionInferenceClient, AppInferenceMemoryReporting,
    AppInferenceTranscriptReporting, AppInferenceEngineReporting, @unchecked Sendable {
    public var currentEngineIdentity: String? {
        connection.withLock { value in
            value.engineInstallation.map {
                $0.lease.manifest.packageID + ":" + $0.lease.manifest.version.directoryName + ":" + $0.identity
            }
        }
    }
    private struct Connection {
        var input: FileHandle?
        var responses: DecodeServiceResponseRouter?
        var loadedDirectory: URL?
        var launchLabel: String?
        var socketPath: String?
        var directProcess: Process?
        var engineInstallation: InferenceEngineInstallation?
    }

    private let connection = Mutex(Connection())
    private let inputWriteLock = NSLock()
    private let requiresHandshake: Bool
    private let serviceURLOverride: URL?
    private let inferenceMemory = Mutex<UInt64?>(nil)
    private let inferenceTowerMemory = Mutex<UInt64?>(nil)
    public let generationTranscriptMailbox = GenerationTranscriptMailbox()

    public var currentInferenceMemoryBytes: UInt64? {
        inferenceMemory.withLock { $0 }
    }

    public var currentInferenceTowerBytes: UInt64? {
        inferenceTowerMemory.withLock { $0 }
    }

    public init(serviceURL: URL? = nil, requiresHandshake: Bool = true) {
        self.serviceURLOverride = serviceURL
        self.requiresHandshake = requiresHandshake
        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
    }

    /// Connect an existing transport without launching a helper process.
    convenience init(input: FileHandle, output: FileHandle) {
        self.init(requiresHandshake: false)
        connection.withLock {
            $0.input = input
            $0.responses = DecodeServiceResponseRouter(output: output)
        }
    }

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                             options: AppRuntimeOptions, forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        do {
            try await load(modelDirectory: modelDirectory, maxContextTokens: maxContextTokens,
                options: options, forceLogitsHead: forceLogitsHead, acquireEngine: true, onState: onState)
        } catch {
            try Task.checkCancellation()
            guard serviceURLOverride == nil, requiresHandshake else { throw error }
            let failedIdentity = connection.withLock { $0.engineInstallation?.identity }
            closeConnection()
            guard let fallback = try? await InferenceEngineEnvironment.acquireKnownGoodModel(
                at: modelDirectory, excluding: failedIdentity) else { throw error }
            connection.withLock { $0.engineInstallation = fallback }
            do {
                try await load(modelDirectory: modelDirectory, maxContextTokens: maxContextTokens,
                    options: options, forceLogitsHead: forceLogitsHead, acquireEngine: false, onState: onState)
            } catch {
                closeConnection()
                throw error
            }
        }
    }

    private func load(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool, acquireEngine: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        onState(.loading(.validatingDirectory))
        // Explicit transports are test/developer injections. Ordinary app loads
        // always acquire an authenticated engine installed in the shared store.
        if acquireEngine && serviceURLOverride == nil && (requiresHandshake || currentHandles() == nil) {
            onState(.loading(.preparingEngine))
            let installed = try await InferenceEngineEnvironment.acquireModel(at: modelDirectory)
            if Task.isCancelled {
                await installed.release()
                throw CancellationError()
            }
            let previous = connection.withLock { $0.engineInstallation }
            if previous?.executableURL == installed.executableURL, currentHandles() != nil {
                await installed.release()
            } else {
                if currentHandles() != nil { await unload() }
                closeConnection()
                connection.withLock { $0.engineInstallation = installed }
            }
        }
        let handles = try await Task.detached(priority: .userInitiated) { [self] in
            try ensureProcess()
        }.value
        if requiresHandshake {
            let architecture = try ManifestReader.resolveArchitecture(directoryURL: modelDirectory)
            let hello = DecodeEngineHello(architectureID: architecture.variant.rawValue,
                requiredFeatures: ["preserve-history"] + (options.retainedConversationBytes == nil ? [] : ["bounded-retention"]))
            try handles.input.write(contentsOf: DecodeFrameCodec.encode(DecodeServiceCommand.hello(hello)))
            let reply = try await handles.responses.next(matching: hello.requestID)
            guard reply.kind == .hello, let capabilities = reply.capabilities,
                  capabilities.accepts(hello) else {
                throw AppInferenceError.modelLoadFailed(reply.error ?? "inference engine handshake failed")
            }
        }
        let request = DecodeLoadRequest(
            modelPath: modelDirectory.path, maxContextTokens: maxContextTokens,
            runtimeOptions: Self.decodeRuntimeOptions(options),
            forceLogitsHead: forceLogitsHead)
        try handles.input.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.load(request)))
        let event = try await handles.responses.next(matching: request.requestID)
        switch event.kind {
        case .ready:
            break
        case .failed:
            throw AppInferenceError.modelLoadFailed(
                event.error ?? "decode service load failed")
        default:
            throw AppInferenceError.modelLoadFailed(
                "decode service returned \(event.kind.rawValue) for a load request")
        }
        inferenceMemory.withLock { $0 = event.currentMemoryBytes }
        inferenceTowerMemory.withLock { $0 = event.visionTowerMappedBytes }
        connection.withLock { $0.loadedDirectory = modelDirectory.standardizedFileURL }
        onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
    }

    public func unload() async {
        guard let handles = currentHandles() else { return }
        let requestID = UUID()
        try? handles.input.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceCommand.unload(requestID)))
        guard let event = try? await handles.responses.next(matching: requestID),
              event.kind == .unloaded else { return }
        connection.withLock { $0.loadedDirectory = nil }
        inferenceMemory.withLock { $0 = nil }
        inferenceTowerMemory.withLock { $0 = nil }
        if connection.withLock({ $0.engineInstallation != nil }) { closeConnection() }
    }

    public func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        makeGenerationStream(request, sessionID: nil, requestID: UUID())
    }

    public func generate(_ request: AppGenerationRequest, sessionID: UUID,
                         requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        makeGenerationStream(request, sessionID: sessionID, requestID: requestID)
    }

    /// Starts one raw helper generation for a serving coordinator. Unlike the
    /// app-facing adapter this preserves the decode-service event envelope,
    /// including terminal kind, diagnostics, memory and token sequencing.
    public func startRaw(_ request: DecodeGenerationRequest, sessionID: UUID)
        -> AsyncThrowingStream<DecodeServiceEvent, Error> {
        AsyncThrowingStream { continuation in
            Task.detached(priority: .userInitiated) { [self] in
                do {
                    guard let handles = currentHandles() else {
                        throw AppInferenceError.modelNotLoaded
                    }
                    try send(.generateSession(sessionID: sessionID, request: request),
                             to: handles.input)
                    while true {
                        let event = try await handles.responses.next(matching: request.generationID)
                        if let bytes = event.currentMemoryBytes { inferenceMemory.withLock { $0 = bytes } }
                        if let bytes = event.visionTowerMappedBytes { inferenceTowerMemory.withLock { $0 = bytes } }
                        continuation.yield(event)
                        if event.kind == .finished || event.kind == .cancelled || event.kind == .failed {
                            continuation.finish()
                            return
                        }
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { [weak self] termination in
                if case .cancelled = termination {
                    // Keep draining the helper's terminal event; ending the
                    // reader early would leave the request/session occupied.
                    self?.cancel(requestID: request.generationID)
                }
            }
        }
    }

    private func makeGenerationStream(_ request: AppGenerationRequest, sessionID: UUID?,
                                      requestID: UUID)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) { [self] in
                do {
                    try request.validate()
                    guard let handles = currentHandles() else {
                        throw AppInferenceError.modelNotLoaded
                    }
                    let generationID = requestID
                    if sessionID == nil { generationTranscriptMailbox.reset() }
                    let command = DecodeGenerationTranslation.decodeRequest(
                        request, generationID: generationID)
                    let serviceCommand: DecodeServiceCommand = sessionID.map {
                        .generateSession(sessionID: $0, request: command)
                    } ?? .generate(command)
                    try send(serviceCommand, to: handles.input)

                    var translator = DecodeGenerationEventTranslator(
                        options: request.runtimeOptions, mailbox: generationTranscriptMailbox)
                    while true {
                        let event = try await handles.responses.next(matching: generationID)
                        // Only when the event carries a figure: an event
                        // without one says nothing about memory, and clearing
                        // the last reading made the display flicker to empty.
                        if let bytes = event.currentMemoryBytes {
                            inferenceMemory.withLock { $0 = bytes }
                        }
                        if let tower = event.visionTowerMappedBytes {
                            inferenceTowerMemory.withLock { $0 = tower }
                        }
                        guard event.generationID == generationID else { continue }
                        if event.kind == .finished,
                           let installation = connection.withLock({ $0.engineInstallation }) {
                            await InferenceEngineEnvironment.markKnownGood(installation, for: request.modelDirectory)
                        }
                        let output = try translator.consume(event)
                        for translated in output.events { continuation.yield(translated) }
                        switch output.terminal {
                        case nil: continue
                        case .finished, .cancelled: continuation.finish()
                        case .failed(let error): continuation.finish(throwing: error)
                        }
                        return
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { [weak self] termination in
                // A finished stream already has a matching terminal event.
                // Sending a global cancel here could land after the broker has
                // admitted the next request and cancel that request instead.
                if case .cancelled = termination {
                    if sessionID != nil { self?.cancel(requestID: requestID) }
                    else { task.cancel(); self?.cancel() }
                }
            }
        }
    }

    public func cancel(requestID: UUID) {
        guard let input = currentHandles()?.input else { return }
        try? send(.cancelRequest(requestID), to: input)
    }

    public func cancelWarmup() {
        guard let input = currentHandles()?.input else { return }
        try? send(.cancelWarmup, to: input)
    }

    public func endSession(_ sessionID: UUID, requestID: UUID) async throws {
        guard let handles = currentHandles() else { throw AppInferenceError.modelNotLoaded }
        try send(.endSession(sessionID: sessionID, requestID: requestID), to: handles.input)
        let event = try await handles.responses.next(matching: requestID)
        switch event.kind {
        case .unloaded: return
        case .failed:
            throw AppInferenceError.unknown(event.error ?? "The chat session could not be closed.")
        default:
            throw AppInferenceError.unknown("The decode service returned an invalid session close event.")
        }
    }

    public func cancel() {
        guard let input = currentHandles()?.input else { return }
        try? send(.cancel, to: input)
    }

    private func send(_ command: DecodeServiceCommand, to input: FileHandle) throws {
        let frame = try DecodeFrameCodec.encode(command)
        inputWriteLock.lock()
        defer { inputWriteLock.unlock() }
        try input.write(contentsOf: frame)
    }

    deinit { closeConnection() }

    private func closeConnection() {
        let state = connection.withLock { value -> Connection in
            defer { value = Connection() }
            return value
        }
        if let input = state.input {
            try? input.write(contentsOf: DecodeFrameCodec.encode(
                DecodeServiceCommand.shutdown))
            try? input.close()
        }
        if let label = state.launchLabel { Self.removeLaunchJob(label: label) }
        state.directProcess?.terminate()
        if let socketPath = state.socketPath { unlink(socketPath) }
    }

    private func ensureProcess() throws
        -> (input: FileHandle, responses: DecodeServiceResponseRouter) {
        if let handles = currentHandles() { return handles }
        return try launchIndependentService()
    }

    private func launchIndependentService() throws
        -> (input: FileHandle, responses: DecodeServiceResponseRouter) {
        guard let serviceURL = serviceURLOverride ?? connection.withLock({ $0.engineInstallation?.executableURL }) else {
            throw InferenceEngineEnvironmentError.missingCatalog
        }
        guard FileManager.default.isExecutableFile(atPath: serviceURL.path) else {
            throw AppInferenceError.modelLoadFailed(
                "decode service executable is missing at \(serviceURL.path); run swift build -c release before launching the app")
        }
        let identifier = "\(getuid()).\(getpid()).\(UUID().uuidString.lowercased())"
        let label = "com.tuff.decode.\(identifier)"
        let socketPath = "/private/tmp/tuff-decode-\(identifier).sock"

        // A kickstart failure is diagnostic, not fatal: the socket is
        // authoritative, and the direct-spawn fallback below may still work.
        // Carry the detail so the final error can name what actually failed.
        var kickstartError: String?
        if let bootstrapped = try? bootstrapViaLaunchd(label: label, socketPath: socketPath, serviceURL: serviceURL) {
            kickstartError = bootstrapped
            if let handles = try? connectSocketWithRetry(socketPath: socketPath, attempts: 100) {
                connection.withLock {
                    $0.input = handles.input
                    $0.responses = handles.responses
                    $0.launchLabel = label
                    $0.socketPath = socketPath
                }
                return handles
            }
        }

        // Some sessions (e.g. terminal multiplexers that spawn the app
        // outside a normal Aqua-attached process) have launchd accept the
        // "gui/<uid>" bootstrap but never actually spawn the RunAtLoad job.
        // Cancel that registration and fall back to a plain child process on
        // the same socket path so the app still works there.
        Self.removeLaunchJob(label: label)
        let process: Process
        do {
            process = try launchDirectProcess(socketPath: socketPath, label: label, serviceURL: serviceURL)
        } catch {
            throw AppInferenceError.modelLoadFailed(
                "decode service could not be started: \(error)")
        }
        do {
            let handles = try connectSocketWithRetry(socketPath: socketPath, attempts: 200)
            connection.withLock {
                $0.input = handles.input
                $0.responses = handles.responses
                $0.directProcess = process
                $0.socketPath = socketPath
            }
            return handles
        } catch {
            process.terminate()
            throw AppInferenceError.modelLoadFailed(
                Self.socketFailureMessage(
                    socketError: String(describing: error),
                    kickstartError: kickstartError))
        }
    }

    /// Register and demand the launchd job. Returns the `launchctl kickstart`
    /// failure detail, if any — bootstrap succeeding while kickstart fails is a
    /// diagnostic, not a fatal error, because the socket is authoritative.
    @discardableResult
    private func bootstrapViaLaunchd(label: String, socketPath: String, serviceURL: URL) throws -> String? {
        let propertyListURL = URL(
            fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("\(label).plist")
        let propertyList: [String: Any] = [
            "Label": label,
            "ProgramArguments": [
                serviceURL.path,
                "--socket", socketPath,
                "--launch-label", label,
            ] + engineLeaseArguments,

            "RunAtLoad": true,
            "KeepAlive": false,
            "ProcessType": "Interactive",
            "EnvironmentVariables": ProcessInfo.processInfo.environment.filter {
                ["TUFF_APPLICATION_SUPPORT_ROOT", "TUFF_ENGINE_CATALOG_PATH"].contains($0.key)
            },
        ]
        let propertyListData = try PropertyListSerialization.data(
            fromPropertyList: propertyList, format: .xml, options: 0)
        try propertyListData.write(to: propertyListURL, options: .atomic)
        defer { try? FileManager.default.removeItem(at: propertyListURL) }

        let launcher = Process()
        let errors = Pipe()
        launcher.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        launcher.arguments = [
            "bootstrap", "gui/\(getuid())", propertyListURL.path,
        ]
        launcher.standardOutput = FileHandle.nullDevice
        launcher.standardError = errors
        try launcher.run()
        launcher.waitUntilExit()
        guard launcher.terminationStatus == 0 else {
            let data = try? errors.fileHandleForReading.readToEnd()
            let detail = data.flatMap { String(data: $0, encoding: .utf8) }?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let message = detail.flatMap { $0.isEmpty ? nil : $0 }
                ?? "launchd could not start the decode service"
            throw AppInferenceError.modelLoadFailed(message)
        }

        // Demand the job without restarting a RunAtLoad winner; the socket is
        // authoritative. `bootstrap` alone does not always leave the job
        // running, and the failure then looked like a socket timeout with no
        // indication that launchd was the cause. Returns the failure detail
        // rather than throwing so the direct-process fallback can still run.
        var kickstartError: String?
        do {
            let starter = Process()
            let starterErrors = Pipe()
            starter.executableURL = URL(fileURLWithPath: "/bin/launchctl")
            starter.arguments = Self.kickstartArguments(uid: getuid(), label: label)
            starter.standardOutput = FileHandle.nullDevice
            starter.standardError = starterErrors
            try starter.run()
            starter.waitUntilExit()
            if starter.terminationStatus != 0 {
                let data = try? starterErrors.fileHandleForReading.readToEnd()
                let detail = data.flatMap { String(data: $0, encoding: .utf8) }?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                kickstartError = detail.flatMap { $0.isEmpty ? nil : $0 }
                    ?? "exit status \(starter.terminationStatus)"
            }
        } catch {
            kickstartError = String(describing: error)
        }
        return kickstartError
    }

    private func launchDirectProcess(socketPath: String, label: String, serviceURL: URL) throws -> Process {
        let process = Process()
        process.executableURL = serviceURL
        process.arguments = ["--socket", socketPath, "--launch-label", label] + engineLeaseArguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        return process
    }

    private func connectSocketWithRetry(socketPath: String, attempts: Int) throws
        -> (input: FileHandle, responses: DecodeServiceResponseRouter) {
        var lastError: Error?
        for _ in 0..<attempts {
            do {
                let handles = try DecodeUnixSocket.connect(path: socketPath)
                let responses = DecodeServiceResponseRouter(output: handles.output)
                return (handles.input, responses)
            } catch {
                lastError = error
                usleep(10_000)
            }
        }
        throw lastError ?? AppInferenceError.modelLoadFailed("unknown error")
    }

    /// `kickstart` demands the job; it deliberately omits `-k`, which would
    /// restart a job that a RunAtLoad bootstrap already won. The socket is what
    /// decides readiness, so restarting a healthy service would only lose it.
    static func kickstartArguments(uid: uid_t, label: String) -> [String] {
        ["kickstart", "gui/\(uid)/\(label)"]
    }

    /// The socket timeout stays the primary diagnostic, because that is what the
    /// caller actually observed; a launchctl failure is appended when there was
    /// one, since it usually explains the timeout.
    static func socketFailureMessage(socketError: String?, kickstartError: String?)
        -> String {
        let kickstartDetail = kickstartError.map {
            "; launchctl kickstart failed: \($0)"
        } ?? ""
        return "decode service socket did not become ready: \(socketError ?? "unknown error")\(kickstartDetail)"
    }

    private func currentHandles()
        -> (input: FileHandle, responses: DecodeServiceResponseRouter)? {
        connection.withLock { state in
            guard let input = state.input, let responses = state.responses else {
                return nil
            }
            return (input, responses)
        }
    }

    static func diagnostics(_ event: DecodeServiceEvent,
                                    options: AppRuntimeOptions) -> AppDiagnostics {
        let stop = AppStopReason(rawValue: event.stopReason ?? "")
            ?? (event.kind == .cancelled
                ? .cancelled
                : event.kind == .failed ? .failed : .maxTokens)
        var diagnostics = AppDiagnostics(
            generatedTokens: event.tokenCount,
            stopReason: stop,
            promptTokenCount: event.promptTokenCount,
            prefillSeconds: event.prefillSeconds,
            timeToFirstTokenSeconds: event.timeToFirstTokenSeconds,
            decodeSeconds: event.decodeSeconds,
            tokensPerSecond: event.tokensPerSecond,
            peakMemoryBytes: event.peakMemoryBytes,
            visionTowerMappedBytes: event.visionTowerMappedBytes,
            runtimeOptions: options,
            prefill: prefillDiagnostics(event.prefill, options: options),
            runner: event.runner.map(runnerDiagnostics),
            droppedTurns: event.droppedTurns ?? 0)
        diagnostics.cachedPromptTokens = event.cachedPromptTokens
        diagnostics.conversationCacheSource = event.conversationCacheSource
        diagnostics.retainedConversations = event.retainedConversations
        diagnostics.retainedConversationBytes = event.retainedConversationBytes
        return diagnostics
    }

    public static func decodeToolRound(_ round: AppToolRound) -> DecodeToolRound {
        DecodeToolRound(
            thinking: round.thinking, content: round.content,
            calls: round.calls.map(\.historical),
            results: round.results.map {
                DecodeToolResult(callID: $0.callID, name: $0.name, content: $0.modelText)
            })
    }

    private static func prefillDiagnostics(
        _ value: DecodePrefillDiagnostics?, options: AppRuntimeOptions
    ) -> PrefillExecutionDiagnostics? {
        guard let value,
              let executedMode = PrefillExecutedMode(rawValue: value.executedMode),
              let completeness = PrefillChunkCompleteness(
                rawValue: value.chunkCompleteness) else { return nil }
        let kvStorage = value.kvStorageMode.flatMap(PrefillKVStorageMode.init(rawValue:))
        return PrefillExecutionDiagnostics(
            config: options.prefillConfig,
            executedMode: executedMode,
            kvStorageMode: kvStorage,
            chunkCompleteness: completeness,
            unsupportedReason: value.unsupportedReason)
    }

    private static func runnerDiagnostics(_ value: DecodeRunnerDiagnostics)
        -> AppRunnerDiagnostics {
        AppRunnerDiagnostics(
            cb1MillisecondsPerToken: value.cb1MillisecondsPerToken,
            ioMillisecondsPerToken: value.ioMillisecondsPerToken,
            cb2MillisecondsPerToken: value.cb2MillisecondsPerToken,
            headMillisecondsPerToken: value.headMillisecondsPerToken,
            rdadviseMillisecondsPerToken: value.rdadviseMillisecondsPerToken,
            rdadviseCallsPerToken: value.rdadviseCallsPerToken,
            rdadviseMegabytesPerToken: value.rdadviseMegabytesPerToken,
            rdadviseSkippedPerToken: value.rdadviseSkippedPerToken,
            rdadviseFailures: value.rdadviseFailures,
            expertReads: value.expertReads,
            exposedPrefetchWaitMillisecondsPerToken: value.exposedPrefetchWaitMillisecondsPerToken)
    }

    public static func decodeRuntimeOptions(_ options: AppRuntimeOptions)
        -> DecodeRuntimeOptions {
        let layers: [Int]?
        switch options.denseResidencyPolicy {
        case .automatic: layers = nil
        case .resident: layers = nil
        case .bounded(let resident): layers = resident.sorted()
        }
        return DecodeRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: options.expertCachePolicy.rawValue,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: options.rdadvisePolicy.rawValue,
            modelVerification: options.modelVerification.rawValue,
            visionResidencyPolicy: options.visionResidencyPolicy.rawValue,
            denseResidentLayers: layers,
            denseWeightBudgetBytes: {
                if case .automatic(let budget) = options.denseResidencyPolicy { return budget }
                return nil
            }(), retainedConversationBytes: options.retainedConversationBytes)
    }

    private static func removeLaunchJob(label: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())/\(label)"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
        process.waitUntilExit()
    }

    private var engineLeaseArguments: [String] {
        connection.withLock { value in
            value.engineInstallation.map { ["--engine-lock", $0.lockFileURL.path] } ?? []
        }
    }
}
