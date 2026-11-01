import Foundation
import Synchronization
import TUFFAppCore
import TUFFDecodeProtocol
import TUFFModelCatalog
import TUFFServerCore

/// The app's inference client: a thin attachment to the per-profile serving
/// host. Model weights, helper processes and GPU scheduling live in the host,
/// so quitting this app cannot kill another client's work and other clients
/// cannot stop this app's.
///
/// Every operation goes through the host's model/session/request namespaces.
/// The host may evict an idle model under memory pressure or restart; `generate`
/// therefore re-asserts the model's preparation (an immediate no-op while it is
/// resident) instead of trusting local state.
public final class ServingHostInferenceClient: AppModelLifecycleClient, AppPooledInferenceClient,
    AppSessionInferenceClient, AppInferenceMemoryReporting, AppInferenceTranscriptReporting,
    AppInferenceEngineReporting, @unchecked Sendable {
    public typealias Connector = @Sendable () throws -> PrivateServingIPCConnection

    private struct KnownModel {
        var preparation: ServingModelPreparation
        var directory: URL
        var options: AppRuntimeOptions
    }

    private struct State {
        var ipc: PrivateServingIPCClient?
        var connecting: (id: UUID, task: Task<PrivateServingIPCClient, any Error>)?
        var models: [String: KnownModel] = [:]
        var currentModelID: String?
        var requestSessions: [UUID: UUID] = [:]
        var cancelledRequests: Set<UUID> = []
        /// Requests whose `.start` frame has been written. The host handles one
        /// connection's commands in order, so a cancel written after this point
        /// always reaches a started request; before it, the cancel is a flag
        /// the request checks before sending `.start` and right after.
        var startedRequests: Set<UUID> = []
        var legacyRequests: Set<UUID> = []
        var memoryBytes: UInt64?
        var towerBytes: UInt64?
    }

    private let connector: Connector
    private let modelsRoot: URL
    private let state = Mutex(State())
    private let gate = KeyedGate()
    public let generationTranscriptMailbox = GenerationTranscriptMailbox()

    /// - Parameter modelsRoot: The host's model folder. Only models inside it
    ///   can be named; clients send identifiers, never arbitrary paths.
    public init(modelsRoot: URL, connector: @escaping Connector) {
        self.modelsRoot = modelsRoot.standardizedFileURL
        self.connector = connector
        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
    }

    /// Attaches to this process's profile host, launching the dedicated host
    /// process when it is not running.
    public convenience init(profile: ServingHostProfile, bundleIdentifier: String,
                            clientName: String = "gui", role: ServingClientRole = .gui,
                            modelsRoot: URL) {
        let clientID = UUID()
        self.init(modelsRoot: modelsRoot) {
            try ServingHostRuntime.connectOrLaunch(
                profile: profile, clientID: clientID, role: role).connection
        }
    }

    public var currentEngineIdentity: String? {
        state.withLock { $0.ipc?.host.buildIdentity }
    }
    public var currentInferenceMemoryBytes: UInt64? { state.withLock { $0.memoryBytes } }
    public var currentInferenceTowerBytes: UInt64? { state.withLock { $0.towerBytes } }

    // MARK: - Connection

    /// The live connection, reconnecting (and launching the host) after a loss.
    private func attach() async throws -> PrivateServingIPCClient {
        let connector = connector
        let pending = state.withLock { current -> (UUID, Task<PrivateServingIPCClient, any Error>) in
            if let existing = current.ipc, !existing.isClosed {
                return (UUID(), Task { existing })
            }
            if let connecting = current.connecting { return (connecting.id, connecting.task) }
            let id = UUID()
            let connecting = Task.detached(priority: .userInitiated) {
                PrivateServingIPCClient(connection: try connector())
            }
            current.connecting = (id, connecting)
            return (id, connecting)
        }
        do {
            let client = try await pending.1.value
            state.withLock { current in
                if current.connecting?.id == pending.0 {
                    current.ipc = client
                    current.connecting = nil
                }
            }
            return client
        } catch {
            state.withLock { current in
                if current.connecting?.id == pending.0 { current.connecting = nil }
            }
            throw AppInferenceError.modelLoadFailed(
                "The TUFF inference host could not be reached: \(error)")
        }
    }

    // MARK: - Model identity

    private func modelID(for directory: URL) throws -> String {
        let standard = directory.standardizedFileURL
        guard standard.deletingLastPathComponent().path == modelsRoot.path else {
            throw AppInferenceError.modelLoadFailed(
                "This model is outside TUFF's managed models folder, so the shared inference host cannot load it.")
        }
        let name = standard.lastPathComponent
        let installed = ServerInstalledModels(modelsRoot: modelsRoot)
        if let descriptor = TUFFModelCatalog.all.first(where: { $0.installDirectoryName == name }) {
            return descriptor.apiModelID
        }
        if let custom = installed.customModels().first(where: {
            $0.descriptor.installDirectoryName == name
        }) { return custom.descriptor.apiModelID }
        throw AppInferenceError.modelNotFound(standard.path)
    }

    // MARK: - Lifecycle

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                             options: AppRuntimeOptions, forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        onState(.loading(.validatingDirectory))
        let modelID = try modelID(for: modelDirectory)
        let preparation: ServingModelPreparation
        do {
            preparation = try ServingModelPreparation(
                modelID: modelID, maxContextTokens: maxContextTokens,
                runtimeOptions: DecodeServiceInferenceClient.decodeRuntimeOptions(options),
                forceLogitsHead: forceLogitsHead)
        } catch { throw AppInferenceError.invalidRequest("\(error)") }
        let directory = modelDirectory.standardizedFileURL
        let seconds = try await prepare(preparation, onState: { phase in
            onState(.loading(phase))
        })
        state.withLock {
            $0.models[modelID] = KnownModel(preparation: preparation, directory: directory,
                                            options: options)
            $0.currentModelID = modelID
        }
        onState(.ready(modelDirectory: modelDirectory, loadSeconds: seconds))
    }

    /// Asks the host for the model, waiting for any load to finish. Returns
    /// the host-reported load seconds.
    @discardableResult
    private func prepare(_ preparation: ServingModelPreparation,
                         onState: (@Sendable (AppModelLoadPhase) -> Void)? = nil,
                         replaceIncompatible: Bool = true) async throws -> Double {
        try await gate.run(key: preparation.modelID) { [self] in
            func attempt(_ preparation: ServingModelPreparation) async throws -> Double {
                let client = try await attach()
                var seconds = 0.0
                do {
                    for try await event in client.events(for: .prepare(preparation)) {
                        guard case .lifecycle(_, _, let loadState) = event else { continue }
                        switch loadState {
                        case .loading(let phase): onState?(Self.appPhase(phase))
                        case .ready(let value): seconds = value
                        case .cancelled: throw CancellationError()
                        default: break
                        }
                    }
                } catch PrivateServingIPCClientError.remote(let message) {
                    throw AppInferenceError.modelLoadFailed(message)
                } catch let error as PrivateServingIPCClientError {
                    throw AppInferenceError.modelLoadFailed("\(error)")
                }
                try Task.checkCancellation()
                return seconds
            }
            do {
                return try await attempt(preparation)
            } catch AppInferenceError.modelLoadFailed(let message)
                        where replaceIncompatible && message.contains("Unload the model before changing") {
                // The pool holds this model with another configuration (a
                // reload with new settings). Replace it once.
                try await unloadFromHost(modelID: preparation.modelID)
                return try await attempt(ServingModelPreparation(
                    modelID: preparation.modelID,
                    maxContextTokens: preparation.maxContextTokens,
                    runtimeOptions: preparation.runtimeOptions,
                    forceLogitsHead: preparation.forceLogitsHead))
            }
        }
    }

    private func unloadFromHost(modelID: String) async throws {
        let client = try await attach()
        do {
            for try await _ in client.events(for: .unloadModel(
                operationID: UUID(), modelID: modelID)) {}
            try Task.checkCancellation()
        } catch PrivateServingIPCClientError.remote(let message) {
            throw AppInferenceError.unknown(message)
        }
    }

    public func unload() async {
        guard let modelID = state.withLock({ $0.currentModelID }) else { return }
        await unload(modelID: modelID)
    }

    public func unload(modelDirectory: URL) async {
        guard let modelID = try? modelID(for: modelDirectory) else { return }
        await unload(modelID: modelID)
    }

    private func unload(modelID: String) async {
        // A model with active requests is refused by the host and stays.
        do { try await gate.run(key: modelID) { [self] in try await unloadFromHost(modelID: modelID) } }
        catch { return }
        state.withLock { current in
            current.models.removeValue(forKey: modelID)
            if current.currentModelID == modelID { current.currentModelID = nil }
            if current.models.isEmpty { current.memoryBytes = nil; current.towerBytes = nil }
        }
    }

    public func residentRuntime(forModelDirectory directory: URL)
        -> (maxContextTokens: Int, options: AppRuntimeOptions, forceLogitsHead: Bool)? {
        let standard = directory.standardizedFileURL
        return state.withLock { current in
            current.models.values.first { $0.directory == standard }.map {
                ($0.preparation.maxContextTokens, $0.options, $0.preparation.forceLogitsHead)
            }
        }
    }

    /// Live state of the host: its models, work and attached adapters.
    public func hostStatus() async throws -> ServingHostStatus {
        let client = try await attach()
        for try await event in client.events(for: .hostStatus(operationID: UUID())) {
            if case .hostStatus(_, let status) = event { return status }
        }
        throw AppInferenceError.unknown("The inference host did not answer its status request.")
    }

    // MARK: - Generation

    public func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        // A request outside any chat gets a private session that is closed
        // as soon as it ends.
        let sessionID = UUID(), requestID = UUID()
        _ = state.withLock { $0.legacyRequests.insert(requestID) }
        let inner = generate(request, sessionID: sessionID, requestID: requestID)
        return AsyncThrowingStream(bufferingPolicy: .bufferingOldest(PrivateServingIPCClient.streamCapacity)) { continuation in
            let task = Task {
                var overflowed = false
                do {
                    for try await event in inner {
                        if !overflowed, case .dropped = continuation.yield(event) {
                            overflowed = true
                            cancel(requestID: requestID)
                            continuation.finish(throwing: PrivateServingIPCClientError.slowConsumer)
                        }
                    }
                }
                catch { continuation.finish(throwing: error); await closeLegacy(sessionID, requestID); return }
                continuation.finish()
                await closeLegacy(sessionID, requestID)
            }
            continuation.onTermination = { [weak self] termination in
                if case .cancelled = termination { self?.cancel(requestID: requestID); task.cancel() }
            }
        }
    }

    private func closeLegacy(_ sessionID: UUID, _ requestID: UUID) async {
        _ = state.withLock { $0.legacyRequests.remove(requestID) }
        try? await endSession(sessionID, requestID: requestID)
    }

    public func generate(_ request: AppGenerationRequest, sessionID: UUID,
                         requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingOldest(PrivateServingIPCClient.streamCapacity)) { continuation in
            state.withLock { $0.requestSessions[requestID] = sessionID }
            let task = Task.detached(priority: .userInitiated) { [self] in
                defer {
                    state.withLock {
                        $0.requestSessions.removeValue(forKey: requestID)
                        $0.cancelledRequests.remove(requestID)
                        $0.startedRequests.remove(requestID)
                    }
                }
                do {
                    try request.validate()
                    let modelID = try modelID(for: request.modelDirectory)
                    guard let known = state.withLock({ $0.models[modelID] }) else {
                        throw AppInferenceError.modelNotLoaded
                    }
                    if state.withLock({ $0.cancelledRequests.contains(requestID) }) {
                        continuation.finish()
                        return
                    }
                    let status = try await hostStatus()
                    let loaded = status.loadedModels.first { $0.modelID == modelID }
                    let preparation = try loaded.map {
                        try ServingModelPreparation(modelID: modelID,
                            maxContextTokens: $0.maxContextTokens, runtimeOptions: $0.runtimeOptions,
                            forceLogitsHead: $0.forceLogitsHead)
                    } ?? known.preparation
                    try await prepare(preparation, replaceIncompatible: false)
                    try Task.checkCancellation()
                    let client = try await attach()
                    if state.withLock({ $0.cancelledRequests.contains(requestID) }) {
                        continuation.finish()
                        return
                    }
                    let payload = DecodeGenerationTranslation.decodeRequest(
                        request, generationID: requestID)
                    let routed: ServingGenerationRequest
                    do {
                        routed = try ServingGenerationRequest(
                            modelID: modelID, sessionID: sessionID, requestID: requestID,
                            payload: payload)
                    } catch { throw AppInferenceError.invalidRequest("\(error)") }
                    let legacy = state.withLock { $0.legacyRequests.contains(requestID) }
                    var translator = DecodeGenerationEventTranslator(
                        options: request.runtimeOptions, mailbox: legacy ? generationTranscriptMailbox : nil)
                    var finished = false
                    var overflowed = false
                    do {
                        let events = client.events(for: .start(routed))
                        // The start frame is on the wire now. A cancel that
                        // arrived while it was being sent only set the flag;
                        // send it here, after the start, so it cannot be
                        // dropped as "unknown request".
                        let cancelAlreadyRequested = state.withLock { current -> Bool in
                            _ = current.startedRequests.insert(requestID)
                            return current.cancelledRequests.contains(requestID)
                        }
                        if cancelAlreadyRequested {
                            try? client.send(.cancel(sessionID: sessionID, requestID: requestID))
                        }
                        for try await event in events {
                            switch event {
                            case .generation(_, let decodeEvent):
                                state.withLock {
                                    if let bytes = decodeEvent.currentMemoryBytes { $0.memoryBytes = bytes }
                                    if let bytes = decodeEvent.visionTowerMappedBytes { $0.towerBytes = bytes }
                                }
                                let output = try translator.consume(decodeEvent)
                                if !overflowed {
                                    for translated in output.events {
                                        if case .dropped = continuation.yield(translated) {
                                            overflowed = true
                                            cancel(requestID: requestID)
                                            continuation.finish(throwing: PrivateServingIPCClientError.slowConsumer)
                                            break
                                        }
                                    }
                                }
                                switch output.terminal {
                                case nil: continue
                                case .finished, .cancelled: continuation.finish()
                                case .failed(let error): continuation.finish(throwing: error)
                                }
                                finished = true
                            case .failed(_, let message):
                                let error = AppInferenceError.unknown(message)
                                continuation.yield(.failed(error, partial: nil))
                                continuation.finish(throwing: error)
                                finished = true
                            default: continue
                            }
                            if finished { return }
                        }
                    } catch PrivateServingIPCClientError.remote(let message),
                            PrivateServingIPCClientError.hostFault(let message) {
                        let error = AppInferenceError.unknown(message)
                        continuation.yield(.failed(error, partial: nil))
                        continuation.finish(throwing: error)
                        return
                    }
                    if !finished {
                        let error = AppInferenceError.unknown(
                            "The TUFF inference host connection ended before the generation finished.")
                        continuation.yield(.failed(error, partial: nil))
                        continuation.finish(throwing: error)
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { [weak self] termination in
                if case .cancelled = termination {
                    self?.cancel(requestID: requestID)
                    task.cancel()
                }
            }
        }
    }

    public func cancel(requestID: UUID) {
        let target = state.withLock { current in
            guard let session = current.requestSessions[requestID] else {
                return (nil as UUID?, nil as PrivateServingIPCClient?)
            }
            current.cancelledRequests.insert(requestID)
            // Before `.start` is written the flag alone is enough: the request
            // checks it and never starts, or sends this cancel itself right
            // after its start.
            guard current.startedRequests.contains(requestID) else {
                return (nil as UUID?, nil as PrivateServingIPCClient?)
            }
            return (session, current.ipc)
        }
        guard let sessionID = target.0, let client = target.1 else { return }
        try? client.send(.cancel(sessionID: sessionID, requestID: requestID))
    }

    /// Cancels requests that were not issued through a scoped chat session.
    public func cancel() {
        let targets = state.withLock { Array($0.legacyRequests) }
        for requestID in targets { cancel(requestID: requestID) }
    }

    /// Returns only after the host has released the session's state. A failed
    /// cleanup throws and leaves the session open on the host for a retry.
    public func endSession(_ sessionID: UUID, requestID: UUID) async throws {
        let client = try await attach()
        do {
            for try await _ in client.events(for: .endSession(sessionID: sessionID)) {}
            try Task.checkCancellation()
        } catch PrivateServingIPCClientError.remote(let message) {
            throw AppInferenceError.unknown(message)
        }
    }

    private static func appPhase(_ phase: ServingLoadPhase) -> AppModelLoadPhase {
        switch phase {
        case .preparingEngine: .preparingEngine
        case .validatingDirectory: .validatingDirectory
        case .tokenizer: .tokenizer
        case .verifyingWeights: .verifyingWeights
        case .preparingRunner: .preparingRunner
        case .mappingImageTower: .mappingImageTower
        }
    }
}

/// FIFO mutual exclusion per key. Model preparations and unloads for one model
/// never overlap, while different models proceed independently.
private actor KeyedGate {
    private var held: Set<String> = []
    private var waiting: [String: [CheckedContinuation<Void, Never>]] = [:]

    func run<T: Sendable>(key: String, _ body: @Sendable () async throws -> T) async throws -> T {
        await acquire(key)
        defer { release(key) }
        return try await body()
    }

    private func acquire(_ key: String) async {
        if held.insert(key).inserted { return }
        await withCheckedContinuation { waiting[key, default: []].append($0) }
    }

    private func release(_ key: String) {
        if var queue = waiting[key], !queue.isEmpty {
            let next = queue.removeFirst()
            waiting[key] = queue.isEmpty ? nil : queue
            next.resume()
        } else {
            held.remove(key)
        }
    }
}
