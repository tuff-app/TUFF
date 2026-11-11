import Foundation
import Synchronization
import TUFFAppCore
import TUFFDecodeProtocol
import TUFFInferenceTypes
import TUFFModelCatalog
import TUFFRuntimeDescriptions

/// Serves every installed model through the per-profile serving host.
///
/// The host owns the model helpers; this provider is one client of it, next to
/// the app. It never loads a model itself, so an API request and a chat in
/// the app use the same resident weights. Requests run one at a time, with the
/// configured number allowed to wait, as before; the host runs them.
public final class HostRoutedModelProvider: ServerModelProvider {
    public static let defaultModelAlias = "default"
    /// Prompt-cache sessions kept open in the host. Each holds retained
    /// conversation state, so the number is small.
    static let maximumKeyedSessions = 4

    public let installed: ServerInstalledModels
    public let settings: ServerSettingsSource
    let link: HostServingLink
    let appleAvailability: @Sendable () -> AppleModelAvailability
    let gate: ServerCoordinator
    /// The allocation a load requests. Tests substitute a fixed plan.
    let automaticPlan: @Sendable (TUFFModelDescriptor) -> AppAutomaticMemoryPlan?

    struct KeyedSession {
        var id: UUID
        var model: String
        var lastUsed: Date
    }

    struct State {
        /// Models this adapter caused the host to load, with the configuration
        /// it asked for. Only these are unloaded when the API goes idle.
        var apiLoaded: [String: ServingModelPreparation] = [:]
        /// Whether the last request for a model ran in the configuration the
        /// adapter planned, which is when Auto may learn from it.
        var ranOwnPlan: [String: Bool] = [:]
        var keyed: [String: KeyedSession] = [:]
        var failedSessionEnds: Set<UUID> = []
        var transition: String?
        var idleDeadline: Date?
        var idleTask: Task<Void, Never>?
        var runGeneration: UInt64 = 0
    }
    let state = Mutex(State())

    public init(installed: ServerInstalledModels, settings: ServerSettingsSource,
                link: HostServingLink,
                automaticPlan: (@Sendable (TUFFModelDescriptor) -> AppAutomaticMemoryPlan?)? = nil,
                appleAvailability: @escaping @Sendable () -> AppleModelAvailability = { .current() },
                onActivity: @escaping @Sendable (ServerCoordinatorActivity) -> Void = { _ in }) {
        self.installed = installed
        self.settings = settings
        self.link = link
        self.appleAvailability = appleAvailability
        self.automaticPlan = automaticPlan ?? { installed.automaticPlan(for: $0) }
        gate = ServerCoordinator(queueLimit: settings.current.queueLimit, onActivity: onActivity)
        // Staged images from API processes that died are not swept by anyone
        // else; a live app's directory is left alone.
        AppImageAttachmentStore.sweepAbandoned()
    }

    // MARK: - Discovery

    public var parserVisionCapability: String {
        installed.available().contains { installed.visionCapability(for: $0) == "ready" }
            ? "ready" : "missing"
    }

    public func modelList() -> OpenAIModelList {
        var models: [OpenAIModelList.Model] = installed.available().map { descriptor in
            let context = installed.inferencePlan(for: descriptor).contextTokens
            return .init(id: descriptor.apiModelID, object: "model", created: 0, ownedBy: "tuff",
                  capabilities: installed.visionCapability(for: descriptor) == "ready"
                    ? ["text", "image"] : ["text"],
                  contextLength: context, maxOutputTokens: min(4_096, context / 4))
        }
        if appleAvailability().canGenerate {
            models.append(.init(id: AppleModelKind.onDevice.rawValue, object: "model", created: 0, ownedBy: "apple", capabilities: ["text"]))
        }
        return OpenAIModelList(object: "list", data: models)
    }

    public func route(_ requestedModel: String) throws -> ServerRoutedModel {
        let name = requestedModel == Self.defaultModelAlias
            ? settings.current.defaultModel : requestedModel
        if ["apple", AppleModelKind.onDevice.rawValue].contains(name) {
            let availability = appleAvailability()
            guard availability.canGenerate else { throw ServerRequestError.modelUnavailable(availability.label) }
            return ServerRoutedModel(id: AppleModelKind.onDevice.rawValue, requestedID: requestedModel,
                                     dialect: .chatml, visionCapability: "unsupported")
        }
        guard let descriptor = installed.descriptor(named: name) else {
            throw ServerRequestError.unknownModel
        }
        guard installed.isInstalled(descriptor) else {
            throw ServerRequestError.modelUnavailable(
                "\(descriptor.displayName) is not installed. Download it in TUFF first.")
        }
        guard installed.isEligible(descriptor) else {
            throw ServerRequestError.modelUnavailable(
                "\(descriptor.displayName) needs more memory than this Mac has.")
        }
        return ServerRoutedModel(
            id: descriptor.apiModelID, requestedID: requestedModel,
            dialect: ServerInstalledModels.dialect(for: descriptor),
            visionCapability: installed.visionCapability(for: descriptor))
    }

    // MARK: - Status

    /// What the host holds right now. Never starts a host.
    func residentModels() async -> [String] {
        guard let status = try? await link.status(launch: false) else { return [] }
        return status.loadedModelIDs
    }

    public func health() async -> ServerHealth {
        let resident = await residentModels()
        let defaultName = installed.descriptor(named: settings.current.defaultModel)?.apiModelID
        let known = resident.filter { installed.descriptor(named: $0) != nil }
        return ServerHealth(
            mode: "router", vision: parserVisionCapability,
            loadedModel: known.first { $0 == defaultName } ?? known.first,
            activeRequests: await gate.isActive ? 1 : 0,
            queuedRequests: await gate.queuedCount)
    }

    public func controlStatus(version: String) async -> ServerControlStatus {
        let resident = await residentModels()
        let current = settings.current
        let snapshot = state.withLock { ($0.transition, $0.idleDeadline) }
        return ServerControlStatus(
            version: version, mode: "router",
            residentModel: resident.first { installed.descriptor(named: $0) != nil },
            transition: snapshot.0,
            activeRequests: await gate.isActive ? 1 : 0,
            queuedRequests: await gate.queuedCount,
            idleUnloadInSeconds: snapshot.1.map { max(0, Int($0.timeIntervalSinceNow.rounded(.up))) },
            defaultModel: current.defaultModel,
            unloadDelaySeconds: current.unloadDelay.seconds)
    }

    public var queuedCount: Int { get async { await gate.queuedCount } }
    public var isActive: Bool { get async { await gate.isActive } }

    // MARK: - Running

    public func run<Prepared: Sendable, T: Sendable>(
        _ model: ServerRoutedModel,
        onQueued: @escaping @Sendable () -> Void,
        prepare: @escaping @Sendable (any ServerInferenceBackend) async throws -> Prepared,
        operation: @escaping @Sendable (any ServerInferenceBackend, Prepared) async throws -> T
    ) async throws -> T {
        if model.id == AppleModelKind.onDevice.rawValue {
            return try await gate.run(onQueued: onQueued) { [self] in
                let availability = appleAvailability()
                guard availability.canGenerate else {
                    throw ServerRequestError.modelUnavailable(availability.label)
                }
                beginRun()
                defer { endRun() }
                let backend = AppleServerInferenceBackend()
                let prepared = try await prepare(backend)
                return try await operation(backend, prepared)
            }
        }
        guard let descriptor = installed.descriptor(named: model.id) else {
            throw ServerRequestError.unknownModel
        }
        // The slot covers preparation too: preparing starts the generation in
        // the host, which must not overlap another API request's.
        return try await gate.run(onQueued: onQueued) { [self] in
            beginRun()
            let backend = HostServingBackend(provider: self, descriptor: descriptor)
            do {
                let prepared = try await prepare(backend)
                let value = try await operation(backend, prepared)
                await backend.finish(succeeded: true)
                endRun()
                return value
            } catch {
                await backend.finish(succeeded: false)
                endRun()
                throw error
            }
        }
    }

    public func completionRecorder(for model: ServerRoutedModel) -> (@Sendable (ServerCompletion) -> Void)? {
        guard let descriptor = installed.descriptor(named: model.id),
              let record = installed.autoRecorder(for: descriptor) else { return nil }
        return { [self] completion in
            // Auto learns only from runs in the configuration it chose; a run
            // that adopted the app's settings says nothing about them.
            if state.withLock({ $0.ranOwnPlan[model.id] ?? false }) { record(completion) }
        }
    }

    public func shutdown() async {
        state.withLock { current in
            current.idleTask?.cancel()
            current.idleTask = nil
            current.idleDeadline = nil
        }
        await gate.shutdown()
        _ = await releaseIdleModels()
        link.close()
    }

    // MARK: - Idle release

    /// The control endpoint: release what this adapter loaded, only when no
    /// request is running or waiting. False when busy.
    public func unloadIfIdle() async -> Bool {
        guard !(await gate.isActive), await gate.queuedCount == 0 else { return false }
        let generation = state.withLock { $0.runGeneration }
        do {
            return try await gate.run { [self] in
                guard state.withLock({ $0.runGeneration }) == generation else { return false }
                return await releaseIdleModels()
            }
        } catch {
            return false
        }
    }

    private func beginRun() {
        state.withLock { current in
            current.runGeneration &+= 1
            current.idleTask?.cancel()
            current.idleTask = nil
            current.idleDeadline = nil
        }
    }

    private func endRun() {
        let (generation, hasState) = state.withLock { current in
            (current.runGeneration, !current.apiLoaded.isEmpty || !current.keyed.isEmpty)
        }
        guard hasState else { return }
        let delay = max(0, settings.current.unloadDelay.seconds)
        let task = Task { [weak self] in
            if delay > 0 { do { try await Task.sleep(for: .seconds(delay)) } catch { return } }
            await self?.idleTimerFired(generation: generation)
        }
        state.withLock { current in
            current.idleTask?.cancel()
            current.idleTask = task
            current.idleDeadline = Date().addingTimeInterval(TimeInterval(delay))
        }
    }

    private func idleTimerFired(generation: UInt64) async {
        guard state.withLock({ $0.runGeneration }) == generation,
              !(await gate.isActive), await gate.queuedCount == 0 else { return }
        // Taking the slot makes the release wait for any request that arrived
        // in the meantime; the generation check then skips it.
        _ = try? await gate.run { [self] in
            guard state.withLock({ $0.runGeneration }) == generation else { return false }
            state.withLock { $0.idleDeadline = nil }
            return await releaseIdleModels()
        }
    }
}
