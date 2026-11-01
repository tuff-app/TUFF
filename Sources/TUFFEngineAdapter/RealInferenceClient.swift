import TUFFAppCore
import TUFFRuntimeManagement
import CryptoKit
import Foundation
import Metal
import TUFFEngine
import TUFFModelCatalog
import Synchronization

/// Real-model inference client for the Mac app. Wraps the same raw-completion
/// loop the CLI uses (`runRawCompletion`, BOS + verbatim encode, no chat
/// template) behind the `AppInferenceClient` event stream, with an explicit
/// load lifecycle so the resident weights stay warm across generations.
public final class RealInferenceClient: AppModelLifecycleClient, @unchecked Sendable {
    private let session: RealInferenceSession
    /// Bytes of image tower held mapped, readable without awaiting the session.
    public var currentVisionTowerBytes: UInt64? {
        session.towerBytes.withLock { $0 }
    }

    private let memorySampler: AppMemorySampler
    private let generationTasks = GenerationTaskRegistry()
    private let concurrentTasks = ConcurrentGenerationTasks()
    private let peers = ConcurrentInferencePeers()
    private let admission = InferenceAdmission()

    public init(memorySampler: AppMemorySampler = AppMemorySampler(),
                residencyCoordinator: AppResidencyCoordinator? = nil) {
        self.memorySampler = memorySampler
        self.session = RealInferenceSession(residencyCoordinator: residencyCoordinator)
    }

    public func ensureLoaded(modelDirectory: URL,
                             maxContextTokens: Int,
                             options: AppRuntimeOptions,
                             forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        await admission.beginLifecycle()
        defer { admission.endLifecycle() }
        try Task.checkCancellation()
        let key = SessionLoadKey(directory: modelDirectory.standardizedFileURL,
                                 maxContext: maxContextTokens, options: options,
                                 forceLogitsHead: forceLogitsHead)
        if await session.isLoaded(key) { return }
        concurrentTasks.cancelAll()
        await concurrentTasks.waitUntilIdle()
        await peers.removeAll()
        try await session.ensureLoaded(key: key, onState: onState)
    }

    public func unload() async {
        await admission.beginLifecycle()
        defer { admission.endLifecycle() }
        cancel()
        concurrentTasks.cancelAll()
        await waitUntilIdle()
        await peers.removeAll()
        await session.unload()
    }

    /// Waits for the producer's cleanup, including pending GPU and prefetch
    /// work. A cancelled stream consumer may finish before its producer does.
    public func waitUntilIdle() async {
        await generationTasks.waitUntilIdle()
        await concurrentTasks.waitUntilIdle()
    }

    public func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let generationID = UUID()
            guard admission.beginLegacy() else {
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            guard generationTasks.reserve(generationID) else {
                admission.endLegacy()
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            let task = Task(priority: .userInitiated) { [self] in
                await session.run(request: request,
                                  memorySampler: memorySampler,
                                  continuation: continuation)
                generationTasks.clear(generationID)
                admission.endLegacy()
            }
            generationTasks.attach(task, to: generationID)

            continuation.onTermination = { [generationTasks] _ in
                generationTasks.take(generationID)?.cancel()
            }
        }
    }

    public func cancel() {
        generationTasks.takeCurrent()?.cancel()
    }

    /// Used by the multiplexed helper protocol. A session owns its runner and
    /// cache; the request ID is used solely for targeted cancellation.
    public func generate(_ request: AppGenerationRequest, sessionID: UUID,
                         requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            guard admission.beginShared() else {
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            guard concurrentTasks.reserve(requestID: requestID, sessionID: sessionID) else {
                admission.endShared()
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            let task = Task(priority: .userInitiated) { [self] in
                do {
                    let peer = try await peers.session(for: sessionID, owner: session)
                    try Task.checkCancellation()
                    await peer.run(request: request, memorySampler: memorySampler,
                                   continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
                concurrentTasks.clear(requestID: requestID)
                admission.endShared()
            }
            concurrentTasks.attach(task, requestID: requestID)
            continuation.onTermination = { [concurrentTasks] _ in
                concurrentTasks.cancel(requestID: requestID)
            }
        }
    }

    public func cancel(requestID: UUID) { concurrentTasks.cancel(requestID: requestID) }
    public func waitUntilRequestIdle(_ requestID: UUID) async {
        await concurrentTasks.waitUntilRequestIdle(requestID)
    }
    public func cancelAllRequests() { cancel(); concurrentTasks.cancelAll() }
    public func cancelWarmup() { Task { await session.cancelWarmup() } }

    public func endSession(_ sessionID: UUID) async {
        guard concurrentTasks.beginEnding(sessionID) else { return }
        await concurrentTasks.waitUntilSessionIdle(sessionID)
        await peers.remove(sessionID)
        concurrentTasks.finishEnding(sessionID)
    }

}

private actor ConcurrentInferencePeers {
    private struct Entry {
        let session: RealInferenceSession
        let isOwner: Bool
    }
    private var sessions: [UUID: Entry] = [:]
    private var ownerClaimed = false

    func session(for id: UUID, owner: RealInferenceSession) async throws -> RealInferenceSession {
        if let existing = sessions[id] { return existing.session }
        let isOwner = !ownerClaimed
        if isOwner { ownerClaimed = true }
        let created: RealInferenceSession
        if isOwner { created = await owner.claimSharedOwner() }
        else { created = try await owner.makeSharedPeer() }
        // Only one active request per session is admitted by the registry.
        sessions[id] = Entry(session: created, isOwner: isOwner)
        return created
    }

    func remove(_ id: UUID) async {
        guard let entry = sessions.removeValue(forKey: id) else { return }
        if entry.isOwner {
            await entry.session.resetSharedOwnerSession()
            ownerClaimed = false
        } else { await entry.session.unload() }
    }

    func removeAll() async {
        let old = Array(sessions.values)
        sessions.removeAll()
        ownerClaimed = false
        for entry in old {
            if entry.isOwner { await entry.session.resetSharedOwnerSession() }
            else { await entry.session.unload() }
        }
    }
}

struct SessionLoadKey: Equatable, Sendable {
    var directory: URL
    var maxContext: Int
    var options: AppRuntimeOptions
    var forceLogitsHead: Bool

    init(directory: URL,
         maxContext: Int,
         options: AppRuntimeOptions,
         forceLogitsHead: Bool = false) {
        self.directory = directory.standardizedFileURL
        self.maxContext = maxContext
        self.options = options
        self.forceLogitsHead = forceLogitsHead
    }
}

struct TokenizerDirectoryCache: Equatable, Sendable {
    private(set) var directory: URL?

    func shouldReload(for modelDirectory: URL) -> Bool {
        directory != modelDirectory.standardizedFileURL
    }

    mutating func markLoaded(for modelDirectory: URL) {
        directory = modelDirectory.standardizedFileURL
    }

    mutating func clear() {
        directory = nil
    }
}

/// Owns the loaded model and serializes load / unload / generate. All Metal
/// command-buffer waits happen inside this actor, off the main actor; one
/// cooperative-pool thread is occupied for the duration of a generation,
/// which is acceptable for the app's single session. The 8 GB rule lives
/// here: a reload releases the loaded model, runner, and scratch before constructing
/// replacements, so two models are never alive at once.
/// The immutable model mapping is shared; each peer constructs its own
/// runner, KV state, and scratch. Mutable model staging is protected by the
/// same forward lease in every peer.
private struct SharedInferenceResources: @unchecked Sendable {
    let model: Model
    let context: MetalContext
    let tokenizer: GFTokenizer
    let key: SessionLoadKey
    let lease: FairForwardLease
    let residencyLease: TUFFResidencyLease
    let visionRuntime: VisionRuntime?
    let visionRuntimeError: Error?
    let residencyCoordinator: AppResidencyCoordinator
    let ownerResidencyLease: TUFFResidencyLease
    let visionWorkspaceLease: TUFFResidencyLease?
}

/// The runtime and model are owned by this session; the shared lease protects
/// model expert residency until image encoding and its GPU work finish.
private struct SharedVisionEncode: @unchecked Sendable {
    let runtime: VisionRuntime
    let model: Model
    let fileURL: URL
    let policy: VisionResidencyPolicy
}

actor RealInferenceSession {
    private let residencyCoordinator: AppResidencyCoordinator?
    private var residencyLease: TUFFResidencyLease?

    init(residencyCoordinator: AppResidencyCoordinator? = nil) {
        self.residencyCoordinator = residencyCoordinator
    }
    private init(shared: SharedInferenceResources) throws {
        self.residencyCoordinator = shared.residencyCoordinator
        let configuration = try shared.key.options.resolvedRuntimeConfiguration(
            forceLogitsHead: shared.key.forceLogitsHead)
        self.loadedKey = shared.key
        self.ctx = shared.context
        self.tokenizer = shared.tokenizer
        self.runner = try ModelForwardRunner(model: shared.model, context: shared.context,
            maxContext: shared.key.maxContext, runtimeConfiguration: configuration)
        self.scratch = try RawCompletionScratch(context: shared.context,
            vocab: shared.model.config.vocabSize,
            logitSoftcap: Float(shared.model.config.finalLogitSoftcap))
        self.model = shared.model
        self.sharedForwardLease = shared.lease
        self.residencyLease = shared.residencyLease
        self.visionRuntime = shared.visionRuntime
        self.visionRuntimeError = shared.visionRuntimeError
        self.ownerResidencyLease = shared.ownerResidencyLease
        self.visionWorkspaceLease = shared.visionWorkspaceLease
        self.conversations = ConversationStateStore(budgetBytes: Self.retainedConversationBudget(
            model: shared.model, key: shared.key, imagePackInstalled: shared.visionRuntime != nil))
        self.conversationDomain = Self.conversationDomain(
            model: shared.model, key: shared.key, runtime: configuration)
    }
    private var loadedKey: SessionLoadKey?
    private var ctx: MetalContext?
    private var tokenizer: GFTokenizer?
    private var tokenizerDirectoryCache = TokenizerDirectoryCache()
    private var runner: ModelForwardRunner?
    private var scratch: RawCompletionScratch?
    private var model: Model?
    private var residentWarmup: ResidentWarmup?
    private var sharedForwardLease: FairForwardLease?
    private var loadEpoch = UUID()
    private var visionWorkspaceLease: TUFFResidencyLease?
    private var ownerResidencyLease: TUFFResidencyLease?
    /// The conversation the runner holds and any retained ones. Replaced with
    /// the session, so nothing outlives the model it was computed by.
    private var conversations = ConversationStateStore(budgetBytes: 0)
    private var conversationDomain: ConversationCacheDomain?
    /// Chats saved on disk, so one reopened after a restart resumes instead
    /// of reading its whole history again. Nil when conversation state is off.
    private var savedChats: PrefixSnapshotDiskStore?

    /// Bytes of image tower held mapped right now, published outside the actor
    /// so a reader does not have to await it mid-decode.
    nonisolated let towerBytes = Mutex<UInt64?>(nil)

    /// The owner keeps the model residency lease alive while peer sessions
    /// exist. The caller must reserve each peer's additional working memory.
    func makeSharedPeer() async throws -> RealInferenceSession {
        guard let model, let ctx, let tokenizer, let loadedKey,
              let residencyCoordinator, let residencyLease else {
            throw AppInferenceError.modelNotLoaded
        }
        residentWarmup?.cancel()
        residentWarmup = nil
        let epoch = loadEpoch
        let peerReservation = try await residencyCoordinator.reserveSession(
            directory: loadedKey.directory, context: loadedKey.maxContext, options: loadedKey.options)
        guard epoch == loadEpoch else {
            peerReservation.release()
            throw AppInferenceError.modelNotLoaded
        }
        let lease = sharedForwardLease ?? FairForwardLease()
        sharedForwardLease = lease
        return try RealInferenceSession(shared: SharedInferenceResources(
            model: model, context: ctx, tokenizer: tokenizer, key: loadedKey,
            lease: lease, residencyLease: peerReservation,
            visionRuntime: visionRuntime, visionRuntimeError: visionRuntimeError,
            residencyCoordinator: residencyCoordinator, ownerResidencyLease: residencyLease,
            visionWorkspaceLease: visionWorkspaceLease))
    }

    private var visionRuntime: VisionRuntime? {
        didSet { publishTowerBytes() }
    }
    private var visionRuntimeError: Error?

    func isLoaded(_ key: SessionLoadKey) -> Bool { loadedKey == key && runner != nil }

    func cancelWarmup() {
        residentWarmup?.cancel()
        residentWarmup = nil
    }

    /// The first session reuses the already-admitted owner's runner and vision
    /// runtime. Additional peers receive separately admitted private state.
    func claimSharedOwner() -> RealInferenceSession {
        cancelWarmup()
        if sharedForwardLease == nil { sharedForwardLease = FairForwardLease() }
        return self
    }

    func resetSharedOwnerSession() {
        runner?.reset()
        let budget = conversations.budgetBytes
        conversations = ConversationStateStore(budgetBytes: budget)
        savedChats?.flush()
    }

    /// Called after anything that maps or releases tower regions.
    private func publishTowerBytes() {
        let value = visionRuntime.map { UInt64($0.retainedWeightBytes) }
        towerBytes.withLock { $0 = value }
    }

    func ensureLoaded(key: SessionLoadKey,
                      onState: @Sendable (AppModelLoadState) -> Void) async throws {
        if loadedKey == key, runner != nil { return }

        loadEpoch = UUID()
        residentWarmup?.cancel()
        residentWarmup = nil
        conversations = ConversationStateStore(budgetBytes: 0)
        conversationDomain = nil
        savedChats?.flush()
        savedChats = nil
        runner = nil
        scratch = nil
        model = nil
        loadedKey = nil
        visionRuntime = nil
        visionRuntimeError = nil
        visionWorkspaceLease = nil
        residencyLease = nil
        ownerResidencyLease = nil

        let start = Date()
        do {
            onState(.loading(.validatingDirectory))
            let manifest = key.directory.appendingPathComponent("manifest.json")
            guard FileManager.default.fileExists(atPath: manifest.path) else {
                throw AppInferenceError.modelNotFound(key.directory.path)
            }

            if let residencyCoordinator {
                residencyLease = try await residencyCoordinator.reserve(
                    directory: key.directory, context: key.maxContext, slots: key.options.expertCacheSlots,
                    chunk: key.options.prefillChunkTokens, options: key.options)
            }
            var loadSucceeded = false
            defer { if !loadSucceeded { residencyLease = nil } }
            // The shader compiler runs while the tokenizer and weights load.
            let context: MetalContext
            if let ctx {
                context = ctx
            } else {
                context = try MetalContext(kernelGroups: [])
                ctx = context
            }
            context.prewarmKernels(forModelDirectory: key.directory, maxContext: key.maxContext,
                                   includeVision: VisionRuntime.isSupported(on: context.device))
            onState(.loading(.tokenizer))
            // The tokenizer is read while the weights load; neither needs the other.
            let tokenizerLoad: Task<GFTokenizer, Error>?
            if tokenizer == nil || tokenizerDirectoryCache.shouldReload(for: key.directory) {
                let directory = key.directory
                tokenizerLoad = Task.detached(priority: .userInitiated) {
                    try await Self.loadTokenizer(for: directory)
                }
            } else {
                tokenizerLoad = nil
            }
            defer { tokenizerLoad?.cancel() }
            try Task.checkCancellation()

            onState(.loading(.verifyingWeights))
            let runtimeConfiguration = try key.options.resolvedRuntimeConfiguration(
                forceLogitsHead: key.forceLogitsHead)
            let loadedModel = try Model.load(
                directoryURL: key.directory,
                device: context.device,
                streamingMode: .pread(slotCount: runtimeConfiguration.expertCacheSlots),
                denseResidency: key.options.denseResidencyPolicy,
                expertCachePolicy: runtimeConfiguration.modelExpertCachePolicy,
                integrityPolicy: key.options.modelVerification.runtimeValue)
            try Task.checkCancellation()

            onState(.loading(.preparingRunner))
            let loadedRunner = try ModelForwardRunner(
                model: loadedModel,
                context: context,
                maxContext: key.maxContext,
                runtimeConfiguration: runtimeConfiguration)
            let loadedScratch = try RawCompletionScratch(context: context,
                                                         vocab: loadedModel.config.vocabSize,
                                                         logitSoftcap: Float(loadedModel.config.finalLogitSoftcap))
            if let tokenizerLoad {
                do {
                    tokenizer = try await tokenizerLoad.value
                    tokenizerDirectoryCache.markLoaded(for: key.directory)
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    throw AppInferenceError.tokenizerUnavailable("\(error)")
                }
            }
            try Task.checkCancellation()

            let loadedVisionRuntime: VisionRuntime?
            let loadedVisionRuntimeError: Error?
            var loadedVisionReservation: TUFFResidencyLease?
            do {
                let runtime = try VisionRuntime.open(
                    textModelURL: key.directory,
                    context: context)
                // Keep Ready maps the tower during the load rather than on the
                // first image, so the wait is where the user asked for it.
                if key.options.visionResidencyPolicy == .keepReady {
                    onState(.loading(.mappingImageTower))
                    if let residencyCoordinator, let residencyLease {
                        loadedVisionReservation = try residencyCoordinator.reserveVision(allowing: [residencyLease])
                    }
                    try runtime.prewarmWeightRegions()
                }
                loadedVisionRuntime = runtime
                loadedVisionRuntimeError = nil
            } catch {
                // A missing or invalid pack only means images are unavailable;
                // the reason travels to the first image turn.
                loadedVisionReservation = nil
                loadedVisionRuntime = nil
                loadedVisionRuntimeError = error
            }
            try Task.checkCancellation()

            loadSucceeded = true
            runner = loadedRunner
            scratch = loadedScratch
            model = loadedModel
            loadedKey = key
            conversations = ConversationStateStore(
                budgetBytes: Self.retainedConversationBudget(
                    model: loadedModel, key: key,
                    imagePackInstalled: loadedVisionRuntime != nil))
            conversationDomain = Self.conversationDomain(
                model: loadedModel, key: key, runtime: runtimeConfiguration)
            savedChats = conversations.budgetBytes > 0 && Self.savesChats()
                ? .standard(modelID: loadedModel.modelID, buildIdentity: TUFFVersion.current)
                : nil
            visionRuntime = loadedVisionRuntime
            visionRuntimeError = loadedVisionRuntimeError
            visionWorkspaceLease = loadedVisionReservation
            context.logLoadStatistics(key.directory.lastPathComponent)
            onState(.ready(modelDirectory: key.directory,
                           loadSeconds: Date().timeIntervalSince(start)))
            residentWarmup = loadedModel.startResidentWarmup()
        } catch is CancellationError {
            throw CancellationError()
        } catch let appError as AppInferenceError {
            onState(.failed(appError))
            throw appError
        } catch {
            let appError = AppInferenceError.modelLoadFailed("\(error)")
            onState(.failed(appError))
            throw appError
        }
    }

    private static func loadTokenizer(for modelDirectory: URL) async throws -> GFTokenizer {
        try await GFTokenizer.load(forModelDirectory: modelDirectory)
    }

    static func forceLogitsHead(for request: AppGenerationRequest) -> Bool {
        !request.isPureGreedy
    }

    static func generationConfig(for request: AppGenerationRequest,
                                 maxNewTokens: Int? = nil) -> GenerationConfig {
        GenerationConfig(maxNewTokens: maxNewTokens ?? request.maxNewTokens,
                         temperature: request.temperature,
                         topK: request.topK,
                         topP: request.topP,
                         repetitionPenalty: request.repetitionPenalty,
                         seed: request.seed)
    }

    static func effectiveMaxNewTokens(requested: Int,
                                      promptTokenCount: Int,
                                      maxContext: Int) -> Int {
        min(requested, max(0, maxContext - promptTokenCount))
    }

    struct RenderedConversation {
        let tokens: [Int32]
        let trim: AppConversationTrim
    }

    struct RenderedMultimodalConversation {
        let input: MultimodalPrefillInput
        let trim: AppConversationTrim
    }

    /// Whether completed turns render their reasoning back: Harmony keeps
    /// its analysis channel, and Qwen 3.6 does when asked to preserve it.
    static func rendersFinalThinking(request: AppGenerationRequest,
                                     modelVariant: ModelVariant?) -> Bool {
        modelVariant == .gptOss_20B || modelVariant == .gptOss_120B
            || (modelVariant == .qwen36_35B_A3B && request.preserveThinking)
    }

    /// Encodes `turns` plus the current message through the template the
    /// request needs: Harmony for GPT-OSS, the model's own tool template when
    /// tools are declared or present, and the plain chat renderer otherwise.
    static func encode(request: AppGenerationRequest,
                       turns: ArraySlice<AppChatTurn>,
                       tokenizer: GFTokenizer,
                       modelVariant: ModelVariant?,
                       harmonyDate: String) throws -> [Int32] {
        let isHarmony = modelVariant == .gptOss_20B || modelVariant == .gptOss_120B
        let preservesChatMLThinking = modelVariant == .qwen36_35B_A3B
            && request.preserveThinking
        let messages = AppConversationMessages.messages(
            for: request, turns: turns,
            finalThinking: rendersFinalThinking(request: request, modelVariant: modelVariant))
        if isHarmony {
            return try tokenizer.encodeHarmonyChat(
                messages: messages,
                tools: request.tools,
                reasoningEffort: request.reasoningEffort ?? .medium,
                currentDate: harmonyDate)
        }
        // The generation prompt ends exactly where the assistant's text
        // begins, so a partial answer appended here is continued rather than
        // restarted. Harmony is excluded upstream: it opens a channel after
        // that point, and text placed here would not land in the reply.
        if request.usesToolTemplate {
            return try tokenizer.encodeToolChat(
                messages: messages, tools: request.tools,
                reasoning: request.reasoning, preserveThinking: preservesChatMLThinking)
                + (request.assistantPrefix.isEmpty
                    ? [] : tokenizer.encode(request.assistantPrefix, addBOS: false))
        }
        let rendered = try tokenizer.applyChatTemplate(
            messages,
            modelVariant: modelVariant,
            reasoning: request.reasoning,
            preserveThinking: preservesChatMLThinking)
        return tokenizer.encode(rendered + request.assistantPrefix, addBOS: false)
    }

    /// How much of `promptIDs` the next request will render the same way: up
    /// to where this turn's answer begins. A template may open the answer
    /// with something a finished turn drops, such as Qwen's `<think>` with
    /// thinking on, so the turn is rendered with a placeholder answer and
    /// only the tokens that agree count. The server finds its checkpoints
    /// the same way.
    static func turnCheckpointPosition(request: AppGenerationRequest,
                                       promptIDs: [Int32],
                                       tokenizer: GFTokenizer,
                                       modelVariant: ModelVariant?,
                                       harmonyDate: String) -> Int? {
        var probe = request
        probe.history.append(AppChatTurn(prompt: request.prompt, response: "\u{1}"))
        probe.prompt = "\u{1}"
        probe.currentRounds = []
        probe.assistantPrefix = ""
        guard let rendered = try? encode(request: probe, turns: probe.history[...],
                                         tokenizer: tokenizer, modelVariant: modelVariant,
                                         harmonyDate: harmonyDate) else { return nil }
        let agreeing = zip(rendered, promptIDs).prefix { $0 == $1 }.count
        return agreeing > 0 ? agreeing : nil
    }

    /// Render the conversation to tokens, dropping oldest turns until it fits.
    ///
    /// A conversation grows without bound while the context window does not, so
    /// something has to give. Dropping from the front keeps the newest exchanges
    /// — the ones a follow-up question actually depends on — and keeps the app
    /// usable instead of failing the moment history outgrows the window.
    ///
    /// The current prompt and its tool rounds are never dropped: if they alone
    /// do not fit, that is a real context overflow and is reported as one.
    static func renderConversation(
        request: AppGenerationRequest,
        tokenizer: GFTokenizer,
        maxContext: Int,
        modelVariant: ModelVariant? = nil,
        harmonyDate: String = HarmonyPromptRenderer.calendarDate()
    ) throws -> RenderedConversation {
        var dropped = 0
        while true {
            let tokens = try encode(request: request, turns: request.history[dropped...],
                                    tokenizer: tokenizer, modelVariant: modelVariant,
                                    harmonyDate: harmonyDate)
            if tokens.count < maxContext {
                return RenderedConversation(
                    tokens: tokens,
                    trim: AppConversationTrim(droppedTurns: dropped,
                                              promptTokens: tokens.count))
            }
            if request.preservesFullHistory {
                throw AppInferenceError.contextOverflow(prompt: tokens.count, maxNew: 0, maxContext: maxContext)
            }
            guard dropped < request.history.count else {
                // Nothing left to drop; the prompt itself overflows.
                throw AppInferenceError.contextOverflow(
                    prompt: tokens.count,
                    maxNew: request.maxNewTokens,
                    maxContext: maxContext)
            }
            dropped += 1
        }
    }

    /// Every image a request needs encoded, current message first and each
    /// earlier turn after it, with duplicates removed by attachment id.
    ///
    /// Order matters: encoding is the expensive part of an image prompt, and
    /// the current message is the one that must be encoded even if the context
    /// leaves no room for the rest.
    static func attachmentsToEncode(
        for request: AppGenerationRequest
    ) -> [AppImageAttachment] {
        var seen = Set<UUID>()
        var ordered: [AppImageAttachment] = []
        for attachment in request.imageAttachments + request.history.reversed()
            .flatMap(\.images) where seen.insert(attachment.id).inserted {
            ordered.append(attachment)
        }
        return ordered
    }

    static func renderMultimodalConversation(
        request: AppGenerationRequest,
        features: [UUID: VisionFeatures],
        tokenizer: GFTokenizer,
        maxContext: Int,
        family: ModelFamily = .gemma4,
        modelVariant: ModelVariant? = nil
    ) throws -> RenderedMultimodalConversation {
        let preservesChatMLThinking = family == .qwen36 && request.preserveThinking
        var dropped = 0
        while true {
            // The renderer refuses features nothing refers to, and dropping a
            // turn to fit the context drops its images with it, so the map is
            // rebuilt for each attempt rather than handed the whole set once.
            let built = AppConversationMessages.multimodalMessages(
                for: request, turns: request.history[dropped...], features: features,
                finalThinking: preservesChatMLThinking)
            let input = try MultimodalPromptRenderer.render(
                messages: built.messages,
                featuresByID: built.used,
                tokenizer: tokenizer,
                tools: request.tools,
                family: family,
                modelVariant: modelVariant,
                reasoning: request.reasoning,
                preserveThinking: preservesChatMLThinking)
            if input.effectiveTokenIDs.count < maxContext {
                return RenderedMultimodalConversation(
                    input: input,
                    trim: AppConversationTrim(
                        droppedTurns: dropped,
                        promptTokens: input.effectiveTokenIDs.count))
            }
            guard dropped < request.history.count else {
                throw AppInferenceError.contextOverflow(
                    prompt: input.effectiveTokenIDs.count,
                    maxNew: request.maxNewTokens,
                    maxContext: maxContext)
            }
            dropped += 1
        }
    }

    /// The untrimmed conversation as the cache sees it. A request whose
    /// render had to drop turns neither publishes nor matches through this.
    static func transcript(for request: AppGenerationRequest,
                           modelVariant: ModelVariant?,
                           harmonyDate: String?) -> ConversationTranscript {
        let isHarmony = modelVariant == .gptOss_20B || modelVariant == .gptOss_120B
        let turns = request.history[...]
        return ConversationTranscript(
            messages: AppConversationMessages.messages(
                for: request, turns: turns,
                finalThinking: rendersFinalThinking(request: request, modelVariant: modelVariant)),
            imageIdentities: AppConversationMessages.imageIdentities(for: request, turns: turns),
            tools: request.tools,
            reasoning: request.reasoning,
            reasoningEffort: request.reasoningEffort,
            harmonyCurrentDate: isHarmony ? harmonyDate : nil,
            preserveThinking: request.preserveThinking)
    }

    /// The app bridges a finished exchange to the next user message only for
    /// ordinary, non-thinking turns, as it always has. With reasoning on,
    /// the cached tokens hold reasoning a fresh render would drop, so the
    /// next message is matched by an exact rendered prefix instead. Tool
    /// results always continue the KV, because the templates keep the
    /// in-progress turn's reasoning.
    static func allowsTextBridge(_ request: AppGenerationRequest) -> Bool {
        request.assistantPrefix.isEmpty && request.reasoning == .off
            && !request.preserveThinking && request.reasoningEffort == nil
    }

    /// Below this a chat is read again faster than its state is written.
    static let minimumSavedChatTokens = 1_024

    /// A snapshot holds the whole state, hundreds of megabytes for a long
    /// chat on a large model, so it is rewritten only once the chat has
    /// grown by a quarter (and at least 512 tokens). A long chat is saved a
    /// handful of times rather than after every reply, and a restart reads
    /// at most that quarter again.
    static func shouldSaveChat(at position: Int, saved: Int?) -> Bool {
        guard position >= minimumSavedChatTokens else { return false }
        guard let saved, saved < position else { return saved == nil }
        return position - saved >= max(512, saved / 4)
    }

    /// `TUFF_SAVED_CHATS=off` keeps chat state in memory only.
    static func savesChats(environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        environment["TUFF_SAVED_CHATS"] != "off"
    }

    static func retainedConversationBudget(model: Model, key: SessionLoadKey,
                                           imagePackInstalled: Bool) -> Int {
        if let bytes = key.options.retainedConversationBytes {
            return ConversationRetentionBudget.budget(safeBudgetBytes: UInt64(bytes), workingSetBytes: 0, capBytes: UInt64(bytes))
        }
        guard let descriptor = TUFFModelCatalog.all.first(where: {
            $0.architecture.id.rawValue == model.config.variant.rawValue
        }) else { return 0 }
        return descriptor.retainedConversationBudgetBytes(
            contextTokens: key.maxContext,
            expertCacheSlots: key.options.expertCacheSlots,
            prefillChunkTokens: key.options.prefillChunkTokens,
            device: TUFFDeviceCapabilities.current(),
            imagePackInstalled: imagePackInstalled)
    }

    static func conversationDomain(model: Model, key: SessionLoadKey,
                                   runtime: RuntimeConfiguration) -> ConversationCacheDomain {
        let switches = ProcessInfo.processInfo.environment
            .filter { $0.key.hasPrefix("TUFF_") }
            .map { "\($0.key)=\($0.value)" }.sorted()
        let identity = ([GFTokenizer.toolChatTemplateIdentity, InferenceEngineEnvironment.computationIdentity,
                         String(runtime.expertCacheSlots), runtime.expertCachePolicy.rawValue,
                         runtime.prefillPolicy.rawValue, String(runtime.prefillChunkTokens),
                         runtime.headPath.rawValue, String(key.forceLogitsHead)] + switches)
            .joined(separator: ":")
        let template = GFTokenizer.tokenizerFolder(forModelDirectory: key.directory)
            .flatMap { try? Data(contentsOf: $0.appendingPathComponent("chat_template.jinja")) }
            .map(Self.sha256Hex) ?? "builtin"
        return ConversationCacheDomain(
            modelID: model.modelID, sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: Self.sha256Hex(Data(identity.utf8)),
            maximumContext: key.maxContext, kvStorage: PrefillKVStorageMode.fp16.rawValue,
            fp16RingEnabled: runtime.fp16RingEnabled, templateSHA256: template)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    func unload() {
        loadEpoch = UUID()
        residentWarmup?.cancel()
        residentWarmup = nil
        // Unload is also the helper's way out when the app quits, so a chat
        // still being written finishes first instead of being lost.
        savedChats?.flush()
        savedChats = nil
        conversations = ConversationStateStore(budgetBytes: 0)
        conversationDomain = nil
        visionRuntime = nil
        visionRuntimeError = nil
        visionWorkspaceLease = nil
        model = nil
        runner = nil
        scratch = nil
        tokenizer = nil
        tokenizerDirectoryCache.clear()
        loadedKey = nil
        residencyLease = nil
        ownerResidencyLease = nil
    }

    func run(request: AppGenerationRequest,
             memorySampler: AppMemorySampler,
             continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async {
        residentWarmup?.cancel()
        residentWarmup = nil
        var prefillConfig = request.runtimeOptions.prefillConfig
        // Image spans only run under chunked prefill, and the app's prefill toggle
        // can select `.off`. Coerce rather than fail after the encodes: whether
        // images work is not a performance preference.
        //
        // An image carried in from an earlier turn puts spans in the prompt just
        // as one attached to this message does, so the test is the whole
        // conversation's images, not this message's.
        let encodableAttachments = Self.attachmentsToEncode(for: request)
        if !encodableAttachments.isEmpty,
           let coerced = prefillConfig.coercedForImagePrompt() {
            prefillConfig = coerced
        }
        let progress = ProgressState()
        var completionStarted = false
        var imageReservation: TUFFResidencyLease?
        defer { imageReservation?.release() }
        do {
            try request.validate()
            if !encodableAttachments.isEmpty, visionWorkspaceLease == nil,
               let residencyCoordinator, let residencyLease {
                imageReservation = try residencyCoordinator.reserveVision(
                    allowing: [residencyLease] + (ownerResidencyLease.map { [$0] } ?? []))
            }
            let executedPrefillMode: PrefillExecutedMode =
                prefillConfig.mode == .chunked ? .chunked : .off
            let prefillDiagnostics = PrefillExecutionDiagnostics(config: prefillConfig,
                                                                 executedMode: executedPrefillMode,
                                                                 kvStorageMode: .fp16)
            let requestKey = SessionLoadKey(
                directory: request.modelDirectory.standardizedFileURL,
                maxContext: request.maxContextTokens,
                options: request.runtimeOptions,
                forceLogitsHead: Self.forceLogitsHead(for: request))
            guard let loadedKey else { throw AppInferenceError.modelNotLoaded }
            guard loadedKey == requestKey else { throw AppInferenceError.reloadRequired }
            guard let runner, let tokenizer, let ctx, let scratch, let model,
                  let conversationDomain else {
                throw AppInferenceError.modelLoadFailed("session lost its loaded state")
            }
            let variant = model.config.variant
            let harmonyDate = HarmonyPromptRenderer.calendarDate()
            let transcript = Self.transcript(for: request, modelVariant: variant,
                                             harmonyDate: harmonyDate)

            var promptIds: [Int32]
            var multimodalInput: MultimodalPrefillInput?
            var conversationTrim: AppConversationTrim
            var completionStart = RawCompletionStart.reset
            var cacheSource = ConversationStateStore.Source.cold

            // Text-only prompts render first, so an exact rendered prefix can
            // match. With images the rendered ids cannot identify them, so
            // only the message history can establish a match, and a hit skips
            // encoding the images already inside the cached state.
            let renderedText: RenderedConversation? = encodableAttachments.isEmpty
                ? try Self.renderConversation(request: request, tokenizer: tokenizer,
                                              maxContext: runner.maxContext,
                                              modelVariant: variant, harmonyDate: harmonyDate)
                : nil
            let plan = conversations.plan(
                domain: conversationDomain, transcript: transcript,
                renderedPromptIDs: renderedText.map {
                    $0.trim.droppedTurns == 0 ? $0.tokens : []
                },
                tokenizer: tokenizer, modelVariant: variant,
                conversationKey: request.conversationKey,
                runner: runner,
                allowsTextBridge: Self.allowsTextBridge(request))
            let plannedEntry = plan.match.isHit ? conversations.active : nil
            if case .hit(let effective, _) = plan.match, effective.count < runner.maxContext {
                promptIds = effective
                multimodalInput = nil
                conversationTrim = AppConversationTrim(droppedTurns: 0,
                                                        promptTokens: effective.count)
                completionStart = plan.start
                cacheSource = plan.source
            } else if let renderedText {
                promptIds = renderedText.tokens
                multimodalInput = nil
                conversationTrim = renderedText.trim
                // Nothing in memory matched: this chat may have been saved by
                // an earlier process. Restoring replaces the runner's state,
                // so the active entry no longer describes it.
                if renderedText.trim.droppedTurns == 0, let key = request.conversationKey,
                   let savedChats, runner.supportsPrefixCheckpoints,
                   let snapshot = savedChats.loadChat(
                    conversationKey: key, prompt: renderedText.tokens,
                    domain: conversationDomain, runner: runner, device: ctx.device) {
                    conversations.invalidateActive()
                    do {
                        try runner.restoreState(snapshot)
                        completionStart = .resume(cachedPromptTokens: snapshot.position)
                        cacheSource = .saved
                    } catch {
                        // The runner reset itself; read the prompt from scratch.
                        completionStart = .reset
                    }
                }
            } else {
                guard let visionRuntime else {
                    throw AppInferenceError.invalidRequest(
                        "Image support is unavailable: "
                            + (visionRuntimeError.map(String.init(describing:))
                                ?? "the image companion pack is not installed"))
                }
                var features: [UUID: VisionFeatures] = [:]
                features.reserveCapacity(encodableAttachments.count)
                for attachment in encodableAttachments {
                    try Task.checkCancellation()
                    // The file may have changed between selection and send.
                    let actualDigest = try Sha256Verifier.hashFile(
                        at: attachment.fileURL, chunkBytes: 256 * 1_024)
                    guard actualDigest == attachment.sha256 else {
                        throw AppInferenceError.invalidRequest(
                            "Image \(attachment.displayName) changed after selection.")
                    }
                    defer { publishTowerBytes() }
                    if let lease = sharedForwardLease {
                        let work = SharedVisionEncode(runtime: visionRuntime, model: model,
                            fileURL: attachment.fileURL,
                            policy: request.runtimeOptions.visionResidencyPolicy)
                        features[attachment.id] = try await lease.withLease {
                            try work.runtime.encodeImage(at: work.fileURL, languageModel: work.model,
                                residencyPolicy: work.policy,
                                checkCancellation: { try Task.checkCancellation() })
                        }
                    } else {
                        features[attachment.id] = try visionRuntime.encodeImage(
                            at: attachment.fileURL, languageModel: model,
                            residencyPolicy: request.runtimeOptions.visionResidencyPolicy,
                            checkCancellation: { try Task.checkCancellation() })
                    }
                }
                let rendered = try Self.renderMultimodalConversation(
                    request: request,
                    features: features,
                    tokenizer: tokenizer,
                    maxContext: runner.maxContext,
                    family: model.config.family,
                    modelVariant: variant)
                promptIds = rendered.input.effectiveTokenIDs
                multimodalInput = rendered.input
                conversationTrim = rendered.trim
            }
            if request.preservesFullHistory && conversationTrim.droppedTurns > 0 {
                throw AppInferenceError.invalidRequest("The full conversation exceeds this Auto context. Increase context or start a new chat; no history was removed.")
            }
            progress.promptTokenCount = promptIds.count
            progress.conversationTrim = conversationTrim
            memorySampler.resetPeak()
            _ = memorySampler.sample()
            let config = Self.generationConfig(
                for: request,
                maxNewTokens: Self.effectiveMaxNewTokens(
                    requested: request.maxNewTokens,
                    promptTokenCount: promptIds.count,
                    maxContext: runner.maxContext))
            progress.prefillStart = Date()

            let toolNames = Set(request.tools.map(\.name))
            let assistantDecoder = StructuredAssistantDecoder(
                tokenizer: tokenizer,
                // A tool-shaped reply naming anything undeclared is refused.
                allowedTools: toolNames,
                promptOpensThinking: StructuredAssistantDecoder.promptOpensThinking(
                    tokenizer: tokenizer, reasoning: request.reasoning),
                parameterSchemas: Dictionary(
                    request.tools.map { ($0.name, $0.parameters) },
                    uniquingKeysWith: { first, _ in first }))
            let publishAssistantEvents: @Sendable (
                [StructuredAssistantEvent], Int, Double
            ) -> Void = { events, index, elapsed in
                for event in events {
                    let token = { (text: String) in AppTokenEvent(
                        index: index,
                        textDelta: text,
                        elapsedDecodeSeconds: elapsed)
                    }
                    switch event {
                    case .content(let text):
                        progress.responseText += text
                        continuation.yield(.token(token(text)))
                    case .thinking(let text):
                        progress.thinkingText += text
                        continuation.yield(.thinking(token(text)))
                    case .toolCall(let call):
                        progress.toolCalls.append(AppToolCall(
                            id: call.id, name: call.name, arguments: call.arguments))
                    }
                }
            }

            // A template that rewrites a finished turn when the next one is
            // rendered (GPT-OSS always, Gemma and Qwen with thinking on) is
            // resumed where the user turn began. Tool rounds keep it.
            let capturesTurnCheckpoint = multimodalInput == nil
                && !Self.allowsTextBridge(request)
                && request.assistantPrefix.isEmpty
                && transcript.messages.last?.role == .user
            let turnPosition = multimodalInput == nil && request.assistantPrefix.isEmpty
                && transcript.messages.last?.role == .user
                ? Self.turnCheckpointPosition(
                    request: request, promptIDs: promptIds, tokenizer: tokenizer,
                    modelVariant: variant, harmonyDate: harmonyDate)
                : nil
            // The chat is saved where this user turn begins: the next turn's
            // prompt starts with exactly those tokens whatever the template
            // does to finished turns.
            let chatSavePosition = turnPosition.flatMap { position -> Int? in
                guard let key = request.conversationKey, let savedChats,
                      Self.shouldSaveChat(at: position, saved: savedChats.savedChatTokenCount(
                        conversationKey: key, prompt: promptIds)) else { return nil }
                return position
            }

            completionStarted = true
            let producer: any LogitProducer = sharedForwardLease.map {
                LeasedModelRunner(runner: runner, lease: $0)
            } ?? runner
            let result = try await runRawCompletion(
                producer: producer, tokenizer: tokenizer, promptIds: promptIds,
                multimodalInput: multimodalInput,
                config: config, context: ctx, scratch: scratch,
                prefillConfig: prefillConfig, start: completionStart,
                prefixCheckpointPositions: capturesTurnCheckpoint || chatSavePosition != nil
                    ? [turnPosition].compactMap { $0 }
                    : [],
                stateSnapshotPosition: chatSavePosition,
                checkpointBudgetBytes: max(0, conversations.budgetBytes
                    - conversations.retainedBytes - conversations.activeCheckpointBytes)) { @Sendable event in
                switch event {
                case .prefill(let done, let total):
                    if done == total {
                        progress.decodeStart = Date()
                        progress.countersAtDecodeStart = RunnerCounterSnapshot(runner)
                    }
                    continuation.yield(.prefillProgress(done: done, total: total))
                case .token(let index, let tokenID, let delta):
                    if progress.firstTokenDate == nil { progress.firstTokenDate = Date() }
                    progress.generated = index + 1
                    if index % 8 == 0 { _ = memorySampler.sample() }
                    guard progress.assistantDecodeError == nil else { break }
                    do {
                        publishAssistantEvents(
                            try assistantDecoder.consume(
                                tokenID: tokenID, delta: delta),
                            index,
                            progress.elapsedDecodeSeconds)
                    } catch {
                        progress.assistantDecodeError = error
                    }
                case .tail(let text):
                    guard progress.assistantDecodeError == nil else { break }
                    do {
                        publishAssistantEvents(
                            try assistantDecoder.consumeTail(text),
                            max(progress.generated - 1, 0),
                            progress.elapsedDecodeSeconds)
                    } catch {
                        progress.assistantDecodeError = error
                    }
                }
            }
            // A stop token that closes a structure (Harmony's `<|call|>`) is not
            // reported as `.token`; the decoder must see it before `finish()`.
            if progress.assistantDecodeError == nil {
                do {
                    for tokenID in result.undeliveredBoundaryTokenIDs {
                        publishAssistantEvents(
                            try assistantDecoder.consume(tokenID: tokenID, delta: ""),
                            max(progress.generated - 1, 0), progress.elapsedDecodeSeconds)
                    }
                } catch {
                    progress.assistantDecodeError = error
                }
            }
            if let assistantDecodeError = progress.assistantDecodeError {
                throw Self.structuredFailure(assistantDecodeError)
            }
            // A call still open when the length limit ended generation was cut
            // off, not malformed: nothing of it runs, the answer so far stands
            // as a length stop, and the state holding the partial call is not
            // offered for reuse.
            let truncatedCall = try Self.finish(assistantDecoder, reason: result.reason)
            if truncatedCall { progress.toolCalls = [] }
            if !toolNames.isEmpty, result.reason == .toolCalls, progress.toolCalls.isEmpty {
                throw AppInferenceError.malformedToolCall("a tool response marker without a call")
            }
            // An awaited completion must not publish into a session that was
            // unloaded or replaced while that completion was in flight.
            if self.loadedKey == requestKey, self.runner === runner {
                conversations.publish(conversationTrim.droppedTurns == 0 && !truncatedCall
                    ? ConversationCache.entry(
                        domain: conversationDomain, transcript: transcript,
                        content: request.assistantPrefix + progress.responseText,
                        thinking: progress.thinkingText,
                        calls: progress.toolCalls.map {
                            ParsedToolCall(id: $0.id, name: $0.name, arguments: $0.arguments,
                                           argumentsJSON: "")
                        },
                        result: result,
                        conversationKey: request.conversationKey,
                        prefixCheckpoints: result.prefixCheckpoints.compactMap {
                            ConversationPrefixCheckpoint(
                                tokenIDs: Array(promptIds.prefix($0.position)), snapshot: $0)
                        } + (plannedEntry?.prefixCheckpoints ?? []))
                    : nil)
            }
            if let snapshot = result.stateSnapshot, let key = request.conversationKey,
               let savedChats, conversationTrim.droppedTurns == 0 {
                savedChats.saveChat(snapshot, tokens: Array(promptIds.prefix(snapshot.position)),
                                    domain: conversationDomain, conversationKey: key)
            }
            progress.cachedPromptTokens = result.cachedPromptTokens
            progress.cacheSource = result.cachedPromptTokens > 0 ? cacheSource : .cold

            if !progress.toolCalls.isEmpty {
                continuation.yield(.toolCalls(progress.toolCalls))
            }
            let diagnostics = makeDiagnostics(request: request,
                                              memorySampler: memorySampler,
                                              progress: progress,
                                              stopReason: progress.toolCalls.isEmpty
                                                ? Self.stopReason(result.reason) : .toolCalls,
                                              prefillSeconds: result.prefillSeconds,
                                              decodeSeconds: result.decodeSeconds,
                                              generated: result.newTokens,
                                              prefill: prefillDiagnostics)
            continuation.yield(.finished(diagnostics))
            continuation.finish()
        } catch is CancellationError {
            if completionStarted { conversations.invalidateActive() }
            let diagnostics = makeDiagnostics(request: request,
                                              memorySampler: memorySampler,
                                              progress: progress,
                                              stopReason: .cancelled,
                                              prefillSeconds: progress.elapsedPrefillSeconds,
                                              decodeSeconds: progress.elapsedDecodeSeconds,
                                              generated: progress.generated,
                                              prefill: PrefillExecutionDiagnostics(
                                                config: prefillConfig,
                                                executedMode: prefillConfig.mode == .chunked ? .chunked : .off,
                                                kvStorageMode: .fp16))
            continuation.yield(.cancelled(diagnostics))
            continuation.finish(throwing: AppInferenceError.cancelled)
        } catch let prefillError as PrefillError {
            if completionStarted { conversations.invalidateActive() }
            let diagnostics = Self.prefillFailureDiagnostics(config: prefillConfig,
                                                             kvStorageMode: .fp16,
                                                             reason: prefillError.description)
            failGeneration(.unknown(prefillError.description),
                           request: request,
                           memorySampler: memorySampler,
                           progress: progress,
                           continuation: continuation,
                           prefill: diagnostics,
                           forcePartialDiagnostics: true)
        } catch let appError as AppInferenceError {
            if completionStarted { conversations.invalidateActive() }
            failGeneration(appError, request: request, memorySampler: memorySampler,
                           progress: progress, continuation: continuation)
        } catch {
            if completionStarted { conversations.invalidateActive() }
            failGeneration(.unknown("\(error)"), request: request, memorySampler: memorySampler,
                           progress: progress, continuation: continuation)
        }
    }

    /// Finishes structured decoding. Returns true when a tool call was still
    /// open as the length limit ended generation: a truncation, reported as a
    /// length stop with no calls run. Any other open or broken call throws.
    static func finish(_ decoder: StructuredAssistantDecoder, reason: StopReason) throws -> Bool {
        do {
            try decoder.finish()
            return false
        } catch ToolCallParserError.malformed where reason == .maxTokens {
            return true
        } catch {
            throw structuredFailure(error)
        }
    }

    /// Parser failures are tool-call failures the app can retry; anything else
    /// stays a generation error.
    static func structuredFailure(_ error: Error) -> Error {
        switch error {
        case let parser as ToolCallParserError:
            switch parser {
            case .unknownTool(let name):
                return AppInferenceError.malformedToolCall("unknown tool \(name.prefix(40))")
            case .malformed:
                return AppInferenceError.malformedToolCall("malformed call")
            case .oversized:
                return AppInferenceError.malformedToolCall("call larger than the parser accepts")
            }
        default:
            return error
        }
    }

    private func failGeneration(_ error: AppInferenceError,
                                request: AppGenerationRequest,
                                memorySampler: AppMemorySampler,
                                progress: ProgressState,
                                continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation,
                                prefill: PrefillExecutionDiagnostics? = nil,
                                forcePartialDiagnostics: Bool = false) {
        let partial = progress.generated > 0 || forcePartialDiagnostics
            ? makeDiagnostics(request: request, memorySampler: memorySampler,
                              progress: progress, stopReason: .failed,
                              prefillSeconds: progress.elapsedPrefillSeconds,
                              decodeSeconds: progress.elapsedDecodeSeconds,
                              generated: progress.generated,
                              prefill: prefill)
            : nil
        continuation.yield(.failed(error, partial: partial))
        continuation.finish(throwing: error)
    }

    private func makeDiagnostics(request: AppGenerationRequest,
                                 memorySampler: AppMemorySampler,
                                 progress: ProgressState,
                                 stopReason: AppStopReason,
                                 prefillSeconds: Double? = nil,
                                 decodeSeconds: Double,
                                 generated: Int,
                                 prefill: PrefillExecutionDiagnostics? = nil) -> AppDiagnostics {
        _ = memorySampler.sample()
        let ttft: Double?
        if let first = progress.firstTokenDate, let start = progress.decodeStart {
            ttft = first.timeIntervalSince(start)
        } else {
            ttft = nil
        }
        var diagnostics = AppDiagnostics(
            generatedTokens: generated,
            stopReason: stopReason,
            promptTokenCount: progress.promptTokenCount,
            prefillSeconds: prefillSeconds,
            timeToFirstTokenSeconds: ttft,
            decodeSeconds: decodeSeconds,
            tokensPerSecond: decodeSeconds > 0 ? Double(generated) / decodeSeconds : 0,
            peakMemoryBytes: memorySampler.peakBytes,
            visionTowerMappedBytes: visionRuntime.map { UInt64($0.retainedWeightBytes) },
            runtimeOptions: request.runtimeOptions,
            prefill: prefill,
            runner: runnerDiagnostics(progress: progress, generated: generated),
            droppedTurns: progress.conversationTrim?.droppedTurns ?? 0)
        let statistics = conversations.statistics
        diagnostics.cachedPromptTokens = progress.cachedPromptTokens
        diagnostics.conversationCacheSource = progress.cachedPromptTokens == nil
            ? nil : progress.cacheSource.rawValue
        diagnostics.retainedConversations = statistics.retainedConversations
        diagnostics.retainedConversationBytes = statistics.retainedBytes
        return diagnostics
    }

    /// Per-token buckets as diffs of the runner's cumulative counters from the
    /// decode start (excludes prefill), divided by the decode forward count.
    /// The forward count is `generated - 1`: each loop iteration that continues
    /// ends with one `produce`; the final sampled token never runs a forward.
    private func runnerDiagnostics(progress: ProgressState, generated: Int) -> AppRunnerDiagnostics? {
        guard let runner, let base = progress.countersAtDecodeStart, generated > 1 else { return nil }
        let now = RunnerCounterSnapshot(runner)
        let forwards = Double(generated - 1)
        func ms(_ end: UInt64, _ start: UInt64) -> Double {
            Double(end &- start) / 1_000_000 / forwards
        }
        return AppRunnerDiagnostics(
            cb1MillisecondsPerToken: ms(now.cb1, base.cb1),
            ioMillisecondsPerToken: ms(now.io, base.io),
            cb2MillisecondsPerToken: ms(now.cb2, base.cb2),
            headMillisecondsPerToken: ms(now.head, base.head),
            rdadviseMillisecondsPerToken: ms(now.rdadvise, base.rdadvise),
            rdadviseCallsPerToken: Double(now.rdadviseCalls &- base.rdadviseCalls) / forwards,
            rdadviseMegabytesPerToken: Double(now.rdadviseBytes &- base.rdadviseBytes) / 1_048_576.0 / forwards,
            rdadviseSkippedPerToken: Double(now.rdadviseSkipped &- base.rdadviseSkipped) / forwards,
            rdadviseFailures: now.rdadviseFailures &- base.rdadviseFailures,
            expertReads: now.expertReads.subtracting(base.expertReads),
            exposedPrefetchWaitMillisecondsPerToken: ms(now.prefetchWait, base.prefetchWait))
    }

    private static func stopReason(_ reason: StopReason) -> AppStopReason {
        switch reason {
        case .eos: return .eos
        case .endOfTurn: return .endOfTurn
        case .maxTokens: return .maxTokens
        case .stopString: return .stopString
        case .cancelled: return .cancelled
        case .toolCalls: return .toolCalls
        }
    }

    internal static func prefillFailureDiagnostics(config: PrefillRuntimeConfig,
                                                   kvStorageMode: PrefillKVStorageMode,
                                                   reason: String) -> PrefillExecutionDiagnostics {
        PrefillExecutionDiagnostics.unsupported(config: config,
                                                kvStorageMode: kvStorageMode,
                                                reason: reason)
    }
}

/// Mutable per-generation state shared between the progress callback and the
/// surrounding actor method. Access is sequential: the callback runs synchronously
/// inside `runRawCompletion` while the actor method awaits it.
private final class ProgressState: @unchecked Sendable {
    var generated = 0
    var promptTokenCount: Int?
    var conversationTrim: AppConversationTrim?
    var prefillStart: Date?
    var decodeStart: Date?
    var firstTokenDate: Date?
    var countersAtDecodeStart: RunnerCounterSnapshot?
    var assistantDecodeError: Error?
    var responseText = ""
    var thinkingText = ""
    var toolCalls: [AppToolCall] = []
    var cachedPromptTokens: Int?
    var cacheSource: ConversationStateStore.Source = .cold

    var elapsedDecodeSeconds: Double {
        guard let decodeStart else { return 0 }
        return Date().timeIntervalSince(decodeStart)
    }

    var elapsedPrefillSeconds: Double? {
        guard let prefillStart else { return nil }
        let end = decodeStart ?? Date()
        return max(end.timeIntervalSince(prefillStart), 0)
    }
}

private struct RunnerCounterSnapshot {
    let expertReads: ExpertReadMetrics
    let prefetchWait: UInt64
    let cb1: UInt64
    let io: UInt64
    let cb2: UInt64
    let head: UInt64
    let rdadvise: UInt64
    let rdadviseCalls: UInt64
    let rdadviseBytes: UInt64
    let rdadviseFailures: UInt64
    let rdadviseSkipped: UInt64

    init(_ runner: ModelForwardRunner) {
        expertReads = runner.expertReadMetrics
        prefetchWait = runner.exposedPrefetchWaitNanos
        cb1 = runner.totalCb1Nanos
        io = runner.totalIoNanos
        cb2 = runner.totalCb2Nanos
        head = runner.totalHeadNanos &+ runner.totalHeadFusedNanos
        rdadvise = runner.totalRDAdviseNanos
        rdadviseCalls = runner.totalRDAdviseCalls
        rdadviseBytes = runner.totalRDAdviseBytes
        rdadviseFailures = runner.totalRDAdviseFailures
        rdadviseSkipped = runner.totalRDAdviseSkipped
    }
}
