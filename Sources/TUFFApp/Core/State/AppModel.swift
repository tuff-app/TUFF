import Foundation
import Synchronization
import TUFFModelCatalog
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFModelMetadata
import TUFFRepackCore
import TUFFRuntimeManagement
import Observation

@MainActor
@Observable
public final class AppModel {
    public private(set) var customCheckpointImportError: String?
    public private(set) var isImportingCustomCheckpoint = false
    public var customCheckpointInstalls: [ModelInstallCoordinator] {
        installs.filter { $0.descriptor.isCustomCheckpoint }
    }

    public func importCustomCheckpoint(directory: URL, name: String, templateID: TUFFModelID) async {
        guard !isImportingCustomCheckpoint, !isRunning, !loadState.isLoading else { return }
        isImportingCustomCheckpoint = true
        customCheckpointImportError = nil
        defer { isImportingCustomCheckpoint = false }
        do {
            let sourcePath = directory.standardizedFileURL.resolvingSymlinksInPath().path
            guard !(try AppCustomCheckpointStore.load()).contains(where: { $0.sourcePath == sourcePath }) else {
                throw AppCustomCheckpoint.ImportError("This checkpoint is already registered.")
            }
            let record = try await Task.detached(priority: .utility) {
                try AppCustomCheckpoint.inspect(directory: directory, name: name, templateID: templateID)
            }.value
            guard !isRunning, !loadState.isLoading else {
                throw AppCustomCheckpoint.ImportError("Finish the current operation before importing.")
            }
            var records = try AppCustomCheckpointStore.load()
            guard !records.contains(where: { $0.sourcePath == record.sourcePath }) else {
                throw AppCustomCheckpoint.ImportError("This checkpoint is already registered.")
            }
            let alias = AppCustomCheckpointStore.directory.appendingPathComponent(record.descriptor.installDirectoryName)
            try FileManager.default.createDirectory(at: AppCustomCheckpointStore.directory, withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(atPath: alias.path, withDestinationPath: record.sourcePath)
            do {
                let engine = try await InferenceEngineEnvironment.acquireModel(at: alias)
                await engine.release()
                guard !isRunning, !loadState.isLoading else {
                    throw AppCustomCheckpoint.ImportError("Finish the current operation before importing.")
                }
                records.append(record)
                try AppCustomCheckpointStore.save(records)
            } catch {
                try? await InferenceEngineEnvironment.unbindModel(at: alias)
                try? FileManager.default.removeItem(at: alias)
                throw error
            }
            let coordinator = ModelInstallCoordinator(descriptor: record.descriptor, directoryURL: alias,
                client: AppCustomCheckpointInstaller(descriptor: record.descriptor))
            installs.append(coordinator)
            updateSettingsProfile(.defaults(for: record.templateID.rawValue), for: coordinator)
            persistSettings()
        } catch { customCheckpointImportError = String(describing: error) }
    }

    public func removeCustomCheckpoint(_ coordinator: ModelInstallCoordinator) {
        guard coordinator.descriptor.isCustomCheckpoint else { return }
        guard !isRunning, !loadState.isLoading,
              coordinator.id != selectedModelID || !loadState.isReady else {
            customCheckpointImportError = "Unload the checkpoint before removing its entry."
            return
        }
        do {
            var records = try AppCustomCheckpointStore.load()
            guard let record = records.first(where: { $0.descriptor.id == coordinator.id }) else { return }
            let alias = AppCustomCheckpointStore.directory.appendingPathComponent(record.descriptor.installDirectoryName)
            guard try FileManager.default.destinationOfSymbolicLink(atPath: alias.path) == record.sourcePath else {
                throw AppCustomCheckpoint.ImportError("The registered checkpoint link changed; its files were left untouched.")
            }
            records.removeAll { $0.id == record.id }
            try AppCustomCheckpointStore.save(records)
            try FileManager.default.removeItem(at: alias)
            if coordinator.id == selectedModelID,
               let fallback = installs.first(where: { !$0.descriptor.isCustomCheckpoint && $0.isInstalled })
                    ?? installs.first(where: { !$0.descriptor.isCustomCheckpoint }) {
                selectModel(fallback)
            }
            installs.removeAll { $0.id == coordinator.id }
            Task { try? await InferenceEngineEnvironment.unbindModel(at: alias) }
            customCheckpointImportError = nil
        } catch { customCheckpointImportError = String(describing: error) }
    }

    public typealias RunState = AppRunState

    public let conversationStore: AppConversationStore
    public let modelLibraryStore = AppModelLibraryStore()
    public let settingsStore = AppSettingsStore()
    public let inferenceStore = AppSharedInferenceStore()
    public let networkStatus: AppNetworkStatus
    /// Folders, the local index, provider keys and the tools built from them.
    public let toolStore: AppToolStore
    #if os(macOS)
    /// The web sandbox: started and stopped here, used only through the
    /// tool store's sandbox access.
    @ObservationIgnored public private(set) var sandbox: SandboxController
    @ObservationIgnored private var sandboxTask: Task<Void, Never>?
    /// The files sandbox: a second VM with no network that reads the folders
    /// chosen for file search. Started when indexing needs it, never for
    /// anything else.
    @ObservationIgnored public private(set) var filesSandbox: SandboxController
    @ObservationIgnored private var filesSandboxTask: Task<Void, Never>?
    #endif
    public let deviceCapabilities: TUFFDeviceCapabilities
    private var runsAfterCurrentLoad = false
    private var autoWorkloadKey = "text-cold-short-t0"


    public var modelPathText: String {
        get { modelLibraryStore.modelPathText }
        set { modelLibraryStore.modelPathText = newValue }
    }
    public var promptText: String {
        get { conversationStore.promptText }
        set { conversationStore.promptText = newValue }
    }
    public private(set) var imageAttachments: [AppImageAttachment] {
        get { conversationStore.imageAttachments }
        set { conversationStore.imageAttachments = newValue }
    }
    public private(set) var imageAttachmentError: String? {
        get { conversationStore.imageAttachmentError }
        set { conversationStore.imageAttachmentError = newValue }
    }
    /// A count, not a flag. The picker and a drop can both be staging at once,
    /// and whichever finished first cleared a shared Bool — reopening `canRun`
    /// while the other was still copying, so Generate ran against a partial set
    /// and the remaining images were appended after the run had snapshotted its
    /// own, where `removeImage` and `clearImages` are no-ops.
    private var addingImagesCount: Int {
        get { conversationStore.addingImagesCount }
        set { conversationStore.addingImagesCount = newValue }
    }
    public var isAddingImages: Bool { addingImagesCount > 0 }
    /// Set by the Model menu's Remove Image Support item; the window presents
    /// the confirmation.
    public var isConfirmingVisionPackRemoval: Bool {
        get { modelLibraryStore.isConfirmingVisionPackRemoval }
        set { modelLibraryStore.isConfirmingVisionPackRemoval = newValue }
    }
    public private(set) var outputPromptText: String {
        get { conversationStore.outputPromptText }
        set { conversationStore.outputPromptText = newValue }
    }
    public private(set) var outputImageAttachments: [AppImageAttachment] {
        get { conversationStore.outputImageAttachments }
        set { conversationStore.outputImageAttachments = newValue }
    }
    public var outputText: String {
        get { conversationStore.outputText }
        set { conversationStore.outputText = newValue }
    }
    public private(set) var outputThinkingText: String {
        get { conversationStore.outputThinkingText }
        set { conversationStore.outputThinkingText = newValue }
    }
    /// Completed exchanges, oldest first. The in-flight exchange lives in
    /// `outputPromptText` / `outputText` and only joins this once it finishes.
    public private(set) var conversation: [AppChatTurn] {
        get { conversationStore.conversation }
        set { conversationStore.conversation = newValue }
    }
    public var runState: RunState {
        get { inferenceStore.runState }
        set { inferenceStore.runState = newValue }
    }
    public var runtimeOptions: AppRuntimeOptions {
        get { settingsStore.runtimeOptions }
        set { settingsStore.runtimeOptions = newValue }
    }
    public var automaticMemory: Bool {
        get { settingsStore.automaticMemory }
        set { settingsStore.automaticMemory = newValue }
    }
    public var automaticMemoryProfile: AppAutomaticMemoryProfile {
        get { settingsStore.automaticMemoryProfile }
        set { settingsStore.automaticMemoryProfile = newValue }
    }
    /// Lets this Mac install and load models its measured capabilities refuse,
    /// and load a model at manual settings above Auto's safe budget.
    public var bypassModelRestrictions: Bool {
        get { settingsStore.bypassModelRestrictions }
        set {
            guard settingsStore.bypassModelRestrictions != newValue else { return }
            settingsStore.bypassModelRestrictions = newValue
            persistSettings()
        }
    }
    public var maxNewTokensOverride: Int? {
        get { settingsStore.maxNewTokensOverride }
        set { settingsStore.maxNewTokensOverride = newValue }
    }
    public var reasoning: ChatReasoning {
        get { settingsStore.reasoning }
        set { settingsStore.reasoning = newValue }
    }
    public var reasoningEffort: GPTOSSReasoningEffort {
        get { settingsStore.reasoningEffort }
        set { settingsStore.reasoningEffort = newValue }
    }
    public var preserveThinking: Bool {
        get { settingsStore.preserveThinking }
        set { settingsStore.preserveThinking = newValue }
    }
    public var maxContextTokens: Int {
        get { settingsStore.maxContextTokens }
        set { settingsStore.maxContextTokens = newValue }
    }
    public var temperature: Double {
        get { settingsStore.temperature }
        set { settingsStore.temperature = newValue }
    }
    public var topKEnabled: Bool {
        get { settingsStore.topKEnabled }
        set { settingsStore.topKEnabled = newValue }
    }
    public var topK: Int {
        get { settingsStore.topK }
        set { settingsStore.topK = newValue }
    }
    public var topPEnabled: Bool {
        get { settingsStore.topPEnabled }
        set { settingsStore.topPEnabled = newValue }
    }
    public var topP: Double {
        get { settingsStore.topP }
        set { settingsStore.topP = newValue }
    }
    // App-wide settings persist from their setters so views can bind directly.
    public var newlineShortcut: AppNewlineShortcut {
        get { settingsStore.newlineShortcut }
        set {
            guard settingsStore.newlineShortcut != newValue else { return }
            settingsStore.newlineShortcut = newValue
            persistSettings()
        }
    }
    public var smoothStreaming: Bool {
        get { settingsStore.smoothStreaming }
        set {
            guard settingsStore.smoothStreaming != newValue else { return }
            settingsStore.smoothStreaming = newValue
            persistSettings()
        }
    }
    public var showPromptExamples: Bool {
        get { settingsStore.showPromptExamples }
        set {
            guard settingsStore.showPromptExamples != newValue else { return }
            settingsStore.showPromptExamples = newValue
            persistSettings()
        }
    }
    public var sentPromptBehavior: AppSentPromptBehavior {
        get { settingsStore.sentPromptBehavior }
        set {
            guard settingsStore.sentPromptBehavior != newValue else { return }
            settingsStore.sentPromptBehavior = newValue
            persistSettings()
        }
    }
    /// Whether launching the app should load the model straight away. Off by
    /// default, because loading takes minutes and holds gigabytes.
    public var loadModelOnLaunch: Bool {
        get { settingsStore.loadModelOnLaunch }
        set {
            guard settingsStore.loadModelOnLaunch != newValue else { return }
            settingsStore.loadModelOnLaunch = newValue
            persistSettings()
        }
    }
    public var translucentWindow: Bool {
        get { settingsStore.translucentWindow }
        set {
            guard settingsStore.translucentWindow != newValue else { return }
            settingsStore.translucentWindow = newValue
            persistSettings()
        }
    }
    public var appearanceMode: AppAppearanceMode {
        get { settingsStore.appearanceMode }
        set {
            guard settingsStore.appearanceMode != newValue else { return }
            settingsStore.appearanceMode = newValue
            persistSettings()
        }
    }
    public var accentColorMode: AppAccentColorMode {
        get { settingsStore.accentColorMode }
        set {
            guard settingsStore.accentColorMode != newValue else { return }
            settingsStore.accentColorMode = newValue
            persistSettings()
        }
    }
    /// Updated through `setCustomAccentColorHex`, which rejects malformed input.
    public private(set) var customAccentColorHex: String {
        get { settingsStore.customAccentColorHex }
        set { settingsStore.customAccentColorHex = newValue }
    }
    public var zoomLevel: AppZoomLevel {
        get { settingsStore.zoomLevel }
        set {
            guard settingsStore.zoomLevel != newValue else { return }
            settingsStore.zoomLevel = newValue
            persistSettings()
        }
    }
    public var diagnostics: AppDiagnostics? {
        get { inferenceStore.diagnostics }
        set { inferenceStore.diagnostics = newValue }
    }
    public var error: AppInferenceError? {
        get { inferenceStore.error }
        set { inferenceStore.error = newValue }
    }
    /// One coordinator per catalog model. Each owns its own directory,
    /// installer, and progress, so their downloads run independently of each
    /// other and of whichever model is selected.
    public private(set) var installs: [ModelInstallCoordinator] {
        get { modelLibraryStore.installs }
        set { modelLibraryStore.installs = newValue }
    }
    /// `AppModelInstallDescriptor.id` of the model the app is focused on: the
    /// one it will load, and the one the single-model properties below report.
    public private(set) var selectedModelID: String {
        get { modelLibraryStore.selectedModelID }
        set { modelLibraryStore.selectedModelID = newValue }
    }
    /// The companion download is 1.5 GB and deserves the same answer to "how
    /// long is this going to take" as the model download. Kept separate because
    /// both can be in flight in principle and an estimator holds per-download
    /// rate state.
    public private(set) var visionInstallETAPresentation: DownloadETAPresentation {
        get { modelLibraryStore.visionInstallETAPresentation }
        set { modelLibraryStore.visionInstallETAPresentation = newValue }
    }
    public private(set) var visionInstallETAText: String? {
        get { modelLibraryStore.visionInstallETAText }
        set { modelLibraryStore.visionInstallETAText = newValue }
    }
    public var visionInstallState: AppModelInstallState {
        get { modelLibraryStore.visionInstallState }
        set { modelLibraryStore.visionInstallState = newValue }
    }
    /// How far activation's hash of the companion weights has got, 0 to 1.
    /// Activation reads about 1.5 GB, which was a bare spinner with no way to
    /// tell a slow verify from a stuck one.
    public private(set) var visionActivationProgress: Double? {
        get { modelLibraryStore.visionActivationProgress }
        set { modelLibraryStore.visionActivationProgress = newValue }
    }
    public private(set) var visionInstallReadiness: AppModelInstallReadiness {
        get { modelLibraryStore.visionInstallReadiness }
        set { modelLibraryStore.visionInstallReadiness = newValue }
    }
    public private(set) var visionInstallationStatus: AppVisionPackInstallationStatus {
        get { modelLibraryStore.visionInstallationStatus }
        set { modelLibraryStore.visionInstallationStatus = newValue }
    }

    public var loadState: AppModelLoadState {
        get { inferenceStore.loadState }
        set { inferenceStore.loadState = newValue }
    }
    public private(set) var loadedRuntimeKey: AppLoadedRuntimeKey? {
        get { inferenceStore.loadedRuntimeKey }
        set { inferenceStore.loadedRuntimeKey = newValue }
    }
    public private(set) var phase: AppGenerationPhase {
        get { inferenceStore.phase }
        set { inferenceStore.phase = newValue }
    }
    public private(set) var liveTokenCount: Int {
        get { inferenceStore.liveTokenCount }
        set { inferenceStore.liveTokenCount = newValue }
    }
    public private(set) var liveElapsedDecodeSeconds: Double {
        get { inferenceStore.liveElapsedDecodeSeconds }
        set { inferenceStore.liveElapsedDecodeSeconds = newValue }
    }
    public private(set) var livePrefillDone: Int {
        get { inferenceStore.livePrefillDone }
        set { inferenceStore.livePrefillDone = newValue }
    }
    public private(set) var livePrefillTotal: Int {
        get { inferenceStore.livePrefillTotal }
        set { inferenceStore.livePrefillTotal = newValue }
    }
    public private(set) var liveMemoryBytes: UInt64? {
        get { inferenceStore.liveMemoryBytes }
        set { inferenceStore.liveMemoryBytes = newValue }
    }
    /// Resident bytes of the inference process. The footprint above is what
    /// the system counts against the process; this is what it actually holds,
    /// including the mapped weights the footprint omits. A 26B model reports
    /// about 160 MB of footprint right after loading, which is true and reads
    /// as nonsense without this beside it.
    public private(set) var liveResidentBytes: UInt64? {
        get { inferenceStore.liveResidentBytes }
        set { inferenceStore.liveResidentBytes = newValue }
    }
    /// Tower weights the inference process is holding mapped, reported
    /// separately because no per-process counter attributes them.
    public private(set) var visionTowerMappedBytes: UInt64? {
        get { inferenceStore.visionTowerMappedBytes }
        set { inferenceStore.visionTowerMappedBytes = newValue }
    }
    public private(set) var isCancellationPending: Bool {
        get {
            guard let id = conversationStore.selectedConversationID else { return false }
            return activeAnswerRuns[id]?.cancellationPending ?? false
        }
        set {
            guard let id = conversationStore.selectedConversationID,
                  let run = activeAnswerRuns[id] else { return }
            run.cancellationPending = newValue
            inferenceStore.isCancellationPending = newValue
        }
    }
    /// Increments when a generation starts. The transcript watches it to put
    /// the newest turn on screen: with several images attached, the prompt and
    /// its thumbnails are tall enough to push the answer out of view, so
    /// scrolling only when the reader was already at the bottom left them
    /// looking at their own attachments while the model worked.
    public private(set) var runIdentity: Int {
        get { conversationStore.runIdentity }
        set { conversationStore.runIdentity = newValue }
    }

    private let client: any AppInferenceClient
    private let visionInstaller: any AppVisionPackInstallerClient
    private let deletionCleanup: AppModelDeletionCleanup
    /// Models being deleted, so a second click cannot start another delete.
    public private(set) var deletingModelIDs: Set<String> = []
    private final class ActiveAnswerRun {
        let conversationID: UUID
        let runtimeKey: AppLoadedRuntimeKey
        let modelProfileKey: String
        let scopedClient: any AppInferenceClient
        let cancellation = AppAnswerCancellation()
        var task: Task<Void, Never>?
        var currentGenerationThinking = ""
        var hasHandledTerminalEvent = false
        var cancellationPending = false
        var observationStart: AppAutoObservationWindow?
        var phase: AppGenerationPhase = .prefill
        var tokenCount = 0
        var elapsedDecodeSeconds = 0.0
        var prefillDone = 0
        var prefillTotal = 0

        init(conversationID: UUID, runtimeKey: AppLoadedRuntimeKey,
             modelProfileKey: String, scopedClient: any AppInferenceClient) {
            self.conversationID = conversationID
            self.runtimeKey = runtimeKey
            self.modelProfileKey = modelProfileKey
            self.scopedClient = scopedClient
        }
    }
    private var activeAnswerRuns: [UUID: ActiveAnswerRun] = [:]
    private var sessionCleanups: [UUID: Task<Void, Never>] = [:]
    #if os(macOS)
    private var sessionPressureSource: DispatchSourceMemoryPressure?
    #endif
    /// Set only while one answer event is synchronously applied on MainActor.
    private var handlingAnswerConversationID: UUID?
    private var loadTask: Task<Void, Never>?
    private var visionInstallTask: Task<Void, Never>?
    private var unloadTask: Task<Void, Never>?
    private var loadGeneration: UInt64 = 0
    /// The highest load-phase sequence already applied. Each `onState` callback
    /// hops to the main actor in its own task, and ordering between separately
    /// created tasks is not guaranteed, so `.ready` could be applied before the
    /// `.loading(.preparingRunner)` that preceded it and leave the UI showing a
    /// phase the runtime had already left.
    private var appliedLoadSequence: UInt64 = 0
    private var unloadGeneration: UInt64 = 0
    private var visionInstallGeneration: UInt64 = 0
    private var visionInstallCancellationRequested = false
    /// The catalog model whose optional companion owns the current vision
    /// transaction. This is deliberately separate from `selectedModelID`:
    /// downloading Qwen image support must not silently switch away from a
    /// loaded Gemma text session (or vice versa).
    public private(set) var visionInstallTargetModelID: String? {
        get { modelLibraryStore.visionInstallTargetModelID }
        set { modelLibraryStore.visionInstallTargetModelID = newValue }
    }
    private var visionInstallTargetDirectory: URL?
    private var pendingExplicitLoadRuntimeKey: AppLoadedRuntimeKey?
    private let memorySampler: AppMemorySampler
    private let settingsPersistenceEnabled: Bool
    /// The debounced settings write. See `persistSettingsSoon`.
    @ObservationIgnored private var pendingSettingsSave: Task<Void, Never>?
    /// Non-selected profiles waiting for the next coalesced save.
    @ObservationIgnored private var unsavedProfileKeys: Set<String> = []
    private let installETAClock: SuspendingClock
    private let installETAOrigin: SuspendingClock.Instant
    private var visionInstallETAEstimator = DownloadETAEstimator()
    private let attachmentStore: AppImageAttachmentStore
    public let isVisionRuntimeSupported: Bool

    public static var currentDeviceSupportsVisionRuntime: Bool {
        TUFFDeviceCapabilities.current().appleSiliconGeneration >= 2
    }

    /// `otherInstalls` replaces the coordinators the catalog would build for
    /// the models `installer` does not cover. Callers that want the shipped
    /// catalog leave it nil; tests pass their own so no catalog model reaches
    /// the network or a real install directory.
    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = DecodeServiceInferenceClient(),
                installer: any AppModelInstallerClient = RepackModelInstallerClient(descriptor: .default),
                otherInstalls: [ModelInstallCoordinator]? = nil,
                visionInstaller: any AppVisionPackInstallerClient = RepackVisionPackInstallerClient(),
                memorySampler: AppMemorySampler = AppMemorySampler(),
                attachmentStore: AppImageAttachmentStore = AppImageAttachmentStore(),
                conversationStore: AppConversationStore = AppConversationStore(),
                visionRuntimeSupported: Bool = true,
                settingsPersistenceEnabled: Bool = false,
                deviceCapabilities: TUFFDeviceCapabilities = .current(),
                toolStore: AppToolStore? = nil,
                networkStatus: AppNetworkStatus? = nil,
                deletionCleanup: AppModelDeletionCleanup? = nil) {
        let directory = (modelDirectory ?? AppModelLocation.defaultURL()).standardizedFileURL
        let installETAClock = SuspendingClock()
        let settingsProfileKey = installer.descriptor.settingsProfileKey
        let settings = settingsPersistenceEnabled
            ? MacAppSettingsFileStore.loadOrCreate(
                forModelDirectory: directory,
                profileKey: settingsProfileKey)
            : MacAppSettings(modelProfiles: [
                settingsProfileKey: .defaults(for: settingsProfileKey)
            ])
        let savedModelSettings = settings.profile(for: settingsProfileKey)
        let modelSettings = AppAutomaticMemoryPlanner.applying(
            savedModelSettings,
            for: installer.descriptor,
            on: deviceCapabilities, modelDirectory: directory, useBundledLayout: settingsPersistenceEnabled)
        // Initialize AppModel's stored dependencies before writing through the
        // domain-store proxy properties below. Swift treats those computed
        // properties as accesses to self even though each store already has a
        // default value.
        self.conversationStore = conversationStore
        let tools = toolStore ?? .inMemory()
        self.toolStore = tools
        #if os(macOS)
        let sandbox = SandboxController()
        self.sandbox = sandbox
        let liveSettings = self.settingsStore
        tools.sandboxWeb = SandboxWebAccess(
            controller: sandbox, mayRestart: { await liveSettings.runWebInSandbox })
        let filesSandbox = SandboxController(dependencies: .live(profile: .files))
        self.filesSandbox = filesSandbox
        tools.sandboxFiles = SandboxFilesService(
            controller: filesSandbox,
            access: SandboxWebAccess(controller: filesSandbox,
                                     mayRestart: { await liveSettings.runFilesInSandbox }))
        #endif
        self.networkStatus = networkStatus ?? AppNetworkStatus(monitorsConnectivity: settingsPersistenceEnabled)
        self.deviceCapabilities = deviceCapabilities
        self.client = client
        self.visionInstaller = visionInstaller
        self.deletionCleanup = deletionCleanup ?? (settingsPersistenceEnabled ? .standard : .none)
        self.memorySampler = memorySampler
        self.attachmentStore = attachmentStore
        self.isVisionRuntimeSupported = visionRuntimeSupported
        self.settingsPersistenceEnabled = settingsPersistenceEnabled
        self.installETAClock = installETAClock
        self.installETAOrigin = installETAClock.now
        self.modelPathText = directory.path
        // The app always releases the image tower after each image. Keeping it
        // resident saves a few hundred milliseconds on a run of images and
        // holds about 1 GB of page cache to do it — a trade worth exposing to
        // a CLI or server operator, not to someone using the app, where it was
        // one more setting whose effect no figure on screen could show.
        // `keepReady` remains available through AppRuntimeOptions for those.
        let initialPlan = modelSettings.automaticMemory ? AppAutomaticMemoryPlanner.plan(
            for: installer.descriptor, on: deviceCapabilities, profile: modelSettings.automaticMemoryProfile,
            modelDirectory: directory, useBundledLayout: settingsPersistenceEnabled) : nil
        self.runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: modelSettings.expertCacheSlots,
            prefillEnabled: modelSettings.prefillEnabled,
            prefillChunkTokens: initialPlan?.prefillChunkTokens ?? installer.descriptor
                .recommendedPrefillChunkTokens(on: deviceCapabilities),
            rdadvisePolicy: modelSettings.rdadvisePolicy,
            visionResidencyPolicy: .onDemand,
            denseResidencyPolicy: initialPlan?.denseResidencyPolicy ?? modelSettings.manualDenseResidencyPolicy(for: installer.descriptor),
            retainedConversationBytes: initialPlan?.retainedConversationBytes)
        self.automaticMemory = modelSettings.automaticMemory
        self.automaticMemoryProfile = modelSettings.automaticMemoryProfile
        self.maxContextTokens = modelSettings.contextTokens
        self.temperature = modelSettings.temperature
        self.topKEnabled = modelSettings.topKEnabled
        self.topK = modelSettings.topK
        self.topPEnabled = modelSettings.topPEnabled
        self.topP = modelSettings.topP
        self.reasoning = installer.descriptor.reasoningControl == .alwaysOn
            ? .on : modelSettings.defaultReasoning
        self.reasoningEffort = modelSettings.defaultReasoningEffort
        self.preserveThinking = modelSettings.preserveThinking
        self.settingsStore.systemPrompt = modelSettings.systemPrompt
        self.settingsStore.newlineShortcut = settings.newlineShortcut
        self.settingsStore.showPromptExamples = settings.showPromptExamples
        self.settingsStore.sentPromptBehavior = settings.sentPromptBehavior
        self.settingsStore.loadModelOnLaunch = settings.loadModelOnLaunch
        self.settingsStore.smoothStreaming = settings.smoothStreaming
        self.settingsStore.appearanceMode = settings.appearanceMode
        self.settingsStore.translucentWindow = settings.translucentWindow
        self.settingsStore.accentColorMode = settings.accentColorMode
        self.customAccentColorHex = settings.customAccentColorHex
        self.settingsStore.zoomLevel = settings.zoomLevel
        self.settingsStore.bypassModelRestrictions = settings.bypassModelRestrictions
        self.settingsStore.webSearchEnabled = settings.webSearchEnabled
        self.settingsStore.pauseWebSearchWhenOffline = settings.pauseWebSearchWhenOffline
        self.settingsStore.runWebInSandbox = settings.runWebInSandbox
        self.settingsStore.runFilesInSandbox = settings.runFilesInSandbox
        self.toolStore.indexInSandbox = settings.runFilesInSandbox
        self.settingsStore.fileSearchEnabled = settings.fileSearchEnabled
        self.settingsStore.codingWriteEnabled = settings.codingWriteEnabled
        self.settingsStore.codingCommandEnabled = settings.codingCommandEnabled
        self.settingsStore.codingNetworkEnabled = settings.codingNetworkEnabled
        self.settingsStore.codingFolderID = settings.codingFolderID
        self.settingsStore.searchProvider = settings.searchProvider
        self.visionInstallationStatus = AppVisionPackInstallationProbe.status(at: directory)

        // The injected installer owns the passed-in directory and becomes the
        // selected model; the rest of the catalog gets shipped coordinators at
        // their own default locations. Catalog order drives the UI listing.
        let selected = ModelInstallCoordinator(
            descriptor: installer.descriptor,
            directoryURL: directory,
            client: installer)
        self.selectedModelID = selected.id
        if let otherInstalls {
            self.installs = [selected] + otherInstalls.filter { $0.id != selected.id }
        } else {
            self.installs = AppModelInstallDescriptor.catalog.map { candidate in
                candidate.id == selected.id ? selected : .shipped(candidate)
            }
            if !self.installs.contains(where: { $0.id == selected.id }) {
                self.installs.insert(selected, at: 0)
            }
        }
        if settingsPersistenceEnabled {
            do {
                for record in try AppCustomCheckpointStore.load() {
                    let alias = AppCustomCheckpointStore.directory.appendingPathComponent(record.descriptor.installDirectoryName)
                    if !FileManager.default.fileExists(atPath: alias.path),
                       (try? FileManager.default.destinationOfSymbolicLink(atPath: alias.path)) == nil {
                        try FileManager.default.createSymbolicLink(atPath: alias.path, withDestinationPath: record.sourcePath)
                    }
                    self.installs.append(ModelInstallCoordinator(descriptor: record.descriptor, directoryURL: alias,
                        client: AppCustomCheckpointInstaller(descriptor: record.descriptor)))
                }
            } catch { self.customCheckpointImportError = String(describing: error) }
        }
        // Cache every profile so edits to non-selected models are observable.
        self.settingsStore.modelProfiles = Self.storedProfiles(
            for: self.installs,
            selected: selected,
            selectedProfile: savedModelSettings,
            selectedSettings: settings,
            persistenceEnabled: settingsPersistenceEnabled)
        for coordinator in self.installs {
            coordinator.onInstalled = { [weak self] finished in
                self?.modelDidInstall(finished)
            }
            // Concurrent downloads share one disk, so each model's free-space
            // check has to net out what the others still owe.
            coordinator.reservedByOtherInstalls = { [weak self, weak coordinator] in
                guard let self, let coordinator else { return 0 }
                let textBytes = self.installs
                    .filter { $0 !== coordinator }
                    .reduce(UInt64(0)) { partial, install in
                        let addition = partial.addingReportingOverflow(
                            install.outstandingBytes)
                        return addition.overflow ? .max : addition.partialValue
                    }
                let addition = textBytes.addingReportingOverflow(
                    self.visionInstallOutstandingBytes)
                return addition.overflow ? .max : addition.partialValue
            }
        }
        if settingsPersistenceEnabled {
            for kind in AppleModelKind.allCases {
                appleSystemPrompts[kind.rawValue] = UserDefaults.standard.string(forKey: kind.rawValue + ".systemPrompt") ?? ""
            }
        }
        applySelectedConversationModelBinding()
        // Staged images of runs that were killed before they could clean up;
        // nothing else ever removes them. The v2 root is swept too, or the
        // rename would strand every copy an older build staged.
        AppImageAttachmentStore.sweepAbandoned()
        AppImageAttachmentStore.sweepAbandoned(
            in: AppImageAttachmentStore.legacyRoot)
        refreshVisionInstallReadiness()
        #if os(macOS)
        if client is any AppSessionInferenceClient {
            let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .main)
            source.setEventHandler { [weak self] in
                Task { @MainActor [weak self] in self?.reclaimIdleConversationSessions() }
            }
            sessionPressureSource = source
            source.resume()
        }
        #endif
    }

    public static let appleModelID = "apple.on-device"
    public var selectedAppleModel: AppleModelKind? { AppleModelKind(rawValue: selectedModelID) }
    public var isAppleSelected: Bool { selectedAppleModel != nil }
    public var selectedProviderKey: String { isAppleSelected ? selectedModelID : selectedDescriptor.settingsProfileKey }
    public var selectedModelName: String { selectedAppleModel?.name ?? selectedDescriptor.shortName }
    public private(set) var appleAvailability = AppleModelAvailability.current()
    @ObservationIgnored var appleAvailabilityProvider: () -> AppleModelAvailability = { .current() }
    @ObservationIgnored var appleCloudAvailabilityProvider: () -> AppleModelAvailability = { .current(for: .privateCloudCompute) }
    public private(set) var appleCloudAvailability = AppleModelAvailability.current(for: .privateCloudCompute)
    private var lastLocalModelID: String?
    private var appleSystemPrompts: [String: String] = [:]
    public var availableAppleModels: [AppleModelKind] {
        AppleModelKind.allCases.filter { availability(for: $0).canGenerate }
    }
    public func availability(for kind: AppleModelKind) -> AppleModelAvailability {
        kind == .onDevice ? appleAvailability : appleCloudAvailability
    }
    public var selectedAppleAvailability: AppleModelAvailability {
        availability(for: selectedAppleModel ?? .onDevice)
    }
    public func refreshAppleAvailability() {
        appleAvailability = appleAvailabilityProvider()
        appleCloudAvailability = appleCloudAvailabilityProvider()
    }
    public func selectAppleModel(_ kind: AppleModelKind = .onDevice) {
        refreshAppleAvailability()
        guard !isRunning, !loadState.isLoading else { return }
        guard !conversationHasAttachments, conversation.allSatisfy({ $0.toolRounds.isEmpty }) else {
            error = .invalidRequest("Start a text-only chat to use Apple.")
            return
        }
        if let id = conversationStore.selectedConversationID { endConversationSession(id) }
        persistSettings()
        if !isAppleSelected { lastLocalModelID = selectedModelID }
        selectedModelID = kind.rawValue
        if conversationStore.selectedConversationID == nil { conversationStore.startNewConversation(modelID: kind.rawValue) }
        conversationStore.rebindConversation(to: kind.rawValue)
        error = nil
    }

    // MARK: - Model catalog

    /// The coordinator for the selected model. Every single-model property
    /// below reports this one.
    public var selectedInstall: ModelInstallCoordinator {
        installs.first { $0.id == selectedModelID }
            ?? installs.first { $0.id == lastLocalModelID } ?? installs[0]
    }

    public var selectedDescriptor: AppModelInstallDescriptor {
        selectedInstall.descriptor
    }

    /// Models that are actually on this Mac. The chat picker offers only these:
    /// choosing one that is not downloaded replaced the conversation view with
    /// the installer, which is not what picking a model in a chat should do.
    public var installedInstalls: [ModelInstallCoordinator] {
        installs.filter { $0.isInstalled || $0.id == selectedModelID }
    }

    /// The short name of the model a saved turn was answered by. Turns written
    /// before chats recorded it fall back to the chat's model.
    public func modelShortName(forProfileKey key: String?) -> String {
        guard let key else { return selectedModelName }
        if let kind = AppleModelKind(rawValue: key) { return kind.name }
        return installs.first {
            $0.descriptor.settingsProfileKey == key
        }?.descriptor.shortName ?? key
    }

    private func applySelectedConversationModelBinding() {
        if let key = conversationStore.selectedConversation?.modelID, AppleModelKind(rawValue: key) != nil {
            if !isAppleSelected { lastLocalModelID = selectedModelID }
            selectedModelID = key
            return
        }
        guard let modelID = conversationStore.selectedConversation?.modelID,
              let coordinator = installs.first(where: {
                  $0.descriptor.settingsProfileKey == modelID
              }), coordinator.id != selectedModelID else { return }
        selectedModelID = coordinator.id
        applySelectedModelDirectory(coordinator.directoryURL)
    }

    /// Whether any model in the catalog is downloading right now — the
    /// selected one or not.
    public var isInstallingAnyModel: Bool { installs.contains { $0.isInstalling } }

    public func canSelectModel(_ coordinator: ModelInstallCoordinator) -> Bool {
        // With a shared serving pool, other chats keep generating on their own
        // models while the selection moves. The selected chat itself is the
        // exception: rebinding it would change the model it is answering on.
        guard !runsBlockSelectionChange, !loadState.isLoading,
              coordinator.id != selectedModelID,
              hardwareRequirementsSatisfied(for: coordinator) else { return false }
        // A transfer keeps writing beside its original text model. Wait for it
        // to reach a saved state before changing selection; once prepared, the
        // target model itself can be selected for activation.
        if isVisionCompanionOperationInProgress { return false }
        if case .readyToActivate = visionInstallState,
           let target = visionInstallTargetModelID {
            return coordinator.id == target
        }
        return true
    }

    /// Focus a different model. A download already running for either model is
    /// left alone — only the loaded runtime and the active directory change.
    public func selectModel(_ coordinator: ModelInstallCoordinator) {
        guard canSelectModel(coordinator) else { return }
        // An existing chat carries on through the new model. History is stored
        // as turns and re-rendered in the destination's prompt format on the
        // next submission, so nothing has to be replayed in the old dialect.
        //
        // Images are the exception. A transcript holding attachments cannot
        // move to a model without image input: answering as though the images
        // were never sent is exactly the failure the image path is built to
        // avoid, so the switch is refused instead.
        if conversationHasAttachments,
           !coordinator.descriptor.supportsImageInput {
            error = .invalidRequest(
                "\(coordinator.descriptor.shortName) cannot read images. "
                + "Start a new chat to use it, or pick a model with image input.")
            return
        }
        // Capture edits to the current model before `selectedModelID` changes
        // which profile key persistence resolves.
        persistSettings()
        if let id = conversationStore.selectedConversationID { endConversationSession(id) }
        selectedModelID = coordinator.id
        conversationStore.rebindConversation(
            to: coordinator.descriptor.settingsProfileKey)
        applySelectedModelDirectory(coordinator.directoryURL)
    }

    /// The files attached to the message being composed.
    public private(set) var documentAttachments: [AppDocumentAttachment] {
        get { conversationStore.documentAttachments }
        set { conversationStore.documentAttachments = newValue }
    }

    /// Documents that went out with the message currently being answered.
    public private(set) var outputDocumentAttachments: [AppDocumentAttachment] {
        get { conversationStore.outputDocumentAttachments }
        set { conversationStore.outputDocumentAttachments = newValue }
    }

    /// At most this many files on one message. The limit is about the composer
    /// staying readable, not about the model: the context check at submission
    /// is what refuses text that does not fit.
    public static let maximumDocumentAttachments = 10

    /// Attach documents as attachments rather than as prompt text.
    ///
    /// A document is not a model capability: every model reads text, so the
    /// extracted text still goes into the prompt at send time. What changed is
    /// that the composer and the transcript hold the file, so a two-line
    /// question with a PDF on it still reads as a two-line question and the
    /// file can be taken back off.
    public func attachDocuments(_ urls: [URL]) {
        guard !isSelectedConversationRunning else { return }
        var added: [AppDocumentAttachment] = []
        var refusedForCapacity = false
        for url in urls {
            guard documentAttachments.count + added.count
                    < Self.maximumDocumentAttachments else {
                refusedForCapacity = true
                break
            }
            do {
                let document = try DocumentTextExtractor.extract(from: url)
                added.append(AppDocumentAttachment(
                    displayName: document.displayName, text: document.text))
            } catch {
                self.error = .invalidRequest("\(error)")
                return
            }
        }
        guard !added.isEmpty || refusedForCapacity else { return }
        documentAttachments.append(contentsOf: added)
        updateDocumentAttachmentNotice(refusedForCapacity: refusedForCapacity)
    }

    public func removeDocument(id: UUID) {
        guard !isSelectedConversationRunning else { return }
        documentAttachments.removeAll { $0.id == id }
        updateDocumentAttachmentNotice()
    }

    /// The refusal comes first when there is one: it is the part the user has to
    /// act on. Writing the size line over it is how the dropped files became
    /// invisible.
    private func updateDocumentAttachmentNotice(refusedForCapacity: Bool = false) {
        guard !documentAttachments.isEmpty else {
            documentAttachmentNotice = refusedForCapacity
                ? "At most \(Self.maximumDocumentAttachments) files fit on one message."
                : nil
            return
        }
        let size = "Roughly \(documentAttachments.estimatedTokens) tokens of "
            + "document text attached. The context limit for this model is "
            + "\(maxContextTokens)."
        documentAttachmentNotice = refusedForCapacity
            ? "At most \(Self.maximumDocumentAttachments) files fit on one message; "
                + "the rest were not attached. " + size
            : size
    }

    public func clearDocuments() {
        guard !isSelectedConversationRunning else { return }
        documentAttachments.removeAll()
        documentAttachmentNotice = nil
    }

    /// Set after a document is attached, so the size stays with its draft chat.
    public var documentAttachmentNotice: String? {
        get { conversationStore.documentAttachmentNotice }
        set { conversationStore.documentAttachmentNotice = newValue }
    }

    /// Whether the selected chat holds any image attachment, in a completed
    /// turn or in the draft being composed.
    var conversationHasAttachments: Bool {
        if !imageAttachments.isEmpty || !outputImageAttachments.isEmpty { return true }
        if conversation.contains(where: { !$0.images.isEmpty }) { return true }
        guard let record = conversationStore.selectedConversation else { return false }
        return record.turns.contains { !$0.attachments.isEmpty }
    }

    /// Restore a named chat and the model it was created with. This is the one
    /// model switch that must not create another chat: the destination already
    /// owns the history and prompt dialect being restored.
    /// Whether opening this chat would do anything.
    ///
    /// The rule used to live only inside `selectConversation`, which returned
    /// silently when it did not hold, while the sidebar row stayed fully
    /// enabled — so during a load, clicking a chat did nothing and said
    /// nothing. The view reads this, so the row it offers is a row that works.
    public func canSelectConversation(_ record: AppConversationRecord) -> Bool {
        guard !loadState.isLoading else { return false }
        // A single-model client cannot switch models under a running chat; a
        // pooled client serves every model concurrently.
        return client is any AppPooledInferenceClient || !isRunning
            || activeAnswerRuns.values.allSatisfy { $0.modelProfileKey == record.modelID }
    }

    public func selectConversation(_ record: AppConversationRecord) {
        guard canSelectConversation(record) else { return }
        if !isRunning { releaseTranscriptImages() }
        conversationStore.selectConversation(id: record.id)
        if AppleModelKind(rawValue: record.modelID) != nil {
            if !isAppleSelected { lastLocalModelID = selectedModelID }
            selectedModelID = record.modelID
        }
        if let coordinator = installs.first(where: {
            $0.descriptor.settingsProfileKey == record.modelID
        }), coordinator.id != selectedModelID {
            persistSettings()
            selectedModelID = coordinator.id
            applySelectedModelDirectory(coordinator.directoryURL)
        }
        if !isRunning { generationTranscriptMailbox?.reset() }
        diagnostics = nil
        error = nil
        restoreSelectedRunPresentation()
    }

    private func restoreSelectedRunPresentation() {
        guard let selected = conversationStore.selectedConversationID,
              let run = activeAnswerRuns[selected] else {
            phase = .idle
            inferenceStore.isCancellationPending = false
            return
        }
        phase = run.phase
        liveTokenCount = run.tokenCount
        liveElapsedDecodeSeconds = run.elapsedDecodeSeconds
        livePrefillDone = run.prefillDone
        livePrefillTotal = run.prefillTotal
        inferenceStore.isCancellationPending = run.cancellationPending
    }

    public func renameConversation(_ record: AppConversationRecord, to title: String) {
        conversationStore.renameConversation(id: record.id, title: title)
    }

    /// Whether deleting this chat can leave the active runtime untouched.
    /// Deleting a background chat is always harmless, but deleting the
    /// selected one implicitly selects another chat and may switch models.
    public func canDeleteConversation(_ record: AppConversationRecord) -> Bool {
        guard conversationStore.conversations.contains(where: { $0.id == record.id }) else {
            return false
        }
        if isRunning {
            return activeAnswerRuns[record.id] == nil
                && record.id != conversationStore.selectedConversationID
        }
        guard conversationStore.selectedConversationID == record.id else { return true }
        return !loadState.isLoading
    }

    private func endConversationSession(_ conversationID: UUID) {
        guard sessionCleanups[conversationID] == nil,
              let sessionID = conversationStore.existingSessionID(for: conversationID),
              let sessionClient = client as? any AppSessionInferenceClient else { return }
        sessionCleanups[conversationID] = Task { [weak self] in
            do {
                try await sessionClient.endSession(sessionID, requestID: UUID())
                self?.conversationStore.forgetSessionID(sessionID, for: conversationID)
            } catch {
                // Keep the identity so a later cleanup can retry.
            }
            self?.sessionCleanups.removeValue(forKey: conversationID)
        }
    }
    func reclaimIdleConversationSessions() {
        for id in conversationStore.existingSessionIDs.keys where activeAnswerRuns[id] == nil {
            endConversationSession(id)
        }
    }


    public func deleteConversation(_ record: AppConversationRecord) {
        guard canDeleteConversation(record) else { return }
        if conversationStore.selectedConversationID == record.id {
            // Deletion may select a chat for another model. Save this profile
            // before that changes which profile key persistence resolves.
            persistSettings()
            releaseTranscriptImages()
            generationTranscriptMailbox?.reset()
            diagnostics = nil
            error = nil
        }
        endConversationSession(record.id)
        conversationStore.deleteConversation(id: record.id)
        applySelectedConversationModelBinding()
    }

    /// A finished install selects its model when nothing else is loaded, so a
    /// first-run download lands the user in the conversation view.
    private func modelDidInstall(_ coordinator: ModelInstallCoordinator) {
        guard coordinator.id == selectedModelID else {
            if loadState == .notLoaded, !isRunning, !selectedInstall.isInstalled {
                selectModel(coordinator)
            }
            return
        }
        modelPathText = coordinator.directoryURL.path
        loadState = .notLoaded
        refreshVisionInstallReadiness()
    }

    private var activeRunConversationID: UUID? {
        if let selected = conversationStore.selectedConversationID,
           activeAnswerRuns[selected] != nil { return selected }
        return activeAnswerRuns.keys.first
    }

    private var activeRunRuntimeKey: AppLoadedRuntimeKey? {
        activeRunConversationID.flatMap { activeAnswerRuns[$0]?.runtimeKey }
    }

    public var isRunning: Bool { !activeAnswerRuns.isEmpty }

    /// A run is using the selected model's loaded runtime.
    private var selectedModelIsRunning: Bool {
        activeAnswerRuns.values.contains { $0.modelProfileKey == selectedProviderKey }
    }

    /// Whether running chats forbid loading, reloading or unloading the
    /// selected model. A pooled client only needs its own model to be idle.
    private var runsBlockModelLifecycle: Bool {
        client is any AppPooledInferenceClient ? selectedModelIsRunning : isRunning
    }

    private var runsBlockSelectionChange: Bool {
        client is any AppPooledInferenceClient ? isSelectedConversationRunning : isRunning
    }
    public var isSelectedConversationRunning: Bool {
        conversationStore.selectedConversationID.map { activeAnswerRuns[$0] != nil } ?? false
    }

    public var isModelAvailable: Bool { isAppleSelected ? selectedAppleAvailability.canGenerate : loadState.isReady }

    public var hasStaleLoadedRuntime: Bool {
        guard !isAppleSelected, loadState.isReady, let loadedRuntimeKey else { return false }
        return loadedRuntimeKey != currentRuntimeKey
    }

    public var canLoadModel: Bool {
        !isAppleSelected && isModelInstalled && hardwareRequirementsSatisfied(for: selectedInstall)
            && memoryRequirementsSatisfied
            && !runsBlockModelLifecycle
            && !isVisionCompanionOperationInProgress
            && (loadState == .notLoaded || loadState.isFailed)
    }

    public var canCancelLoad: Bool {
        if case .loading = loadState { return loadTask != nil }
        return false
    }

    public var canReloadModel: Bool {
        !isAppleSelected && isModelInstalled && hardwareRequirementsSatisfied(for: selectedInstall)
            && memoryRequirementsSatisfied
            && !runsBlockModelLifecycle
            && !isVisionCompanionOperationInProgress
            && loadState.isReady && hasStaleLoadedRuntime
    }

    public var canUnloadModel: Bool {
        !isAppleSelected && isModelInstalled && !runsBlockModelLifecycle
            && !isVisionCompanionOperationInProgress
            && loadState.isReady
    }

    // MARK: - Selected model's install (delegates to `selectedInstall`)

    public var installState: AppModelInstallState { selectedInstall.state }

    public var installReadiness: AppModelInstallReadiness { selectedInstall.readiness }

    public var installationStatus: AppModelInstallationStatus {
        selectedInstall.installationStatus
    }

    public var installETAPresentation: DownloadETAPresentation {
        selectedInstall.etaPresentation
    }

    public var installETAText: String? { selectedInstall.etaText }

    public var isModelInstalled: Bool { selectedInstall.isInstalled }

    public var requiresModelInstallation: Bool { !isAppleSelected && !isModelInstalled }

    public var installDescriptor: AppModelInstallDescriptor { selectedDescriptor }

    public var installRequirement: AppModelInstallRequirement? {
        installReadiness.requirement
    }

    public var isInstallingModel: Bool { installState.isInstalling }

    public var canInstallModel: Bool {
        canInstallModel(selectedInstall)
    }

    public var selectedModelHardwareEligibility: AppModelHardwareEligibility {
        hardwareEligibility(for: selectedInstall)
    }

    public var selectedContextHardwareEligibility: AppModelContextEligibility {
        contextEligibility(
            for: selectedInstall,
            contextTokens: maxContextTokens,
            expertCacheSlots: runtimeOptions.expertCacheSlots)
    }

    public func hardwareEligibility(
        for coordinator: ModelInstallCoordinator
    ) -> AppModelHardwareEligibility {
        coordinator.descriptor.hardwareEligibility(on: deviceCapabilities)
    }

    /// Whether this model's measured hardware requirements are satisfied, or
    /// deliberately waived. Every gate that used to read
    /// `hardwareEligibility(for:).isCompatible` reads this instead, so one
    /// switch covers download, selection, and load rather than three.
    public func hardwareRequirementsSatisfied(
        for coordinator: ModelInstallCoordinator
    ) -> Bool {
        bypassModelRestrictions || hardwareEligibility(for: coordinator).isCompatible
    }

    /// Whether the selected model's current context and cache fit the safe
    /// budget, or the person has chosen to run past it.
    public var memoryRequirementsSatisfied: Bool {
        bypassModelRestrictions || selectedContextHardwareEligibility.isCompatible
    }

    /// Why the selected model cannot be loaded right now, in words that name
    /// the setting responsible. Nil when nothing is blocking it. Without this
    /// a model whose context was set above the budget simply refused to load
    /// with no visible reason anywhere outside the Settings memory section.
    public var loadBlockedReason: String? {
        guard isModelInstalled, !bypassModelRestrictions else { return nil }
        if let explanation = selectedModelHardwareEligibility.explanation {
            return explanation
                + " Turn on Bypass model restrictions in Settings to run it anyway."
        }
        if let explanation = selectedContextHardwareEligibility.explanation {
            return explanation
                + " Lower Context or Expert cache in Settings, switch that model "
                + "to Auto, or turn on Bypass model restrictions to run it anyway."
        }
        return nil
    }

    public func contextEligibility(
        for coordinator: ModelInstallCoordinator,
        contextTokens: Int,
        expertCacheSlots: Int
    ) -> AppModelContextEligibility {
        if coordinator.id == selectedModelID,
           let estimate = AppAutomaticMemoryPlanner.estimatedAllocation(directory: coordinator.directoryURL,
                context: contextTokens, options: AppRuntimeOptions(
                    expertCacheSlots: expertCacheSlots, prefillChunkTokens: runtimeOptions.prefillChunkTokens,
                    denseResidencyPolicy: runtimeOptions.denseResidencyPolicy,
                    retainedConversationBytes: runtimeOptions.retainedConversationBytes)) {
            return AppModelContextEligibility(estimatedWorkingSetBytes: estimate.peakAllocatedBytes,
                safeBudgetBytes: deviceCapabilities.safeAppMemoryBudgetBytes)
        }
        return coordinator.descriptor.contextEligibility(
            contextTokens: contextTokens,
            expertCacheSlots: expertCacheSlots,
            prefillChunkTokens: coordinator.id == selectedModelID
                ? runtimeOptions.prefillChunkTokens
                : coordinator.descriptor.recommendedPrefillChunkTokens(on: deviceCapabilities),
            on: deviceCapabilities)
    }

    public func canInstallModel(_ coordinator: ModelInstallCoordinator) -> Bool {
        hardwareRequirementsSatisfied(for: coordinator)
            && coordinator.canInstall && !isRunning && !loadState.isLoading
            && !isVisionCompanionOperationInProgress
    }

    public var canCancelInstall: Bool { selectedInstall.canCancel }

    public var isVisionPackInstalled: Bool {
        visionInstallationStatus == .complete
    }

    public var isInstallingVisionPack: Bool { visionInstallState.isInstalling }

    public var visionInstallOutstandingBytes: UInt64 {
        guard let targetID = visionInstallTargetModelID,
              let target = installs.first(where: { $0.id == targetID }),
              target.descriptor.supportsImageInput else { return 0 }
        switch visionInstallState {
        case .checking, .downloadingMetadata, .planning, .reservingOutput,
             .hashingOutput, .finalizing:
            return visionInstallDescriptor(for: target)?.installedBytes ?? 0
        case .copyingPayload(let reused, let downloaded, let total):
            let completed = reused.addingReportingOverflow(downloaded)
            guard !completed.overflow, total > completed.partialValue else { return 0 }
            return total - completed.partialValue
        case .idle, .activating, .cancelling, .discarding, .cancelled,
             .readyToActivate, .recoverable, .installed, .failed:
            return 0
        }
    }

    /// `nil` when the selected model ships no image pack, which text-only
    /// models such as GPT-OSS do not.
    public var visionInstallDescriptor: AppModelInstallDescriptor? {
        visionInstallDescriptor(for: selectedInstall)
    }

    public func visionInstallDescriptor(
        for coordinator: ModelInstallCoordinator
    ) -> AppModelInstallDescriptor? {
        if coordinator.descriptor.catalogID != nil {
            return coordinator.descriptor.imageCompanion
        }
        guard let variant = coordinator.descriptor.modelVariant else {
            return .visionCompanion(for: coordinator.descriptor.family)
        }
        return .visionCompanion(for: variant)
    }

    public func visionInstallationStatus(
        for coordinator: ModelInstallCoordinator
    ) -> AppVisionPackInstallationStatus {
        AppVisionPackInstallationProbe.status(at: coordinator.directoryURL)
    }

    public func isVisionPackInstalled(
        for coordinator: ModelInstallCoordinator
    ) -> Bool {
        visionInstallationStatus(for: coordinator) == .complete
    }

    public func isVisionInstallTarget(
        _ coordinator: ModelInstallCoordinator
    ) -> Bool {
        visionInstallTargetModelID == coordinator.id
    }

    /// Selected-model surfaces must not relabel another catalog row's active
    /// transfer as though it belonged to the model currently in use.
    public var selectedModelOwnsVisionInstallState: Bool {
        visionInstallTargetModelID == nil
            || visionInstallTargetModelID == selectedModelID
    }

    public func visionDownloadButtonLabel(
        for coordinator: ModelInstallCoordinator
    ) -> String {
        let verb: String
        if hasPartialVisionPackDownload(for: coordinator) {
            verb = "Resume"
        } else if hasVisionPackDirectory(for: coordinator) {
            verb = "Repair"
        } else {
            verb = "Download"
        }
        let name = visionInstallDescriptor(for: coordinator)?.displayName
            ?? "Image Support"
        return "\(verb) \(name)"
    }

    /// Every companion Download, Resume, Verify, Activate, Repair, and Remove
    /// operation is one app-blocking state: model actions stay disabled until it
    /// reaches a resting state, so a companion transaction never overlaps a
    /// loaded session or another companion operation.
    public var isVisionCompanionOperationInProgress: Bool {
        visionInstallState.isInstalling
    }

    /// Downloading prepares a sibling directory and does not touch the loaded
    /// runtime or the active companion. Activation and destructive mutations
    /// still require an unloaded session below.
    public var canPrepareVisionCompanionOperation: Bool {
        !isRunning && !loadState.isLoading
            && !isVisionCompanionOperationInProgress
    }

    /// Activation, discard, and removal mutate companion state and therefore
    /// require an unloaded model session.
    public var canBeginVisionCompanionOperation: Bool {
        canPrepareVisionCompanionOperation && !loadState.isReady
    }

    public var canInstallVisionPack: Bool {
        canInstallVisionPack(for: selectedInstall)
    }

    public func canInstallVisionPack(
        for coordinator: ModelInstallCoordinator
    ) -> Bool {
        guard isVisionRuntimeSupported,
              coordinator.descriptor.supportsImageInput,
              hardwareRequirementsSatisfied(for: coordinator) else { return false }
        // A layout with nowhere to put a companion cannot be repaired by
        // downloading one, so do not offer to.
        let status = visionInstallationStatus(for: coordinator)
        guard status != .unsupportedLayout, status != .complete,
              coordinator.isInstalled else { return false }
        if isVisionInstallTarget(coordinator),
           case .readyToActivate = visionInstallState { return false }
        guard canPrepareVisionCompanionOperation else { return false }
        do {
            return try checkedVisionInstallRequirement(
                textModelDirectory: coordinator.directoryURL).canInstall
        } catch {
            return false
        }
    }

    public func visionInstallRequirement(
        for coordinator: ModelInstallCoordinator
    ) -> AppModelInstallRequirement? {
        try? checkedVisionInstallRequirement(
            textModelDirectory: coordinator.directoryURL)
    }

    public var canActivateVisionPack: Bool {
        guard isVisionRuntimeSupported else { return false }
        guard case .readyToActivate = visionInstallState else { return false }
        guard visionInstallTargetModelID == nil
                || visionInstallTargetModelID == selectedModelID else { return false }
        return canBeginVisionCompanionOperation
    }

    public var canCancelVisionInstall: Bool { visionInstallState.canCancel }

    public var visionInstallProgressFraction: Double? {
        // Activation hashes about 1.5 GB, so it gets a bar of its own rather
        // than an indeterminate spinner for minutes.
        if case .activating = visionInstallState { return visionActivationProgress }
        guard case .copyingPayload(let reused, let downloaded, let total) = visionInstallState,
              total > 0 else { return nil }
        let addition = reused.addingReportingOverflow(downloaded)
        let done = addition.overflow ? UInt64.max : addition.partialValue
        return min(max(Double(done) / Double(total), 0), 1)
    }

    public var visionInstallPhaseLabel: String {
        switch visionInstallState {
        case .idle: return isVisionPackInstalled ? "Installed" : "Not installed"
        case .checking: return "Checking image support"
        case .downloadingMetadata: return "Downloading metadata"
        case .planning: return "Planning image support"
        case .reservingOutput: return "Reserving storage"
        case .copyingPayload: return "Downloading image support"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing download"
        case .activating:
            guard let fraction = visionActivationProgress else {
                return "Activating image support"
            }
            return "Verifying image support \(Int(fraction * 100))%"
        case .cancelling: return "Cancelling"
        case .discarding: return "Cleaning up"
        case .cancelled: return "Download paused"
        case .readyToActivate: return "Ready to activate"
        case .recoverable: return "Saved download needs attention"
        case .installed: return "Installed"
        case .failed: return "Installation failed"
        }
    }

    public var installDownloadedBytes: UInt64? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState else {
            return nil
        }
        return min(reused.addingReportingOverflow(downloaded).partialValue, total)
    }

    public var installTotalBytes: UInt64? {
        guard case .copyingPayload(_, _, let total) = installState else {
            return nil
        }
        return total
    }

    public var installReusedBytes: UInt64? {
        guard case .copyingPayload(let reused, _, _) = installState else {
            return nil
        }
        return reused
    }

    public var installDownloadedThisRunBytes: UInt64? {
        guard case .copyingPayload(_, let downloaded, _) = installState else {
            return nil
        }
        return downloaded
    }

    public var installProgressFraction: Double? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState,
              total > 0 else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        let done = addition.overflow ? UInt64.max : addition.partialValue
        return min(max(Double(done) / Double(total), 0), 1)
    }

    public var installPhaseLabel: String {
        switch installState {
        case .idle: return "Model required"
        case .checking: return "Checking installation"
        case .downloadingMetadata: return "Downloading metadata"
        case .planning: return "Planning installation"
        case .reservingOutput: return "Reserving storage"
        case .copyingPayload: return "Downloading model"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing installation"
        case .activating:
            guard let fraction = visionActivationProgress else {
                return "Activating image support"
            }
            return "Verifying image support \(Int(fraction * 100))%"
        case .cancelling: return "Cancelling"
        case .discarding: return "Discarding download"
        case .cancelled: return "Download paused"
        case .readyToActivate: return "Ready to activate"
        case .recoverable: return "Saved download needs attention"
        case .installed: return "Model installed"
        case .failed: return "Installation failed"
        }
    }

    public var canRun: Bool {
        if isAppleSelected {
            return selectedAppleAvailability.canGenerate && !isRunning && !loadState.isLoading && !isAddingImages
                && imageAttachments.isEmpty && conversation.allSatisfy { $0.images.isEmpty && $0.toolRounds.isEmpty }
                && hasComposedInput
        }
        // Staging copies the files a request will carry. Starting a run while
        // it is in flight sent a request without those images and then landed
        // them on the next message instead.
        let selectedID = conversationStore.selectedConversationID
        let selectedAlreadyRunning = selectedID.map { activeAnswerRuns[$0] != nil } ?? false
        let requestedRuntime = currentRuntimeKey
        let admissionCompatible = activeAnswerRuns.count < 4
            && (activeAnswerRuns.isEmpty || client is any AppSessionInferenceClient)
            && activeAnswerRuns.values.allSatisfy {
            // Chats on another model do not constrain this model's runtime.
            $0.modelProfileKey != selectedProviderKey
                || $0.runtimeKey == requestedRuntime
        }
        return !selectedAlreadyRunning && !(selectedID.map { sessionCleanups[$0] != nil } ?? false)
            && admissionCompatible && !isAddingImages && isModelAvailable && !loadState.isLoading
            && !isVisionCompanionOperationInProgress
            && !hasStaleLoadedRuntime
            && hasComposedInput
    }

    /// Anything worth sending: typed text, an image, or an attached file.
    public var hasComposedInput: Bool {
        !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !imageAttachments.isEmpty
            || !documentAttachments.isEmpty
    }

    /// A submission can start immediately against a ready runtime or begin the
    /// selected model's normal load and run once that same load reaches ready.
    public var canSubmit: Bool {
        guard hasComposedInput, !runsAfterCurrentLoad,
              !(conversationStore.selectedConversationID.map { sessionCleanups[$0] != nil } ?? false)
        else { return false }
        return canRun || canLoadModel
    }

    public var isLoadingForSubmission: Bool {
        runsAfterCurrentLoad && loadState.isLoading
    }

    public var canCancel: Bool {
        isSelectedConversationRunning && !isCancellationPending
    }

    public var hasOutputTranscript: Bool {
        !conversation.isEmpty || hasLiveMessage
    }

    /// The exchange the transcript draws below the recorded ones: the one being
    /// generated, or the one whose run ended without becoming history because
    /// it was cancelled or failed.
    ///
    /// A completed exchange stays in the output slot — the copy commands and
    /// the diagnostics describe the last run — so "recorded" rather than
    /// "empty" is what takes it off the bottom of the transcript.
    public var hasLiveMessage: Bool {
        guard !conversationStore.outputTurnIsRecorded else { return false }
        let selectedRunIsActive = isSelectedConversationRunning
        return selectedRunIsActive || !outputPromptText.isEmpty
            || !outputImageAttachments.isEmpty
            || !outputDocumentAttachments.isEmpty || !outputText.isEmpty
            || !outputThinkingText.isEmpty
    }

    /// Completed turns the next prompt will carry as context.
    public var conversationTurnCount: Int { conversation.count }

    public var outputResponsePlainText: String {
        // A continued run generates only the remainder, so the part that was
        // already written is carried here. Without it the finished turn would
        // replace a long truncated answer with its own tail.
        let liveTranscript = activeRunConversationID == conversationStore.selectedConversationID
            && !isAppleSelected
            && !(client is any AppSessionInferenceClient)
            ? generationTranscriptMailbox?.completeText : nil
        return conversationStore.outputAssistantPrefix + (liveTranscript ?? outputText)
    }

    /// The already-written part of the answer this run is carrying on. Empty
    /// for every ordinary run.
    private var activeAssistantPrefix: String { conversationStore.outputAssistantPrefix }

    public var outputConversationPlainText: String {
        let response = outputResponsePlainText
        let modelLabel = selectedModelName
        switch (outputPromptText.isEmpty, response.isEmpty) {
        case (true, true):
            return ""
        case (false, true):
            return "You:\n\(outputPromptText)"
        case (true, false):
            return "\(modelLabel):\n\(response)"
        case (false, false):
            return "You:\n\(outputPromptText)\n\n\(modelLabel):\n\(response)"
        }
    }

    /// The transcript exactly as it is drawn: saved turns followed by an
    /// unrecorded live, cancelled, or failed exchange. A completed output slot
    /// remains available to "Copy Last Response", but must not be appended a
    /// second time after that turn has joined `conversation`.
    public var conversationPlainText: String {
        var blocks = conversation.map { turn in
            let name = modelShortName(forProfileKey: turn.modelID)
            return "You:\n\(turn.modelPrompt)\n\n\(name):\n\(turn.response)"
        }
        if hasLiveMessage {
            let live = outputConversationPlainText
            if !live.isEmpty { blocks.append(live) }
        }
        return blocks.joined(separator: "\n\n")
    }

    public var liveTokensPerSecond: Double {
        liveElapsedDecodeSeconds > 0 ? Double(liveTokenCount) / liveElapsedDecodeSeconds : 0
    }

    public var presentation: AppPresentationState {
        AppPresentationState.resolve(AppPresentationSnapshot(
            requiresInstallation: requiresModelInstallation,
            installState: installState,
            installReadiness: installReadiness,
            loadState: isAppleSelected ? (selectedAppleAvailability.canGenerate ? .ready(modelDirectory: URL(fileURLWithPath: "/"), loadSeconds: 0) : .notLoaded) : loadState,
            hasStaleRuntime: hasStaleLoadedRuntime,
            isRunning: isSelectedConversationRunning,
            isGenerationCancellationPending: isCancellationPending,
            generationPhase: phase,
            livePrefillDone: livePrefillDone,
            livePrefillTotal: livePrefillTotal,
            lastStopReason: diagnostics?.stopReason,
            isVisionCompanionOperationInProgress: isVisionCompanionOperationInProgress,
            loadBlockedReason: loadBlockedReason))
    }

    public var currentProcessMemoryBytes: UInt64? {
        guard loadState.isReady || isRunning else { return nil }
        // `liveMemoryBytes` first because it is a tracked property: reading the
        // reporter alone told the truth but was invisible to observation, so
        // the figure only refreshed when something else — a generated token —
        // happened to redraw the view. Through prefill, nothing did.
        if let liveMemoryBytes { return liveMemoryBytes }
        // When inference runs in another process, its memory is the only
        // memory worth showing. Falling back to this app's own sampler put the
        // UI's footprint in a row labelled as the model's.
        if let reporter = client as? any AppInferenceMemoryReporting {
            return reporter.currentInferenceMemoryBytes
        }
        return memorySampler.sample()
    }

    public var generationTranscriptMailbox: GenerationTranscriptMailbox? {
        guard !(client is any AppSessionInferenceClient) else { return nil }
        return (client as? any AppInferenceTranscriptReporting)?.generationTranscriptMailbox
    }

    private var currentRuntimeKey: AppLoadedRuntimeKey {
        AppLoadedRuntimeKey(modelDirectory: URL(fileURLWithPath: modelPathText),
                            maxContextTokens: maxContextTokens,
                            options: runtimeOptions,
                            forceLogitsHead: currentForceLogitsHead)
    }

    private var currentForceLogitsHead: Bool {
        temperature != 0
    }

    /// Point the selected model at a different directory. Any install running
    /// against the old location is abandoned; other models are untouched.
    public func setModelURL(_ url: URL) {
        guard !isRunning else { return }
        let path = url.standardizedFileURL.path
        guard path != modelPathText else { return }
        // Save against the old directory before the coordinator moves.
        persistSettings()
        selectedInstall.setDirectory(url)
        applySelectedModelDirectory(url, adoptingStoredProfile: false)
    }

    /// Shared tail of selecting a model and repointing one: adopt the
    /// directory's persisted settings and drop everything tied to the runtime
    /// that was loaded from the previous one.
    private func applySelectedModelDirectory(
        _ url: URL,
        adoptingStoredProfile: Bool = true
    ) {
        let path = url.standardizedFileURL.path
        let preservesPreparedVisionPack = visionInstallTargetModelID == selectedModelID
            && visionInstallTargetDirectory?.standardizedFileURL.path == path
            && {
                if case .readyToActivate = visionInstallState { return true }
                return false
            }()
        modelPathText = path
        clearImages()
        applyPersistedSettings(
            forModelDirectory: URL(fileURLWithPath: path, isDirectory: true),
            adoptingStoredProfile: adoptingStoredProfile)
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        if !preservesPreparedVisionPack {
            visionInstallGeneration &+= 1
            visionInstallCancellationRequested = false
            visionInstallTask?.cancel()
            visionInstaller.cancel()
            visionInstallTask = nil
            visionInstallTargetModelID = nil
            visionInstallTargetDirectory = nil
            resetVisionInstallETA()
            visionInstallState = .idle
        }
        pendingExplicitLoadRuntimeKey = nil
        loadedRuntimeKey = nil
        loadState = .notLoaded
        diagnostics = nil
        error = nil
        phase = .idle
        selectedInstall.refresh()
        visionInstallationStatus = AppVisionPackInstallationProbe.status(
            at: URL(fileURLWithPath: path))
        refreshVisionInstallReadiness()

        if let pooled = client as? any AppPooledInferenceClient {
            // The pool keeps the previous model resident (other chats may be
            // generating on it) and releases it only under memory pressure or
            // an explicit unload. A model the pool still holds is ready as is.
            let directory = URL(fileURLWithPath: path, isDirectory: true)
            if let resident = pooled.residentRuntime(forModelDirectory: directory) {
                loadedRuntimeKey = AppLoadedRuntimeKey(
                    modelDirectory: directory, maxContextTokens: resident.maxContextTokens,
                    options: resident.options, forceLogitsHead: resident.forceLogitsHead)
                loadState = .ready(modelDirectory: directory, loadSeconds: 0)
            }
        } else if let lifecycle = client as? AppModelLifecycleClient {
            unloadGeneration &+= 1
            let generation = unloadGeneration
            let task = Task { [weak self, lifecycle] in
                await lifecycle.unload()
                self?.clearUnloadTask(generation: generation)
            }
            unloadTask = task
        }
    }

    public func loadModel() {
        guard canLoadModel else { return }
        // What Auto learned from the answers since the last load.
        if automaticMemory, settingsPersistenceEnabled {
            applySettingsProfile(storedSettingsProfile(for: selectedInstall))
        }
        beginLoad()
    }

    public func perform(_ action: AppModelAction) {
        switch action {
        case .install: installModel()
        case .cancelInstall: cancelInstall()
        case .load, .retryLoad: loadModel()
        case .cancelLoad: cancelLoad()
        case .reload: reloadModel()
        case .unload: unloadModel()
        }
    }

    /// Ignores a hex string that fails to parse, so a malformed value typed
    /// into the settings field never reaches disk or the shared theme.
    public func setCustomAccentColorHex(_ hex: String) {
        guard AppHexColor(hexString: hex) != nil, customAccentColorHex != hex else { return }
        customAccentColorHex = hex
        persistSettings()
    }

    /// Loads every profile once, sharing reads for models in the same directory.
    private static func storedProfiles(
        for installs: [ModelInstallCoordinator],
        selected: ModelInstallCoordinator,
        selectedProfile: AppModelSettingsProfile,
        selectedSettings: MacAppSettings,
        persistenceEnabled: Bool
    ) -> [String: AppModelSettingsProfile] {
        var profiles = [selected.descriptor.settingsProfileKey: selectedProfile]
        let others = installs.filter { $0.id != selected.id }
        guard persistenceEnabled else {
            for install in others {
                let profileKey = install.descriptor.settingsProfileKey
                profiles[profileKey] = .defaults(for: profileKey)
            }
            return profiles
        }
        var loaded = [
            MacAppSettingsFileStore.fileURL(forModelDirectory: selected.directoryURL):
                selectedSettings
        ]
        for install in others {
            let profileKey = install.descriptor.settingsProfileKey
            let fileURL = MacAppSettingsFileStore.fileURL(
                forModelDirectory: install.directoryURL)
            let settings = loaded[fileURL] ?? MacAppSettingsFileStore.loadOrCreate(
                forModelDirectory: install.directoryURL,
                profileKey: profileKey)
            loaded[fileURL] = settings
            profiles[profileKey] = settings.profile(for: profileKey)
        }
        return profiles
    }

    /// The saved profile before Auto resolves its runtime values.
    private func storedSettingsProfile(
        for coordinator: ModelInstallCoordinator
    ) -> AppModelSettingsProfile {
        let profileKey = coordinator.descriptor.settingsProfileKey
        return settingsStore.modelProfiles[profileKey] ?? .defaults(for: profileKey)
    }

    /// The effective profile, including Auto's resolved runtime values.
    public func settingsProfile(
        for coordinator: ModelInstallCoordinator
    ) -> AppModelSettingsProfile {
        if coordinator.id == selectedModelID {
            return currentSettingsProfile
        }
        return AppAutomaticMemoryPlanner.applying(
            storedSettingsProfile(for: coordinator),
            for: coordinator.descriptor,
            on: deviceCapabilities, modelDirectory: coordinator.directoryURL, useBundledLayout: settingsPersistenceEnabled)
    }

    /// Captures live settings without replacing Auto's saved manual values.
    private func captureSelectedSettingsProfile() {
        let profileKey = selectedProviderKey
        settingsStore.modelProfiles[profileKey] = Self.preservingManualMemorySettings(
            in: currentSettingsProfile,
            from: storedSettingsProfile(for: selectedInstall))
    }

    public func automaticMemoryPlan(
        for coordinator: ModelInstallCoordinator
    ) -> AppAutomaticMemoryPlan? {
        automaticMemoryPlan(for: coordinator,
                            profile: settingsProfile(for: coordinator).automaticMemoryProfile)
    }

    /// The plan a given Auto profile would resolve for this model, so the
    /// settings picker can label each choice with what it would actually do.
    public func automaticMemoryPlan(
        for coordinator: ModelInstallCoordinator,
        profile: AppAutomaticMemoryProfile
    ) -> AppAutomaticMemoryPlan? {
        AppAutomaticMemoryPlanner.plan(
            for: coordinator.descriptor,
            on: deviceCapabilities,
            profile: profile, workloadKey: coordinator.id == selectedModelID ? autoWorkloadKey : "text-cold-short-t0",
            modelDirectory: coordinator.directoryURL, useBundledLayout: settingsPersistenceEnabled,
            engineIdentity: coordinator.id == selectedModelID ? (client as? any AppInferenceEngineReporting)?.currentEngineIdentity : nil)
    }

    /// Switches Auto without discarding the manual memory profile underneath
    /// it. Auto's resolved values drive the loaded runtime; the saved context,
    /// cache, and prefill choices return when Auto is turned off.
    public func setAutomaticMemory(
        _ enabled: Bool,
        for coordinator: ModelInstallCoordinator
    ) {
        var stored = storedSettingsProfile(for: coordinator)
        guard stored.automaticMemory != enabled else { return }
        stored.automaticMemory = enabled
        let profileKey = coordinator.descriptor.settingsProfileKey
        settingsStore.modelProfiles[profileKey] = stored
        if coordinator.id == selectedModelID {
            applySettingsProfile(stored)
            persistSettings()
        } else {
            markProfileUnsaved(profileKey)
            persistSettingsSoon()
        }
    }

    /// Updates observable state immediately and coalesces the file write.
    public func updateSettingsProfile(
        _ profile: AppModelSettingsProfile,
        for coordinator: ModelInstallCoordinator
    ) {
        guard profile.isValid() else { return }
        let profileKey = coordinator.descriptor.settingsProfileKey
        settingsStore.modelProfiles[profileKey] = Self.preservingManualMemorySettings(
            in: profile,
            from: storedSettingsProfile(for: coordinator))
        if coordinator.id == selectedModelID {
            applySettingsProfile(profile)
        } else {
            markProfileUnsaved(profileKey)
        }
        persistSettingsSoon()
    }

    private func markProfileUnsaved(_ profileKey: String) {
        guard settingsPersistenceEnabled else { return }
        unsavedProfileKeys.insert(profileKey)
    }

    /// Starts the launch load if it is switched on and the model can be loaded.
    /// Called once, when the window first appears; a model that is missing,
    /// already loading, or busy with a companion operation is left alone.
    public func loadModelAtLaunchIfEnabled() {
        guard loadModelOnLaunch, canLoadModel else { return }
        loadModel()
    }

    /// Downloads the engine each installed model needs, in the background.
    /// An update that ships new engines otherwise made the first load of each
    /// model family wait for its download under "Preparing inference
    /// engine". A load started meanwhile joins the same download, and an
    /// engine already in place is only checked.
    public func prepareEnginesAtLaunch() {
        guard settingsPersistenceEnabled else { return }
        let directories = installs.filter(\.isInstalled).map(\.directoryURL)
        Task.detached(priority: .utility) {
            var prepared: Set<String> = []
            for directory in directories {
                guard let architecture = try? ManifestReader.resolveArchitecture(directoryURL: directory),
                      prepared.insert(architecture.variant.rawValue).inserted,
                      let engine = try? await InferenceEngineEnvironment.acquire(
                        architectureID: architecture.variant.rawValue,
                        quantizationLayout: architecture.family == .gptOss ? "mxfp4" : "affine")
                else { continue }
                await engine.release()
            }
        }
    }

    /// Whether an image can be attached at all.
    ///
    /// The runtime flag only says this build *can* use images; the companion
    /// pack is what makes it possible for this model. Gating on the flag alone
    /// offered an Add-images button with no tower behind it, and the failure
    /// only surfaced when the user pressed Generate.
    public var isImageInputAvailable: Bool {
        !isAppleSelected && selectedDescriptor.supportsImageInput
            && isVisionRuntimeSupported && isVisionPackInstalled
    }

    /// Image support is part of this build. Hardware support and companion-pack
    /// availability are separate so the inspector can explain either absence.
    public var visionRuntimeEnabled: Bool { !isAppleSelected && selectedDescriptor.supportsImageInput }

    /// Room left for the prompt when working out how many images fit. The
    /// runtime still rejects a combination that does not fit, so this only has
    /// to be a defensible reserve rather than an exact prompt measurement.
    static let reservedPromptTokens = 1_024

    /// How many images this conversation can hold, derived from the context
    /// exactly as the server derives its budget. It used to be a fixed four,
    /// which meant the same set of images was accepted over the API and refused
    /// in the app.
    public var maximumImageAttachments: Int {
        guard selectedDescriptor.supportsImageInput else { return 0 }
        // The context a run will actually use, which is the loaded session's
        // until it is reloaded. Capping on the pending setting instead let the
        // composer accept images the request then refused.
        return max(1, VisionImageTokenBudget.capacity(
            maxContext: effectiveMaxContextTokens,
            reservedTextTokens: Self.reservedPromptTokens,
            family: selectedDescriptor.family))
    }

    /// The context a generation would run with right now.
    public var effectiveMaxContextTokens: Int {
        (loadedRuntimeKey ?? currentRuntimeKey).maxContextTokens
    }

    /// `discardingSourceDirectory` is the temp directory a file-promise drop
    /// wrote into. It is ours, it holds nothing but those copies, and staging
    /// takes its own copy — so it must not outlive the staging that consumed
    /// it, which is exactly how it leaked.
    public func addImages(_ urls: [URL], discardingSourceDirectory: URL? = nil) {
        // Every early return has to discard the promise directory itself. The
        // staging task's `defer` below owns it only once that task exists, so a
        // return above it strands the full-size copies with nothing left to
        // delete them: the attachment sweep only covers the staging root.
        func discardSource() {
            if let discardingSourceDirectory {
                try? FileManager.default.removeItem(at: discardingSourceDirectory)
            }
        }
        guard isImageInputAvailable, !urls.isEmpty else {
            discardSource()
            return
        }
        guard !isSelectedConversationRunning else {
            // A promise drop admitted before the run started can be delivered
            // after it. Returning silently made the images look as though they
            // had simply vanished.
            imageAttachmentError =
                "Wait for the current run to finish before attaching images."
            discardSource()
            return
        }
        let capacity = maximumImageAttachments
        let available = max(0, capacity - imageAttachments.count)
        guard available > 0 else {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
            discardSource()
            return
        }
        addingImagesCount += 1
        imageAttachmentError = nil
        // Dropping the rest silently left the user believing every image they
        // chose was attached.
        let selected = Array(urls.prefix(available))
        if selected.count < urls.count {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
        }
        let store = attachmentStore
        Task.detached(priority: .userInitiated) { [weak self] in
            var staged: [AppImageAttachment] = []
            defer {
                if let discardingSourceDirectory {
                    try? FileManager.default.removeItem(at: discardingSourceDirectory)
                }
            }
            do {
                for url in selected {
                    staged.append(try store.stage(url))
                }
                await self?.finishAddingImages(staged)
            } catch {
                // The batch is all-or-nothing, so the copies made before the
                // failure are referenced by nothing and would never be deleted.
                for attachment in staged { store.remove(attachment) }
                await self?.finishAddingImages(error: error)
            }
        }
    }

    /// Attaches image bytes that have no file behind them: an image copied out
    /// of another app arrives on the pasteboard as data, and a drag from an app
    /// that has not written the file yet arrives as a promise.
    public func addImageData(_ data: Data, displayName: String) {
        guard isImageInputAvailable, !isSelectedConversationRunning else { return }
        let capacity = maximumImageAttachments
        guard imageAttachments.count < capacity else {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
            return
        }
        addingImagesCount += 1
        imageAttachmentError = nil
        let store = attachmentStore
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let staged = try store.stage(data: data, displayName: displayName)
                await self?.finishAddingImages([staged])
            } catch {
                await self?.finishAddingImages(error: error)
            }
        }
    }

    static func imageCapacityMessage(capacity: Int, context: Int) -> String {
        "At most \(capacity) image\(capacity == 1 ? "" : "s") fit in the "
            + "\(context / 1_024)K context this session is running with. Raise "
            + "Context in Memory and reload the model to send more."
    }

    public func reportImageAttachmentError(_ error: Error) {
        imageAttachmentError = String(describing: error)
    }

    public func reportImageAttachmentError(_ message: String) {
        imageAttachmentError = message
    }

    public func removeImage(id: UUID) {
        guard !isSelectedConversationRunning,
              let index = imageAttachments.firstIndex(where: { $0.id == id }) else { return }
        let attachment = imageAttachments.remove(at: index)
        attachmentStore.remove(attachment)
        imageAttachmentError = nil
    }

    public func clearImages() {
        guard !isSelectedConversationRunning else { return }
        for attachment in imageAttachments { attachmentStore.remove(attachment) }
        imageAttachments.removeAll()
        imageAttachmentError = nil
    }

    private func finishAddingImages(_ staged: [AppImageAttachment]) {
        // Two adds can be in flight at once — the picker and a drop — and each
        // sized itself against the count it saw at admission, so the second to
        // land can push past the cap. Re-check against the real count here and
        // delete what does not fit, rather than leaving staged copies that
        // nothing references.
        defer { addingImagesCount = max(0, addingImagesCount - 1) }
        // The counter keeps `canRun` closed until every batch lands, so a run
        // should not be able to start underneath one. If it ever does, the run
        // has already snapshotted its images: appending here would attach them
        // to the *next* message with no way to take them off, which is worse
        // than saying so. `addImages` refuses a mid-run drop the same way.
        guard !isSelectedConversationRunning else {
            for attachment in staged { attachmentStore.remove(attachment) }
            imageAttachmentError =
                "Wait for the current run to finish before attaching images."
            return
        }
        let capacity = maximumImageAttachments
        let available = max(0, capacity - imageAttachments.count)
        let accepted = staged.prefix(available)
        for attachment in staged.dropFirst(accepted.count) {
            attachmentStore.remove(attachment)
        }
        imageAttachments.append(contentsOf: accepted)
        if accepted.count < staged.count {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
        }
    }

    private func finishAddingImages(error: Error) {
        addingImagesCount = max(0, addingImagesCount - 1)
        imageAttachmentError = String(describing: error)
    }

    public func reloadModel() {
        guard canReloadModel else { return }
        beginLoad()
    }

    public func tuneAutomaticMemory() {
        guard canTuneAutomaticMemory else { return }
        beginLoad(calibrate: true)
    }

    public var canTuneAutomaticMemory: Bool {
        !isAppleSelected && automaticMemory && isModelInstalled && loadState.isReady && !isRunning
            && !isVisionCompanionOperationInProgress && hardwareRequirementsSatisfied(for: selectedInstall)
    }

    private func beginLoad(calibrate: Bool = false) {
        guard !isAppleSelected else { return }
        guard let lifecycle = client as? AppModelLifecycleClient else {
            loadState = .failed(.modelLoadFailed("This client has no model load lifecycle."))
            return
        }
        let directory = URL(fileURLWithPath: modelPathText)
        let maxContext = maxContextTokens
        let forceLogitsHead = currentForceLogitsHead
        let runtimeKey = AppLoadedRuntimeKey(modelDirectory: directory,
                                             maxContextTokens: maxContext,
                                             options: runtimeOptions,
                                             forceLogitsHead: forceLogitsHead)
        // The session is loaded with the same normalized options a run sends.
        // Loading with the raw settings instead meant a control that is off but
        // still carries a non-default value — RDADVISE off with its policy left
        // on `bounded` — produced a loaded session no run could match, and the
        // staleness check compares two normalized keys, so nothing ever offered
        // the reload that would have cleared it.
        let options = runtimeKey.options(prefillEnabled: runtimeOptions.prefillEnabled,
                                        prefillChunkTokens: runtimeOptions.prefillChunkTokens)
        let descriptor = selectedDescriptor
        let device = deviceCapabilities
        let calibration = calibrate ? AppAutomaticMemoryPlanner.layoutSummary(at: directory) : nil
        if calibration != nil { toolStore.inferenceActivity.set(true) }
        let baseline = calibration.flatMap { AppAutomaticMemoryPlanner.plan(for: descriptor,
            on: device, profile: automaticMemoryProfile, layoutSummary: $0) }
        let inferenceClient = client
        let pendingUnload = unloadTask
        loadGeneration &+= 1
        let generation = loadGeneration
        pendingExplicitLoadRuntimeKey = runtimeKey
        error = nil
        appliedLoadSequence = 0
        loadState = .loading(.validatingDirectory)
        let emitted = Mutex<UInt64>(0)
        loadTask = Task.detached { [weak self, lifecycle, pendingUnload] in
            do {
                await pendingUnload?.value
                try Task.checkCancellation()
                try await lifecycle.ensureLoaded(modelDirectory: directory,
                                                 maxContextTokens: maxContext,
                                                 options: options,
                                                 forceLogitsHead: forceLogitsHead) { [weak self] state in
                    if calibration != nil && baseline != nil && state.isReady { return }
                    // Stamped where the phase is emitted, in order; checked
                    // where it is applied, which is not.
                    let sequence = emitted.withLock { value -> UInt64 in
                        value += 1
                        return value
                    }
                    Task { @MainActor in
                        self?.applyLoadState(state, generation: generation,
                                             sequence: sequence)
                    }
                }
                if let calibration, let baseline {
                    let tuningSequence = emitted.withLock { $0 += 1; return $0 }
                    await self?.applyLoadState(.loading(.calibratingAuto), generation: generation, sequence: tuningSequence)
                    let selected = try await AppAutoCalibrator.run(client: inferenceClient, lifecycle: lifecycle,
                        directory: directory, descriptor: descriptor, device: device, summary: calibration,
                        baseline: baseline, options: options, forceLogitsHead: forceLogitsHead)
                    await self?.acceptAutomaticCalibration(selected, directory: directory,
                        options: options, forceLogitsHead: forceLogitsHead, generation: generation)
                }
            } catch is CancellationError {
            } catch let appError as AppInferenceError {
                await self?.applyLoadState(.failed(appError), generation: generation)
            } catch {
                await self?.applyLoadState(
                    .failed(.modelLoadFailed(AppLoadFailureMessage.text(for: error))),
                    generation: generation)
            }
            await self?.clearLoadTask(generation: generation)
        }
    }

    private func acceptAutomaticCalibration(_ plan: AppAutomaticMemoryPlan, directory: URL,
                                             options: AppRuntimeOptions, forceLogitsHead: Bool,
                                             generation: UInt64) {
        guard generation == loadGeneration else { return }
        var selected = options
        selected.expertCacheSlots = plan.expertCacheSlots
        selected.prefillChunkTokens = plan.prefillChunkTokens
        selected.prefillEnabled = plan.expertCacheSlots >= 16
        selected.denseResidencyPolicy = plan.denseResidencyPolicy
        selected.retainedConversationBytes = plan.retainedConversationBytes
        runtimeOptions = selected
        maxContextTokens = plan.contextTokens
        autoWorkloadKey = "calibration-short-t\(ProcessInfo.processInfo.thermalState.rawValue)"
        pendingExplicitLoadRuntimeKey = AppLoadedRuntimeKey(modelDirectory: directory,
            maxContextTokens: plan.contextTokens, options: selected, forceLogitsHead: forceLogitsHead)
        applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0), generation: generation)
    }

    public func cancelLoad() {
        guard canCancelLoad, let lifecycle = client as? AppModelLifecycleClient else { return }
        if case .loading(.calibratingAuto) = loadState { client.cancel() }
        runsAfterCurrentLoad = false
        loadState = .cancelling
        loadGeneration &+= 1
        loadTask?.cancel()
        toolStore.inferenceActivity.set(false)
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
        unloadGeneration &+= 1
        let generation = unloadGeneration
        let directory = URL(fileURLWithPath: modelPathText)
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload(modelDirectory: directory)
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func unloadModel() {
        guard canUnloadModel, let lifecycle = client as? AppModelLifecycleClient else { return }
        runsAfterCurrentLoad = false
        loadState = .unloading
        unloadGeneration &+= 1
        let generation = unloadGeneration
        let directory = URL(fileURLWithPath: modelPathText)
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload(modelDirectory: directory)
            guard let self, generation == self.unloadGeneration else { return }
            if let pooled = lifecycle as? any AppPooledInferenceClient,
               pooled.residentRuntime(forModelDirectory: directory) != nil {
                // The shared host refuses to unload a model another client is
                // still generating on; say so rather than showing it unloaded.
                self.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
                self.error = .invalidRequest(
                    "This model is still in use by another request, so it stays loaded.")
                self.clearUnloadTask(generation: generation)
                return
            }
            self.loadedRuntimeKey = nil
            self.liveMemoryBytes = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func installModel() {
        installModel(selectedInstall)
    }

    public func installModel(_ coordinator: ModelInstallCoordinator) {
        guard canInstallModel(coordinator) else { return }
        coordinator.install()
    }

    public func cancelInstall() {
        selectedInstall.cancel()
    }

    public var hasPartialModelDownload: Bool { selectedInstall.hasPartialDownload }

    public var canDiscardModelDownload: Bool {
        selectedInstall.canDiscard && !isRunning
    }

    public func discardModelDownload() {
        guard !isRunning else { return }
        selectedInstall.discard()
    }

    /// Re-probe every catalog model, so a download that finished in another
    /// model's coordinator and a model installed outside the app both show up.

    public var hasPartialVisionPackDownload: Bool {
        hasPartialVisionPackDownload(for: selectedInstall)
    }

    public func hasPartialVisionPackDownload(
        for coordinator: ModelInstallCoordinator
    ) -> Bool {
        guard let output = try? VisionPackLocation.companionURL(
            forTextModel: coordinator.directoryURL),
              let paths = try? RemoteInstallPaths(outputDirectory: output.path) else {
            return false
        }
        return FileManager.default.fileExists(atPath: paths.partialDirectory)
            || FileManager.default.fileExists(atPath: paths.checkpointFile)
    }

    public var canDiscardVisionPackDownload: Bool {
        canDiscardVisionPackDownload(for: selectedInstall)
    }

    public var canRemoveVisionPack: Bool {
        canRemoveVisionPack(for: selectedInstall)
    }

    public func canDiscardVisionPackDownload(
        for coordinator: ModelInstallCoordinator
    ) -> Bool {
        hasPartialVisionPackDownload(for: coordinator)
            && canMutateVisionPack(for: coordinator)
    }

    public func canRemoveVisionPack(
        for coordinator: ModelInstallCoordinator
    ) -> Bool {
        hasVisionPackDirectory(for: coordinator)
            && canMutateVisionPack(for: coordinator)
    }

    public var hasVisionPackDirectory: Bool {
        hasVisionPackDirectory(for: selectedInstall)
    }

    public func hasVisionPackDirectory(
        for coordinator: ModelInstallCoordinator
    ) -> Bool {
        guard let output = try? VisionPackLocation.companionURL(
            forTextModel: coordinator.directoryURL) else {
            return false
        }
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(
            atPath: output.path,
            isDirectory: &isDirectory) && isDirectory.boolValue
    }

    public func installVisionPack() {
        installVisionPack(for: selectedInstall)
    }

    public func installVisionPack(for coordinator: ModelInstallCoordinator) {
        guard canInstallVisionPack(for: coordinator) else { return }
        visionInstallCancellationRequested = false
        visionInstallTask?.cancel()
        visionInstaller.cancel()
        let textModelDirectory = coordinator.directoryURL.standardizedFileURL
        visionInstallTargetModelID = coordinator.id
        visionInstallTargetDirectory = textModelDirectory
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        visionInstallState = .checking
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                for try await event in visionInstaller.install(
                    textModelDirectory: textModelDirectory) {
                    guard let self else { return }
                    self.applyVisionInstallEvent(event, generation: generation)
                }
                self?.finishVisionInstallStream(generation: generation)
            } catch is CancellationError {
                self?.finishVisionInstallCancellation(generation: generation)
            } catch {
                self?.finishVisionInstallFailure(error, generation: generation)
            }
        }
    }

    public func cancelVisionInstall() {
        guard canCancelVisionInstall else { return }
        visionInstallCancellationRequested = true
        visionInstallState = .cancelling
        visionInstaller.cancel()
        // A cancel raised before the stream registers its own task would find no
        // active install; cancelling the consumer terminates the stream, which
        // routes through the same cooperative drain-to-checkpoint path.
        visionInstallTask?.cancel()
    }

    public func activateVisionPack() {
        guard canActivateVisionPack else { return }
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        visionInstallCancellationRequested = false
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        resetVisionInstallETA()
        visionInstallState = .activating
        visionActivationProgress = 0
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                let output = try await visionInstaller.activatePreparedInstall(
                    textModelDirectory: directory,
                    onVerifyProgress: { [weak self] fraction in
                        Task { @MainActor in
                            guard let self,
                                  generation == self.visionInstallGeneration,
                                  case .activating = self.visionInstallState else { return }
                            // Each hop is its own task, and tasks are not
                            // ordered against each other, so a late one must
                            // not walk the bar backwards.
                            guard fraction >= (self.visionActivationProgress ?? 0)
                            else { return }
                            self.visionActivationProgress = fraction
                        }
                    })
                // Applied first: a progress hop still in flight is dropped
                // once the state is no longer `.activating`, so clearing before
                // this could be undone by a late update.
                self?.applyVisionInstallEvent(
                    .installed(output), generation: generation)
                self?.visionActivationProgress = nil
            } catch is CancellationError {
                // Cancellation can only land during verification, before
                // anything is renamed, so the prepared pack is untouched and
                // still activatable.
                self?.finishVisionActivationCancelled(generation: generation)
                self?.visionActivationProgress = nil
            } catch {
                self?.finishVisionInstallFailure(
                    error, generation: generation, phase: .activation)
                self?.visionActivationProgress = nil
            }
        }
    }

    private func finishVisionActivationCancelled(generation: UInt64) {
        guard generation == visionInstallGeneration else { return }
        resetVisionInstallETA()
        visionInstallTask = nil
        visionInstallCancellationRequested = false
        visionInstallState = .idle
        refreshVisionInstallReadiness()
    }

    public func discardVisionPackDownload() {
        discardVisionPackDownload(for: selectedInstall)
    }

    public func discardVisionPackDownload(
        for coordinator: ModelInstallCoordinator
    ) {
        guard canDiscardVisionPackDownload(for: coordinator) else { return }
        let directory = coordinator.directoryURL.standardizedFileURL
        visionInstallCancellationRequested = false
        visionInstallTargetModelID = coordinator.id
        visionInstallTargetDirectory = directory
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        visionInstallState = .discarding
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                try await visionInstaller.discardPartialInstall(
                    textModelDirectory: directory)
                guard let self, generation == self.visionInstallGeneration else { return }
                self.visionInstallTask = nil
                self.visionInstallState = .idle
                self.visionInstallTargetModelID = nil
                self.visionInstallTargetDirectory = nil
                self.refreshVisionInstallReadiness()
            } catch {
                self?.finishVisionInstallFailure(error, generation: generation)
            }
        }
    }

    /// Drives the confirmation the Model menu puts in front of `removeVisionPack`.
    ///
    /// The Inspector hides its own Remove button once the pack is installed, so
    /// the menu item is the only reachable way to delete 1.14 GB — and it called
    /// straight through, with the dialog sitting on an unreachable branch.
    public func requestVisionPackRemoval() {
        guard canRemoveVisionPack else { return }
        isConfirmingVisionPackRemoval = true
    }

    /// A loaded or selected model cannot be deleted, and nothing may be
    /// generating or mutating companion state. Another model's download is
    /// unaffected, so it does not block.
    public func canDeleteInstalledModel(_ coordinator: ModelInstallCoordinator) -> Bool {
        coordinator.canDeleteInstalled
            && coordinator.id != selectedModelID
            && !deletingModelIDs.contains(coordinator.id)
            && canPrepareVisionCompanionOperation
    }

    /// Asks the Background API to let go of the model, then removes its image
    /// add-on and its folder, then its saved snapshots and an engine nothing
    /// else needs. Callers confirm first; this does not ask.
    public func deleteInstalledModel(_ coordinator: ModelInstallCoordinator) {
        guard canDeleteInstalledModel(coordinator) else { return }
        deletingModelIDs.insert(coordinator.id)
        Task { [weak self, visionInstaller, deletionCleanup] in
            defer { self?.deletingModelIDs.remove(coordinator.id) }
            let descriptor = coordinator.descriptor
            // Deleting files out from under a request the server is answering
            // would fail it partway, so a busy server keeps the model.
            guard await deletionCleanup.releaseFromBackgroundAPI(descriptor.apiModelID) else {
                self?.error = .invalidRequest(
                    "The Background API is using \(descriptor.displayName). "
                    + "Delete it when its requests have finished.")
                return
            }
            do {
                if self?.hasVisionPackDirectory(for: coordinator) == true {
                    try await visionInstaller.removeInstalled(
                        textModelDirectory: coordinator.directoryURL)
                }
                try await coordinator.deleteInstalled()
            } catch {
                self?.error = .invalidRequest(
                    "Could not delete \(descriptor.displayName): \(error)")
                return
            }
            await deletionCleanup.afterDeletion(descriptor)
        }
    }

    public func removeVisionPack() {
        isConfirmingVisionPackRemoval = false
        removeVisionPack(for: selectedInstall)
    }

    public func removeVisionPack(for coordinator: ModelInstallCoordinator) {
        guard canRemoveVisionPack(for: coordinator) else { return }
        let directory = coordinator.directoryURL.standardizedFileURL
        visionInstallCancellationRequested = false
        visionInstallTargetModelID = coordinator.id
        visionInstallTargetDirectory = directory
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        visionInstallState = .discarding
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                try await visionInstaller.removeInstalled(
                    textModelDirectory: directory)
                guard let self, generation == self.visionInstallGeneration else { return }
                self.visionInstallTask = nil
                self.visionInstallState = .idle
                if coordinator.id == self.selectedModelID {
                    self.visionInstallationStatus = .missing
                }
                self.visionInstallTargetModelID = nil
                self.visionInstallTargetDirectory = nil
                self.refreshVisionInstallReadiness()
            } catch {
                self?.finishVisionInstallFailure(error, generation: generation)
            }
        }
    }

    public func refreshInstallReadiness() {
        for coordinator in installs { coordinator.refresh() }
    }

    /// Adopt whatever path the user typed into the location field, then
    /// re-probe. The field is authoritative here — this is the "Check Again"
    /// action after pointing the app at an existing install.
    public func recheckModelAtCurrentLocation() {
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        modelPathText = directory.path
        selectedInstall.setDirectory(directory)
        refreshInstallReadiness()
        refreshVisionInstallReadiness(at: directory)
    }

    public func refreshVisionInstallReadiness() {
        refreshVisionInstallReadiness(
            at: URL(fileURLWithPath: modelPathText, isDirectory: true)
                .standardizedFileURL)
    }

    private func refreshVisionInstallReadiness(at textModelDirectory: URL) {
        visionInstallationStatus = AppVisionPackInstallationProbe.status(
            at: textModelDirectory)
        // Removing the companion leaves any attached image unsendable, and the
        // composer would keep offering it with nothing able to encode it.
        // Only once the dust has settled: the probe verifies the pack on disk,
        // and a companion operation renames that directory underneath it, so
        // refreshing mid-operation can briefly report no image support. Acting
        // on that would delete images the user had staged.
        if !isImageInputAvailable, !isVisionCompanionOperationInProgress,
           !imageAttachments.isEmpty {
            for attachment in imageAttachments { attachmentStore.remove(attachment) }
            imageAttachments.removeAll()
            // Say so. Clearing the error alongside the images removed them and
            // the only explanation for their absence in one step, so the
            // composer just quietly emptied itself.
            imageAttachmentError =
                "Image support is unavailable, so the attached images were removed."
        }
        guard isModelInstalled else {
            visionInstallReadiness = .failed("Install the text model first")
            return
        }
        guard !isVisionPackInstalled else { return }
        if visionInstaller.preparedInstallIsValid(
            textModelDirectory: textModelDirectory) {
            let output = try? VisionPackLocation.companionURL(
                forTextModel: textModelDirectory)
            // A pack that failed to activate must not be re-offered for
            // activation: `preparedInstallIsValid` does not hash the weights,
            // so a corrupt pack still looks ready and the user would loop
            // between Activate and the same failure.
            let reportedBroken: Bool
            switch visionInstallState {
            case .recoverable, .failed: reportedBroken = true
            default: reportedBroken = false
            }
            if let output, !isInstallingVisionPack, !reportedBroken {
                visionInstallTargetModelID = selectedModelID
                visionInstallTargetDirectory = textModelDirectory
                visionInstallState = .readyToActivate(output)
            }
        }
        visionInstallReadiness = .checking
        do {
            let requirement = try checkedVisionInstallRequirement(
                textModelDirectory: textModelDirectory)
            visionInstallReadiness = requirement.canInstall
                ? .ready(requirement)
                : .insufficientSpace(requirement)
        } catch {
            visionInstallReadiness = .failed("\(error)")
        }
    }

    private var textInstallOutstandingBytes: UInt64 {
        installs.reduce(UInt64(0)) { partial, install in
            let addition = partial.addingReportingOverflow(install.outstandingBytes)
            return addition.overflow ? .max : addition.partialValue
        }
    }

    private func checkedVisionInstallRequirement(
        textModelDirectory: URL
    ) throws -> AppModelInstallRequirement {
        let measured = try visionInstaller.checkInstallRequirement(
            textModelDirectory: textModelDirectory)
        let reserved = textInstallOutstandingBytes
        return AppModelInstallRequirement(
            probePath: measured.probePath,
            requiredBytes: measured.requiredBytes,
            availableBytes: measured.availableBytes > reserved
                ? measured.availableBytes - reserved : 0)
    }

    private func canMutateVisionPack(
        for coordinator: ModelInstallCoordinator
    ) -> Bool {
        canPrepareVisionCompanionOperation
            && (coordinator.id != selectedModelID || !loadState.isReady)
    }

    private func applyVisionInstallEvent(
        _ event: AppModelInstallEvent,
        generation: UInt64
    ) {
        guard generation == visionInstallGeneration else { return }
        if visionInstallCancellationRequested {
            switch event {
            case .readyToActivate, .installed:
                // Work that finished before the cancel landed is reported as it
                // actually ended, not as a pause.
                visionInstallCancellationRequested = false
            default:
                return
            }
        }
        switch event {
        case .checking:
            resetVisionInstallETA()
            visionInstallState = .checking
        case .downloadingMetadata:
            resetVisionInstallETA()
            visionInstallState = .downloadingMetadata
        case .planning:
            resetVisionInstallETA()
            visionInstallState = .planning
        case .reservingOutput:
            resetVisionInstallETA()
            visionInstallState = .reservingOutput
        case .copyingPayload(let reused, let downloaded, let total):
            visionInstallState = .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloaded,
                totalBytes: total)
            updateVisionInstallETA(
                reusedBytes: reused,
                downloadedThisRunBytes: downloaded,
                totalBytes: total)
        case .hashingOutput(let file):
            resetVisionInstallETA()
            visionInstallState = .hashingOutput(file)
        case .finalizing:
            resetVisionInstallETA()
            visionInstallState = .finalizing
        case .readyToActivate(let directory):
            resetVisionInstallETA()
            visionInstallState = .readyToActivate(directory)
            visionInstallTask = nil
        case .installed:
            resetVisionInstallETA()
            let textModelDirectory = visionInstallTargetDirectory
                ?? URL(fileURLWithPath: modelPathText, isDirectory: true)
                    .standardizedFileURL
            let status = AppVisionPackInstallationProbe.status(at: textModelDirectory)
            if visionInstallTargetModelID == selectedModelID {
                visionInstallationStatus = status
            }
            guard status == .complete else {
                finishVisionInstallFailure(
                    RepackError.configurationInvalid(
                        detail: "completed vision install failed verification"),
                    generation: generation)
                return
            }
            visionInstallState = .installed(modelDirectory: textModelDirectory)
            visionInstallTask = nil
        }
    }

    private func finishVisionInstallStream(generation: UInt64) {
        guard generation == visionInstallGeneration,
              visionInstallTask != nil else { return }
        if visionInstallCancellationRequested || visionInstallState == .cancelling {
            finishVisionInstallCancellation(generation: generation)
        } else if !isVisionPackInstalled {
            finishVisionInstallFailure(
                RepackError.configurationInvalid(
                    detail: "vision installer ended before completion"),
                generation: generation)
        }
    }

    private func finishVisionInstallCancellation(generation: UInt64) {
        guard generation == visionInstallGeneration else { return }
        resetVisionInstallETA()
        visionInstallCancellationRequested = false
        visionInstallTask = nil
        visionInstallState = .cancelled
        if visionInstallTargetModelID == selectedModelID {
            refreshVisionInstallReadiness()
        }
    }

    /// Which phase failed. Only a download failure may leave a prepared pack
    /// that is worth activating; a verification failure must never send the
    /// user back to Activate, or the same corrupt pack is offered forever.
    enum VisionFailurePhase { case download, activation }

    func finishVisionInstallFailure(
        _ error: Error, generation: UInt64,
        phase: VisionFailurePhase = .download
    ) {
        guard generation == visionInstallGeneration else { return }
        // An error raised because the user cancelled is a pause with saved
        // progress, not an installation failure.
        guard !visionInstallCancellationRequested else {
            finishVisionInstallCancellation(generation: generation)
            return
        }
        resetVisionInstallETA()
        visionInstallTask = nil
        let textModelDirectory = visionInstallTargetDirectory
            ?? URL(fileURLWithPath: modelPathText, isDirectory: true)
                .standardizedFileURL
        let target = installs.first { $0.id == visionInstallTargetModelID }
        let hasSavedDownload = target.map(hasPartialVisionPackDownload(for:))
            ?? hasPartialVisionPackDownload
        // A download that finished and verifies is activatable whatever went
        // wrong afterwards. Reporting it as "needs attention" hid the Activate
        // button behind a Resume that only repeats work already done. The one
        // failure that must not come back here is verification itself, or the
        // same corrupt pack is offered forever — but a lock held by another
        // process is contention, not corruption.
        let isContention: Bool
        if let repackError = error as? RepackError, case .installBusy = repackError {
            isContention = true
        } else {
            isContention = false
        }
        if phase == .download || isContention, hasSavedDownload,
           let output = try? VisionPackLocation.companionURL(
            forTextModel: textModelDirectory),
            visionInstaller.preparedInstallIsValid(
            textModelDirectory: textModelDirectory) {
            visionInstallState = .readyToActivate(output)
            if visionInstallTargetModelID == selectedModelID {
                refreshVisionInstallReadiness(at: textModelDirectory)
            }
            return
        }
        visionInstallState = hasSavedDownload
            ? .recoverable("\(error)")
            : .failed("\(error)")
        if let repackError = error as? RepackError,
           case .diskSpaceInsufficient(let path, let required, let available) = repackError {
            visionInstallReadiness = .insufficientSpace(AppModelInstallRequirement(
                probePath: path,
                requiredBytes: required,
                availableBytes: available))
        } else {
            if visionInstallTargetModelID == selectedModelID {
                refreshVisionInstallReadiness()
            }
            if hasSavedDownload {
                visionInstallState = .recoverable("\(error)")
            }
        }
    }

    private var installETATimestamp: Double {
        let components = installETAOrigin.duration(to: installETAClock.now).components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func updateVisionInstallETA(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    ) {
        let observation = DownloadETAObservation(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: downloadedThisRunBytes,
            totalBytes: totalBytes)
        let presentation = visionInstallETAEstimator.update(
            observation, timestamp: installETATimestamp)
        visionInstallETAPresentation = presentation
        visionInstallETAText = DownloadETAFormatter.string(for: presentation)
    }

    private func resetVisionInstallETA() {
        visionInstallETAEstimator.reset()
        visionInstallETAPresentation = .hidden
        visionInstallETAText = nil
    }

    /// Loads app-wide settings and adopts either the cached or on-disk profile.
    private func applyPersistedSettings(forModelDirectory modelDirectory: URL,
                                        adoptingStoredProfile: Bool) {
        if settingsPersistenceEnabled {
            let profileKey = selectedProviderKey
            let settings = MacAppSettingsFileStore.loadOrCreate(
                forModelDirectory: modelDirectory,
                profileKey: profileKey)
            if !adoptingStoredProfile || settingsStore.modelProfiles[profileKey] == nil {
                settingsStore.modelProfiles[profileKey] = settings.profile(for: profileKey)
            }
            settingsStore.newlineShortcut = settings.newlineShortcut
            settingsStore.showPromptExamples = settings.showPromptExamples
            settingsStore.sentPromptBehavior = settings.sentPromptBehavior
            settingsStore.loadModelOnLaunch = settings.loadModelOnLaunch
            settingsStore.smoothStreaming = settings.smoothStreaming
            settingsStore.appearanceMode = settings.appearanceMode
            settingsStore.translucentWindow = settings.translucentWindow
            settingsStore.accentColorMode = settings.accentColorMode
            customAccentColorHex = settings.customAccentColorHex
            settingsStore.zoomLevel = settings.zoomLevel
            settingsStore.bypassModelRestrictions = settings.bypassModelRestrictions
            settingsStore.webSearchEnabled = settings.webSearchEnabled
            settingsStore.pauseWebSearchWhenOffline = settings.pauseWebSearchWhenOffline
            settingsStore.runWebInSandbox = settings.runWebInSandbox
            settingsStore.runFilesInSandbox = settings.runFilesInSandbox
            toolStore.indexInSandbox = settings.runFilesInSandbox
            settingsStore.fileSearchEnabled = settings.fileSearchEnabled
            settingsStore.codingWriteEnabled = settings.codingWriteEnabled
            settingsStore.codingCommandEnabled = settings.codingCommandEnabled
            settingsStore.codingNetworkEnabled = settings.codingNetworkEnabled
            settingsStore.codingFolderID = settings.codingFolderID
            settingsStore.searchProvider = settings.searchProvider
        }
        applySettingsProfile(storedSettingsProfile(for: selectedInstall))
    }

    private var currentSettingsProfile: AppModelSettingsProfile {
        AppModelSettingsProfile(
            automaticMemory: automaticMemory,
            automaticMemoryProfile: automaticMemoryProfile,
            contextTokens: maxContextTokens,
            expertCacheSlots: runtimeOptions.expertCacheSlots,
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            prefillEnabled: runtimeOptions.prefillEnabled,
            visionResidencyPolicy: runtimeOptions.visionResidencyPolicy,
            rdadvisePolicy: runtimeOptions.rdadvisePolicy,
            defaultReasoning: reasoning,
            defaultReasoningEffort: reasoningEffort,
            preserveThinking: preserveThinking,
            systemPrompt: modelSystemPrompt, denseResidentLayerCount: {
                if case .bounded(let layers) = runtimeOptions.denseResidencyPolicy { return layers.count }
                return nil
            }())
    }

    private func applySettingsProfile(_ profile: AppModelSettingsProfile) {
        var profile = AppAutomaticMemoryPlanner.applying(
            profile,
            for: selectedDescriptor,
            on: deviceCapabilities, modelDirectory: URL(fileURLWithPath: modelPathText), workloadKey: autoWorkloadKey,
            useBundledLayout: settingsPersistenceEnabled,
            engineIdentity: (client as? any AppInferenceEngineReporting)?.currentEngineIdentity)
        var autoPlan = profile.automaticMemory ? AppAutomaticMemoryPlanner.plan(
            for: selectedDescriptor, on: deviceCapabilities, profile: profile.automaticMemoryProfile, workloadKey: autoWorkloadKey,
            modelDirectory: URL(fileURLWithPath: modelPathText), useBundledLayout: settingsPersistenceEnabled,
            engineIdentity: (client as? any AppInferenceEngineReporting)?.currentEngineIdentity) : nil
        let conversationReserve = contextUsage.estimatedTokens + (maxNewTokensOverride ?? 1_024)
            + (effectiveChatCapabilities.isEmpty ? 256 : 2_048)
        if let baseline = autoPlan, baseline.contextTokens < conversationReserve,
           let summary = AppAutomaticMemoryPlanner.layoutSummary(at: URL(fileURLWithPath: modelPathText)) {
            autoPlan = AppContextLengthOption.options(for: selectedDescriptor).map(\.tokens).sorted()
                .lazy.filter { $0 >= conversationReserve }.compactMap {
                    AppAutomaticMemoryPlanner.plan(for: self.selectedDescriptor, on: self.deviceCapabilities,
                        profile: profile.automaticMemoryProfile, layoutSummary: summary,
                        workloadKey: self.autoWorkloadKey, modelDirectory: URL(fileURLWithPath: self.modelPathText),
                        contextTokens: $0, engineIdentity: (self.client as? any AppInferenceEngineReporting)?.currentEngineIdentity)
                }.first ?? baseline
            if let autoPlan {
                profile.contextTokens = autoPlan.contextTokens
                profile.expertCacheSlots = autoPlan.expertCacheSlots
                profile.prefillEnabled = autoPlan.expertCacheSlots >= 16
            }
        }
        runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: profile.expertCacheSlots,
            prefillEnabled: profile.prefillEnabled,
            prefillChunkTokens: autoPlan?.prefillChunkTokens ?? selectedDescriptor
                .recommendedPrefillChunkTokens(on: deviceCapabilities),
            rdadvisePolicy: profile.rdadvisePolicy,
            // Pinned for the same reason as `init`: the app always releases the
            // image tower. Reading the persisted value here would let a
            // `keepReady` written by an older build resurrect ~1 GB of resident
            // tower on a machine with no control that shows or clears it.
            visionResidencyPolicy: .onDemand,
            denseResidencyPolicy: autoPlan?.denseResidencyPolicy ?? profile.manualDenseResidencyPolicy(for: selectedDescriptor),
            retainedConversationBytes: autoPlan?.retainedConversationBytes)
        automaticMemory = profile.automaticMemory
        automaticMemoryProfile = profile.automaticMemoryProfile
        maxContextTokens = profile.contextTokens
        temperature = profile.temperature
        topKEnabled = profile.topKEnabled
        topK = profile.topK
        topPEnabled = profile.topPEnabled
        topP = profile.topP
        reasoning = selectedDescriptor.reasoningControl == .alwaysOn
            ? .on : profile.defaultReasoning
        reasoningEffort = profile.defaultReasoningEffort
        preserveThinking = profile.preserveThinking
        settingsStore.systemPrompt = profile.systemPrompt
    }

    /// Coalesces the writes behind a control that changes continuously.
    ///
    /// Every save rewrites the whole file — a read to merge the profiles of
    /// the models that are not selected, then an atomic replace — and the
    /// Settings sliders call one per tick while the system prompt editor calls
    /// one per keystroke, all on the main actor. The in-memory state still
    /// changes immediately; only the disk write waits, and because each save
    /// serialises whatever is current rather than a captured snapshot, a
    /// pending one can never write back a stale value.
    ///
    /// `flushPendingSettings` runs on quit so the last edit is not lost.
    private func persistSettingsSoon() {
        guard settingsPersistenceEnabled else { return }
        pendingSettingsSave?.cancel()
        pendingSettingsSave = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled, let self else { return }
            self.pendingSettingsSave = nil
            self.persistSettings()
        }
    }

    /// Writes whatever `persistSettingsSoon` is still holding, now.
    public func flushPendingSettings() {
        guard pendingSettingsSave != nil || !unsavedProfileKeys.isEmpty else { return }
        pendingSettingsSave?.cancel()
        pendingSettingsSave = nil
        persistSettings()
    }

    private func persistSettings() {
        // Keep the profile cache current even when persistence is disabled.
        if !isAppleSelected { captureSelectedSettingsProfile() }
        guard settingsPersistenceEnabled else {
            unsavedProfileKeys.removeAll()
            return
        }
        let modelDirectory = URL(fileURLWithPath: modelPathText, isDirectory: true)
        let profileKey = selectedDescriptor.settingsProfileKey
        var settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory,
            profileKey: profileKey)
        settings.setProfile(storedSettingsProfile(for: selectedInstall), for: profileKey)
        settings.newlineShortcut = newlineShortcut
        settings.showPromptExamples = showPromptExamples
        settings.sentPromptBehavior = sentPromptBehavior
        settings.loadModelOnLaunch = loadModelOnLaunch
        settings.smoothStreaming = smoothStreaming
        settings.appearanceMode = appearanceMode
        settings.translucentWindow = translucentWindow
        settings.accentColorMode = accentColorMode
        settings.customAccentColorHex = customAccentColorHex
        settings.zoomLevel = zoomLevel
        settings.bypassModelRestrictions = bypassModelRestrictions
        settings.webSearchEnabled = settingsStore.webSearchEnabled
        settings.pauseWebSearchWhenOffline = settingsStore.pauseWebSearchWhenOffline
        settings.runWebInSandbox = settingsStore.runWebInSandbox
        settings.runFilesInSandbox = settingsStore.runFilesInSandbox
        settings.fileSearchEnabled = settingsStore.fileSearchEnabled
        settings.codingWriteEnabled = settingsStore.codingWriteEnabled
        settings.codingCommandEnabled = settingsStore.codingCommandEnabled
        settings.codingNetworkEnabled = settingsStore.codingNetworkEnabled
        settings.codingFolderID = settingsStore.codingFolderID
        settings.searchProvider = settingsStore.searchProvider
        if (try? MacAppSettingsFileStore.save(
            settings,
            forModelDirectory: modelDirectory)) != nil {
            unsavedProfileKeys.remove(profileKey)
        }
        persistUnsavedProfiles(excluding: profileKey)
    }

    /// Merges dirty non-selected profiles into their own settings files.
    private func persistUnsavedProfiles(excluding selectedKey: String) {
        let pending = unsavedProfileKeys.subtracting([selectedKey])
        guard !pending.isEmpty else { return }
        for coordinator in installs
        where pending.contains(coordinator.descriptor.settingsProfileKey) {
            let profileKey = coordinator.descriptor.settingsProfileKey
            var settings = MacAppSettingsFileStore.loadOrCreate(
                forModelDirectory: coordinator.directoryURL,
                profileKey: profileKey)
            settings.setProfile(
                storedSettingsProfile(for: coordinator), for: profileKey)
            if (try? MacAppSettingsFileStore.save(
                settings,
                forModelDirectory: coordinator.directoryURL)) != nil {
                unsavedProfileKeys.remove(profileKey)
            }
        }
    }

    private static func preservingManualMemorySettings(
        in effective: AppModelSettingsProfile,
        from saved: AppModelSettingsProfile
    ) -> AppModelSettingsProfile {
        guard effective.automaticMemory else { return effective }
        var stored = effective
        stored.contextTokens = saved.contextTokens
        stored.expertCacheSlots = saved.expertCacheSlots
        stored.prefillEnabled = saved.prefillEnabled
        stored.denseResidentLayerCount = saved.denseResidentLayerCount
        return stored
    }

    func applyLoadState(_ state: AppModelLoadState) {
        applyLoadState(state, generation: loadGeneration)
    }

    /// `sequence` orders the phases a load emits. It is 0 for states this model
    /// raises itself, which bypass the ordering check.
    func applyLoadState(_ state: AppModelLoadState, generation: UInt64,
                        sequence: UInt64 = 0) {
        guard generation == loadGeneration else { return }
        if sequence > 0 {
            guard sequence > appliedLoadSequence else { return }
            appliedLoadSequence = sequence
        }
        if case .ready(let directory, _) = state,
           directory.standardizedFileURL.path
            != URL(fileURLWithPath: modelPathText).standardizedFileURL.path {
            return
        }
        loadState = state
        // An outcome closes the load. Phases emitted before it but delivered
        // after it must not reopen one that has already finished: `.failed` is
        // raised here at sequence 0, so it never advanced the counter, and a
        // late `.loading` hop could put the UI back into a load with no task
        // left to cancel and no way to start another. `beginLoad` resets the
        // counter, so the seal lasts exactly one load.
        switch state {
        case .notLoaded, .ready, .failed:
            appliedLoadSequence = .max
        case .loading, .cancelling, .unloading:
            break
        }
        switch state {
        case .notLoaded:
            loadedRuntimeKey = nil
        case .loading, .cancelling, .unloading:
            break
        case .ready(_, let seconds):
            loadedRuntimeKey = pendingExplicitLoadRuntimeKey
                ?? activeRunRuntimeKey
                ?? currentRuntimeKey
            pendingExplicitLoadRuntimeKey = nil
            // The freshly loaded model's footprint, so the figure is right
            // before the first generation rather than after it.
            sampleLiveMemory()
            _ = seconds
        case .failed(let loadError):
            pendingExplicitLoadRuntimeKey = nil
            runsAfterCurrentLoad = false
            error = loadError
        }
        if case .ready = state, runsAfterCurrentLoad {
            runsAfterCurrentLoad = false
            run()
        }
    }

    /// Clear the transcript and start a fresh conversation. The next prompt
    /// carries no history, so the model starts from nothing.

    /// Releases the transcript's own references to the images this chat was
    /// showing. They are separate files from the composer's, so nothing else
    /// frees them — and they outlive the message they were sent with, because
    /// every earlier turn in the chat is still drawing its own.
    private func releaseTranscriptImages() {
        for attachment in retainedRunImages { attachmentStore.remove(attachment) }
        retainedRunImages = []
        outputImageAttachments = []
    }

    private var retainedRunImages: [AppImageAttachment] {
        get { conversationStore.retainedRunImages }
        set { conversationStore.retainedRunImages = newValue }
    }

    /// Deletes every file this session staged. Called when the app is quitting,
    /// which is the only moment they are all certainly unwanted.
    public func releaseAllAttachments() {
        releaseTranscriptImages()
        for attachment in imageAttachments { attachmentStore.remove(attachment) }
        imageAttachments.removeAll()
        attachmentStore.removeAll()
    }

    /// Memory comes from the process doing the work: the decode service when
    /// there is one, this process otherwise.
    private func sampleLiveMemory() {
        if let reporter = client as? any AppInferenceMemoryReporting {
            if let bytes = reporter.currentInferenceMemoryBytes {
                liveMemoryBytes = bytes
            }
            if let resident = reporter.currentInferenceResidentBytes {
                liveResidentBytes = resident
            }
            // Refreshed on every sample, so the tower figure tracks a run
            // instead of appearing only in its final diagnostics.
            if let tower = reporter.currentInferenceTowerBytes {
                visionTowerMappedBytes = tower
            }
        } else {
            liveMemoryBytes = memorySampler.sample()
            // The occupied figure, not the footprint again: this row is the one
            // that includes the mapped weights, and feeding it the footprint
            // made both numbers report the same thing.
            liveResidentBytes = memorySampler.occupiedSample()
        }
    }

    /// Resident bytes for display, on the same terms as
    /// `currentProcessMemoryBytes`.
    public var currentProcessResidentBytes: UInt64? {
        guard loadState.isReady || isRunning else { return nil }
        if let liveResidentBytes { return liveResidentBytes }
        if let reporter = client as? any AppInferenceMemoryReporting {
            return reporter.currentInferenceResidentBytes
        }
        return memorySampler.occupiedSample()
    }

    public func clearOutput() {
        if isRunning {
            beginNewConversation(modelID: selectedProviderKey)
            // The new chat is idle; show its state, not the running chat's.
            diagnostics = nil
            error = nil
            restoreSelectedRunPresentation()
            return
        }
        releaseTranscriptImages()
        beginNewConversation(modelID: selectedProviderKey)
        if !(client is any AppSessionInferenceClient) { generationTranscriptMailbox?.reset() }
        diagnostics = nil
        error = nil
    }

    private func beginNewConversation(modelID: String) {
        conversationStore.startNewConversation(modelID: modelID)
    }

    /// Whether this turn sends an image to a session whose expert cache is
    /// too small for the chunked prefill images need.
    var imageTurnNeedsLargerExpertCache: Bool {
        guard selectedDescriptor.usesExpertCache,
              (loadedRuntimeKey ?? currentRuntimeKey).expertCacheSlots
                < RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill else { return false }
        return !imageAttachments.isEmpty || conversation.contains { !$0.images.isEmpty }
    }

    public func run() {
        if isAppleSelected { refreshAppleAvailability() }
        guard canRun else { return }
        if !isAppleSelected {
            if automaticMemory, let summary = AppAutomaticMemoryPlanner.layoutSummary(at: URL(fileURLWithPath: modelPathText)) {
                // Leave room for the answer and tool results before selecting a
                // larger window. The helper checks the actual tokenizer count.
                let reserve = maxNewTokensOverride ?? 1_024
                let required = contextUsage.estimatedTokens + reserve
                    + ((effectiveChatCapabilities.web || effectiveChatCapabilities.files) ? 2_048 : 256)
                if required > effectiveMaxContextTokens {
                    guard !selectedModelIsRunning else {
                        error = .invalidRequest("This chat needs a larger context. Wait for the other chat to finish before changing the shared model runtime.")
                        return
                    }
                    let choices = AppContextLengthOption.options(for: selectedDescriptor).map(\.tokens).sorted()
                    let plan = choices.lazy.filter { $0 >= required }.compactMap {
                        AppAutomaticMemoryPlanner.plan(for: self.selectedDescriptor, on: self.deviceCapabilities,
                            profile: self.automaticMemoryProfile, layoutSummary: summary, workloadKey: self.autoWorkloadKey,
                            modelDirectory: URL(fileURLWithPath: self.modelPathText), contextTokens: $0,
                            engineIdentity: (self.client as? any AppInferenceEngineReporting)?.currentEngineIdentity)
                    }.first
                    guard let plan else {
                        error = .invalidRequest("The full conversation and answer reserve cannot fit safely on this Mac. Reduce the answer limit or start a new chat; no history was removed.")
                        return
                    }
                    maxContextTokens = plan.contextTokens
                    runtimeOptions.expertCacheSlots = plan.expertCacheSlots
                    runtimeOptions.prefillChunkTokens = plan.prefillChunkTokens
                    runtimeOptions.prefillEnabled = plan.expertCacheSlots >= 16
                    runtimeOptions.denseResidencyPolicy = plan.denseResidencyPolicy
                    runtimeOptions.retainedConversationBytes = plan.retainedConversationBytes
                    runsAfterCurrentLoad = true
                    beginLoad()
                    return
                }
            }
            // Image spans need chunked prefill, and chunked prefill needs 16 expert
            // slots. Auto can settle below that to avoid paging, and the image then
            // failed partway through its prompt. Auto reloads with a cache that
            // can take it; a manual setting is the user's to change.
            if imageTurnNeedsLargerExpertCache {
                guard !selectedModelIsRunning else {
                    error = .invalidRequest("Image input needs a different runtime configuration. Wait for the other chat to finish before reloading the model.")
                    return
                }
                let directory = URL(fileURLWithPath: modelPathText)
                // The installed layout when it can be read, else the catalog's.
                if automaticMemory,
                   let plan = AppAutomaticMemoryPlanner.plan(for: selectedDescriptor, on: deviceCapabilities,
                        profile: automaticMemoryProfile, workloadKey: autoWorkloadKey,
                        modelDirectory: directory, useBundledLayout: true, contextTokens: effectiveMaxContextTokens,
                        engineIdentity: (client as? any AppInferenceEngineReporting)?.currentEngineIdentity,
                        imageInput: true),
                   plan.expertCacheSlots >= RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill {
                    maxContextTokens = plan.contextTokens
                    runtimeOptions.expertCacheSlots = plan.expertCacheSlots
                    runtimeOptions.prefillChunkTokens = plan.prefillChunkTokens
                    runtimeOptions.prefillEnabled = true
                    runtimeOptions.denseResidencyPolicy = plan.denseResidencyPolicy
                    runtimeOptions.retainedConversationBytes = plan.retainedConversationBytes
                    runsAfterCurrentLoad = true
                    beginLoad()
                    return
                }
                error = .invalidRequest(automaticMemory
                    ? "Images need at least \(RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill) expert cache slots, and they don't fit safely on this Mac right now. Close other apps and try again, or remove the image."
                    : "Images need at least \(RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill) expert cache slots. Raise Slots to \(RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill) in this model's settings, or remove the image.")
                return
            }
            if automaticMemory, let loadedRuntimeKey, loadedRuntimeKey != currentRuntimeKey {
                runsAfterCurrentLoad = true
                beginLoad()
                return
            }
        }
        var request: AppGenerationRequest
        let capabilities = effectiveChatCapabilities
        do {
            request = try makeRequest(capabilities: capabilities)
        } catch let appError as AppInferenceError {
            pendingAssistantPrefix = nil
            error = appError
            return
        } catch {
            pendingAssistantPrefix = nil
            let appError = AppInferenceError.unknown("\(error)")
            self.error = appError
            return
        }

        // The run reads the transcript's own hard links rather than the
        // composer's files, so clearing the composer below cannot delete an
        // image this request has not opened yet. One set of files, one owner.
        // A failed retain leaves no reference that is guaranteed to outlive
        // the composer, so the run is refused instead of started against files
        // that are about to be removed.
        var retained: [AppImageAttachment] = []
        do {
            for attachment in request.imageAttachments {
                retained.append(try attachmentStore.retain(attachment))
            }
        } catch {
            for attachment in retained { attachmentStore.remove(attachment) }
            imageAttachmentError = String(describing: error)
            self.error = .invalidRequest(
                "Could not prepare the attached images for this run: \(error)")
            return
        }
        request.imageAttachments = retained

        persistSettings()

        if !(client is any AppSessionInferenceClient) { generationTranscriptMailbox?.reset() }
        runIdentity &+= 1
        conversationStore.outputAssistantPrefix = request.assistantPrefix
        pendingAssistantPrefix = nil
        conversationStore.outputTurnIsRecorded = false
        // The transcript shows what was typed, not the flattened text the
        // model reads: the attached files are drawn as files beside it.
        outputPromptText = promptText
        outputImageAttachments = retained
        retainedRunImages.append(contentsOf: retained)
        outputDocumentAttachments = documentAttachments
        outputText = ""
        outputThinkingText = ""
        conversationStore.clearOutputTools()
        conversationStore.outputCapabilities = capabilities
        diagnostics = nil
        error = nil
        let runRuntimeKey = AppLoadedRuntimeKey(
            modelDirectory: request.modelDirectory,
            maxContextTokens: request.maxContextTokens,
            options: request.runtimeOptions,
            forceLogitsHead: !request.isPureGreedy)
        liveTokenCount = 0
        liveElapsedDecodeSeconds = 0
        livePrefillDone = 0
        livePrefillTotal = 0
        sampleLiveMemory()
        phase = .prefill
        runState = .running
        // The chat exists from here, not from the first finished answer. This
        // names it, puts it at the top of the sidebar, and writes the message to
        // disk before a single token is generated, so stopping the model — or
        // quitting — leaves the question that was asked rather than nothing.
        conversationStore.beginTurn(
            AppChatTurn(
                prompt: outputPromptText,
                response: "",
                documents: outputDocumentAttachments,
                images: retained,
                modelID: selectedProviderKey,
                capabilities: capabilities),
            attachments: retained,
            modelID: selectedProviderKey)
        guard let conversationID = conversationStore.selectedConversationID else {
            error = .unknown("Could not create a chat for this answer.")
            runState = activeAnswerRuns.isEmpty ? .idle : .running
            return
        }
        conversationStore.outputAssistantPrefix = request.assistantPrefix
        // The chat exists now, so its identity is known even for a first
        // message.
        request.conversationKey = conversationStore.selectedConversationID?.uuidString
        let firstSourceID = conversationStore.nextSourceID
        let userText = promptText
        let hasLocalContext = !outputDocumentAttachments.isEmpty || !retained.isEmpty
            || conversation.contains { !$0.documents.isEmpty || !$0.images.isEmpty }
        if sentPromptBehavior == .clear {
            promptText = ""
            documentAttachments.removeAll()
            documentAttachmentNotice = nil
            // Images used to survive a cleared prompt, so the next Run silently
            // re-attached them to a completely different message. Dropping them
            // is safe because the request above was repointed at the retained
            // links: removing these files cannot pull the ground out from under
            // a run that has not opened its images yet.
            for attachment in imageAttachments { attachmentStore.remove(attachment) }
            imageAttachments.removeAll()
            imageAttachmentError = nil
        }

        let scopedClient: any AppInferenceClient
        if isAppleSelected {
            scopedClient = AppleInferenceClient(kind: selectedAppleModel ?? .onDevice)
        } else if let sessionClient = client as? any AppSessionInferenceClient {
            scopedClient = sessionClient.scopedSession(conversationStore.sessionID(for: conversationID))
        } else {
            scopedClient = client
        }
        let answerRun = ActiveAnswerRun(
            conversationID: conversationID, runtimeKey: runRuntimeKey,
            modelProfileKey: selectedProviderKey,
            scopedClient: scopedClient)
        activeAnswerRuns[conversationID] = answerRun
        runState = .running
        let webInSandbox = settingsStore.runWebInSandbox
        #if os(macOS)
        // After an idle stop or a failed start the sandbox is started again
        // here, before any tool runs; a web request that finds it still
        // starting waits for it. Nothing else starts it.
        if webInSandbox, capabilities.web { startSandbox() }
        #endif
        let workspace = selectedCodingFolder?.path
        var toolLimits = AppToolLimits.standard
        if capabilities.fileWrite || capabilities.command { toolLimits.maximumToolRounds = 12 }
        let runner = AppAnswerRunner(
            client: scopedClient,
            toolbox: toolStore.makeToolbox(provider: settingsStore.searchProvider, limits: toolLimits,
                                       webInSandbox: webInSandbox, workspace: workspace,
                                       workspaceWrite: capabilities.fileWrite, commandNetwork: capabilities.commandNetwork,
                                       codingAllowed: { [weak self] call in
                await MainActor.run {
                    guard let self, self.selectedCodingFolder?.path == workspace else { return false }
                    let current = self.effectiveChatCapabilities
                    switch call {
                    case .listFiles, .readFile: return current.workspaceRead
                    case .writeFile, .editFile: return current.fileWrite
                    case .runCommand:
                        return current.command && (!capabilities.fileWrite || current.fileWrite)
                            && (!capabilities.commandNetwork || current.commandNetwork)
                    default: return false
                    }
                }
            }), limits: toolLimits)
        let activity = toolStore.inferenceActivity
        activity.set(true)
        answerRun.observationStart = .begin()
        let answerRequest = request
        let cancellation = answerRun.cancellation
        answerRun.task = Task.detached { [weak self, runner, answerRequest] in
            let model = self
            await runner.run(answerRequest, capabilities: capabilities,
                             firstSourceID: firstSourceID, userText: userText,
                             hasLocalContext: hasLocalContext, cancellation: cancellation) { [weak model] event in
                await model?.apply(answer: event, for: conversationID)
            }
            // The terminal event clears activity on the main actor. Clearing
            // it again here can race with a new answer that has just started.
        }
    }

    public func submit() {
        guard canSubmit else { return }
        if canRun {
            run()
        } else {
            runsAfterCurrentLoad = true
            loadModel()
        }
    }

    public func cancel() {
        guard canCancel else { return }
        guard let conversationID = conversationStore.selectedConversationID,
              let run = activeAnswerRuns[conversationID] else { return }
        isCancellationPending = true
        run.cancellation.cancel()
        run.scopedClient.cancel()
    }

    /// What the model reads for the message being composed: attached files
    /// first, then what was typed.
    public var composedPromptText: String {
        let preamble = documentAttachments.promptPreamble
        if preamble.isEmpty { return promptText }
        if promptText.isEmpty { return preamble }
        return preamble + "\n\n" + promptText
    }

    // MARK: - Rewinding a conversation

    /// Whether a recorded message can be put back in the composer or sent
    /// again. Both rewind the chat, so both need it to be idle.
    public func canRewind(to turn: AppChatTurn) -> Bool {
        !isSelectedConversationRunning && !loadState.isLoading
            && conversation.contains { $0.id == turn.id }
    }

    /// Drops this message and everything after it, and puts it back in the
    /// composer to be edited.
    ///
    /// Its images are re-staged from the archive rather than reused in place:
    /// the composer owns files it may delete, and the archive's copies are
    /// read-only and shared with any earlier turn still carrying them.
    @discardableResult
    public func editMessage(_ turn: AppChatTurn) -> Bool {
        guard canRewind(to: turn) else { return false }
        guard let removed = conversationStore.truncate(from: turn.id) else { return false }
        releaseTranscriptImages()
        clearImages()
        clearDocuments()
        promptText = removed.prompt
        documentAttachments = removed.documents
        let urls = turn.images
            .map(\.fileURL)
            .filter { FileManager.default.fileExists(atPath: $0.path) }
        if !urls.isEmpty { addImages(urls) }
        generationTranscriptMailbox?.reset()
        diagnostics = nil
        error = nil
        return true
    }

    /// Sends the same message again for a different answer.
    ///
    /// Staging images is asynchronous, so the run is queued behind it rather
    /// than started against a half-attached message: `submit` refuses while
    /// `isAddingImages`, which is exactly the case this creates.
    public func regenerate(_ turn: AppChatTurn) {
        guard editMessage(turn) else { return }
        resendAfterStagingCompletes()
    }

    private func resendAfterStagingCompletes() {
        guard isAddingImages else {
            submit()
            return
        }
        Task { @MainActor [weak self] in
            for _ in 0..<600 {
                guard let self else { return }
                if !self.isAddingImages { break }
                try? await Task.sleep(for: .milliseconds(20))
            }
            guard let self, !self.isAddingImages else { return }
            self.submit()
        }
    }

    /// Whether the last answer looks cut off and could be carried on.
    ///
    /// Only ever offered for the newest message: continuing an older one would
    /// mean rewriting the middle of a conversation the model has already been
    /// shown.
    public var canContinueLastAnswer: Bool {
        guard !isSelectedConversationRunning, !loadState.isLoading, isModelAvailable,
              !hasStaleLoadedRuntime, promptText.isEmpty,
              imageAttachments.isEmpty, documentAttachments.isEmpty,
              supportsAssistantContinuation,
              let last = conversation.last,
              !last.response.isEmpty else { return false }
        // Images send the request down the multimodal renderer, which builds
        // its prompt as spans rather than a string and has nowhere to put a
        // trailing prefix. Offering Continue there would silently start over.
        guard conversation.allSatisfy(\.images.isEmpty) else { return false }
        return lastRunWasCutOff
    }

    /// Harmony writes its answer inside a channel the prompt has not opened at
    /// the point generation begins, so text appended after the generation
    /// prompt would not land in the reply. Gemma and ChatML both end their
    /// prompt exactly where the assistant's text starts.
    private var supportsAssistantContinuation: Bool {
        !isAppleSelected && selectedDescriptor.family != .gptOss
    }

    private var lastRunWasCutOff: Bool {
        if error == .cancelled { return true }
        guard let reason = diagnostics?.stopReason else { return false }
        // Ran out of room, rather than deciding it was finished.
        return reason == .maxTokens
    }

    /// Carries on the last answer from where it stopped.
    ///
    /// The partial text is handed back as the start of the reply rather than
    /// re-asked as a new question, so the model finishes the sentence it was in
    /// rather than starting again.
    public func continueLastAnswer() {
        guard canContinueLastAnswer, let last = conversation.last else { return }
        guard editMessage(last) else { return }
        pendingAssistantPrefix = last.response
        resendAfterStagingCompletes()
    }

    /// Text the next run should treat as the beginning of the answer. Cleared
    /// as soon as it is spent, so it applies to one run only.
    private var pendingAssistantPrefix: String?

    // MARK: - System prompt

    /// The selected model's system prompt, sent ahead of every chat with it.
    /// Set in Settings and stored with the rest of that model's profile, so
    /// each model carries its own.
    public var modelSystemPrompt: String {
        get { isAppleSelected ? (appleSystemPrompts[selectedModelID] ?? "") : settingsStore.systemPrompt }
        set {
            if isAppleSelected {
                appleSystemPrompts[selectedModelID] = newValue
                if settingsPersistenceEnabled { UserDefaults.standard.set(newValue, forKey: selectedModelID + ".systemPrompt") }
                return
            }
            guard settingsStore.systemPrompt != newValue else { return }
            settingsStore.systemPrompt = newValue
            persistSettings()
        }
    }

    /// What the next run will send, or nil when there is nothing to send.
    public var effectiveSystemPrompt: String? {
        modelSystemPrompt.isEmpty ? nil : modelSystemPrompt
    }

    /// Turns the last run had to drop to make its prompt fit, or nil when it
    /// dropped none.
    ///
    /// The renderer has always computed this; until v3 it was stored on a
    /// struct nothing read, so a long chat quietly forgot its beginning.
    public var lastRunDroppedTurns: Int? {
        guard let dropped = diagnostics?.droppedTurns, dropped > 0 else { return nil }
        return dropped
    }

    /// How full the context window would be if the draft were sent now.
    ///
    /// Every term is an estimate. See `AppContextUsage`: the real tokenizer
    /// lives in the decode service, and asking it would put the figure on
    /// screen only after the moment it was useful.
    public var contextUsage: AppContextUsage {
        let family = selectedDescriptor.family
        var history = 0
        for turn in conversation {
            history += AppContextUsage.tokens(inText: turn.modelPrompt)
            history += AppContextUsage.tokens(inText: turn.response)
            history += AppContextUsage.tokens(
                forImages: turn.images.count, family: family)
            history += 2 * AppContextUsage.overheadTokensPerMessage
        }
        var draft = AppContextUsage.tokens(inText: composedPromptText)
        draft += AppContextUsage.tokens(
            forImages: imageAttachments.count, family: family)
        if draft > 0 {
            if let systemPrompt = effectiveSystemPrompt {
                draft += AppContextUsage.tokens(inText: systemPrompt)
                    + AppContextUsage.overheadTokensPerMessage
            }
            draft += AppContextUsage.overheadTokensPerMessage
        }
        return AppContextUsage(
            historyTokens: history,
            draftTokens: draft,
            maxTokens: effectiveMaxContextTokens)
    }

    /// Completed turns as the inference client should see them: documents
    /// flattened back into the prompt, and images kept on the newest turns that
    /// still fit the image budget.
    ///
    /// The budget matters. Every image carried costs an encode on every turn,
    /// and the context cannot hold more of them than `maximumImageAttachments`
    /// anyway — a conversation that has collected a dozen would otherwise
    /// encode all twelve only for the renderer to drop most of them.
    func requestHistory(imageBudget: Int) -> [AppChatTurn] {
        var remaining = imageBudget
        var reversed: [AppChatTurn] = []
        // A message that was never answered is not an exchange. Since a turn is
        // now saved the moment it is sent, a run stopped before its first token
        // — or one the app was quit during — leaves a turn with no response,
        // and rendering that as an empty assistant message would tell the model
        // it had already replied with nothing. It still shows in the transcript
        // as the unanswered question it is.
        for turn in conversation.reversed() where !turn.response.isEmpty {
            // A file the archive no longer holds cannot be encoded, and sending
            // its path fails the whole run rather than one image.
            let readable = turn.images.filter {
                FileManager.default.fileExists(atPath: $0.fileURL.path)
            }
            let carries = !readable.isEmpty && readable.count <= remaining
            if carries { remaining -= readable.count }
            reversed.append(turn.requestTurn(carrying: carries ? readable : []))
        }
        return reversed.reversed()
    }

    public func makeRequest() throws -> AppGenerationRequest {
        try makeRequest(capabilities: effectiveChatCapabilities)
    }

    public func makeRequest(capabilities: AppChatCapabilities) throws -> AppGenerationRequest {
        // A run executes against the session that is actually loaded. Sending
        // the current settings instead meant that changing Context, Slots or
        // image residency and pressing Generate — without reloading first —
        // was refused outright with "generation runtime options do not match
        // the loaded session". The settings still apply on reload, which is
        // what the Memory section promises; they simply no longer break the
        // run in the meantime.
        if isAppleSelected {
            refreshAppleAvailability()
            guard selectedAppleAvailability.canGenerate else { throw AppInferenceError.invalidRequest(selectedAppleAvailability.label) }
            return AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/"), prompt: composedPromptText,
                systemPrompt: effectiveSystemPrompt ?? "", history: conversation,
                maxNewTokens: maxNewTokensOverride ?? 1024, temperature: Float(temperature), topK: nil, topP: nil)
        }
        let effective = loadedRuntimeKey ?? currentRuntimeKey
        let instructions = AppToolCatalog.systemInstructions(
            for: capabilities, currentDate: Self.toolDate())
        let workspaceInstruction: String
        if capabilities.workspaceRead || capabilities.fileWrite || capabilities.command, let folder = selectedCodingFolder {
            workspaceInstruction = "The selected coding workspace is \(String(reflecting: AppFileNameText.safe(folder.path))). list_files, read_file, write_file and edit_file use paths relative to that workspace; search_files can search all selected folders."
        } else { workspaceInstruction = "" }
        let systemPrompt = [effectiveSystemPrompt ?? "", instructions, workspaceInstruction]
            .filter { !$0.isEmpty }.joined(separator: "\n\n")
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: modelPathText),
            // Attached files become prompt text here, at the one point where
            // the model's view of the message is built. Everything above this
            // line keeps them as attachments.
            prompt: composedPromptText,
            systemPrompt: systemPrompt,
            assistantPrefix: pendingAssistantPrefix ?? "",
            history: requestHistory(
                imageBudget: max(0, maximumImageAttachments - imageAttachments.count)),
            imageAttachments: imageAttachments,
            maxNewTokens: maxNewTokensOverride ?? effective.maxContextTokens,
            maxContextTokens: effective.maxContextTokens,
            reasoning: selectedDescriptor.family == .gptOss
                ? .off
                : (selectedDescriptor.reasoningControl == .alwaysOn ? .on : reasoning),
            reasoningEffort: selectedDescriptor.family == .gptOss
                ? reasoningEffort : nil,
            preserveThinking: selectedDescriptor.family == .qwen36
                && preserveThinking,
            temperature: Float(temperature),
            topK: topKEnabled ? topK : nil,
            topP: topKEnabled && topPEnabled ? Float(topP) : nil,
            repetitionPenalty: 1.0,
            runtimeOptions: effective.options(
                prefillEnabled: runtimeOptions.prefillEnabled,
                prefillChunkTokens: runtimeOptions.prefillChunkTokens),
            tools: AppToolCatalog.definitions(for: capabilities),
            conversationKey: conversationStore.selectedConversationID?.uuidString,
            preservesFullHistory: automaticMemory)
        try request.validate(requireModelDirectory: true)
        return request
    }

    /// The day, as tool instructions state it. Kept to the day so the system
    /// prompt, and the cached state built on it, is stable within a day.
    static func toolDate(_ date: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "MMMM d, yyyy"
        return formatter.string(from: date)
    }

    func apply(_ event: AppInferenceEvent) {
        let isSelectedRun = handlingAnswerConversationID == conversationStore.selectedConversationID
        switch event {
        case .memorySample:
            if isSelectedRun { sampleLiveMemory() }
        case .prefillProgress(let done, let total):
            handlingAnswerRun?.phase = .prefill
            handlingAnswerRun?.prefillDone = done
            handlingAnswerRun?.prefillTotal = total
            if isSelectedRun {
                phase = .prefill
                livePrefillDone = done
                livePrefillTotal = total
                sampleLiveMemory()
            }
        case .token(let token):
            handlingAnswerRun?.phase = .decode
            handlingAnswerRun?.tokenCount = token.index + 1
            handlingAnswerRun?.elapsedDecodeSeconds = token.elapsedDecodeSeconds
            if isSelectedRun {
                phase = .decode
                liveTokenCount = token.index + 1
                liveElapsedDecodeSeconds = token.elapsedDecodeSeconds
                sampleLiveMemory()
            }
            if !token.textDelta.isEmpty {
                outputText += token.textDelta
            }
        case .thinking(let token):
            handlingAnswerRun?.phase = .decode
            handlingAnswerRun?.tokenCount = token.index + 1
            handlingAnswerRun?.elapsedDecodeSeconds = token.elapsedDecodeSeconds
            if isSelectedRun {
                phase = .decode
                liveTokenCount = token.index + 1
                liveElapsedDecodeSeconds = token.elapsedDecodeSeconds
                sampleLiveMemory()
            }
            if !token.textDelta.isEmpty {
                outputThinkingText += token.textDelta
                handlingAnswerRun?.currentGenerationThinking += token.textDelta
            }
        case .toolCalls:
            // Calls reach the app through the answer runner, which runs them.
            break
        case .finished(let diagnostics):
            if isSelectedRun { visionTowerMappedBytes = diagnostics.visionTowerMappedBytes }
            finishSuccessfully(diagnostics)
        case .cancelled(let diagnostics):
            finishCancelled(diagnostics)
        case .failed(let appError, let partial):
            diagnostics = partial
            materializeServiceTranscript()
            finishWithError(appError)
        }
    }

    private var handlingAnswerRun: ActiveAnswerRun? {
        handlingAnswerConversationID.flatMap { activeAnswerRuns[$0] }
    }

    private func recordAutoPerformance(_ diagnostics: AppDiagnostics, completed: Bool = true) {
        // The run's own model, not whatever is selected now: the selection may
        // have moved to another model while this chat kept generating.
        guard AppleModelKind(rawValue: handlingAnswerRun?.modelProfileKey ?? "") == nil, automaticMemory, settingsPersistenceEnabled, diagnostics.toolRounds == 0,
              let runKey = handlingAnswerRun?.runtimeKey,
              handlingAnswerRun?.modelProfileKey == selectedProviderKey,
              let summary = AppAutomaticMemoryPlanner.layoutSummary(at: runKey.modelDirectory) else { return }
        guard let layers = AppAutoPerformanceRecords.residentDenseLayers(
            diagnostics.runtimeOptions.denseResidencyPolicy, summary: summary) else { return }
        let key = AppAutoPerformanceRecords.workloadKey(promptTokens: diagnostics.promptTokenCount ?? 0,
            reusedPrefix: (diagnostics.cachedPromptTokens ?? 0) > 0, imageInput: !outputImageAttachments.isEmpty,
            sampled: loadedRuntimeKey?.forceLogitsHead ?? currentForceLogitsHead)
        autoWorkloadKey = key
        let start = handlingAnswerRun?.observationStart
        let observedPressure = start?.pressure() ?? AppAutoMemoryPressureMonitor.shared.snapshot.pressure
        let paging = start?.pagingBytes() ?? 0
        guard completed || observedPressure != .normal || paging > 0 else { return }
        AppAutoPerformanceRecords.record(identity: AppAutoPerformanceRecords.identity(summary: summary, device: deviceCapabilities,
            engineIdentity: (client as? any AppInferenceEngineReporting)?.currentEngineIdentity,
            modelDirectory: URL(fileURLWithPath: modelPathText)),
            observation: InferencePlanObservation(candidateID: AppAutoPerformanceRecords.candidateID(
                    contextTokens: effectiveMaxContextTokens,
                    expertCacheSlots: diagnostics.runtimeOptions.expertCacheSlots,
                    residentDenseLayers: layers,
                    prefillChunkTokens: diagnostics.runtimeOptions.prefillChunkTokens),
                workloadKey: key, decodeTokens: diagnostics.generatedTokens, decodeSeconds: diagnostics.decodeSeconds,
                timeToFirstTokenSeconds: diagnostics.requestStartTimeToFirstTokenSeconds ?? diagnostics.timeToFirstTokenSeconds ?? 0,
                pressure: observedPressure, swapGrowthBytes: paging))
    }

    private func finishSuccessfully(_ diagnostics: AppDiagnostics) {
        guard let run = handlingAnswerRun, !run.hasHandledTerminalEvent else { return }
        run.hasHandledTerminalEvent = true
        materializeServiceTranscript()
        if run.conversationID == conversationStore.selectedConversationID {
            self.diagnostics = diagnostics
        }
        recordAutoPerformance(diagnostics)
        recordAnswer()
        finishTerminalRun()
    }

    /// Files whatever the run produced against the turn `run` already saved.
    ///
    /// Every terminal outcome comes through here, a stop included. An answer
    /// cut off halfway is what the reader has in front of them, and discarding
    /// it — which is what a cancelled run used to do — threw away the whole
    /// exchange, question and all.
    private func recordAnswer() {
        let thinking = conversationStore.outputToolRounds.isEmpty
            ? outputThinkingText : (handlingAnswerRun?.currentGenerationThinking ?? "")
        conversationStore.completeTurn(
            response: outputResponsePlainText,
            thinking: thinking.isEmpty ? nil : thinking,
            toolRounds: conversationStore.outputToolRounds,
            sources: conversationStore.outputSources)
    }

    private func finishCancelled(_ diagnostics: AppDiagnostics) {
        guard let run = handlingAnswerRun, !run.hasHandledTerminalEvent else { return }
        run.hasHandledTerminalEvent = true
        materializeServiceTranscript()
        if run.conversationID == conversationStore.selectedConversationID {
            self.diagnostics = diagnostics
        }
        recordAutoPerformance(diagnostics, completed: false)
        if run.conversationID == conversationStore.selectedConversationID { error = .cancelled }
        recordAnswer()
        finishTerminalRun()
    }

    private func materializeServiceTranscript() {
        guard AppleModelKind(rawValue: handlingAnswerRun?.modelProfileKey ?? "") == nil, !(client is any AppSessionInferenceClient), activeAnswerRuns.count <= 1 else { return }
        guard let reporter = client as? any AppInferenceTranscriptReporting else { return }
        outputText = reporter.generationTranscriptMailbox.completeText
    }

    private func finishWithError(_ appError: AppInferenceError) {
        guard let run = handlingAnswerRun, !run.hasHandledTerminalEvent else { return }
        run.hasHandledTerminalEvent = true
        if appError.isEngineTransportFailure, loadedRuntimeKey == run.runtimeKey {
            loadedRuntimeKey = nil
            loadState = .notLoaded
        }
        if run.conversationID == conversationStore.selectedConversationID { error = appError }
        recordAnswer()
        finishTerminalRun()
    }

    private func finishStreamFailure(_ appError: AppInferenceError) {
        materializeServiceTranscript()
        finishWithError(appError)
    }

    private func finishTerminalRun() {
        guard let conversationID = handlingAnswerConversationID else { return }
        activeAnswerRuns.removeValue(forKey: conversationID)
        runState = activeAnswerRuns.isEmpty ? .idle : .running
        if activeAnswerRuns.isEmpty { toolStore.inferenceActivity.set(false) }
        if activeAnswerRuns.isEmpty, AppAutoMemoryPressureMonitor.shared.snapshot.pressure != .normal {
            reclaimIdleConversationSessions()
        }
        restoreSelectedRunPresentation()
        // Auto learns from every answer, but its next plan waits for the next
        // load. Applying it here changed the settings of the model already in
        // memory, so after an ordinary answer the composer asked for a reload
        // nobody had caused (#54). `run` still reloads with a larger window
        // when the conversation outgrows the loaded one.
        if !isAppleSelected, activeAnswerRuns.isEmpty, automaticMemory, settingsPersistenceEnabled, !loadState.isReady {
            applySettingsProfile(storedSettingsProfile(for: selectedInstall))
        }
    }

    // MARK: - Answers with tools

    /// What the composer's Web and Files switches allow for the next message,
    /// limited by the model and by having folders to search.
    public var effectiveChatCapabilities: AppChatCapabilities {
        if isAppleSelected { return .none }
        let support = toolSupport
        guard support.allowsTools else { return .none }
        return AppChatCapabilities(
            web: settingsStore.webSearchEnabled && !isWebSearchPaused,
            files: settingsStore.fileSearchEnabled && !toolStore.folders.isEmpty,
            workspaceRead: settingsStore.fileSearchEnabled && selectedCodingFolder != nil,
            fileWrite: settingsStore.fileSearchEnabled && selectedCodingFolder != nil && settingsStore.codingWriteEnabled,
            command: settingsStore.fileSearchEnabled && selectedCodingFolder != nil && settingsStore.codingCommandEnabled,
            commandNetwork: settingsStore.fileSearchEnabled && selectedCodingFolder != nil
                && settingsStore.codingCommandEnabled && settingsStore.codingNetworkEnabled)
    }

    public var toolSupport: AppToolSupport {
        if isAppleSelected { return .unavailable("Apple supports text chat only; tools are unavailable.") }
        guard let id = selectedDescriptor.catalogID else { return .untested }
        return AppToolSupport.forModel(id)
    }

    public var isWebSearchPaused: Bool {
        settingsStore.webSearchEnabled && pauseWebSearchWhenOffline && networkStatus.isOffline
    }

    public var pauseWebSearchWhenOffline: Bool {
        get { settingsStore.pauseWebSearchWhenOffline }
        set {
            guard settingsStore.pauseWebSearchWhenOffline != newValue else { return }
            settingsStore.pauseWebSearchWhenOffline = newValue
            persistSettings()
        }
    }

    /// Web searches and page reads go through the sandbox, a small virtual
    /// machine that holds the web code. Off by default; turning it on starts
    /// the sandbox, turning it off stops it.
    public var runWebInSandbox: Bool {
        get { settingsStore.runWebInSandbox }
        set {
            guard settingsStore.runWebInSandbox != newValue else { return }
            settingsStore.runWebInSandbox = newValue
            persistSettings()
            #if os(macOS)
            sandboxTask = Task { [weak self] in await self?.applySandboxSetting() }
            #endif
        }
    }

    /// Reading the folders chosen for file search happens in a second
    /// sandbox with no network, so a file the app cannot parse safely is never
    /// opened in the app. Off by default; turning it on indexes through the
    /// sandbox from then on, turning it off stops the sandbox.
    public var runFilesInSandbox: Bool {
        get { settingsStore.runFilesInSandbox }
        set {
            guard settingsStore.runFilesInSandbox != newValue else { return }
            settingsStore.runFilesInSandbox = newValue
            toolStore.stopIndexing()
            toolStore.indexInSandbox = newValue
            persistSettings()
            #if os(macOS)
            filesSandboxTask?.cancel()
            filesSandboxTask = Task { [weak self] in await self?.applyFilesSandboxSetting() }
            #endif
        }
    }

    #if os(macOS)
    /// Makes the index follow the setting: indexing in progress is stopped,
    /// a sandbox that is no longer wanted is stopped, and the folders are
    /// brought up to date by the reader the setting now names.
    private func applyFilesSandboxSetting() async {
        toolStore.stopIndexing()
        if !settingsStore.runFilesInSandbox { await toolStore.sandboxFiles?.stop() }
        // A file one reader skipped may be fine for the other.
        toolStore.forgetSkippedFiles()
        guard !Task.isCancelled else { return }
        if settingsStore.fileSearchEnabled { toolStore.reindex() }
    }

    /// The sandbox needs macOS 26 or newer; the setting is hidden below it.
    public var isSandboxSupported: Bool { deviceCapabilities.macOSMajorVersion >= 26 }

    /// Makes the sandbox match the setting. The latest choice wins if the
    /// switch is flipped quickly: a start that was cut short by an earlier
    /// "off" is made again while the setting is on.
    private func applySandboxSetting() async {
        for _ in 0..<3 {
            guard !Task.isCancelled else { return }
            if !settingsStore.runWebInSandbox {
                await sandbox.stop()
                // A start that was waiting behind an earlier stop may have
                // brought the sandbox back meanwhile: look again.
                if !settingsStore.runWebInSandbox, sandbox.state == .off { return }
                continue
            }
            await sandbox.start()
            if settingsStore.runWebInSandbox, sandbox.state != .off { return }
        }
    }

    private func startSandbox() {
        sandboxTask = Task { [weak self] in await self?.sandbox.start() }
    }

    /// At app launch: removes a container an earlier run left behind (only
    /// when the sandbox has been used on this Mac) and starts the sandbox if
    /// the setting is on.
    public func prepareSandboxAtLaunch() {
        let used = FileManager.default.fileExists(
            atPath: SandboxLocations.defaultStateDirectory().path)
        guard settingsStore.runWebInSandbox || settingsStore.runFilesInSandbox || used else { return }
        sandboxTask = Task { [weak self] in
            guard let self else { return }
            await self.sandbox.cleanUpLeftovers()
            await self.filesSandbox.cleanUpLeftovers()
            if self.settingsStore.runWebInSandbox { await self.sandbox.start() }
        }
    }

    /// When the app quits: stops the sandbox, waiting a few seconds.
    public func stopSandboxWhenQuitting() {
        sandboxTask?.cancel()
        filesSandboxTask?.cancel()
        toolStore.stopIndexing()
        sandbox.stopWhenQuitting()
        filesSandbox.stopWhenQuitting()
    }

    func finishSandboxWork() async {
        await sandboxTask?.value
        await filesSandboxTask?.value
    }

    /// Tests give the model a sandbox with a fake `container`.
    func installSandbox(_ controller: SandboxController,
                        makeClient: (@Sendable (SandboxLease) -> SandboxClient)? = nil) {
        sandbox = controller
        let settings = settingsStore
        var access = SandboxWebAccess(
            controller: controller, mayRestart: { await settings.runWebInSandbox })
        if let makeClient { access.makeClient = makeClient }
        toolStore.sandboxWeb = access
    }

    /// The same for the files sandbox.
    func installFilesSandbox(_ controller: SandboxController,
                             makeClient: (@Sendable (SandboxLease) -> SandboxClient)? = nil) {
        filesSandbox = controller
        let settings = settingsStore
        var access = SandboxWebAccess(
            controller: controller, mayRestart: { await settings.runFilesInSandbox })
        if let makeClient { access.makeClient = makeClient }
        toolStore.sandboxFiles = SandboxFilesService(controller: controller, access: access)
    }
    #endif

    public var webSearchEnabled: Bool {
        get { settingsStore.webSearchEnabled }
        set {
            guard settingsStore.webSearchEnabled != newValue else { return }
            settingsStore.webSearchEnabled = newValue
            persistSettings()
        }
    }

    public var fileSearchEnabled: Bool {
        get { settingsStore.fileSearchEnabled }
        set {
            guard settingsStore.fileSearchEnabled != newValue else { return }
            settingsStore.fileSearchEnabled = newValue
            persistSettings()
            if newValue { toolStore.reindex() }
        }
    }

    /// The workspace for coding tools. Only an explicit choice counts: Local
    /// Files alone never grants raw file access to the first search folder.
    public var selectedCodingFolder: AppSearchFolder? {
        guard let id = settingsStore.codingFolderID else { return nil }
        return toolStore.folders.first { $0.id.uuidString == id }
    }
    public var codingFolderID: String {
        get { selectedCodingFolder?.id.uuidString ?? "" }
        set { settingsStore.codingFolderID = newValue; persistSettings() }
    }
    public var codingWriteEnabled: Bool {
        get { settingsStore.codingWriteEnabled }
        set { settingsStore.codingWriteEnabled = newValue; persistSettings() }
    }
    public var codingCommandEnabled: Bool {
        get { settingsStore.codingCommandEnabled }
        set { settingsStore.codingCommandEnabled = newValue; persistSettings() }
    }
    public var codingNetworkEnabled: Bool {
        get { settingsStore.codingNetworkEnabled }
        set { settingsStore.codingNetworkEnabled = newValue; persistSettings() }
    }

    public var searchProvider: AppSearchProviderKind {
        get { settingsStore.searchProvider }
        set {
            guard settingsStore.searchProvider != newValue else { return }
            settingsStore.searchProvider = newValue
            persistSettings()
        }
    }

    public var outputToolActivities: [AppToolActivity] { conversationStore.outputToolActivities }
    public var outputToolRounds: [AppToolRound] { conversationStore.outputToolRounds }
    public var outputSources: [AppSource] { conversationStore.outputSources }

    func apply(answer event: AppAnswerEvent, for conversationID: UUID) {
        guard activeAnswerRuns[conversationID] != nil else { return }
        handlingAnswerConversationID = conversationID
        conversationStore.beginOutputUpdate(for: conversationID)
        defer {
            conversationStore.endOutputUpdate()
            handlingAnswerConversationID = nil
        }
        switch event {
        case .inference(let inference):
            apply(inference)
        case .generationStarted(let round):
            handlingAnswerRun?.phase = .prefill
            if conversationStore.selectedConversationID == conversationID {
                phase = .prefill
            }
            handlingAnswerRun?.currentGenerationThinking = ""
            if round > 0 {
                // The text before the calls belongs to the round now.
                outputText = ""
                if !(client is any AppSessionInferenceClient) {
                    generationTranscriptMailbox?.reset()
                }
            }
        case .toolActivity(let activity):
            handlingAnswerRun?.phase = .tools
            if conversationStore.selectedConversationID == conversationID { phase = .tools }
            var activities = conversationStore.outputToolActivities
            if let index = activities.firstIndex(where: { $0.id == activity.id }) {
                activities[index] = activity
            } else {
                activities.append(activity)
            }
            conversationStore.outputToolActivities = activities
        case .toolRoundFinished(let round, let sources):
            conversationStore.outputToolRounds.append(round)
            conversationStore.outputSources = sources
            outputText = ""
        case .retryingMalformedToolCall:
            outputText = ""
            conversationStore.outputToolActivities.append(AppToolActivity(
                id: "retry-\(conversationStore.outputToolActivities.count)", name: "",
                title: "Retrying", detail: "The model wrote a tool call TUFF could not read.",
                state: .refused, summary: "Generating once more"))
        case .finished(let diagnostics):
            apply(.finished(diagnostics))
        case .cancelled(let diagnostics):
            if let diagnostics {
                apply(.cancelled(diagnostics))
            } else {
                finishCancelledWithoutDiagnostics()
            }
        case .failed(let error, let partial):
            apply(.failed(error, partial: partial))
        }
    }

    private func finishCancelledWithoutDiagnostics() {
        guard let run = handlingAnswerRun, !run.hasHandledTerminalEvent else { return }
        run.hasHandledTerminalEvent = true
        if run.conversationID == conversationStore.selectedConversationID { error = .cancelled }
        recordAnswer()
        finishTerminalRun()
    }

    private func clearLoadTask(generation: UInt64) {
        guard generation == loadGeneration else { return }
        if !isRunning { toolStore.inferenceActivity.set(false) }
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
    }

    private func clearUnloadTask(generation: UInt64) {
        guard generation == unloadGeneration else { return }
        unloadTask = nil
    }
}
