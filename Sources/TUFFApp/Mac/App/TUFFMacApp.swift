import AppKit
import TUFFAppCore
import TUFFAppServer
import TUFFAppUpdater
import TUFFMacPresentation
import TUFFServerCore
import SwiftUI

// Run as a regular foreground app even when launched as a bare SwiftPM
// executable (no .app bundle): Dock icon, click-to-activate, full main menu
// with Quit (Cmd+Q).
private final class ForegroundAppDelegate: NSObject, NSApplicationDelegate {
    /// Set by the scene so quitting can release this session's staged images.
    @MainActor static var model: AppModel?

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            // Settings edits are coalesced, so the newest one can still be
            // waiting to be written when the app is asked to quit.
            Self.model?.flushPendingSettings()
            Self.model?.releaseAllAttachments()
            Self.model?.stopSandboxWhenQuitting()
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        // A packaged app owns a system-rendered, correctly inset Icon Composer
        // icon. Replacing it with the full-canvas PNG enlarges the Dock tile.
        if Bundle.main.object(forInfoDictionaryKey: "CFBundleIconName") == nil,
           Bundle.main.object(forInfoDictionaryKey: "CFBundleIconFile") == nil,
           let icon = MacAppIcon.dockFallback() {
            NSApp.applicationIconImage = icon
            NSApp.dockTile.display()
        }
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

struct TUFFMacApp: App {
    @NSApplicationDelegateAdaptor private var appDelegate: ForegroundAppDelegate
    @State private var isReportingBug = false
    @State private var model: AppModel
    @State private var backgroundAPI = AppBackgroundAPIController()
    @State private var updateController: AppUpdateController
    private let inferenceClient: ServingHostInferenceClient
    private let loadModelRequested: Bool

    init() {
        // Move any models still under the old Application Support directory
        // before anything resolves a model path or opens a pack.
        AppSupportMigration.migrateModels(
            applicationSupport: AppSupportMigration.applicationSupportURL())
        // One serving pool per profile: this window, the foreground server and
        // the optional Background API all attach to the same dedicated host.
        let support = AppSupportMigration.applicationSupportURL()
        let inferenceClient: ServingHostInferenceClient
        do {
            let profile = try ServingHostProfile.forCurrentProcess()
            inferenceClient = ServingHostInferenceClient(
                profile: profile,
                bundleIdentifier: Bundle.main.bundleIdentifier ?? "tuff.unbundled",
                modelsRoot: support.appendingPathComponent("TUFF/Models", isDirectory: true))
        } catch {
            fatalError("The TUFF serving profile could not be prepared: \(error)")
        }
        let model = AppModel(
            client: inferenceClient,
            installer: RepackModelInstallerClient(descriptor: .selected),
            conversationStore: .persistentDefault(),
            visionRuntimeSupported: AppModel.currentDeviceSupportsVisionRuntime,
            settingsPersistenceEnabled: true,
            toolStore: .standard())
        let updateController = AppUpdateController()
        self.inferenceClient = inferenceClient
        self.loadModelRequested = CommandLine.arguments.contains("--load-model")
        _model = State(initialValue: model)
        _updateController = State(initialValue: updateController)
        MainActor.assumeIsolated { ForegroundAppDelegate.model = model }
    }

    var body: some Scene {
        Window("TUFF", id: "main") {
            RootView(
                model: model,
                backgroundAPI: backgroundAPI,
                updateController: updateController)
                .sheet(isPresented: $isReportingBug) { BugReportSheet(model: model) }
                .task { backgroundAPI.refreshRegistrationAfterUpdate() }
                .alert("Recovery Update", isPresented: Binding(
                    get: { updateController.recoveryMessage != nil },
                    set: { if !$0 { updateController.recoveryMessage = nil } })) {
                    Button("OK") { updateController.recoveryMessage = nil }
                    Button("Check for Updates…") { updateController.checkForUpdates() }
                        .disabled(!updateController.canCheckForUpdates)
                } message: { Text(updateController.recoveryMessage ?? "") }
                // Scaled with the zoom: at 150% the text and the columns that
                // hold it are half again as wide, so the smallest window that
                // still fits the layout grows with it.
                .frame(
                    minWidth: AppWindowLayout.minimumWidth * model.zoomLevel.scale,
                    minHeight: AppWindowLayout.minimumHeight * model.zoomLevel.scale)
                // Once, when the window first appears: the setting is read
                // from disk in init, and loadModelAtLaunchIfEnabled ignores a
                // model that is missing or already busy.
                .task {
                    model.prepareSandboxAtLaunch()
                    if loadModelRequested {
                        model.loadModel()
                    } else {
                        model.loadModelAtLaunchIfEnabled()
                    }
                    model.prepareEnginesAtLaunch()
                }
                // On the window, not in the Inspector: the menu item works
                // whether or not the Inspector is open.
                .confirmationDialog(
                    "Remove downloaded image support?",
                    isPresented: Bindable(model).isConfirmingVisionPackRemoval,
                    titleVisibility: .visible
                ) {
                    Button("Remove Image Support", role: .destructive) {
                        model.removeVisionPack()
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text("Text generation will continue to work. "
                        + "Getting image support back means downloading the "
                        + "pack again.")
                }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(
            width: AppWindowLayout.defaultWidth,
            height: AppWindowLayout.defaultHeight)
        .windowResizability(.contentMinSize)
        .commands {
            AppNavigationCommands()
            CommandGroup(replacing: .help) {
                Button("Report a Bug…") { isReportingBug = true }
            }
            // Into the View menu macOS already puts Enter Full Screen in, not a
            // `CommandMenu("View")` — that builds a second menu with the same
            // name and the app ends up with two of them in the bar.
            CommandGroup(after: .toolbar) {
                Button("Zoom In") { model.zoomLevel = model.zoomLevel.zoomedIn }
                    .keyboardShortcut("=", modifiers: .command)
                    .disabled(model.zoomLevel == AppZoomLevel.allCases.last)
                Button("Zoom Out") { model.zoomLevel = model.zoomLevel.zoomedOut }
                    .keyboardShortcut("-", modifiers: .command)
                    .disabled(model.zoomLevel == AppZoomLevel.allCases.first)
                Button("Actual Size") { model.zoomLevel = .default }
                    .keyboardShortcut("0", modifiers: .command)
                    .disabled(model.zoomLevel == .default)
            }
            CommandGroup(replacing: .appInfo) {
                Button("About TUFF") {
                    NSApp.orderFrontStandardAboutPanel(
                        options: AboutPanelPresentation.options(
                            infoDictionary: Bundle.main.infoDictionary,
                            icon: MacAppIcon.load()))
                }
            }
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") {
                    updateController.checkForUpdates()
                }
                .disabled(!updateController.canCheckForUpdates)
                Button("Check for Recovery Update…") { updateController.checkForRecoveryUpdate() }
                    .disabled(!updateController.canCheckForUpdates)
            }
            CommandMenu("Generation") {
                Button("New Chat") { model.clearOutput() }
                    .keyboardShortcut("n", modifiers: .command)
                    .disabled(model.isSelectedConversationRunning || !model.hasOutputTranscript)
                Divider()
                Button("Cancel Generation") { model.cancel() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!model.canCancel)
                Button("Cancel Model Installation") { model.cancelInstall() }
                    .disabled(!model.canCancelInstall)
            }
            CommandMenu("Model") {
                Button("Load Model", action: model.loadModel)
                    .disabled(!model.canLoadModel)
                Button("Reload Model", action: model.reloadModel)
                    .disabled(!model.canReloadModel)
                Button("Unload Model", action: model.unloadModel)
                    .disabled(!model.canUnloadModel)
                Divider()
                Button("Reveal Model in Finder", action: revealModel)
                    .disabled(modelRevealTarget == .unavailable)
                // The Inspector shows image support only while there is
                // something to decide, so reclaiming the pack lives here:
                // rare, deliberate, and destructive. Which is why it asks
                // first — this is the only reachable way to delete the pack,
                // and it used to call straight through.
                Button("Remove Image Support", action: model.requestVisionPackRemoval)
                    .disabled(!model.canRemoveVisionPack)
            }
            CommandMenu("Settings") {
                Picker("Send Message With",
                       selection: Bindable(model).newlineShortcut) {
                    ForEach(AppNewlineShortcut.sendMessageOptions) { shortcut in
                        Text(shortcut.sendMessageLabel).tag(shortcut)
                    }
                }
                Picker("Prompt Examples",
                       selection: Bindable(model).showPromptExamples) {
                    Text("Show").tag(true)
                    Text("Hide").tag(false)
                }
                Picker("After Sending",
                       selection: Bindable(model).sentPromptBehavior) {
                    ForEach(AppSentPromptBehavior.allCases) { behavior in
                        Text(behavior.settingsLabel).tag(behavior)
                    }
                }
                Picker("Load Model At Launch",
                       selection: Bindable(model).loadModelOnLaunch) {
                    Text("Off").tag(false)
                    Text("On").tag(true)
                }
            }
        }
    }

    private var modelRevealTarget: ModelRevealTarget {
        ModelRevealPolicy.target(
            forModelPath: model.modelPathText,
            fileExists: FileManager.default.fileExists(atPath:))
    }

    private func revealModel() {
        switch modelRevealTarget {
        case .selectItem(let url):
            NSWorkspace.shared.activateFileViewerSelecting([url])
        case .openContainer(let url):
            NSWorkspace.shared.open(url)
        case .unavailable:
            break
        }
    }
}
