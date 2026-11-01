import AppKit
import TUFFAppCore
import TUFFAppServer
import TUFFAppUpdater
import TUFFMacPresentation
import SwiftUI

/// Whether the window is in full screen.
///
/// The title bar is hidden there, so the chrome that normally sits clear of it
/// — the sidebar header, the status strip's room for the traffic lights — has
/// to make its own space instead.
private struct WindowIsFullScreenKey: EnvironmentKey {
    static let defaultValue = false
}

extension EnvironmentValues {
    var windowIsFullScreen: Bool {
        get { self[WindowIsFullScreenKey.self] }
        set { self[WindowIsFullScreenKey.self] = newValue }
    }
}

/// Identity for `.task(id:)` below: re-runs the sync into
/// `TUFFAccentThemeStore` whenever either persisted accent setting changes.
private struct AccentSettingsKey: Equatable {
    let mode: AppAccentColorMode
    let hex: String
}

struct RootView: View {
    let model: AppModel
    let backgroundAPI: AppBackgroundAPIController
    let updateController: AppUpdateController
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @State private var navigation = AppNavigationState()
    @State private var benchmarks = BenchmarkController()
    @State private var renameTarget: AppConversationRecord?
    @State private var renameText = ""
    /// The chat a Delete is waiting on confirmation for. Deleting has no
    /// undo, so one stray click in a context menu used to lose a chat.
    @State private var deletionTarget: AppConversationRecord?
    @State private var isFullScreen = false

    var body: some View {
        splitView
        // Every `appFont` in the app resolves its point size against this.
        // macOS has no Dynamic Type, and scaling the rendered view instead
        // only magnifies a bitmap, so the zoom has to reach the fonts
        // themselves.
        .environment(\.appFontScale, model.zoomLevel.scale)
        // The default font as well, for the text that never asks for one —
        // the sidebar's own labels among it. At 100% this is the body font
        // the app was already inheriting.
        .font(AppFont.body.resolved(scale: model.zoomLevel.scale))
        .task(id: AccentSettingsKey(mode: model.accentColorMode, hex: model.customAccentColorHex)) {
            TUFFAccentThemeStore.shared.update(
                mode: model.accentColorMode, customHex: model.customAccentColorHex)
        }
        .environment(\.windowIsFullScreen, isFullScreen)
        .background {
            WindowFullScreenReader(isFullScreen: $isFullScreen)
                .accessibilityHidden(true)
        }
        .focusedSceneValue(
            \.appNavigationAction,
            AppNavigationAction(
                select: { navigation.select($0) },
                stepChat: canStepChat ? { stepChat(by: $0) } : nil,
                focusComposer: {
                    navigation.select(.chat)
                    NotificationCenter.default.post(name: .tuffFocusComposer, object: nil)
                }))
        .containerBackground(for: .window) {
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(nsColor: .windowBackgroundColor).mix(
                        with: TUFFMacTheme.accentColor,
                        by: 0.04),
                ],
                startPoint: .top,
                endPoint: .bottom)
        }
        .tint(TUFFMacTheme.accentColor)
        .animation(.smooth(duration: 0.3), value: model.requiresModelInstallation)
        .animation(.smooth(duration: 0.25), value: model.error)
        .transaction { transaction in
            if model.isRunning { transaction.animation = nil }
        }
        .alert("Rename Chat", isPresented: renameAlertIsPresented) {
            TextField("Name", text: $renameText)
            Button("Cancel", role: .cancel) { renameTarget = nil }
            Button("Rename") {
                if let renameTarget {
                    model.renameConversation(renameTarget, to: renameText)
                }
                renameTarget = nil
            }
            .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .confirmationDialog(
            "Delete \u{201C}\(deletionTarget?.title ?? "")\u{201D}?",
            isPresented: deletionDialogIsPresented,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let deletionTarget { model.deleteConversation(deletionTarget) }
                deletionTarget = nil
            }
            Button("Cancel", role: .cancel) { deletionTarget = nil }
        } message: {
            Text("This chat is removed from this Mac and can't be restored.")
        }
    }

    private var deletionDialogIsPresented: Binding<Bool> {
        Binding {
            deletionTarget != nil
        } set: { presented in
            if !presented { deletionTarget = nil }
        }
    }

    /// The sidebar and detail columns are sized in points, so they are scaled
    /// with the text they hold — at 150% an unscaled 160pt sidebar clipped its
    /// own labels.
    private var zoom: CGFloat { model.zoomLevel.scale }

    @ViewBuilder private var splitView: some View {
        if isFullScreen {
            HSplitView {
                sidebar
                    .frame(
                        minWidth: AppWindowLayout.sidebarMinimumWidth * zoom,
                        idealWidth: AppWindowLayout.sidebarIdealWidth * zoom,
                        maxWidth: AppWindowLayout.sidebarMaximumWidth * zoom,
                        maxHeight: .infinity)
                workspace
                    .frame(
                        minWidth: AppWindowLayout.detailMinimumWidth * zoom,
                        maxWidth: .infinity,
                        maxHeight: .infinity)
            }
        } else {
            NavigationSplitView {
                sidebar
                    .navigationSplitViewColumnWidth(
                        min: AppWindowLayout.sidebarMinimumWidth * zoom,
                        ideal: AppWindowLayout.sidebarIdealWidth * zoom,
                        max: AppWindowLayout.sidebarMaximumWidth * zoom)
            } detail: {
                workspace
            }
            .navigationSplitViewStyle(.balanced)
        }
    }

    private var workspace: some View {
        AppWorkspaceView(
            destination: navigation.destination,
            model: model,
            backgroundAPI: backgroundAPI,
            updateController: updateController,
            benchmarks: benchmarks)
    }

    private var sidebar: some View {
        List {
            Section {
                ForEach(AppDestination.allCases) { item in
                    Button {
                        navigation.select(item)
                    } label: {
                        HStack(spacing: 9) {
                            // Stated rather than inherited: the sidebar list
                            // style puts its own font on row content, so the
                            // zoom does not reach these labels otherwise.
                            Label(item.title, systemImage: item.systemImage)
                                .appFont(.body)
                            Spacer(minLength: 0)
                            if item == .benchmarks, benchmarks.isRunning {
                                ProgressView()
                                    .controlSize(.mini)
                                    .accessibilityLabel("Benchmark running")
                            }
                        }
                        .contentShape(Rectangle())
                    }
                        .buttonStyle(.plain)
                        .listRowBackground(SidebarSelection(isSelected: navigation.destination == item))
                        .foregroundStyle(navigation.destination == item
                                         ? TUFFMacTheme.accentColor
                                         : .primary)
                        .accessibilityIdentifier("navigation.\(item.rawValue)")
                        .accessibilityValue(navigation.destination == item
                                            ? "Selected" : "")
                        .accessibilityAddTraits(
                            navigation.destination == item ? .isSelected : [])
                }
            }
            // Always, not only on the Chat screen. Chats are the thing the app
            // is for; hiding them behind the destination meant leaving Chat to
            // check a download made every conversation disappear, and coming
            // back was a two-step trip.
            // No section header: a sidebar list gives one a collapse arrow on
            // hover, and a heading plus a small pencil icon read as a label
            // rather than a way to start a chat.
            Section {
                newChatButton
                    .listRowBackground(Color.clear)
                    .padding(.top, 6)
                if model.conversationStore.conversations.isEmpty {
                    chatsEmptyState
                        .listRowBackground(Color.clear)
                } else {
                    ForEach(AppConversationDateGroup.grouped(
                        model.conversationStore.conversations), id: \.group) { section in
                        Text(section.group.rawValue.uppercased())
                            .appFont(.caption2.weight(.semibold))
                            .tracking(0.6)
                            .foregroundStyle(.tertiary)
                            .padding(.top, 8)
                            .listRowBackground(Color.clear)
                            .accessibilityAddTraits(.isHeader)
                        ForEach(section.conversations) { conversation in
                            conversationRow(conversation)
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        // Windowed, NavigationSplitView supplies the native floating sidebar.
        // In full screen the content stays inside its safe area, but the
        // sidebar surface itself extends through the top, bottom, and leading
        // safe areas so it becomes a conventional full-height side column.
        .scrollContentBackground(.hidden)
        .safeAreaInset(edge: .top, spacing: 0) { brandHeader }
        .safeAreaInset(edge: .bottom, spacing: 0) { selectedModelFooter }
        // Behind the header and footer as well as the list. Painted under the
        // list alone, the header and footer showed the translucent sidebar
        // material instead, as two lighter bands. A stable surface also keeps
        // bright wallpaper from washing out the sidebar, follows light and
        // dark appearance, and works with Reduce Transparency.
        .background {
            Color(nsColor: .windowBackgroundColor)
                .ignoresSafeArea(
                    .container,
                    edges: isFullScreen ? [.top, .bottom, .leading] : [.top])
        }
        .navigationTitle("TUFF")
    }

    /// Starting a chat is the main thing the sidebar does, so it is a full
    /// button with its shortcut rather than an icon beside a heading.
    private var newChatButton: some View {
        SidebarNewChatButton(isEnabled: canStartNewChat) {
            navigation.select(.chat)
            model.clearOutput()
        }
    }

    /// Where the chats will be. The New Chat button sits just above it.
    private var chatsEmptyState: some View {
        VStack(spacing: 6) {
            Image(systemName: "bubble.left.and.bubble.right")
                .appFont(.title3)
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            Text("No chats yet")
                .appFont(.callout.weight(.medium))
                .foregroundStyle(.secondary)
            Text("Your chats are saved here, on this Mac.")
                .appFont(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 18)
    }

    /// Next and previous chat need another chat to go to, and follow the
    /// same rule as clicking one in the sidebar.
    private var canStepChat: Bool {
        let chats = model.conversationStore.conversations
        guard let first = chats.first else { return false }
        return model.canSelectConversation(first)
            && (chats.count > 1 || model.conversationStore.selectedConversationID != first.id
                || navigation.destination != .chat)
    }

    /// Moves through the chats in sidebar order. From another screen, the
    /// step starts as if no chat were open.
    private func stepChat(by offset: Int) {
        let current = navigation.destination == .chat
            ? model.conversationStore.selectedConversationID : nil
        guard let target = AppConversationStepping.target(
            in: model.conversationStore.conversations, from: current, offset: offset)
        else { return }
        navigation.select(.chat)
        model.selectConversation(target)
    }

    /// Whether the New Chat button does anything. A chat that has not been
    /// written to yet is already a new chat.
    private var canStartNewChat: Bool {
        guard navigation.destination == .chat else { return true }
        return model.hasOutputTranscript
            || model.conversationStore.selectedConversation == nil
    }

    private func isOpen(_ conversation: AppConversationRecord) -> Bool {
        navigation.destination == .chat
            && model.conversationStore.selectedConversationID == conversation.id
    }

    /// The open chat stays enabled even though clicking it changes nothing:
    /// disabling it would grey out the one row that is meant to look selected.
    private func canOpen(_ conversation: AppConversationRecord) -> Bool {
        isOpen(conversation) || model.canSelectConversation(conversation)
    }

    private func conversationRow(_ conversation: AppConversationRecord) -> some View {
        SidebarChatRow(
            title: conversation.title,
            subtitle: modelName(for: conversation.modelID),
            timestamp: AppConversationDateGroup.shortTimestamp(for: conversation.updatedAt),
            isSelected: isOpen(conversation)
        ) {
            // Opening a chat means showing it. Selecting one from Models or
            // Settings used to load it silently behind whatever screen was up,
            // so the click looked like it had done nothing.
            navigation.select(.chat)
            model.selectConversation(conversation)
        }
        .listRowBackground(SidebarSelection(isSelected: isOpen(conversation)))
        .disabled(!canOpen(conversation))
        .contextMenu {
            Button("Rename…") {
                renameText = conversation.title
                renameTarget = conversation
            }
            Divider()
            Button("Delete…", role: .destructive) {
                deletionTarget = conversation
            }
            .disabled(!model.canDeleteConversation(conversation))
        }
        .accessibilityLabel("\(conversation.title), \(modelName(for: conversation.modelID))")
        .accessibilityValue(isOpen(conversation) ? "Selected" : "")
        .accessibilityHint("Opens this saved chat")
        .accessibilityAddTraits(isOpen(conversation) ? .isSelected : [])
        .accessibilityAction(named: "Rename chat") {
            renameText = conversation.title
            renameTarget = conversation
        }
        .accessibilityAction(named: "Delete chat") {
            deletionTarget = conversation
        }
    }

    private func modelName(for profileKey: String) -> String {
        model.installs.first {
            $0.descriptor.settingsProfileKey == profileKey
        }?.descriptor.shortName ?? profileKey
    }

    /// Presentation follows the optional target; dismissal clears it.
    private var renameAlertIsPresented: Binding<Bool> {
        Binding {
            renameTarget != nil
        } set: { presented in
            if !presented { renameTarget = nil }
        }
    }

    private var brandHeader: some View {
        HStack(spacing: 10) {
            TUFFMarkView()
                .padding(3)
                .frame(width: 32, height: 32)
                .background(
                    TUFFMacTheme.surfaceStyle(
                        reduceTransparency: reduceTransparency,
                        material: .thin),
                    in: RoundedRectangle(cornerRadius: 8))
            Text("TUFF")
                .appFont(.title2.weight(.bold))
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .padding(.bottom, 10)
    }

    private var selectedModelFooter: some View {
        Button {
            navigation.select(.models)
        } label: {
            HStack(alignment: .center, spacing: 8) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("CURRENT MODEL")
                        .appFont(.caption2.weight(.semibold))
                        .foregroundStyle(.tertiary)
                    Text(model.selectedDescriptor.displayName)
                        .appFont(.caption.weight(.medium))
                        .lineLimit(2)
                    HStack(spacing: 5) {
                        Circle()
                            .fill(model.loadState.isReady
                                  ? AnyShapeStyle(TUFFMacTheme.accentColor)
                                  : AnyShapeStyle(.tertiary))
                            .frame(width: 6, height: 6)
                        Text(model.presentation.label)
                            .appFont(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.right")
                    .appFont(.caption2.weight(.semibold))
                    .foregroundStyle(.tertiary)
                    .accessibilityHidden(true)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            TUFFMacTheme.surfaceStyle(
                reduceTransparency: reduceTransparency,
                material: .thin),
            in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(.separator.opacity(0.6), lineWidth: 0.5))
        .padding(.horizontal, 10)
        .padding(.bottom, 10)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Current model")
        .accessibilityValue("\(model.selectedDescriptor.displayName), \(model.presentation.label)")
        .accessibilityHint("Opens Models")
    }

}

/// The selected-row highlight: an inset rounded pill, as in Finder and
/// Mail, rather than a band running edge to edge.
/// One saved chat: its title, the model it was with, and when it was last
/// used, with a hover highlight so the list reads as clickable.
private struct SidebarChatRow: View {
    let title: String
    let subtitle: String
    let timestamp: String
    let isSelected: Bool
    let action: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            // The time sits under the title, beside the model, so a narrow
            // sidebar gives the title the whole width.
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .appFont(.body)
                    .lineLimit(1)
                    .truncationMode(.tail)
                HStack(spacing: 4) {
                    Text(subtitle)
                        .lineLimit(1)
                    Text("·")
                    Text(timestamp)
                        .monospacedDigit()
                        .lineLimit(1)
                        .layoutPriority(1)
                }
                .appFont(.caption2)
                .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 2)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(isSelected ? TUFFMacTheme.accentColor : .primary)
        .background {
            if isHovered && !isSelected {
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(Color.primary.opacity(0.06))
                    .padding(.horizontal, -8)
                    .padding(.vertical, -2)
            }
        }
        .onHover { isHovered = $0 }
        .help(title)
    }
}

/// The sidebar's New Chat control: an accent tile with the icon, the label,
/// and the shortcut, across the sidebar's width.
private struct SidebarNewChatButton: View {
    let isEnabled: Bool
    let action: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: "plus")
                    .appFont(.callout.weight(.bold))
                    .foregroundStyle(.white)
                    .frame(width: 24, height: 24)
                    .background(TUFFMacTheme.accentColor,
                                in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                Text("New Chat")
                    .appFont(.body.weight(.medium))
                    .foregroundStyle(.primary)
                Spacer(minLength: 4)
                Text("⌘N")
                    .appFont(.caption)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 7)
            .background {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(TUFFMacTheme.accentColor.opacity(isHovered && isEnabled ? 0.20 : 0.11))
            }
            .overlay {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(TUFFMacTheme.accentColor.opacity(0.28), lineWidth: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .opacity(isEnabled ? 1 : 0.5)
        .disabled(!isEnabled)
        .onHover { isHovered = $0 }
        .animation(.easeOut(duration: 0.12), value: isHovered)
        .help("New Chat (⌘N)")
        .accessibilityLabel("New Chat")
    }
}

private struct SidebarSelection: View {
    let isSelected: Bool

    var body: some View {
        RoundedRectangle(cornerRadius: 7, style: .continuous)
            .fill(isSelected ? TUFFMacTheme.accentColor.opacity(0.16) : .clear)
            .padding(.horizontal, 8)
            .padding(.vertical, 1)
    }
}

/// Reports whether the hosting window is in full screen.
///
/// The style mask is read on attach as well as on the notifications: a window
/// restored into full screen never posts one, and the sidebar would sit against
/// the top of the screen until the user toggled full screen off and on again.
private struct WindowFullScreenReader: NSViewRepresentable {
    @Binding var isFullScreen: Bool

    func makeNSView(context: Context) -> NSView {
        let view = ObservingView()
        view.onChange = { isFullScreen = $0 }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        (nsView as? ObservingView)?.onChange = { isFullScreen = $0 }
    }

    final class ObservingView: NSView {
        var onChange: ((Bool) -> Void)?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            NotificationCenter.default.removeObserver(self)
            guard let window else { return }
            for name in [NSWindow.didEnterFullScreenNotification,
                         NSWindow.didExitFullScreenNotification,
                         // A window restored into full screen posts neither of
                         // the above; it does become key, and the style mask is
                         // correct by then.
                         NSWindow.didBecomeKeyNotification] {
                NotificationCenter.default.addObserver(
                    self, selector: #selector(windowFullScreenChanged),
                    name: name, object: window)
            }
            report()
        }

        @objc private func windowFullScreenChanged() { report() }

        private func report() {
            let fullScreen = window?.styleMask.contains(.fullScreen) ?? false
            // Deferred: this runs inside AppKit's view-hierarchy change, which
            // SwiftUI counts as its own update pass.
            DispatchQueue.main.async { [weak self] in
                self?.onChange?(fullScreen)
            }
        }
    }
}

struct AppWorkspaceView: View {
    let destination: AppDestination
    let model: AppModel
    let backgroundAPI: AppBackgroundAPIController
    let updateController: AppUpdateController
    let benchmarks: BenchmarkController

    var body: some View {
        Group {
            switch destination {
            case .chat:
                ChatWorkspaceView(model: model)
            case .models:
                ModelsWorkspaceView(model: model)
            case .benchmarks:
                BenchmarksWorkspaceView(model: model, controller: benchmarks)
            case .server:
                ServerWorkspaceView(
                    model: model, controller: backgroundAPI)
            case .settings:
                SettingsWorkspaceView(
                    model: model, updateController: updateController)
            }
        }
        .accessibilityIdentifier("workspace.\(destination.rawValue)")
    }
}

private struct ChatWorkspaceView: View {
    let model: AppModel
    @State private var conversationChromeHeight: CGFloat = 0

    var body: some View {
        primaryContent
            .frame(
                minWidth: AppWindowLayout.detailMinimumWidth,
                maxWidth: .infinity,
                maxHeight: .infinity)
            .safeAreaInset(edge: .top, spacing: 0) {
                StatusHUDView(model: model)
            }
    }

    private var primaryContent: some View {
        Group {
            if model.requiresModelInstallation {
                ModelInstallView(model: model)
            } else {
                conversationView
            }
        }
    }

    private var conversationView: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                if model.hasOutputTranscript {
                    OutputPaneView(model: model)
                        .padding(.bottom, conversationChromeHeight)
                } else if conversationChromeHeight > 0 {
                    OutputPaneView(model: model)
                        .frame(height: max(0, geometry.size.height - conversationChromeHeight))
                        .frame(maxHeight: .infinity, alignment: .top)
                }

                conversationChrome
                    .background {
                        GeometryReader { chromeGeometry in
                            Color.clear.preference(
                                key: ConversationChromeHeightKey.self,
                                value: chromeGeometry.size.height)
                        }
                    }
            }
            .onPreferenceChange(ConversationChromeHeightKey.self) { height in
                guard height > 0 else { return }
                var transaction = Transaction()
                transaction.disablesAnimations = true
                withTransaction(transaction) { conversationChromeHeight = height }
            }
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 10) {
            ErrorBanner(model: model)
            // Not just an empty prompt box: a message that is so far only an
            // attached file is still a message being written, and offering
            // "try one of these instead" over it reads as a dead end.
            if !model.hasComposedInput && model.showPromptExamples && !model.isRunning {
                PromptExamplesView { preset in model.promptText = preset.prompt }
            }
            ModelActionBanner(model: model)
            PromptComposerView(model: model)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
        .animation(.smooth(duration: 0.2), value: model.hasComposedInput)
        .animation(.smooth(duration: 0.2), value: model.showPromptExamples)
    }
}

private struct ModelsWorkspaceView: View {
    let model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                WorkspaceTitle(
                    title: "Models",
                    subtitle: "Download and manage the models that run on this Mac.")
                ModelLibraryColumnsView(model: model)
            }
            .frame(maxWidth: 1_180, alignment: .leading)
            .padding(28)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

private struct SettingsWorkspaceView: View {
    let model: AppModel
    let updateController: AppUpdateController

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceTitle(
                title: "Settings",
                subtitle: "App behavior and per-model generation defaults.")
                .padding(.horizontal, 28)
                .padding(.top, 24)
                .frame(maxWidth: .infinity, alignment: .leading)
            AppSettingsView(
                model: model, updateController: updateController)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct WorkspaceTitle: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .appFont(.largeTitle.bold())
                .accessibilityHeading(.h1)
            Text(subtitle)
                .appFont(.callout)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .contain)
    }
}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
