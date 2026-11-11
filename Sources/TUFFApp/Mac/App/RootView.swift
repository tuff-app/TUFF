import AppKit
import TUFFAppCore
import TUFFAppServer
import TUFFAppUpdater
import TUFFMacPresentation
import SwiftUI

extension EnvironmentValues {
    /// Whether the windowed sidebar is collapsed, so the traffic lights sit
    /// over the detail pane and its top chrome must leave room for them.
    @Entry var windowSidebarIsHidden = false

    /// True for a page shown as a tab of Settings: the Settings header is
    /// already above it, so it drops its own title and top margin.
    @Entry var workspaceEmbedded = false
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
    @Environment(\.colorScheme) private var colorScheme
    /// `TUFF_OPEN_SCREEN=models` (or server, settings…) opens that screen
    /// first, for screenshots and testing a screen without clicking to it.
    @State private var navigation = AppNavigationState(
        destination: ProcessInfo.processInfo.environment["TUFF_OPEN_SCREEN"]
            .flatMap(AppDestination.init(rawValue:)) ?? .chat)
    @State private var benchmarks = BenchmarkController()
    @State private var renameTarget: AppConversationRecord?
    @State private var renameText = ""
    /// The chat a Delete is waiting on confirmation for. Deleting has no
    /// undo, so one stray click in a context menu used to lose a chat.
    @State private var deletionTarget: AppConversationRecord?
    @State private var columnVisibility = NavigationSplitViewVisibility.all
    @State private var footerHovered = false
    /// Each chat row's frame in the list, for the one glass pill that marks
    /// the open chat and slides between rows.
    @State private var chatRowFrames: [UUID: CGRect] = [:]
    /// The row under the pointer while the pill is being dragged.
    @State private var draggedChatID: UUID?
    /// The pointer's y while dragging; the glass follows it directly.
    @State private var chatDragY: CGFloat?
    /// The Settings tab last shown among the app's own settings, kept while
    /// the person goes back to Chat.
    /// `TUFF_OPEN_SECTION=search` (general, models, search, advanced) picks
    /// the Settings tab, alongside `TUFF_OPEN_SCREEN`, for screenshots.
    @State private var settingsSection: AppSettingsSection = ProcessInfo.processInfo
        .environment["TUFF_OPEN_SECTION"].flatMap(AppSettingsSection.init(rawValue:)) ?? .general

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
        // No full screen. There macOS lays the window out its own way: the
        // toolbar becomes a band that slides in over the status strip, and
        // the sidebar panel loses its inset and its corners. The green
        // button fills the screen with the window instead.
        .windowFullScreenBehavior(.disabled)
        #if DEBUG
        // Debug builds only: `TUFF_DEBUG_SIDEBAR=hidden` opens with the
        // sidebar collapsed, for checking that layout without driving the UI.
        .task {
            if ProcessInfo.processInfo.environment["TUFF_DEBUG_SIDEBAR"] == "hidden" {
                columnVisibility = .detailOnly
            }
        }
        #endif
        #if DEBUG
        // Debug builds only: `TUFF_DEBUG_PROMPT` sends one prompt after launch,
        // for exercising streaming without driving the UI.
        .task {
            guard let prompt = ProcessInfo.processInfo.environment["TUFF_DEBUG_PROMPT"] else { return }
            try? await Task.sleep(for: .seconds(2))
            model.promptText = prompt
            model.submit()
        }
        #endif
        .focusedSceneValue(
            \.appNavigationAction,
            AppNavigationAction(
                select: { navigation.select($0) },
                stepChat: canStepChat ? { stepChat(by: $0) } : nil,
                focusComposer: {
                    navigation.select(.chat)
                    NotificationCenter.default.post(name: .tuffFocusComposer, object: nil)
                }))
        // Translucent: the desktop shows through behind the chat, blurred and
        // tinted with the page color, which is also what the glass controls
        // refract. Reduce Transparency keeps it solid.
        .containerBackground(for: .window) {
            if reduceTransparency || !model.translucentWindow {
                TUFFDesign.pageColor
            } else {
                ZStack {
                    // In light appearance the window-background material is
                    // itself a nearly opaque grey, so with a tint on top the
                    // desktop barely showed; the sidebar material stays
                    // translucent in light.
                    WindowBlurView(material: colorScheme == .dark ? .underWindowBackground : .fullScreenUI)
                    TUFFDesign.pageColor.opacity(colorScheme == .dark ? 0.45 : 0.3)
                }
            }
        }
        .tint(TUFFMacTheme.accentTextColor)
        // The whole window draws as the active window even when another app
        // is in front: text fields, spinners and other system-drawn controls
        // otherwise dimmed when TUFF was clicked away from.
        .environment(\.controlActiveState, .active)
        // Set on the app rather than the view so AppKit colors, menus and
        // sheets follow it too.
        .task(id: model.appearanceMode) {
            switch model.appearanceMode {
            case .system: NSApp.appearance = nil
            case .light: NSApp.appearance = NSAppearance(named: .aqua)
            case .dark: NSApp.appearance = NSAppearance(named: .darkAqua)
            }
        }
        .animation(TUFFMotion.standard, value: model.requiresModelInstallation)
        .animation(TUFFMotion.standard, value: model.error)
        .transaction { transaction in
            if model.isRunning { transaction.animation = nil }
        }
        // App sheets rather than system alerts: an alert draws Delete in red
        // and its default button in the system accent, whatever accent TUFF
        // is set to.
        .tuffConfirm("Rename Chat", isPresented: renameAlertIsPresented,
                     symbol: "pencil", confirm: "Rename", text: $renameText,
                     canConfirm: !renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) {
            if let renameTarget { model.renameConversation(renameTarget, to: renameText) }
            renameTarget = nil
        }
        .tuffConfirm("Delete \u{201C}\(deletionTarget?.title ?? "")\u{201D}?",
                     isPresented: deletionDialogIsPresented,
                     message: "This chat is removed from this Mac and can't be restored.",
                     symbol: "trash", confirm: "Delete") {
            if let deletionTarget { model.deleteConversation(deletionTarget) }
            deletionTarget = nil
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in model.refreshAppleAvailability() }
        .onChange(of: model.error) { _, error in
            if let error, GenericErrorBannerPolicy.shouldShow(error: error, loadState: model.loadState) {
                TUFFNoticeCenter.shared.show(.init(
                    key: "error",
                    symbol: "exclamationmark.triangle.fill",
                    title: "Something went wrong",
                    message: error.userMessage,
                    help: error.technicalDetail,
                    actions: [.secondary("Copy Details") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(error.technicalDetail, forType: .string)
                        model.error = nil
                    }],
                    onClose: { model.error = nil }))
            } else {
                TUFFNoticeCenter.shared.dismiss(key: "error")
            }
        }
        // Now and then, once the app has been opened a few times, ask for a
        // star; only while the chat sits idle.
        .task {
            #if DEBUG
            let forced = ProcessInfo.processInfo.environment["TUFF_DEBUG_STAR_PROMPT"] == "1"
            #else
            let forced = false
            #endif
            guard StarPrompt.registerLaunch() || forced else { return }
            try? await Task.sleep(for: .seconds(forced ? 3 : 25))
            guard navigation.destination == .chat, !model.isRunning,
                  !model.loadState.isLoading, TUFFNoticeCenter.shared.current == nil else { return }
            TUFFNoticeCenter.shared.show(StarPrompt.notice())
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
        NavigationSplitView(columnVisibility: $columnVisibility) {
            sidebar
                // Fixed width: dragging the divider only made the chat
                // titles and the header reflow. Scaled with the zoom.
                .navigationSplitViewColumnWidth(AppWindowLayout.sidebarIdealWidth * zoom)
                .background {
                    SidebarWidthLock(width: AppWindowLayout.sidebarIdealWidth * zoom)
                        .accessibilityHidden(true)
                }
                // Our own toggle (below) replaces the system one.
                .toolbar(removing: .sidebarToggle)
        } detail: {
            workspace
                .environment(\.windowSidebarIsHidden, columnVisibility == .detailOnly)
                // Keeps the window's toolbar once the system sidebar
                // toggle is gone; without any toolbar item the sidebar
                // stopped running up under the title bar.
                .toolbar {
                    InvisibleToolbarItem()
                    // Collapsed, outside Chat: a real toolbar button
                    // beside the traffic lights. A button drawn over the
                    // title bar there never received the click.
                    CollapsedSidebarToolbarButton(
                        isShown: columnVisibility == .detailOnly && navigation.destination != .chat,
                        action: toggleSidebar)
                }
        }
        .navigationSplitViewStyle(.balanced)
        // The sidebar's width is fixed, but its edge still showed the
        // resize pointer. This strip over the edge keeps the arrow.
        .overlay(alignment: .topLeading) {
            if columnVisibility != .detailOnly {
                Color.clear
                    .frame(width: 12)
                    .contentShape(Rectangle())
                    .pointerStyle(.default)
                    .padding(.leading, SidebarToggleButton.panelInset + AppWindowLayout.sidebarIdealWidth * zoom - 6)
                    .ignoresSafeArea()
            }
        }
        .overlay(alignment: .topLeading) {
            if columnVisibility != .detailOnly || navigation.destination == .chat {
                SidebarToggleButton(collapsed: columnVisibility == .detailOnly,
                                    besideStrip: navigation.destination == .chat,
                                    sidebarWidth: AppWindowLayout.sidebarIdealWidth * zoom,
                                    action: toggleSidebar)
                    .ignoresSafeArea()
                    .transition(.opacity)
            }
        }

        // The native split-view container also reads this value, outside
        // the sidebar content's environment.
        .environment(\.controlActiveState, .active)
        .background(SidebarActiveAppearance())
    }

    private var workspace: some View {
        AppWorkspaceView(
            destination: navigation.destination,
            settingsSection: $settingsSection,
            select: { navigation.select($0) },
            model: model,
            backgroundAPI: backgroundAPI,
            updateController: updateController,
            benchmarks: benchmarks)
            // The app's look for every button that does not choose one. Set
            // here, not on the window: there it also restyled the system's
            // sidebar toggle in the title bar.
            .buttonStyle(.tuffSecondary)
            // Errors and the occasional prompt, as one card centred over the
            // content, just below the status strip.
            .overlay { TUFFNoticeOverlay(center: TUFFNoticeCenter.shared) }
            .disclosureGroupStyle(.tuff)
    }

    /// A plain scrolling stack rather than a `List`: every row already draws
    /// its own selection and hover, and a sidebar list drew the right-click
    /// outline in the system accent whatever accent the app was set to.
    private var sidebar: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 4) {
                // The sidebar is for chats. Settings, which holds Models,
                // Server and Benchmarks as tabs, is the cog in the header.
                newChatButton
                    .padding(.bottom, 4)
                if model.conversationStore.conversations.isEmpty {
                    chatsEmptyState
                        .transition(.opacity)
                } else {
                    ForEach(AppConversationDateGroup.grouped(
                        model.conversationStore.conversations), id: \.group) { section in
                        Text(section.group.rawValue.uppercased())
                            .appFont(.caption2.weight(.semibold))
                            .tracking(0.6)
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, SidebarMetrics.rowInset)
                            .padding(.top, 12)
                            .padding(.bottom, 4)
                            .accessibilityAddTraits(.isHeader)
                            .transition(.opacity)
                        ForEach(section.conversations) { conversation in
                            conversationRow(conversation)
                                .transition(.opacity)
                        }
                    }
                }
            }
            // One glass pill marks the open chat. It glides to another chat
            // when one is opened, and can be dragged down the list to open
            // the chat it is let go on.
            .background(alignment: .topLeading) {
                if let frame = chatPillFrame {
                    TUFFGlassPill(cornerRadius: SidebarMetrics.rowRadius,
                                  isHovered: draggedChatID != nil)
                        .frame(width: frame.width, height: frame.height)
                        .offset(x: frame.minX, y: frame.minY)
                }
            }
            .simultaneousGesture(
                DragGesture(minimumDistance: 6, coordinateSpace: .named(Self.chatListSpace))
                    .onChanged { value in
                        guard openChatID != nil else { return }
                        var follow = Transaction()
                        follow.animation = .interactiveSpring(response: 0.12, dampingFraction: 0.9)
                        withTransaction(follow) {
                            chatDragY = value.location.y
                            if let hit = chatRowFrames.first(where: {
                                $0.value.minY <= value.location.y && value.location.y <= $0.value.maxY
                            })?.key {
                                draggedChatID = hit
                            }
                        }
                    }
                    .onEnded { _ in
                        let id = draggedChatID
                        withAnimation(TUFFMotion.standard) {
                            draggedChatID = nil
                            chatDragY = nil
                            if let id, id != openChatID,
                               let target = model.conversationStore.conversations.first(where: { $0.id == id }),
                               canOpen(target) {
                                model.selectConversation(target)
                            }
                        }
                    })
            .coordinateSpace(name: Self.chatListSpace)
            .animation(chatDragY == nil ? TUFFMotion.standard : nil, value: openChatID)
            // A chat added or deleted: the rest, and the pill, slide to
            // their places.
            .animation(TUFFMotion.standard, value: model.conversationStore.conversations.map(\.id))
            .padding(.horizontal, SidebarMetrics.edgeInset)
            .padding(.bottom, 8)
        }
        .scrollIndicators(.automatic)
        // Keep labels, accents and controls at their focused appearance when
        // another window becomes key; focus must not restyle navigation.
        .environment(\.controlActiveState, .active)
        .safeAreaInset(edge: .top, spacing: 0) { brandHeader }
        .safeAreaInset(edge: .bottom, spacing: 0) { selectedModelFooter }
        .background {
            // The floating panel's tone. Opaque on purpose: a blurred,
            // see-through panel took on the wallpaper's color (blue over a
            // blue desktop).
            TUFFDesign.sidebarColor
                .ignoresSafeArea(.container, edges: [.top, .bottom])
        }
        // A quiet edge on the floating panel, drawn by us so it stays when
        // the window is not key.
        .overlay {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.10), lineWidth: 1)
                .ignoresSafeArea(.container, edges: [.top, .bottom])
                .allowsHitTesting(false)
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

    private static let chatListSpace = "TUFFChatList"

    private func toggleSidebar() {
        withAnimation(TUFFMotion.sidebar) {
            columnVisibility = columnVisibility == .detailOnly ? .all : .detailOnly
        }
    }

    /// The glass pill's frame: the open (or dragged-over) chat's row, or
    /// while dragging, that row's size centred on the pointer, kept within
    /// the list.
    private var chatPillFrame: CGRect? {
        guard let id = draggedChatID ?? openChatID, var frame = chatRowFrames[id] else { return nil }
        if let y = chatDragY {
            let tops = chatRowFrames.values.map(\.minY)
            let bottoms = chatRowFrames.values.map(\.maxY)
            frame.origin.y = min(max(y - frame.height / 2, tops.min() ?? 0),
                                 (bottoms.max() ?? frame.maxY) - frame.height)
        }
        return frame
    }

    /// The open chat, when Chat is the screen showing.
    private var openChatID: UUID? {
        navigation.destination == .chat ? model.conversationStore.selectedConversationID : nil
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
        .disabled(!canOpen(conversation))
        .onGeometryChange(for: CGRect.self) { $0.frame(in: .named(Self.chatListSpace)) } action: {
            chatRowFrames[conversation.id] = $0
        }
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
        model.modelShortName(forProfileKey: profileKey)
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
                .lineLimit(1)
                .fixedSize()
            Spacer(minLength: 0)
            SidebarSettingsButton(
                isActive: navigation.destination != .chat,
                isBusy: benchmarks.isRunning) {
                navigation.select(navigation.destination == .chat ? .settings : .chat)
            }
        }
        .padding(.horizontal, SidebarMetrics.edgeInset + SidebarMetrics.rowInset - 3)
        .padding(.top, 10)
        .padding(.bottom, 12)
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
                    Text(model.selectedModelName)
                        .appFont(.caption.weight(.medium))
                        .lineLimit(2)
                    HStack(spacing: 5) {
                        Circle()
                            .fill(model.isModelAvailable
                                  ? AnyShapeStyle(TUFFMacTheme.accentTextColor)
                                  : AnyShapeStyle(.tertiary))
                            .frame(width: 6, height: 6)
                        Text(model.isAppleSelected ? model.selectedAppleAvailability.label : model.presentation.label)
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
        .tuffGlass(RoundedRectangle(cornerRadius: 14, style: .continuous),
                   highlighted: footerHovered)
        .onHover { footerHovered = $0 }
        .animation(TUFFMotion.hover, value: footerHovered)
        .padding(.horizontal, SidebarMetrics.edgeInset)
        .padding(.bottom, SidebarMetrics.edgeInset)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Current model")
        .accessibilityValue("\(model.selectedModelName), \(model.presentation.label)")
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
                    .appFont(isSelected ? .body.weight(.semibold) : .body)
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
                // Grey in the selected row as well: the accent at secondary
                // strength was too faint to read on the accent tint.
                .foregroundStyle(TUFFDesign.secondaryTextColor)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, SidebarMetrics.rowInset)
            .padding(.vertical, 7)
            // The open chat's glass is drawn once for the whole list, so it
            // can slide between rows; each row only shows hover.
            .background(SidebarSelection(isSelected: false, isHovered: isHovered && !isSelected))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.primary)
        .onHover { isHovered = $0 }
        .animation(TUFFMotion.hover, value: isHovered)
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
                    .foregroundStyle(TUFFMacTheme.onAccentColor)
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
            .tuffGlass(RoundedRectangle(cornerRadius: 14, style: .continuous),
                       tint: TUFFMacTheme.accentColor.opacity(0.18),
                       highlighted: isHovered && isEnabled)
            .scaleEffect(isHovered && isEnabled ? 1.02 : 1)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // Not dimmed when the open chat is already empty: a faded button
        // read as broken. It just does nothing then.
        .disabled(!isEnabled)
        .onHover { isHovered = $0 }
        .animation(TUFFMotion.hover, value: isHovered)
        .help("New Chat (⌘N)")
        .accessibilityLabel("New Chat")
    }
}

/// The selected row sits on the same Liquid Glass as the selected Settings
/// tab; other rows get a faint wash under the pointer.
private struct SidebarSelection: View {
    let isSelected: Bool
    var isHovered = false

    var body: some View {
        if isSelected {
            TUFFGlassPill(cornerRadius: SidebarMetrics.rowRadius, isHovered: isHovered)
        } else {
            RoundedRectangle(cornerRadius: SidebarMetrics.rowRadius, style: .continuous)
                .fill(isHovered ? Color.primary.opacity(0.06) : .clear)
        }
    }
}

/// Settings, from the sidebar header: a bare cog, accent-colored while
/// Settings is open, with a spinner while a benchmark runs there.
private struct SidebarSettingsButton: View {
    let isActive: Bool
    let isBusy: Bool
    let action: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            ZStack {
                if isBusy {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: "gearshape")
                        .appFont(.body.weight(.medium))
                }
            }
            .foregroundStyle(isActive ? AnyShapeStyle(TUFFMacTheme.accentTextColor)
                                      : AnyShapeStyle(TUFFDesign.secondaryTextColor))
            .frame(width: 32, height: 32)
            // Just the icon: tinted while Settings is open, brighter on hover.
            .opacity(isHovered || isActive ? 1 : 0.8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .animation(TUFFMotion.hover, value: isHovered)
        .help(isActive ? "Back to Chat" : "Settings (⌘,)")
        .accessibilityLabel("Settings")
        .accessibilityAddTraits(isActive ? .isSelected : [])
    }
}

/// Spacing shared by every sidebar element, so the brand mark, the row icons,
/// the chat titles and the footer all start on one line.
private enum SidebarMetrics {
    /// From the sidebar's edge to a row's highlight.
    static let edgeInset: CGFloat = 10
    /// From a row's highlight to its content.
    static let rowInset: CGFloat = 10
    static let rowRadius: CGFloat = 14
}

private struct SidebarDestinationRow: View {
    let item: AppDestination
    let isSelected: Bool
    let isBusy: Bool
    let select: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: select) {
            HStack(spacing: 9) {
                Label(item.title, systemImage: item.systemImage).appFont(.body)
                Spacer(minLength: 0)
                if isBusy { ProgressView().controlSize(.mini) }
            }
            .padding(.horizontal, SidebarMetrics.rowInset)
            .frame(minHeight: 34)
            .background(SidebarSelection(isSelected: isSelected, isHovered: isHovered))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .animation(TUFFMotion.hover, value: isHovered)
        .foregroundStyle(isSelected ? TUFFMacTheme.accentTextColor : .primary)
        .accessibilityIdentifier("navigation.\(item.rawValue)")
        .accessibilityValue(isSelected ? "Selected" : "")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

/// Keep the native sidebar material active without changing keyboard focus.
private struct SidebarActiveAppearance: NSViewRepresentable {
    func makeNSView(context: Context) -> Observer { Observer() }
    // Not on every SwiftUI update: the walk covers the whole window, and the
    // material only changes state when the window's key status does.
    func updateNSView(_ view: Observer, context: Context) {}

    final class Observer: NSView {
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            NotificationCenter.default.removeObserver(self)
            guard let window else { return }
            for name in [NSWindow.didBecomeKeyNotification, NSWindow.didResignKeyNotification] {
                NotificationCenter.default.addObserver(
                    self, selector: #selector(refresh), name: name, object: window)
            }
            refresh()
        }

        @objc func refresh() {
            DispatchQueue.main.async { [weak self] in
                guard let root = self?.window?.contentView?.superview else { return }
                func activateSidebar(in view: NSView) {
                    if let effect = view as? NSVisualEffectView, effect.material == .sidebar {
                        effect.state = .active
                        effect.isEmphasized = true
                    }
                    for child in view.subviews { activateSidebar(in: child) }
                }
                activateSidebar(in: root)
            }
        }
    }
}

/// Holds the sidebar column at one width.
///
/// `navigationSplitViewColumnWidth` only sets the width the sidebar starts
/// at: the divider could still be dragged, and macOS kept the dragged width
/// for later launches.
private struct SidebarWidthLock: NSViewRepresentable {
    let width: CGFloat

    func makeNSView(context: Context) -> LockView {
        let view = LockView()
        view.width = width
        return view
    }

    func updateNSView(_ view: LockView, context: Context) {
        view.width = width
    }

    final class LockView: NSView {
        var width: CGFloat = 0 {
            didSet { if width != oldValue { apply() } }
        }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            apply()
        }

        private func apply() {
            // Deferred: this is called inside a SwiftUI update, and changing
            // the split view there would re-enter its layout.
            DispatchQueue.main.async { [weak self] in
                guard let self, self.window != nil, self.width > 0 else { return }
                var ancestor = self.superview
                while let view = ancestor, !(view is NSSplitView) { ancestor = view.superview }
                guard let splitView = ancestor as? NSSplitView,
                      let item = (splitView.delegate as? NSSplitViewController)?.splitViewItems
                        .first(where: { self.isDescendant(of: $0.viewController.view) }) else { return }
                if item.minimumThickness != self.width { item.minimumThickness = self.width }
                if item.maximumThickness != self.width { item.maximumThickness = self.width }
            }
        }
    }
}

struct AppWorkspaceView: View {
    let destination: AppDestination
    @Binding var settingsSection: AppSettingsSection
    let select: (AppDestination) -> Void
    let model: AppModel
    let backgroundAPI: AppBackgroundAPIController
    let updateController: AppUpdateController
    let benchmarks: BenchmarkController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        // A short cross-fade with a few points of rise when switching screens,
        // like a sidebar app's detail pane. Reduce Motion keeps only the fade.
        ZStack {
            workspace
                // Settings and its tabs are one page: moving between them
                // switches the tab, not the whole screen.
                .id(destination == .chat ? "chat" : "settings")
                .transition(reduceMotion
                    ? .opacity
                    : .opacity.combined(with: .offset(y: 6)))
        }
        // Only between Chat and Settings: Settings tabs that are their own
        // destinations (Models, Server, Benchmarks) ran this spring over the
        // whole page on every tab change, which felt slow.
        .animation(TUFFMotion.standard, value: destination == .chat)
        .accessibilityIdentifier("workspace.\(destination.rawValue)")
    }

    @ViewBuilder
    private var workspace: some View {
        Group {
            switch destination {
            case .chat:
                ChatWorkspaceView(model: model, openModels: { select(.models) })
            case .models, .benchmarks, .server, .settings:
                SettingsHubView(
                    tab: SettingsHubTab(destination: destination, section: settingsSection),
                    choose: { tab in
                        if let section = tab.settingsSection { settingsSection = section }
                        select(tab.destination)
                    },
                    model: model,
                    backgroundAPI: backgroundAPI,
                    updateController: updateController,
                    benchmarks: benchmarks)
            }
        }
    }
}

private struct ChatWorkspaceView: View {
    let model: AppModel
    let openModels: () -> Void
    @State private var conversationChromeHeight: CGFloat = 0
    @Environment(\.windowSidebarIsHidden) private var sidebarIsHidden
    @State private var transcriptWidth: CGFloat = 0
    @State private var frozenTranscriptWidth: CGFloat?
    /// The message box's latest height while the sidebar slides, applied
    /// once it settles.
    @State private var heldChromeHeight: CGFloat?

    var body: some View {
        primaryContent
            .frame(
                minWidth: AppWindowLayout.detailMinimumWidth,
                maxWidth: .infinity,
                maxHeight: .infinity)
            .safeAreaInset(edge: .top, spacing: 0) {
                StatusHUDView(model: model)
            }
            .ignoresSafeArea(.container, edges: .top)
            .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { transcriptWidth = $0 }
            .onChange(of: sidebarIsHidden) { _, _ in
                frozenTranscriptWidth = transcriptWidth
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.26) {
                    var settle = Transaction()
                    settle.disablesAnimations = true
                    withTransaction(settle) {
                        frozenTranscriptWidth = nil
                        if let held = heldChromeHeight { conversationChromeHeight = held }
                        heldChromeHeight = nil
                    }
                }
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
                    // Runs under the glass composer, which floats over it.
                    // While the sidebar slides, held at the width it had so
                    // the text re-wraps once at the end, not on every frame.
                    OutputPaneView(model: model, bottomInset: conversationChromeHeight)
                        .frame(width: frozenTranscriptWidth, alignment: .leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .clipped()
                        .transition(.opacity)
                } else if conversationChromeHeight > 0 {
                    OutputPaneView(model: model)
                        .frame(height: max(0, geometry.size.height - conversationChromeHeight))
                        .frame(maxHeight: .infinity, alignment: .top)
                        .transition(.opacity)
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
            // The first message, a new chat, and moving between an empty
            // chat and one with messages fade across rather than cut.
            .animation(TUFFMotion.standard, value: model.hasOutputTranscript)
            .onPreferenceChange(ConversationChromeHeightKey.self) { height in
                // Held while the sidebar slides: the message box re-wrapping
                // at each width changed the space under the conversation on
                // every frame, and the view re-pinned to the bottom each time.
                guard height > 0 else { return }
                guard frozenTranscriptWidth == nil else { heldChromeHeight = height; return }
                var transaction = Transaction()
                transaction.disablesAnimations = true
                withTransaction(transaction) { conversationChromeHeight = height }
            }
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 10) {
            // Not just an empty prompt box: a message that is so far only an
            // attached file is still a message being written, and offering
            // "try one of these instead" over it reads as a dead end.
            if !model.hasComposedInput && model.showPromptExamples && !model.isRunning {
                PromptExamplesView { preset in model.promptText = preset.prompt }
            }
            ModelActionBanner(model: model)
            PromptComposerView(model: model, openModels: openModels)
        }
        .padding(.horizontal, StatusHUDView.contentInset)
        .padding(.bottom, StatusHUDView.contentInset)
        .animation(TUFFMotion.standard, value: model.hasComposedInput)
        .animation(TUFFMotion.standard, value: model.showPromptExamples)
        .animation(TUFFMotion.standard, value: model.presentation.conversationAction)
        .animation(TUFFMotion.standard, value: model.presentation.conversationNotice)
    }
}

private struct ModelsWorkspaceView: View {
    let model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                WorkspaceTitle(
                    title: "Models")
                ModelLibraryColumnsView(model: model)
            }
            .workspaceColumn()
        }
        .workspacePageChrome()
    }
}

/// The tabs of the Settings page. Models, Server and Benchmarks are their
/// own destinations (the Navigate menu and its shortcuts still open them),
/// shown here as tabs; the rest are the app's settings sections.
enum SettingsHubTab: Hashable, Identifiable, CaseIterable {
    case general, models, generation, tools, server, benchmarks, advanced

    var id: Self { self }

    init(destination: AppDestination, section: AppSettingsSection) {
        switch destination {
        case .models: self = .models
        case .server: self = .server
        case .benchmarks: self = .benchmarks
        case .chat, .settings:
            switch section {
            case .general: self = .general
            case .models: self = .generation
            case .search: self = .tools
            case .advanced: self = .advanced
            }
        }
    }

    var destination: AppDestination {
        switch self {
        case .models: .models
        case .server: .server
        case .benchmarks: .benchmarks
        case .general, .generation, .tools, .advanced: .settings
        }
    }

    var settingsSection: AppSettingsSection? {
        switch self {
        case .general: .general
        case .generation: .models
        case .tools: .search
        case .advanced: .advanced
        case .models, .server, .benchmarks: nil
        }
    }

    var title: String {
        switch self {
        case .models: AppDestination.models.title
        case .server: AppDestination.server.title
        case .benchmarks: AppDestination.benchmarks.title
        default: settingsSection!.title
        }
    }

    var systemImage: String {
        switch self {
        case .models: AppDestination.models.systemImage
        case .server: AppDestination.server.systemImage
        case .benchmarks: AppDestination.benchmarks.systemImage
        default: settingsSection!.systemImage
        }
    }
}

private struct SettingsHubView: View {
    @Environment(\.windowSidebarIsHidden) private var sidebarIsHidden
    let tab: SettingsHubTab
    let choose: (SettingsHubTab) -> Void
    let model: AppModel
    let backgroundAPI: AppBackgroundAPIController
    let updateController: AppUpdateController
    let benchmarks: BenchmarkController

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            WorkspaceTitle(
                title: "Settings")
                .workspaceColumn()
                .padding(.bottom, -TUFFDesign.Space.xl)
            TUFFTabBar(
                tabs: SettingsHubTab.allCases,
                selection: Binding(get: { tab }, set: { choose($0) }),
                title: \.title,
                systemImage: \.systemImage)
                .padding(.horizontal, TUFFDesign.Space.page)
                .padding(.top, TUFFDesign.Space.md)
                .padding(.bottom, TUFFDesign.Space.xs)
            content
                .environment(\.workspaceEmbedded, true)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                // Content fades out under the tab bar instead of being cut off.
                .mask {
                    VStack(spacing: 0) {
                        LinearGradient(colors: [.clear, .black], startPoint: .top, endPoint: .bottom)
                            .frame(height: TUFFDesign.Space.sm)
                        Color.black
                    }
                }
                .id(tab)
                .transition(.opacity)
        }
        // A quick crossfade between tabs; the spring was too slow here.
        .animation(.easeOut(duration: 0.15), value: tab)
        // The title, the tabs and the page move as one when the sidebar
        // opens or closes; the tab bar used to jump while the title glided.
        .animation(TUFFMotion.sidebar, value: sidebarIsHidden)
        .workspacePageChrome()
    }

    @ViewBuilder private var content: some View {
        switch tab {
        case .models: ModelsWorkspaceView(model: model)
        case .server: ServerWorkspaceView(model: model, controller: backgroundAPI)
        case .benchmarks: BenchmarksWorkspaceView(model: model, controller: benchmarks)
        default:
            AppSettingsView(model: model, updateController: updateController,
                            section: tab.settingsSection ?? .general)
        }
    }
}

/// The column every workspace page lays its content in: left-aligned, the
/// page inset on every side, and the same distance from the window's top as
/// the chat's status strip. Used by each page so their titles and cards line
/// up when switching between them.
struct WorkspaceColumn: ViewModifier {
    var maxWidth: CGFloat
    @Environment(\.windowSidebarIsHidden) private var sidebarIsHidden
    @Environment(\.workspaceEmbedded) private var embedded

    func body(content: Content) -> some View {
        content
            .frame(maxWidth: maxWidth, alignment: .leading)
            .padding(.horizontal, TUFFDesign.Space.page)
            // With the sidebar collapsed, the title moves below the sidebar
            // toggle, which sits under the traffic lights.
            // Collapsed, the title clears the title bar row, where the
            // sidebar toggle sits beside the traffic lights.
            .padding(.top, embedded ? TUFFDesign.Space.sm
                     : sidebarIsHidden ? 60 : TUFFDesign.Space.page)
            .padding(.bottom, TUFFDesign.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
            // Moves on the sidebar's own curve, so the title travels with
            // the column instead of jumping when it starts.
            .animation(TUFFMotion.sidebar, value: sidebarIsHidden)
    }
}

extension View {
    /// Full width by default: a capped column left a dead band beside it,
    /// most of the window once the sidebar was collapsed.
    func workspaceColumn(maxWidth: CGFloat = .infinity) -> some View {
        modifier(WorkspaceColumn(maxWidth: maxWidth))
    }

    /// A page's scroll view runs to the top of the window, as the chat does;
    /// the column's own top padding keeps the title clear of it.
    func workspacePageChrome() -> some View {
        modifier(WorkspacePageChrome())
    }
}

private struct WorkspacePageChrome: ViewModifier {
    @Environment(\.workspaceEmbedded) private var embedded

    func body(content: Content) -> some View {
        if embedded {
            content.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        } else {
            content
                .ignoresSafeArea(.container, edges: .top)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .tuffPageBackground()
        }
    }
}

struct WorkspaceTitle: View {
    let title: String
    let subtitle: String?

    init(title: String, subtitle: String? = nil) {
        self.title = title
        self.subtitle = subtitle
    }
    @Environment(\.workspaceEmbedded) private var embedded

    var body: some View {
        if !embedded { titleBlock }
    }

    private var titleBlock: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .appFont(.largeTitle.bold())
                .accessibilityHeading(.h1)
            if let subtitle, !subtitle.isEmpty {
                Text(subtitle)
                    .appFont(.callout)
                    .foregroundStyle(.secondary)
            }
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

/// The desktop behind the window, blurred. Held active so it does not dim
/// when TUFF is not the frontmost app.
private struct WindowBlurView: NSViewRepresentable {
    var material: NSVisualEffectView.Material

    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = material
        view.blendingMode = .behindWindow
        view.state = .active
        return view
    }

    func updateNSView(_ view: NSVisualEffectView, context: Context) {
        view.material = material
        view.state = .active
    }
}

/// A toolbar item with nothing in it, so the window keeps a toolbar.
private struct InvisibleToolbarItem: ToolbarContent {
    var body: some ToolbarContent {
        if #available(macOS 26.0, *) {
            ToolbarItem(placement: .primaryAction) { Color.clear.frame(width: 1, height: 1) }
                .sharedBackgroundVisibility(.hidden)
        } else {
            ToolbarItem(placement: .primaryAction) { Color.clear.frame(width: 1, height: 1) }
        }
    }
}

/// Shows or hides the sidebar. Open, a small glass circle at the sidebar's
/// top right, where the system toggle sat. Collapsed, the same button grows
/// to the chat strip's height and glass and sits just before the strip, so
/// the two read as one row. One view throughout, so it moves between the
/// two places instead of disappearing and reappearing.
private struct SidebarToggleButton: View {
    let collapsed: Bool
    /// In Chat, collapsed, it joins the status strip; elsewhere there is no
    /// strip and it stays small in the title bar row.
    let besideStrip: Bool
    let sidebarWidth: CGFloat
    let action: () -> Void
    @State private var isHovered = false

    /// The floating sidebar panel's inset from the window and its corner
    /// radius. Open, the button is concentric with that corner: its centre
    /// is the corner's centre, so the gap to the top, the right and the
    /// curve between them is the same.
    static let panelInset: CGFloat = 8.5
    private static let panelRadius: CGFloat = 20
    private static let openSize: CGFloat = 30
    private static let extraInset: CGFloat = 4

    var body: some View {
        let large = collapsed && besideStrip
        let size: CGFloat = large ? StatusHUDView.stripHeight : Self.openSize
        // Moved in by the same amount from the top and the right, so the
        // two gaps stay equal with a little more room from the corner.
        let openCentre = CGPoint(x: Self.panelInset + sidebarWidth - Self.panelRadius - Self.extraInset,
                                 y: Self.panelInset + Self.panelRadius + Self.extraInset)
        Button(action: action) {
            Image(systemName: "sidebar.left")
                .appFont(large ? .title3.weight(.medium) : .body.weight(.medium))
                .foregroundStyle(isHovered ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                .frame(width: size, height: size)
                .tuffGlass(Circle(), highlighted: isHovered)
                .contentShape(Circle())
                .scaleEffect(isHovered ? 1.05 : 1)
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        // Collapsed, it sits below the traffic lights at the page's left
        // margin: in Chat at the strip's size, first in the strip's row;
        // elsewhere small, above the page title. Kept out of the title bar
        // row, where the window's toolbar took the clicks.
        .padding(.top, collapsed ? StatusHUDView.collapsedTop
                                 : openCentre.y - Self.openSize / 2)
        .padding(.leading, collapsed ? StatusHUDView.contentInset
                                     : openCentre.x - Self.openSize / 2)
        .animation(TUFFMotion.sidebar, value: collapsed)
        .animation(TUFFMotion.sidebar, value: large)
        .animation(TUFFMotion.hover, value: isHovered)
        .help("Show or Hide Sidebar (⌃⌘S)")
        .accessibilityLabel("Toggle sidebar")
    }
}

/// The sidebar toggle as a toolbar button, shown beside the traffic lights
/// while the sidebar is collapsed on a screen without the chat strip.
private struct CollapsedSidebarToolbarButton: ToolbarContent {
    let isShown: Bool
    let action: () -> Void

    var body: some ToolbarContent {
        ToolbarItem(placement: .navigation) {
            if isShown {
                Button(action: action) {
                    Image(systemName: "sidebar.left")
                }
                .help("Show Sidebar (⌃⌘S)")
                .accessibilityLabel("Show sidebar")
            }
        }
    }
}
