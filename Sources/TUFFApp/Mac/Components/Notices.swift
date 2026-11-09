import AppKit
import SwiftUI
import TUFFAppCore
import TUFFMacPresentation

// MARK: - The card

/// The one look every popup in the app shares: an accent icon, a title, a
/// short message, then the actions, with the main one in the accent. Used
/// for questions (centred over the window) and for notices (sliding in
/// from the top), where the app used to mix system alerts, confirmation
/// dialogs, sheets and an inline error banner.
struct TUFFNoticeCard: View {
    var symbol: String
    var title: String
    var message: String?
    var text: Binding<String>?
    var actions: [TUFFNoticeAction]
    var close: (() -> Void)?
    var help: String?
    @FocusState private var fieldFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: symbol)
                    .appFont(.body.weight(.semibold))
                    .foregroundStyle(TUFFMacTheme.accentTextColor)
                    .frame(width: 34, height: 34)
                    .background(TUFFMacTheme.accentColor.opacity(0.16), in: Circle())
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .appFont(.headline)
                        .fixedSize(horizontal: false, vertical: true)
                    if let message {
                        Text(message)
                            .appFont(.callout)
                            .foregroundStyle(TUFFDesign.secondaryTextColor)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .help(help ?? "")
                Spacer(minLength: 0)
                if let close {
                    Button(action: close) {
                        Image(systemName: "xmark")
                            .appFont(.caption.weight(.bold))
                            .foregroundStyle(TUFFDesign.secondaryTextColor)
                            .frame(width: 26, height: 26)
                            .contentShape(Circle())
                            .tuffHoverGlass()
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Close")
                }
            }
            if let text {
                TextField("Name", text: text)
                    .tuffField()
                    .focused($fieldFocused)
                    .onSubmit { actions.last(where: \.isPrimary)?.run() }
            }
            if !actions.isEmpty {
                HStack(spacing: 8) {
                    Spacer(minLength: 0)
                    ForEach(actions) { action in
                        if action.isPrimary {
                            Button(action.title, action: action.run)
                                .buttonStyle(.tuffProminent)
                                .keyboardShortcut(.defaultAction)
                                .disabled(!action.isEnabled)
                        } else {
                            Button(action.title, action: action.run)
                                .buttonStyle(.tuffSecondary)
                                .keyboardShortcut(action.isCancel ? .cancelAction : nil)
                        }
                    }
                }
            }
        }
        .padding(20)
        .frame(width: 400)
        .onAppear { fieldFocused = text != nil }
    }
}

struct TUFFNoticeAction: Identifiable {
    let id = UUID()
    let title: String
    var isPrimary = false
    var isCancel = false
    var isEnabled = true
    let run: () -> Void

    static func primary(_ title: String, enabled: Bool = true,
                        _ run: @escaping () -> Void) -> TUFFNoticeAction {
        TUFFNoticeAction(title: title, isPrimary: true, isEnabled: enabled, run: run)
    }

    static func secondary(_ title: String, _ run: @escaping () -> Void) -> TUFFNoticeAction {
        TUFFNoticeAction(title: title, run: run)
    }

    static func cancel(_ title: String = "Cancel", _ run: @escaping () -> Void = {}) -> TUFFNoticeAction {
        TUFFNoticeAction(title: title, isCancel: true, run: run)
    }
}

// MARK: - Questions, centred over the window

extension View {
    /// Asks before doing something, in the app's card: a stand-in for
    /// `confirmationDialog` and `alert`, whose system styling drew Delete in
    /// red and default buttons in the system accent.
    func tuffConfirm(
        _ title: String,
        isPresented: Binding<Bool>,
        message: String? = nil,
        symbol: String = "questionmark.circle",
        confirm: String,
        cancel: String = "Cancel",
        text: Binding<String>? = nil,
        canConfirm: Bool = true,
        action: @escaping () -> Void
    ) -> some View {
        sheet(isPresented: isPresented) {
            TUFFNoticeCard(
                symbol: symbol, title: title, message: message, text: text,
                actions: [
                    .cancel(cancel) { isPresented.wrappedValue = false },
                    .primary(confirm, enabled: canConfirm) {
                        action()
                        isPresented.wrappedValue = false
                    },
                ])
                .presentationBackground(TUFFDesign.cardColor)
        }
    }
}

// MARK: - Notices, from the top

/// One notice at a time, shown at the top of the window: errors, and the
/// occasional prompt. Stays until it is dismissed or acted on, unless it
/// says otherwise.
@MainActor @Observable
final class TUFFNoticeCenter {
    static let shared = TUFFNoticeCenter()

    struct Notice: Identifiable {
        let id = UUID()
        /// Lets a caller find its own notice again, to replace or dismiss it.
        var key: String
        var symbol: String
        var title: String
        var message: String?
        var help: String?
        var actions: [TUFFNoticeAction] = []
        var autoDismissAfter: Double?
        /// Shown small, inside the chat's status strip on its right, for
        /// prompts that should stay out of the way; the overlay skips it.
        var compact = false
        /// Run when the notice is closed with its close button.
        var onClose: () -> Void = {}
    }

    private(set) var current: Notice?

    func show(_ notice: Notice) {
        withAnimation(TUFFMotion.standard) { current = notice }
        if let delay = notice.autoDismissAfter {
            let id = notice.id
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                if self?.current?.id == id { self?.dismiss() }
            }
        }
    }

    /// Runs one of the current notice's actions and puts the notice away.
    func run(_ action: TUFFNoticeAction) {
        action.run()
        dismiss()
    }

    func dismiss(key: String? = nil) {
        guard let current, key == nil || current.key == key else { return }
        withAnimation(TUFFMotion.standard) { self.current = nil }
    }
}

/// Shows the notice center's current notice at the top of the window,
/// centred under the chat's status strip. Compact notices show in the
/// strip instead.
struct TUFFNoticeOverlay: View {
    let center: TUFFNoticeCenter

    var body: some View {
        VStack {
            if let notice = center.current, !notice.compact {
                TUFFNoticeCard(
                    symbol: notice.symbol, title: notice.title, message: notice.message,
                    actions: notice.actions.map { action in
                        // Every action also puts the notice away.
                        TUFFNoticeAction(title: action.title, isPrimary: action.isPrimary,
                                         isCancel: action.isCancel, isEnabled: action.isEnabled) {
                            center.run(action)
                        }
                    },
                    close: {
                        notice.onClose()
                        center.dismiss()
                    },
                    help: notice.help)
                    .tuffChrome(RoundedRectangle(cornerRadius: 22, style: .continuous))
                    .shadow(color: .black.opacity(0.18), radius: 18, y: 8)
                    .id(notice.id)
                    .transition(.offset(y: -16).combined(with: .opacity))
            }
            Spacer(minLength: 0)
        }
        // Below the chat's status strip (28 pt margin + 54 pt strip), and
        // the same height on other screens so notices always land in one
        // place.
        .padding(.top, 96)
        .frame(maxWidth: .infinity)
    }
}

/// A compact notice as one short row at the right of the status strip: its
/// symbol, title and actions, with no card of its own.
struct TUFFStripNotice: View {
    let notice: TUFFNoticeCenter.Notice
    let center: TUFFNoticeCenter

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: notice.symbol)
                .appFont(.caption.weight(.semibold))
                .foregroundStyle(TUFFMacTheme.accentTextColor)
            Text(notice.title)
                .appFont(.caption.weight(.medium))
                .foregroundStyle(TUFFDesign.secondaryTextColor)
                .lineLimit(1)
                .help(notice.message ?? "")
            ForEach(notice.actions) { action in
                Button(action.title) { center.run(action) }
                    .buttonStyle(action.isPrimary
                                 ? AnyButtonStyle(.tuffProminent) : AnyButtonStyle(.tuffSecondary))
            }
        }
        .controlSize(.small)
        .fixedSize()
    }
}

/// Lets the strip pick a button style per action.
private struct AnyButtonStyle: ButtonStyle {
    private let make: (Configuration) -> AnyView
    init<S: ButtonStyle>(_ style: S) { make = { AnyView(style.makeBody(configuration: $0)) } }
    func makeBody(configuration: Configuration) -> some View { make(configuration) }
}

// MARK: - Asking for a star

/// Asks, now and then, for a star on GitHub. First on the fifth launch,
/// some time after the window opens and only while the chat is idle.
/// "Star on GitHub" ends it; "Later" (or closing it) asks again ten
/// launches later.
@MainActor
enum StarPrompt {
    static let repository = URL(string: "https://github.com/tuff-app/TUFF")!
    private static let launchesKey = "TUFFStarPromptLaunches"
    private static let nextKey = "TUFFStarPromptNextLaunch"
    private static let doneKey = "TUFFStarPromptDone"
    private static let firstAsk = 5
    private static let laterGap = 10

    /// Counts this launch; true when this is a launch to ask on.
    static func registerLaunch(defaults: UserDefaults = .standard) -> Bool {
        guard !defaults.bool(forKey: doneKey) else { return false }
        let launches = defaults.integer(forKey: launchesKey) + 1
        defaults.set(launches, forKey: launchesKey)
        let next = defaults.object(forKey: nextKey) as? Int ?? firstAsk
        return launches >= next
    }

    static func notice(defaults: UserDefaults = .standard) -> TUFFNoticeCenter.Notice {
        let later = {
            defaults.set(defaults.integer(forKey: launchesKey) + laterGap, forKey: nextKey)
        }
        return TUFFNoticeCenter.Notice(
            key: "star",
            symbol: "star.fill",
            title: "Enjoying TUFF?",
            message: "A star on GitHub helps other people find TUFF.",
            actions: [
                .secondary("Later", later),
                .primary("Star on GitHub") {
                    defaults.set(true, forKey: doneKey)
                    NSWorkspace.shared.open(repository)
                },
            ],
            compact: true,
            onClose: later)
    }
}
