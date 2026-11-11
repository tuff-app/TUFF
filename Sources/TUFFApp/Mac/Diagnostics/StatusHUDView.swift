import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

struct StatusHUDView: View {
    let model: AppModel
    @Environment(\.windowSidebarIsHidden) private var sidebarIsHidden
    @State private var ejectTaps = 0
    private let noticeCenter = TUFFNoticeCenter.shared

    /// The transcript's own side margin, so the strip and the messages under
    /// it share a left edge.
    static let contentInset: CGFloat = TUFFDesign.Space.page

    /// The strip's height: its 34 pt row plus 10 pt above and below.
    static let stripHeight: CGFloat = 54
    /// With the sidebar collapsed, the top of the row below the traffic
    /// lights that holds the sidebar toggle and the strip.
    static let collapsedTop: CGFloat = 40

    var body: some View {
        let collapsed = sidebarIsHidden
        strip
            // Same height whether or not the sidebar is shown. Collapsed, it
            // only moves right to make room for the sidebar toggle, which
            // takes the strip's height and glass and sits just before it.
            .padding(.top, collapsed ? Self.collapsedTop : Self.contentInset)
            .padding(.leading, collapsed
                     ? Self.contentInset + Self.stripHeight + 10
                     : Self.contentInset)
            .padding(.trailing, Self.contentInset)
            // The extra top room when collapsed comes out of the bottom
            // spacing, so the strip's total height, and with it the space
            // held above the conversation, never changes. It did, and the
            // conversation jumped a line while the sidebar slid.
            .padding(.bottom, collapsed ? 12 - (Self.collapsedTop - Self.contentInset) : 12)
            .animation(TUFFMotion.sidebar, value: collapsed)
            // No surface behind the strip: the transcript scrolls under it,
            // and the glass blurs it.
    }

    private var strip: some View {
        HStack(spacing: 12) {
            ModelStatusBadge(model: model)
            if !model.isAppleSelected {
                Divider().frame(height: 16)
                PhaseLabel(model: model)
            }
            Spacer(minLength: 12)
            if let notice = noticeCenter.current, notice.compact {
                TUFFStripNotice(notice: notice, center: noticeCenter)
                    .transition(.offset(x: 12).combined(with: .opacity))
            } else if showsMetrics {
                HUDMetricView(value: rateText, label: "tok/s", animated: !model.isRunning)
                HUDMetricView(value: tokensText, label: "tokens", animated: !model.isRunning)
                HStack(spacing: 2) {
                    HUDMetricView(value: memoryText, label: "memory", animated: !model.isRunning)
                        .help(memoryHelp)
                    InfoPopoverButton(subject: "Memory", text: memoryHelp, arrowEdge: .bottom)
                    ejectButton
                }
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 34)
        // Attached here so they run even while the window holds animations
        // back during a reply.
        .animation(TUFFMotion.standard, value: model.canUnloadModel)
        .animation(TUFFMotion.standard, value: showsMetrics)
        .animation(TUFFMotion.standard, value: noticeCenter.current?.id)
        .padding(.leading, 18)
        .padding(.trailing, 14)
        .padding(.vertical, 10)
        .tuffChrome(Capsule())
    }

    /// Unloading is what actually returns the model's memory, so the control
    /// sits with the memory readout rather than in a menu.
    ///
    /// Clicked, the symbol gives a quick upward bounce and lifts out as the
    /// model unloads; it scales back in when a model is loaded again.
    @ViewBuilder
    private var ejectButton: some View {
        if model.canUnloadModel {
            Button {
                ejectTaps += 1
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.18) { model.unloadModel() }
            } label: {
                Image(systemName: "eject.fill")
                    .imageScale(.small)
                    .symbolEffect(.bounce.up, value: ejectTaps)
                    .frame(width: 26, height: 26)
                    .contentShape(Rectangle())
                    .tuffHoverGlass()
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("Unload this model and free its memory")
            .accessibilityLabel("Unload model")
            .transition(.asymmetric(
                insertion: .opacity.combined(with: .scale(scale: 0.6)),
                removal: .opacity.combined(with: .offset(y: -12))))
        }
    }

    private var rateText: String {
        if model.phase == .decode { return MetricFormat.rate(model.liveTokensPerSecond) }
        if let d = model.diagnostics { return MetricFormat.rate(d.tokensPerSecond) }
        return "\u{2014}"
    }

    private var tokensText: String {
        if model.isRunning { return "\(model.liveTokenCount)" }
        if let d = model.diagnostics { return "\(d.generatedTokens)" }
        return "\u{2014}"
    }

    /// `phys_footprint`: what this process is charged. Measured, not assumed —
    /// the model and image tower weights are memory-mapped and read by the GPU,
    /// and no per-process counter attributes them: 1,144 MB of tower retained
    /// moves the footprint by 5 MB. Those bytes are page cache, owned by the
    /// kernel and reclaimable, so the popover explains them in words and the
    /// diagnostics section carries their measured row.
    private var memoryText: String {
        MetricFormat.memory(model.currentProcessMemoryBytes)
    }

    private var memoryHelp: String {
        MemoryFootprintExplanation.text(chargedBytes: model.currentProcessMemoryBytes)
    }

    /// Only while a model is loaded: the last reply's numbers would
    /// otherwise linger after an eject.
    private var showsMetrics: Bool {
        !model.isAppleSelected && (model.loadState.isReady || model.isRunning)
    }
}

private struct PhaseLabel: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            switch content {
            case .loading(let label):
                ProgressView().controlSize(.mini)
                Text(label)
            case .pulse(let label):
                if !model.isAppleSelected { PulsingDot() }
                RollingText(label)
            case .steady(let label):
                Circle().fill(TUFFMacTheme.accentColor).frame(width: 7, height: 7)
                Text(label).contentTransition(.opacity)
            case .quiet(let label):
                Text(label)
                    .foregroundStyle(.secondary)
                    .contentTransition(.opacity)
            }
        }
        .appFont(.caption.weight(.medium))
        .lineLimit(1)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Model status")
        .accessibilityValue(statusText)
    }

    private enum Content {
        case loading(String)
        case pulse(String)
        case steady(String)
        case quiet(String)
    }

    private var content: Content {
        if model.isAppleSelected {
            return model.isSelectedConversationRunning
                ? .pulse("Generating")
                : .quiet(model.selectedAppleAvailability.label)
        }
        let presentation = model.presentation
        if presentation.showsActivity { return .loading(presentation.label) }
        if model.isRunning && model.phase == .prefill { return .pulse(presentation.label) }
        if model.isRunning && model.phase == .decode { return .steady(presentation.label) }
        return .quiet(presentation.label)
    }

    private var statusText: String {
        switch content {
        case .loading(let label), .pulse(let label), .steady(let label), .quiet(let label):
            label
        }
    }
}

private struct PulsingDot: View {
    var body: some View {
        Circle()
            .fill(TUFFMacTheme.accentColor)
            .frame(width: 7, height: 7)
            .phaseAnimator([0.4, 1.0]) { dot, opacity in
                dot.opacity(opacity)
            } animation: { _ in
                .easeInOut(duration: 0.7)
            }
    }
}
