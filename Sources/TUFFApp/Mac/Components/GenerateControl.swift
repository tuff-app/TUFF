import AppKit
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

struct GenerateControl: View {
    let model: AppModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private let controlHeight: CGFloat = 32
    @State private var sendHovered = false

    enum Mode: Equatable { case running, loading, otherChatRunning, send }

    static func mode(selectedRunning: Bool, loading: Bool, anyRunning: Bool, canSubmit: Bool) -> Mode {
        if selectedRunning { return .running }
        if loading { return .loading }
        if anyRunning && !canSubmit { return .otherChatRunning }
        return .send
    }

    var body: some View {
        let mode = Self.mode(selectedRunning: model.isSelectedConversationRunning,
                             loading: model.isLoadingForSubmission,
                             anyRunning: model.isRunning, canSubmit: model.canSubmit)
        ZStack {
            switch mode {
            case .running: runningPill.transition(swap)
            case .loading: loadingPill.transition(swap)
            case .otherChatRunning: anotherChatRunningPill.transition(swap)
            case .send: generateButton.transition(swap)
            }
        }
        .animation(reduceMotion ? .easeInOut(duration: 0.2) : TUFFMotion.standard, value: mode)
    }

    /// Send, Stop and the loading pill trade places with a quick fade and a
    /// slight scale, the way a toolbar control changes state. Reduce Motion
    /// keeps the fade only.
    private var swap: AnyTransition {
        reduceMotion ? .opacity : .opacity.combined(with: .scale(scale: 0.94))
    }

    private var generateButton: some View {
        Button {
            model.submit()
        } label: {
            // Always "Send". An unloaded model is loaded first, but that is the
            // app's business, not a different button.
            Label("Send", systemImage: "arrow.up")
                .appFont(.callout.weight(.semibold))
                .padding(.horizontal, 18)
                .frame(minWidth: 92, minHeight: controlHeight)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        // Disabled reads as a quiet grey pill, not a washed-out accent one
        // that looked pressable and muddy at once.
        .foregroundStyle(model.canSubmit ? AnyShapeStyle(TUFFMacTheme.onAccentColor) : AnyShapeStyle(.tertiary))
        // Accent glass when it can send; clear glass when there is nothing to send.
        .tuffGlass(Capsule(), tint: model.canSubmit ? TUFFMacTheme.accentColor : nil,
                   highlighted: sendHovered && model.canSubmit)
        .scaleEffect(sendHovered && model.canSubmit ? 1.03 : 1)
        .animation(TUFFMotion.hover, value: sendHovered)
        .onHover { sendHovered = $0 }
        .background {
            // Command-Return sends without the flight, which is for clicks.
            Button("Send", action: model.submit)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!model.canSubmit)
                .opacity(0)
                .allowsHitTesting(false)
                .accessibilityHidden(true)
        }
        .disabled(!model.canSubmit)
        .animation(TUFFMotion.hover, value: model.canSubmit)
        .accessibilityHint(model.canRun
            ? "Sends the prompt to the selected local model"
            : "Loads the selected local model, then sends the prompt")
    }

    private var loadingPill: some View {
        Button(action: model.cancelLoad) {
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text(model.presentation.label)
                    .appFont(.callout.weight(.medium))
                Label("Cancel model load", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
            }
            .padding(.horizontal, 16)
            .frame(minWidth: 140, minHeight: controlHeight)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(TUFFMacTheme.onAccentColor)
        .tuffGlass(Capsule(), tint: TUFFMacTheme.accentColor)
        .disabled(!model.canCancelLoad)
        .help("Cancel loading and keep this draft")
        .accessibilityLabel("Cancel model load")
        .accessibilityValue(model.presentation.label)
    }

    private var anotherChatRunningPill: some View {
        HStack(spacing: 9) {
            ProgressView().controlSize(.small)
            Text("Another chat is running")
                .appFont(.callout.weight(.medium))
        }
        .padding(.horizontal, 16)
        .frame(minWidth: 170, minHeight: controlHeight)
        .foregroundStyle(.secondary)
        .tuffGlass(Capsule(), interactive: false)
        .accessibilityLabel("Another chat is generating")
    }

    private var runningPill: some View {
        Button {
            model.cancel()
        } label: {
            HStack(spacing: 10) {
                if model.isCancellationPending {
                    Text("Stopping")
                        .appFont(.callout.weight(.medium))
                } else if model.phase == .prefill {
                    RollingText(model.presentation.label)
                        .appFont(.callout.weight(.medium))
                } else {
                    RollingText("\(MetricFormat.rate(model.liveTokensPerSecond)) tok/s")
                        .appFont(.callout.weight(.semibold))
                }
                Label("Stop generation", systemImage: "stop.fill")
                    .labelStyle(.iconOnly)
                    .appFont(.callout)
                    .frame(width: 28, height: 28)
            }
            .padding(.leading, 18)
            .padding(.trailing, 4)
            .frame(minWidth: 140, minHeight: controlHeight)
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(TUFFMacTheme.onAccentColor)
        .tuffGlass(Capsule(), tint: TUFFMacTheme.accentColor)
        .keyboardShortcut(.cancelAction)
        .disabled(!model.canCancel)
        .help("Stop generation")
        .accessibilityLabel(model.isCancellationPending
                            ? "Stopping generation" : "Stop generation")
        .accessibilityValue(model.presentation.label)
        .animation(TUFFMotion.standard, value: model.presentation.label)
    }
}
