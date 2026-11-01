import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

struct GenerateControl: View {
    let model: AppModel
    private let controlHeight: CGFloat = 34

    enum Mode: Equatable { case running, loading, otherChatRunning, send }

    static func mode(selectedRunning: Bool, loading: Bool, anyRunning: Bool, canSubmit: Bool) -> Mode {
        if selectedRunning { return .running }
        if loading { return .loading }
        if anyRunning && !canSubmit { return .otherChatRunning }
        return .send
    }

    var body: some View {
        switch Self.mode(selectedRunning: model.isSelectedConversationRunning,
                         loading: model.isLoadingForSubmission,
                         anyRunning: model.isRunning, canSubmit: model.canSubmit) {
        case .running: runningPill
        case .loading: loadingPill
        case .otherChatRunning: anotherChatRunningPill
        case .send: generateButton
        }
    }

    private var generateButton: some View {
        Button {
            model.submit()
        } label: {
            // Always "Send". An unloaded model is loaded first, but that is the
            // app's business, not a different button.
            Label("Send", systemImage: "arrow.up")
                .appFont(.callout.weight(.semibold))
                .padding(.horizontal, 24)
                .frame(minWidth: 124, minHeight: controlHeight)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(TUFFMacTheme.accentColor, in: .capsule)
        .overlay {
            Capsule().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(!model.canSubmit)
        .opacity(model.canSubmit ? 1 : 0.62)
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
        .foregroundStyle(.white)
        .background(TUFFMacTheme.accentColor, in: .capsule)
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
        .background(.quaternary, in: .capsule)
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
                    Text(model.presentation.label)
                        .appFont(.callout.weight(.medium))
                        .monospacedDigit()
                        .contentTransition(.numericText())
                } else {
                    Text("\(MetricFormat.rate(model.liveTokensPerSecond)) tok/s")
                        .appFont(.callout.weight(.semibold))
                        .monospacedDigit()
                        .contentTransition(.numericText())
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
        .foregroundStyle(.white)
        .background(TUFFMacTheme.accentColor, in: .capsule)
        .overlay {
            Capsule().stroke(.white.opacity(0.16), lineWidth: 0.5)
        }
        .keyboardShortcut(.cancelAction)
        .disabled(!model.canCancel)
        .help("Stop generation")
        .accessibilityLabel(model.isCancellationPending
                            ? "Stopping generation" : "Stop generation")
        .accessibilityValue(model.presentation.label)
        .animation(.smooth(duration: 0.2), value: model.presentation.label)
    }
}
