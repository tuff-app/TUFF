import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

struct ModelActionBanner: View {
    @Bindable var model: AppModel

    var body: some View {
        // Not for a model that is merely unloaded: Send loads it first, and
        // the status strip already says "Not loaded". A permanent bar over
        // the transcript for that was noise.
        if let action = model.presentation.conversationAction,
           action != .load,
           model.hasOutputTranscript {
            banner(tint: iconColor(for: action)) {
                Image(systemName: iconName(for: action))
                    .foregroundStyle(iconColor(for: action))
                    .accessibilityHidden(true)
                Text(message(for: action))
                    .appFont(.callout)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                Button(buttonTitle(for: action)) {
                    model.perform(action)
                }
                .buttonStyle(.tuffSecondary)
                .controlSize(.regular)
            }
            .transition(.tuffArrival(offsetY: 6))
        } else if let notice = model.presentation.conversationNotice {
            // Shown whether or not anything has been said yet: this is the
            // state where the composer refuses and there is no button to
            // press, so the reason has to be on screen from the start.
            banner(tint: TUFFMacTheme.accentTextColor) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(TUFFMacTheme.accentTextColor)
                    .accessibilityHidden(true)
                Text(notice)
                    .appFont(.callout)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
            }
            .transition(.tuffArrival(offsetY: 6))
        }
    }

    @ViewBuilder
    private func banner(
        tint: Color,
        @ViewBuilder content: () -> some View
    ) -> some View {
        HStack(spacing: 10) {
            content()
        }
        .padding(.leading, 16)
        .padding(.trailing, 10)
        .padding(.vertical, 10)
        .frame(minHeight: 48)
        // The card and hairline every other surface uses; the icon carries
        // the tone. A colored outline read as an alarm for a routine step.
        .tuffGlass(RoundedRectangle(cornerRadius: PromptComposerMetrics.cornerRadius,
                                        style: .continuous), interactive: false)
        .accessibilityElement(children: .contain)
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    private func message(for action: AppModelAction) -> String {
        switch action {
        case .load:
            return "The model is unloaded. Load it to generate."
        case .retryLoad:
            return model.presentation.detail ?? "The model could not be loaded."
        case .reload:
            return "Settings changed. Reload the model to continue."
        case .install, .cancelInstall, .cancelLoad, .unload:
            return model.presentation.label
        }
    }

    private func buttonTitle(for action: AppModelAction) -> String {
        switch action {
        case .load: return "Load Model"
        case .retryLoad: return "Retry Load"
        case .reload: return "Reload Model"
        case .install: return "Install"
        case .cancelInstall: return "Cancel"
        case .cancelLoad: return "Cancel Load"
        case .unload: return "Unload Model"
        }
    }

    private func iconName(for action: AppModelAction) -> String {
        switch action {
        case .retryLoad: return "exclamationmark.triangle.fill"
        case .reload: return "arrow.clockwise.circle.fill"
        case .load, .install, .cancelInstall, .cancelLoad, .unload:
            return "externaldrive.fill"
        }
    }

    /// An unloaded model is a step to take, not a fault: it wears the app's
    /// purple like every other "do this next" control. Orange is kept for the
    /// things that really are wrong, and red for a load that failed.
    private func iconColor(for action: AppModelAction) -> Color {
        action == .retryLoad ? TUFFMacTheme.accentTextColor : TUFFMacTheme.accentTextColor
    }
}
