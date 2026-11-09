import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

struct ModelStatusBadge: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            statusDot
            Text(model.installDescriptor.displayName)
                .appFont(.callout.weight(.semibold))
                .lineLimit(1)
                .help(model.installDescriptor.repoID)
                .accessibilityLabel("Model")
                .accessibilityValue(model.installDescriptor.repoID)
        }
    }

    /// One dot whose color eases between states instead of being swapped.
    private var statusDot: some View {
        Circle()
            .fill(dotColor)
            .frame(width: 8, height: 8)
            .animation(TUFFMotion.standard, value: model.presentation.severity)
            .accessibilityHidden(true)
    }

    private var dotColor: Color {
        switch model.presentation.severity {
        case .neutral: .gray
        // Working, not failing: the same purple the phase label and the send
        // button use, rather than a caution colour for an ordinary load.
        case .active: TUFFMacTheme.accentTextColor
        case .warning: TUFFMacTheme.accentTextColor
        case .success: .green
        case .error: TUFFMacTheme.accentTextColor
        }
    }
}
