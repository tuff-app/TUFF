import SwiftUI
import TUFFMacPresentation

struct HUDMetricView: View {
    let value: String
    let label: String
    var animated = true

    var body: some View {
        VStack(spacing: 1) {
            RollingText(value, animated: animated)
                .appFont(.system(.callout, design: .rounded).weight(.semibold))
            Text(label)
                .appFont(.caption2)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 56)
    }
}
