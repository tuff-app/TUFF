import TUFFAppCore
import SwiftUI
import TUFFMacPresentation

struct PromptExamplesView: View {
    let select: (AppPromptPreset) -> Void
    /// The panel's corner and padding; the cards inside use the difference,
    /// so their corners run concentric with the panel's.
    private static let radius: CGFloat = 24
    private static let padding: CGFloat = 12

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Try an example")
                        .appFont(.headline)
                }
                Spacer()
                moreExamples
            }

            LazyVGrid(columns: columns, alignment: .leading, spacing: 10) {
                primaryCards
            }
        }
        .padding(Self.padding)
        .padding(.top, 4)
        // Floats over the transcript like the message box under it.
        .tuffChrome(RoundedRectangle(cornerRadius: Self.radius, style: .continuous))
        .transition(.opacity.combined(with: .move(edge: .bottom)))
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder
    private var primaryCards: some View {
        ForEach(AppPromptPreset.primary) { preset in
            ExampleCard(preset: preset,
                        radius: Self.radius - Self.padding) { select(preset) }
            .accessibilityLabel(preset.title)
            .accessibilityValue(preset.prompt)
            .accessibilityHint("Copies this prompt into the prompt editor")
        }
    }

    private var columns: [GridItem] {
        Array(repeating: GridItem(.flexible(minimum: 0), spacing: 10, alignment: .top), count: 3)
    }

    private var moreExamples: some View {
        Menu {
            ForEach(AppPromptPreset.secondary) { preset in
                Button {
                    select(preset)
                } label: {
                    VStack(alignment: .leading) {
                        Text(preset.title)
                        Text(preset.prompt)
                    }
                }
            }
        } label: {
            MoreExamplesLabel()
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .accessibilityLabel("More examples")
        .accessibilityHint("Shows additional prompts")
    }
}

/// One example: a glass tile that brightens and lifts under the pointer.
private struct ExampleCard: View {
    let preset: AppPromptPreset
    let radius: CGFloat
    let select: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: select) {
            VStack(alignment: .leading, spacing: 6) {
                Text(preset.title)
                    .appFont(.caption.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                Text(preset.prompt)
                    .appFont(.caption2)
                    .foregroundStyle(TUFFDesign.secondaryTextColor)
                    .multilineTextAlignment(.leading)
                    .lineLimit(3)
                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, minHeight: 96, alignment: .topLeading)
            .padding(12)
            .tuffGlass(RoundedRectangle(cornerRadius: radius, style: .continuous),
                       highlighted: isHovered)
            .contentShape(.rect)
            .scaleEffect(isHovered ? 1.02 : 1)
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .animation(TUFFMotion.hover, value: isHovered)
    }
}

private struct MoreExamplesLabel: View {
    @State private var isHovered = false

    var body: some View {
        HStack(spacing: 6) {
            Text("More examples")
            Image(systemName: "chevron.down")
                .appFont(.caption2.weight(.bold))
                .foregroundStyle(TUFFDesign.secondaryTextColor)
        }
        .appFont(.callout.weight(.medium))
        .padding(.horizontal, 12)
        .frame(minHeight: 28)
        .tuffGlass(Capsule(), highlighted: isHovered)
        .contentShape(Capsule())
        .scaleEffect(isHovered ? 1.03 : 1)
        .onHover { isHovered = $0 }
        .animation(TUFFMotion.hover, value: isHovered)
    }
}
