import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// The composer's model and thinking controls, as quiet icons. The model's
/// name is already in the message placeholder, so the bar itself stays small;
/// thinking opens its choices in place and folds back once one is picked.
struct ChatControlsView: View {
    @Bindable var model: AppModel
    @State private var showsReasoningChoices = false

    var body: some View {
        HStack(spacing: 6) {
            modelMenu
            if !model.isAppleSelected { reasoningControl }
        }
        .appFont(.callout)
        .disabled(model.isRunning || model.loadState.isLoading)
        .animation(TUFFMotion.standard, value: showsReasoningChoices)
    }

    // Only what is on this Mac. Offering the rest of the catalogue here meant
    // picking one replaced the conversation with the downloader; the Models
    // screen is where downloads live.
    private var modelMenu: some View {
        Menu {
            ForEach(model.availableAppleModels) { kind in
                Button(kind.name) { model.selectAppleModel(kind) }
            }
            ForEach(model.installedInstalls) { install in
                Button {
                    model.selectModel(install)
                } label: {
                    if install.id == model.selectedModelID {
                        Label(install.descriptor.shortName, systemImage: "checkmark")
                    } else {
                        Text(install.descriptor.shortName)
                    }
                }
            }
        } label: {
            ComposerIcon(systemImage: "shippingbox", isOn: false)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Model: \(model.selectedModelName)")
        .accessibilityLabel("Chat model")
        .accessibilityValue(model.selectedModelName)
    }

    @ViewBuilder
    private var reasoningControl: some View {
        switch model.selectedDescriptor.reasoningControl {
        case .toggle, .toggleWithPreservation:
            ReasoningMorph(isOn: model.reasoning == .on,
                           help: "Thinking: \(model.reasoning == .on ? "on" : "off")",
                           options: [("Off", ChatReasoning.off), ("On", ChatReasoning.on)],
                           selection: $model.reasoning,
                           isOpen: $showsReasoningChoices)
        case .graded:
            ReasoningMorph(isOn: true,
                           help: "Reasoning: \(model.reasoningEffort.rawValue)",
                           options: [("Low", GPTOSSReasoningEffort.low), ("Medium", .medium), ("High", .high)],
                           selection: $model.reasoningEffort,
                           isOpen: $showsReasoningChoices)
        case .alwaysOn:
            ComposerIcon(systemImage: "brain", isOn: true)
                .help("This model always thinks before answering")
                .accessibilityLabel("Thinking always on")
        case nil:
            EmptyView()
        }
    }
}

/// The thinking control. Clicked, the round brain button grows into one glass
/// capsule holding the choices; picking one (or clicking the brain again)
/// folds it back into the button. One shape throughout, so it reads as the
/// button opening rather than a second control appearing beside it.
private struct ReasoningMorph<Value: Hashable>: View {
    let isOn: Bool
    let help: String
    let options: [(String, Value)]
    @Binding var selection: Value
    @Binding var isOpen: Bool
    @State private var isHovered = false

    var body: some View {
        HStack(spacing: 2) {
            Button {
                isOpen.toggle()
            } label: {
                Image(systemName: "brain")
                    .appFont(.body)
                    .foregroundStyle(isOn ? AnyShapeStyle(TUFFMacTheme.accentTextColor)
                                     : isHovered ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
                    .rotationEffect(.degrees(isOpen ? -8 : 0))
                    .frame(width: 32, height: 32)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help(help)
            .accessibilityLabel("Thinking")
            .accessibilityValue(help)
            if isOpen {
                // The Settings glass selector: the capsule slides to the
                // choice (or follows a drag), then the control folds back.
                TUFFTabBar(
                    tabs: options.map(\.1),
                    selection: Binding(
                        get: { selection },
                        set: { value in
                            selection = value
                            // Let the glass settle on the choice, then fold.
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) {
                                withAnimation(TUFFMotion.emphasized) { isOpen = false }
                            }
                        }),
                    title: { value in options.first { $0.1 == value }?.0 ?? "" },
                    compact: true,
                    showsTrack: false)
                    .fixedSize()
                    // Opens out of the brain and folds back into it the
                    // same way.
                    .transition(.opacity
                        .combined(with: .scale(scale: 0.5, anchor: .leading))
                        .combined(with: .offset(x: -16)))
            }
        }
        // No extra inset: the selector's own 3 pt padding is the same on
        // every side, so its capsule sits concentric inside this one.
        .tuffGlass(Capsule(),
                   tint: isOn && !isOpen ? TUFFMacTheme.accentColor.opacity(0.3) : nil,
                   highlighted: isHovered && !isOpen)
        .scaleEffect(isHovered && !isOpen ? 1.06 : 1)
        .onHover { isHovered = $0 }
        .fixedSize()
        .animation(TUFFMotion.emphasized, value: isOpen)
        .animation(TUFFMotion.hover, value: isHovered)
    }
}

/// A round icon control for the composer, tinted while its option is on.
struct ComposerIcon: View {
    let systemImage: String
    let isOn: Bool
    var badge: Color?
    @State private var isHovered = false

    var body: some View {
        Image(systemName: systemImage)
            .appFont(.body)
            .foregroundStyle(isOn ? AnyShapeStyle(TUFFMacTheme.accentTextColor)
                             : isHovered ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
            .frame(width: 32, height: 32)
            // Glass, tinted with the accent while its option is on.
            .tuffGlass(Circle(), tint: isOn ? TUFFMacTheme.accentColor.opacity(0.3) : nil,
                       highlighted: isHovered)
            .scaleEffect(isHovered ? 1.06 : 1)
            // Switching on or off springs the tint in or out with a small
            // bounce of the symbol.
            .symbolEffect(.bounce, value: isOn)
            .animation(TUFFMotion.standard, value: isOn)
            .animation(TUFFMotion.hover, value: isHovered)
            .onHover { isHovered = $0 }
            .overlay(alignment: .topTrailing) {
                if let badge {
                    Circle().fill(badge).frame(width: 7, height: 7).offset(x: -2, y: 2)
                }
            }
            .contentShape(Circle())
    }

}
