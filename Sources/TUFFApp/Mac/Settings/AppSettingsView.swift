import AppKit
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFAppCore
import TUFFAppUpdater
import TUFFMacPresentation
import SwiftUI

struct AppSettingsView: View {
    @Bindable var model: AppModel
    @Bindable var updateController: AppUpdateController
    /// Chosen by the Settings tab bar, which this view sits under.
    let section: AppSettingsSection
    @State private var profileModelID = ""
    /// Holds text the user is mid-typing into the hex field until it parses,
    /// so an incomplete value (e.g. "6F4") isn't snapped back to the last
    /// valid one on every keystroke.
    @State private var pendingAccentHex: String?
    @State private var autoLearningMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Group {
                switch section {
                case .general: generalSettings
                case .models:
                    if model.isAppleSelected { appleManagedSettings }
                    else { modelSettings(advanced: false) }
                case .search: SearchSettingsView(model: model)
                case .advanced:
                    if model.isAppleSelected { appleManagedSettings }
                    else { modelSettings(advanced: true) }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .id(section)
            .transition(.opacity)
        }
        .animation(TUFFMotion.standard, value: section)
        // A setting that shows or hides the rows under it: they ease in and
        // the cards below move, instead of the page jumping.
        .animation(TUFFMotion.standard, value: rowsKey)
        .tuffPageBackground()
    }

    private var appleManagedSettings: some View {
        Form {
            Section("System Prompt") {
                TextEditor(text: $model.modelSystemPrompt)
                    .appFont(.body)
                    .scrollContentBackground(.hidden)
                    .padding(10)
                    .frame(minHeight: 120)
                    .background(TUFFDesign.insetColor, in: RoundedRectangle(cornerRadius: TUFFDesign.Radius.control))
                    .accessibilityLabel("Apple system prompt")
                Text("Applies from the next message.")
                    .settingsNote()
                Button("Clear System Prompt") { model.modelSystemPrompt = "" }
                    .disabled(model.modelSystemPrompt.isEmpty)
            }
        }
        .formStyle(TUFFFormStyle())
    }

    /// The settings whose value decides which rows the page has.
    private var rowsKey: String {
        let profile = selectedProfile
        return "\(model.accentColorMode)\(model.bypassModelRestrictions)\(updateController.isAvailable)"
            + "\(profile.automaticMemory)\(profile.topKEnabled)\(profile.topPEnabled)"
            + "\(autoLearningMessage ?? "")\(selectedProfileInstall.id)"
    }

    private var generalSettings: some View {
        Form {
            Section("Chat") {
                GlassMenuPicker("Send message with", selection: $model.newlineShortcut,
                                options: AppNewlineShortcut.sendMessageOptions,
                                label: { $0.sendMessageLabel })
                GlassMenuPicker("After sending", selection: $model.sentPromptBehavior,
                                options: AppSentPromptBehavior.allCases,
                                label: { $0.settingsLabel })
                Toggle("Show prompt examples", isOn: $model.showPromptExamples)
                Toggle("Smooth text while it streams", isOn: $model.smoothStreaming)
                    .help("Shows an answer at a steady pace instead of in bursts. "
                        + "The answer and its speed are the same either way.")
            }
            Section("Appearance") {
                GlassChoiceRow(title: "Appearance", selection: $model.appearanceMode,
                               options: AppAppearanceMode.allCases, label: \.settingsLabel)
                LabeledContent("Transparency") {
                    HStack(spacing: 10) {
                        Slider(value: $model.windowTransparency, in: 0...1, step: 0.1)
                            .frame(minWidth: 80, maxWidth: 180)
                            .accessibilityLabel("Window transparency")
                        Text("\(Int((model.windowTransparency * 100).rounded()))%")
                            .monospacedDigit()
                            .frame(width: 42 * model.zoomLevel.scale, alignment: .trailing)
                    }
                }
                .help("Adjusts how much of the blurred desktop shows behind chat. "
                    + "Reduce Transparency in System Settings keeps it opaque.")
                GlassChoiceRow(title: "Zoom", selection: $model.zoomLevel,
                               options: AppZoomLevel.allCases, label: \.settingsLabel)
                    .help("Scales text, icons and controls. Also in the View menu: "
                    + "Zoom In (⌘+), Zoom Out (⌘−) and Actual Size (⌘0).")
                GlassChoiceRow(title: "Accent color", selection: $model.accentColorMode,
                               options: AppAccentColorMode.allCases, label: \.settingsLabel)
                if model.accentColorMode == .custom {
                    // One row: the hex value and a round swatch that opens the
                    // color panel.
                    LabeledContent("Custom color") {
                        HStack(spacing: TUFFDesign.Space.xs) {
                            TextField("6F4DFF", text: customAccentColorHexFieldBinding)
                                .tuffField()
                                .frame(width: 90)
                                .multilineTextAlignment(.trailing)
                                .monospaced()
                            ColorPicker("Custom color", selection: customAccentColorBinding,
                                        supportsOpacity: false)
                                .labelsHidden()
                                .frame(width: 28, height: 24)
                        }
                    }
                }
            }
            Section("Model restrictions") {
                Toggle("Bypass model restrictions",
                       isOn: $model.bypassModelRestrictions)
                Text("Allows models larger than this Mac's memory budget.")
                    .settingsNote()
                    .help("TUFF normally hides Download and Load for models this Mac "
                    + "does not meet the requirements for, and refuses to load "
                    + "any model whose context and expert cache exceed "
                    + "\(MetricFormat.memory(model.deviceCapabilities.safeAppMemoryBudgetBytes)), "
                    + "the share of this Mac's memory it plans against. Turning "
                    + "this on removes both gates.")
                if model.bypassModelRestrictions {
                    Label("Models that don't fit can swap heavily or be quit by macOS.",
                          systemImage: "exclamationmark.triangle.fill")
                        .settingsNote(tone: .warning)
                }
            }
            Section("Startup") {
                Toggle("Load the selected model at launch",
                       isOn: $model.loadModelOnLaunch)
                Text("Only if the model fits this Mac's memory budget.")
                    .settingsNote()
                    .help("Only when the model is installed and fits this Mac's "
                    + "memory budget, or restrictions are bypassed.")
            }
            Section("Updates") {
                if updateController.isAvailable {
                    Toggle(
                        "Check for updates automatically",
                        isOn: $updateController.automaticallyChecksForUpdates)
                    Toggle(
                        "Download and install updates automatically",
                        isOn: $updateController.automaticallyDownloadsUpdates)
                        .disabled(!updateController.automaticallyChecksForUpdates
                            || !updateController.allowsAutomaticUpdates)
                    Button("Check for Updates Now") {
                        updateController.checkForUpdates()
                    }
                    .disabled(!updateController.canCheckForUpdates)
                        .help("Updates come from TUFF's GitHub Releases feed. Each "
                        + "download must pass Sparkle's embedded EdDSA signature "
                        + "check before it can be installed.")
                } else {
                    Label(
                        updateController.unavailableReason
                            ?? "Updates are unavailable in this build.",
                        systemImage: "exclamationmark.shield")
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                    Text("Only packaged builds can install updates.")
                        .settingsNote()
                        .help("Clone builds stay usable, but only packaged builds with "
                        + "TUFF's update-signing public key can install releases.")
                }
            }
        }
        .formStyle(TUFFFormStyle())
        .scrollContentBackground(.hidden)
    }

    private func modelSettings(advanced: Bool) -> some View {
        Form {
            Section("Profile") {
                GlassMenuPicker("Model", selection: profileModelBinding,
                                options: model.installs.map(\.id),
                                label: { id in model.installs.first { $0.id == id }?.descriptor.shortName ?? "" })
                if selectedProfileInstall.id == model.selectedModelID,
                   model.hasStaleLoadedRuntime {
                    Label("Reload the model to apply memory changes.",
                          systemImage: "arrow.clockwise")
                        .settingsNote()
                }
            }
            if advanced {
                CustomCheckpointSettingsView(model: model)
                advancedProfileSettings
                InferenceEngineSettingsView()
            } else {
                systemPromptSettings
                memorySettings
                reasoningSettings
                samplingSettings
            }
        }
        .formStyle(TUFFFormStyle())
        .scrollContentBackground(.hidden)
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionCompanionOperationInProgress)
    }

    private var memorySettings: some View {
        Section("Memory") {
            Toggle("Auto", isOn: automaticMemoryBinding)
            if selectedProfile.automaticMemory {
                GlassChoiceRow(title: "Auto profile",
                               selection: profileBinding(\.automaticMemoryProfile),
                               options: AppAutomaticMemoryProfile.allCases,
                               label: { profileLabel($0) })
                Text(selectedProfile.automaticMemoryProfile.explanation)
                    .settingsNote()
                DisclosureGroup("Auto tuning") {
                    Button("Tune Auto on this Mac") { model.tuneAutomaticMemory() }
                        .disabled(selectedProfileInstall.id != model.selectedModelID || !model.canTuneAutomaticMemory)
                        .help("Runs local probes for up to 15 seconds. No chats are saved.")
                    Button("Reset Auto learning") {
                        do {
                            try AppAutoPerformanceRecords.reset()
                            model.updateSettingsProfile(selectedProfile, for: selectedProfileInstall)
                            autoLearningMessage = "Auto learning reset."
                        } catch { autoLearningMessage = String(describing: error) }
                    }
                    if let autoLearningMessage {
                        Text(autoLearningMessage).appFont(.caption).foregroundStyle(TUFFDesign.secondaryTextColor)
                    }
                }
            }
            Group {
                GlassMenuPicker("Context", selection: profileBinding(\.contextTokens),
                                options: AppContextLengthOption.options(for: selectedProfileInstall.descriptor).map(\.tokens),
                                label: { tokens in
                                    guard let option = AppContextLengthOption.options(for: selectedProfileInstall.descriptor)
                                        .first(where: { $0.tokens == tokens }) else { return MetricFormat.tokens(tokens) }
                                    return contextLabel(option, eligibility: contextEligibility(
                                        tokens: tokens, slots: selectedProfile.expertCacheSlots))
                                },
                                isEnabled: { tokens in
                                    contextEligibility(tokens: tokens, slots: selectedProfile.expertCacheSlots).isCompatible
                                })
                if selectedProfileInstall.descriptor.usesExpertCache {
                    GlassMenuPicker("Expert cache", selection: expertCacheSlotsBinding,
                                    options: selectedProfileInstall.descriptor.usefulExpertCacheSlotCounts,
                                    label: { slots in
                                        selectedProfile.automaticMemory ? "\(slots) slots"
                                            : "\(slots) slots (" + MetricFormat.memory(contextEligibility(
                                                tokens: selectedProfile.contextTokens, slots: slots).estimatedWorkingSetBytes) + " total)"
                                    },
                                    isEnabled: { slots in
                                        contextEligibility(tokens: selectedProfile.contextTokens, slots: slots).isCompatible
                                    })
                } else {
                    LabeledContent("Expert cache", value: "Not used by this dense model")
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                }
            }
            .disabled(selectedProfile.automaticMemory)
            let eligibility = contextEligibility(
                tokens: selectedProfile.contextTokens,
                slots: selectedProfile.expertCacheSlots)
            let automaticPlan = selectedProfile.automaticMemory
                ? model.automaticMemoryPlan(for: selectedProfileInstall) : nil
            LabeledContent("Calculated allocation") {
                Text(MetricFormat.memory(automaticPlan?.estimatedWorkingSetBytes ?? eligibility.estimatedWorkingSetBytes))
                    .monospacedDigit()
            }
            if automaticPlan == nil, let explanation = eligibility.explanation {
                Label(explanation, systemImage: "exclamationmark.triangle.fill")
                    .settingsNote(tone: .warning)
                Text(model.bypassModelRestrictions
                     ? "Bypass model restrictions is on, so TUFF will still load "
                        + "this. If the Mac runs out of memory the app may be "
                        + "terminated by the system."
                     : "TUFF will not load the model at these settings. Lower "
                        + "Context or Expert cache, switch this model to Auto, "
                        + "or turn on Bypass model restrictions in General.")
                    .settingsNote()
            }
        }
    }

    /// This model's system prompt. Stored with the rest of its profile, so
    /// each model carries its own.
    private var systemPromptSettings: some View {
        Section("System prompt") {
            // A padded, rounded well like the message box, not a bare black
            // text view flush with the card.
            TextEditor(text: profileBinding(\.systemPrompt))
                .appFont(.body)
                .scrollContentBackground(.hidden)
                .padding(.horizontal, 7)
                .padding(.vertical, 8)
                .frame(minHeight: 96)
                .background(TUFFDesign.insetColor,
                            in: RoundedRectangle(cornerRadius: TUFFDesign.Radius.control, style: .continuous))
                .accessibilityLabel("System prompt")
            HStack {
                Text("Applies from the next message.")
                    .appFont(.caption)
                    .foregroundStyle(TUFFDesign.secondaryTextColor)
                    .help("Sent as a system message before the conversation. It applies "
                        + "from the next message; answers already given were not "
                        + "written with it.")
                Spacer()
                Button("Restore Default") {
                    var profile = selectedProfile
                    profile.systemPrompt = selectedProfileInstall.descriptor.defaultSystemPrompt
                    model.updateSettingsProfile(profile, for: selectedProfileInstall)
                }
                .disabled(selectedProfile.systemPrompt
                    == selectedProfileInstall.descriptor.defaultSystemPrompt)
                .help("Restore TUFF's default prompt for this model")
                .controlSize(.small)
            }
        }
    }

    @ViewBuilder
    private var reasoningSettings: some View {
        if let control = selectedProfileInstall.descriptor.reasoningControl {
            Section("Reasoning default") {
                switch control {
                case .toggle, .toggleWithPreservation:
                    GlassChoiceRow(title: "Thinking", selection: profileBinding(\.defaultReasoning),
                                   options: [ChatReasoning.off, .on],
                                   label: { $0 == .on ? "On" : "Off" })
                case .graded:
                    GlassChoiceRow(title: "Effort", selection: profileBinding(\.defaultReasoningEffort),
                                   options: GPTOSSReasoningEffort.allCases,
                                   label: { $0.rawValue.capitalized })
                case .alwaysOn:
                    LabeledContent("Thinking", value: "Always On")
                        .help("This model always reasons before answering.")
                }
                if control != .alwaysOn {
                    Text("Chat can override this for the current conversation.")
                        .settingsNote()
                }
            }
        }
    }

    private var samplingSettings: some View {
        Section("Sampling") {
            LabeledContent("Temperature") {
                HStack(spacing: 10) {
                    Slider(value: profileBinding(\.temperature), in: 0...2, step: 0.05)
                    Text(selectedProfile.temperature,
                         format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 38, alignment: .trailing)
                }
            }
            Toggle("Top-K", isOn: profileBinding(\.topKEnabled))
            if selectedProfile.topKEnabled {
                Stepper(value: profileBinding(\.topK), in: 1...256) {
                    LabeledContent("K value", value: "\(selectedProfile.topK)")
                }
                Toggle("Top-P", isOn: profileBinding(\.topPEnabled))
                if selectedProfile.topPEnabled {
                    LabeledContent("P value") {
                        HStack(spacing: 10) {
                            Slider(value: profileBinding(\.topP),
                                   in: 0.01...1, step: 0.01)
                            Text(selectedProfile.topP,
                                 format: .number.precision(.fractionLength(2)))
                                .monospacedDigit()
                                .frame(width: 38, alignment: .trailing)
                        }
                    }
                }
            }
        }
    }

    private var advancedProfileSettings: some View {
        Group {
            Section("Runtime") {
                if selectedProfileInstall.descriptor.projectionLayerCount > 0 {
                    GlassMenuPicker("Projection residency", selection: projectionResidencyBinding,
                                    options: ["resident", "partial", "streamed"],
                                    label: { $0.capitalized }).disabled(selectedProfile.automaticMemory)
                    if let count = selectedProfile.denseResidentLayerCount, count > 0 {
                        Stepper(value: projectionLayerBinding,
                                in: 1...max(1, selectedProfileInstall.descriptor.projectionLayerCount - 1)) {
                            Text("\(count) projection layers resident")
                        }.disabled(selectedProfile.automaticMemory)
                    }
                    Text("Resident keeps projection weights in RAM. Streamed reads each layer as needed and may be very slow. Auto manages this choice while preserving your manual setting.")
                        .appFont(.caption).foregroundStyle(TUFFDesign.secondaryTextColor)
                }
                Toggle("Chunked prefill", isOn: profileBinding(\.prefillEnabled))
                    .disabled(selectedProfile.automaticMemory
                        || selectedProfile.expertCacheSlots
                        < RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill)
                if selectedProfile.automaticMemory {
                    Text("Auto manages chunked prefill with this model's memory plan.")
                        .settingsNote()
                } else if selectedProfile.expertCacheSlots
                    < RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill {
                    Text("Chunked prefill needs at least "
                        + "\(RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill) "
                        + "expert-cache slots.")
                        .settingsNote()
                }
                GlassMenuPicker("RDADVISE", selection: profileBinding(\.rdadvisePolicy),
                                options: AppRDAdvicePolicy.allCases,
                                label: { $0.label })
                Text("Experimental. Applies after a reload.")
                    .settingsNote()
                    .help("RDADVISE is experimental. Changes to memory mapping apply "
                    + "after the selected model is reloaded.")
            }
            if selectedProfileInstall.descriptor.reasoningControl
                == .toggleWithPreservation {
                Section("Thinking history") {
                    Toggle("Preserve thinking in later prompts",
                           isOn: profileBinding(\.preserveThinking))
                    Text("Sends earlier thinking back to the model.")
                        .settingsNote()
                        .help("When enabled, Qwen receives its earlier thinking text as "
                        + "part of conversation history. TUFF always keeps that text "
                        + "available for you in the transcript.")
                }
            }
            Section("Location") {
                Text(selectedProfileInstall.directoryURL.path)
                    .appFont(.caption.monospaced())
                    .textSelection(.enabled)
            }
            if selectedProfileInstall.id == model.selectedModelID,
               let diagnostics = model.diagnostics {
                RunnerDiagnosticsSection(diagnostics: diagnostics)
            }
        }
    }

    private var selectedProfileInstall: ModelInstallCoordinator {
        model.installs.first { $0.id == profileModelID } ?? model.selectedInstall
    }

    private var selectedProfile: AppModelSettingsProfile {
        model.settingsProfile(for: selectedProfileInstall)
    }

    /// An unset picker follows the app's selected model.
    private var profileModelBinding: Binding<String> {
        Binding {
            profileModelID.isEmpty ? model.selectedModelID : profileModelID
        } set: { profileModelID = $0 }
    }

    /// Writes a field through profile validation and persistence.
    private func profileBinding<Value>(
        _ keyPath: WritableKeyPath<AppModelSettingsProfile, Value>
    ) -> Binding<Value> {
        Binding {
            selectedProfile[keyPath: keyPath]
        } set: { value in
            var profile = selectedProfile
            profile[keyPath: keyPath] = value
            model.updateSettingsProfile(profile, for: selectedProfileInstall)
        }
    }

    /// Each profile is labelled with the context it would actually resolve on
    /// this Mac, so the choice is between concrete outcomes rather than words.
    private func profileLabel(_ profile: AppAutomaticMemoryProfile) -> String {
        guard let plan = model.automaticMemoryPlan(
            for: selectedProfileInstall, profile: profile) else {
            return profile.title
        }
        let context = AppContextLengthOption(rawValue: plan.contextTokens)?.shortLabel
            ?? "\(plan.contextTokens)"
        return "\(profile.title) · \(context)"
    }

    /// The model preserves manual memory values while Auto is enabled.
    private var automaticMemoryBinding: Binding<Bool> {
        Binding {
            selectedProfile.automaticMemory
        } set: { enabled in
            model.setAutomaticMemory(enabled, for: selectedProfileInstall)
        }
    }

    /// Keep prefill and cache-slot invariants in one edit.
    private var expertCacheSlotsBinding: Binding<Int> {
        Binding {
            selectedProfile.expertCacheSlots
        } set: { slots in
            var profile = selectedProfile
            profile.expertCacheSlots = slots
            if slots < RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill {
                profile.prefillEnabled = false
            }
            model.updateSettingsProfile(profile, for: selectedProfileInstall)
        }
    }

    private func contextEligibility(
        tokens: Int,
        slots: Int
    ) -> AppModelContextEligibility {
        model.contextEligibility(
            for: selectedProfileInstall,
            contextTokens: tokens,
            expertCacheSlots: slots)
    }

    private func contextLabel(
        _ option: AppContextLengthOption,
        eligibility: AppModelContextEligibility
    ) -> String {
        if selectedProfile.automaticMemory { return option.shortLabel }
        // The total with this option, not the option's own share: the
        // selected row always equals the working set below, which read as
        // three settings that each cost the same.
        return option.shortLabel + " (" + MetricFormat.memory(eligibility.estimatedWorkingSetBytes)
            + " total)"
    }


    private var projectionResidencyBinding: Binding<String> {
        Binding(get: {
            guard let count = selectedProfile.denseResidentLayerCount else { return "resident" }
            return count == 0 ? "streamed" : "partial"
        }, set: { mode in
            var profile = selectedProfile
            profile.denseResidentLayerCount = mode == "resident" ? nil
                : (mode == "streamed" ? 0 : max(1, selectedProfileInstall.descriptor.projectionLayerCount / 2))
            model.updateSettingsProfile(profile, for: selectedProfileInstall)
        })
    }
    private var projectionLayerBinding: Binding<Int> {
        Binding(get: { selectedProfile.denseResidentLayerCount ?? 1 }, set: { count in
            var profile = selectedProfile
            profile.denseResidentLayerCount = count
            model.updateSettingsProfile(profile, for: selectedProfileInstall)
        })
    }

    private var customAccentColorBinding: Binding<Color> {
        Binding {
            let hex = AppHexColor(hexString: model.customAccentColorHex) ?? .defaultPurple
            return Color(red: hex.red, green: hex.green, blue: hex.blue)
        } set: { newColor in
            guard let hex = Self.hexColor(from: newColor) else { return }
            pendingAccentHex = nil
            model.setCustomAccentColorHex(hex.hexString)
        }
    }

    private var customAccentColorHexFieldBinding: Binding<String> {
        Binding {
            pendingAccentHex
                ?? model.customAccentColorHex.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
        } set: { newValue in
            pendingAccentHex = newValue
            let candidate = "#" + newValue.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
            guard AppHexColor(hexString: candidate) != nil else { return }
            model.setCustomAccentColorHex(candidate)
            pendingAccentHex = nil
        }
    }

    private static func hexColor(from color: Color) -> AppHexColor? {
        guard let rgb = NSColor(color).usingColorSpace(.deviceRGB) else { return nil }
        return AppHexColor(
            red: rgb.redComponent, green: rgb.greenComponent, blue: rgb.blueComponent)
    }
}

/// Options on a capsule track with one Liquid Glass capsule under the chosen
/// one. The capsule is a single shape that moves and resizes, so it glides
/// across any distance. Hovering brightens it, or shows a faint wash under an
/// option that is not chosen. Used for the Settings tabs (with icons) and for
/// every segmented choice in Settings (compact, text only). In a narrow
/// window, tabs with icons drop the labels of the unchosen ones.
struct TUFFTabBar<Tab: Hashable>: View {
    let tabs: [Tab]
    @Binding var selection: Tab
    let title: (Tab) -> String
    var systemImage: ((Tab) -> String)? = nil
    var compact = false
    /// Off where the bar sits inside another glass shape (the thinking
    /// control), which already is the track.
    var showsTrack = true
    /// Each tab's frame, per layout (with and without labels), since the
    /// window can switch between them.
    @State private var frames: [FrameKey: CGRect] = [:]
    @State private var hovered: Tab?
    /// The option under the pointer while the glass is being dragged; it is
    /// chosen when the drag ends.
    @State private var dragged: Tab?
    /// The pointer's x while dragging; the glass follows it directly.
    @State private var dragX: CGFloat?
    @Namespace private var pillSpace
    /// Just after a drag: the dragged glass glides into the chosen slot
    /// before the tab's own glass takes over, so it does not pop.
    @State private var settling = false

    /// The glass's frame: the chosen option's, or while dragging, the
    /// width of the option under the pointer centred on the pointer,
    /// kept inside the bar.
    private func pillFrame(labels: Bool) -> CGRect? {
        guard let base = frames[FrameKey(labels: labels, tab: dragged ?? selection)] else { return nil }
        guard let x = dragX else { return base }
        let all = tabs.compactMap { frames[FrameKey(labels: labels, tab: $0)] }
        let lower = all.map(\.minX).min() ?? base.minX
        let upper = (all.map(\.maxX).max() ?? base.maxX) - base.width
        var frame = base
        frame.origin.x = min(max(x - base.width / 2, lower), upper)
        return frame
    }

    private struct FrameKey: Hashable {
        let labels: Bool
        let tab: Tab
    }

    var body: some View {
        if systemImage == nil {
            bar(labels: true)
        } else {
            ViewThatFits(in: .horizontal) {
                bar(labels: true)
                bar(labels: false)
            }
        }
    }

    private func bar(labels: Bool) -> some View {
        HStack(spacing: 2) {
            ForEach(tabs, id: \.self) { tab in
                let isSelected = tab == selection
                Button {
                    selection = tab
                } label: {
                    Group {
                        if let systemImage {
                            if labels || isSelected {
                                Label(title(tab), systemImage: systemImage(tab))
                            } else {
                                Image(systemName: systemImage(tab))
                            }
                        } else {
                            Text(title(tab))
                        }
                    }
                    .appFont(.callout.weight(isSelected ? .semibold : .medium))
                    .foregroundStyle((dragged ?? selection) == tab ? AnyShapeStyle(.primary)
                                                : AnyShapeStyle(TUFFDesign.secondaryTextColor))
                    .lineLimit(1)
                    .fixedSize()
                    .padding(.horizontal, compact ? 12 : 14)
                    .frame(minHeight: compact ? 26 : 30)
                    .background {
                        if isSelected && dragX == nil && !settling {
                            // Laid out with the tab itself, so the glass moves
                            // in the same pass as the tabs (when the whole bar
                            // moves with the sidebar, say), and slides to the
                            // next tab as one matched shape.
                            TUFFGlassPill(isHovered: hovered == tab)
                                .matchedGeometryEffect(id: labels, in: pillSpace)
                        } else if !isSelected && hovered == tab {
                            Capsule().fill(Color.primary.opacity(0.06))
                        }
                    }
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .onHover { inside in
                    if inside { hovered = tab } else if hovered == tab { hovered = nil }
                }
                .help(title(tab))
                .accessibilityLabel(title(tab))
                .accessibilityAddTraits(isSelected ? .isSelected : [])
                .onGeometryChange(for: CGRect.self) { $0.frame(in: .named(TabSpace.name)) } action: {
                    frames[FrameKey(labels: labels, tab: tab)] = $0
                }
            }
        }
        .padding(compact ? 3 : 4)
        // While dragging, the glass follows the pointer, placed from the
        // measured tab frames.
        .background(alignment: .topLeading) {
            if dragX != nil || settling, let frame = pillFrame(labels: labels) {
                TUFFGlassPill(isHovered: true)
                    .frame(width: frame.width, height: frame.height)
                    .offset(x: frame.minX, y: frame.minY)
            }
        }
        // Drag the glass across the options: it follows the pointer, and
        // the option it is let go on is chosen.
        .simultaneousGesture(
            DragGesture(minimumDistance: 4, coordinateSpace: .named(TabSpace.name))
                .onChanged { value in
                    let x = value.location.x
                    var follow = Transaction()
                    follow.animation = .interactiveSpring(response: 0.12, dampingFraction: 0.9)
                    withTransaction(follow) {
                        dragX = x
                        if let tab = tabs.first(where: { tab in
                            guard let frame = frames[FrameKey(labels: labels, tab: tab)] else { return false }
                            return x >= frame.minX - 1 && x <= frame.maxX + 1
                        }) {
                            dragged = tab
                        }
                    }
                }
                .onEnded { _ in
                    settling = true
                    withAnimation(TUFFMotion.standard) {
                        if let dragged { selection = dragged }
                        dragged = nil
                        dragX = nil
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                        var still = Transaction()
                        still.disablesAnimations = true
                        withTransaction(still) { settling = false }
                    }
                })
        .coordinateSpace(name: TabSpace.name)
        .background(showsTrack ? Color.primary.opacity(0.06) : .clear, in: Capsule())
        .animation(dragX == nil ? TUFFMotion.standard : nil, value: selection)
        .animation(TUFFMotion.hover, value: hovered)
        .accessibilityElement(children: .contain)
    }
}

private enum TabSpace { static let name = "TUFFTabBar" }

/// A labeled settings row whose choice is a glass segmented control.
struct GlassChoiceRow<Value: Hashable>: View {
    let title: String
    @Binding var selection: Value
    let options: [Value]
    let label: (Value) -> String

    var body: some View {
        LabeledContent(title) {
            TUFFTabBar(tabs: options, selection: $selection, title: label, compact: true)
                .fixedSize()
        }
    }
}

/// A dropdown as a glass capsule: the current choice and a small chevron,
/// brightening on hover, opening a menu of plain buttons with a checkmark on
/// the chosen one. (A `Picker` nested in the menu drew misaligned checkmarks
/// and an empty checked row.)
struct GlassMenuPicker<Value: Hashable>: View {
    let title: String
    @Binding var selection: Value
    let options: [Value]
    let label: (Value) -> String
    var isEnabled: ((Value) -> Bool)? = nil
    @State private var isHovered = false
    @Environment(\.isEnabled) private var controlEnabled

    init(_ title: String, selection: Binding<Value>, options: [Value],
         label: @escaping (Value) -> String, isEnabled: ((Value) -> Bool)? = nil) {
        self.title = title
        _selection = selection
        self.options = options
        self.label = label
        self.isEnabled = isEnabled
    }

    var body: some View {
        LabeledContent(title) {
            Menu {
                ForEach(options, id: \.self) { option in
                    Button {
                        selection = option
                    } label: {
                        if option == selection {
                            Label(label(option), systemImage: "checkmark")
                        } else {
                            Text(label(option))
                        }
                    }
                    .disabled(!(isEnabled?(option) ?? true))
                }
            } label: {
                HStack(spacing: 7) {
                    Text(label(selection))
                        .lineLimit(1)
                    Image(systemName: "chevron.up.chevron.down")
                        .appFont(.caption2.weight(.bold))
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                }
                .appFont(.callout.weight(.medium))
                .foregroundStyle(controlEnabled ? AnyShapeStyle(.primary)
                                                : AnyShapeStyle(TUFFDesign.tertiaryTextColor))
                .padding(.leading, 14)
                .padding(.trailing, 11)
                .frame(minHeight: 30)
                .tuffGlass(Capsule(), highlighted: isHovered && controlEnabled)
                .contentShape(Capsule())
                .scaleEffect(isHovered && controlEnabled ? 1.03 : 1)
                .animation(TUFFMotion.hover, value: isHovered)
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
            .fixedSize()
            .onHover { isHovered = $0 }
            .accessibilityLabel(title)
            .accessibilityValue(label(selection))
        }
    }
}
