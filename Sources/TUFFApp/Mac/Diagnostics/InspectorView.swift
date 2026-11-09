import AppKit
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel

    var body: some View {
        Form {
            modelSection
            catalogSection
            // Second, beside Model: image support is an install concern, not a
            // diagnostic. Last put it under the runner diagnostics and below the
            // fold, where the one screen that must mention it - the empty state
            // before any model exists - could not.
            if showsVisionSection {
                visionSection
            }
            memorySection
            generationSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .tuffPageBackground()
    }

    /// Shown while it has something to say, hidden once it does not. Unlike
    /// root this does not require an installed text model: the empty state is
    /// exactly where someone decides whether this app does what they need, and
    /// hiding image support until after a 14.62 GB download meant nobody found
    /// out it existed. It still collapses once the pack is installed and healthy.
     private var showsVisionSection: Bool {
        guard model.selectedModelOwnsVisionInstallState else { return false }
        return VisionSectionVisibility.shows(
            visionRuntimeEnabled: model.visionRuntimeEnabled,
            visionRuntimeSupported: model.isVisionRuntimeSupported,
            isModelInstalled: model.isModelInstalled,
            isVisionPackInstalled: model.isVisionPackInstalled,
            isCompanionOperationInProgress: model.isVisionCompanionOperationInProgress,
            installState: model.visionInstallState)
    }

    private var visionSection: some View {
        Section("Image Support") {
            LabeledContent("State") {
                Text(visionStatusLabel)
                    .appFont(.caption)
                    .foregroundStyle(visionStatusColor)
            }
            if model.isVisionRuntimeSupported, !model.isVisionPackInstalled,
               let descriptor = model.visionInstallDescriptor {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(descriptor.approximateDownloadBytes))
                        .appFont(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if let fraction = model.visionInstallProgressFraction {
                ProgressView(value: fraction)
                    .accessibilityValue(visionAccessibleProgress(fraction: fraction))
                HStack(alignment: .firstTextBaseline) {
                    Text(MetricFormat.percent(fraction * 100))
                    Spacer(minLength: 8)
                    if let eta = model.visionInstallETAText {
                        Text(eta)
                    }
                }
                .appFont(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            } else if model.isInstallingVisionPack {
                ProgressView()
                    .controlSize(.small)
                if let eta = model.visionInstallETAText {
                    Text(eta)
                        .appFont(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if !model.isVisionRuntimeSupported {
                Text("Image support requires an M2 or newer Mac. "
                    + "Text generation remains available on this Mac.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            } else if case .failed(let message) = model.visionInstallState {
                Text(message)
                    .appFont(.caption)
                    .foregroundStyle(TUFFMacTheme.accentTextColor)
            } else if case .recoverable(let message) = model.visionInstallState {
                Text(message)
                    .appFont(.caption)
                    .foregroundStyle(TUFFMacTheme.accentTextColor)
            } else if case .partial(let message) = model.visionInstallationStatus,
                      !model.isInstallingVisionPack {
                Text(message)
                    .appFont(.caption)
                    .foregroundStyle(TUFFMacTheme.accentTextColor)
            } else if case .unsupportedLayout = model.visionInstallationStatus {
                Text("Image support needs a model directory named "
                    + "\"<name>.gturbo\", which is where the companion pack lives.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            } else if case .failed(let message) = model.visionInstallReadiness,
                      !model.isInstallingVisionPack {
                Text(message)
                    .appFont(.caption)
                    .foregroundStyle(TUFFMacTheme.accentTextColor)
            } else if case .insufficientSpace(let requirement) = model.visionInstallReadiness,
                      !model.isInstallingVisionPack {
                Text("Free \(MetricFormat.storage(requirement.shortfallBytes)) more storage.")
                    .appFont(.caption)
                    .foregroundStyle(TUFFMacTheme.accentTextColor)
            } else if model.isVisionCompanionOperationInProgress {
                Text("Model actions stay unavailable until this finishes. "
                    + "Your prompt, images, and transcript are kept.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            } else if case .readyToActivate = model.visionInstallState,
                      model.loadState.isReady {
                Text("The separate image-support download is ready. "
                    + "Unload the model once to activate it.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            } else if !model.isVisionPackInstalled && model.loadState.isReady {
                Text("Image support downloads separately while the text model "
                    + "stays installed. Unload only when the download is ready to activate.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            } else if model.isVisionPackInstalled && model.loadState.isReady {
                Text("Unload the model before removing image support.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            }

            HStack {
                if model.isInstallingVisionPack {
                    Button("Cancel", action: model.cancelVisionInstall)
                        .disabled(!model.canCancelVisionInstall)
                } else if case .readyToActivate = model.visionInstallState {
                    Button("Discard", role: .destructive) {
                        model.discardVisionPackDownload()
                    }
                    .disabled(!model.canDiscardVisionPackDownload)
                    if model.isVisionRuntimeSupported {
                        Button("Activate", action: model.activateVisionPack)
                            .buttonStyle(.tuffProminent)
                            .disabled(!model.canActivateVisionPack)
                    }
                } else if model.isVisionPackInstalled {
                    Button("Remove", role: .destructive) {
                        model.requestVisionPackRemoval()
                    }
                    .disabled(!model.canRemoveVisionPack)
                } else {
                    if model.hasVisionPackDirectory {
                        Button("Remove", role: .destructive) {
                            model.requestVisionPackRemoval()
                        }
                        .disabled(!model.canRemoveVisionPack)
                    }
                    if model.hasPartialVisionPackDownload {
                        Button("Discard", role: .destructive) {
                            model.discardVisionPackDownload()
                        }
                        .disabled(!model.canDiscardVisionPackDownload)
                    }
                    if model.isVisionRuntimeSupported {
                        Button(visionInstallButtonLabel) {
                            model.installVisionPack()
                        }
                        .buttonStyle(.tuffProminent)
                        .disabled(!model.canInstallVisionPack)
                    }
                }
            }
        }
    }

    private var visionInstallButtonLabel: String {
        let name = model.visionInstallDescriptor?.displayName ?? "Image Support"
        if model.hasPartialVisionPackDownload { return "Resume \(name)" }
        if model.hasVisionPackDirectory { return "Repair \(name)" }
        return "Download \(name)"
    }

    private func visionAccessibleProgress(fraction: Double) -> String {
        let percent = MetricFormat.percent(fraction * 100)
        guard let eta = model.visionInstallETAText else { return percent }
        return "\(percent), \(eta)"
    }

    private var visionStatusLabel: String {
        guard model.isVisionRuntimeSupported else { return "Requires M2 or newer" }
        if model.visionInstallState != .idle {
            return model.visionInstallPhaseLabel
        }
        switch model.visionInstallationStatus {
        case .missing: return "Not installed"
        case .partial: return "Needs repair"
        case .complete: return "Installed"
        case .unsupportedLayout: return "Not available for this model"
        }
    }

    private var visionStatusColor: Color {
        guard model.isVisionRuntimeSupported else { return .secondary }
        switch model.visionInstallationStatus {
        case .partial: return TUFFMacTheme.accentTextColor
        case .missing, .complete, .unsupportedLayout: return .secondary
        }
    }

    private var modelSection: some View {
        Section("Model") {
            LabeledContent("Path") {
                HStack(spacing: 6) {
                    Text(model.modelPathText)
                        .appFont(.caption)
                        .truncationMode(.middle)
                        .lineLimit(1)
                        .foregroundStyle(.secondary)
                        .help(model.modelPathText)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.modelPathText, forType: .string)
                    } label: {
                        Label("Copy model path", systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Copy model path")
                }
            }
            if model.canUnloadModel {
                Button("Unload Model", action: model.unloadModel)
            }
            LabeledContent("State") {
                Text(model.presentation.label)
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.requiresModelInstallation {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(model.installDescriptor.approximateDownloadBytes))
                        .appFont(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Installed size") {
                    Text(MetricFormat.storage(model.installDescriptor.installedBytes))
                        .appFont(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let requirement = model.installRequirement {
                    LabeledContent("Available") {
                        Text(MetricFormat.storage(requirement.availableBytes))
                            .appFont(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel
            || model.isVisionCompanionOperationInProgress)
    }

    /// Not disabled while a download runs: switching models and starting a
    /// second download are exactly the things this section exists to allow
    /// mid-transfer. The individual controls guard themselves.
    private var catalogSection: some View {
        Section("Models") {
            ModelCatalogView(model: model, compact: true)
                .listRowInsets(EdgeInsets())
            Text("Downloads are per model and keep running while you use "
                 + "another one.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var memorySection: some View {
        Section("Memory") {
            LabeledContent("Context") {
                Picker("Context", selection: $model.maxContextTokens) {
                    ForEach(AppContextLengthOption.options(for: model.installDescriptor)) { option in
                        Text(option.shortLabel).tag(option.tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            LabeledContent("Slots") {
                Picker("Slots", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            Text("Longer context and more expert slots use additional memory. The cost depends on the model. Changes apply after reloading the model.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionCompanionOperationInProgress)
    }

    private var generationSection: some View {
        Section("Generation") {
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Text("0 uses deterministic greedy decoding. Higher values make sampling more varied.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
            if model.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)").monospacedDigit()
                    }
                    .fixedSize()
                }
            }
            Toggle("Top-P", isOn: $model.topPEnabled)
                .toggleStyle(.switch)
                .disabled(!model.topKEnabled)
            if model.topKEnabled && model.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                        Text(model.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionCompanionOperationInProgress)
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            Toggle("Prefill", isOn: $model.runtimeOptions.prefillEnabled)
            VStack(alignment: .leading, spacing: 8) {
                Text("RDADVISE")
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            Text("RDADVISE is experimental. It may speed up short decodes but slow down long decodes.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionCompanionOperationInProgress)
    }

}
