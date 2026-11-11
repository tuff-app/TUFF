import TUFFAppCore
import TUFFMacPresentation
import SwiftUI
import AppKit

/// The model catalog: every supported model, each with its own download.
///
/// A row's download controls act on that model alone, so a download can run
/// for a model the app is not currently using — including while another model
/// is loaded and generating. Selection is a separate action from downloading:
/// picking a model decides which one loads, and never starts or stops a
/// transfer.
struct ModelCatalogView: View {
    let model: AppModel
    /// Compact drops the per-model summary and storage detail for the
    /// inspector, where the row sits in a narrow column.
    var compact: Bool = false
    var showsAddOns: Bool = true
    @State private var isAppleExpanded = true
    @State private var expandedGroups: Set<ModelCatalogGroup> = [.streaming, .resident]

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 8 : 14) {
            DisclosureGroup(isExpanded: $isAppleExpanded) {
                VStack(alignment: .leading, spacing: 12) {
                    if model.availableAppleModels.isEmpty {
                        Text("No Apple Models Found On This Mac").appFont(.body.weight(.medium))
                        Text(model.appleAvailability.label)
                            .appFont(.caption).foregroundStyle(TUFFDesign.secondaryTextColor)
                        if model.appleAvailability.action == .openSystemSettings {
                            Button(model.appleAvailability == .updateRequired ? "Open Software Update" : "Set Up Apple Intelligence") {
                                let pane = model.appleAvailability == .updateRequired ? "com.apple.Software-Update-Settings.extension" : "com.apple.Siri-Settings.extension"
                                if let url = URL(string: "x-apple.systempreferences:" + pane) { NSWorkspace.shared.open(url) }
                            }
                        }
                        Button("Check Again") { model.refreshAppleAvailability() }
                    } else {
                        ForEach(model.availableAppleModels) { kind in
                            VStack(alignment: .leading, spacing: 8) {
                                Text(kind == .onDevice ? kind.name : "Private Cloud Compute").appFont(.body.weight(.semibold))
                                Text(kind == .onDevice ? "Runs on this Mac." : "Runs on Apple's Private Cloud Compute. Requires a network connection.")
                                    .appFont(.caption).foregroundStyle(TUFFDesign.secondaryTextColor)
                                Button(model.selectedAppleModel == kind ? "Selected" : "Use Model") { model.selectAppleModel(kind) }
                                    .disabled(model.selectedAppleModel == kind || model.isRunning || model.loadState.isLoading)
                            }
                        }
                    }
                }.tuffCard(padding: TUFFDesign.Radius.cardPadding)
                .padding(.top, 8)
            } label: {
                Text("Apple").appFont(.headline).accessibilityAddTraits(.isHeader)
            }
            // Legacy models stay available in their own catalog section.
            ForEach(ModelCatalogGroup.allCases) { group in
                let installs = model.installs.filter {
                    group.contains($0.descriptor)
                }
                if !installs.isEmpty {
                    DisclosureGroup(isExpanded: Binding(
                        get: { expandedGroups.contains(group) },
                        set: { expanded in
                            if expanded { expandedGroups.insert(group) }
                            else { expandedGroups.remove(group) }
                        }
                    )) {
                        VStack(alignment: .leading, spacing: compact ? 8 : 12) {
                            ForEach(installs) { install in
                                ModelCatalogRow(model: model, install: install,
                                    compact: compact, showsAddOns: showsAddOns)
                            }
                        }
                        .padding(.top, 8)
                    } label: {
                        groupHeader(group)
                    }
                }
            }
        }
    }

    private func groupHeader(_ group: ModelCatalogGroup) -> some View {
        Text(group.title)
            .appFont(.headline)
            .accessibilityAddTraits(.isHeader)
    }
}

/// Maintenance status takes precedence over the maintained-model layout groups.
enum ModelCatalogGroup: String, CaseIterable, Identifiable {
    case streaming
    case resident
    case legacy

    var id: String { rawValue }

    func contains(_ descriptor: AppModelInstallDescriptor) -> Bool {
        switch self {
        case .streaming: !descriptor.isLegacy && descriptor.usesExpertCache
        case .resident: !descriptor.isLegacy && !descriptor.usesExpertCache
        case .legacy: descriptor.isLegacy
        }
    }

    var title: String {
        switch self {
        case .streaming: "MoE"
        case .resident: "Dense"
        case .legacy: "Legacy"
        }
    }
}

struct ModelCatalogRow: View {
    let model: AppModel
    let install: ModelInstallCoordinator
    var compact: Bool
    var showsAddOns: Bool

    @State private var showingDiscardConfirmation = false
    @State private var showingVisionDiscardConfirmation = false
    @State private var showingVisionRemoveConfirmation = false
    @State private var showingDeleteConfirmation = false

    private var isSelected: Bool { install.id == model.selectedModelID }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            if !compact {
                storage
            }
            if install.isInstalling {
                progress
                    .transition(.opacity)
            }
            message
            actions
            if showsAddOns && install.descriptor.supportsImageInput {
                visionSupport
            }
        }
        // A download starting, pausing or finishing: the progress bar and
        // the buttons for that stage fade across, and the card resizes
        // around them, instead of swapping in one frame.
        .animation(TUFFMotion.standard, value: stage)
        .tuffCard(padding: compact ? TUFFDesign.Space.sm : TUFFDesign.Radius.cardPadding)
        .overlay {
            if isSelected {
                RoundedRectangle(cornerRadius: TUFFDesign.Radius.card, style: .continuous)
                    .strokeBorder(TUFFMacTheme.accentTextColor, lineWidth: 1.5)
                    .transition(.opacity)
            }
        }
        .animation(TUFFMotion.standard, value: isSelected)
        .accessibilityElement(children: .contain)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
        .tuffConfirm("Discard the saved \(install.descriptor.displayName) download?",
                     isPresented: $showingDiscardConfirmation,
                     message: "Downloaded ranges for this model will be removed. "
                        + "Other models and any installed model are untouched.",
                     symbol: "arrow.down.circle", confirm: "Discard Download",
                     cancel: "Keep Download") { install.discard() }
        .tuffConfirm("Discard the saved \(visionDisplayName) download?",
                     isPresented: $showingVisionDiscardConfirmation,
                     message: "Downloaded add-on ranges will be removed. The text model is untouched.",
                     symbol: "arrow.down.circle", confirm: "Discard Download",
                     cancel: "Keep Download") { model.discardVisionPackDownload(for: install) }
        .tuffConfirm("Remove \(visionDisplayName)?",
                     isPresented: $showingVisionRemoveConfirmation,
                     message: "Text generation will keep working. Restoring image input "
                        + "requires downloading this add-on again.",
                     symbol: "photo", confirm: "Remove Image Support",
                     cancel: "Keep Image Support") { model.removeVisionPack(for: install) }
        .tuffConfirm("Delete \(install.descriptor.displayName)?",
                     isPresented: $showingDeleteConfirmation,
                     message: deleteMessage,
                     symbol: "trash", confirm: "Delete Model",
                     cancel: "Keep Model") { model.deleteInstalledModel(install) }
        // Grey means "this Mac cannot run it". With restrictions bypassed the
        // card stays in colour, because the buttons work, and the requirement
        // is still stated below in orange.
        .saturation(requirementsSatisfied ? 1 : 0)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(install.descriptor.displayName)
                .appFont(compact ? .callout.weight(.medium) : .headline)
            // No badges beside the name: the selected card's accent border
            // says it is selected, and "Recommended" was removed.
            Spacer(minLength: 8)
            statusBadge
        }
    }

    /// The add-on names itself, not its model: it lives inside that model's
    /// card, so repeating the name was the model's name twice on one row.
    private static let imageSupportName = "Image Support"

    private var visionDisplayName: String {
        guard install.descriptor.supportsImageInput else { return "image support" }
        return Self.imageSupportName
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch status {
        case .installed:
            Label("Installed", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .labelStyle(.titleAndIcon)
                .appFont(.caption)
        case .downloading:
            Label("Downloading", systemImage: "arrow.down.circle")
                .foregroundStyle(TUFFMacTheme.accentTextColor)
                .appFont(.caption)
        case .paused:
            Label("Paused", systemImage: "pause.circle")
                .foregroundStyle(TUFFDesign.secondaryTextColor)
                .appFont(.caption)
        case .notInstalled:
            EmptyView()
        }
    }

    private enum RowStatus { case installed, downloading, paused, notInstalled }

    private var status: RowStatus {
        if install.isInstalled { return .installed }
        if install.isInstalling { return .downloading }
        if install.hasPartialDownload { return .paused }
        return .notInstalled
    }

    private var storage: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text((install.descriptor.customIdentity == nil
                ? MetricFormat.storage(install.descriptor.approximateDownloadBytes) + " download · "
                : "Local checkpoint · ")
                + MetricFormat.memory(hardwareEligibility.minimumUnifiedMemoryBytes)
                + " RAM minimum")
                .appFont(.caption.monospacedDigit())
                .foregroundStyle(TUFFDesign.secondaryTextColor)
            if !hardwareEligibility.isCompatible,
               let explanation = hardwareEligibility.explanation {
                warning(explanation, symbol: "memorychip")
            }
            if let requirement = install.requirement, !install.isInstalled,
               !requirement.canInstall {
                warning("Free " + MetricFormat.storage(requirement.shortfallBytes)
                    + " more storage to download.", symbol: "externaldrive")
            }
        }
    }

    private func warning(_ text: String, symbol: String) -> some View {
        Label(text, systemImage: symbol)
            .appFont(.caption)
            .foregroundStyle(TUFFMacTheme.accentTextColor)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Where this model's download stands, as far as the card's layout goes.
    private var stage: String {
        "\(install.isInstalling)\(install.isInstalled)\(install.hasPartialDownload)\(install.isInstallPathBlocked)"
            + "\(model.isVisionInstallTarget(install) && model.isInstallingVisionPack)\(visionIsInstalled)\(visionIsPrepared)"
    }

    @ViewBuilder
    private var progress: some View {
        VStack(alignment: .leading, spacing: 6) {
            if case .copyingPayload(let reused, let downloaded, let total) = install.state,
               total > 0 {
                let completed = reused + downloaded
                let fraction = min(1, Double(completed) / Double(total))
                ProgressView(value: fraction)
                    .animation(.linear(duration: 0.25), value: fraction)
                    .accessibilityLabel("\(install.descriptor.displayName) download")
                    .accessibilityValue(Text(accessibleProgressValue(fraction: fraction)))
                HStack {
                    Text("\(MetricFormat.storage(completed)) of \(MetricFormat.storage(total))")
                    Spacer(minLength: 12)
                    Text(MetricFormat.percent(fraction * 100))
                    if let eta = install.etaText {
                        Text(eta)
                    }
                }
                .appFont(.caption.monospacedDigit())
                .foregroundStyle(TUFFDesign.secondaryTextColor)
                if reused > 0 {
                    Text("Reused \(MetricFormat.storage(reused)) from the saved download")
                        .appFont(.caption)
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                }
            } else {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(stageLabel).appFont(.caption).foregroundStyle(TUFFDesign.secondaryTextColor)
                    Spacer()
                }
            }
        }
    }

    /// Says what the confirmation removes, including the image add-on when
    /// one is installed, since it is deleted with the model.
    private var deleteMessage: String {
        let addOn = model.hasVisionPackDirectory(for: install)
            ? " and its \(visionDisplayName) add-on"
            : ""
        return "The downloaded files\(addOn) are removed from this Mac, with its "
            + "saved prompt cache and its engine if no other model uses it. "
            + "Download it again to use it later. Other models are untouched."
    }

    private var deleteHelp: String {
        if model.deletingModelIDs.contains(install.id) {
            return "Deleting…"
        }
        if isSelected {
            return "Select a different model to delete this one."
        }
        if model.isRunning || model.loadState.isLoading {
            return "Available when the current reply has finished."
        }
        return "Delete the downloaded files for this model."
    }

    private var blockedPathHelp: String {
        let entries = install.blockedInstallEntries
        let names = entries.map { ($0 as NSString).lastPathComponent }.joined(separator: ", ")
        return "Delete \(names) and download again. A symlink is unlinked; "
            + "whatever it points at is left alone."
    }

    @ViewBuilder
    private var message: some View {
        if install.isInstallPathBlocked {
            Label("Something is already in the way at this location. "
                  + "Remove and Retry clears it and downloads again.",
                  systemImage: "exclamationmark.triangle.fill")
                .appFont(.caption)
                .foregroundStyle(TUFFMacTheme.accentTextColor)
                .fixedSize(horizontal: false, vertical: true)
        }
        switch install.state {
        case .failed(let text):
            Label(text, systemImage: "exclamationmark.triangle.fill")
                .appFont(.caption)
                .foregroundStyle(TUFFMacTheme.accentTextColor)
                .fixedSize(horizontal: false, vertical: true)
        case .recoverable(let text):
            Label(text, systemImage: "exclamationmark.triangle.fill")
                .appFont(.caption)
                .foregroundStyle(TUFFMacTheme.accentTextColor)
                .fixedSize(horizontal: false, vertical: true)
        default:
            if case .failed(let text) = install.readiness, !install.isInstalled {
                Label(text, systemImage: "exclamationmark.triangle.fill")
                    .appFont(.caption)
                    .foregroundStyle(TUFFMacTheme.accentTextColor)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var actions: some View {
        VStack(alignment: .trailing, spacing: 3) {
        HStack(spacing: 8) {
            if install.isInstalling {
                Button("Cancel") { install.cancel() }
                    .disabled(!install.canCancel)
            } else {
                if install.hasPartialDownload && !install.isInstalled {
                    Button("Discard", role: .destructive) {
                        showingDiscardConfirmation = true
                    }
                    .disabled(!install.canDiscard)
                }
                if install.isInstallPathBlocked {
                    Button("Remove and Retry", role: .destructive) {
                        install.clearBlockedInstallPath()
                    }
                    .buttonStyle(.tuffProminent)
                    .disabled(!install.canClearInstallPath)
                    .help(blockedPathHelp)
                } else if !install.isInstalled {
                    Button(install.hasPartialDownload ? "Resume" : "Download") {
                        model.installModel(install)
                    }
                    .buttonStyle(.tuffProminent)
                    .disabled(!model.canInstallModel(install))
                    .help(downloadHelp)
                }
            }
            Spacer(minLength: 8)
            // A custom checkpoint is the person's own folder; its entry is
            // removed in Settings, which leaves the folder alone.
            if install.isInstalled, !install.descriptor.isCustomCheckpoint {
                Button(role: .destructive) {
                    showingDeleteConfirmation = true
                } label: {
                    Label("Delete", systemImage: "trash")
                }
                .disabled(!model.canDeleteInstalledModel(install))
                .help(deleteHelp)
            }
            if !isSelected {
                Button("Use This Model") { model.selectModel(install) }
                    .disabled(!model.canSelectModel(install))
                    .help(selectionHelp)
            }
        }
        if !isSelected, model.isRunning, install.isInstalled {
            Text("Stop the reply to switch")
                .appFont(.caption2)
                .foregroundStyle(TUFFDesign.secondaryTextColor)
        }
        }
        .controlSize(compact ? .small : .regular)
    }

    private var visionSupport: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Label(Self.imageSupportName, systemImage: "photo")
                    .appFont(.caption.weight(.semibold))
                Spacer(minLength: 8)
                Text(visionStatusText)
                    .appFont(.caption2)
                    .foregroundStyle(visionStatusColor)
            }

            if model.isVisionInstallTarget(install),
               let fraction = model.visionInstallProgressFraction {
                ProgressView(value: fraction)
                    .animation(.linear(duration: 0.25), value: fraction)
                    .accessibilityLabel("Image support download")
                    .accessibilityValue(Text(MetricFormat.percent(fraction * 100)))
                HStack {
                    Text(MetricFormat.percent(fraction * 100))
                    Spacer()
                    if let eta = model.visionInstallETAText { Text(eta) }
                }
                .appFont(.caption2.monospacedDigit())
                .foregroundStyle(TUFFDesign.secondaryTextColor)
            }

            visionMessage

            HStack(alignment: .firstTextBaseline, spacing: 8) {
                visionAction
                Spacer(minLength: 8)
                if !visionIsInstalled, !visionIsPrepared,
                   let visionDescriptor {
                    Text(MetricFormat.storage(visionDescriptor.approximateDownloadBytes))
                        .appFont(.caption2.monospacedDigit())
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                }
            }
            .controlSize(compact ? .small : .regular)
        }
    }

    @ViewBuilder
    private var visionAction: some View {
        if model.isVisionInstallTarget(install), model.isInstallingVisionPack {
            Button("Cancel Download") {
                model.cancelVisionInstall()
            }
            .disabled(!model.canCancelVisionInstall)
        } else if visionIsInstalled {
            Button("Remove", role: .destructive) {
                showingVisionRemoveConfirmation = true
            }
            .disabled(!model.canRemoveVisionPack(for: install))
        } else if visionIsPrepared {
            if isSelected {
                if model.loadState.isReady {
                    Button("Unload to Activate Image Support") {
                        model.unloadModel()
                    }
                    .disabled(!model.canUnloadModel)
                } else {
                    Button("Activate Image Support") {
                        model.activateVisionPack()
                    }
                    .buttonStyle(.tuffProminent)
                    .disabled(!model.canActivateVisionPack)
                }
            } else {
                Button("Use This Model to Activate") {
                    model.selectModel(install)
                }
                .disabled(!model.canSelectModel(install))
            }
        } else {
            if model.hasPartialVisionPackDownload(for: install) {
                Button("Discard", role: .destructive) {
                    showingVisionDiscardConfirmation = true
                }
                .disabled(!model.canDiscardVisionPackDownload(for: install))
            }
            Button(model.visionDownloadButtonLabel(for: install)) {
                model.installVisionPack(for: install)
            }
            .buttonStyle(.tuffSecondary)
            .disabled(!model.canInstallVisionPack(for: install))
            .help("Optional download for this model only. The text model stays installed.")
        }
    }

    @ViewBuilder
    private var visionMessage: some View {
        if !model.isVisionRuntimeSupported {
            Label("Image input requires an M2 or newer Mac", systemImage: "memorychip")
                .foregroundStyle(TUFFMacTheme.accentTextColor)
        } else if model.isVisionInstallTarget(install),
                  case .failed(let text) = model.visionInstallState {
            Label(text, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(TUFFMacTheme.accentTextColor)
        } else if model.isVisionInstallTarget(install),
                  case .recoverable(let text) = model.visionInstallState {
            Label(text, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(TUFFMacTheme.accentTextColor)
        } else if let requirement = model.visionInstallRequirement(for: install),
                  !requirement.canInstall, !visionIsInstalled {
            Label("Free \(MetricFormat.storage(requirement.shortfallBytes)) more storage.",
                  systemImage: "internaldrive")
                .foregroundStyle(TUFFMacTheme.accentTextColor)
        }
    }

    private var visionDescriptor: AppModelInstallDescriptor? {
        model.visionInstallDescriptor(for: install)
    }

    private var hardwareEligibility: AppModelHardwareEligibility {
        model.hardwareEligibility(for: install)
    }

    private var requirementsSatisfied: Bool {
        model.hardwareRequirementsSatisfied(for: install)
    }

    private var downloadHelp: String {
        if !requirementsSatisfied, let explanation = hardwareEligibility.explanation {
            return explanation
        }
        if let requirement = install.requirement, !requirement.canInstall {
            return "Requires \(MetricFormat.storage(requirement.shortfallBytes)) more disk space."
        }
        return "Download and install \(install.descriptor.displayName)"
    }

    private var selectionHelp: String {
        if !requirementsSatisfied, let explanation = hardwareEligibility.explanation {
            return explanation
        }
        if model.isRunning {
            return "Stop the current reply to switch models"
        }
        return install.isInstalled
            ? "Load and generate with this model"
            : "Make this the model the app loads once it is installed"
    }

    private var visionIsInstalled: Bool {
        model.isVisionPackInstalled(for: install)
    }

    private var visionIsPrepared: Bool {
        guard model.isVisionInstallTarget(install) else { return false }
        if case .readyToActivate = model.visionInstallState { return true }
        return false
    }

    private var visionStatusText: String {
        guard model.isVisionRuntimeSupported else { return "Requires M2 or newer" }
        if model.isVisionInstallTarget(install) {
            if model.visionInstallState != .idle {
                return model.visionInstallPhaseLabel
            }
        }
        switch model.visionInstallationStatus(for: install) {
        case .complete: return "Installed"
        case .partial: return "Needs repair"
        case .missing: return ""
        case .unsupportedLayout: return "Unavailable"
        }
    }

    private var visionStatusColor: Color {
        if visionIsInstalled { return .green }
        if case .partial = model.visionInstallationStatus(for: install) {
            return TUFFMacTheme.accentTextColor
        }
        return .secondary
    }

    private var stageLabel: String {
        switch install.state {
        case .checking: return "Checking"
        case .downloadingMetadata: return "Reading the checkpoint index"
        case .planning: return "Planning the repack"
        case .reservingOutput: return "Reserving disk space"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing"
        case .cancelling: return "Cancelling"
        case .discarding: return "Discarding the saved download"
        default: return "Working"
        }
    }

    private func accessibleProgressValue(fraction: Double) -> String {
        let percent = MetricFormat.percent(fraction * 100)
        guard let eta = install.etaText else { return percent }
        return "\(percent), \(eta)"
    }
}
