import Foundation
import TUFFModelCatalog
import TUFFRuntimeDescriptions
import TUFFRepackCore
import TUFFRuntimeManagement

public enum AppReasoningControl: Equatable, Sendable {
    case toggle
    case toggleWithPreservation
    case graded
    case alwaysOn
}

public enum AppModelHardwareIssue: Equatable, Sendable {
    case requiresRealModelValidation
    case insufficientUnifiedMemory(requiredBytes: UInt64, actualBytes: UInt64)
    case unsupportedMacOS(requiredMajorVersion: Int, actualMajorVersion: Int)
    case unsupportedAppleSilicon(requiredGeneration: Int, actualGeneration: Int)
}

public struct AppModelHardwareEligibility: Equatable, Sendable {
    public let minimumUnifiedMemoryBytes: UInt64
    public let actualUnifiedMemoryBytes: UInt64
    public let issues: [AppModelHardwareIssue]

    public init(
        minimumUnifiedMemoryBytes: UInt64,
        actualUnifiedMemoryBytes: UInt64,
        issues: [AppModelHardwareIssue]
    ) {
        self.minimumUnifiedMemoryBytes = minimumUnifiedMemoryBytes
        self.actualUnifiedMemoryBytes = actualUnifiedMemoryBytes
        self.issues = issues
    }

    public var isCompatible: Bool { issues.isEmpty }

    public var explanation: String? {
        guard let issue = issues.first else { return nil }
        switch issue {
        case .requiresRealModelValidation:
            return "Needs a real validation run on qualifying Apple Silicon before download."
        case .insufficientUnifiedMemory(let required, let actual):
            return "Requires \(Self.memoryLabel(required)) unified memory; "
                + "this Mac has \(Self.memoryLabel(actual))."
        case .unsupportedMacOS(let required, let actual):
            return "Requires macOS \(required) or newer; this Mac is running macOS \(actual)."
        case .unsupportedAppleSilicon(let required, let actual):
            return "Requires an M\(required) or newer Mac; this Mac reports M\(actual)."
        }
    }

    private static func memoryLabel(_ bytes: UInt64) -> String {
        let gibibytes = Double(bytes) / Double(TUFFModelCatalog.oneGiB)
        return "\(gibibytes.formatted(.number.precision(.fractionLength(0...1)))) GB"
    }
}

public struct AppModelContextEligibility: Equatable, Sendable {
    public let estimatedWorkingSetBytes: UInt64
    public let safeBudgetBytes: UInt64

    public init(estimatedWorkingSetBytes: UInt64, safeBudgetBytes: UInt64) {
        self.estimatedWorkingSetBytes = estimatedWorkingSetBytes
        self.safeBudgetBytes = safeBudgetBytes
    }

    public var isCompatible: Bool { estimatedWorkingSetBytes <= safeBudgetBytes }

    public var explanation: String? {
        guard !isCompatible else { return nil }
        return "Estimated memory is \(Self.memoryLabel(estimatedWorkingSetBytes)); "
            + "this Mac's safe app budget is \(Self.memoryLabel(safeBudgetBytes))."
    }

    private static func memoryLabel(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: bytes),
                                  countStyle: .memory)
    }
}

public struct AppModelInstallDescriptor: Equatable, Sendable {
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let rangeStagingBytes: UInt64
    public let reserveBytes: UInt64
    public let customIdentity: String?
    public let customArchitectureTemplate: TUFFModelID?
    public let customManifestSHA256: String?

    public init(catalog descriptor: TUFFModelDescriptor) {
        let source = descriptor.source
        self.init(
            displayName: descriptor.displayName,
            repoID: source.repoID,
            revision: source.revision,
            sourceIndexSHA256: source.sourceIndexSHA256,
            approximateDownloadBytes: source.approximateDownloadBytes,
            installedBytes: source.installedBytes,
            rangeStagingBytes: descriptor.architecture.weightLayout == .prismQ2
                ? source.approximateDownloadBytes + 8 * 1024 * 1024
                : UInt64(RemoteChunkPolicy.defaultBytes) * 2,
            reserveBytes: source.reserveBytes)
    }

    public init(addon: TUFFModelAddonDescriptor) {
        let source = addon.source
        self.init(
            displayName: addon.displayName,
            repoID: source.repoID,
            revision: source.revision,
            sourceIndexSHA256: source.sourceIndexSHA256,
            approximateDownloadBytes: source.approximateDownloadBytes,
            installedBytes: source.installedBytes,
            rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes) * 2,
            reserveBytes: source.reserveBytes)
    }

    public init(displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                rangeStagingBytes: UInt64,
                reserveBytes: UInt64,
                customIdentity: String? = nil,
                customArchitectureTemplate: TUFFModelID? = nil,
                customManifestSHA256: String? = nil) {
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.rangeStagingBytes = rangeStagingBytes
        self.reserveBytes = reserveBytes
        self.customIdentity = customIdentity
        self.customArchitectureTemplate = customArchitectureTemplate
        self.customManifestSHA256 = customManifestSHA256
    }

    public var requiredFreeBytes: UInt64 {
        installedBytes + rangeStagingBytes + reserveBytes + engineInstallationTemporaryBytes
    }

    private var engineRelease: InferenceEngineRelease? {
        guard let architecture = planningCatalogID.flatMap(TUFFModelCatalog.model(id:))?.architecture.id.rawValue,
              let controller = try? InferenceEngineEnvironment.controller() else { return nil }
        return try? controller.catalog.release(architectureID: architecture,
            macOSMajorVersion: ProcessInfo.processInfo.operatingSystemVersion.majorVersion)
    }
    public var engineDownloadBytes: UInt64 { engineRelease?.archiveBytes ?? 0 }
    public var projectionLayerCount: Int {
        guard let architecture = planningCatalogID.flatMap(TUFFModelCatalog.model(id:))?.architecture,
              architecture.weightLayout == .affine,
              let variant = ModelVariant(rawValue: architecture.id.rawValue) else { return 0 }
        return ArchConfig.registeredArchitectures[variant]?.numLayers ?? 0
    }
    public var engineInstalledBytes: UInt64 { engineRelease?.installedBytes ?? 0 }
    /// Conservative peak if the shared family engine needs installation/repair.
    public var engineInstallationTemporaryBytes: UInt64 {
        guard let release = engineRelease else { return 0 }
        return release.archiveBytes + release.installedBytes * 2 + (1 << 20)
    }

    public static let `default` = AppModelInstallDescriptor(
        catalog: TUFFModelCatalog.gemma4_26B_A4B)

    public static let gemma4E4B = AppModelInstallDescriptor(
        catalog: TUFFModelCatalog.gemma4_E4B)

    public static let qwen36 = AppModelInstallDescriptor(
        catalog: TUFFModelCatalog.qwen36_35B_A3B)

    public static let minimaxM27 = AppModelInstallDescriptor(
        catalog: TUFFModelCatalog.minimaxM27)
    public static let qwen38FlashNext = AppModelInstallDescriptor(
        catalog: TUFFModelCatalog.qwen38FlashNext)

    /// The shipped descriptor for a model family, if one exists.
    public static func descriptor(for family: ModelFamily) -> AppModelInstallDescriptor? {
        catalog.first { $0.family == family }
    }

    public static func descriptor(for variant: ModelVariant) -> AppModelInstallDescriptor? {
        catalog.first { $0.catalogDescriptor?.architecture.id.rawValue == variant.rawValue }
    }

    /// Basename of the installed `.gturbo` directory for this descriptor.
    public var installDirectoryName: String {
        if let customIdentity { return "\(customIdentity).gturbo" }
        return catalogDescriptor?.installDirectoryName
            ?? (self == .qwen36 ? "qwen36.gturbo" : "gemma4.gturbo")
    }

    /// Stable identity for catalog lookups and UI selection. The install
    /// directory basename already distinguishes the shipped models and is what
    /// separates their downloads on disk, so it is the natural key.
    public var id: String { installDirectoryName }

    /// Stable registry identity used by settings and other persisted app data.
    /// Test-only/custom descriptors keep their install basename as a safe
    /// fallback so they never collide with a shipped model profile.
    public var catalogID: TUFFModelID? {
        customIdentity == nil ? catalogDescriptor?.id : nil
    }
    public var planningCatalogID: TUFFModelID? { customArchitectureTemplate ?? catalogID }
    public var isCustomCheckpoint: Bool { customIdentity != nil }

    public var isLegacy: Bool { catalogDescriptor?.maintenanceStatus == .legacy }

    public var settingsProfileKey: String {
        catalogID?.rawValue ?? id
    }

    public var apiModelID: String {
        isCustomCheckpoint ? settingsProfileKey : (catalogDescriptor?.apiModelID ?? settingsProfileKey)
    }

    /// The model family this descriptor installs.
    public var family: ModelFamily {
        catalogDescriptor.flatMap { ModelFamily(rawValue: $0.family.rawValue) }
            ?? (self == .qwen36 ? .qwen36 : .gemma4)
    }

    public var chatDialect: ChatDialect {
        switch family {
        case .gemma4: .gemma
        case .qwen36, .qwen4Exp: .chatml
        case .flint: .flint
        case .gptOss: .harmony
        case .minimaxM2: .minimax
        }
    }

    /// Whether this exact checkpoint has a qualified image companion. Custom
    /// descriptors retain the v1 behavior for tests and local builds.
    public var supportsImageInput: Bool {
        !isCustomCheckpoint && (catalogDescriptor?.capabilities.contains(.imageInput) ?? true)
    }

    /// Chunk size the app runs prefill at for this model on `device`; see
    /// `TUFFModelDescriptor.recommendedPrefillChunkTokens(on:)`. A descriptor
    /// outside the catalog keeps the size the app always used.
    public func recommendedPrefillChunkTokens(on device: TUFFDeviceCapabilities) -> Int {
        catalogDescriptor?.recommendedPrefillChunkTokens(on: device) ?? 128
    }

    public var isRecommended: Bool {
        catalogID == TUFFModelCatalog.default.id
    }

    public var usesExpertCache: Bool {
        catalogDescriptor?.architecture.feedForwardKind == .mixtureOfExperts
    }

    /// Cache sizes that can help this model. Counts below routed top-k cannot
    /// execute a token, while counts above the model's expert count only create
    /// idle buffers. Custom descriptors retain the shared runtime choices.
    public var usefulExpertCacheSlotCounts: [Int] {
        guard let architectureID = catalogDescriptor?.architecture.id,
              let variant = ModelVariant(rawValue: architectureID.rawValue),
              let architecture = ArchConfig.registeredArchitectures[variant],
              architecture.numExperts > 0 else {
            return AppRuntimeOptions.allowedSlotCounts
        }
        return AppRuntimeOptions.allowedSlotCounts.filter {
            $0 >= architecture.topKExperts && $0 <= architecture.numExperts
        }
    }

    /// Unified memory to run this model comfortably at its default settings,
    /// as opposed to the floor below which it is refused outright.
    ///
    /// The minimum is a hard gate: at it, the context and expert cache have to
    /// be cut back before the model will load. This is the size where the
    /// defaults fit with macOS and other applications still running — the
    /// measured working set doubled, plus 4 GiB for the system, rounded up to a
    /// size Macs are actually sold in. It is a rule of thumb and the interface
    /// labels it as one; the working-set figure it is derived from is measured.
    public var recommendedUnifiedMemoryBytes: UInt64 {
        let minimum = catalogDescriptor?.hardware.minimumUnifiedMemoryBytes ?? 0
        guard let memory = catalogDescriptor?.memory else { return minimum }
        let workingSet = memory.estimatedWorkingSetBytes(
            contextTokens: memory.defaultContextTokens,
            expertCacheSlots: memory.defaultExpertCacheSlots)
        guard workingSet != .max else { return minimum }
        let wanted = workingSet * 2 + 4 * 1_073_741_824
        return Self.unifiedMemoryTiers.first { $0 >= max(wanted, minimum) }
            ?? max(wanted, minimum)
    }

    /// Unified memory sizes Apple Silicon Macs ship with. Rounding to these
    /// keeps the recommendation something a buyer can act on.
    static let unifiedMemoryTiers: [UInt64] = [8, 16, 24, 32, 36, 48, 64, 96, 128]
        .map { UInt64($0) * 1_073_741_824 }

    public func hardwareEligibility(
        on device: TUFFDeviceCapabilities
    ) -> AppModelHardwareEligibility {
        guard let catalogDescriptor else {
            return AppModelHardwareEligibility(
                minimumUnifiedMemoryBytes: 0,
                actualUnifiedMemoryBytes: device.unifiedMemoryBytes,
                issues: [])
        }
        let compatibility = catalogDescriptor.compatibility(with: device)
        let issues = compatibility.issues.compactMap { issue -> AppModelHardwareIssue? in
            switch issue {
            case .requiresRealModelValidation:
                return .requiresRealModelValidation
            case .insufficientUnifiedMemory(let required, let actual):
                return .insufficientUnifiedMemory(
                    requiredBytes: required, actualBytes: actual)
            case .unsupportedMacOS(let required, let actual):
                return .unsupportedMacOS(
                    requiredMajorVersion: required, actualMajorVersion: actual)
            case .unsupportedAppleSilicon(let required, let actual):
                return .unsupportedAppleSilicon(
                    requiredGeneration: required, actualGeneration: actual)
            case .contextExceedsSafeMemory:
                return nil
            }
        }
        return AppModelHardwareEligibility(
            minimumUnifiedMemoryBytes: catalogDescriptor.hardware.minimumUnifiedMemoryBytes,
            actualUnifiedMemoryBytes: device.unifiedMemoryBytes,
            issues: issues)
    }

    public func contextEligibility(
        contextTokens: Int,
        expertCacheSlots: Int,
        prefillChunkTokens: Int? = nil,
        on device: TUFFDeviceCapabilities
    ) -> AppModelContextEligibility {
        guard let catalogDescriptor else {
            return AppModelContextEligibility(
                estimatedWorkingSetBytes: 0,
                safeBudgetBytes: device.unifiedMemoryBytes)
        }
        let estimate = catalogDescriptor.estimatedInferenceWorkingSetBytes(
            contextTokens: contextTokens,
            expertCacheSlots: expertCacheSlots,
            prefillChunkTokens: prefillChunkTokens ?? recommendedPrefillChunkTokens(on: device))
        let budget = device.safeAppMemoryBudgetBytes
        return AppModelContextEligibility(
            estimatedWorkingSetBytes: estimate,
            safeBudgetBytes: budget)
    }

    public var reasoningControl: AppReasoningControl? {
        switch catalogDescriptor?.reasoningControl {
        case .toggle: .toggle
        case .toggleWithPreservation: .toggleWithPreservation
        case .graded: .graded
        case .alwaysOn: .alwaysOn
        case nil: nil
        }
    }

    /// Every model the app can install, in the order the UI lists them.
    public static let catalog: [AppModelInstallDescriptor] =
        TUFFModelCatalog.all.map(AppModelInstallDescriptor.init(catalog:))

    /// A short line describing what the model is good for, shown beside its
    /// name in the picker.
    public var summary: String {
        catalogDescriptor?.summary ?? displayName
    }

    public var shortName: String {
        isCustomCheckpoint ? displayName : (catalogDescriptor?.shortName ?? displayName)
    }

    public var defaultSystemPrompt: String {
        catalogDescriptor?.defaultSystemPrompt
            ?? "You are \(shortName), a helpful AI assistant. "
                + "You are running locally on a Mac in an app called TUFF."
    }

    /// The descriptor the app products select at launch. Defaults to Gemma 4.
    /// `TUFF_MODEL=qwen36` in the environment wins; otherwise the persisted
    /// preference (`defaults write TUFF model qwen36`) applies, so GUI launches
    /// without an environment also select Qwen.
    ///
    /// v3 renamed both the variable and the preference domain from the old
    /// `TURBO_FIELDFARE_MODEL` / `TurboFieldfare`. The old names are still read,
    /// after the new ones, so a machine that already selected Qwen keeps it.
    public static var selected: AppModelInstallDescriptor {
        let environment = ProcessInfo.processInfo.environment
        let environmentValue = environment["TUFF_MODEL"]
            ?? environment["TURBO_FIELDFARE_MODEL"]
        let suite = TUFFApplicationDirectories.preferencesSuiteName(environment: environment)
        let preferenceValue = UserDefaults(suiteName: suite)?
            .string(forKey: "model")
            ?? (suite == "TUFF" ? UserDefaults(suiteName: "TurboFieldfare")?.string(forKey: "model") : nil)
        guard let selector = environmentValue ?? preferenceValue,
              let descriptor = TUFFModelCatalog.model(selector: selector) else {
            return .default
        }
        return AppModelInstallDescriptor(catalog: descriptor)
    }

    public static let visionCompanion = AppModelInstallDescriptor(
        addon: TUFFModelCatalog.gemma4_26B_A4B.addons[0])

    public static let qwen36VisionCompanion = AppModelInstallDescriptor(
        addon: TUFFModelCatalog.qwen36_35B_A3B.addons[0])

    public static let qwen38FlashNextVisionCompanion = AppModelInstallDescriptor(
        addon: TUFFModelCatalog.qwen38FlashNext.addons[0])

    /// Family-only routing, for descriptors that match no catalog row. `nil`
    /// for a family that ships no image pack at all.
    ///
    /// Prefer the `ModelVariant` overload: Gemma 4 checkpoints do not share a
    /// vision tensor layout, so routing by family alone hands every Gemma the
    /// 26B pack and silently installs a valid pack for the wrong model.
    public static func visionCompanion(
        for family: ModelFamily
    ) -> AppModelInstallDescriptor? {
        switch family {
        case .gemma4: return .visionCompanion
        case .qwen36: return .qwen36VisionCompanion
        case .gptOss: return nil
        case .minimaxM2, .flint: return nil
        // Flash Next's tower is Qwen3.6's, but its merger projects into a
        // wider residual, so the two packs are not interchangeable and each
        // family routes to its own.
        case .qwen4Exp: return .qwen38FlashNextVisionCompanion
        }
    }

    /// The image pack belonging to one exact text architecture, or `nil` when
    /// that architecture ships none.
    ///
    /// Returning `nil` rather than trapping: the only guard in front of the
    /// inspector's copy of this call is whether the *device* supports the
    /// vision runtime, so selecting a text-only model on a capable Mac reached
    /// a `preconditionFailure` and took the app down.
    public static func visionCompanion(
        for variant: ModelVariant
    ) -> AppModelInstallDescriptor? {
        guard let model = TUFFModelCatalog.all.first(where: {
            $0.architecture.id.rawValue == variant.rawValue
        }), let addon = model.addons.first(where: { $0.kind == .imageInput }) else {
            return nil
        }
        return AppModelInstallDescriptor(addon: addon)
    }

    public var modelVariant: ModelVariant? {
        catalogDescriptor.flatMap { ModelVariant(rawValue: $0.architecture.id.rawValue) }
    }

    /// A companion belongs to the checkpoint, not merely its architecture.
    public var imageCompanion: AppModelInstallDescriptor? {
        guard !isCustomCheckpoint, let catalogDescriptor,
              let addon = catalogDescriptor.addons.first(where: { $0.kind == .imageInput }) else { return nil }
        return AppModelInstallDescriptor(addon: addon)
    }

    private var catalogDescriptor: TUFFModelDescriptor? {
        if let customArchitectureTemplate { return TUFFModelCatalog.model(id: customArchitectureTemplate) }
        return TUFFModelCatalog.all.first {
            $0.source.repoID == repoID
                && $0.source.revision == revision
                && $0.source.sourceIndexSHA256 == sourceIndexSHA256
                && $0.source.installedBytes == installedBytes
        }
    }
}

public struct AppModelInstallRequirement: Equatable, Sendable {
    public let probePath: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(probePath: String = "", requiredBytes: UInt64, availableBytes: UInt64) {
        self.probePath = probePath
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }

    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public enum AppModelInstallReadiness: Equatable, Sendable {
    case checking
    case ready(AppModelInstallRequirement)
    case insufficientSpace(AppModelInstallRequirement)
    case failed(String)

    public var requirement: AppModelInstallRequirement? {
        switch self {
        case .ready(let requirement), .insufficientSpace(let requirement):
            return requirement
        case .checking, .failed:
            return nil
        }
    }
}
