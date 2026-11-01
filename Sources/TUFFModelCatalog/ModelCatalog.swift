import Foundation
import Darwin

public enum TUFFModelID: String, Codable, CaseIterable, Sendable {
    case flint40B = "flint-40b-a2.5b"
    case bonsai2_2B = "ternary-bonsai-2-2b"
    case bonsai2_27B = "ternary-bonsai-2-27b"
    case gemma4_E2B = "gemma4-e2b"
    case gemma4_E4B = "gemma4-e4b"
    case gemma4_12B_QAT = "gemma4-12b-qat"
    case gemma4_26B_A4B = "gemma4-26b-a4b"
    case qwen36_35B_A3B = "qwen36-35b-a3b"
    case gptOss_20B = "gpt-oss-20b"
    case gptOss_120B = "gpt-oss-120b"
    case minimaxM27 = "minimax-m2.7"
    case qwen38FlashNext = "qwen3.8-flash-next"
    case ornith15_9B = "ornith-1.5-9b"
    case ornith15_35B_A3B = "ornith-1.5-35b-a3b"
    case ornith15_122B = "ornith-1.5-122b"
    case ornith15_397B = "ornith-1.5-397b"
    case qwen38_27B = "qwen3.8-27b"
    case gemma4_26B_Heretic = "gemma4-26b-heretic"
}

public enum TUFFModelFamily: String, Codable, Sendable {
    case flint
    case gemma4
    case qwen36
    case gptOss = "gpt-oss"
    case minimaxM2 = "minimax-m2"
    case qwen4Exp = "qwen4-exp"
}

public enum TUFFArchitectureID: String, Codable, Sendable {
    case flint40B = "flint-40b-a2.5b"
    case bonsai2_2B = "ternary-bonsai-2-2b"
    case bonsai2_27B = "ternary-bonsai-2-27b"
    case gemma4_E2B = "gemma4-e2b"
    case gemma4_E4B = "gemma4-e4b"
    case gemma4_12B_QAT = "gemma4-12b-qat"
    case gemma4_26B_A4B = "gemma4-26b-a4b"
    case qwen36_35B_A3B = "qwen36-35b-a3b"
    case gptOss_20B = "gpt-oss-20b"
    case gptOss_120B = "gpt-oss-120b"
    case minimaxM27 = "minimax-m2.7"
    case qwen38FlashNext = "qwen3.8-flash-next"
    case ornith15_9B = "ornith-1.5-9b"
    case ornith15_35B_A3B = "ornith-1.5-35b-a3b"
    case ornith15_122B = "ornith-1.5-122b"
    case ornith15_397B = "ornith-1.5-397b"
    case qwen38_27B = "qwen3.8-27b"
}

public enum TUFFFeedForwardKind: String, Codable, Sendable {
    case dense
    case mixtureOfExperts = "moe"
}

public enum TUFFWeightLayout: String, Codable, Sendable {
    case prismQ2 = "prism-q2"
    case affine
    case mxfp4
}

public struct TUFFArchitectureProfile: Codable, Equatable, Sendable {
    public let id: TUFFArchitectureID
    public let family: TUFFModelFamily
    public let feedForwardKind: TUFFFeedForwardKind
    public let weightLayout: TUFFWeightLayout

    public init(id: TUFFArchitectureID,
                family: TUFFModelFamily,
                feedForwardKind: TUFFFeedForwardKind,
                weightLayout: TUFFWeightLayout) {
        self.id = id
        self.family = family
        self.feedForwardKind = feedForwardKind
        self.weightLayout = weightLayout
    }
}

public extension TUFFArchitectureProfile {
    static let flint40B = TUFFArchitectureProfile(id: .flint40B, family: .flint, feedForwardKind: .mixtureOfExperts, weightLayout: .affine)
    static let bonsai2_2B = TUFFArchitectureProfile(id: .bonsai2_2B, family: .qwen36, feedForwardKind: .dense, weightLayout: .prismQ2)
    static let bonsai2_27B = TUFFArchitectureProfile(id: .bonsai2_27B, family: .qwen36, feedForwardKind: .dense, weightLayout: .prismQ2)
    static let gemma4E2B = TUFFArchitectureProfile(
        id: .gemma4_E2B,
        family: .gemma4,
        feedForwardKind: .dense,
        weightLayout: .affine)

    static let gemma4E4B = TUFFArchitectureProfile(
        id: .gemma4_E4B,
        family: .gemma4,
        feedForwardKind: .dense,
        weightLayout: .affine)

    static let gemma4_12B_QAT = TUFFArchitectureProfile(
        id: .gemma4_12B_QAT,
        family: .gemma4,
        feedForwardKind: .dense,
        weightLayout: .affine)

    static let gemma4_26B_A4B = TUFFArchitectureProfile(
        id: .gemma4_26B_A4B,
        family: .gemma4,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .affine)

    static let qwen36_35B_A3B = TUFFArchitectureProfile(
        id: .qwen36_35B_A3B,
        family: .qwen36,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .affine)

    static let gptOss_20B = TUFFArchitectureProfile(
        id: .gptOss_20B,
        family: .gptOss,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .mxfp4)

    static let gptOss_120B = TUFFArchitectureProfile(
        id: .gptOss_120B,
        family: .gptOss,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .mxfp4)

    static let minimaxM27 = TUFFArchitectureProfile(
        id: .minimaxM27,
        family: .minimaxM2,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .affine)

    static let qwen38FlashNext = TUFFArchitectureProfile(
        id: .qwen38FlashNext,
        family: .qwen4Exp,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .affine)
}

public extension TUFFArchitectureProfile {
    static let ornith15_9B = TUFFArchitectureProfile(id: .ornith15_9B, family: .qwen36, feedForwardKind: .dense, weightLayout: .affine)
    static let ornith15_35B_A3B = TUFFArchitectureProfile(id: .ornith15_35B_A3B, family: .qwen36, feedForwardKind: .mixtureOfExperts, weightLayout: .affine)
    static let ornith15_122B = TUFFArchitectureProfile(id: .ornith15_122B, family: .qwen36, feedForwardKind: .mixtureOfExperts, weightLayout: .affine)
    static let ornith15_397B = TUFFArchitectureProfile(id: .ornith15_397B, family: .qwen36, feedForwardKind: .mixtureOfExperts, weightLayout: .affine)
    static let qwen38_27B = TUFFArchitectureProfile(id: .qwen38_27B, family: .qwen36, feedForwardKind: .dense, weightLayout: .affine)
}

public enum TUFFReasoningControl: String, Codable, Sendable {
    case toggle
    case toggleWithPreservation
    case graded
    case alwaysOn
}

public enum TUFFModelMaintenanceStatus: String, Codable, Sendable {
    case maintained
    case legacy
}

public enum TUFFModelQualification: String, Codable, Sendable {
    case qualified
    case requiresRealModelValidation
}

public enum TUFFModelCapability: String, Codable, Sendable {
    case textGeneration
    case imageInput
    case reasoning
}

public enum TUFFModelAddonKind: String, Codable, Sendable {
    case imageInput
}

public struct TUFFModelSource: Codable, Equatable, Sendable {
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let manifestModelID: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    public init(repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                manifestModelID: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                reserveBytes: UInt64) {
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.manifestModelID = manifestModelID
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.reserveBytes = reserveBytes
    }
}

public struct TUFFModelHardwareRequirements: Codable, Equatable, Sendable {
    public let minimumUnifiedMemoryBytes: UInt64
    public let minimumMacOSMajorVersion: Int
    public let minimumAppleSiliconGeneration: Int

    public init(minimumUnifiedMemoryBytes: UInt64,
                minimumMacOSMajorVersion: Int = 15,
                minimumAppleSiliconGeneration: Int = 1) {
        self.minimumUnifiedMemoryBytes = minimumUnifiedMemoryBytes
        self.minimumMacOSMajorVersion = minimumMacOSMajorVersion
        self.minimumAppleSiliconGeneration = minimumAppleSiliconGeneration
    }
}

public struct TUFFKVCacheProfile: Codable, Equatable, Sendable {
    public let fullAttentionBytesPerToken: UInt64
    public let slidingAttentionBytesPerToken: UInt64
    public let slidingWindowCapacityTokens: Int

    public init(fullAttentionBytesPerToken: UInt64,
                slidingAttentionBytesPerToken: UInt64 = 0,
                slidingWindowCapacityTokens: Int = 0) {
        self.fullAttentionBytesPerToken = fullAttentionBytesPerToken
        self.slidingAttentionBytesPerToken = slidingAttentionBytesPerToken
        self.slidingWindowCapacityTokens = slidingWindowCapacityTokens
    }

    public func estimatedBytes(contextTokens: Int) -> UInt64 {
        guard contextTokens > 0 else { return 0 }
        let full = fullAttentionBytesPerToken.multipliedReportingOverflow(
            by: UInt64(contextTokens))
        let slidingRows = min(contextTokens, max(0, slidingWindowCapacityTokens))
        let sliding = slidingAttentionBytesPerToken.multipliedReportingOverflow(
            by: UInt64(slidingRows))
        guard !full.overflow, !sliding.overflow else { return .max }
        let total = full.partialValue.addingReportingOverflow(sliding.partialValue)
        return total.overflow ? .max : total.partialValue
    }
}

public struct TUFFModelMemoryProfile: Codable, Equatable, Sendable {
    public let qualifiedDefaultWorkingSetBytes: UInt64
    public let defaultContextTokens: Int
    public let defaultExpertCacheSlots: Int
    public let expertCacheBytesPerSlot: UInt64
    public let kvCache: TUFFKVCacheProfile

    public init(qualifiedDefaultWorkingSetBytes: UInt64,
                defaultContextTokens: Int = 8_192,
                defaultExpertCacheSlots: Int = 16,
                expertCacheBytesPerSlot: UInt64,
                kvCache: TUFFKVCacheProfile) {
        self.qualifiedDefaultWorkingSetBytes = qualifiedDefaultWorkingSetBytes
        self.defaultContextTokens = defaultContextTokens
        self.defaultExpertCacheSlots = defaultExpertCacheSlots
        self.expertCacheBytesPerSlot = expertCacheBytesPerSlot
        self.kvCache = kvCache
    }

    public func estimatedWorkingSetBytes(contextTokens: Int,
                                         expertCacheSlots: Int) -> UInt64 {
        let defaultKV = kvCache.estimatedBytes(contextTokens: defaultContextTokens)
        let requestedKV = kvCache.estimatedBytes(contextTokens: contextTokens)
        let defaultSlots = expertCacheBytesPerSlot.multipliedReportingOverflow(
            by: UInt64(max(0, defaultExpertCacheSlots)))
        let requestedSlots = expertCacheBytesPerSlot.multipliedReportingOverflow(
            by: UInt64(max(0, expertCacheSlots)))
        guard defaultKV != .max, requestedKV != .max,
              !defaultSlots.overflow, !requestedSlots.overflow else { return .max }

        let removableKV = min(defaultKV, qualifiedDefaultWorkingSetBytes)
        let afterKV = qualifiedDefaultWorkingSetBytes - removableKV
        let removableSlots = min(defaultSlots.partialValue, afterKV)
        var estimate = afterKV - removableSlots
        for value in [requestedKV, requestedSlots.partialValue] {
            let addition = estimate.addingReportingOverflow(value)
            guard !addition.overflow else { return .max }
            estimate = addition.partialValue
        }
        return estimate
    }
}

public struct TUFFDeviceCapabilities: Codable, Equatable, Sendable {
    public let unifiedMemoryBytes: UInt64
    /// Free, inactive, and speculative pages reclaimable when TUFF launched.
    /// Nil for synthetic/test capability descriptions and older archives.
    ///
    /// Recorded as a diagnostic only. It used to cap `safeAppMemoryBudgetBytes`,
    /// which made the budget depend on whatever else was open at launch and
    /// refused models a Mac was qualified for; see that property.
    public let availableMemoryBytes: UInt64?
    public let macOSMajorVersion: Int
    public let appleSiliconGeneration: Int

    public init(unifiedMemoryBytes: UInt64,
                availableMemoryBytes: UInt64? = nil,
                macOSMajorVersion: Int,
                appleSiliconGeneration: Int) {
        self.unifiedMemoryBytes = unifiedMemoryBytes
        self.availableMemoryBytes = availableMemoryBytes
        self.macOSMajorVersion = macOSMajorVersion
        self.appleSiliconGeneration = appleSiliconGeneration
    }

    public static func current() -> TUFFDeviceCapabilities {
        let memory = sysctlUInt64(named: "hw.memsize") ?? ProcessInfo.processInfo.physicalMemory
        let generation = sysctlString(named: "machdep.cpu.brand_string")
            .flatMap(appleSiliconGeneration(brandString:)) ?? 1
        return TUFFDeviceCapabilities(
            unifiedMemoryBytes: memory,
            availableMemoryBytes: hostAvailableMemoryBytes(),
            macOSMajorVersion: ProcessInfo.processInfo.operatingSystemVersion.majorVersion,
            appleSiliconGeneration: generation)
    }

    /// Share of installed unified memory TUFF plans against.
    public static let safeAppMemoryShareNumerator: UInt64 = 3
    public static let safeAppMemoryShareDenominator: UInt64 = 4

    /// Memory TUFF can use while retaining a system reserve: a fixed 75% of
    /// installed unified memory.
    ///
    /// This was previously capped by how much memory happened to be
    /// reclaimable when the app launched. That made the budget depend on
    /// whatever else was open at the time, and on a 16 GB Mac it routinely
    /// refused the very models that were qualified on 16 GB — the machine
    /// could host them, but the snapshot said otherwise. Weights are
    /// memory-mapped and their pages are evictable, so the kernel can reclaim
    /// under pressure; a fixed share of installed memory describes what the
    /// machine is, which is the thing a model requirement is written against.
    public var safeAppMemoryBudgetBytes: UInt64 {
        unifiedMemoryBytes / Self.safeAppMemoryShareDenominator
            * Self.safeAppMemoryShareNumerator
    }

    private static func hostAvailableMemoryBytes() -> UInt64? {
        var pageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else {
            return nil
        }
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size
                / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(
                to: integer_t.self,
                capacity: Int(count)
            ) { rebound in
                host_statistics64(
                    mach_host_self(),
                    HOST_VM_INFO64,
                    rebound,
                    &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        let pages = UInt64(stats.free_count)
            + UInt64(stats.inactive_count)
            + UInt64(stats.speculative_count)
        let bytes = pages.multipliedReportingOverflow(by: UInt64(pageSize))
        return bytes.overflow ? nil : bytes.partialValue
    }

    static func appleSiliconGeneration(brandString: String) -> Int? {
        let normalized = brandString.lowercased()
        guard let marker = normalized.range(of: "apple m") else { return nil }
        let suffix = normalized[marker.upperBound...]
        let digits = suffix.prefix { $0.isNumber }
        guard !digits.isEmpty else { return nil }
        return Int(digits)
    }

    private static func sysctlUInt64(named name: String) -> UInt64? {
        var value: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0,
              size == MemoryLayout<UInt64>.size else { return nil }
        return value
    }

    private static func sysctlString(named name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 1 else { return nil }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return nil }
        return String(
            decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) },
            as: UTF8.self)
    }
}

public enum TUFFCompatibilityIssue: Codable, Equatable, Sendable {
    case requiresRealModelValidation
    case insufficientUnifiedMemory(requiredBytes: UInt64, actualBytes: UInt64)
    case unsupportedMacOS(requiredMajorVersion: Int, actualMajorVersion: Int)
    case unsupportedAppleSilicon(requiredGeneration: Int, actualGeneration: Int)
    case contextExceedsSafeMemory(estimatedBytes: UInt64, budgetBytes: UInt64)
}

public struct TUFFModelCompatibility: Codable, Equatable, Sendable {
    public let issues: [TUFFCompatibilityIssue]

    public init(issues: [TUFFCompatibilityIssue]) {
        self.issues = issues
    }

    public var isCompatible: Bool { issues.isEmpty }
}

public struct TUFFModelRuntimeDefaults: Codable, Equatable, Sendable {
    public let contextTokens: Int
    public let expertCacheSlots: Int
    public let temperature: Double
    public let topK: Int
    public let topP: Double

    public init(contextTokens: Int = 8_192,
                expertCacheSlots: Int = 16,
                temperature: Double = 0.2,
                topK: Int = 64,
                topP: Double = 0.95) {
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
    }
}

public struct TUFFModelAddonDescriptor: Codable, Equatable, Sendable {
    public let id: String
    public let displayName: String
    public let kind: TUFFModelAddonKind
    public let source: TUFFModelSource
    public let hardware: TUFFModelHardwareRequirements

    public init(id: String,
                displayName: String,
                kind: TUFFModelAddonKind,
                source: TUFFModelSource,
                hardware: TUFFModelHardwareRequirements) {
        self.id = id
        self.displayName = displayName
        self.kind = kind
        self.source = source
        self.hardware = hardware
    }
}

public struct TUFFModelDescriptor: Codable, Equatable, Sendable, Identifiable {
    public let id: TUFFModelID
    /// Native checkpoint limits, without optional RoPE extrapolation.
    public var maximumContextTokens: Int {
        switch id {
        case .gemma4_E2B, .gemma4_E4B, .gptOss_20B, .gptOss_120B: 131_072
        case .minimaxM27: 204_800
        case .flint40B, .bonsai2_2B, .bonsai2_27B, .gemma4_12B_QAT, .gemma4_26B_A4B, .gemma4_26B_Heretic, .qwen36_35B_A3B, .qwen38FlashNext,
             .ornith15_9B, .ornith15_35B_A3B, .ornith15_122B, .ornith15_397B, .qwen38_27B: 262_144
        }
    }

    public let selector: String
    public let aliases: [String]
    public let apiModelID: String
    public let displayName: String
    public let shortName: String
    public let summary: String
    public let family: TUFFModelFamily
    public let architecture: TUFFArchitectureProfile
    public let installDirectoryName: String
    public let source: TUFFModelSource
    public let hardware: TUFFModelHardwareRequirements
    public let memory: TUFFModelMemoryProfile
    public let qualification: TUFFModelQualification
    public let maintenanceStatus: TUFFModelMaintenanceStatus
    public let runtimeDefaults: TUFFModelRuntimeDefaults
    public let capabilities: Set<TUFFModelCapability>
    public let reasoningControl: TUFFReasoningControl?
    public let addons: [TUFFModelAddonDescriptor]

    public init(id: TUFFModelID,
                selector: String,
                aliases: [String] = [],
                apiModelID: String,
                displayName: String,
                shortName: String,
                summary: String,
                family: TUFFModelFamily,
                architecture: TUFFArchitectureProfile,
                installDirectoryName: String,
                source: TUFFModelSource,
                hardware: TUFFModelHardwareRequirements,
                memory: TUFFModelMemoryProfile,
                qualification: TUFFModelQualification = .qualified,
                runtimeDefaults: TUFFModelRuntimeDefaults = TUFFModelRuntimeDefaults(),
                capabilities: Set<TUFFModelCapability>,
                reasoningControl: TUFFReasoningControl?,
                addons: [TUFFModelAddonDescriptor] = [],
                maintenanceStatus: TUFFModelMaintenanceStatus = .maintained) {
        self.id = id
        self.selector = selector
        self.aliases = aliases
        self.apiModelID = apiModelID
        self.displayName = displayName
        self.shortName = shortName
        self.summary = summary
        self.family = family
        self.architecture = architecture
        self.installDirectoryName = installDirectoryName
        self.source = source
        self.hardware = hardware
        self.memory = memory
        self.qualification = qualification
        self.maintenanceStatus = maintenanceStatus
        self.runtimeDefaults = runtimeDefaults
        self.capabilities = capabilities
        self.reasoningControl = reasoningControl
        self.addons = addons
    }

    private enum CodingKeys: String, CodingKey {
        case id, selector, aliases, apiModelID, displayName, shortName, summary
        case family, architecture, installDirectoryName, source, hardware, memory
        case qualification, maintenanceStatus, runtimeDefaults, capabilities, reasoningControl, addons
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let id = try values.decode(TUFFModelID.self, forKey: .id)
        let defaultMaintenance: TUFFModelMaintenanceStatus
        switch id {
        case .gptOss_20B, .gptOss_120B, .minimaxM27: defaultMaintenance = .legacy
        default: defaultMaintenance = .maintained
        }
        self.init(id: id,
            selector: try values.decode(String.self, forKey: .selector),
            aliases: try values.decode([String].self, forKey: .aliases),
            apiModelID: try values.decode(String.self, forKey: .apiModelID),
            displayName: try values.decode(String.self, forKey: .displayName),
            shortName: try values.decode(String.self, forKey: .shortName),
            summary: try values.decode(String.self, forKey: .summary),
            family: try values.decode(TUFFModelFamily.self, forKey: .family),
            architecture: try values.decode(TUFFArchitectureProfile.self, forKey: .architecture),
            installDirectoryName: try values.decode(String.self, forKey: .installDirectoryName),
            source: try values.decode(TUFFModelSource.self, forKey: .source),
            hardware: try values.decode(TUFFModelHardwareRequirements.self, forKey: .hardware),
            memory: try values.decode(TUFFModelMemoryProfile.self, forKey: .memory),
            qualification: try values.decode(TUFFModelQualification.self, forKey: .qualification),
            runtimeDefaults: try values.decode(TUFFModelRuntimeDefaults.self, forKey: .runtimeDefaults),
            capabilities: try values.decode(Set<TUFFModelCapability>.self, forKey: .capabilities),
            reasoningControl: try values.decodeIfPresent(TUFFReasoningControl.self, forKey: .reasoningControl),
            addons: try values.decode([TUFFModelAddonDescriptor].self, forKey: .addons),
            maintenanceStatus: try values.decodeIfPresent(TUFFModelMaintenanceStatus.self,
                forKey: .maintenanceStatus) ?? defaultMaintenance)
    }

    public var defaultSystemPrompt: String {
        "You are \(shortName), a helpful AI assistant. "
            + "You are running locally on a Mac in an app called TUFF."
    }

    /// Prefill chunk size for this model on `device`.
    ///
    /// Every chunk streams each layer's routed experts again, so chunk count
    /// drives SSD traffic, but only when the experts cannot stay in the page
    /// cache. Measured on a 16 GB M2 over a 7K-token prompt: Qwen 3.6 (19.5 GB
    /// installed) prefilled 2.2x faster at 2,048 than at 256, while Gemma 4
    /// 26B-A4B (14.3 GB) gained 3% for the same memory. So large chunks go
    /// only to a mixture-of-experts model bigger than the Mac's memory; one
    /// that fits gets a moderate chunk, and a dense model, which has no
    /// experts to re-read, keeps the size TUFF shipped with. Below 16 GB the
    /// large size is halved, since its scratch and KV ring come out of the
    /// same small budget as the expert cache.
    public func recommendedPrefillChunkTokens(on device: TUFFDeviceCapabilities) -> Int {
        guard architecture.feedForwardKind == .mixtureOfExperts else { return 256 }
        guard source.installedBytes > device.unifiedMemoryBytes else { return 512 }
        return device.unifiedMemoryBytes >= 16 * TUFFModelCatalog.oneGiB ? 2_048 : 1_024
    }

    public func compatibility(
        with device: TUFFDeviceCapabilities,
        contextTokens: Int? = nil,
        expertCacheSlots: Int? = nil
    ) -> TUFFModelCompatibility {
        var issues: [TUFFCompatibilityIssue] = []
        if qualification == .requiresRealModelValidation {
            issues.append(.requiresRealModelValidation)
        }
        if device.unifiedMemoryBytes < hardware.minimumUnifiedMemoryBytes {
            issues.append(.insufficientUnifiedMemory(
                requiredBytes: hardware.minimumUnifiedMemoryBytes,
                actualBytes: device.unifiedMemoryBytes))
        }
        if device.macOSMajorVersion < hardware.minimumMacOSMajorVersion {
            issues.append(.unsupportedMacOS(
                requiredMajorVersion: hardware.minimumMacOSMajorVersion,
                actualMajorVersion: device.macOSMajorVersion))
        }
        if device.appleSiliconGeneration < hardware.minimumAppleSiliconGeneration {
            issues.append(.unsupportedAppleSilicon(
                requiredGeneration: hardware.minimumAppleSiliconGeneration,
                actualGeneration: device.appleSiliconGeneration))
        }

        if let contextTokens {
            let estimate = memory.estimatedWorkingSetBytes(
                contextTokens: contextTokens,
                expertCacheSlots: expertCacheSlots ?? runtimeDefaults.expertCacheSlots)
            let budget = device.safeAppMemoryBudgetBytes
            if estimate > budget {
                issues.append(.contextExceedsSafeMemory(
                    estimatedBytes: estimate,
                    budgetBytes: budget))
            }
        }
        return TUFFModelCompatibility(issues: issues)
    }
}

public enum TUFFModelCatalog {
    public static let oneGiB: UInt64 = 1_073_741_824
    public static let eightGiB: UInt64 = 8 * oneGiB

    private static let gemmaSource = TUFFModelSource(
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        manifestModelID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: oneGiB)

    /// Pins verified against the live repository: `revision` is the commit the
    /// Hugging Face API reports for `main`, and `sourceIndexSHA256` is the
    /// SHA-256 of `model.safetensors.index.json` at that commit.
    ///
    /// The byte figures are the tensors this installer actually streams — the
    /// `language_model.*` ranges plus the tokenizer — not the whole file. The
    /// checkpoint also carries a vision tower and an audio tower that the text
    /// install skips, which is why the download is smaller than the repository.
    ///
    /// The method was checked by recomputing E4B's pinned fingerprint from its
    /// own repository: it reproduced `f8accac5…0093` exactly, and the byte sum
    /// landed within 400 KB of the figure pinned below.
    private static let gemmaE2BSource = TUFFModelSource(
        repoID: "mlx-community/gemma-4-e2b-it-4bit",
        revision: "238767527555cb75a05732a84dff5d6ba0dd6809",
        sourceIndexSHA256:
            "edb157dbf495e23f37377af4a628a9ad13c4ee7937f93ccb36ec9e9a19940f16",
        manifestModelID: "mlx-community/gemma-4-e2b-it-4bit",
        approximateDownloadBytes: 2_636_500_000,
        // Measured from a completed install: 2,636,263,860 bytes on disk. The
        // previous figure rounded *down* past that, so the free-space gate
        // asked for less than the install actually needs.
        installedBytes: 2_636_400_000,
        reserveBytes: oneGiB)

    private static let gemmaE4BSource = TUFFModelSource(
        repoID: "mlx-community/gemma-4-e4b-it-4bit",
        revision: "475b9088d29754a3379866cf5aeb6b41acd313c2",
        sourceIndexSHA256:
            "f8accac59ee7efe87e0c298c854610b262c3cadd477407503147c71209ff0093",
        manifestModelID: "mlx-community/gemma-4-e4b-it-4bit",
        approximateDownloadBytes: 4_231_600_000,
        installedBytes: 4_231_300_000,
        reserveBytes: oneGiB)

    private static let gemma12BQATSource = TUFFModelSource(
        repoID: "mlx-community/gemma-4-12B-it-qat-4bit",
        revision: "e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6",
        sourceIndexSHA256:
            "b87c93774de5d13ca9d0e21b045793e42e5df032fb5e7622212524f56f9695f2",
        manifestModelID: "mlx-community/gemma-4-12B-it-qat-4bit",
        approximateDownloadBytes: 10_947_021_920,
        installedBytes: 10_978_056_429,
        reserveBytes: oneGiB)

    private static let qwenSource = TUFFModelSource(
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256:
            "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        manifestModelID: "qwen3.6-35b-a3b-4bit",
        // The planner's own totalSourceBytes for this exact revision. The
        // previous figure understated the transfer by 617 MB.
        approximateDownloadBytes: 20_146_183_200,
        // Measured from a completed install: 19,551,402,438 bytes on disk,
        // which the previous planned figure understated by 4.9 MB.
        installedBytes: 19_551_500_000,
        reserveBytes: oneGiB)

    private static let gptOss20BSource = TUFFModelSource(
        repoID: "openai/gpt-oss-20b",
        revision: "6cee5e81ee83917806bbde320786a8fb61efebee",
        sourceIndexSHA256:
            "0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1",
        manifestModelID: "openai/gpt-oss-20b",
        approximateDownloadBytes: 13_790_000_000,
        installedBytes: 13_792_000_000,
        reserveBytes: oneGiB)

    private static let gptOss120BSource = TUFFModelSource(
        repoID: "openai/gpt-oss-120b",
        revision: "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        sourceIndexSHA256:
            "ede2655fdc05008561983b6e0829c600727c28d591e071077377059f03a6c00e",
        manifestModelID: "openai/gpt-oss-120b",
        approximateDownloadBytes: 65_300_000_000,
        installedBytes: 65_400_000_000,
        reserveBytes: oneGiB)

    /// Pinned against the live repository on 2026-09-06: `revision` is the
    /// commit the Hugging Face API reports for `main`, and `sourceIndexSHA256`
    /// is the SHA-256 of `model.safetensors.index.json` at that commit.
    ///
    /// Byte figures were computed from the checkpoint's own safetensors
    /// headers rather than estimated. The repository holds 103.86 GiB across
    /// 22 shards; a text install fetches only the `language_model.*` ranges,
    /// which coalesce at the 64 MiB chunk policy into 1,884 requests totalling
    /// 110,840,572,952 bytes — 220 MB of that being gaps between interleaved
    /// tensors. The 0.84 GiB vision tower is skipped. The installed figure adds
    /// the tokenizer sidecars and 16 KB page rounding on 24,576 routed expert
    /// blobs (192 MiB), whose 3,072,000-byte payload is not page-aligned.
    private static let qwen38FlashNextSource = TUFFModelSource(
        repoID: "mlx-community/Qwen3.8-Flash-Next-4bit",
        revision: "07b5dc6c54600a359b87f1e53e7adf6351c72a2c",
        sourceIndexSHA256:
            "3581f8d40a330d40009d0417359f5f75b7cf79e9f8fc48ba0b1461eabd43dd5f",
        manifestModelID: "mlx-community/Qwen3.8-Flash-Next-4bit",
        approximateDownloadBytes: 110_860_700_000,
        installedBytes: 110_850_000_000,
        reserveBytes: oneGiB)

    private static let minimaxM27Source = TUFFModelSource(
        repoID: "mlx-community/MiniMax-M2.7-4bit",
        revision: "66d2e5cb7c5cda05251b4625c504af4b034df7ff",
        sourceIndexSHA256:
            "8b2204b5a4741cb323a49d3ad6cfc5523c72c7c8ffa6e668a5627fb36ee13e52",
        manifestModelID: "mlx-community/MiniMax-M2.7-4bit",
        approximateDownloadBytes: 128_700_000_000,
        installedBytes: 128_700_000_000,
        reserveBytes: oneGiB)

    /// Small text-only launch model. Image and audio remain intentionally
    /// absent until separately packaged add-ons pass their own qualification.
    public static let gemma4_E2B = TUFFModelDescriptor(
        id: .gemma4_E2B,
        selector: "gemma4-e2b",
        aliases: ["e2b"],
        apiModelID: "gemma-4-e2b-it",
        displayName: "Gemma 4 E2B Q4",
        shortName: "Gemma 4 E2B",
        summary: "The smallest Gemma. Same architecture as E4B — per-layer embeddings and shared KV projections — at roughly half the weights.",
        family: .gemma4,
        architecture: .gemma4E2B,
        installDirectoryName: "gemma4-e2b.gturbo",
        source: gemmaE2BSource,
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        // Retain E4B's measured profile as a conservative compatibility
        // guardrail. E2B completed a pinned install and an 8K-context runtime
        // generation on the audit machine, but that single short run is not a
        // replacement for the cross-device qualification sweep that produced
        // the catalogue figures. E2B has fewer layers, KV heads and weights,
        // although its final 20 MLPs are wider than E4B's, so the old claim
        // that it was smaller in every dimension was not true.
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 1_833_438_160,
            defaultExpertCacheSlots: 0,
            expertCacheBytesPerSlot: 0,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 16_384,
                slidingAttentionBytesPerToken: 40_960,
                slidingWindowCapacityTokens: 512)),
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 8_192,
            expertCacheSlots: 16,
            temperature: 1.0,
            topK: 64,
            topP: 0.95),
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggle,
        addons: [TUFFModelAddonDescriptor(
            id: "gemma4-e2b-image-input",
            displayName: "Gemma 4 E2B Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: gemmaE2BSource.repoID,
                revision: gemmaE2BSource.revision,
                sourceIndexSHA256: gemmaE2BSource.sourceIndexSHA256,
                manifestModelID: gemmaE2BSource.manifestModelID,
                approximateDownloadBytes: 1_169_504_854,
                installedBytes: 336_950_725,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: eightGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let gemma4_E4B = TUFFModelDescriptor(
        id: .gemma4_E4B,
        selector: "gemma4-e4b",
        aliases: ["e4b"],
        apiModelID: "gemma-4-e4b-it",
        displayName: "Gemma 4 E4B Q4",
        shortName: "Gemma 4 E4B",
        summary: "Small dense Gemma with per-layer embeddings and shared KV projections.",
        family: .gemma4,
        architecture: .gemma4E4B,
        installDirectoryName: "gemma4-e4b.gturbo",
        source: gemmaE4BSource,
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 1_833_438_160,
            defaultExpertCacheSlots: 0,
            expertCacheBytesPerSlot: 0,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 16_384,
                slidingAttentionBytesPerToken: 40_960,
                slidingWindowCapacityTokens: 512)),
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 8_192,
            // Dense execution ignores expert residency, but the shared v1
            // runtime contract still requires a valid slot count.
            expertCacheSlots: 16,
            temperature: 1.0,
            topK: 64,
            topP: 0.95),
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggle,
        addons: [TUFFModelAddonDescriptor(
            id: "gemma4-e4b-image-input",
            displayName: "Gemma 4 E4B Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: gemmaE4BSource.repoID,
                revision: gemmaE4BSource.revision,
                sourceIndexSHA256: gemmaE4BSource.sourceIndexSHA256,
                manifestModelID: gemmaE4BSource.manifestModelID,
                // Coalesced range total for E4B's own layout, not E2B's plus
                // the payload delta: the two checkpoints interleave their
                // vision tensors differently, so E4B needs 31 requests where
                // E2B needs 23 and pulls 143 MB more of the gaps between them.
                approximateDownloadBytes: 1_313_596_274,
                installedBytes: 337_376_704,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: eightGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let gemma4_12B_QAT = TUFFModelDescriptor(
        id: .gemma4_12B_QAT,
        selector: "gemma4-12b-qat",
        aliases: ["12b", "12b-qat"],
        apiModelID: "gemma-4-12b-it-qat",
        displayName: "Gemma 4 12B QAT",
        shortName: "Gemma 4 12B QAT",
        summary: "Dense Gemma 4 Unified with 8-bit QAT MLPs and optional image input.",
        family: .gemma4,
        architecture: .gemma4_12B_QAT,
        installDirectoryName: "gemma4-12b-qat.gturbo",
        source: gemma12BQATSource,
        hardware: TUFFModelHardwareRequirements(
            minimumUnifiedMemoryBytes: 16 * oneGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 6_000_000_000,
            defaultExpertCacheSlots: 0,
            expertCacheBytesPerSlot: 0,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 16_384,
                slidingAttentionBytesPerToken: 327_680,
                slidingWindowCapacityTokens: 1_024)),
        // Qualified with a complete pinned install plus real text and image
        // generation on a 16 GB Apple-silicon Mac. Keep the 6 GB catalog
        // estimate conservative: file-backed Metal mappings are not fully
        // represented by the process RSS reported by `time`.
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 8_192,
            expertCacheSlots: 16,
            temperature: 1.0,
            topK: 64,
            topP: 0.95),
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggle,
        addons: [TUFFModelAddonDescriptor(
            id: "gemma4-12b-qat-image-input",
            displayName: "Gemma 4 12B QAT Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: gemma12BQATSource.repoID,
                revision: gemma12BQATSource.revision,
                sourceIndexSHA256: gemma12BQATSource.sourceIndexSHA256,
                manifestModelID: gemma12BQATSource.manifestModelID,
                approximateDownloadBytes: 102_556_672,
                installedBytes: 40_581_222,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: 16 * oneGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let gemma4_26B_A4B = TUFFModelDescriptor(
        id: .gemma4_26B_A4B,
        selector: "gemma4",
        apiModelID: "gemma-4-26b-a4b-it",
        displayName: "Gemma 4 26B-A4B Q4",
        shortName: "Gemma 4 26B-A4B",
        summary: "26B total, 3.9B active. A balanced Gemma model with optional image input.",
        family: .gemma4,
        architecture: .gemma4_26B_A4B,
        installDirectoryName: "gemma4.gturbo",
        source: gemmaSource,
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 2_254_857_830,
            expertCacheBytesPerSlot: 100_761_600,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 20_480,
                slidingAttentionBytesPerToken: 204_800,
                slidingWindowCapacityTokens: 1_304)),
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggle,
        addons: [TUFFModelAddonDescriptor(
            id: "gemma4-image-input",
            displayName: "Gemma 4 Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: gemmaSource.repoID,
                revision: gemmaSource.revision,
                sourceIndexSHA256: gemmaSource.sourceIndexSHA256,
                manifestModelID: gemmaSource.manifestModelID,
                approximateDownloadBytes: 1_539_478_890,
                installedBytes: 1_148_567_552,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: eightGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let qwen36_35B_A3B = TUFFModelDescriptor(
        id: .qwen36_35B_A3B,
        selector: "qwen36",
        apiModelID: "qwen3.6-35b-a3b",
        displayName: "Qwen 3.6 35B-A3B Q4",
        shortName: "Qwen 3.6 35B-A3B",
        summary: "35B total, 3B active. Hybrid linear/full attention, so its "
            + "KV cache stays small at long context; larger install.",
        family: .qwen36,
        architecture: .qwen36_35B_A3B,
        installDirectoryName: "qwen36.gturbo",
        source: qwenSource,
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 1_610_612_736,
            expertCacheBytesPerSlot: 70_778_880,
            kvCache: TUFFKVCacheProfile(fullAttentionBytesPerToken: 20_480)),
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggleWithPreservation,
        addons: [TUFFModelAddonDescriptor(
            id: "qwen36-image-input",
            displayName: "Qwen3.6 Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: qwenSource.repoID,
                revision: qwenSource.revision,
                sourceIndexSHA256: qwenSource.sourceIndexSHA256,
                manifestModelID: qwenSource.manifestModelID,
                // The coalesced range total, not the vision payload sum. Qwen
                // interleaves its 333 vision tensors through shard 1 across
                // 129 gaps, so fetching them costs 245 MB more than the
                // tensors themselves weigh.
                approximateDownloadBytes: 1_137_999_008,
                installedBytes: 900_808_704,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: eightGiB,
                minimumAppleSiliconGeneration: 2))])

    /// Official Harmony checkpoint, qualified at 4K context on a 16 GB M2.
    public static let gptOss_20B = TUFFModelDescriptor(
        id: .gptOss_20B,
        selector: "gpt-oss-20b",
        aliases: ["gpt-oss"],
        apiModelID: "gpt-oss-20b",
        displayName: "GPT-OSS 20B Q4",
        shortName: "GPT-OSS 20B",
        summary: "Official MXFP4 checkpoint with Harmony reasoning and tool calls.",
        family: .gptOss,
        architecture: .gptOss_20B,
        installDirectoryName: "gpt-oss-20b.gturbo",
        source: gptOss20BSource,
        hardware: TUFFModelHardwareRequirements(
            minimumUnifiedMemoryBytes: 16 * oneGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 5_487_695_296,
            defaultContextTokens: 4_096,
            defaultExpertCacheSlots: 4,
            expertCacheBytesPerSlot: 317_718_528,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 24_576,
                slidingAttentionBytesPerToken: 24_576,
                slidingWindowCapacityTokens: 384)),
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 4_096,
            expertCacheSlots: 4,
            temperature: 1.0,
            topK: 0,
            topP: 1.0),
        capabilities: [.textGeneration, .reasoning],
        reasoningControl: .graded,
        maintenanceStatus: .legacy)

    /// Official Harmony checkpoint, qualified at 4K context on a 16 GB M2.
    /// Streamed experts keep the full 61 GiB installation out of the working
    /// set while an FP32 residual stream prevents late-layer overflow.
    public static let gptOss_120B = TUFFModelDescriptor(
        id: .gptOss_120B,
        selector: "gpt-oss-120b",
        apiModelID: "gpt-oss-120b",
        displayName: "GPT-OSS 120B Q4",
        shortName: "GPT-OSS 120B",
        summary: "Maximum-quality GPT-OSS checkpoint with streamed MXFP4 experts.",
        family: .gptOss,
        architecture: .gptOss_120B,
        installDirectoryName: "gpt-oss-120b.gturbo",
        source: gptOss120BSource,
        hardware: TUFFModelHardwareRequirements(
            minimumUnifiedMemoryBytes: 16 * oneGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 7_990_582_952,
            defaultContextTokens: 4_096,
            defaultExpertCacheSlots: 4,
            expertCacheBytesPerSlot: 476_577_792,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 36_864,
                slidingAttentionBytesPerToken: 36_864,
                slidingWindowCapacityTokens: 384)),
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 4_096,
            expertCacheSlots: 4,
            temperature: 1.0,
            topK: 0,
            topP: 1.0),
        capabilities: [.textGeneration, .reasoning],
        reasoningControl: .graded,
        maintenanceStatus: .legacy)

    /// MiniMax M2.7 streamed from the pinned MLX 4-bit checkpoint. Its 229B
    /// total parameters remain file-backed; only the selected eight experts
    /// per layer enter the bounded cache.
    public static let minimaxM27 = TUFFModelDescriptor(
        id: .minimaxM27,
        selector: "minimax-m2.7",
        aliases: ["minimax", "m2.7"],
        apiModelID: "minimax-m2.7",
        displayName: "MiniMax M2.7 Q4",
        shortName: "MiniMax M2.7",
        summary: "229B total with eight active experts, streamed for 16 GB Macs.",
        family: .minimaxM2,
        architecture: .minimaxM27,
        installDirectoryName: "minimax-m2.7.gturbo",
        source: minimaxM27Source,
        hardware: TUFFModelHardwareRequirements(
            minimumUnifiedMemoryBytes: 16 * oneGiB,
            minimumAppleSiliconGeneration: 2),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 11_250_000_000,
            defaultContextTokens: 4_096,
            defaultExpertCacheSlots: 16,
            expertCacheBytesPerSlot: 493_682_688,
            kvCache: TUFFKVCacheProfile(fullAttentionBytesPerToken: 253_952)),
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 4_096,
            expertCacheSlots: 16,
            temperature: 1.0,
            topK: 40,
            topP: 0.95),
        capabilities: [.textGeneration, .reasoning],
        reasoningControl: .alwaysOn,
        maintenanceStatus: .legacy)

    /// Qwen3.8 Flash Next, the `qwen4_exp` architecture: 48 decoder layers of
    /// which 36 are gated-DeltaNet linear attention and 12 are full attention,
    /// 512 routed experts per layer with 10 active, a sigmoid-gated shared
    /// expert, four-stream hyper-connections in place of a single residual,
    /// and an n-gram per-layer-embedding table on layer 1.
    ///
    /// The checkpoint splits well for streaming. Of its 103.02 GiB of
    /// `language_model` tensors, 70.31 GiB are routed experts and 29.80 GiB
    /// are the n-gram table, leaving 2.91 GiB that has to stay resident. Only
    /// 12 layers keep a KV cache, at 2 KV heads and a 256 head dim, so a token
    /// costs 24 KiB of cache against MiniMax M2.7's 248 KiB.
    ///
    /// The validation run is done: the checkpoint installs, loads and decodes,
    /// and every stage of a forward pass was compared against mlx-vlm's own
    /// `qwen4_exp` on the real weights, layer by layer. On the 16 GB M2 it
    /// answers at ~2.6 tok/s.
    ///
    /// The original 2K qualification remains the memory-estimate baseline.
    /// The runner now implements the compressed-block QSA indexer required
    /// above that threshold; the native checkpoint limit is 262,144 tokens.
    public static let qwen38FlashNext = TUFFModelDescriptor(
        id: .qwen38FlashNext,
        selector: "qwen38-flash-next",
        aliases: ["flash-next", "qwen-flash"],
        apiModelID: "qwen3.8-flash-next",
        displayName: "Qwen 3.8 Flash Next Q4",
        shortName: "Qwen 3.8 Flash Next",
        summary: "512 experts per layer with 10 active, hybrid linear "
            + "attention, and a 24 KiB-per-token KV cache.",
        family: .qwen4Exp,
        architecture: .qwen38FlashNext,
        installDirectoryName: "qwen38-flash-next.gturbo",
        source: qwen38FlashNextSource,
        hardware: TUFFModelHardwareRequirements(
            minimumUnifiedMemoryBytes: 16 * oneGiB,
            minimumAppleSiliconGeneration: 2),
        // The parts that are counted rather than estimated: 3,123,406,360
        // bytes of resident weights and 119,144,448 bytes of gated-DeltaNet
        // recurrent and convolution state, both summed from the checkpoint's
        // safetensors headers; 50,331,648 bytes of KV cache at 2K; and
        // 4,731,174,912 bytes of routed-expert cache. The remainder is an
        // allowance for scratch, logits and staging buffers.
        //
        // Earlier long-answer measurements slightly favored 32 slots. A 512-expert layer argues
        // for a large cache — 32 slots hold 6.3% of one — but each slot is its
        // own Metal buffer and a command buffer that references it pays for it
        // whether or not the token routed there. On a 160-token answer:
        //
        //     16 slots  2.711 tok/s
        //     32 slots  2.842 tok/s
        //     48 slots  1.633 tok/s
        //
        // 48 was the first choice here, from the hit-rate argument alone. It
        // is 1.7x slower than 32 and costs 2.4 GB more, which is the same
        // trade `AppAutomaticMemoryPlanner` documents for Gemma 4 26B-A4B.
        // 9.0 starts at 16: the earlier 16/32 results are within the 5% tie
        // threshold, while new 16 GB tests show 32 can cause swap writes.
        // Calibration may still select a larger cache with safe evidence.
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 8_634_412_544,
            defaultContextTokens: 2_048,
            defaultExpertCacheSlots: 32,
            expertCacheBytesPerSlot: 147_849_216,
            kvCache: TUFFKVCacheProfile(fullAttentionBytesPerToken: 28_560)),
        qualification: .qualified,
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 2_048,
            expertCacheSlots: 16,
            temperature: 1.0,
            topK: 20,
            topP: 0.95),
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggleWithPreservation,
        // The tower is Qwen 3.6's, tensor for tensor: 27 blocks at hidden
        // 1152, patch 16, spatial merge 2, the same 333 tensors and the same
        // `<|vision_start|>` / `<|image_pad|>` / `<|vision_end|>` ids. Only the
        // merger's output differs, at 2,560 wide instead of 2,048 — which is
        // exactly the 4,719,616 bytes by which this pack exceeds Qwen 3.6's.
        addons: [TUFFModelAddonDescriptor(
            id: "qwen38-flash-next-image-input",
            displayName: "Qwen3.8 Flash Next Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: qwen38FlashNextSource.repoID,
                revision: qwen38FlashNextSource.revision,
                sourceIndexSHA256: qwen38FlashNextSource.sourceIndexSHA256,
                manifestModelID: qwen38FlashNextSource.manifestModelID,
                // The coalesced range total, not the tensor sum: the 333
                // vision tensors are interleaved through shard 1, so fetching
                // them costs more than they weigh.
                approximateDownloadBytes: 1_058_209_792,
                installedBytes: 901_332_992,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: 16 * oneGiB,
                minimumAppleSiliconGeneration: 2))])


    public static let ornith15_9B = TUFFModelDescriptor(
        id: .ornith15_9B, selector: "ornith-9b", apiModelID: "ornith-9b",
        displayName: "Ornith 1.5 9B Q4", shortName: "Ornith 1.5 9B",
        summary: "Dense 9B text model. Auto retains whole layers when they fit and streams the remainder.", family: .qwen36, architecture: .ornith15_9B,
        installDirectoryName: "ornith-9b.gturbo",
        source: TUFFModelSource(repoID: "ornith-ai/Ornith-1.5-9B-MLX-4bit", revision: "a48173b246ac705be75c05bedf1a0666db522d53",
            sourceIndexSHA256: "215f9abadc1cea4d465b01c036dffcac5a84c2fb2386777863eff6928b91ed0b", manifestModelID: "ornith-ai/Ornith-1.5-9B-MLX-4bit",
            approximateDownloadBytes: 5100000000, installedBytes: 5100000000, reserveBytes: oneGiB),
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        memory: TUFFModelMemoryProfile(qualifiedDefaultWorkingSetBytes: 1400000000,
            expertCacheBytesPerSlot: 0, kvCache: TUFFKVCacheProfile(fullAttentionBytesPerToken: 32768)),
        qualification: .qualified,
        runtimeDefaults: TUFFModelRuntimeDefaults(temperature: 0.7, topK: 20, topP: 0.8),
        capabilities: [.textGeneration, .imageInput, .reasoning], reasoningControl: .toggleWithPreservation,
        // The MLX conversion left the vision tower out, so the pack comes from
        // Ornith's own BF16 release of the same model.
        addons: [TUFFModelAddonDescriptor(
            id: "ornith-9b-image-input",
            displayName: "Ornith 1.5 9B Image Support",
            kind: .imageInput,
            source: TUFFModelSource(repoID: "ornith-ai/Ornith-1.5-9B",
                revision: "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
                sourceIndexSHA256: "d5c7fee99574e9a05f901282aee04fc4fc3dccf094a659df48c6b8e9f39109c3",
                manifestModelID: "ornith-ai/Ornith-1.5-9B",
                approximateDownloadBytes: 912_020_960,
                installedBytes: 912_020_960,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: 16 * oneGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let ornith15_35B_A3B = TUFFModelDescriptor(
        id: .ornith15_35B_A3B, selector: "ornith-35b", apiModelID: "ornith-35b",
        displayName: "Ornith 1.5 35B-A3B Q4", shortName: "Ornith 1.5 35B-A3B",
        summary: "35B total, 3B active. Streams routed experts with a bounded cache.", family: .qwen36, architecture: .ornith15_35B_A3B,
        installDirectoryName: "ornith-35b.gturbo",
        source: TUFFModelSource(repoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit", revision: "19504d912fa8fc7622bf6b1de3db5d5d890b1f02",
            sourceIndexSHA256: "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36", manifestModelID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
            approximateDownloadBytes: 19600000000, installedBytes: 19800000000, reserveBytes: oneGiB),
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        memory: TUFFModelMemoryProfile(qualifiedDefaultWorkingSetBytes: 2000000000,
            expertCacheBytesPerSlot: 70778880, kvCache: TUFFKVCacheProfile(fullAttentionBytesPerToken: 20480)),
        qualification: .qualified,
        runtimeDefaults: TUFFModelRuntimeDefaults(temperature: 0.7, topK: 20, topP: 0.8),
        capabilities: [.textGeneration, .imageInput, .reasoning], reasoningControl: .toggleWithPreservation,
        // As with 9B, the vision tower is only in Ornith's BF16 release.
        addons: [TUFFModelAddonDescriptor(
            id: "ornith-35b-image-input",
            displayName: "Ornith 1.5 35B-A3B Image Support",
            kind: .imageInput,
            source: TUFFModelSource(repoID: "ornith-ai/Ornith-1.5-35B-A3B",
                revision: "10fbf86fed7ecee4a061f8b499a618f46001cac1",
                sourceIndexSHA256: "0ad9fbfbb1a7514773ed881d3c9403019bd7ae9ff4c6af3cfc65ca7da784d78f",
                manifestModelID: "ornith-ai/Ornith-1.5-35B-A3B",
                approximateDownloadBytes: 893_142_496,
                installedBytes: 893_142_496,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: 16 * oneGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let ornith15_122B = TUFFModelDescriptor(
        id: .ornith15_122B, selector: "ornith-122b", apiModelID: "ornith-122b",
        displayName: "Ornith 1.5 122B Q4", shortName: "Ornith 1.5 122B",
        summary: "122B MoE, about 10B active. Streams routed experts with a bounded cache.",
        family: .qwen36, architecture: .ornith15_122B, installDirectoryName: "ornith-122b.gturbo",
        // The index also names a model-vision.safetensors that was never
        // uploaded. The installer drops those entries for this pinned index.
        source: TUFFModelSource(repoID: "ornith-ai/Ornith-1.5-122B-MLX-4bit",
            revision: "8640002ba242524ee7288ead4a3454dd919e8526",
            sourceIndexSHA256: "deea07c02a2b760c991d27e5646c3c42bae8d5fa8a4a63be349f95655485f3a1",
            manifestModelID: "ornith-ai/Ornith-1.5-122B-MLX-4bit",
            approximateDownloadBytes: 68_740_000_000, installedBytes: 68_800_000_000, reserveBytes: oneGiB),
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: 16 * oneGiB),
        // Summed from the safetensors headers: 3,479,860,224 bytes of resident
        // weights, 156,303,360 of gated-DeltaNet state, 50,331,648 of KV cache
        // at 2K and 4,076,863,488 for 16 expert slots, plus the scratch
        // allowance Flash Next measured.
        memory: TUFFModelMemoryProfile(qualifiedDefaultWorkingSetBytes: 8_400_000_000,
            defaultContextTokens: 2_048, expertCacheBytesPerSlot: 254_803_968, kvCache: TUFFKVCacheProfile(fullAttentionBytesPerToken: 24_576)),
        qualification: .qualified,
        runtimeDefaults: TUFFModelRuntimeDefaults(contextTokens: 2_048, expertCacheSlots: 16,
            temperature: 0.7, topK: 20, topP: 0.8),
        capabilities: [.textGeneration, .imageInput, .reasoning], reasoningControl: .toggleWithPreservation,
        // The MLX conversion's vision shard is missing, so the pack comes from
        // Ornith's own BF16 release, as with 9B and 35B.
        addons: [TUFFModelAddonDescriptor(
            id: "ornith-122b-image-input",
            displayName: "Ornith 1.5 122B Image Support",
            kind: .imageInput,
            source: TUFFModelSource(repoID: "ornith-ai/Ornith-1.5-122B",
                revision: "7681ef608624e7532d8164128bcffdbdbaf1b1af",
                sourceIndexSHA256: "54f60cfac525650a60c4ff0ac034624cce71e94ff0535ca77c9c5e2b7d0f7686",
                manifestModelID: "ornith-ai/Ornith-1.5-122B",
                approximateDownloadBytes: 902_581_728,
                installedBytes: 902_581_728,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: 16 * oneGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let qwen38_27B = TUFFModelDescriptor(
        id: .qwen38_27B, selector: "qwen38-27b", apiModelID: "qwen38-27b",
        displayName: "Qwen 3.8 27B Q4", shortName: "Qwen 3.8 27B",
        summary: "Dense 27B model. Bounded layer streaming supports smaller memory budgets; disk speed limits decode.", family: .qwen36, architecture: .qwen38_27B,
        installDirectoryName: "qwen38-27b.gturbo",
        source: TUFFModelSource(repoID: "mlx-community/Qwen3.8-27B-4bit", revision: "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
            sourceIndexSHA256: "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8", manifestModelID: "mlx-community/Qwen3.8-27B-4bit",
            approximateDownloadBytes: 16200000000, installedBytes: 15500000000, reserveBytes: oneGiB),
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        memory: TUFFModelMemoryProfile(qualifiedDefaultWorkingSetBytes: 1400000000,
            expertCacheBytesPerSlot: 0, kvCache: TUFFKVCacheProfile(fullAttentionBytesPerToken: 65536)),
        qualification: .qualified,
        runtimeDefaults: TUFFModelRuntimeDefaults(temperature: 0.7, topK: 20, topP: 0.8),
        capabilities: [.textGeneration, .imageInput, .reasoning], reasoningControl: .toggleWithPreservation,
        // The checkpoint carries Qwen 3.6's tower, tensor for tensor, with the
        // merger widened to this model's 5,120-wide residual.
        addons: [TUFFModelAddonDescriptor(
            id: "qwen38-27b-image-input",
            displayName: "Qwen 3.8 27B Image Support",
            kind: .imageInput,
            source: TUFFModelSource(repoID: "mlx-community/Qwen3.8-27B-4bit",
                revision: "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
                sourceIndexSHA256: "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8",
                manifestModelID: "mlx-community/Qwen3.8-27B-4bit",
                // The coalesced range total: the 333 vision tensors are spread
                // through the first shard, so fetching them costs about twice
                // what they weigh.
                approximateDownloadBytes: 1_929_719_232,
                installedBytes: 925_014_699,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: 16 * oneGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let ornith15_397B = TUFFModelDescriptor(
        id: .ornith15_397B, selector: "ornith-397b", apiModelID: "ornith-397b",
        displayName: "Ornith 1.5 397B Q4", shortName: "Ornith 1.5 397B",
        summary: "397B MoE. Streams routed experts and retains only the projection layers that fit. Requires a large local install.",
        family: .qwen36, architecture: .ornith15_397B, installDirectoryName: "ornith-397b.gturbo",
        source: TUFFModelSource(repoID: "gratex/Ornith-1.5-397B-W4A16-g32-sym",
            revision: "62fbf868ef177baf41f47c3e0001699a4ec54f02",
            sourceIndexSHA256: "99e7ee7bbbf6f18184967b4be9a118f63c16c2fcbbe0bb8876088eaa9f03c177",
            manifestModelID: "gratex/Ornith-1.5-397B-W4A16-g32-sym",
            approximateDownloadBytes: 239_000_000_000, installedBytes: 249_000_000_000, reserveBytes: oneGiB),
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: 16 * oneGiB),
        memory: TUFFModelMemoryProfile(qualifiedDefaultWorkingSetBytes: 8_000_000_000,
            expertCacheBytesPerSlot: 471_859_200, kvCache: TUFFKVCacheProfile(fullAttentionBytesPerToken: 30_720)),
        qualification: .qualified,
        runtimeDefaults: TUFFModelRuntimeDefaults(contextTokens: 2_048, expertCacheSlots: 16,
            temperature: 0.7, topK: 20, topP: 0.8),
        capabilities: [.textGeneration, .imageInput, .reasoning], reasoningControl: .toggleWithPreservation,
        // The checkpoint keeps its BF16 vision tower beside the 4-bit weights.
        addons: [TUFFModelAddonDescriptor(
            id: "ornith-397b-image-input",
            displayName: "Ornith 1.5 397B Image Support",
            kind: .imageInput,
            source: TUFFModelSource(repoID: "gratex/Ornith-1.5-397B-W4A16-g32-sym",
                revision: "62fbf868ef177baf41f47c3e0001699a4ec54f02",
                sourceIndexSHA256: "99e7ee7bbbf6f18184967b4be9a118f63c16c2fcbbe0bb8876088eaa9f03c177",
                manifestModelID: "gratex/Ornith-1.5-397B-W4A16-g32-sym",
                approximateDownloadBytes: 912_020_960,
                installedBytes: 912_020_960,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: 16 * oneGiB,
                minimumAppleSiliconGeneration: 2))])

    /// Community abliterated derivative. It shares Gemma's architecture and
    /// engine, but never shares a checkpoint identity, install or settings.
    public static let gemma4_26B_Heretic = TUFFModelDescriptor(
        id: .gemma4_26B_Heretic, selector: "gemma4-heretic", apiModelID: "gemma4-heretic",
        displayName: "Gemma 4 26B-A4B Heretic Q4", shortName: "Gemma 4 26B-A4B Heretic",
        summary: "An abliterated Gemma variant for text and images, converted by mlx-community.",
        family: .gemma4, architecture: .gemma4_26B_A4B,
        installDirectoryName: "gemma4-heretic.gturbo",
        source: TUFFModelSource(repoID: "mlx-community/gemma-4-26B-A4B-it-heretic-4bit",
            revision: "1f1ae59040a78b1d717618f44c82757255925d26",
            sourceIndexSHA256: "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80",
            manifestModelID: "mlx-community/gemma-4-26B-A4B-it-heretic-4bit",
            approximateDownloadBytes: 15_000_000_000, installedBytes: 15_000_000_000,
            reserveBytes: oneGiB),
        hardware: gemma4_26B_A4B.hardware,
        memory: TUFFModelMemoryProfile(qualifiedDefaultWorkingSetBytes: 2_600_000_000,
            expertCacheBytesPerSlot: 100_761_600, kvCache: gemma4_26B_A4B.memory.kvCache),
        qualification: .qualified,
        runtimeDefaults: gemma4_26B_A4B.runtimeDefaults,
        capabilities: [.textGeneration, .imageInput, .reasoning], reasoningControl: .toggle,
        addons: [TUFFModelAddonDescriptor(
            id: "gemma4-heretic-image-input", displayName: "Gemma Heretic Image Support", kind: .imageInput,
            source: TUFFModelSource(repoID: "mlx-community/gemma-4-26B-A4B-it-heretic-4bit",
                revision: "1f1ae59040a78b1d717618f44c82757255925d26",
                sourceIndexSHA256: "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80",
                manifestModelID: "mlx-community/gemma-4-26B-A4B-it-heretic-4bit",
                approximateDownloadBytes: 1_580_286_570, installedBytes: 1_148_567_552,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB,
                minimumAppleSiliconGeneration: 2))])

    /// Whether an image pack read from the checkpoint whose index hashes to
    /// `visionSourceIndexSHA256` may serve the text model built from
    /// `textSourceSnapshotHash` (`sha256:<hex>`). Normally both are one
    /// checkpoint. A catalog add-on can name another one, pinned by hash, when
    /// the text conversion dropped the vision tower.
    public static func imagePackSourceMatches(textSourceSnapshotHash: String,
                                              visionSourceIndexSHA256: String) -> Bool {
        if textSourceSnapshotHash == "sha256:\(visionSourceIndexSHA256)" { return true }
        return all.contains { model in
            "sha256:\(model.source.sourceIndexSHA256)" == textSourceSnapshotHash
                && model.addons.contains {
                    $0.kind == .imageInput && $0.source.sourceIndexSHA256 == visionSourceIndexSHA256
                }
        }
    }

    /// The publisher INT4 source remains available for local pack identification;
    /// the advertised Flint descriptor uses the pinned OptiQ checkpoint below.
    public static let flint40B = TUFFModelDescriptor(
        id: .flint40B, selector: "flint-40b", apiModelID: "flint-40b",
        displayName: "Recursive Flint 40B A2.5B", shortName: "Flint 40B",
        summary: "40B mixture-of-experts model with streamed INT4 experts.",
        family: .flint, architecture: .flint40B, installDirectoryName: "flint-40b.gturbo",
        source: .init(repoID: "RecursiveSI/flint-int4", revision: "3b93f9053b530488405ae9fca51d4900d6ce21f0",
            sourceIndexSHA256: "319822f10e32a5916b57e23c7f05f9e92b24d02b3a136d02a3846b5edab78109",
            manifestModelID: "RecursiveSI/flint-int4", approximateDownloadBytes: 23_000_000_000,
            installedBytes: 23_000_000_000, reserveBytes: oneGiB),
        hardware: .init(minimumUnifiedMemoryBytes: 16 * oneGiB),
        memory: .init(qualifiedDefaultWorkingSetBytes: 5 * oneGiB, expertCacheBytesPerSlot: 80_000_000,
            kvCache: .init(fullAttentionBytesPerToken: 28_672, slidingAttentionBytesPerToken: 71_680, slidingWindowCapacityTokens: 513)),
        qualification: .requiresRealModelValidation, capabilities: [.textGeneration, .reasoning], reasoningControl: .toggle)

    /// Pinned Flint OptiQ checkpoint (group-64 affine; 8-bit routed experts in
    /// layers 2-12, 4-bit elsewhere). The largest layer slot is 99.4 MB.
    public static let flint40BOptiQ = TUFFModelDescriptor(
        id: .flint40B, selector: "flint-40b", apiModelID: "flint-40b",
        displayName: "Recursive Flint 40B A2.5B", shortName: "Flint 40B",
        summary: "40B mixture-of-experts model with streamed 4/8-bit OptiQ experts.",
        family: .flint, architecture: .flint40B, installDirectoryName: "flint-40b.gturbo",
        source: .init(repoID: "mlx-community/flint-OptiQ-4bit", revision: "d5efe414b30ee0cf2042ec712321a6fa060c2a53",
            sourceIndexSHA256: "54bd24dfdb836f9154da82f2273cc52274a230b74c9996379357285d72330fd0",
            manifestModelID: "mlx-community/flint-OptiQ-4bit", approximateDownloadBytes: 25_600_000_000,
            installedBytes: 27_500_000_000, reserveBytes: oneGiB),
        hardware: .init(minimumUnifiedMemoryBytes: 16 * oneGiB),
        memory: .init(qualifiedDefaultWorkingSetBytes: 5 * oneGiB, expertCacheBytesPerSlot: 100_000_000,
            kvCache: .init(fullAttentionBytesPerToken: 28_672, slidingAttentionBytesPerToken: 71_680, slidingWindowCapacityTokens: 513)),
        qualification: .qualified,
        runtimeDefaults: TUFFModelRuntimeDefaults(expertCacheSlots: 16, temperature: 1.0, topK: 20, topP: 0.95),
        capabilities: [.textGeneration, .reasoning], reasoningControl: .toggle)

    public static let bonsai2_2B = TUFFModelDescriptor(
        id: .bonsai2_2B, selector: "bonsai-2-2b", apiModelID: "bonsai-2-2b",
        displayName: "Ternary Bonsai 2 2B", shortName: "Bonsai 2 2B",
        summary: "Text-only 2B model using Q2 Hadamard projections.",
        family: .qwen36, architecture: .bonsai2_2B, installDirectoryName: "bonsai-2-2b.gturbo",
        source: .init(repoID: "prism-ml/Ternary-Bonsai-2-2B-mlx-2bit", revision: "1c6cbae21f8c38830ce337972b46fdae7b18eaaa",
            sourceIndexSHA256: "da5095b9f34d7f0d5094090afeeadf124eef821c75136dc28957ce586ae7ac0c",
            manifestModelID: "prism-ml/Ternary-Bonsai-2-2B-mlx-2bit", approximateDownloadBytes: 1_220_000_000,
            installedBytes: 600_000_000, reserveBytes: oneGiB),
        hardware: .init(minimumUnifiedMemoryBytes: eightGiB),
        memory: .init(qualifiedDefaultWorkingSetBytes: 2 * oneGiB, expertCacheBytesPerSlot: 0,
            kvCache: .init(fullAttentionBytesPerToken: 12_288)),
        qualification: .qualified,
        runtimeDefaults: TUFFModelRuntimeDefaults(temperature: 0.7, topK: 20, topP: 0.8),
        capabilities: [.textGeneration, .reasoning], reasoningControl: .toggleWithPreservation)

    public static let bonsai2_27B = TUFFModelDescriptor(
        id: .bonsai2_27B, selector: "bonsai-2-27b", apiModelID: "bonsai-2-27b",
        displayName: "Ternary Bonsai 2 27B", shortName: "Bonsai 2 27B",
        summary: "Text-only 27B model using Q2 Hadamard projections.",
        family: .qwen36, architecture: .bonsai2_27B, installDirectoryName: "bonsai-2-27b.gturbo",
        source: .init(repoID: "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", revision: "fcba37d2117a7077eac6b613b2668d14d9779edd",
            sourceIndexSHA256: "130de5925082c168b7866b2e91b52e44abbafc99017e3ca352b77b5b55a269ed",
            manifestModelID: "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", approximateDownloadBytes: 8_620_000_000,
            installedBytes: 7_800_000_000, reserveBytes: oneGiB),
        hardware: .init(minimumUnifiedMemoryBytes: 16 * oneGiB),
        memory: .init(qualifiedDefaultWorkingSetBytes: 10 * oneGiB, expertCacheBytesPerSlot: 0,
            kvCache: .init(fullAttentionBytesPerToken: 65_536)),
        qualification: .qualified,
        runtimeDefaults: TUFFModelRuntimeDefaults(temperature: 0.7, topK: 20, topP: 0.8),
        capabilities: [.textGeneration, .reasoning], reasoningControl: .toggleWithPreservation)

    public static let all: [TUFFModelDescriptor] = [
        gemma4_E2B,
        gemma4_E4B,
        gemma4_12B_QAT,
        gemma4_26B_A4B,
        qwen36_35B_A3B,
        gptOss_20B,
        gptOss_120B,
        minimaxM27,
        qwen38FlashNext,
        ornith15_9B,
        ornith15_35B_A3B,
        ornith15_122B,
        ornith15_397B,
        qwen38_27B,
        gemma4_26B_Heretic,
        flint40BOptiQ,
        bonsai2_2B,
        bonsai2_27B,
    ]
    public static let `default` = gemma4_26B_A4B

    public static func model(id: TUFFModelID) -> TUFFModelDescriptor? {
        all.first { $0.id == id }
    }

    public static func model(selector: String) -> TUFFModelDescriptor? {
        all.first { $0.selector == selector || $0.aliases.contains(selector) }
    }

    /// Resolve reviewed local packs for memory admission without advertising
    /// gated models in the install catalog.
    public static func model(manifestModelID: String) -> TUFFModelDescriptor? {
        (all + [flint40B]).first {
            $0.source.manifestModelID == manifestModelID
        }
    }
}
