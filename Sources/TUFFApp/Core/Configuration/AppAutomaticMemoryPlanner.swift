import TUFFRuntimeDescriptions
import TUFFModelCatalog
import TUFFInferenceTypes
import TUFFModelMetadata
import Foundation
import Synchronization

/// The concrete memory settings Auto resolves for one model on one Mac.
public struct AppAutomaticMemoryPlan: Codable, Equatable, Sendable {
    public let profile: AppAutomaticMemoryProfile
    public let contextTokens: Int
    public let expertCacheSlots: Int
    public let estimatedWorkingSetBytes: UInt64
    public let safeBudgetBytes: UInt64
    public let denseResidencyPolicy: DenseResidencyPolicy
    public let residencyMode: InferenceResidencyMode
    public let residentLayerCount: Int?
    public let streamedBytesPerToken: UInt64
    public let prefillChunkTokens: Int
    public let evidence: InferencePlanEvidence
    public let candidateID: String?
    public let retainedConversationBytes: Int?

    public init(profile: AppAutomaticMemoryProfile = .balanced,
                contextTokens: Int,
                expertCacheSlots: Int,
                estimatedWorkingSetBytes: UInt64,
                safeBudgetBytes: UInt64,
                denseResidencyPolicy: DenseResidencyPolicy = .resident,
                residencyMode: InferenceResidencyMode = .partial,
                residentLayerCount: Int? = nil,
                streamedBytesPerToken: UInt64 = 0,
                prefillChunkTokens: Int = 128,
                evidence: InferencePlanEvidence = .fallback,
                candidateID: String? = nil, retainedConversationBytes: Int? = nil) {
        self.profile = profile
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.estimatedWorkingSetBytes = estimatedWorkingSetBytes
        self.safeBudgetBytes = safeBudgetBytes
        self.denseResidencyPolicy = denseResidencyPolicy
        self.residencyMode = residencyMode
        self.residentLayerCount = residentLayerCount
        self.streamedBytesPerToken = streamedBytesPerToken
        self.prefillChunkTokens = prefillChunkTokens
        self.evidence = evidence
        self.candidateID = candidateID
        self.retainedConversationBytes = retainedConversationBytes
    }
}

/// Resolves context and cache capacity using the shared runtime allocation plan.
/// Cache counts retain qualified defaults; larger context must fit alongside
/// the selected chunk's scratch and ring storage.
public enum AppAutomaticMemoryPlanner {
    public static func estimatedAllocation(directory: URL, context: Int,
                                            options: AppRuntimeOptions) -> InferenceMemoryEstimate? {
        guard options.retainedConversationBytes.map({ (0...(1 << 30)).contains($0) }) ?? true else { return nil }
        guard let summary = layoutSummary(at: directory),
              let catalog = TUFFModelCatalog.model(manifestModelID: summary.modelID)
                ?? TUFFModelCatalog.all.first(where: { $0.architecture.id.rawValue == summary.architectureID }) else { return nil }
        return AppAutomaticLayoutPlanner.estimate(catalog: catalog, summary: summary,
            context: context, slots: options.expertCacheSlots, chunk: options.prefillChunkTokens,
            densePolicy: options.denseResidencyPolicy,
            retainedBytes: UInt64(options.retainedConversationBytes ?? Int(ConversationRetentionBudget.defaultCapBytes)))
    }
    private static let summaries = Mutex<[String: ModelWeightLayoutSummary]>([:])
    private static let plans = Mutex<[String: AppAutomaticMemoryPlan]>([:])
    private struct BundledLayouts: Decodable {
        struct Profile: Decodable { let sourceIndexSHA256: String; let layout: ModelWeightLayoutSummary }
        let schemaVersion: Int
        let profiles: [String: Profile]
    }
    private static let bundledLayouts: BundledLayouts? = {
        let bundle = Bundle.main.resourceURL.flatMap {
            Bundle(url: $0.appendingPathComponent("TUFF_TUFFAppCore.bundle"))
        } ?? Bundle.module
        guard let url = bundle.url(forResource: "model-layouts", withExtension: "json"),
              let data = try? Data(contentsOf: url), data.count <= 1 << 20,
              let decoded = try? JSONDecoder().decode(BundledLayouts.self, from: data),
              decoded.schemaVersion == 1 else { return nil }
        return decoded
    }()
    public static func predictedLayout(for descriptor: AppModelInstallDescriptor) -> ModelWeightLayoutSummary? {
        guard let catalog = descriptor.catalogID.flatMap(TUFFModelCatalog.model(id:)),
              let profile = bundledLayouts?.profiles[catalog.id.rawValue],
              profile.sourceIndexSHA256 == catalog.source.sourceIndexSHA256,
              profile.layout.architectureID == catalog.architecture.id.rawValue else { return nil }
        return profile.layout
    }
    /// The last summary per directory and when it was checked. Settings
    /// views ask for this from their bodies, which run many times per layout
    /// pass; two file stats each time added up to a hang in Settings.
    private static let recent = Mutex<[String: (summary: ModelWeightLayoutSummary, checked: Date)]>([:])

    public static func layoutSummary(at directory: URL) -> ModelWeightLayoutSummary? {
        let path = directory.standardizedFileURL.path
        if let hit = recent.withLock({ $0[path] }), Date().timeIntervalSince(hit.checked) < 2 {
            return hit.summary
        }
        guard let summary = statCheckedLayoutSummary(at: directory) else { return nil }
        recent.withLock { if $0.count >= 64 { $0.removeAll() }; $0[path] = (summary, Date()) }
        return summary
    }

    private static func statCheckedLayoutSummary(at directory: URL) -> ModelWeightLayoutSummary? {
        let manifest = directory.appendingPathComponent("manifest.json")
        let weights = directory.appendingPathComponent("model_weights.bin")
        guard let metadata = try? FileManager.default.attributesOfItem(atPath: manifest.path),
              let payload = try? FileManager.default.attributesOfItem(atPath: weights.path) else { return nil }
        let key = "\(directory.standardizedFileURL.path):\(metadata[.modificationDate] ?? ""):\(metadata[.size] ?? ""):\(payload[.modificationDate] ?? ""):\(payload[.size] ?? "")"
        if let cached = summaries.withLock({ $0[key] }) { return cached }
        guard let summary = try? ModelWeightLayoutSummary.read(directory: directory) else { return nil }
        summaries.withLock { if $0.count >= 64 { $0.removeAll() }; $0[key] = summary }
        return summary
    }
    public static func plan(
        for descriptor: AppModelInstallDescriptor,
        on device: TUFFDeviceCapabilities,
        profile: AppAutomaticMemoryProfile = .balanced,
        layoutSummary: ModelWeightLayoutSummary? = nil,
        observations: [InferencePlanObservation] = [],
        workloadKey: String = "text-cold-short-t0",
        modelDirectory: URL? = nil,
        useBundledLayout: Bool = false,
        contextTokens: Int? = nil,
        engineIdentity: String? = nil,
        imageInput: Bool = false
    ) -> AppAutomaticMemoryPlan? {
        guard let id = descriptor.planningCatalogID,
              let catalog = TUFFModelCatalog.model(id: id) else { return nil }
        if let layoutSummary = layoutSummary ?? modelDirectory.flatMap({ self.layoutSummary(at: $0) })
            ?? (useBundledLayout ? predictedLayout(for: descriptor) : nil) {
            let identity = AppAutoPerformanceRecords.identity(summary: layoutSummary, device: device,
                engineIdentity: engineIdentity, modelDirectory: modelDirectory)
            let key = "\(identity):\(profile.rawValue):\(contextTokens ?? 0):\(workloadKey):\(imageInput):\(AppAutoMemoryPressureMonitor.shared.snapshot.pressure.rawValue):\(AppAutoPerformanceRecords.revisionToken)"
            if observations.isEmpty, let cached = plans.withLock({ $0[key] }) { return cached }
            let selected = AppAutomaticLayoutPlanner.plan(catalog: catalog, descriptor: descriptor,
                device: device, profile: profile, summary: layoutSummary,
                observations: observations.isEmpty
                    ? AppAutoPerformanceRecords.observations(identity: identity)
                    : observations, workloadKey: workloadKey, contextOverride: contextTokens,
                imageInput: imageInput)
            if observations.isEmpty, let selected {
                plans.withLock { if $0.count >= 128 { $0.removeAll() }; $0[key] = selected }
            }
            return selected
        }

        let budget = device.safeAppMemoryBudgetBytes
        let chunk = catalog.recommendedPrefillChunkTokens(on: device)
        let qualifiedContext = catalog.runtimeDefaults.contextTokens
        func fits(context: Int, slots: Int) -> Bool {
            guard catalog.estimatedInferenceWorkingSetBytes(contextTokens: context,
                                            expertCacheSlots: slots,
                                            prefillChunkTokens: chunk) <= budget
            else { return false }
            return descriptor.usesExpertCache
                || denseResidentBytes(catalog, context: context) <= budget
        }

        // The qualified slot count, clamped to what this model can use, then
        // raised just far enough to reach chunked prefill. GPT-OSS 20B is
        // qualified at four slots, and the chunked prefill path needs sixteen;
        // leaving it at four would trade a much slower prompt for memory Auto
        // is not otherwise spending. Nothing grows past that — see the note
        // above for why filling the budget with slots is a loss.
        let slots: Int
        if descriptor.usesExpertCache {
            let options = descriptor.usefulExpertCacheSlotCounts.sorted()
            let qualified = catalog.runtimeDefaults.expertCacheSlots
            let clamped = options.contains(qualified)
                ? qualified
                : (options.last(where: { $0 <= qualified })
                    ?? options.first
                    ?? qualified)
            let prefillFloor = options.first {
                $0 >= RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
            }
            if let prefillFloor, prefillFloor > clamped,
               fits(context: qualifiedContext, slots: prefillFloor) {
                slots = prefillFloor
            } else {
                slots = clamped
            }
        } else {
            slots = catalog.runtimeDefaults.expertCacheSlots
        }

        let contextOptions = AppContextLengthOption.options(for: descriptor)
            .map(\.tokens).sorted()

        // Find what this checkpoint can afford on this Mac, then space the
        // profiles across that capacity. A 2K qualification must not trap
        // Speed/Balanced at 2K/4K on machines with gigabytes left for KV.
        let affordable = contextOptions.last {
            fits(context: $0, slots: slots)
        } ?? qualifiedContext
        let target = max(qualifiedContext, affordable / profile.contextCapacityDivisor)
        let context = contextOptions.last { $0 <= target } ?? qualifiedContext

        return AppAutomaticMemoryPlan(
            profile: profile,
            contextTokens: context,
            expertCacheSlots: slots,
            estimatedWorkingSetBytes: catalog.estimatedInferenceWorkingSetBytes(
                contextTokens: context,
                expertCacheSlots: slots,
                prefillChunkTokens: chunk),
            safeBudgetBytes: budget,
            residencyMode: descriptor.usesExpertCache ? .partial : .resident,
            prefillChunkTokens: chunk)
    }

    /// What a dense model really keeps resident: every weight, because every
    /// token reads all of them, plus its KV cache and a runtime allowance.
    /// The qualified working set leaves file-backed weights out, which is
    /// right for streamed experts but not here. Measured on a 16 GB M2,
    /// Gemma 4 12B decoded at 6.3 tok/s with 8K of context and 3.0 with the
    /// 131K Auto used to choose, and as low as 0.34 once a long run added
    /// more state. Only extra context is refused this way; a model's
    /// qualified context is always kept.
    static func denseResidentBytes(_ catalog: TUFFModelDescriptor, context: Int) -> UInt64 {
        let kv = catalog.memory.kvCache.estimatedBytes(contextTokens: context)
        guard kv != .max else { return .max }
        let total = catalog.source.installedBytes.addingReportingOverflow(kv)
        guard !total.overflow else { return .max }
        let withRuntime = total.partialValue.addingReportingOverflow(denseRuntimeAllowanceBytes)
        return withRuntime.overflow ? .max : withRuntime.partialValue
    }

    /// Scratch, tokenizer, sampling and the app itself, beside the weights
    /// and KV cache.
    static let denseRuntimeAllowanceBytes: UInt64 = TUFFModelCatalog.oneGiB

    public static func applying(
        _ profile: AppModelSettingsProfile,
        for descriptor: AppModelInstallDescriptor,
        on device: TUFFDeviceCapabilities,
        modelDirectory: URL? = nil,
        workloadKey: String = "text-cold-short-t0",
        useBundledLayout: Bool = false,
        engineIdentity: String? = nil
    ) -> AppModelSettingsProfile {
        guard profile.automaticMemory,
              let plan = plan(for: descriptor,
                              on: device,
                              profile: profile.automaticMemoryProfile,
                              workloadKey: workloadKey,
                              modelDirectory: modelDirectory,
                              useBundledLayout: useBundledLayout, engineIdentity: engineIdentity) else { return profile }
        var resolved = profile
        resolved.contextTokens = plan.contextTokens
        resolved.expertCacheSlots = plan.expertCacheSlots
        // Auto owns the memory-dependent prefill decision too. This lets a
        // GPT-OSS profile whose manual four-slot cache disables prefill gain
        // the faster chunked path when Auto can afford at least 16 slots.
        resolved.prefillEnabled = resolved.expertCacheSlots
            >= RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        return resolved
    }
}
