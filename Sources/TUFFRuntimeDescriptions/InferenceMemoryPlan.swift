import TUFFModelCatalog
import Darwin
import Foundation

/// Admission estimate, not resident pages or a measured process footprint.
/// Adds runtime allocations conservatively to the qualified catalog baseline:
/// its historical scratch allowance is not subtracted without new evidence.
public struct InferenceMemoryPlan: Sendable, Equatable {
    public let prefillScratchBytes: UInt64
    public let kvCacheBytes: UInt64
    public let additionalRingBytes: UInt64
    public let cacheTrackingMetadataReserveBytes: UInt64
    public let estimatedWorkingSetBytes: UInt64

    public init(descriptor: TUFFModelDescriptor, config: ArchConfig,
                contextTokens: Int, expertCacheSlots: Int,
                prefillChunkTokens: Int) {
        let context = max(1, contextTokens)
        cacheTrackingMetadataReserveBytes = Self.trackingMetadataReserve(
            config: config, slots: expertCacheSlots)
        let chunk = max(PrefillRuntimeConfig.baselineChunkTokens,
                        min(PrefillRuntimeConfig.maxChunkTokens, prefillChunkTokens))
        // The affine runner reserves image capacity even for a text request.
        let visionRows = config.family.acceptsImageInput
            ? min(context, VisionConfig(family: config.family).maximumPooledTokens) : 0
        let kv = KVCacheMemoryPlan(config: config, maxContext: context,
                                  fp16RingEnabled: true,
                                  maxPrefillChunkTokens: max(chunk, visionRows))
        kvCacheBytes = kv.totalBytes
        let baselineKV = descriptor.memory.kvCache.estimatedBytes(contextTokens: context)
        additionalRingBytes = kv.totalBytes > baselineKV ? kv.totalBytes - baselineKV : 0
        if config.family == .inkling2 {
            prefillScratchBytes = Self.inklingScratchBytes(config: config, contextTokens: context,
                                                           chunkTokens: chunk)
        } else if config.family == .gptOss {
            let expert = GPTOSSExpertScratchLayout(hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize, topK: config.topKExperts,
                queryCapacity: chunk)
            let rowBytes = config.hiddenSize * 8
                + config.numHeads * config.headDim * 4
                + config.numKVHeads * config.headDim * 4
                + config.numExperts * 4 + config.topKExperts * 4
            let batched = RuntimeAllocationGeometry.maximumBatchedExpertRows
                * (3 * config.hiddenSize + 3 * config.moeIntermediateSize) * 2
            prefillScratchBytes = UInt64(chunk * rowBytes + expert.totalBytes + batched)
        } else {
            let scratch = PrefillChunkScratchLayout(config: config, chunkTokens: chunk)
            // Old and new chunk scratch coexist while a larger chunk is allocated.
            let batched = config.numExperts > 0
                ? RuntimeAllocationGeometry.maximumBatchedExpertRows
                    * (2 * config.hiddenSize + 3 * config.moeIntermediateSize) * 2 : 0
            prefillScratchBytes = UInt64(2 * scratch.totalPersistentBytes + batched)
        }
        var estimate = descriptor.memory.estimatedWorkingSetBytes(
            contextTokens: context,
            expertCacheSlots: min(max(0, expertCacheSlots), max(1, config.numExperts)))
        // Flash Next's full KV grows geometrically; the old buffers can coexist
        // with their replacements. Reserve a full additional KV allocation.
        // Inkling's full caches double too, all of them in one step: the old
        // buffers, at most half the size of their replacements, coexist with them.
        let growth = config.family == .qwen4Exp ? kv.totalBytes
            : config.family == .inkling2 ? kv.growthOverlapBytes / 2 : 0
        for bytes in [prefillScratchBytes, additionalRingBytes, growth, cacheTrackingMetadataReserveBytes] {
            let sum = estimate.addingReportingOverflow(bytes)
            estimate = sum.overflow ? .max : sum.partialValue
        }
        estimatedWorkingSetBytes = estimate
    }

    /// Inkling's forward scratch: about 0.8 MB a prompt row, and one attention
    /// score per head and key for as many rows as fit beside the longest cache.
    private static func inklingScratchBytes(config: ArchConfig, contextTokens: Int,
                                            chunkTokens: Int) -> UInt64 {
        let rows = UInt64(min(max(1, chunkTokens), InklingLoopConfig.maximumBlockRows))
        let ring = UInt64(config.slidingWindow) + rows
        let scores = max(UInt64(config.numHeads) * max(UInt64(contextTokens), ring) * 4, 32 << 20)
        return rows * (1 << 20) + scores + (16 << 20)
    }

    private static func trackingMetadataReserve(config: ArchConfig, slots: Int) -> UInt64 {
        guard config.numExperts > 0, config.numLayers > 0 else { return 0 }
        let page = UInt64(max(1, sysconf(_SC_PAGESIZE)))
        // Two new host arrays per opened expert layer: prediction counts and
        // unused-prefetch flags. Allow 64 bytes per allocation and whole pages.
        func allocation(_ bytes: UInt64) -> UInt64 {
            let header = bytes.addingReportingOverflow(64)
            let rounded = header.partialValue.addingReportingOverflow(page - 1)
            guard !header.overflow && !rounded.overflow else { return .max }
            return rounded.partialValue / page * page
        }
        let counters = UInt64(config.numExperts).multipliedReportingOverflow(
            by: UInt64(MemoryLayout<Int>.stride))
        guard !counters.overflow else { return .max }
        let flags = UInt64(min(max(1, slots), config.numExperts))
        let layer = allocation(counters.partialValue).addingReportingOverflow(allocation(flags))
        guard !layer.overflow else { return .max }
        let total = layer.partialValue.multipliedReportingOverflow(by: UInt64(config.numLayers))
        return total.overflow ? .max : total.partialValue
    }
}

public extension TUFFModelDescriptor {
    func estimatedInferenceWorkingSetBytes(contextTokens: Int, expertCacheSlots: Int,
                                           prefillChunkTokens: Int) -> UInt64 {
        guard let variant = ModelVariant(rawValue: architecture.id.rawValue),
              let config = ArchConfig.registeredArchitectures[variant] else { return .max }
        return InferenceMemoryPlan(descriptor: self, config: config,
            contextTokens: contextTokens, expertCacheSlots: expertCacheSlots,
            prefillChunkTokens: prefillChunkTokens).estimatedWorkingSetBytes
    }
}

public extension TUFFModelDescriptor {
    /// Bytes of retained conversation state a loaded session may keep, after
    /// the working set and, when the image pack is installed, an allowance for
    /// mapping the image tower and its transient encode work. The figure is
    /// charged against the same safe budget as the working set, so a session
    /// reserves it with its model rather than finding it later.
    func retainedConversationBudgetBytes(contextTokens: Int, expertCacheSlots: Int,
                                         prefillChunkTokens: Int,
                                         device: TUFFDeviceCapabilities,
                                         imagePackInstalled: Bool,
                                         environment: [String: String]
                                            = ProcessInfo.processInfo.environment) -> Int {
        var working = estimatedInferenceWorkingSetBytes(
            contextTokens: contextTokens, expertCacheSlots: expertCacheSlots,
            prefillChunkTokens: prefillChunkTokens)
        if imagePackInstalled,
           let addon = addons.first(where: { $0.kind == .imageInput }) {
            let allowance = addon.source.installedBytes
                .addingReportingOverflow(addon.source.reserveBytes)
            let total = working.addingReportingOverflow(
                allowance.overflow ? .max : allowance.partialValue)
            working = total.overflow ? .max : total.partialValue
        }
        return ConversationRetentionBudget.budget(safeBudgetBytes: device.safeAppMemoryBudgetBytes,
                                             workingSetBytes: working,
                                             environment: environment)
    }
}
