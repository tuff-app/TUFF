import Testing
import Darwin
import TUFFModelCatalog
import TUFFRuntimeDescriptions
@testable import TUFFEngine

@Suite struct InferenceMemoryPlanTests {
    @Test func growingFullAttentionReservesOneLayerAboveTheFirstAllocation() {
        for config in [ArchConfig.gemma4_12B_QAT, ArchConfig.qwen38_27B, .qwen38FlashNext,
                       .gemma4_26B_A4B, .gptOss_20B] {
            let initial = KVCacheMemoryPlan(config: config, maxContext: 2_048,
                fp16RingEnabled: true, maxPrefillChunkTokens: 256)
            let growing = KVCacheMemoryPlan(config: config, maxContext: 8_192,
                fp16RingEnabled: true, maxPrefillChunkTokens: 256)
            #expect(initial.growthOverlapBytes == 0)
            #expect(growing.growthOverlapBytes > 0)
            #expect(growing.growthOverlapBytes <= growing.totalBytes)
        }
    }
    /// Inkling runs 12 of its 24 layers three times, each pass with its own
    /// keys and values: 8 full-attention visits and 40 sliding ones, at 4,096
    /// bytes a position. Its blocks stop at 512 rows whatever chunk is asked for.
    @Test func inklingCachesAreCountedByVisit() {
        let config = ArchConfig.inkling2Mini
        for (chunk, rows) in [(128, 128), (512, 512), (2_048, 512)] {
            let plan = KVCacheMemoryPlan(config: config, maxContext: 8_192, fp16RingEnabled: true,
                                         maxPrefillChunkTokens: chunk)
            #expect(plan.capacities.count == 48)
            #expect(plan.totalBytes == UInt64(8 * 8_192 * 4_096 + 40 * (1_024 + rows) * 4_096))
            #expect(plan.growthOverlapBytes == UInt64(8 * 8_192 * 4_096))
        }
        // A context shorter than the window never fills a ring.
        let short = KVCacheMemoryPlan(config: config, maxContext: 600, fp16RingEnabled: true,
                                      maxPrefillChunkTokens: 128)
        #expect(short.totalBytes == UInt64(48 * 600 * 4_096))
        let descriptor = TUFFModelCatalog.inkling2Mini
        // The admission estimate never assumes a chunk below 256 tokens.
        let small = InferenceMemoryPlan(descriptor: descriptor, config: config, contextTokens: 8_192,
                                        expertCacheSlots: 16, prefillChunkTokens: 128)
        let large = InferenceMemoryPlan(descriptor: descriptor, config: config, contextTokens: 8_192,
                                        expertCacheSlots: 16, prefillChunkTokens: 512)
        #expect(small.kvCacheBytes == UInt64(8 * 8_192 * 4_096 + 40 * 1_280 * 4_096))
        // Scratch is about a megabyte a row.
        #expect(large.prefillScratchBytes - small.prefillScratchBytes == UInt64(256 << 20))
        #expect(large.estimatedWorkingSetBytes > small.estimatedWorkingSetBytes)
    }

    @Test func newCacheTrackingArraysHaveAConservativeAllLayerReserve() throws {
        let page = UInt64(sysconf(_SC_PAGESIZE))
        for descriptor in TUFFModelCatalog.all {
            let variant = try #require(ModelVariant(rawValue: descriptor.architecture.id.rawValue))
            let config = try #require(ArchConfig.registeredArchitectures[variant])
            let plan = InferenceMemoryPlan(descriptor: descriptor, config: config,
                contextTokens: 4096, expertCacheSlots: descriptor.runtimeDefaults.expertCacheSlots,
                prefillChunkTokens: 512)
            // Flash's 512-entry counter needs two pages on a 4 KB host once
            // its allocation header is included; other arrays fit one page.
            let pages: UInt64 = page == 4096 && config.numExperts == 512 ? 3 : 2
            let expected = config.numExperts > 0 ? pages * page * UInt64(config.numLayers) : 0
            #expect(plan.cacheTrackingMetadataReserveBytes == expected)
            #expect(plan.estimatedWorkingSetBytes >= descriptor.memory.estimatedWorkingSetBytes(
                contextTokens: 4096, expertCacheSlots: descriptor.runtimeDefaults.expertCacheSlots)
                + expected)
        }
    }

    @Test func everyCatalogModelIncludesChunkScratchAndRuntimeKV() throws {
        for descriptor in TUFFModelCatalog.all {
            let variant = try #require(ModelVariant(rawValue: descriptor.architecture.id.rawValue))
            let config = try #require(ArchConfig.registeredArchitectures[variant])
            let small = InferenceMemoryPlan(descriptor: descriptor, config: config,
                contextTokens: 4096, expertCacheSlots: descriptor.runtimeDefaults.expertCacheSlots,
                prefillChunkTokens: 256)
            let large = InferenceMemoryPlan(descriptor: descriptor, config: config,
                contextTokens: 4096, expertCacheSlots: descriptor.runtimeDefaults.expertCacheSlots,
                prefillChunkTokens: 2048)
            #expect(large.prefillScratchBytes > small.prefillScratchBytes)
            #expect(large.estimatedWorkingSetBytes > small.estimatedWorkingSetBytes)
            let vision = config.family.acceptsImageInput
                ? min(4096, VisionConfig(family: config.family).maximumPooledTokens) : 0
            let expectedKV = KVCacheMemoryPlan(config: config, maxContext: 4096,
                fp16RingEnabled: true, maxPrefillChunkTokens: max(2048, vision))
            #expect(large.kvCacheBytes == expectedKV.totalBytes)
            #expect(large.estimatedWorkingSetBytes == descriptor.estimatedInferenceWorkingSetBytes(
                contextTokens: 4096, expertCacheSlots: descriptor.runtimeDefaults.expertCacheSlots,
                prefillChunkTokens: 2048))
        }
    }

    @Test func allLayerSlotCostsMatchPackedRecords() {
        let records: [(TUFFModelDescriptor, UInt64, UInt64)] = [
            (TUFFModelCatalog.gemma4_26B_A4B, 30, 3_358_720), (TUFFModelCatalog.qwen36_35B_A3B, 40, 1_769_472),
            (TUFFModelCatalog.gptOss_20B, 24, 13_238_272), (TUFFModelCatalog.gptOss_120B, 36, 13_238_272),
            (TUFFModelCatalog.minimaxM27, 62, 7_962_624), (TUFFModelCatalog.qwen38FlashNext, 48, 3_080_192)
        ]
        for (descriptor, layers, stride) in records {
            #expect(descriptor.memory.expertCacheBytesPerSlot == layers * stride)
            let four = descriptor.memory.estimatedWorkingSetBytes(contextTokens: 4096, expertCacheSlots: 4)
            let sixteen = descriptor.memory.estimatedWorkingSetBytes(contextTokens: 4096, expertCacheSlots: 16)
            #expect(sixteen - four == 12 * layers * stride)
        }
    }

    @Test func unsupportedHugeContextSaturatesInsteadOfWrapping() {
        #expect(TUFFModelCatalog.gptOss_120B.estimatedInferenceWorkingSetBytes(
            contextTokens: Int.max, expertCacheSlots: 16, prefillChunkTokens: 2048) == .max)
    }
}
