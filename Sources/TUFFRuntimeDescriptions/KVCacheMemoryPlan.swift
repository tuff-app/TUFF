package struct KVCacheMemoryPlan: Sendable, Equatable {
    package let capacities: [Int]
    package let strides: [Int]
    package let totalBytes: UInt64
    /// Full-attention buffers start at 2,048 rows and double as a chat grows
    /// (`KVCacheManager.initialFullAttentionCapacity`). While one doubles, the
    /// old buffer and its replacement coexist; the replacement can be as large
    /// as the full window, so a whole layer is reserved. Static rings never
    /// grow.
    package let growthOverlapBytes: UInt64

    package init(config: ArchConfig,
         maxContext: Int,
         fp16RingEnabled: Bool,
         slidingWindow: Int? = nil,
         maxPrefillChunkTokens: Int = 128,
         fp16RingCapacityOverride: Int? = nil) {
        precondition(maxContext > 0 && maxPrefillChunkTokens > 0)
        let swaStride = config.numKVHeads * config.headDim * 2
        let fullStride = config.numFullKVHeads * config.fullHeadDim * 2
        // An Inkling layer inside the loop keeps keys and values for each of
        // its passes, so its caches are counted by visit, and its blocks have
        // their own row limit.
        let inkling = config.family == .inkling2
        let loop = config.inklingLoop
        let kinds = inkling
            ? (0..<loop.visits(numLayers: config.numLayers)).map {
                config.fullAttentionLayerMask[loop.layer(ofVisit: $0, numLayers: config.numLayers)]
            }
            : config.fullAttentionLayerMask
        let chunkRows = inkling ? min(maxPrefillChunkTokens, InklingLoopConfig.maximumBlockRows)
            : maxPrefillChunkTokens
        let swaCapacity = min(maxContext, max(1, fp16RingCapacityOverride
            ?? ((slidingWindow ?? config.slidingWindow) + chunkRows)))
        var resolvedCapacities: [Int] = []
        var resolvedStrides: [Int] = []
        var bytes: UInt64 = 0
        var growingBytes: UInt64 = 0
        for kind in kinds {
            if kind == 2 {
                resolvedCapacities.append(0)
                resolvedStrides.append(0)
                continue
            }
            let full = kind != 0
            let capacity = fp16RingEnabled && !full ? swaCapacity : maxContext
            let stride = full ? fullStride : swaStride
            resolvedCapacities.append(capacity)
            resolvedStrides.append(stride)
            let layerBytes = UInt64(capacity)
                .multipliedReportingOverflow(by: UInt64(stride * 2))
            if layerBytes.overflow {
                bytes = .max
            } else if bytes != .max {
                let sum = bytes.addingReportingOverflow(layerBytes.partialValue)
                bytes = sum.overflow ? .max : sum.partialValue
            }
            if full, maxContext > 2_048 {
                let sum = growingBytes.addingReportingOverflow(layerBytes.partialValue)
                growingBytes = layerBytes.overflow || sum.overflow ? .max : sum.partialValue
            }
        }
        capacities = resolvedCapacities
        strides = resolvedStrides
        totalBytes = bytes
        growthOverlapBytes = growingBytes
    }
}
