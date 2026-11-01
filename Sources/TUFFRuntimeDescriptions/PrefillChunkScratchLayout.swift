package struct PrefillChunkScratchLayout: Sendable, Equatable {
    package let chunkTokens: Int
    package let hiddenSize: Int
    package let maxQElementsPerToken: Int
    package let maxKVElementsPerToken: Int
    package let sharedIntermediate: Int
    package let routedIntermediate: Int
    package let topK: Int
    package let routedPairMicrobatchRows: Int
    /// Rows of the widest per-token projection written into `q`: the packed
    /// [query ; gate] q_proj when `attnOutputGate`, and the gated-DeltaNet
    /// `in_proj_qkv` rows on architectures with linear-attention layers.
    package let qProjElementsPerToken: Int
    /// Non-zero when the architecture gates attention output (Qwen): sizes
    /// the split q / gate buffers.
    package let attnGateElementsPerToken: Int
    /// Gated-DeltaNet dims (all zero when the mask has no linear layers).
    package let gdnQKVDim: Int
    package let gdnValueDim: Int
    package let gdnVHeads: Int
    /// Prism's alpha and beta projections retain FP32 outputs.
    package let gdnScalarElementBytes: Int
    /// Non-zero when the shared expert output is scalar-gated (Qwen).
    package let sharedScalarGateElements: Int
    /// Gemma 4 dense PLE dimensions (zero on architectures without PLE).
    package let plePackedWidth: Int
    package let pleWidth: Int
    /// Residual streams. One on an ordinary architecture; four on Qwen4-Exp,
    /// where `hidden` and `normed` carry every stream stacked per token and
    /// the block reads a mixture of them instead of the residual itself.
    package let streamCount: Int
    /// Hyper-connection low-rank width (zero when there are no streams).
    package let hcLowRank: Int
    /// Width of one n-gram embedding row, zero without an n-gram layer. The
    /// chunk path stages a row per token here, so this has to be sized from
    /// the layout rather than a constant: an image span is emitted as a single
    /// chunk of its own pooled length, which is larger than `maxChunkTokens`
    /// for any ordinary photo.
    package let ngramEmbedDim: Int

    package init(config: ArchConfig,
                chunkTokens: Int,
                routedPairMicrobatchRows: Int = 32,
                chunkTokenLimit: Int = PrefillRuntimeConfig.maxChunkTokens) {
        self.chunkTokens = max(1, min(chunkTokens, chunkTokenLimit))
        self.hiddenSize = config.hiddenSize
        self.maxQElementsPerToken = config.numHeads * max(config.headDim, config.fullHeadDim)
        self.maxKVElementsPerToken = max(config.numKVHeads * config.headDim,
                                         config.numFullKVHeads * config.fullHeadDim)
        // The widest layer, since this scratch is reused across all of them.
        self.sharedIntermediate = config.maxFFNIntermediateSize
        self.routedIntermediate = config.moeIntermediateSize
        self.topK = config.topKExperts
        self.routedPairMicrobatchRows = max(1, min(routedPairMicrobatchRows, 128))
        let qDim = config.numHeads * max(config.headDim, config.fullHeadDim)
        let hasLinear = config.hasLinearAttentionLayers
        self.qProjElementsPerToken = max(config.attnOutputGate ? 2 * qDim : qDim,
                                         hasLinear ? config.linearAttention.qkvDim : 0)
        self.attnGateElementsPerToken = config.attnOutputGate ? qDim : 0
        self.gdnQKVDim = hasLinear ? config.linearAttention.qkvDim : 0
        self.gdnValueDim = hasLinear ? config.linearAttention.valueDim : 0
        self.gdnVHeads = hasLinear ? config.linearAttention.numVHeads : 0
        self.gdnScalarElementBytes = (config.variant == .bonsai2_2B || config.variant == .bonsai2_27B)
            ? MemoryLayout<Float>.stride : MemoryLayout<Float16>.stride
        self.sharedScalarGateElements = config.sharedExpertGated ? 1 : 0
        self.plePackedWidth = config.hasPerLayerInputs
            ? config.numLayers * config.hiddenSizePerLayerInput : 0
        self.pleWidth = config.hiddenSizePerLayerInput
        self.streamCount = config.hyperConnection.isEnabled
            ? config.hyperConnection.streamCount : 1
        self.hcLowRank = config.hyperConnection.isEnabled
            ? config.hyperConnection.lowRank : 0
        self.ngramEmbedDim = config.ngramEmbedding.isEnabled
            ? config.ngramEmbedding.embedDim : 0
    }


    /// The residual and its norm carry every stream, so both are
    /// `streamCount` times wider than the block's own working width.
    package var hiddenElements: Int { chunkTokens * hiddenSize * streamCount }
    package var normedElements: Int { hiddenElements }
    package var hcMixedElements: Int { streamCount > 1 ? chunkTokens * hiddenSize : 0 }
    package var hcLowRankElements: Int { chunkTokens * hcLowRank }
    package var hcMixUpElements: Int { streamCount > 1 ? chunkTokens * hiddenSize * streamCount : 0 }
    package var hcInjectElements: Int { streamCount > 1 ? chunkTokens * streamCount : 0 }
    package var ngramEmbeddingElements: Int { chunkTokens * ngramEmbedDim }
    package var qElements: Int { chunkTokens * qProjElementsPerToken }
    package var attnElements: Int { chunkTokens * maxQElementsPerToken }
    package var attnQElements: Int { chunkTokens * attnGateElementsPerToken }
    package var attnGateElements: Int { attnQElements }
    package var gdnConvOutElements: Int { chunkTokens * gdnQKVDim }
    package var gdnZElements: Int { chunkTokens * gdnValueDim }
    package var gdnAElements: Int { chunkTokens * gdnVHeads }
    package var gdnBElements: Int { gdnAElements }
    package var gdnYElements: Int { gdnZElements }
    package var sharedScalarGateBufferElements: Int { chunkTokens * sharedScalarGateElements }
    package var pleIdentityElements: Int { chunkTokens * plePackedWidth }
    package var pleContextElements: Int { pleIdentityElements }
    package var pleLayerElements: Int { chunkTokens * pleWidth }
    package var pleGateElements: Int { pleLayerElements }
    package var kStageElements: Int { chunkTokens * maxKVElementsPerToken }
    package var vStageElements: Int { kStageElements }
    package var attentionOutputElements: Int { chunkTokens * max(maxQElementsPerToken, gdnValueDim) }
    package var denseXElements: Int { hiddenElements }
    package var routedXElements: Int { hiddenElements }
    package var routerXElements: Int { hiddenElements }
    package var h1Elements: Int { hiddenElements }
    package var h2Elements: Int { hiddenElements }
    package var routePartialElements: Int { chunkTokens * topK * hiddenSize }
    package var routeIDElements: Int { chunkTokens * topK }
    package var routeWeightElements: Int { routeIDElements }
    /// Gate, up and activation rows for the whole chunk: the batched MLP
    /// path writes every token's intermediate at once.
    package var sharedExpertScratchElements: Int {
        max(1, chunkTokens) * sharedIntermediate
    }
    package var routedGateUpActElements: Int { 3 * routedPairMicrobatchRows * routedIntermediate }
    package var routedDownOutputElements: Int { routedPairMicrobatchRows * hiddenSize }

    package var devicePrivateBytes: Int {
        let fp16Elements = hiddenElements
            + normedElements
            + qElements
            + kStageElements
            + vStageElements
            + attentionOutputElements
            + denseXElements
            + routedXElements
            + routerXElements
            + h1Elements
            + h2Elements
            + routePartialElements
            + 3 * sharedExpertScratchElements
            + routedGateUpActElements
            + routedDownOutputElements
            + attnQElements
            + attnGateElements
            + gdnConvOutElements
            + gdnZElements
            + (gdnAElements + gdnBElements) * (gdnScalarElementBytes / MemoryLayout<Float16>.stride)
            + gdnYElements
            + sharedScalarGateBufferElements
            + pleIdentityElements
            + pleContextElements
            + pleLayerElements
            + pleGateElements
            + hcMixedElements
            + hcLowRankElements
            + hcMixUpElements
            + hcInjectElements
        return fp16Elements * MemoryLayout<Float16>.stride
    }

    package var sharedMetadataBytes: Int {
        routeIDElements * MemoryLayout<UInt32>.stride
            + routeWeightElements * MemoryLayout<Float16>.stride
            // Written by the CPU as the chunk is encoded, so shared rather
            // than private.
            + ngramEmbeddingElements * MemoryLayout<Float16>.stride
    }

    package var totalPersistentBytes: Int {
        devicePrivateBytes + sharedMetadataBytes
    }
}
