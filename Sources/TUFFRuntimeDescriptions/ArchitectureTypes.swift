import Foundation

/// Model family discriminator. Selects the tensor-name contract, the layer
/// graph shape (norm sandwich vs plain pre-norm), and family-specific kernel
/// behavior. Stored in `manifest.json -> arch.family`; absent means Gemma 4
/// (the format's original architecture).
public enum ModelFamily: String, Sendable, Equatable {
    case gemma4 = "gemma4"
    case qwen36 = "qwen36"
    case gptOss = "gpt-oss"
    case minimaxM2 = "minimax-m2"
    case qwen4Exp = "qwen4-exp"
    case flint = "flint"

    /// Whether this family's image input runs Qwen's vision tower: 27 blocks
    /// at hidden 1152, patch 16, spatial merge 2, and the `<|vision_start|>`
    /// / `<|image_pad|>` / `<|vision_end|>` markers. Qwen 3.6 and Qwen3.8
    /// Flash Next ship the same 333 tensors and the same token ids, and
    /// differ only in the width the merger projects into — so the tower code
    /// belongs behind one predicate rather than a growing list of families.
    public var usesQwenVisionTower: Bool {
        switch self {
        case .qwen36, .qwen4Exp: true
        case .gemma4, .gptOss, .minimaxM2, .flint: false
        }
    }

    /// Whether the family accepts image input at all.
    public var acceptsImageInput: Bool {
        switch self {
        case .gemma4, .qwen36, .qwen4Exp: true
        case .gptOss, .minimaxM2, .flint: false
        }
    }
}

public enum ModelVariant: String, Sendable, Equatable {
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
    case bonsai2_2B = "ternary-bonsai-2-2b"
    case bonsai2_27B = "ternary-bonsai-2-27b"
    case flint40B = "flint-40b-a2.5b"

    package static func legacyDefault(for family: ModelFamily) -> ModelVariant {
        switch family {
        case .gemma4: return .gemma4_26B_A4B
        case .qwen36: return .qwen36_35B_A3B
        case .gptOss: return .gptOss_20B
        case .minimaxM2: return .minimaxM27
        case .qwen4Exp: return .qwen38FlashNext
        case .flint: return .flint40B
        }
    }
}

public enum FeedForwardKind: String, Sendable, Equatable {
    case dense
    case mixtureOfExperts = "moe"
}

/// YaRN RoPE scaling parameters used by GPT-OSS. Frequencies inside the
/// original context are blended between extrapolation and interpolation;
/// positions beyond it keep the same continuous frequency schedule.
public struct YaRNRopeConfig: Sendable, Equatable {
    public let originalContextLength: Int
    public let scalingFactor: Double
    public let betaFast: Double
    public let betaSlow: Double

    public init(originalContextLength: Int,
                scalingFactor: Double,
                betaFast: Double,
                betaSlow: Double) {
        self.originalContextLength = originalContextLength
        self.scalingFactor = scalingFactor
        self.betaFast = betaFast
        self.betaSlow = betaSlow
    }
}

/// Gated-DeltaNet (linear attention) dimensions. Zeroed for architectures
/// without linear-attention layers.
public struct LinearAttentionConfig: Sendable, Equatable {
    /// Activation applied to the z projection in the gated output norm. Qwen
    /// 3.6 uses the model's own `hidden_act`; Qwen4-Exp sets
    /// `output_gate_type: sigmoid`, which is not the same function and not a
    /// rescaling of it — silu(z) is z * sigmoid(z), so the gate carries the
    /// magnitude of z as well as its sign.
    public enum OutputGate: String, Sendable, Equatable {
        case silu
        case sigmoid
    }

    public let numKHeads: Int
    public let numVHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let convKernelSize: Int
    public let outputGate: OutputGate

    public init(numKHeads: Int, numVHeads: Int,
                keyHeadDim: Int, valueHeadDim: Int,
                convKernelSize: Int,
                outputGate: OutputGate = .silu) {
        self.numKHeads = numKHeads
        self.numVHeads = numVHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convKernelSize = convKernelSize
        self.outputGate = outputGate
    }

    public static let none = LinearAttentionConfig(
        numKHeads: 0, numVHeads: 0, keyHeadDim: 0, valueHeadDim: 0,
        convKernelSize: 0)

    /// Fused qkv projection rows: 2 * K-dim + V-dim. Also the depthwise conv
    /// channel count.
    public var qkvDim: Int { 2 * numKHeads * keyHeadDim + numVHeads * valueHeadDim }
    /// Value dim, also the z-gate projection rows and out_proj columns.
    public var valueDim: Int { numVHeads * valueHeadDim }
}

/// Hyper-connection residual topology. `streamCount` residual streams run in
/// parallel where other architectures carry one, and each block reads a
/// low-rank mixture of them and writes back through a learned per-stream
/// injection. Zeroed for architectures with an ordinary residual.
///
/// The streams replace the pre-norms too: a `qwen4_exp` layer has no
/// `input_layernorm` or `post_attention_layernorm`, only the `hc_norm` inside
/// each of its two hyper-connection modules, and a model-level
/// `hyper_connection_mixer` collapses the streams before the LM head.
public struct HyperConnectionConfig: Sendable, Equatable {
    public let streamCount: Int
    public let lowRank: Int

    public init(streamCount: Int, lowRank: Int) {
        self.streamCount = streamCount
        self.lowRank = lowRank
    }

    public static let none = HyperConnectionConfig(streamCount: 0, lowRank: 0)

    public var isEnabled: Bool { streamCount > 0 }
    /// Width of the stacked residual streams, and of `hc_norm`.
    public func stackedWidth(hiddenSize: Int) -> Int { streamCount * hiddenSize }
}

/// n-gram per-layer embedding table. One decoder layer looks up a rolling hash
/// of the last `ngramSize` tokens in a table of `heads` independently-hashed
/// slices, each roughly `vocabSizeBase` rows, and feeds the result through a
/// depthwise convolution and key/value projections.
///
/// Unlike Gemma's per-layer embeddings this table is far too large to hold
/// resident — 29.80 GiB in the pinned checkpoint, against 2.91 GiB for every
/// other resident tensor combined — while a token touches only `heads` rows of
/// it. Zeroed for architectures without one.
public struct NgramEmbeddingConfig: Sendable, Equatable {
    /// Decoder layer carrying the module. The pinned checkpoint declares
    /// `ple_layer_ids: [2]` but emits its tensors under `layers.1`, and the
    /// tensor names are what the loader resolves against.
    public let layer: Int
    public let ngramSize: Int
    /// Independently-hashed slices of the table, concatenated to `embedDim`.
    public let heads: Int
    public let headsPerNgram: Int
    /// Nominal rows per head. Each head's real vocabulary is the next value at
    /// or above this that the checkpoint's `ngram_heads_vocab_sizes` records.
    public let vocabSizeBase: Int
    /// Files the table is split across.
    public let shardCount: Int
    public let embedDim: Int
    public let convKernelSize: Int
    /// The token that both pads a fresh sequence and bounds a segment. The
    /// hash never reaches across it, so n-grams from one turn cannot mix with
    /// the previous one's.
    public let eosTokenID: Int32

    public init(layer: Int, ngramSize: Int, heads: Int, headsPerNgram: Int,
                vocabSizeBase: Int, shardCount: Int, embedDim: Int,
                convKernelSize: Int, eosTokenID: Int32) {
        self.layer = layer
        self.ngramSize = ngramSize
        self.heads = heads
        self.headsPerNgram = headsPerNgram
        self.vocabSizeBase = vocabSizeBase
        self.shardCount = shardCount
        self.embedDim = embedDim
        self.convKernelSize = convKernelSize
        self.eosTokenID = eosTokenID
    }

    public static let none = NgramEmbeddingConfig(
        layer: -1, ngramSize: 0, heads: 0, headsPerNgram: 0,
        vocabSizeBase: 0, shardCount: 0, embedDim: 0, convKernelSize: 0,
        eosTokenID: 0)

    public var isEnabled: Bool { shardCount > 0 }
    /// Dimensions each head contributes to the concatenated embedding.
    public var headDim: Int { heads > 0 ? embedDim / heads : 0 }
}

/// Sparse attention selection. Full-attention layers score keys through a
/// narrow side projection and attend to only `budget` of them.
///
/// A context no longer than `budget` selects every position, so the indexer is
/// an identity there and only changes results above it. Zeroed for
/// architectures that attend densely.
public struct AttentionIndexerConfig: Sendable, Equatable {
    public let budget: Int
    public let compressRatio: Int
    public let headDim: Int
    public let numHeads: Int
    public let numKVHeads: Int

    public init(budget: Int, compressRatio: Int, headDim: Int,
                numHeads: Int, numKVHeads: Int) {
        self.budget = budget
        self.compressRatio = compressRatio
        self.headDim = headDim
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
    }

    public static let none = AttentionIndexerConfig(
        budget: 0, compressRatio: 0, headDim: 0, numHeads: 0, numKVHeads: 0)

    public var isEnabled: Bool { budget > 0 }
    /// Rows the fused index projection emits: query heads plus key heads.
    public var projectionRows: Int { (numHeads + numKVHeads) * headDim }
}

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String

    /// Per-layer embeddings (PLE) used by Gemma 4 E2B/E4B. Zero means the
    /// architecture does not carry the auxiliary per-token residual stream.
    public let hiddenSizePerLayerInput: Int
    public let vocabSizePerLayerInput: Int
    /// Final decoder layers that reuse K/V projections from the most recent
    /// non-sharing layer of the same attention kind.
    public let numKVSharedLayers: Int
    /// Backing store for `doubleWideFFNFromLayer`; -1 means "no wide layers".
    public let ffnDoubleWideFromLayer: Int

    // Family-dependent extensions. Defaults describe Gemma 4 so that legacy
    // manifests (which omit them) validate unchanged.
    public let family: ModelFamily
    /// Checkpoint architecture identity. Unlike `family`, this distinguishes
    /// dense and MoE variants that share tokenizer and prompt behavior.
    public let variant: ModelVariant
    public let feedForwardKind: FeedForwardKind
    /// Full-attention q_proj emits `2 * numHeads * fullHeadDim` rows: per-head
    /// [query ; gate] halves. Attention output is multiplied by sigmoid(gate)
    /// before o_proj.
    public let attnOutputGate: Bool
    /// Softmax scale for full attention. Gemma 4 uses 1.0.
    public let attentionScale: Double
    /// Embedding lookup is multiplied by sqrt(hiddenSize) (Gemma) or not (Qwen).
    public let embeddingScaledBySqrtHidden: Bool
    /// Router has `router.scale` (input multiplier) and `per_expert_scale`
    /// tensors (Gemma). False: plain quantized linear router with renormalized
    /// top-k softmax weights and no auxiliary scale tensors.
    public let routerScaled: Bool
    /// Gemma's dual-branch FFN sandwich: pre/post feedforward norms plus a
    /// per-layer residual scalar. False = plain pre-norm residual block.
    public let ffnSandwichNorms: Bool
    /// Shared expert output is gated by sigmoid(shared_expert_gate(x)) (Qwen).
    public let sharedExpertGated: Bool
    /// Partial RoPE convention. False (Gemma): pairs (i, headDim/2 + i) for
    /// i < rotatedPairs with frequency divisor = headDim. True (Qwen/NeoX
    /// sub-dim): rotation confined to the first `rotaryDim` elements, pairing
    /// (i, rotaryDim/2 + i), frequency divisor = rotaryDim.
    public let ropeNeoxSubdim: Bool
    /// Each query head has a learned scalar softmax logit whose value vector
    /// is zero. GPT-OSS uses this as an attention sink.
    public let attentionSinks: Bool
    /// Optional YaRN frequency scaling. Nil means unscaled RoPE.
    public let yarnRope: YaRNRopeConfig?
    /// Upper clamp used by GPT-OSS capped SwiGLU. Zero means ordinary SwiGLU.
    public let swigluLimit: Double
    /// Gated-DeltaNet dimensions for layers with mask value 2.
    public let linearAttention: LinearAttentionConfig
    /// Multi-stream residual topology. `.none` means one residual stream and
    /// the usual pre-norms.
    public let hyperConnection: HyperConnectionConfig
    /// n-gram per-layer embedding table. `.none` means no such table.
    public let ngramEmbedding: NgramEmbeddingConfig
    /// Sparse key selection on full-attention layers. `.none` means dense.
    public let attentionIndexer: AttentionIndexerConfig
    /// Affine group size of the INT4 weights: one scale and bias per this many
    /// quantized values. 64 for every checkpoint until Qwen3.8 Flash Next,
    /// whose 4-bit tensors are grouped at 32 because its n-gram PLE rows are
    /// 160 values wide. INT8 tensors — the router and the shared-expert gate —
    /// stay at 64 in every architecture, so this describes the INT4 path only.
    public let int4GroupSize: Int

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String,
        hiddenSizePerLayerInput: Int = 0,
        vocabSizePerLayerInput: Int = 0,
        numKVSharedLayers: Int = 0,
        ffnDoubleWideFromLayer: Int = -1,
        family: ModelFamily = .gemma4,
        variant: ModelVariant? = nil,
        feedForwardKind: FeedForwardKind = .mixtureOfExperts,
        attnOutputGate: Bool = false,
        attentionScale: Double = 1.0,
        embeddingScaledBySqrtHidden: Bool = true,
        routerScaled: Bool = true,
        ffnSandwichNorms: Bool = true,
        sharedExpertGated: Bool = false,
        ropeNeoxSubdim: Bool = false,
        attentionSinks: Bool = false,
        yarnRope: YaRNRopeConfig? = nil,
        swigluLimit: Double = 0,
        linearAttention: LinearAttentionConfig = .none,
        hyperConnection: HyperConnectionConfig = .none,
        ngramEmbedding: NgramEmbeddingConfig = .none,
        attentionIndexer: AttentionIndexerConfig = .none,
        int4GroupSize: Int = AffineQuantizationDefaults.groupSize
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.vocabSizePerLayerInput = vocabSizePerLayerInput
        self.numKVSharedLayers = numKVSharedLayers
        self.ffnDoubleWideFromLayer = ffnDoubleWideFromLayer
        self.family = family
        self.variant = variant ?? ModelVariant.legacyDefault(for: family)
        self.feedForwardKind = feedForwardKind
        self.attnOutputGate = attnOutputGate
        self.attentionScale = attentionScale
        self.embeddingScaledBySqrtHidden = embeddingScaledBySqrtHidden
        self.routerScaled = routerScaled
        self.ffnSandwichNorms = ffnSandwichNorms
        self.sharedExpertGated = sharedExpertGated
        self.ropeNeoxSubdim = ropeNeoxSubdim
        self.attentionSinks = attentionSinks
        self.yarnRope = yarnRope
        self.swigluLimit = swigluLimit
        self.linearAttention = linearAttention
        self.hyperConnection = hyperConnection
        self.ngramEmbedding = ngramEmbedding
        self.attentionIndexer = attentionIndexer
        self.int4GroupSize = int4GroupSize
    }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh"
    )

    /// Canonical Gemma 4 E4B text architecture. The final 18 layers reuse K/V
    /// projections by attention kind, and every layer consumes a 256-wide PLE
    /// signal alongside the main 2,560-wide residual stream.
    public static let gemma4_E4B = ArchConfig(
        hiddenSize: 2560,
        intermediateSize: 10_240,
        moeIntermediateSize: 0,
        numHeads: 8,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262_144,
        slidingWindow: 512,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 42,
        numExperts: 0,
        topKExperts: 0,
        tieWordEmbeddings: true,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.gemma4E4BLayerMask(),
        hiddenActivation: "gelu_pytorch_tanh",
        hiddenSizePerLayerInput: 256,
        vocabSizePerLayerInput: 262_144,
        numKVSharedLayers: 18,
        variant: .gemma4_E4B,
        feedForwardKind: .dense)

    /// Gemma 4 E2B: the same architecture as E4B at a smaller scale — per-layer
    /// embeddings, shared KV projections, one KV head, 35 layers.
    ///
    /// Every field is transcribed from the pinned checkpoint's `config.json`.
    /// The mapping was validated by transcribing E4B's config the same way and
    /// checking it against the `gemma4_E4B` entry below, which the runtime
    /// already generates correct output from: every field matched, the layer
    /// mask included.
    public static let gemma4_E2B = ArchConfig(
        hiddenSize: 1536,
        intermediateSize: 6144,
        moeIntermediateSize: 0,
        numHeads: 8,
        numKVHeads: 1,
        numFullKVHeads: 1,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262_144,
        slidingWindow: 512,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 35,
        numExperts: 0,
        topKExperts: 0,
        tieWordEmbeddings: true,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.gemma4E2BLayerMask(),
        hiddenActivation: "gelu_pytorch_tanh",
        hiddenSizePerLayerInput: 256,
        vocabSizePerLayerInput: 262_144,
        numKVSharedLayers: 20,
        // Layers 15-34 hold a 12,288-wide MLP; layers 0-14 hold 6,144.
        ffnDoubleWideFromLayer: 15,
        variant: .gemma4_E2B,
        feedForwardKind: .dense)

    /// Gemma 4 12B Unified QAT. Unlike E2B/E4B this dense model has no PLE or
    /// shared-KV tail; all MLP projections use the checkpoint's 8-bit affine
    /// overrides while attention and embeddings remain 4-bit.
    public static let gemma4_12B_QAT = ArchConfig(
        hiddenSize: 3_840,
        intermediateSize: 15_360,
        moeIntermediateSize: 0,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 1,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262_144,
        slidingWindow: 1_024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 48,
        numExperts: 0,
        topKExperts: 0,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4_12BLayerMask(),
        hiddenActivation: "gelu_pytorch_tanh",
        variant: .gemma4_12B_QAT,
        feedForwardKind: .dense)

    private static func gemma4_12BLayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 48)
        for i in stride(from: 5, to: 48, by: 6) { mask[i] = 1 }
        return mask
    }

    private static func gemma4E2BLayerMask() -> [UInt8] {
        // `layer_types` in the checkpoint marks full attention every fifth
        // layer: indices 4, 9, 14, 19, 24, 29, 34.
        var mask = [UInt8](repeating: 0, count: 35)
        for i in stride(from: 4, to: 35, by: 5) { mask[i] = 1 }
        return mask
    }

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }

    private static func gemma4E4BLayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 42)
        for i in stride(from: 5, to: 42, by: 6) { mask[i] = 1 }
        return mask
    }

    /// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
    /// gated-DeltaNet linear-attention layers and 10 full-attention layers
    /// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
    /// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
    ///
    /// The sliding-window slots (`numKVHeads`/`headDim`/`slidingWindow`/
    /// `ropeTheta`) mirror the full-attention values; the architecture has no
    /// sliding-window layers so they are never used to size storage.
    public static let qwen36_35B_A3B = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 40,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwen36LayerMask(),
        hiddenActivation: "silu",
        family: .qwen36,
        variant: .qwen36_35B_A3B,
        attnOutputGate: true,
        attentionScale: 0.0625,   // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 32,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4)
    )

    /// Qwen3.5 block contract used by the new Qwen and Ornith checkpoints.
    private static func qwen35Contract(variant: ModelVariant, hidden: Int, layers: Int,
                                       intermediate: Int, heads: Int, kvHeads: Int,
                                       valueHeads: Int, experts: Int = 0, topK: Int = 0,
                                       tied: Bool = false) -> ArchConfig {
        ArchConfig(hiddenSize: hidden, intermediateSize: intermediate,
            moeIntermediateSize: experts > 0 ? intermediate : 0,
            numHeads: heads, numKVHeads: kvHeads, numFullKVHeads: kvHeads,
            headDim: 256, fullHeadDim: 256, vocabSize: 248_320, slidingWindow: 0,
            finalLogitSoftcap: 0, ropeTheta: 10_000_000, fullRopeTheta: 10_000_000,
            partialRotaryFactor: 0.25, numLayers: layers, numExperts: experts,
            topKExperts: topK, tieWordEmbeddings: tied, attentionKEqV: false,
            fullAttentionLayerMask: qwen4ExpLayerMask(count: layers, interval: 4),
            hiddenActivation: "silu", family: .qwen36, variant: variant,
            feedForwardKind: experts > 0 ? .mixtureOfExperts : .dense,
            attnOutputGate: true, attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false, routerScaled: false,
            ffnSandwichNorms: false, sharedExpertGated: experts > 0,
            ropeNeoxSubdim: true,
            linearAttention: .init(numKHeads: 16, numVHeads: valueHeads,
                                   keyHeadDim: 128, valueHeadDim: 128, convKernelSize: 4),
            int4GroupSize: variant == .ornith15_397B ? 32 : 64)
    }

    public static let ornith15_9B = qwen35Contract(variant: .ornith15_9B,
        hidden: 4096, layers: 32, intermediate: 12288, heads: 16, kvHeads: 4, valueHeads: 32)
    public static let ornith15_35B_A3B = qwen35Contract(variant: .ornith15_35B_A3B,
        hidden: 2048, layers: 40, intermediate: 512, heads: 16, kvHeads: 2, valueHeads: 32,
        experts: 256, topK: 8)
    public static let ornith15_122B = qwen35Contract(variant: .ornith15_122B,
        hidden: 3072, layers: 48, intermediate: 1024, heads: 32, kvHeads: 2, valueHeads: 64,
        experts: 256, topK: 8)
    public static let ornith15_397B = qwen35Contract(variant: .ornith15_397B,
        hidden: 4096, layers: 60, intermediate: 1024, heads: 32, kvHeads: 2, valueHeads: 64,
        experts: 512, topK: 10)
    public static let qwen38_27B = qwen35Contract(variant: .qwen38_27B,
        hidden: 5120, layers: 64, intermediate: 17408, heads: 24, kvHeads: 4, valueHeads: 48)
    /// The publisher's pinned 2B text backbone; Q2/Hadamard execution and
    /// model installation require additional validation and are not implied.
    public static let bonsai2_2B = qwen35Contract(variant: .bonsai2_2B,
        hidden: 2048, layers: 24, intermediate: 6144, heads: 8, kvHeads: 2, valueHeads: 16, tied: true)
    public static let bonsai2_27B = qwen35Contract(variant: .bonsai2_27B,
        hidden: 5120, layers: 64, intermediate: 17408, heads: 24, kvHeads: 4, valueHeads: 48)

    /// MiniMax M2.7 4-bit MLX checkpoint. All decoder layers use full GQA,
    /// whole-projection Q/K RMSNorm, and eight sigmoid-routed experts selected
    /// with a learned correction bias. There is no shared expert branch.
    public static let minimaxM27 = ArchConfig(
        hiddenSize: 3_072,
        intermediateSize: 1_536,
        moeIntermediateSize: 1_536,
        numHeads: 48,
        numKVHeads: 8,
        numFullKVHeads: 8,
        headDim: 128,
        fullHeadDim: 128,
        vocabSize: 200_064,
        slidingWindow: 0,
        finalLogitSoftcap: 0,
        ropeTheta: 5_000_000,
        fullRopeTheta: 5_000_000,
        partialRotaryFactor: 0.5,
        numLayers: 62,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: [UInt8](repeating: 1, count: 62),
        hiddenActivation: "silu",
        family: .minimaxM2,
        variant: .minimaxM27,
        attentionScale: 1 / Double(128).squareRoot(),
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: true)

    /// Qwen3.8 Flash Next, the `qwen4_exp` architecture, from the pinned MLX
    /// 4-bit checkpoint.
    ///
    /// Its gated-DeltaNet, full-attention and mixture-of-experts blocks use the
    /// same tensor contract as Qwen3.6 35B-A3B — `linear_attn.in_proj_{qkv,z,
    /// a,b}`, per-head `q_norm`/`k_norm`, a gated `q_proj`, `mlp.gate`,
    /// `mlp.shared_expert.*` and `mlp.switch_mlp.*` — at wider dimensions: 48
    /// layers instead of 40, 48 linear value heads instead of 32, and 512
    /// routed experts with 10 active instead of 256 with 8.
    ///
    /// Three mechanisms have no Qwen3.6 counterpart and require their own
    /// runtime paths: `hyperConnection` replaces the single residual stream and
    /// both pre-norms, `ngramEmbedding` adds a 29.80 GiB hashed table on layer
    /// 1, and `attentionIndexer` makes full attention sparse above 2K context.
    public static let qwen38FlashNext = ArchConfig(
        hiddenSize: 2_560,
        intermediateSize: 640,
        moeIntermediateSize: 640,
        numHeads: 24,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 48,
        numExperts: 512,
        topKExperts: 10,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwen4ExpLayerMask(count: 48, interval: 4),
        hiddenActivation: "silu",
        family: .qwen4Exp,
        variant: .qwen38FlashNext,
        attnOutputGate: true,
        attentionScale: 0.0625,   // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 48,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4,
            outputGate: .sigmoid),
        hyperConnection: HyperConnectionConfig(streamCount: 4, lowRank: 320),
        ngramEmbedding: NgramEmbeddingConfig(
            layer: 1,
            ngramSize: 3,
            heads: 16,
            headsPerNgram: 8,
            vocabSizeBase: 20_000_000,
            shardCount: 128,
            embedDim: 2_560,
            convKernelSize: 4,
            eosTokenID: 248_044),
        attentionIndexer: AttentionIndexerConfig(
            budget: 2_048,
            compressRatio: 4,
            headDim: 128,
            numHeads: 4,
            numKVHeads: 1),
        int4GroupSize: 32)

    /// Linear-attention layers with full attention every `interval`-th layer,
    /// counting from the end of the first group.
    private static func qwen4ExpLayerMask(count: Int, interval: Int) -> [UInt8] {
        var mask = [UInt8](repeating: 2, count: count)
        for i in stride(from: interval - 1, to: count, by: interval) { mask[i] = 1 }
        return mask
    }

    private static func qwen36LayerMask() -> [UInt8] {
        // Layer kinds: 2 = gated-DeltaNet linear, 1 = full attention on every
        // 4th layer ((i + 1) % 4 == 0).
        var mask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { mask[i] = 1 }
        return mask
    }

    /// Canonical GPT-OSS 20B profile from the pinned OpenAI checkpoint. The
    /// model alternates 128-token sliding and full attention, routes each
    /// token through four of 32 experts, and stores routed matrices as MXFP4.
    public static let gptOss_20B = ArchConfig(
        hiddenSize: 2_880,
        intermediateSize: 2_880,
        moeIntermediateSize: 2_880,
        numHeads: 64,
        numKVHeads: 8,
        numFullKVHeads: 8,
        headDim: 64,
        fullHeadDim: 64,
        vocabSize: 201_088,
        slidingWindow: 128,
        finalLogitSoftcap: 0,
        ropeTheta: 150_000,
        fullRopeTheta: 150_000,
        partialRotaryFactor: 1,
        numLayers: 24,
        numExperts: 32,
        topKExperts: 4,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.gptOssLayerMask(count: 24),
        hiddenActivation: "swiglu_capped",
        family: .gptOss,
        variant: .gptOss_20B,
        attentionScale: 0.125,
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: true,
        attentionSinks: true,
        yarnRope: YaRNRopeConfig(
            originalContextLength: 4_096,
            scalingFactor: 32,
            betaFast: 32,
            betaSlow: 1),
        swigluLimit: 7)

    /// Canonical GPT-OSS 120B profile from the pinned OpenAI checkpoint. It
    /// shares the 20B attention geometry and Harmony behavior, while expanding
    /// to 36 layers and 128 routed experts.
    public static let gptOss_120B = ArchConfig(
        hiddenSize: 2_880,
        intermediateSize: 2_880,
        moeIntermediateSize: 2_880,
        numHeads: 64,
        numKVHeads: 8,
        numFullKVHeads: 8,
        headDim: 64,
        fullHeadDim: 64,
        vocabSize: 201_088,
        slidingWindow: 128,
        finalLogitSoftcap: 0,
        ropeTheta: 150_000,
        fullRopeTheta: 150_000,
        partialRotaryFactor: 1,
        numLayers: 36,
        numExperts: 128,
        topKExperts: 4,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.gptOssLayerMask(count: 36),
        hiddenActivation: "swiglu_capped",
        family: .gptOss,
        variant: .gptOss_120B,
        attentionScale: 0.125,
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: true,
        attentionSinks: true,
        yarnRope: YaRNRopeConfig(
            originalContextLength: 4_096,
            scalingFactor: 32,
            betaFast: 32,
            betaSlow: 1),
        swigluLimit: 7)

    private static func gptOssLayerMask(count: Int) -> [UInt8] {
        (0..<count).map { $0.isMultiple(of: 2) ? 0 : 1 }
    }

    /// Recursive Flint uses two dense 6144-wide MLPs followed by 47 routed
    /// layers with a 1024-wide shared expert and 512-wide routed experts.
    public static let flint40B = ArchConfig(
        hiddenSize: 2048, intermediateSize: 1024, moeIntermediateSize: 512,
        numHeads: 32, numKVHeads: 4, numFullKVHeads: 4, headDim: 128, fullHeadDim: 128,
        vocabSize: 150144, slidingWindow: 513, finalLogitSoftcap: 0,
        ropeTheta: 10000, fullRopeTheta: 10000, partialRotaryFactor: 1,
        numLayers: 49, numExperts: 256, topKExperts: 6, tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: (0..<49).map { $0 < 2 || $0 % 4 == 1 || $0 == 48 ? 1 : 0 },
        hiddenActivation: "silu", family: .flint, variant: .flint40B,
        feedForwardKind: .mixtureOfExperts, attentionScale: 1 / Double(128).squareRoot(),
        embeddingScaledBySqrtHidden: false, routerScaled: false, ffnSandwichNorms: false,
        sharedExpertGated: false, ropeNeoxSubdim: true, attentionSinks: true)

    /// Variant registry used by both v1.1 manifests and the legacy-family
    /// resolver. Family alone is intentionally not a registry key because
    /// multiple checkpoints can share prompt behavior without sharing shapes.
    public static var registeredArchitectures: [ModelVariant: ArchConfig] {
        let all: [ModelVariant: ArchConfig] = [
        .gemma4_E2B: .gemma4_E2B,
        .gemma4_E4B: .gemma4_E4B,
        .gemma4_12B_QAT: .gemma4_12B_QAT,
        .gemma4_26B_A4B: .gemma4_26B_A4B,
        .qwen36_35B_A3B: .qwen36_35B_A3B,
        .ornith15_9B: .ornith15_9B,
        .ornith15_35B_A3B: .ornith15_35B_A3B,
        .ornith15_122B: .ornith15_122B,
        .ornith15_397B: .ornith15_397B,
        .qwen38_27B: .qwen38_27B,
        .bonsai2_2B: .bonsai2_2B,
        .bonsai2_27B: .bonsai2_27B,
        .flint40B: .flint40B,
        .gptOss_20B: .gptOss_20B,
        .gptOss_120B: .gptOss_120B,
        .minimaxM27: .minimaxM27,
        .qwen38FlashNext: .qwen38FlashNext,
    ]
#if TUFF_ENGINE_GEMMA
        return all.filter { $0.value.family == .gemma4 }
#elseif TUFF_ENGINE_QWEN
        return all.filter { $0.value.family == .qwen36 || $0.value.family == .qwen4Exp }
#elseif TUFF_ENGINE_GPT_OSS
        return all.filter { $0.value.family == .gptOss }
#elseif TUFF_ENGINE_MINIMAX
        return all.filter { $0.value.family == .minimaxM2 }
#elseif TUFF_ENGINE_FLINT
        return all.filter { $0.value.family == .flint }
#else
        return all
#endif
    }

    /// Resident INT4 GEMV shapes this architecture issues during decode, for
    /// pipeline specialization. Constant-folding the loop bounds measurably
    /// raises achieved bandwidth on the narrower projections.
    public var decodeInt4GEMVShapes: [(m: Int, n: Int)] {
        if family == .gptOss { return [] }
        var shapes: [(m: Int, n: Int)] = []
        if attnOutputGate {
            shapes.append((m: 2 * numHeads * fullHeadDim, n: hiddenSize))
        } else {
            shapes.append((m: numHeads * fullHeadDim, n: hiddenSize))
        }
        shapes.append((m: numFullKVHeads * fullHeadDim, n: hiddenSize))
        shapes.append((m: hiddenSize, n: numHeads * fullHeadDim))
        if headDim != fullHeadDim {
            // Sliding-window layers project with the local head geometry.
            shapes.append((m: numHeads * headDim, n: hiddenSize))
            shapes.append((m: numKVHeads * headDim, n: hiddenSize))
            shapes.append((m: hiddenSize, n: numHeads * headDim))
        }
        if hasLinearAttentionLayers {
            let la = linearAttention
            shapes.append((m: la.qkvDim, n: hiddenSize))
            shapes.append((m: la.valueDim, n: hiddenSize))
            shapes.append((m: hiddenSize, n: la.valueDim))
        }
        shapes.append((m: intermediateSize, n: hiddenSize))
        shapes.append((m: hiddenSize, n: intermediateSize))
        if maxFFNIntermediateSize != intermediateSize {
            // E2B doubles its MLP width from layer 15 on.
            shapes.append((m: maxFFNIntermediateSize, n: hiddenSize))
            shapes.append((m: hiddenSize, n: maxFFNIntermediateSize))
        }
        if hiddenSizePerLayerInput > 0 {
            shapes.append((m: numLayers * hiddenSizePerLayerInput, n: hiddenSize))
            shapes.append((m: hiddenSizePerLayerInput, n: hiddenSize))
            shapes.append((m: hiddenSize, n: hiddenSizePerLayerInput))
        }
        if hyperConnection.isEnabled {
            let stacked = hiddenSize * hyperConnection.streamCount
            shapes.append((m: hyperConnection.lowRank, n: stacked))
            shapes.append((m: stacked, n: hyperConnection.lowRank))
            shapes.append((m: hyperConnection.streamCount, n: stacked))
            // This architecture's final mixer cannot use the fused RMSNorm
            // head, so its separate vocabulary projection needs a variant too.
            shapes.append((m: vocabSize, n: hiddenSize))
            if ngramEmbedding.isEnabled {
                shapes.append((m: stacked, n: ngramEmbedding.embedDim))
                shapes.append((m: hiddenSize, n: ngramEmbedding.embedDim))
            }
        }
        if attentionIndexer.isEnabled {
            shapes.append((m: attentionIndexer.projectionRows, n: hiddenSize))
        }
        return shapes
    }

    /// Resident INT8 GEMV shapes issued during decode (router and, when the
    /// architecture has one, the shared-expert scalar gate).
    public var decodeInt8GEMVShapes: [(m: Int, n: Int)] {
        var shapes: [(m: Int, n: Int)] = []
        if feedForwardKind == .mixtureOfExperts {
            shapes.append((m: numExperts, n: hiddenSize))
        }
        if sharedExpertGated { shapes.append((m: 1, n: hiddenSize)) }
        return shapes
    }

    /// Layer kind helpers over the mask encoding.
    public func layerIsFull(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 1 }
    public func layerIsLinear(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 2 }
    public var hasLinearAttentionLayers: Bool { fullAttentionLayerMask.contains(2) }
    public var hasPerLayerInputs: Bool { hiddenSizePerLayerInput > 0 }
    public var hasSharedExpert: Bool {
        feedForwardKind == .dense || family == .gemma4 || family == .qwen36
            || family == .qwen4Exp || family == .flint
    }
    /// Whether norm weights are stored centered at zero and applied as
    /// `1 + w`, rather than centered at one and applied as `w`.
    ///
    /// True for Qwen4-Exp and nothing else. Every norm in that architecture
    /// takes this form except the gated DeltaNet output norm, which is
    /// centered at one like everyone's — so the flag says how a *weight* is
    /// stored, and each call site still has to know which of the two it is
    /// looking at.
    public var normWeightsCenteredAtZero: Bool { family == .qwen4Exp }

    public var usesProjectionWideQKNorm: Bool { family == .minimaxM2 }
    public var usesSigmoidCorrectionRouter: Bool { family == .minimaxM2 }

    /// Index of the first layer whose feed-forward block is twice
    /// `intermediateSize` wide, or nil when every layer is the same width.
    ///
    /// Gemma 4 E2B declares `use_double_wide_mlp`, and it is not a whole-model
    /// property: its first 15 layers hold a 6,144-wide MLP and its last 20 hold
    /// a 12,288-wide one. Sizing every layer from `intermediateSize` rejected
    /// the checkpoint at layer 15 with a shape mismatch. The wide layers are an
    /// ordinary wider MLP — gate and up gain rows, down gains columns — so only
    /// the width varies, never the arithmetic.
    public var doubleWideFFNFromLayer: Int? {
        ffnDoubleWideFromLayer >= 0 ? ffnDoubleWideFromLayer : nil
    }

    /// The feed-forward width of one layer.
    public func ffnIntermediateSize(layer: Int) -> Int {
        if family == .flint && layer < 2 { return 6144 }
        guard let first = doubleWideFFNFromLayer, layer >= first else {
            return intermediateSize
        }
        return intermediateSize * 2
    }

    /// The widest feed-forward block in the model. Scratch buffers are shared
    /// across layers, so they are sized for the widest one.
    public var maxFFNIntermediateSize: Int {
        family == .flint ? 6144 : (doubleWideFFNFromLayer == nil ? intermediateSize : intermediateSize * 2)
    }
    public var firstKVSharedLayer: Int { numLayers - numKVSharedLayers }
    public func layerSharesKV(_ layer: Int) -> Bool {
        numKVSharedLayers > 0 && layer >= firstKVSharedLayer
    }
    public func kvSourceLayer(for layer: Int) -> Int? {
        guard layerSharesKV(layer) else { return nil }
        let kind = fullAttentionLayerMask[layer]
        return stride(from: firstKVSharedLayer - 1, through: 0, by: -1)
            .first { fullAttentionLayerMask[$0] == kind }
    }
}


public enum AffineQuantizationDefaults {
    public static let groupSize = 64
}
