import Foundation
import Metal

public final class RealForwardRunner: ChunkedPrefillRunner, MultimodalPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, SpeculativeVerificationRunner, PrefillHeadSkippingRunner, @unchecked Sendable {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Gemma-only post_feedforward_layernorm_1; nil when the arch has no
        /// FFN sandwich norms.
        let postF1: TensorView?
        /// Qwen-only [1, hidden] scalar gate on the shared expert branch.
        let scalarGate: TensorView?
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let flintEmbedding: FlintEmbedding?
    private let flintResidual: FlintResidual?
    private let flintRouter: FlintRouter?
    private let flintFiniteLogits: FlintFiniteLogits?
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let flintInt8: DequantInt8GEMV?
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE?
    private let fusionHead: LMHeadChainInt4
    private lazy var greedyArgmax: FP16Argmax = try! FP16Argmax(context: ctx)
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue
    private let fusedPostAttentionSetup: FusedPostAttentionSetup
    private let fusedTail: FusedLayerTail

    // Qwen 3.6 kernels. Nil on architectures that never dispatch them.
    private let elementwise: Elementwise?
    private let gdn: GDN?
    private let prism: PrismProjection?
    private let prismGDN: PrismGDN?
    private let prismToken: MTLBuffer?
    private let gdnState: GDNStateManager?
    private let rope: RoPE?
    private let perLayerEmbeddingKernel: PerLayerEmbedding?
    private let int8ScalarGate: DequantInt8GEMV?
    /// Difference between Qwen's logical multimodal RoPE position and the
    /// physical KV index. It remains zero for text-only and every Gemma run.
    private var qwenMultimodalRopeDelta: Int32 = 0
    private let qwenSparseAttention: QwenSparseAttention?

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillHyperConnection: PrefillHyperConnection?
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    /// Batched MPP projections at the checkpoint's INT4 group (64, or 32 for
    /// Qwen 3.8 Flash Next). Nil where MSL 4 tensors are unavailable.
    private let prefillMPPAffine: MPPPrefillAffineQMM?
    /// Opt-in shared-weight projections for 2-31 token prefill blocks. Nil
    /// unless `smallBlockPrefill` is enabled, so the default runner compiles
    /// and dispatches exactly what it did before.
    private let smallBlockPrefill: SmallBlockPrefillPolicy
    private let smallBlockProjection: DequantInt4SmallBlock?
    /// Lets ordinary MoE prefill overlap the shared expert with routed
    /// preparation. Off by default; see `SharedExpertOverlapPolicy`.
    private let sharedExpertOverlap: SharedExpertOverlapPolicy
    /// Test-only view of each layer's routed MoE prefill phase. A throw from it
    /// is treated like a failure at that step. Nil in production.
    var routedPhaseObserver: ((PrefillRoutedPhaseStep) throws -> Void)?
    /// Test-only fault injection after a decode read is registered.
    var decodeLookaheadObserver: ((DispatchGroup) throws -> Void)?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillPostAttention: PrefillPostAttentionSetup
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    /// Routed experts as batched MPP projections, with its bounded scratch.
    /// Nil on dense models and where MPP is unavailable, which keeps the
    /// per-pair kernels.
    private let prefillBatchedRouted: PrefillBatchedRoutedExperts?
    private let prefillBatchedRoutedScratch: PrefillBatchedRoutedExperts.Scratch?
    private let prefillMoE: PrefillMoE
    private let prefillLayerTail: PrefillLayerTail
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    private let hidden: MTLBuffer        // [D] FP16, or [streams * D] for qwen4_exp
    private let normed: MTLBuffer        // [D] FP16, or [streams * D] for qwen4_exp
    // Qwen4-Exp hyper-connection scratch. Nil for every single-stream
    // architecture, which is all of them but this one.
    private let hyperConnection: HyperConnection?
    private let hcMixed: MTLBuffer?      // [D] FP16, the block's input
    private let hcLowRank: MTLBuffer?    // [lowRank] FP16
    private let hcMixUp: MTLBuffer?      // [streams * D] FP16
    private let hcInject: MTLBuffer?     // [streams] FP16, pre-sigmoid
    // Qwen4-Exp n-gram per-layer embedding, on one layer of the model.
    private let ngramPLE: NgramPLE?
    private let ngramRows: NgramRowIndex?
    private let ngramTable: NgramTableReader?
    private let pleEmbedding: MTLBuffer?   // [embedDim] FP16
    private let pleKeys: MTLBuffer?        // [streams * D] FP16
    private let pleValues: MTLBuffer?      // [D] FP16
    private let pleQueries: MTLBuffer?     // [streams * D] FP16
    private let pleGated: MTLBuffer?       // [streams * D] FP16
    private let pleNormed: MTLBuffer?      // [streams * D] FP16
    private let pleConvOut: MTLBuffer?     // [streams * D] FP16
    /// `(taps - 1) * dilation + 1` rows of `streams * D`, the last being the
    /// current token. Carried across tokens so decode sees what prefill saw.
    private let pleConvHistory: MTLBuffer?
    /// Tokens the n-gram hash needs from before this call.
    private var ngramContext: [Int32] = []

    /// What an attention or feed-forward block reads.
    ///
    /// For a single-stream architecture this is the normalized residual, as it
    /// always was. With hyper-connections the residual is `streams` wide and
    /// the block consumes the mixture of them instead, which is one stream
    /// wide — so every block below stays width-correct without knowing which
    /// architecture it is running under.
    private var blockInput: MTLBuffer { hcMixed ?? normed }

    /// What the feed-forward block reads: the router, the shared expert and
    /// routed phase 1 all take the same vector. With hyper-connections that is
    /// the feed-forward mixture; otherwise the post-attention norm's output.
    private var ffnInput: MTLBuffer { hcMixed ?? routedX }

    /// Test-only windows onto the residual and the per-block outputs, so a
    /// forward pass that goes wrong can be bisected a stage at a time rather
    /// than inferred from its logits. These found a routed-expert dispatch
    /// computing eight of ten experts and reducing over uninitialized
    /// threadgroup memory.
    var debugResidual: MTLBuffer { hidden }
    var debugBlockInput: MTLBuffer? { hcMixed }
    var debugAttentionOut: MTLBuffer { oOut }
    var debugSharedExpertOut: MTLBuffer { h1Buf }
    var debugRoutedOut: MTLBuffer { h2Buf }
    var debugFlintQuery: MTLBuffer { qScratch }
    var debugFlintAttentionContext: MTLBuffer { attnOut }
    var debugFlintFFNInput: MTLBuffer { routedX }
    var debugFlintResidual: MTLBuffer? { flintResidual?.values }
    func debugKVSlot(layer: Int, position: Int) -> (key: (buffer: MTLBuffer, offset: Int), value: (buffer: MTLBuffer, offset: Int))? {
        guard let kv else { return nil }
        return (kv.kSlot(layer: layer, position: position), kv.vSlot(layer: layer, position: position))
    }
    private var sharedGateBeforeHalf: Bool {
        cfg.variant == .ornith15_397B && model.sharedExpertWeightBits == 4
    }

    /// Test-only: called with the index of the layer about to run, and once
    /// more with `numLayers`, after every command already encoded for the
    /// token has completed — so `debugResidual` holds that layer's input. A
    /// per-layer trace is the only practical way to bisect a 48-layer forward
    /// against a reference, and setting this forces one commit per layer, so
    /// nothing but a diagnostic should install it.
    var debugLayerSink: ((Int) -> Void)?
    /// Diagnostic observer invoked after router readback, before expert fetch.
    var debugRouterSink: ((Int, [Int]) -> Void)?

    /// Test-only: the chunk path's twin of `debugLayerSink`, called with the
    /// index of the layer just finished and that layer's output for the
    /// chunk's last token — directly comparable to what `debugLayerSink`
    /// reports at the same position on the decode path.
    var debugChunkLayerSink: ((Int, [Float]) -> Void)?
    private let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    private let kStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let vStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let oOut: MTLBuffer          // [D] FP16
    private let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    private let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: MTLBuffer // [F=2112] FP16
    private let denseScratchUp: MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: MTLBuffer  // [F=2112] FP16
    private let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    /// Decode lookahead: the next layer's predicted experts, read ahead.
    private var lookahead = ExpertLookahead()
    private lazy var lookaheadIndices: MTLBuffer = ctx.device.makeBuffer(
        length: max(1, cfg.topKExperts) * MemoryLayout<UInt32>.stride,
        options: .storageModeShared)!
    private lazy var lookaheadWeights: MTLBuffer = ctx.device.makeBuffer(
        length: max(1, cfg.topKExperts) * MemoryLayout<Float16>.stride,
        options: .storageModeShared)!
    /// Reads each prefill layer's routed experts, and the next layer's
    /// predicted ones, into the file cache ahead of the tile fetches. A
    /// 20-token prompt on Gemma 4 26B A4B reads 1,407 experts (4.7 GB), and
    /// one tile's fetch at a time reached 1.5 GB/s of an SSD that does
    /// 3.5 GB/s with several reads in flight. On a 16 GB M2: Qwen 3.6 35B
    /// short prompts 1.35x and a 1,040-token prompt 22.5 -> 16.7 s; Ornith
    /// 1.5 35B 1.33x and 25.2 -> 17.5 s; Gemma 4 26B 1.13x and 1.34x; greedy
    /// output unchanged. `TUFF_PREFILL_READAHEAD=off` turns it off.
    private let prefillReadsAhead =
        ProcessInfo.processInfo.environment["TUFF_PREFILL_READAHEAD"] != "off"

    /// Prefill lookahead: layer L+1's router output for every prompt row.
    private var prefillLookaheadRoutes: (ids: MTLBuffer, weights: MTLBuffer)?
    private func prefillLookaheadBuffers(ids: MTLBuffer, weights: MTLBuffer) -> (ids: MTLBuffer, weights: MTLBuffer)? {
        if let existing = prefillLookaheadRoutes, existing.ids.length >= ids.length,
           existing.weights.length >= weights.length { return existing }
        guard let newIDs = ctx.device.makeBuffer(length: ids.length, options: .storageModeShared),
              let newWeights = ctx.device.makeBuffer(length: weights.length, options: .storageModeShared)
        else { return nil }
        prefillLookaheadRoutes = (newIDs, newWeights)
        return prefillLookaheadRoutes
    }

    /// Decode lookahead counters for TUFF_PHASES.
    public var lookaheadPrecision: Double { lookahead.precision }
    public var lookaheadReadsIssued: Int { lookahead.readsIssued }
    var exposedPrefetchWaitNanos: UInt64 { lookahead.exposedWaitNanos }
    var expertReadMetrics: ExpertReadMetrics { model.expertReadMetrics }
    public var lookaheadEnabled: Bool { lookahead.enabled && !model.stagesDenseLayers }
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    private let moeActs: MTLBuffer       // [topK * FmoE] FP16
    private let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    private let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    private let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    private let speculativeTokenBuffer: MTLBuffer // bounded candidate IDs
    private let speculativeTargetTokenBuf: MTLBuffer // one argmax per candidate row
    // Qwen 3.6 decode scratch (nil on architectures that never use it).
    private let qPackedScratch: MTLBuffer?   // [2 * N_HEADS * head_dim] packed [q ; gate]
    private let attnGateScratch: MTLBuffer?  // [N_HEADS * head_dim]
    private let gdnQKVRaw: MTLBuffer?        // [qkvDim] raw in_proj_qkv output
    private let gdnConvOut: MTLBuffer?       // [qkvDim] conv + SiLU output
    private let gdnZ: MTLBuffer?             // [valueDim]
    private let gdnA: MTLBuffer?             // [numVHeads]
    private let gdnB: MTLBuffer?             // [numVHeads]
    private let gdnY: MTLBuffer?             // [valueDim] delta-rule output
    private let gdnOut: MTLBuffer?           // [valueDim] gated-norm output
    private let sharedScalarGateBuf: MTLBuffer? // [1] shared-expert gate logit
    private let pleIdentity: MTLBuffer?       // [numLayers * P] token PLE lookup
    private let pleContext: MTLBuffer?        // [numLayers * P] context projection
    private let pleLayer: MTLBuffer?          // [P] selected, combined PLE row
    private let pleGate: MTLBuffer?           // [P] mapping gate / activation
    /// BF16 ones over [numExperts]; neutral per_expert_scale when the router
    /// has no auxiliary scale tensors.
    private let onesPerExpertScale: MTLBuffer?
    private var prefillChunkState = PrefillChunkCommitState()
    /// Flint prefill chunk scratch; allocated on first chunked prefill.
    fileprivate var flintPrefill: FlintBatchedPrefillState?
    /// Test-only: (layer, first position, rows x topK expert ids) per prefill chunk layer.
    var debugFlintPrefillRouteSink: ((Int, Int, [UInt32]) -> Void)?
    private var prefillClock = PrefillPhaseClock()
    public var prefillPhaseTimes: PrefillPhaseTimes { prefillClock.times }
    private var prefillScratch: PrefillChunkScratchBuffers?
    /// Largest text chunk the KV ring was sized for at load. A later request
    /// for more is run at this size rather than refused mid-prefill.
    private let prefillTextChunkCapacity: Int
    private var speculativeStartPosition: Int?
    private var speculativeProcessedTokens = 0
    private var collectingSpeculativeMetrics = false
    private var speculativeExpertReads: UInt64 = 0
    private var speculativeExpertBytes: UInt64 = 0
    private var speculativeExpertCacheHits: UInt64 = 0
    private var speculativeExpertCacheMisses: UInt64 = 0
    private var speculativeTargetCommandBuffers: UInt64 = 0

    private static let rdadviseBoundedMissCap = 12
    private static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    private static let rdadviseAdaptiveMissCap = 12
    private static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    private static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections?]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must
    /// either pass `forceLogitsHead: true` or turn the fused head off for the
    /// request with `selectHead(pureGreedy:)`, or they read a never-written
    /// buffer.
    private let fusedGreedyHeadAvailable: Bool
    /// Both heads are built at load, so one runner can serve a greedy request
    /// and then a sampled one. The head writes no KV, so saved state carries
    /// over between them.
    private var fusedGreedyHeadSelected = true
    private var useFusedGreedyHead: Bool { fusedGreedyHeadAvailable && fusedGreedyHeadSelected }
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    private var rdadviseSkipUntilPosition: Int = -1
    private var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    private var rdadviseAdaptivePosition: Int = -1
    private var rdadviseAdaptivePositionBytes: UInt64 = 0
    /// The small-block prefill path follows `TUFF_SMALL_BLOCK_PREFILL` and the
    /// shared-expert overlap follows `TUFF_SHARED_EXPERT_OVERLAP`, both read
    /// once here; see `SmallBlockPrefillPolicy` and `SharedExpertOverlapPolicy`.
    public convenience init(model: Model, context: MetalContext, maxContext: Int,
                            runtimeConfiguration: RuntimeConfiguration = .production) throws {
        let environment = ProcessInfo.processInfo.environment
        try self.init(model: model, context: context, maxContext: maxContext,
                      runtimeConfiguration: runtimeConfiguration,
                      smallBlockPrefill: SmallBlockPrefillPolicy(
                        environment: environment,
                        variant: model.config.variant),
                      sharedExpertOverlap: SharedExpertOverlapPolicy(
                        environment: environment,
                        variant: model.config.variant))
    }

    init(model: Model, context: MetalContext, maxContext: Int,
         runtimeConfiguration: RuntimeConfiguration,
         smallBlockPrefill: SmallBlockPrefillPolicy,
         sharedExpertOverlap: SharedExpertOverlapPolicy = .disabled) throws {
        let config = model.config
        if config.feedForwardKind == .mixtureOfExperts,
           let effectiveSlots = model.routedExpertCacheSlotCount(layer: 0),
           effectiveSlots < config.topKExperts {
            throw RuntimeConfigurationError.expertCacheTooSmall(
                configured: effectiveSlots,
                required: config.topKExperts)
        }
        // Compile the kernel groups this architecture dispatches before any
        // wrapper asks for a pipeline, so loading reads only what it needs.
        try context.prepareKernelGroups(MetalKernelGroup.required(
            for: model.config, sharedExpertWeightBits: model.sharedExpertWeightBits,
            maxContext: maxContext))
        self.model = model
        self.ctx = context
        self.cfg = config
        self.maxContext = maxContext
        self.qwenSparseAttention = config.attentionIndexer.isEnabled
            && maxContext >= config.attentionIndexer.budget + config.attentionIndexer.compressRatio
            ? try QwenSparseAttention(context: context, config: config, maxContext: maxContext) : nil
        // The fused head folds the final RMSNorm into the LM head. An
        // architecture whose "final norm" is a whole hyper-connection mixer —
        // two projections and a gated collapse over four streams — has nothing
        // to fold, so it takes the unfused path.
        let isPrism = config.variant == .bonsai2_2B || config.variant == .bonsai2_27B
        // Flint's fused head argmaxes an INT4 LM head from the FP32-normed row;
        // an 8-bit head (OptiQ) takes the logits path.
        let flintHeadIsInt4 = model.lmHead.length * 2
            == UInt64(config.vocabSize) * UInt64(config.hiddenSize)
        self.fusedGreedyHeadAvailable = runtimeConfiguration.headPath == .fusedRows
            && !config.hyperConnection.isEnabled
            && (config.family != .flint || flintHeadIsInt4)
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        self.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        self.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        let maximumVisionTokens = cfg.family.acceptsImageInput
            ? VisionConfig(family: cfg.family).maximumPooledTokens : 0
        let textChunkCapacity = max(PrefillRuntimeConfig.baselineChunkTokens,
                                    runtimeConfiguration.prefillChunkTokens)
        self.prefillTextChunkCapacity = textChunkCapacity
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: useFP16Ring,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: max(
                                        textChunkCapacity,
                                        min(maxContext, maximumVisionTokens)),
                                     initialFullAttentionCapacityTokens:
                                        KVCacheManager.initialFullAttentionCapacity)

        let silu = cfg.hiddenActivation == "silu"
        // The INT4 group size and the routed-expert count are architecture
        // facts, not runtime options: every quantized kernel below is
        // specialized on them, and at their historical values no function
        // constant is set at all.
        let int4Groups = cfg.int4GroupSize
        self.embedInt4 = try EmbedLookupInt4(context: context,
                                             groupSize: int4Groups)
        self.flintEmbedding = cfg.family == .flint ? try FlintEmbedding(context: context) : nil
        self.flintResidual = cfg.family == .flint ? try FlintResidual(context: context, width: cfg.hiddenSize) : nil
        self.flintRouter = cfg.family == .flint ? try FlintRouter(context: context, maximumExperts: cfg.numExperts) : nil
        self.flintFiniteLogits = cfg.family == .flint ? try FlintFiniteLogits(context: context) : nil
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(
            context: context,
            additionalShapes: cfg.decodeInt4GEMVShapes,
            groupSize: int4Groups)
        self.flintInt8 = cfg.family == .flint && model.sharedExpertWeightBits == 8
            ? try DequantInt8GEMV(context: context, wide: true) : nil
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: isPrism ? 4 : model.sharedExpertWeightBits,
                                                  siluActivation: silu,
                                                  groupSize: int4Groups,
                                                  wideInt8: cfg.family == .flint)
        self.moe       = cfg.feedForwardKind == .mixtureOfExperts
            ? try MoE(context: context,
                      siluActivation: silu,
                      specializedD: UInt32(cfg.hiddenSize),
                      specializedF: UInt32(cfg.moeIntermediateSize),
                      specializedNumExperts: UInt32(cfg.numExperts),
                      topKExperts: cfg.topKExperts,
                      groupSize: int4Groups,
                      wideIntermediate: cfg.family == .flint,
                      floatIntermediate: cfg.family == .flint)
            : nil
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize,
                                              groupSize: int4Groups)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context,
                                             additionalShapes: cfg.attnOutputGate ? [(
                                                qRows: 2 * cfg.numHeads * cfg.fullHeadDim,
                                                kvRows: cfg.numFullKVHeads * cfg.fullHeadDim,
                                                n: cfg.hiddenSize)] : [],
                                             groupSize: int4Groups)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.fusedPostAttentionSetup = try FusedPostAttentionSetup(context: context)
        self.fusedTail = try FusedLayerTail(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context,
                                                       groupSize: int4Groups)
        self.prefillHyperConnection = cfg.hyperConnection.isEnabled
            ? try PrefillHyperConnection(context: context) : nil
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context,
                                             groupSize: int4Groups)
        self.prefillMPPAffine = MPPPrefillAffineQMM(context: context, bits: 4,
                                                    groupSize: int4Groups)
        self.smallBlockPrefill = smallBlockPrefill
        self.sharedExpertOverlap = sharedExpertOverlap
        let smallBlockProjection = smallBlockPrefill.enabled
            ? try DequantInt4SmallBlock(context: context, groupSize: int4Groups)
            : nil
        self.smallBlockProjection = smallBlockProjection
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillPostAttention = try PrefillPostAttentionSetup(context: context)
        // The router keeps the default group: this architecture stores it as
        // INT8 at 64 regardless of the base INT4 group, exactly as the decode
        // path's router does.
        self.prefillRouter = try PrefillRouter(context: context)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: isPrism ? 4 : model.sharedExpertWeightBits,
            siluActivation: silu,
            groupSize: int4Groups,
            smallBlock: smallBlockProjection)
        if cfg.feedForwardKind == .mixtureOfExperts,
           let batchedRouted = PrefillBatchedRoutedExperts(context: context,
                                                           groupSize: int4Groups,
                                                           siluActivation: silu),
           batchedRouted.supports(d: cfg.hiddenSize, f: cfg.moeIntermediateSize) {
            self.prefillBatchedRouted = batchedRouted
            self.prefillBatchedRoutedScratch = try PrefillBatchedRoutedExperts.makeScratch(
                device: context.device, d: cfg.hiddenSize, f: cfg.moeIntermediateSize)
        } else {
            self.prefillBatchedRouted = nil
            self.prefillBatchedRoutedScratch = nil
        }
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context,
                                                             siluActivation: silu,
                                                             groupSize: int4Groups)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context,
                                                               maxD: cfg.hiddenSize,
                                                               groupSize: int4Groups)

        // Qwen 3.6 kernels, keyed off the data flags so architectures that
        // never dispatch them pay no PSO compile cost.
        // Hyper-connection architectures also fold the n-gram embedding into
        // the residual with a plain add, so they need this whether or not they
        // gate their attention output.
        let needsElementwise = cfg.attnOutputGate
            || cfg.hyperConnection.isEnabled
            || cfg.sharedExpertGated
            || !cfg.ffnSandwichNorms
            || cfg.hasLinearAttentionLayers
            || cfg.feedForwardKind == .dense
            || cfg.hasPerLayerInputs
        self.elementwise = needsElementwise ? try Elementwise(context: context) : nil
        if isPrism {
            let block = config.variant == .bonsai2_2B ? 512 : 1024
            self.prism = try PrismProjection(context: context, block: block,
                                             maximumWidth: max(config.hiddenSize, config.intermediateSize,
                                                               config.numHeads * config.fullHeadDim,
                                                               config.linearAttention.valueDim))
            self.prismGDN = try PrismGDN(context: context, config: config.linearAttention)
            guard let token = context.device.makeBuffer(length: MemoryLayout<Int32>.stride,
                                                        options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            self.prismToken = token
        } else {
            self.prism = nil
            self.prismGDN = nil
            self.prismToken = nil
        }
        if cfg.hasLinearAttentionLayers {
            self.gdn = try GDN(context: context, config: cfg.linearAttention,
                               specializedHiddenSize: cfg.hiddenSize,
                               groupSize: int4Groups)
            self.gdnState = try GDNStateManager(device: context.device, config: cfg)
        } else {
            self.gdn = nil
            self.gdnState = nil
        }
        self.rope = (cfg.ropeNeoxSubdim || cfg.numKVSharedLayers > 0)
            ? try RoPE(context: context) : nil
        self.perLayerEmbeddingKernel = cfg.hasPerLayerInputs
            ? try PerLayerEmbedding(context: context) : nil
        self.int8ScalarGate = cfg.sharedExpertGated
            ? try DequantInt8GEMV(context: context,
                                  additionalShapes: cfg.decodeInt8GEMVShapes)
            : nil

        let device = context.device
        let D = cfg.hiddenSize
        // Scratch is shared across layers, so it is sized for the widest
        // feed-forward block rather than the nominal one. E2B's last 20 layers
        // are twice as wide as its first 15.
        let F = cfg.maxFFNIntermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return b
        }
        // The residual is `streamCount` wide for an architecture with
        // hyper-connections and one stream wide otherwise, so the whole layer
        // loop can address it the same way.
        let streams = max(1, cfg.hyperConnection.streamCount)
        self.hidden        = try buf(D * streams)
        self.normed        = try buf(D * streams)
        if cfg.ngramEmbedding.isEnabled {
            let ngram = cfg.ngramEmbedding
            self.ngramPLE = try NgramPLE(context: ctx)
            self.ngramRows = try model.makeNgramRowIndex(
                eosTokenID: ngram.eosTokenID)
            self.ngramTable = try model.makeNgramTableReader()
            self.pleEmbedding = try buf(ngram.embedDim)
            self.pleKeys = try buf(D * streams)
            self.pleValues = try buf(D)
            self.pleQueries = try buf(D * streams)
            self.pleGated = try buf(D * streams)
            self.pleNormed = try buf(D * streams)
            self.pleConvOut = try buf(D * streams)
            let historyRows = (ngram.convKernelSize - 1) * ngram.ngramSize + 1
            let history = try buf(historyRows * D * streams)
            // A fresh Metal buffer is not guaranteed zeroed, and the first
            // token convolves over the whole window — so the taps behind it
            // have to start at zero the way the reference's state does.
            memset(history.contents(), 0, history.length)
            self.pleConvHistory = history
            self.ngramContext = self.ngramRows?.initialContext() ?? []
        } else {
            self.ngramPLE = nil
            self.ngramRows = nil
            self.ngramTable = nil
            self.pleEmbedding = nil
            self.pleKeys = nil
            self.pleValues = nil
            self.pleQueries = nil
            self.pleGated = nil
            self.pleNormed = nil
            self.pleConvOut = nil
            self.pleConvHistory = nil
        }
        if cfg.hyperConnection.isEnabled {
            self.hyperConnection = try HyperConnection(context: ctx)
            self.hcMixed   = try buf(D)
            self.hcLowRank = try buf(cfg.hyperConnection.lowRank)
            self.hcMixUp   = try buf(D * streams)
            self.hcInject  = try buf(streams)
        } else {
            self.hyperConnection = nil
            self.hcMixed = nil
            self.hcLowRank = nil
            self.hcMixUp = nil
            self.hcInject = nil
        }
        self.attnOut       = try buf(maxQ)
        self.qScratch      = try buf(maxQ)
        self.kStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.vStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.oOut          = try buf(D)
        self.h1Buf         = try buf(D)
        self.h2Buf         = try buf(D)
        self.routedX       = try buf(D)
        self.denseX        = try buf(D)
        self.denseScratchGate = try buf(F)
        self.denseScratchUp   = try buf(F)
        self.denseScratchAct  = try buf(F)
        self.routerInput   = try buf(D)
        self.zeroResidual  = try buf(D)
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices    = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.outWeights    = try buf(cfg.topKExperts)
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize,
                                      cfg.family == .flint ? MemoryLayout<Float>.size : MemoryLayout<Float16>.size)
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.greedyTokenBuf = tok
        self.speculativeTokenBuffer = try buf(8, MemoryLayout<UInt32>.size)
        self.speculativeTargetTokenBuf = try buf(8, MemoryLayout<UInt32>.size)

        // Qwen 3.6 decode scratch — allocated once here, never in the hot path.
        if cfg.attnOutputGate {
            self.qPackedScratch = try buf(2 * maxQ)
            self.attnGateScratch = try buf(maxQ)
        } else {
            self.qPackedScratch = nil
            self.attnGateScratch = nil
        }
        if cfg.hasLinearAttentionLayers {
            let la = cfg.linearAttention
            self.gdnQKVRaw = try buf(la.qkvDim)
            self.gdnConvOut = try buf(la.qkvDim)
            self.gdnZ = try buf(la.valueDim)
            self.gdnA = try buf(la.numVHeads, isPrism ? MemoryLayout<Float>.stride : MemoryLayout<Float16>.stride)
            self.gdnB = try buf(la.numVHeads, isPrism ? MemoryLayout<Float>.stride : MemoryLayout<Float16>.stride)
            self.gdnY = try buf(la.valueDim)
            self.gdnOut = try buf(la.valueDim)
        } else {
            self.gdnQKVRaw = nil
            self.gdnConvOut = nil
            self.gdnZ = nil
            self.gdnA = nil
            self.gdnB = nil
            self.gdnY = nil
            self.gdnOut = nil
        }
        self.sharedScalarGateBuf = cfg.sharedExpertGated ? try buf(1) : nil
        if cfg.hasPerLayerInputs {
            let packedPLE = cfg.numLayers * cfg.hiddenSizePerLayerInput
            self.pleIdentity = try buf(packedPLE)
            self.pleContext = try buf(packedPLE)
            self.pleLayer = try buf(cfg.hiddenSizePerLayerInput)
            self.pleGate = try buf(cfg.hiddenSizePerLayerInput)
        } else {
            self.pleIdentity = nil
            self.pleContext = nil
            self.pleLayer = nil
            self.pleGate = nil
        }

        // The validated tensor is authoritative for each projection's shape.
        // Scratch uses the model-wide maximum, but E2B's first 15 MLPs are
        // narrower than its final 20; stamping the maximum onto every view
        // makes those early GEMVs read into the next packed tensor.
        func sharedProj(_ view: TensorView) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: view.shape.0,
                                 cols: view.shape.1)
        }
        var sharedViews: [LayerSharedExpertProjections?] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            guard cfg.hasSharedExpert else {
                sharedViews.append(nil)
                continue
            }
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate),
                up: sharedProj(up),
                down: sharedProj(down),
                postF1: cfg.feedForwardKind == .dense && cfg.family == .gemma4
                    ? try model.postFFN(layer: L)
                    : (cfg.ffnSandwichNorms ? try model.postFFN1(layer: L) : nil),
                scalarGate: cfg.sharedExpertGated
                    ? try model.sharedExpertScalarGate(layer: L) : nil))
        }
        self.sharedExpertProjections = sharedViews

        func bf16OnesBuffer(count: Int, label: String) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: count * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = 0x3F80 }  // BF16 1.0
            buf.label = label
            return buf
        }

        if cfg.feedForwardKind == .dense {
            self.effectiveScaleBuffers = []
            self.onesPerExpertScale = nil
        } else if cfg.routerScaled {
            // Pre-fold 1/sqrt(D) into router.scale per layer. Each layer gets
            // its own BF16 [D] buffer — the kernel reads `effective_scale[i]`
            // and we pay for the multiply once per generation, not per token.
            var perLayer: [MTLBuffer] = []
            perLayer.reserveCapacity(cfg.numLayers)
            let invSqrtD = Float(1.0) / Float(D).squareRoot()
            let dInts = D
            for L in 0..<cfg.numLayers {
                let scaleView = try model.routerScale(layer: L)
                guard let buf = device.makeBuffer(length: dInts * MemoryLayout<UInt16>.size,
                                                  options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                let src = scaleView.buffer.contents()
                    .advanced(by: Int(scaleView.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<dInts {
                    let v = Quantization.bf16ToFloat(src[i]) * invSqrtD
                    dst[i] = Quantization.bf16Bits(v)
                }
                buf.label = "effective_scale.L\(L)"
                perLayer.append(buf)
            }
            self.effectiveScaleBuffers = perLayer
            self.onesPerExpertScale = nil
        } else {
            // Plain linear router (Qwen): one shared BF16 ones buffer keeps
            // the router kernel's effective_scale multiply neutral, and a ones
            // per_expert_scale keeps the top-k weights untouched. (Softmax
            // over top-k then renormalize equals Qwen's softmax over all
            // experts then renormalize the selected top-k.)
            let ones = try bf16OnesBuffer(count: D, label: "effective_scale.ones")
            self.effectiveScaleBuffers = [MTLBuffer](repeating: ones,
                                                     count: cfg.numLayers)
            self.onesPerExpertScale = try bf16OnesBuffer(count: cfg.numExperts,
                                                         label: "per_expert_scale.ones")
        }
    }

    public func reset() {
        _ = lookahead.drain()
        kv?.reset()
        gdnState?.reset()
        resetNgramState()
        qwenMultimodalRopeDelta = 0
        speculativeStartPosition = nil
        speculativeProcessedTokens = 0
        resetTransientState()
    }

    /// Clear the n-gram hash context and the convolution history, so a new
    /// sequence starts from EOS padding and zeroed taps rather than carrying
    /// the previous one's tokens into its first n-grams.
    private func resetNgramState() {
        guard let rowIndex = ngramRows else { return }
        ngramContext = rowIndex.initialContext()
        if let history = pleConvHistory {
            memset(history.contents(), 0, history.length)
        }
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        speculativeStartPosition = nil
        speculativeProcessedTokens = 0
        resetTransientState()
    }

    // MARK: Retained conversation state

    /// Layers whose state a snapshot carries. Layers that read another
    /// layer's KV write none of their own and are skipped.
    private func snapshotSkipsLayer(_ layer: Int) -> Bool {
        cfg.layerSharesKV(layer)
    }

    private var sparseIndexerLayers: [Int] {
        guard qwenSparseAttention != nil, let kv else { return [] }
        return (0..<cfg.numLayers).filter { kv.layerKind($0) != .linear }
    }

    /// Everything carried from one token to the next: KV rows, Gated-DeltaNet
    /// recurrent and convolution state, the sparse indexer's keys, the n-gram
    /// convolution history, and the host-side n-gram context and multimodal
    /// RoPE offset. Prefill chunk state, read-advice state and expert
    /// prediction are per-request and rebuilt on resume.
    private func snapshotBuilder(position: Int) throws -> RunnerStateSnapshotBuilder {
        var builder = RunnerStateSnapshotBuilder()
        kv?.addSnapshotRanges(to: &builder, position: position,
                              skipLayer: snapshotSkipsLayer)
        gdnState?.addSnapshotRanges(to: &builder)
        if let qwenSparseAttention {
            try qwenSparseAttention.addSnapshotRanges(
                to: &builder, layers: sparseIndexerLayers, position: position)
        }
        if let pleConvHistory {
            builder.add("ngram.conv", pleConvHistory, length: pleConvHistory.length)
        }
        return builder
    }

    // Accessors the snapshot extension needs; the state itself stays private.
    var speculativeTransactionIsIdle: Bool { speculativeStartPosition == nil }
    var commandQueueForSnapshots: MTLCommandQueue { ctx.queue }
    var ngramContextForSnapshot: [Int32] { ngramContext }
    var ropeDeltaForSnapshot: Int32 { qwenMultimodalRopeDelta }
    func snapshotSkipsLayerForEstimate(_ layer: Int) -> Bool { snapshotSkipsLayer(layer) }
    var gdnStateSnapshotBytes: Int { gdnState?.snapshotBytes ?? 0 }
    var ngramHistorySnapshotBytes: Int {
        pleConvHistory.map { RunnerStateSnapshotBuilder.aligned($0.length) } ?? 0
    }
    func sparseIndexerSnapshotBytes(position: Int) -> Int {
        qwenSparseAttention?.snapshotBytes(layers: sparseIndexerLayers.count,
                                           position: position) ?? 0
    }
    func snapshotBuilderForCapture(position: Int) throws -> RunnerStateSnapshotBuilder {
        try snapshotBuilder(position: position)
    }

    /// Writes a snapshot into a runner that has just been reset.
    func applySnapshot(_ snapshot: RunnerStateSnapshot) throws {
        let position = snapshot.host.position
        if let kv {
            guard position <= kv.maxContext else {
                throw RunnerStateSnapshotError.layoutMismatch("position exceeds context")
            }
            if position > 0 { try kv.ensureCapacity(through: position, on: ctx.queue) }
        } else if position != 0 {
            throw RunnerStateSnapshotError.layoutMismatch("runner has no KV cache")
        }
        if ngramRows != nil {
            guard snapshot.host.ngramContext.count == ngramContext.count else {
                throw RunnerStateSnapshotError.layoutMismatch("n-gram context length")
            }
        }
        try snapshotBuilder(position: position).restore(snapshot, queue: ctx.queue)
        kv?.restorePosition(position)
        if ngramRows != nil { ngramContext = snapshot.host.ngramContext }
        qwenMultimodalRopeDelta = snapshot.host.ropeDelta
        resetTransientState()
    }

    private func resetTransientState() {
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
    }

    public private(set) var totalIoNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    /// GPU-side execution time (gpuStartTime→gpuEndTime) of the per-layer
    /// command buffers and of the head, for TUFF_PHASES diagnostics.
    public private(set) var totalGPULayerNanos: UInt64 = 0
    public private(set) var totalGPUHeadNanos: UInt64 = 0
    public private(set) var totalGPULayerCommandBuffers: UInt64 = 0
    public private(set) var totalGPURoutedNanos: UInt64 = 0
    /// Shared-expert submissions, including any cache-hit phase 1 work folded into them.
    public private(set) var totalGPUSharedNanos: UInt64 = 0
    /// Decode-only routed-expert traffic. A read is a cache miss that required
    /// an SSD-backed expert fetch; cache hits are reported separately.
    public private(set) var totalRoutedExpertReads: UInt64 = 0
    public private(set) var totalRoutedExpertBytes: UInt64 = 0
    public private(set) var totalRoutedExpertCacheHits: UInt64 = 0
    public private(set) var totalRoutedExpertCacheMisses: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }

    public func selectHead(pureGreedy: Bool) {
        fusedGreedyHeadSelected = pureGreedy
    }

    /// Projections encoded through the small-block prefill path since the
    /// runner was created. Zero whenever the policy is off.
    var smallBlockEncodedProjections: Int {
        smallBlockProjection?.encodedProjections ?? 0
    }
    /// Attention and GDN projections the batched MPP path has encoded, for
    /// tests that check which path a short prefill block took.
    var batchedPrefillProjections: Int {
        prefillMPPAffine?.encodedProjections ?? 0
    }
    /// The Qwen Gated-DeltaNet state is recurrent rather than rewindable KV,
    /// so this POC advertises verification only for the fused-head affine
    /// path without linear-attention layers.
    public var supportsSpeculativeVerification: Bool {
        useFusedGreedyHead && gdnState == nil && cfg.family != .flint
    }
    public private(set) var totalRDAdviseNanos: UInt64 = 0
    public private(set) var totalRDAdviseCalls: UInt64 = 0
    public private(set) var totalRDAdviseBytes: UInt64 = 0
    public private(set) var totalRDAdviseFailures: UInt64 = 0
    public private(set) var totalRDAdviseSkipped: UInt64 = 0

    private func recordRDAdvice(_ result: ExpertIOAdviceResult, wallNanos: UInt64) {
        totalRDAdviseNanos &+= wallNanos
        totalRDAdviseCalls &+= UInt64(result.calls)
        totalRDAdviseBytes &+= result.bytes
        totalRDAdviseFailures &+= UInt64(result.failed)
        totalRDAdviseSkipped &+= UInt64(result.skipped)
    }

    private func shouldSkipRDAdvice(position: Int,
                                    requestedMisses: Int,
                                    estimatedBytes: UInt64,
                                    canOverlapUsefulGPUWork: Bool) -> ExpertIOAdviceResult? {
        switch rdadvisePolicyMode {
        case .bounded:
            if position <= rdadviseSkipUntilPosition {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            if requestedMisses > Self.rdadviseBoundedMissCap {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            return nil
        case .adaptive:
            if position != rdadviseAdaptivePosition {
                rdadviseAdaptivePosition = position
                rdadviseAdaptivePositionBytes = 0
            }
            let cumulativeEstimatedBytes = rdadviseAdaptivePositionBytes &+ estimatedBytes
            let shouldSkip = rdadviseAdaptiveState.shouldSkip(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: cumulativeEstimatedBytes,
                canOverlapUsefulGPUWork: canOverlapUsefulGPUWork)
            rdadviseAdaptivePositionBytes = cumulativeEstimatedBytes
            guard shouldSkip else { return nil }
            return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                bytes: estimatedBytes)
        case .default, .off:
            return nil
        }
    }

    private func updateRDAdvicePolicy(after result: ExpertIOAdviceResult,
                                      position: Int) {
        switch rdadvisePolicyMode {
        case .bounded:
            if result.maxCallNanos > Self.rdadviseBoundedMaxCallNanos {
                rdadviseSkipUntilPosition = max(rdadviseSkipUntilPosition, position + 1)
            }
        case .adaptive:
            rdadviseAdaptiveState.update(after: result, position: position)
        case .default, .off:
            break
        }
    }

    public func verifySpeculativeBlock(tokens: [Int32],
                                       startPosition: Int,
                                       into logits: MTLBuffer) async throws
        -> SpeculativeVerificationResult {
        guard supportsSpeculativeVerification else {
            throw PrefillError.chunkedUnsupported(
                "speculative verification requires the fused affine head without GDN state")
        }
        guard (1...8).contains(tokens.count) else {
            throw SpeculativeDecodingError.invalidBlockSize(
                requested: tokens.count, maximum: 8)
        }
        guard speculativeStartPosition == nil else {
            throw PrefillError.chunkedRunnerDirty(
                "a speculative verification transaction is already active")
        }
        guard let kv, kv.position == startPosition else {
            throw SpeculativeDecodingError.invalidStartPosition(
                expected: kv?.position ?? 0, actual: startPosition)
        }
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < cfg.vocabSize }) else {
            throw GeneratorError.invalidGenerationConfig(
                "speculative token is outside the model vocabulary")
        }
        guard let scratch = try? ensurePrefillScratch(
            config: .production(chunkTokens: 32)) else {
            throw PrefillError.chunkedUnsupported(
                "speculative verification could not allocate bounded prefill scratch")
        }

        speculativeExpertReads = 0
        speculativeExpertBytes = 0
        speculativeExpertCacheHits = 0
        speculativeExpertCacheMisses = 0
        speculativeTargetCommandBuffers = 0
        collectingSpeculativeMetrics = true
        defer { collectingSpeculativeMetrics = false }

        let candidatePointer = speculativeTokenBuffer
            .contents().assumingMemoryBound(to: UInt32.self)
        for (index, token) in tokens.enumerated() {
            candidatePointer[index] = UInt32(bitPattern: token)
        }
        let boundaryToken = lastGreedyToken
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        do {
            try await executePrefillChunk(
                tokens: tokens[...],
                startPosition: startPosition,
                outputMode: .greedyIfAvailable,
                logits: logits,
                scratch: scratch,
                config: .production(chunkTokens: 32),
                writeFinalHead: true,
                speculativeTargetTokens: speculativeTargetTokenBuf,
                tokenBufferOverride: speculativeTokenBuffer)
        } catch {
            // The chunk path marks itself dirty before dispatching layers. A
            // failed/cancelled verification must restore both logical state
            // and the reusable prefill scratch state before propagating.
            kv.rewind(to: startPosition)
            prefillChunkState.markCommitted()
            throw error
        }
        let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start

        let targetPointer = speculativeTargetTokenBuf
            .contents().assumingMemoryBound(to: UInt32.self)
        var targetTokens = [Int32(bitPattern: boundaryToken)]
        targetTokens.reserveCapacity(tokens.count + 1)
        for index in 0..<tokens.count {
            targetTokens.append(Int32(bitPattern: targetPointer[index]))
        }
        speculativeStartPosition = startPosition
        speculativeProcessedTokens = tokens.count
        return SpeculativeVerificationResult(
            startPosition: startPosition,
            proposedTokenIDs: tokens,
            targetTokenIDs: targetTokens,
            processedTokens: tokens.count,
            newPosition: startPosition + tokens.count,
            metrics: SpeculativeVerificationMetrics(
                wallNanos: wallNanos,
                targetCommandBuffers: speculativeTargetCommandBuffers,
                expertReads: speculativeExpertReads,
                expertBytes: speculativeExpertBytes,
                expertCacheHits: speculativeExpertCacheHits,
                expertCacheMisses: speculativeExpertCacheMisses))
    }

    public func speculativeBoundaryToken() async throws -> Int32? {
        guard supportsSpeculativeVerification else { return nil }
        return Int32(bitPattern: lastGreedyToken)
    }

    private func recordSpeculativeFetch(layer: Int,
                                        fetch: PrefillStreamedTileFetchResult) {
        guard collectingSpeculativeMetrics else { return }
        let misses = fetch.usedPlannedFetch
            ? fetch.plannedMissSlots.count
            : fetch.expertIDs.count
        let hits = fetch.usedPlannedFetch ? fetch.plannedHits : 0
        recordDecodeExpertFetch(layer: layer, misses: misses, hits: hits)
        speculativeExpertReads &+= UInt64(misses)
        speculativeExpertCacheMisses &+= UInt64(misses)
        speculativeExpertCacheHits &+= UInt64(hits)
        if let bytes = try? model.routedExpertAdviceByteEstimate(
            layer: layer, missCount: misses) {
            speculativeExpertBytes &+= bytes
        }
    }

    /// The label is what a watchdog or driver failure reports, so it names
    /// the chunk, layer and phase that was running.
    private func makePrefillCommandBuffer(_ label: String) -> MTLCommandBuffer? {
        let commandBuffer = ctx.queue.makeCommandBuffer()
        if prism != nil, let commandBuffer { SharedComputeEncoding.share(commandBuffer) }
        commandBuffer?.label = label
        if collectingSpeculativeMetrics, commandBuffer != nil {
            speculativeTargetCommandBuffers &+= 1
        }
        return commandBuffer
    }

    private func recordDecodeExpertFetch(layer: Int,
                                         plan: RoutedExpertFetchPlan?) {
        guard let plan else { return }
        recordDecodeExpertFetch(layer: layer,
                                misses: plan.misses.count,
                                hits: plan.hits)
    }

    private func recordDecodeExpertFetch(layer: Int,
                                         misses: Int,
                                         hits: Int) {
        guard misses >= 0, hits >= 0 else { return }
        totalRoutedExpertReads &+= UInt64(misses)
        totalRoutedExpertCacheMisses &+= UInt64(misses)
        totalRoutedExpertCacheHits &+= UInt64(hits)
        if let bytes = try? model.routedExpertAdviceByteEstimate(
            layer: layer, missCount: misses) {
            totalRoutedExpertBytes &+= bytes
        }
    }

    public func commitSpeculativePrefix(_ count: Int) throws {
        guard let start = speculativeStartPosition else {
            throw SpeculativeDecodingError.noActiveTransaction
        }
        guard (0...speculativeProcessedTokens).contains(count) else {
            throw SpeculativeDecodingError.invalidCommitCount(
                requested: count, processed: speculativeProcessedTokens)
        }
        kv?.rewind(to: start + count)
        speculativeStartPosition = nil
        speculativeProcessedTokens = 0
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer,
                        appendingToFinalCommandBuffer epilogue: (MTLCommandBuffer) -> Void)
        async throws -> Bool {
        try prefillChunkState.requireClean(operation: "produce")
        var encoded = false
        // `produceToken` calls the epilogue during its own execution, before
        // returning, so borrowing the non-escaping closure is safe here.
        try await withoutActuallyEscaping(epilogue) { epilogue in
            try await produceToken(token: token, position: position, into: logits,
                                   emitHead: true, outputMode: .greedyIfAvailable,
                                   epilogue: { cb in
                                       encoded = true
                                       epilogue(cb)
                                   })
        }
        return encoded
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    /// Flint retains its separate decode path and FP32 residual representation.
    var supportsChunkedPrefill: Bool { true }

    var supportsHeadSkippingPrefill: Bool { prism != nil }

    func consumePrefillToken(_ token: Int32, position: Int, scratchLogits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "consumePrefillToken")
        try await produceToken(token: token, position: position, into: scratchLogits,
                               emitHead: false, outputMode: .logits)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        if cfg.family == .flint {
            return try await prefillFlintBatched(tokens: tokens, startPosition: startPosition,
                                                 outputMode: outputMode, config: config,
                                                 into: logits, onProgress: onProgress)
        }
        let config = config.replacingChunkTokens(
            min(config.chunkTokens, prefillTextChunkCapacity))
        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        try await PrefillSpanIteration.forEachSpan(spans) { spanIndex, span in
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: config,
                writeFinalHead: spanIndex == spans.count - 1)
            onProgress(span.completedCount)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }


    public func prefillMultimodal(input: MultimodalPrefillInput,
                                  startPosition: Int,
                                  outputMode: PrefillOutputMode,
                                  config: PrefillRuntimeConfig,
                                  into logits: MTLBuffer,
                                  onProgress: (Int) -> Void) async throws -> PrefillResult {
        try await prefillMultimodal(
            input: input, startPosition: startPosition, outputMode: outputMode,
            config: config, into: logits, chunkGate: nil, onProgress: onProgress)
    }

    /// `chunkGate` runs one planned forward chunk (a text span or one image
    /// span) so a caller sharing the model can hold its fairness lease for
    /// exactly that chunk.
    public func prefillMultimodal(input: MultimodalPrefillInput,
                                  startPosition: Int,
                                  outputMode: PrefillOutputMode,
                                  config: PrefillRuntimeConfig,
                                  into logits: MTLBuffer,
                                  chunkGate: PrefillChunkGate?,
                                  onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillMultimodal")
        // The same property `coercedForImagePrompt()` repairs, so the guard and
        // the coercion cannot drift apart.
        guard config.servesImagePrompt else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill requires the complete chunked prefill path")
        }
        // Image embeddings override the token gather, but the n-gram PLE
        // still hashes the original image-pad IDs, including their history
        // across the image/text boundary. Hashing the gather's dummy zeros
        // injects unrelated text features into every image residual.
        let tokens = cfg.ngramEmbedding.isEnabled
            ? input.effectiveTokenIDs : input.embeddingTokenIDs
        guard startPosition >= 0,
              startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill range exceeds maxContext \(maxContext)")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        guard input.imageSpans.allSatisfy({
            $0.features.hiddenSize == cfg.hiddenSize
                && $0.features.tokenCount == $0.tokenRange.count
        }) else {
            throw PrefillError.chunkedUnsupported(
                "image features do not match the loaded model architecture")
        }

        // The planner applies the same clamp the scratch layout does. Cutting at
        // the raw `config.chunkTokens` let a caller size a 280-token scratch and
        // then hand it a larger chunk, which died on the guard.
        let config = config.replacingChunkTokens(
            min(config.chunkTokens, prefillTextChunkCapacity))
        let textChunkTokens = config.chunkTokens
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: tokens.count,
            imageRanges: input.imageSpans.map(\.tokenRange),
            chunkTokens: config.chunkTokens)
        let imageBuffers = input.imageSpans.map(\.features.buffer)

        // One layout for the whole turn, sized for the largest item in it.
        // Keying the single-slot cache on each item's exact chunk size made a
        // text-image-text prompt free and reallocate the entire chunk scratch at
        // every boundary, twice per image span, every turn.
        let pooledTokens = input.imageSpans.map(\.tokenRange.count).max() ?? 0
        let hasImages = work.contains(where: \.isImage)
        let layoutTokens = hasImages
            ? max(config.chunkTokens, pooledTokens)
            : config.chunkTokens
        let layoutLimit = hasImages
            ? max(PrefillRuntimeConfig.maxChunkTokens, pooledTokens)
            : PrefillRuntimeConfig.maxChunkTokens
        let scratch = try ensurePrefillScratch(
            config: config.replacingChunkTokens(layoutTokens),
            chunkTokenLimit: layoutLimit)

        var resolvedPositions = input.positionIDs
        var resolvedRopeDelta = input.ropeDelta
        if let positions = resolvedPositions, startPosition > 0,
           positions.temporal.first == 0,
           positions.height.first == 0,
           positions.width.first == 0 {
            let base = Int32(startPosition) + qwenMultimodalRopeDelta
            resolvedRopeDelta = qwenMultimodalRopeDelta + positions.ropeDelta
            resolvedPositions = try positions.offsettingPositions(
                by: base, ropeDelta: resolvedRopeDelta)
        }

        var completed = 0
        for (index, item) in work.enumerated() {
            try Task.checkCancellation()
            let isImage = item.isImage
            let imageFeatures = item.imageIndex.map { imageBuffers[$0] }
            // Every item executes under the chunk size it was planned at.
            let chunkConfig = config.replacingChunkTokens(
                isImage ? item.range.count : textChunkTokens)
            let lower = tokens.index(tokens.startIndex, offsetBy: item.range.lowerBound)
            let upper = tokens.index(tokens.startIndex, offsetBy: item.range.upperBound)
            let chunkStart = startPosition + item.range.lowerBound
            let chunkPositions = try resolvedPositions?.slice(item.range)
            let run: () async throws -> Void = {
                try await self.executePrefillChunk(
                    tokens: tokens[lower..<upper],
                    startPosition: chunkStart,
                    outputMode: outputMode,
                    logits: logits,
                    scratch: scratch,
                    config: chunkConfig,
                    writeFinalHead: index == work.count - 1,
                    embeddingOverride: imageFeatures,
                    bidirectionalBlock: isImage && input.usesBidirectionalImageAttention
                        ? chunkStart..<(startPosition + item.range.upperBound)
                        : nil,
                    positionIDs: chunkPositions)
            }
            if let chunkGate { try await chunkGate(run) } else { try await run() }
            completed += item.range.count
            onProgress(completed)
        }
        if input.family.usesQwenVisionTower {
            qwenMultimodalRopeDelta = resolvedRopeDelta
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    @discardableResult
    private func ensurePrefillScratch(
        config: PrefillRuntimeConfig,
        chunkTokenLimit: Int = PrefillRuntimeConfig.maxChunkTokens
    ) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg,
                                               chunkTokens: config.chunkTokens,
                                               chunkTokenLimit: chunkTokenLimit)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    /// Copy the chunk's last-token residual out of device-private scratch so a
    /// diagnostic can read it. Debug-only: it commits and waits.
    /// Test-only: read an FP16 device buffer that is already shared-storage.
    private func readHalfBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        buffer.contents().withMemoryRebound(to: Float16.self, capacity: count) {
            pointer in (0..<count).map { Float(pointer[$0]) }
        }
    }

    private func readChunkResidualTail(scratch: PrefillChunkScratchBuffers,
                                       tokens: Int,
                                       streams: Int) throws -> [Float] {
        let width = cfg.hiddenSize * streams
        let bytes = width * MemoryLayout<Float16>.stride
        guard let staging = ctx.device.makeBuffer(length: bytes,
                                                  options: .storageModeShared),
              let cb = ctx.queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(from: scratch.hidden,
                  sourceOffset: (tokens - 1) * bytes,
                  to: staging, destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cb.commit()
        try waitForCompletion(cb)
        return staging.contents().withMemoryRebound(
            to: Float16.self, capacity: width
        ) { pointer in
            (0..<width).map { Float(pointer[$0]) }
        }
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool,
                                     embeddingOverride: MTLBuffer? = nil,
                                     bidirectionalBlock: Range<Int>? = nil,
                                     positionIDs: MultimodalPositionIDs? = nil,
                                     speculativeTargetTokens: MTLBuffer? = nil,
                                     tokenBufferOverride: MTLBuffer? = nil) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires FP16 KV")
        }
        // Speculative verification reuses this path during decode; only a
        // prompt is timed as prefill.
        prefillClock.active = speculativeTargetTokens == nil
        defer { prefillClock.active = false }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        try kv?.ensureCapacity(through: startPosition + tokens.count, on: ctx.queue)
        _ = lookahead.drain()
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }

        struct LayerPrefillQKVViews {
            // A hyper-connection architecture has neither: its two
            // hyper-connection modules carry the only norms in the layer.
            let inputNorm: TensorView?
            let postAttention: TensorView?
            let router: TensorView?
            // Softmax-attention layers only (nil on linear-attention layers).
            let q: TensorView?
            let k: TensorView?
            let v: TensorView?
            let o: TensorView?
            let qNorm: TensorView?
            let kNorm: TensorView?
            // Gemma FFN sandwich only.
            let preFFN: TensorView?
            let preFFN2: TensorView?
            let postFFN2: TensorView?
            let postFFN: TensorView?
            let layerScalar: TensorView?
            let routerPerExpertScale: TensorView?
            // Dense Gemma PLE mapping.
            let perLayerInputGate: TensorView?
            let perLayerProjection: TensorView?
            let postPerLayerInputNorm: TensorView?
            // Gated-DeltaNet linear-attention layers only.
            let linQKV: TensorView?
            let linZ: TensorView?
            let linA: TensorView?
            let linB: TensorView?
            let linOut: TensorView?
            let linConv: TensorView?
            let linALog: TensorView?
            let linDtBias: TensorView?
            let linNorm: TensorView?
        }

        let layerViews = try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isLinear = cfg.layerIsLinear(L)
            let dense = cfg.feedForwardKind == .dense && cfg.family == .gemma4
            let sandwich = cfg.ffnSandwichNorms && !dense
            let sharesKV = cfg.layerSharesKV(L)
            return LayerPrefillQKVViews(
                inputNorm: cfg.hyperConnection.isEnabled
                    ? nil : try model.inputNorm(layer: L),
                postAttention: cfg.hyperConnection.isEnabled
                    ? nil : try model.postAttnNorm(layer: L),
                router: cfg.feedForwardKind == .dense ? nil : try model.router(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: (isLinear || sharesKV) ? nil : try model.kProj(layer: L),
                v: (isLinear || sharesKV) ? nil
                    : ((isFull && cfg.attentionKEqV)
                        ? (try model.kProj(layer: L))
                        : (try model.vProj(layer: L))),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: isLinear ? nil : try model.qNorm(layer: L),
                kNorm: (isLinear || sharesKV) ? nil : try model.kNorm(layer: L),
                preFFN: (sandwich || dense) ? try model.preFFN(layer: L) : nil,
                preFFN2: sandwich ? try model.preFFN2(layer: L) : nil,
                postFFN2: sandwich ? try model.postFFN2(layer: L) : nil,
                postFFN: (sandwich || dense) ? try model.postFFN(layer: L) : nil,
                layerScalar: (sandwich || dense) ? try model.layerScalar(layer: L) : nil,
                routerPerExpertScale: cfg.routerScaled
                    && !dense ? try model.routerPerExpertScale(layer: L) : nil,
                perLayerInputGate: dense && cfg.hasPerLayerInputs
                    ? try model.perLayerInputGate(layer: L) : nil,
                perLayerProjection: dense && cfg.hasPerLayerInputs
                    ? try model.perLayerProjection(layer: L) : nil,
                postPerLayerInputNorm: dense && cfg.hasPerLayerInputs
                    ? try model.postPerLayerInputNorm(layer: L) : nil,
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: isLinear ? try model.linearInProjZ(layer: L) : nil,
                linA: isLinear ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear ? try model.linearOutProj(layer: L) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear ? try model.linearALog(layer: L) : nil,
                linDtBias: isLinear ? try model.linearDtBias(layer: L) : nil,
                linNorm: isLinear ? try model.linearNorm(layer: L) : nil)
        }

        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        let tokenBuffer: MTLBuffer
        if let tokenBufferOverride {
            guard tokenBufferOverride.length
                >= tokenIDs.count * MemoryLayout<UInt32>.stride else {
                throw ModelError.residentBufferWrapFailed
            }
            tokenIDs.withUnsafeBytes { bytes in
                if let baseAddress = bytes.baseAddress {
                    memcpy(tokenBufferOverride.contents(), baseAddress, bytes.count)
                }
            }
            tokenBuffer = tokenBufferOverride
        } else {
            guard let allocated = ctx.device.makeBuffer(
                bytes: tokenIDs,
                length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            tokenBuffer = allocated
        }
        // A text-only follow-up still uses the position offset established by
        // the previous image. Decode already applies it; chunked prefill must
        // use the same positions when it appends several text tokens at once.
        let positionIDs: MultimodalPositionIDs? = try positionIDs ?? {
            guard cfg.family.usesQwenVisionTower, qwenMultimodalRopeDelta != 0 else { return nil }
            let values = (0..<tokens.count).map { Int32(startPosition + $0) + qwenMultimodalRopeDelta }
            return try MultimodalPositionIDs(temporal: values, height: values,
                width: values, ropeDelta: qwenMultimodalRopeDelta)
        }()
        let mropeBuffers: (temporal: MTLBuffer, height: MTLBuffer, width: MTLBuffer)?
        if let positionIDs {
            guard positionIDs.count == tokens.count,
                  let temporal = ctx.device.makeBuffer(
                    bytes: positionIDs.temporal,
                    length: positionIDs.count * MemoryLayout<Int32>.stride,
                    options: .storageModeShared),
                  let height = ctx.device.makeBuffer(
                    bytes: positionIDs.height,
                    length: positionIDs.count * MemoryLayout<Int32>.stride,
                    options: .storageModeShared),
                  let width = ctx.device.makeBuffer(
                    bytes: positionIDs.width,
                    length: positionIDs.count * MemoryLayout<Int32>.stride,
                    options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            mropeBuffers = (temporal, height, width)
        } else {
            mropeBuffers = nil
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(D).squareRoot()
            : 1.0
        let t = tokens.count
        let emb = model.embedding
        // Speculative verification keeps the per-token GEMV numerics its
        // acceptance was written against.
        let isSpeculativeVerification = speculativeTargetTokens != nil

        func encodeInt4Projection(commandBuffer: MTLCommandBuffer,
                                  family: PrefillProjectionFamily,
                                  weights: TensorView,
                                  x: MTLBuffer,
                                  y: MTLBuffer,
                                  rows: Int,
                                  columns: Int,
                                  tokenCount: Int,
                                  xStrideElements: Int,
                                  yStrideElements: Int) {
            // Opt-in: 2-31 token blocks read each weight row once per tile
            // instead of once per token. It runs ahead of MPP so the two can
            // be compared on one build. A refused shape falls through.
            if smallBlockPrefill.admits(family: family, tokenCount: tokenCount,
                                        speculativeVerification: isSpeculativeVerification),
               let smallBlockProjection,
               smallBlockProjection.encode(commandBuffer: commandBuffer,
                                           weights: weights.buffer,
                                           weightsOffset: Int(weights.offset),
                                           scales: weights.buffer,
                                           scalesOffset: Int(weights.scaleOffset),
                                           biases: weights.buffer,
                                           biasesOffset: Int(weights.biasOffset),
                                           x: x,
                                           xStrideElements: xStrideElements,
                                           y: y,
                                           yStrideElements: yStrideElements,
                                           rows: rows,
                                           columns: columns,
                                           tokenCount: tokenCount) {
                return
            }
            // Every dense projection of two or more tokens with contiguous rows
            // takes the MPP path; a shape MPP cannot tile falls through to the
            // QMM below.
            if PrefillBatchingPolicy.batchesProjections(
                   tokenCount: tokenCount,
                   speculativeVerification: isSpeculativeVerification),
               family != .routed,
               xStrideElements == columns, yStrideElements == rows,
               let mpp = prefillMPPAffine,
               mpp.encode(commandBuffer: commandBuffer,
                              weights: weights.buffer,
                              weightsOffset: Int(weights.offset),
                              scales: weights.buffer,
                              scalesOffset: Int(weights.scaleOffset),
                              biases: weights.buffer,
                              biasesOffset: Int(weights.biasOffset),
                              x: x, y: y,
                              m: tokenCount, n: rows, k: columns) {
                return
            }
            if PrefillProjectionDispatchPolicy.selectedDispatch(for: family,
                                                                chunkTokens: tokenCount) == .qmm {
                prefillQMM.encode(commandBuffer: commandBuffer,
                                  weights: weights.buffer,
                                  weightsOffset: Int(weights.offset),
                                  scales: weights.buffer,
                                  scalesOffset: Int(weights.scaleOffset),
                                  biases: weights.buffer,
                                  biasesOffset: Int(weights.biasOffset),
                                  x: x,
                                  y: y,
                                  t: tokenCount,
                                  n: rows,
                                  k: columns)
                return
            }
            for row in 0..<tokenCount {
                int4.encode(commandBuffer: commandBuffer,
                            weights: weights.buffer,
                            weightsOffset: Int(weights.offset),
                            scales: weights.buffer,
                            scalesOffset: Int(weights.scaleOffset),
                            biases: weights.buffer,
                            biasesOffset: Int(weights.biasOffset),
                            x: x,
                            xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                            y: y,
                            yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                            m: UInt32(rows),
                            n: UInt32(columns))
            }
        }

        func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                           source: MTLBuffer,
                           destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                           sourceTokenOffset: Int,
                           tokenCount: Int,
                           bytesPerToken: Int) throws {
            guard tokenCount > 0 else { return }
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(from: source,
                      sourceOffset: sourceTokenOffset * bytesPerToken,
                      to: destination.buffer,
                      destinationOffset: destination.offset,
                      size: tokenCount * bytesPerToken)
            blit.endEncoding()
        }

        func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                                  kv: KVCacheManager,
                                  layer: Int,
                                  startPosition: Int,
                                  tokenCount: Int,
                                  keySource: MTLBuffer,
                                  valueSource: MTLBuffer,
                                  bytesPerToken: Int) throws {
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
            let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keyFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            guard firstSpan < tokenCount else { return }

            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
            let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keySecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueSecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
        }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        guard var cb = makePrefillCommandBuffer(
            "prefill start=\(startPosition) count=\(tokens.count) layer=0 phase=embed_attention") else {
            throw ModelError.residentBufferWrapFailed
        }
        let residualStreams = cfg.hyperConnection.isEnabled
            ? cfg.hyperConnection.streamCount : 1
        let residualRowElements = D * residualStreams
        if let embeddingOverride {
            // The override spans the whole chunk — the planner emits image spans
            // as standalone chunks — so the INT4 gather it would cover is
            // skipped rather than encoded and fully overwritten.
            let rowBytes = D * MemoryLayout<Float16>.stride
            let bytes = t * rowBytes
            guard embeddingOverride.length >= bytes,
                  let blit = cb.makeBlitCommandEncoder() else {
                throw VisionRuntimeError.invalidInput(
                    "projected image feature buffer is too small")
            }
            if residualStreams == 1 {
                blit.copy(from: embeddingOverride, sourceOffset: 0,
                          to: scratch.hidden, destinationOffset: 0, size: bytes)
            } else {
                // Image features arrive packed; the residual keeps a token's
                // streams side by side, so each row lands on its own stride
                // and the tiling below fills the rest.
                let destinationStride = residualRowElements * MemoryLayout<Float16>.stride
                for token in 0..<t {
                    blit.copy(from: embeddingOverride,
                              sourceOffset: token * rowBytes,
                              to: scratch.hidden,
                              destinationOffset: token * destinationStride,
                              size: rowBytes)
                }
            }
            blit.endEncoding()
        } else if let prism {
            try prism.encodeEmbedding(commandBuffer: cb, model: model, weight: emb,
                                      tokens: tokenBuffer, output: scratch.hidden,
                                      width: D, vocabulary: cfg.vocabSize, count: t)
        } else {
            prefillEmbed.encode(commandBuffer: cb,
                                table: emb.buffer,
                                tableOffset: Int(emb.offset),
                                scales: emb.buffer,
                                scalesOffset: Int(emb.scaleOffset),
                                biases: emb.buffer,
                                biasesOffset: Int(emb.biasOffset),
                                tokens: tokenBuffer,
                                out: scratch.hidden,
                                t: UInt32(t),
                                d: UInt32(D),
                                outScale: embedOutScale,
                                outStrideElements: UInt32(residualRowElements))
        }

        if cfg.hasPerLayerInputs {
            let plan = PerLayerEmbeddingPlan(config: cfg)
            let identity = model.perLayerEmbedding
            prefillEmbed.encode(
                commandBuffer: cb,
                table: identity.buffer,
                tableOffset: Int(identity.offset),
                scales: identity.buffer,
                scalesOffset: Int(identity.scaleOffset),
                biases: identity.buffer,
                biasesOffset: Int(identity.biasOffset),
                tokens: tokenBuffer,
                out: scratch.pleIdentity,
                t: UInt32(t),
                d: UInt32(plan.packedWidth),
                outScale: plan.tokenIdentityScale)
            encodeInt4Projection(
                commandBuffer: cb,
                family: .shared,
                weights: model.perLayerModelProjection,
                x: scratch.hidden,
                y: scratch.pleContext,
                rows: plan.packedWidth,
                columns: D,
                tokenCount: t,
                xStrideElements: D,
                yStrideElements: plan.packedWidth)
        }

        // Four residual streams start from the same embedding and diverge only
        // through what each block injects, exactly as the decode path tiles a
        // single token's row.
        let streams = cfg.hyperConnection.streamCount
        if cfg.hyperConnection.isEnabled, let hc = prefillHyperConnection {
            hc.encodeTileEmbedding(commandBuffer: cb, hidden: scratch.hidden,
                                   tokens: t, hiddenSize: D, streamCount: streams)
        }

        /// The chunk-scale twin of `encodeHyperConnectionRead`: normalize every
        /// stream, mix them down through the low rank and back up, collapse to
        /// `hcMixed`, and leave the per-stream injection gate in `hcInject`.
        /// `layer: nil` selects the model-level mixer, which only collapses.
        func encodePrefillHyperConnectionRead(
            _ cb: MTLCommandBuffer,
            slot: Model.HyperConnectionSlot,
            layer L: Int?
        ) throws {
            guard let hc = prefillHyperConnection else {
                preconditionFailure("hyper-connection chunk encode without its kernels")
            }
            let lowRank = cfg.hyperConnection.lowRank
            let stacked = D * streams
            let norm = try model.hyperConnectionNorm(slot: slot, layer: L)
            hc.encodeGroupedCenteredNorm(
                commandBuffer: cb, x: scratch.hidden,
                weight: norm.buffer, weightOffset: Int(norm.offset),
                out: scratch.normed, tokens: t, hiddenSize: D,
                streamCount: streams, eps: eps)
            let down = try model.hyperConnectionMixDown(slot: slot, layer: L)
            encodeInt4Projection(commandBuffer: cb, family: .shared, weights: down,
                                 x: scratch.normed, y: scratch.hcLowRank,
                                 rows: lowRank, columns: stacked,
                                 tokenCount: t,
                                 xStrideElements: stacked,
                                 yStrideElements: lowRank)
            hc.encodeLowRankSilu(commandBuffer: cb, x: scratch.hcLowRank,
                                 tokens: t, lowRank: lowRank, streamCount: streams)
            let up = try model.hyperConnectionMixUp(slot: slot, layer: L)
            encodeInt4Projection(commandBuffer: cb, family: .shared, weights: up,
                                 x: scratch.hcLowRank, y: scratch.hcMixUp,
                                 rows: stacked, columns: lowRank,
                                 tokenCount: t,
                                 xStrideElements: lowRank,
                                 yStrideElements: stacked)
            hc.encodeCombine(commandBuffer: cb, up: scratch.hcMixUp,
                             normed: scratch.normed, mixed: scratch.hcMixed,
                             tokens: t, hiddenSize: D, streamCount: streams)
            guard let L else { return }
            let gate = try model.hyperConnectionInject(slot: slot, layer: L)
            encodeInt4Projection(commandBuffer: cb, family: .shared, weights: gate,
                                 x: scratch.normed, y: scratch.hcInject,
                                 rows: streams, columns: stacked,
                                 tokenCount: t,
                                 xStrideElements: stacked,
                                 yStrideElements: streams)
        }

        func encodePrefillHyperConnectionWrite(_ cb: MTLCommandBuffer,
                                               branch: MTLBuffer) {
            guard let hc = prefillHyperConnection else {
                preconditionFailure("hyper-connection chunk write without its kernels")
            }
            hc.encodeInject(commandBuffer: cb, hidden: scratch.hidden,
                            branch: branch, injectionRaw: scratch.hcInject,
                            tokens: t, hiddenSize: D, streamCount: streams)
        }

        // What a block reads. With hyper-connections it is the stream mixture;
        // otherwise the norm that precedes the block, as before.
        let usesHyperConnections = cfg.hyperConnection.isEnabled
        let blockInput = usesHyperConnections ? scratch.hcMixed : scratch.normed
        let ffnInput = usesHyperConnections ? scratch.hcMixed : scratch.routedX

        func finishStagedDenseChunkLayer(_ layer: Int) throws {
            guard model.stagesDenseLayers else { return }
            cb.commit()
            try waitForCompletion(cb)
            guard let next = makePrefillCommandBuffer("dense staged layer \(layer + 1)") else {
                throw ModelError.residentBufferWrapFailed
            }
            cb = next
        }

        for L in 0..<cfg.numLayers {
            if cfg.feedForwardKind == .mixtureOfExperts {
                model.beginOpeningRoutedExpertStreamer(layer: L)
            }
            try model.prepareDenseLayer(L)
            let views = layerViews[L]
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            if usesHyperConnections {
                // The n-gram embedding joins the residual before the layer's
                // first hyper-connection reads it, one token at a time so the
                // dilated convolution sees the same window a sequential decode
                // would.
                if cfg.ngramEmbedding.isEnabled, L == cfg.ngramEmbedding.layer {
                    let stride = D * streams * MemoryLayout<Float16>.stride
                    for (index, token) in tokens.enumerated() {
                        try encodeNgramPLE(cb, layer: L, token: token,
                                           residual: scratch.hidden,
                                           residualOffset: index * stride,
                                           embeddingStaging: scratch.ngramEmbedding,
                                           embeddingRow: index)
                    }
                }
                // The two hyper-connection modules carry the only norms in the
                // layer; there is no `input_layernorm` to encode.
                try encodePrefillHyperConnectionRead(cb, slot: .attention, layer: L)
            } else if let prism {
                prism.encodeNorm(commandBuffer: cb, weight: views.inputNorm!,
                                 input: scratch.hidden, output: scratch.normed,
                                 width: D, rows: t, eps: eps)
            } else {
                prefillRMS.encodeBF16W(commandBuffer: cb,
                                       x: scratch.hidden,
                                       weight: views.inputNorm!.buffer,
                                       weightOffset: Int(views.inputNorm!.offset),
                                       out: scratch.normed,
                                       t: UInt32(t),
                                       d: UInt32(D),
                                       eps: eps)
            }
            if isLinear {
                // Gated-DeltaNet linear attention over the chunk: batched
                // projections, causal conv (+ tail carry), delta-rule
                // recurrence, gated norm, out_proj. No KV writes, no
                // attention, no blit.
                guard let gdn, let gdnState else {
                    preconditionFailure("linear-attention layer without GDN kernels")
                }
                let la = cfg.linearAttention
                if let prism, let prismGDN {
                    try prism.encode(commandBuffer: cb, model: model, weight: views.linQKV!,
                                     input: blockInput, output: scratch.q, columns: D,
                                     outputs: la.qkvDim, rows: t, inputStride: D, outputStride: la.qkvDim)
                    try prism.encode(commandBuffer: cb, model: model, weight: views.linZ!,
                                     input: blockInput, output: scratch.gdnZ, columns: D,
                                     outputs: la.valueDim, rows: t, inputStride: D, outputStride: la.valueDim)
                    // One dispatch per step for the whole chunk. Through 10.3
                    // these ran once per token: six dispatches per token per
                    // GDN layer, which dominated Bonsai's prefill.
                    prism.encodeFloatRows(commandBuffer: cb, weight: views.linA!, input: blockInput,
                                          output: scratch.gdnA, columns: D, outputs: la.numVHeads,
                                          rows: t, inputStride: D)
                    prism.encodeFloatRows(commandBuffer: cb, weight: views.linB!, input: blockInput,
                                          output: scratch.gdnB, columns: D, outputs: la.numVHeads,
                                          rows: t, inputStride: D)
                    prismGDN.encodeConvPrefill(commandBuffer: cb, tail: gdnState.convTailBuffer(layer: L),
                                               qkv: scratch.q, weight: views.linConv!,
                                               output: scratch.gdnConvOut, rows: t)
                    gdn.encodeQKNorm(commandBuffer: cb, convOut: scratch.gdnConvOut, rows: t)
                    prismGDN.encodeDeltaPrefill(commandBuffer: cb, convOut: scratch.gdnConvOut,
                                                a: scratch.gdnA, b: scratch.gdnB, aLog: views.linALog!,
                                                dtBias: views.linDtBias!, state: gdnState.stateBuffer(layer: L),
                                                output: scratch.gdnY, rows: t)
                    prismGDN.encodeGatedNormPrefill(commandBuffer: cb, y: scratch.gdnY, z: scratch.gdnZ,
                                                    weight: views.linNorm!, output: scratch.attentionOutput,
                                                    rows: t)
                    try prism.encode(commandBuffer: cb, model: model, weight: views.linOut!,
                                     input: scratch.attentionOutput, output: scratch.h1,
                                     columns: la.valueDim, outputs: D, rows: t,
                                     inputStride: la.valueDim, outputStride: D)
                } else {
                encodeInt4Projection(commandBuffer: cb,
                                     family: .q,
                                     weights: views.linQKV!,
                                     x: blockInput,
                                     y: scratch.q,
                                     rows: la.qkvDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.qkvDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linZ!,
                                     x: blockInput,
                                     y: scratch.gdnZ,
                                     rows: la.valueDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.valueDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linA!,
                                     x: blockInput,
                                     y: scratch.gdnA,
                                     rows: la.numVHeads,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.numVHeads)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linB!,
                                     x: blockInput,
                                     y: scratch.gdnB,
                                     rows: la.numVHeads,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.numVHeads)
                let convW = views.linConv!
                let tail = gdnState.convTailBuffer(layer: L)
                gdn.encodeConvPrefill(commandBuffer: cb,
                                      tail: tail,
                                      qkvRows: scratch.q,
                                      convWeight: convW.buffer,
                                      convWeightOffset: Int(convW.offset),
                                      out: scratch.gdnConvOut,
                                      rows: t)
                gdn.encodeConvTailUpdate(commandBuffer: cb,
                                         tail: tail,
                                         qkvRows: scratch.q,
                                         rows: t)
                gdn.encodeQKNorm(commandBuffer: cb,
                                 convOut: scratch.gdnConvOut,
                                 rows: t)
                let aLog = views.linALog!
                let dtBias = views.linDtBias!
                gdn.encodeDeltaStepPrefill(commandBuffer: cb,
                                           convOut: scratch.gdnConvOut,
                                           aProj: scratch.gdnA,
                                           bProj: scratch.gdnB,
                                           aLog: aLog.buffer,
                                           aLogOffset: Int(aLog.offset),
                                           dtBias: dtBias.buffer,
                                           dtBiasOffset: Int(dtBias.offset),
                                           state: gdnState.stateBuffer(layer: L),
                                           y: scratch.gdnY,
                                           rows: t)
                let gatedNormW = views.linNorm!
                gdn.encodeGatedNorm(commandBuffer: cb,
                                    y: scratch.gdnY,
                                    z: scratch.gdnZ,
                                    weight: gatedNormW.buffer,
                                    weightOffset: Int(gatedNormW.offset),
                                    out: scratch.attentionOutput,
                                    rows: t)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .o,
                                     weights: views.linOut!,
                                     x: scratch.attentionOutput,
                                     y: scratch.h1,
                                     rows: D,
                                     columns: la.valueDim,
                                     tokenCount: t,
                                     xStrideElements: la.valueDim,
                                     yStrideElements: D)
                }
            } else {
                let qProjRows = cfg.attnOutputGate ? 2 * qDim : qDim
                if let prism {
                    for (weight, output, width) in [(views.q!, scratch.q, qProjRows),
                                                    (views.k!, scratch.kStage, kvDim),
                                                    (views.v!, scratch.vStage, kvDim)] {
                        try prism.encode(commandBuffer: cb, model: model, weight: weight,
                                         input: blockInput, output: output, columns: D,
                                         outputs: width, rows: t, inputStride: D, outputStride: width)
                    }
                } else {
                encodeInt4Projection(commandBuffer: cb,
                                     family: .q,
                                     weights: views.q!,
                                     x: blockInput,
                                     y: scratch.q,
                                     rows: qProjRows,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: qProjRows)
                if !cfg.layerSharesKV(L) {
                    encodeInt4Projection(commandBuffer: cb,
                                         family: .kv,
                                         weights: views.k!,
                                         x: blockInput,
                                         y: scratch.kStage,
                                         rows: kvDim,
                                         columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: kvDim)
                    encodeInt4Projection(commandBuffer: cb,
                                         family: .kv,
                                         weights: views.v!,
                                         x: blockInput,
                                         y: scratch.vStage,
                                         rows: kvDim,
                                         columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: kvDim)
                }
                }

                // The attention input Q: the packed q_proj output is split
                // into per-head query/gate halves for gated architectures.
                let attnQ: MTLBuffer
                if cfg.attnOutputGate {
                    elementwise!.encodeSplitQGate(commandBuffer: cb,
                                                  packed: scratch.q,
                                                  q: scratch.attnQ,
                                                  gate: scratch.attnGate,
                                                  heads: cfg.numHeads,
                                                  dim: headDim,
                                                  rows: t)
                    attnQ = scratch.attnQ
                } else {
                    attnQ = scratch.q
                }

                if let prism {
                    prism.encodeNorm(commandBuffer: cb, weight: views.qNorm!, input: attnQ,
                                     output: attnQ, width: headDim, rows: t * cfg.numHeads, eps: eps)
                    prism.encodeNorm(commandBuffer: cb, weight: views.kNorm!, input: scratch.kStage,
                                     output: scratch.kStage, width: headDim, rows: t * numKVHeads, eps: eps)
                    for row in 0..<t {
                        let position = UInt32(startPosition + row)
                        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
                        rope!.encodeNeoxSubdim(commandBuffer: cb, data: attnQ,
                                               dataOffset: row * qDim * 2, position: position,
                                               headDim: UInt32(headDim), numHeads: UInt32(cfg.numHeads),
                                               rotaryDim: rotaryDim, theta: Float(cfg.fullRopeTheta))
                        rope!.encodeNeoxSubdim(commandBuffer: cb, data: scratch.kStage,
                                               dataOffset: row * kvDim * 2, position: position,
                                               headDim: UInt32(headDim), numHeads: UInt32(numKVHeads),
                                               rotaryDim: rotaryDim, theta: Float(cfg.fullRopeTheta))
                    }
                } else if cfg.layerSharesKV(L) {
                    let rotatedPairs = isFull
                        ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                        : UInt32(headDim / 2)
                    prefillQKVEpilogue.encodeGemmaQOnly(
                        commandBuffer: cb,
                        q: attnQ,
                        qWeight: views.qNorm!.buffer,
                        qWeightOffset: Int(views.qNorm!.offset),
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        qTokenStrideElements: UInt32(qDim),
                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                        rotatedPairs: rotatedPairs,
                        eps: eps)
                } else if cfg.usesProjectionWideQKNorm {
                    let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
                    prefillRMS.encodeBF16W(
                        commandBuffer: cb,
                        x: attnQ,
                        weight: views.qNorm!.buffer,
                        weightOffset: Int(views.qNorm!.offset),
                        out: attnQ,
                        t: UInt32(t), d: UInt32(qDim), eps: eps)
                    prefillRMS.encodeBF16W(
                        commandBuffer: cb,
                        x: scratch.kStage,
                        weight: views.kNorm!.buffer,
                        weightOffset: Int(views.kNorm!.offset),
                        out: scratch.kStage,
                        t: UInt32(t), d: UInt32(kvDim), eps: eps)
                    prefillQKVEpilogue.encodeProjectionWideNeoxSubdimRoPE(
                        commandBuffer: cb,
                        q: attnQ, k: scratch.kStage,
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKVHeads),
                        qTokenStrideElements: UInt32(qDim),
                        kvTokenStrideElements: UInt32(kvDim),
                        theta: Float(cfg.fullRopeTheta),
                        rotaryDim: rotaryDim)
                } else if cfg.ropeNeoxSubdim {
                    let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
                    if let mropeBuffers {
                        prefillQKVEpilogue.encodeMRoPENeoxSubdimNoVNorm(
                            commandBuffer: cb,
                            q: attnQ,
                            k: scratch.kStage,
                            qWeight: views.qNorm!.buffer,
                            qWeightOffset: Int(views.qNorm!.offset),
                            kWeight: views.kNorm!.buffer,
                            kWeightOffset: Int(views.kNorm!.offset),
                            queryCount: UInt32(t),
                            headDim: UInt32(headDim),
                            numQHeads: UInt32(cfg.numHeads),
                            numKVHeads: UInt32(numKVHeads),
                            qTokenStrideElements: UInt32(qDim),
                            kvTokenStrideElements: UInt32(kvDim),
                            theta: Float(cfg.fullRopeTheta),
                            rotaryDim: rotaryDim,
                            eps: eps,
                            temporalPositions: mropeBuffers.temporal,
                            heightPositions: mropeBuffers.height,
                            widthPositions: mropeBuffers.width,
                            centered: cfg.normWeightsCenteredAtZero)
                    } else {
                        prefillQKVEpilogue.encodeNeoxSubdimNoVNorm(
                        commandBuffer: cb,
                        q: attnQ,
                        k: scratch.kStage,
                        qWeight: views.qNorm!.buffer,
                        qWeightOffset: Int(views.qNorm!.offset),
                        kWeight: views.kNorm!.buffer,
                        kWeightOffset: Int(views.kNorm!.offset),
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKVHeads),
                        qTokenStrideElements: UInt32(qDim),
                        kvTokenStrideElements: UInt32(kvDim),
                        theta: Float(cfg.fullRopeTheta),
                        rotaryDim: rotaryDim,
                        eps: eps,
                        centered: cfg.normWeightsCenteredAtZero)
                    }
                } else {
                    let rotatedPairs = isFull
                        ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                        : UInt32(headDim / 2)
                    prefillQKVEpilogue.encode(commandBuffer: cb,
                                               q: attnQ,
                                               k: scratch.kStage,
                                               v: scratch.vStage,
                                               qWeight: views.qNorm!.buffer,
                                               qWeightOffset: Int(views.qNorm!.offset),
                                               kWeight: views.kNorm!.buffer,
                                               kWeightOffset: Int(views.kNorm!.offset),
                                               startPosition: UInt32(startPosition),
                                               queryCount: UInt32(t),
                                               headDim: UInt32(headDim),
                                               numQHeads: UInt32(cfg.numHeads),
                                               numKVHeads: UInt32(numKVHeads),
                                               qTokenStrideElements: UInt32(qDim),
                                               kvTokenStrideElements: UInt32(kvDim),
                                               theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                               rotatedPairs: rotatedPairs,
                                               eps: eps)
                }

                if let kv, !cfg.layerSharesKV(L) {
                    let bytes = t * kvDim * MemoryLayout<Float16>.stride
                    try copyPrefillKVToCache(commandBuffer: cb,
                                             kv: kv,
                                             layer: L,
                                             startPosition: startPosition,
                                             tokenCount: t,
                                             keySource: scratch.kStage,
                                             valueSource: scratch.vStage,
                                             bytesPerToken: bytes / t)
                }
                let params = PrefillAttentionParams(
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKVHeads),
                        kvValidCount: UInt32(startPosition + t),
                        slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                        kvTokenStrideElements: UInt32(kvDim),
                        qTokenStrideElements: UInt32(qDim),
                        oTokenStrideElements: UInt32(qDim),
                        scale: Float(cfg.attentionScale),
                        bidirectionalBlockStart: UInt32(
                            bidirectionalBlock?.lowerBound ?? 0),
                        bidirectionalBlockEnd: UInt32(
                            bidirectionalBlock?.upperBound ?? 0))
                if let kv {
                        let kvLayer = cfg.kvSourceLayer(for: L) ?? L
                        let keyBuffer = kv.keyBuffer(layer: kvLayer, validTokenCount: startPosition + t)
                        let valueBuffer = kv.valueBuffer(layer: kvLayer, validTokenCount: startPosition + t)
                        let ringCapacity = kv.ringCapacity(layer: kvLayer)
                        let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                            ? UInt32(ringCapacity)
                            : 0
                        if let qsa = qwenSparseAttention {
                            let weights = try model.resident(name:
                                "language_model.model.layers.\(L).self_attn.indexer.index_qk_proj.weight")
                            let projected = try qsa.projectionBuffer(tokens: t)
                            encodeInt4Projection(commandBuffer: cb, family: .q, weights: weights,
                                x: blockInput, y: projected,
                                rows: cfg.attentionIndexer.projectionRows, columns: D,
                                tokenCount: t, xStrideElements: D,
                                yStrideElements: cfg.attentionIndexer.projectionRows)
                            try qsa.encode(commandBuffer: cb, model: model, layer: L,
                                projected: projected, start: startPosition, tokens: t,
                                positions: positionIDs, ropeDelta: qwenMultimodalRopeDelta,
                                q: attnQ, k: keyBuffer, v: valueBuffer, out: scratch.attentionOutput)
                        }
                        if qwenSparseAttention?.usesSparseAttention(endPosition: startPosition + t) != true {
                        prefillAttention.encodeCausal(commandBuffer: cb,
                                                      q: attnQ,
                                                      k: keyBuffer,
                                                      v: valueBuffer,
                                                      out: scratch.attentionOutput,
                                                      params: params,
                                                      kvRingCapacity: activeRingCapacity,
                                                      layerKind: isFull ? .full : .slidingWindow,
                                                      allowsBidirectionalFullAttention:
                                                          cfg.variant == .gemma4_12B_QAT,
                                                      path: prefillAttentionPath)
                        }
                } else {
                    throw PrefillError.chunkedUnsupported(
                        "chunked prefill attention requires FP16 KV")
                }
                if cfg.attnOutputGate {
                    elementwise!.encodeSigmoidGateMul(commandBuffer: cb,
                                                      out: scratch.attentionOutput,
                                                      gate: scratch.attnGate,
                                                      count: t * qDim)
                }
                if let prism {
                    try prism.encode(commandBuffer: cb, model: model, weight: views.o!,
                                     input: scratch.attentionOutput, output: scratch.h1,
                                     columns: qDim, outputs: D, rows: t, inputStride: qDim, outputStride: D)
                } else {
                encodeInt4Projection(commandBuffer: cb,
                                         family: .o,
                                         weights: views.o!,
                                         x: scratch.attentionOutput,
                                         y: scratch.h1,
                                         rows: D,
                                         columns: qDim,
                                         tokenCount: t,
                                         xStrideElements: qDim,
                                         yStrideElements: D)
                }
            }
            if let prism {
                elementwise!.encodeResidualAdd(commandBuffer: cb, hidden: scratch.hidden,
                                               delta: scratch.h1, count: t * D)
                prism.encodeNorm(commandBuffer: cb, weight: views.postAttention!, input: scratch.hidden,
                                 output: scratch.denseX, width: D, rows: t, eps: eps)
                let gate = try model.sharedExpertGate(layer: L)
                let up = try model.sharedExpertUp(layer: L)
                let down = try model.sharedExpertDown(layer: L)
                let intermediate = Int(gate.shape.0)
                try prism.encode(commandBuffer: cb, model: model, weight: gate, input: scratch.denseX,
                                 output: scratch.sharedGateScratch, columns: D, outputs: intermediate,
                                 rows: t, inputStride: D, outputStride: intermediate)
                try prism.encode(commandBuffer: cb, model: model, weight: up, input: scratch.denseX,
                                 output: scratch.sharedUpScratch, columns: D, outputs: intermediate,
                                 rows: t, inputStride: D, outputStride: intermediate)
                prism.encodeSiluMul(commandBuffer: cb, gate: scratch.sharedGateScratch, up: scratch.sharedUpScratch,
                                    output: scratch.sharedActScratch, count: t * intermediate)
                try prism.encode(commandBuffer: cb, model: model, weight: down, input: scratch.sharedActScratch,
                                 output: scratch.h1, columns: intermediate, outputs: D,
                                 rows: t, inputStride: intermediate, outputStride: D)
                elementwise!.encodeResidualAdd(commandBuffer: cb, hidden: scratch.hidden,
                                               delta: scratch.h1, count: t * D)
                try finishStagedDenseChunkLayer(L)
                continue
            }
            if cfg.feedForwardKind == .dense && cfg.family == .qwen36 {
                elementwise!.encodeResidualAdd(commandBuffer: cb, hidden: scratch.hidden,
                                               delta: scratch.h1, count: t * D)
                prefillRMS.encodeBF16W(commandBuffer: cb, x: scratch.hidden,
                    weight: views.postAttention!.buffer,
                    weightOffset: Int(views.postAttention!.offset), out: scratch.denseX,
                    t: UInt32(t), d: UInt32(D), eps: eps)
                let projections = sharedExpertProjections[L]!
                try prefillSharedExpert.encodeBlock(commandBuffer: cb, x: scratch.denseX,
                    y: scratch.h1, gate: projections.gate, up: projections.up,
                    down: projections.down, scratchGate: scratch.sharedGateScratch,
                    scratchUp: scratch.sharedUpScratch, scratchAct: scratch.sharedActScratch,
                    queryCount: t, d: D, intermediate: cfg.intermediateSize,
                    xStrideElements: D, yStrideElements: D,
                    allowSmallBlock: smallBlockPrefill.admits(family: .shared, tokenCount: t,
                        speculativeVerification: isSpeculativeVerification),
                    speculativeVerification: isSpeculativeVerification)
                elementwise!.encodeResidualAdd(commandBuffer: cb, hidden: scratch.hidden,
                                               delta: scratch.h1, count: t * D)
                try finishStagedDenseChunkLayer(L)
                continue
            }
            if cfg.feedForwardKind == .dense {
                // Official Gemma 4 dense order: post-normalized attention
                // residual, dense FFN sandwich, PLE mapping residual, scale.
                prefillRMS.encodeBF16W(
                    commandBuffer: cb,
                    x: scratch.h1,
                    weight: views.postAttention!.buffer,
                    weightOffset: Int(views.postAttention!.offset),
                    out: scratch.h2,
                    t: UInt32(t), d: UInt32(D), eps: eps)
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: scratch.hidden,
                                               delta: scratch.h2,
                                               count: t * D)
                prefillRMS.encodeBF16W(
                    commandBuffer: cb,
                    x: scratch.hidden,
                    weight: views.preFFN!.buffer,
                    weightOffset: Int(views.preFFN!.offset),
                    out: scratch.denseX,
                    t: UInt32(t), d: UInt32(D), eps: eps)
                let denseProjections = sharedExpertProjections[L]!
                try prefillSharedExpert.encodeBlock(
                    commandBuffer: cb,
                    x: scratch.denseX,
                    y: scratch.h1,
                    gate: denseProjections.gate,
                    up: denseProjections.up,
                    down: denseProjections.down,
                    scratchGate: scratch.sharedGateScratch,
                    scratchUp: scratch.sharedUpScratch,
                    scratchAct: scratch.sharedActScratch,
                    queryCount: t,
                    d: D,
                    intermediate: cfg.ffnIntermediateSize(layer: L),
                    xStrideElements: D,
                    yStrideElements: D,
                    allowSmallBlock: smallBlockPrefill.admits(
                        family: .shared, tokenCount: t,
                        speculativeVerification: isSpeculativeVerification),
                    speculativeVerification: isSpeculativeVerification)
                prefillRMS.encodeBF16W(
                    commandBuffer: cb,
                    x: scratch.h1,
                    weight: views.postFFN!.buffer,
                    weightOffset: Int(views.postFFN!.offset),
                    out: scratch.h1,
                    t: UInt32(t), d: UInt32(D), eps: eps)
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: scratch.hidden,
                                               delta: scratch.h1,
                                               count: t * D)

                let scalarView = views.layerScalar!
                let scalarBits = scalarView.buffer.contents()
                    .advanced(by: Int(scalarView.offset))
                    .assumingMemoryBound(to: UInt16.self)[0]
                let layerScale = Quantization.bf16ToFloat(scalarBits)
                guard cfg.hasPerLayerInputs else {
                    elementwise!.encodeScale(
                        commandBuffer: cb,
                        values: scratch.hidden,
                        count: t * D,
                        scale: layerScale)
                    try finishStagedDenseChunkLayer(L)
                    continue
                }

                let plePlan = PerLayerEmbeddingPlan(config: cfg)
                let pleNorm = model.perLayerProjectionNorm
                perLayerEmbeddingKernel!.encodePrepareLayer(
                    commandBuffer: cb,
                    identity: scratch.pleIdentity,
                    context: scratch.pleContext,
                    normWeight: pleNorm.buffer,
                    normWeightOffset: Int(pleNorm.offset),
                    out: scratch.pleLayer,
                    rows: t,
                    packedWidth: plePlan.packedWidth,
                    pleWidth: plePlan.perLayerHiddenSize,
                    layer: L,
                    contextScale: plePlan.contextProjectionScale,
                    combinedScale: plePlan.combinedScale,
                    eps: eps)
                encodeInt4Projection(
                    commandBuffer: cb,
                    family: .shared,
                    weights: views.perLayerInputGate!,
                    x: scratch.hidden,
                    y: scratch.pleGate,
                    rows: plePlan.perLayerHiddenSize,
                    columns: D,
                    tokenCount: t,
                    xStrideElements: D,
                    yStrideElements: plePlan.perLayerHiddenSize)
                elementwise!.encodeGELUMul(
                    commandBuffer: cb,
                    gate: scratch.pleGate,
                    value: scratch.pleLayer,
                    out: scratch.pleGate,
                    count: t * plePlan.perLayerHiddenSize)
                encodeInt4Projection(
                    commandBuffer: cb,
                    family: .shared,
                    weights: views.perLayerProjection!,
                    x: scratch.pleGate,
                    y: scratch.h1,
                    rows: D,
                    columns: plePlan.perLayerHiddenSize,
                    tokenCount: t,
                    xStrideElements: plePlan.perLayerHiddenSize,
                    yStrideElements: D)
                prefillRMS.encodeBF16W(
                    commandBuffer: cb,
                    x: scratch.h1,
                    weight: views.postPerLayerInputNorm!.buffer,
                    weightOffset: Int(views.postPerLayerInputNorm!.offset),
                    out: scratch.h1,
                    t: UInt32(t), d: UInt32(D), eps: eps)
                elementwise!.encodeResidualAddScale(
                    commandBuffer: cb,
                    hidden: scratch.hidden,
                    delta: scratch.h1,
                    count: t * D,
                    scale: layerScale)
                try finishStagedDenseChunkLayer(L)
                continue
            }
            if cfg.ffnSandwichNorms {
                prefillPostAttention.encode(commandBuffer: cb,
                                                hidden: scratch.hidden,
                                                attn: scratch.h1,
                                                denseX: scratch.denseX,
                                                routedX: scratch.routedX,
                                                routerX: scratch.routerX,
                                                postAttentionWeight: views.postAttention!.buffer,
                                                postAttentionWeightOffset: Int(views.postAttention!.offset),
                                                preFFNWeight: views.preFFN!.buffer,
                                                preFFNWeightOffset: Int(views.preFFN!.offset),
                                                preFFN2Weight: views.preFFN2!.buffer,
                                                preFFN2WeightOffset: Int(views.preFFN2!.offset),
                                                queryCount: UInt32(t),
                                                d: UInt32(D),
                                                hiddenStrideElements: UInt32(D),
                                                attnStrideElements: UInt32(D),
                                                denseStrideElements: UInt32(D),
                                                routedStrideElements: UInt32(D),
                                                routerStrideElements: UInt32(D),
                                                eps: eps)
            } else if usesHyperConnections {
                // The attention branch reaches every stream through its own
                // gate, and the layer's second hyper-connection — not a
                // post-attention norm — is what the feed-forward block reads.
                encodePrefillHyperConnectionWrite(cb, branch: scratch.h1)
                try encodePrefillHyperConnectionRead(cb, slot: .feedForward, layer: L)
            } else {
                // Plain pre-norm residual block: hidden += attention branch,
                // then one post-attention norm feeds router, shared expert,
                // and routed phase 1 (routedX doubles as moeX).
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: scratch.hidden,
                                               delta: scratch.h1,
                                               count: t * D)
                prefillRMS.encodeBF16W(commandBuffer: cb,
                                       x: scratch.hidden,
                                       weight: views.postAttention!.buffer,
                                       weightOffset: Int(views.postAttention!.offset),
                                       out: scratch.routedX,
                                       t: UInt32(t),
                                       d: UInt32(D),
                                       eps: eps)
            }
            let perExpertScale: (buffer: MTLBuffer, offset: Int)
            if cfg.routerScaled {
                let view = views.routerPerExpertScale!
                perExpertScale = (view.buffer, Int(view.offset))
            } else {
                perExpertScale = (onesPerExpertScale!, 0)
            }
            if cfg.usesSigmoidCorrectionRouter {
                let correction = try model.routerCorrectionBias(layer: L)
                prefillRouter.encodeMiniMaxBlock(
                        commandBuffer: cb,
                        weights: views.router!.buffer,
                        weightsOffset: Int(views.router!.offset),
                        scales: views.router!.buffer,
                        scalesOffset: Int(views.router!.scaleOffset),
                        biases: views.router!.buffer,
                        biasesOffset: Int(views.router!.biasOffset),
                        hidden: ffnInput,
                        effectiveScale: effectiveScaleBuffers[L],
                        correctionBias: correction.buffer,
                        correctionBiasOffset: Int(correction.offset),
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))
            } else {
            prefillRouter.encodeGemma4Block(
                        commandBuffer: cb,
                        weights: views.router!.buffer,
                        weightsOffset: Int(views.router!.offset),
                        scales: views.router!.buffer,
                        scalesOffset: Int(views.router!.scaleOffset),
                        biases: views.router!.buffer,
                        biasesOffset: Int(views.router!.biasOffset),
                        hidden: cfg.ffnSandwichNorms ? scratch.routerX : ffnInput,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: perExpertScale.offset,
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))
            }
            // Prefill lookahead: layer L+1's router over this layer's input
            // predicts which experts the next layer reads, so those reads go
            // to the file cache while this layer's tiles run. See
            // `prefillReadsAhead`.
            var prefillLookahead: (ids: MTLBuffer, weights: MTLBuffer)?
            if prefillReadsAhead, !isSpeculativeVerification, !model.stagesDenseLayers, L + 1 < cfg.numLayers,
               let nextRouter = try? model.router(layer: L + 1),
               let buffers = prefillLookaheadBuffers(ids: scratch.routeIDs, weights: scratch.routeWeights) {
                if cfg.usesSigmoidCorrectionRouter {
                    let correction = try model.routerCorrectionBias(layer: L + 1)
                    prefillRouter.encodeMiniMaxBlock(
                        commandBuffer: cb,
                        weights: nextRouter.buffer, weightsOffset: Int(nextRouter.offset),
                        scales: nextRouter.buffer, scalesOffset: Int(nextRouter.scaleOffset),
                        biases: nextRouter.buffer, biasesOffset: Int(nextRouter.biasOffset),
                        hidden: ffnInput,
                        effectiveScale: effectiveScaleBuffers[L + 1],
                        correctionBias: correction.buffer,
                        correctionBiasOffset: Int(correction.offset),
                        outIndices: buffers.ids, outWeights: buffers.weights,
                        queryCount: UInt32(t), numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D), topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))
                } else {
                    let nextScale: (buffer: MTLBuffer, offset: Int)
                    if cfg.routerScaled {
                        let view = try model.routerPerExpertScale(layer: L + 1)
                        nextScale = (view.buffer, Int(view.offset))
                    } else {
                        nextScale = (onesPerExpertScale!, 0)
                    }
                    prefillRouter.encodeGemma4Block(
                        commandBuffer: cb,
                        weights: nextRouter.buffer, weightsOffset: Int(nextRouter.offset),
                        scales: nextRouter.buffer, scalesOffset: Int(nextRouter.scaleOffset),
                        biases: nextRouter.buffer, biasesOffset: Int(nextRouter.biasOffset),
                        hidden: cfg.ffnSandwichNorms ? scratch.routerX : ffnInput,
                        effectiveScale: effectiveScaleBuffers[L + 1],
                        perExpertScale: nextScale.buffer, perExpertScaleOffset: nextScale.offset,
                        outIndices: buffers.ids, outWeights: buffers.weights,
                        queryCount: UInt32(t), numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D), topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))
                }
                prefillLookahead = buffers
            }

                    cb.commit()
                    try waitForCompletion(cb)

                    let routeCount = t * cfg.topKExperts
                    let idPtr = scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    var routeIDs = [UInt32]()
                    routeIDs.reserveCapacity(routeCount)
                    var routeWeights = [Float16]()
                    routeWeights.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        routeWeights.append(weightPtr[i])
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                                   weights: routeWeights,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                    if let prefillLookahead {
                        let predictedPtr = prefillLookahead.ids.contents()
                            .bindMemory(to: UInt32.self, capacity: routeCount)
                        let predicted = Set((0..<routeCount).map {
                            min(Int(predictedPtr[$0]), cfg.numExperts - 1)
                        })
                        let offsets = model.routedExpertPhysicalOffsets(layer: L + 1)
                        try model.readAheadRoutedExperts(
                            layer: L + 1, experts: predicted.sorted { offsets[$0] < offsets[$1] },
                            lowPriority: true)
                    }
                    let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                    let routeTileExpertCount: Int
                    if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
                        guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                            throw PrefillError.chunkedUnsupported(
                                "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
                        }
                        routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
                    } else {
                        routeTileExpertCount = schedulerConfig.tileExperts
                    }
                    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: routeTileExpertCount,
                        expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

                    // With the overlap off the shared expert finishes before any
                    // routed preparation, as it always has. With it on, the
                    // shared buffer runs while the metadata is built and the
                    // first tile is fetched, and is joined just before the first
                    // routed dispatch. Speculative verification never overlaps.
                    let overlapSharedExpert = sharedExpertOverlap.admits(
                        speculativeVerification: isSpeculativeVerification)
                    guard let sharedCB = makePrefillCommandBuffer(
                        "prefill start=\(startPosition) count=\(tokens.count) layer=\(L) phase=shared_expert") else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    if let sharedProj = sharedExpertProjections[L] {
                    if sharedGateBeforeHalf {
                        let view = sharedProj.scalarGate!
                        for row in 0..<t {
                            int8ScalarGate!.encode(commandBuffer: sharedCB,
                                weights: view.buffer, weightsOffset: Int(view.offset),
                                scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                                biases: view.buffer, biasesOffset: Int(view.biasOffset),
                                x: ffnInput, xOffset: row * D * 2,
                                y: scratch.sharedScalarGate, yOffset: row * 2,
                                m: 1, n: UInt32(D))
                        }
                    }
                    try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                        x: cfg.ffnSandwichNorms
                                                            ? scratch.denseX
                                                            : ffnInput,
                                                        y: scratch.h1,
                                                        gate: sharedProj.gate,
                                                        up: sharedProj.up,
                                                        down: sharedProj.down,
                                                        scratchGate: scratch.sharedGateScratch,
                                                        scratchUp: scratch.sharedUpScratch,
                                                        scratchAct: scratch.sharedActScratch,
                                                        queryCount: t,
                                                        d: D,
                                                        intermediate: cfg.ffnIntermediateSize(layer: L),
                                                        xStrideElements: D,
                                                        yStrideElements: D,
                                                        allowSmallBlock: smallBlockPrefill.admits(
                                                            family: .shared, tokenCount: t,
                                                            speculativeVerification: isSpeculativeVerification),
                                                        speculativeVerification: isSpeculativeVerification,
                                                        outputGate: sharedGateBeforeHalf ? scratch.sharedScalarGate : nil)
                    if cfg.ffnSandwichNorms {
                        let postF1 = sharedProj.postF1!
                        prefillRMS.encodeBF16W(commandBuffer: sharedCB,
                                               x: scratch.h1,
                                               weight: postF1.buffer,
                                               weightOffset: Int(postF1.offset),
                                               out: scratch.h1,
                                               t: UInt32(t),
                                               d: UInt32(D),
                                               eps: eps)
                    } else if cfg.sharedExpertGated && !sharedGateBeforeHalf {
                        // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX),
                        // per chunk row.
                        let gateView = sharedProj.scalarGate!
                        let halfBytes = MemoryLayout<Float16>.stride
                        for row in 0..<t {
                            int8ScalarGate!.encode(
                                commandBuffer: sharedCB,
                                weights: gateView.buffer,
                                weightsOffset: Int(gateView.offset),
                                scales: gateView.buffer,
                                scalesOffset: Int(gateView.scaleOffset),
                                biases: gateView.buffer,
                                biasesOffset: Int(gateView.biasOffset),
                                x: ffnInput,
                                xOffset: row * D * halfBytes,
                                y: scratch.sharedScalarGate,
                                yOffset: row * halfBytes,
                                m: 1, n: UInt32(D))
                        }
                        for row in 0..<t {
                            elementwise!.encodeSigmoidScalarMul(
                                commandBuffer: sharedCB,
                                y: scratch.h1,
                                yOffset: row * D * halfBytes,
                                gate: scratch.sharedScalarGate,
                                gateOffset: row * halfBytes,
                                count: D)
                        }
                    }
                    }
                    sharedCB.commit()

                    struct PendingPrefillTile {
                        let tileIndex: Int
                        let commandBuffer: MTLCommandBuffer
                        let fetch: PrefillStreamedTileFetchResult
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                    }
                    // Everything below that has been committed and not yet
                    // joined. Any failure joins all of it before the error
                    // leaves this layer.
                    var sharedInFlight: MTLCommandBuffer? = sharedCB
                    var pendingTiles: [PendingPrefillTile] = []
                    let metadata: PrefillGroupedRoutedMoEStreamedMetadataBuffers
                    do {
                    try routedPhaseObserver?(.sharedSubmitted(layer: L))
                    func joinSharedExpert() throws {
                        guard let shared = sharedInFlight else { return }
                        sharedInFlight = nil
                        try waitForCompletion(shared)
                        try routedPhaseObserver?(.sharedJoined(layer: L))
                    }
                    if !overlapSharedExpert {
                        try joinSharedExpert()
                    }

                    metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    try routedPhaseObserver?(.metadataPrepared(layer: L))
                    let routedOffsets = model.routedExpertOffsets(layer: L)
                    var tileLifetime = PrefillStreamedTileSlotLifetime()
                    func drainOldestPendingTile() throws {
                        guard !pendingTiles.isEmpty else { return }
                        let pending = pendingTiles.removeFirst()
                        try withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                            try waitForCompletion(pending.commandBuffer)
                        }
                        try routedPhaseObserver?(.tileJoined(layer: L, tile: pending.tileIndex))
                        if !pending.fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    // The first tile's experts are read by its own fetch right
                    // away; the rest of the layer's go to the file cache in the
                    // background, in tile order, so later fetches copy from
                    // memory and the SSD keeps several reads in flight.
                    if prefillReadsAhead, routes.tiles.count > 1, !isSpeculativeVerification {
                        let firstTile = Int(routes.tiles[0].groupCount)
                        try model.readAheadRoutedExperts(
                            layer: L, experts: routes.groups.dropFirst(firstTile).map { Int($0.expert) })
                    }
                    let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                    for (tileIndex, tile) in routes.tiles.enumerated() {
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                        var plannedFetch: RoutedExpertFetchPlan?
                        if !pendingTiles.isEmpty {
                            let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                            if !pendingAssignedSlots.isEmpty {
                                let pendingSlots = Set(pendingAssignedSlots)
                                let plan = try model.planRoutedExpertsIfPossible(
                                    layer: L,
                                    experts: expertIDs,
                                    avoidingSlots: pendingSlots)
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: pendingAssignedSlots,
                                        avoidingSlotPlanAvailable: plan != nil))
                                switch decision {
                                case .prefetchNext:
                                    guard let plan else {
                                        throw ModelError.indexCorrupt(
                                            detail: "routed tile scheduler requested missing plan")
                                    }
                                    plannedFetch = plan
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler ignored pending tile")
                                }
                            } else {
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: [],
                                        avoidingSlotPlanAvailable: false))
                                switch decision {
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending, .prefetchNext:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler failed to drain empty-slot pending tile")
                                }
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: false,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .issueWithoutPending:
                                break
                            case .prefetchNext, .drainBeforeIssue:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler requested pending action without pending tile")
                            }
                        }
                        try Task.checkCancellation()
                        try routedPhaseObserver?(.fetchStarting(layer: L, tile: tileIndex))
                        let readStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                        defer { prefillClock.expertRead(nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - readStart) }
                        let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                            model: model,
                            layer: L,
                            tileIndex: tileIndex,
                            routes: routes,
                            plannedFetch: plannedFetch,
                            avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                        try Task.checkCancellation()
                        recordSpeculativeFetch(layer: L, fetch: fetch)
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        try routedPhaseObserver?(.tileBound(layer: L, tile: tileIndex))
                        if !fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedMissSlots)
                        }
                        let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                            device: ctx.device,
                            binding: fetch.binding)
                        let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: fetch.binding,
                            offsets: routedOffsets)
                        guard let tileCB = makePrefillCommandBuffer(
                            "prefill start=\(startPosition) count=\(tokens.count) layer=\(L) phase=routed_tile") else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        if PrefillBatchingPolicy.batchesRoutedExperts(
                               tokenCount: tokens.count,
                               speculativeVerification: isSpeculativeVerification),
                           let batchedRouted = prefillBatchedRouted,
                           let batchedScratch = prefillBatchedRoutedScratch {
                            let groupStart = Int(tile.groupStart)
                            try batchedRouted.encodeTile(
                                commandBuffer: tileCB,
                                hidden: ffnInput,
                                hiddenStrideElements: D,
                                sortedPairs: metadata.sortedPairs,
                                groups: Array(routes.groups[groupStart..<(groupStart + Int(tile.groupCount))]),
                                views: fetch.binding.views,
                                offsets: routedOffsets,
                                routePartials: scratch.routePartials,
                                scratch: batchedScratch,
                                d: D, f: cfg.moeIntermediateSize, topK: cfg.topKExperts)
                        } else {
                        try prefillGroupedMoE.encodeStreamedBatched(
                            commandBuffer: tileCB,
                            hidden: ffnInput,
                            sortedPairs: metadata.sortedPairs,
                            routePartials: scratch.routePartials,
                            gateUpActScratch: scratch.routedGateUpActScratch,
                            downScratch: scratch.routedDownScratch,
                            argumentBuffer: argumentBuffer,
                            binding: fetch.binding,
                            params: streamedParams,
                            pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                        }
                        // The shared expert must have finished without error
                        // before any routed work reaches the GPU.
                        try Task.checkCancellation()
                        try joinSharedExpert()
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        try routedPhaseObserver?(.tileDispatched(layer: L, tile: tileIndex))
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    // A layer with no routed tiles still joins the shared expert.
                    try joinSharedExpert()
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    } catch {
                        // Join every buffer still submitted, retaining what it
                        // reads, before the original error leaves this layer.
                        var joins: [() throws -> Void] = []
                        if let shared = sharedInFlight {
                            joins.append { [self] in try waitForCompletion(shared) }
                        }
                        for pending in pendingTiles {
                            joins.append { [self] in
                                try withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                                    try waitForCompletion(pending.commandBuffer)
                                }
                            }
                        }
                        sharedInFlight = nil
                        pendingTiles.removeAll()
                        PrefillSubmittedWorkDrain.joinAll(joins)
                        try? routedPhaseObserver?(.drainedAfterFailure(layer: L, buffers: joins.count))
                        throw error
                    }
                    guard let tailCB = makePrefillCommandBuffer(
                        "prefill start=\(startPosition) count=\(tokens.count) layer=\(L) phase=routed_tail") else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                      routePartials: scratch.routePartials,
                                                      routeWeights: scratch.routeWeights,
                                                      h2: scratch.h2,
                                                      queryCount: UInt32(t),
                                                      topK: UInt32(cfg.topKExperts),
                                                      d: UInt32(D))
                    if cfg.ffnSandwichNorms {
                        let layerScalarView = views.layerScalar!
                        let scalarBits = layerScalarView.buffer.contents()
                            .advanced(by: Int(layerScalarView.offset))
                            .assumingMemoryBound(to: UInt16.self)[0]
                        prefillLayerTail.encode(commandBuffer: tailCB,
                                                h2: scratch.h2,
                                                h1: scratch.h1,
                                                hidden: scratch.hidden,
                                                postFFN2Weight: views.postFFN2!.buffer,
                                                postFFN2WeightOffset: Int(views.postFFN2!.offset),
                                                postFFNWeight: views.postFFN!.buffer,
                                                postFFNWeightOffset: Int(views.postFFN!.offset),
                                                queryCount: UInt32(t),
                                                d: UInt32(D),
                                                h2StrideElements: UInt32(D),
                                                h1StrideElements: UInt32(D),
                                                hiddenStrideElements: UInt32(D),
                                                eps: eps,
                                                layerScalar: Quantization.bf16ToFloat(scalarBits))
                    } else if usesHyperConnections {
                        // The gate multiplies the whole feed-forward branch,
                        // so the shared and routed halves are summed before
                        // the injection rather than added to the residual
                        // separately.
                        if cfg.hasSharedExpert {
                            elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                           hidden: scratch.h2,
                                                           delta: scratch.h1,
                                                           count: t * D)
                        }
                        encodePrefillHyperConnectionWrite(tailCB, branch: scratch.h2)
                    } else {
                        // Plain pre-norm tail: hidden += gated shared branch
                        // + routed branch.
                        if cfg.hasSharedExpert {
                            elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                           hidden: scratch.hidden,
                                                           delta: scratch.h1,
                                                           count: t * D)
                        }
                        elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                       hidden: scratch.hidden,
                                                       delta: scratch.h2,
                                                       count: t * D)
                    }
                    tailCB.commit()
                    try withExtendedLifetime(metadata) {
                        try waitForCompletion(tailCB)
                    }
                    if let sink = debugChunkLayerSink {
                        sink(L, try readChunkResidualTail(scratch: scratch,
                                                          tokens: t,
                                                          streams: streams))
                    }
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = makePrefillCommandBuffer(
                            "prefill start=\(startPosition) count=\(tokens.count) layer=\(L + 1) phase=attention") else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        cb = nextCB
                    }
                    continue
        }

        if cfg.feedForwardKind == .dense {
            cb.commit()
            try waitForCompletion(cb)
        }

        if writeFinalHead {
            let lm = model.lmHead
            guard let finalCB = makePrefillCommandBuffer(
                "prefill start=\(startPosition) count=\(tokens.count) phase=final_head") else {
                throw ModelError.residentBufferWrapFailed
            }
            if usesHyperConnections {
                // No `model.norm` here either: the mixer that collapses the
                // streams is the last thing before the head and carries the
                // only normalization left, so the head reads its output
                // directly rather than norming the residual. The fused greedy
                // head folds a final RMSNorm into the LM head and has nothing
                // to fold here, which is why it is off for this architecture.
                try encodePrefillHyperConnectionRead(finalCB, slot: .attention,
                                                     layer: nil)
                int4.encode(commandBuffer: finalCB,
                            weights: lm.buffer, weightsOffset: Int(lm.offset),
                            scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                            biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                            x: scratch.hcMixed,
                            xOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                            y: logits,
                            m: UInt32(cfg.vocabSize), n: UInt32(D))
                finalCB.commit()
                try waitForCompletion(finalCB)
                kv?.advance(by: tokens.count)
                prefillChunkState.markCommitted()
                return
            }
            let finalNorm = model.finalNorm
            if let prism {
                prism.encodeNorm(commandBuffer: finalCB, weight: finalNorm, input: scratch.hidden,
                                 output: scratch.normed, width: D, rows: 1, eps: eps,
                                 inputOffset: (t - 1) * D * 2)
                try prism.encode(commandBuffer: finalCB, model: model, weight: lm,
                                 input: scratch.normed, output: logits, columns: D,
                                 outputs: cfg.vocabSize, rows: 1, inputStride: D, outputStride: cfg.vocabSize)
                if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                    greedyArgmax.encode(commandBuffer: finalCB, values: logits,
                                        count: cfg.vocabSize, output: greedyTokenBuf)
                }
            } else if let speculativeTargetTokens {
                fusionHead.encodeGreedyDecodeRows(
                    commandBuffer: finalCB,
                    hidden: scratch.hidden,
                    hiddenStride: D,
                    rowCount: t,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lm.buffer,
                    weightsOffset: Int(lm.offset),
                    scales: lm.buffer,
                    scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer,
                    biasesOffset: Int(lm.biasOffset),
                    outTokens: speculativeTargetTokens,
                    d: UInt32(D),
                    vocab: UInt32(cfg.vocabSize),
                    rmsEps: eps)
            } else if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                fusionHead.encodeGreedyDecode(
                    commandBuffer: finalCB,
                    hidden: scratch.hidden,
                    hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lm.buffer,
                    weightsOffset: Int(lm.offset),
                    scales: lm.buffer,
                    scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer,
                    biasesOffset: Int(lm.biasOffset),
                    outToken: greedyTokenBuf,
                    d: UInt32(D),
                    vocab: UInt32(cfg.vocabSize),
                    rmsEps: eps)
            } else {
                prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                                 hiddenBlock: scratch.hidden,
                                                 row: t - 1,
                                                 rowStrideElements: D,
                                                 normWeight: finalNorm.buffer,
                                                 normWeightOffset: Int(finalNorm.offset),
                                                 weights: lm.buffer,
                                                 weightsOffset: Int(lm.offset),
                                                 scales: lm.buffer,
                                                 scalesOffset: Int(lm.scaleOffset),
                                                 biases: lm.buffer,
                                                 biasesOffset: Int(lm.biasOffset),
                                                 logits: logits,
                                                 d: UInt32(D),
                                                 vocab: UInt32(cfg.vocabSize),
                                                 rmsEps: eps)
            }
            finalCB.commit()
            try waitForCompletion(finalCB)
            if speculativeTargetTokens == nil,
               outputMode == .greedyIfAvailable, useFusedGreedyHead {
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }

        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode,
                              epilogue: ((MTLCommandBuffer) -> Void)? = nil) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        if let profiler = GPUOpProfiler.shared {
            profiler.install(device: ctx.device, queue: ctx.queue)
            profiler.beginStep()
        }
        try kv?.ensureCapacity(through: position + 1, on: ctx.queue)
        // Settle every read before returning or throwing to the lease owner.
        // A layer's own defer dispatches any registered read before this runs.
        defer { _ = lookahead.drain() }
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(cfg.hiddenSize).squareRoot()
            : 1.0
        struct PendingRoutedCommand {
            let cb: MTLCommandBuffer
            let sharedCB: MTLCommandBuffer?
            let phase1HitCB: MTLCommandBuffer?
            let encodeAndCommitNanos: UInt64
        }
        var pendingRoutedCommand: PendingRoutedCommand?

        func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                        waitIfNeeded: Bool) throws {
            if waitIfNeeded {
                if let sharedCB = pending.sharedCB {
                    waitUntilCompleted(sharedCB)
                }
                if let phase1HitCB = pending.phase1HitCB {
                    waitUntilCompleted(phase1HitCB)
                }
                waitUntilCompleted(pending.cb)
            }
            if let sharedCB = pending.sharedCB {
                try checkCommandBufferError(sharedCB)
            }
            if let phase1HitCB = pending.phase1HitCB {
                try checkCommandBufferError(phase1HitCB)
            }
            try checkCommandBufferError(pending.cb)
            totalGPURoutedNanos &+= UInt64(max(0, (pending.cb.gpuEndTime - pending.cb.gpuStartTime) * 1e9))
            if let sharedCB = pending.sharedCB {
                totalGPUSharedNanos &+= UInt64(max(0, (sharedCB.gpuEndTime - sharedCB.gpuStartTime) * 1e9))
            }
            totalCb2Nanos &+= pending.encodeAndCommitNanos
        }

        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }

        func flintProjectionBits(_ view: TensorView, rows: UInt32, columns: UInt32) -> Int {
            let elements = UInt64(rows) * UInt64(columns)
            if view.length == elements { return 8 }
            precondition(view.length * 2 == elements, "invalid Flint affine projection length")
            return 4
        }

        func encodeFlintProjection(_ cb: MTLCommandBuffer, view: TensorView,
                                   x: MTLBuffer, y: MTLBuffer, yOffset: Int = 0,
                                   rows: UInt32, columns: UInt32) {
            if flintProjectionBits(view, rows: rows, columns: columns) == 8 {
                flintInt8!.encode(commandBuffer: cb,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: x, y: y, yOffset: yOffset, m: rows, n: columns)
            } else {
                int4.encode(commandBuffer: cb,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: x, y: y, yOffset: yOffset, m: rows, n: columns)
            }
        }

        // Embed lookup + sqrt(H) fused. Keep these encodes on the first
        // layer's command buffer: nothing on the CPU consumes `hidden` or the
        // optional PLE scratch before that layer, so the old standalone
        // command-buffer commit and synchronous wait were pure token overhead.
        let emb = model.embedding
        var firstLayerCommandBuffer: MTLCommandBuffer? = ctx.queue.makeCommandBuffer()!
        if prism != nil { SharedComputeEncoding.share(firstLayerCommandBuffer!) }
        if let prism, let prismToken {
            prismToken.contents().storeBytes(of: token, as: Int32.self)
            try prism.encodeEmbedding(commandBuffer: firstLayerCommandBuffer!, model: model,
                                      weight: emb, tokens: prismToken, output: hidden,
                                      width: cfg.hiddenSize, vocabulary: cfg.vocabSize, count: 1)
        } else if let flintEmbedding {
            flintEmbedding.encode(commandBuffer: firstLayerCommandBuffer!, table: emb,
                                  output: hidden, residual: flintResidual!.values,
                                  token: UInt32(bitPattern: token), width: D)
        } else {
            embedInt4.encode(commandBuffer: firstLayerCommandBuffer!,
                             table: emb.buffer, tableOffset: Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden, tokenId: UInt32(bitPattern: token), d: D,
                             outScale: embedOutScale)
        }
        if cfg.hyperConnection.isEnabled {
            // The reference tiles the embedding across the streams, so each
            // starts from the same token vector and they diverge only through
            // what every block's gate injects.
            let streams = cfg.hyperConnection.streamCount
            let rowBytes = cfg.hiddenSize * MemoryLayout<Float16>.size
            if let blit = firstLayerCommandBuffer!.makeBlitCommandEncoder() {
                for stream in 1..<streams {
                    blit.copy(from: hidden, sourceOffset: 0,
                              to: hidden, destinationOffset: stream * rowBytes,
                              size: rowBytes)
                }
                blit.endEncoding()
            }
        }
        if cfg.hasPerLayerInputs {
            let plan = PerLayerEmbeddingPlan(config: cfg)
            let identity = model.perLayerEmbedding
            embedInt4.encode(
                commandBuffer: firstLayerCommandBuffer!,
                table: identity.buffer, tableOffset: Int(identity.offset),
                scales: identity.buffer, scalesOffset: Int(identity.scaleOffset),
                biases: identity.buffer, biasesOffset: Int(identity.biasOffset),
                out: pleIdentity!,
                tokenId: UInt32(bitPattern: token),
                d: UInt32(plan.packedWidth),
                outScale: plan.tokenIdentityScale)
            let projection = model.perLayerModelProjection
            int4.encode(
                commandBuffer: firstLayerCommandBuffer!,
                weights: projection.buffer, weightsOffset: Int(projection.offset),
                scales: projection.buffer, scalesOffset: Int(projection.scaleOffset),
                biases: projection.buffer, biasesOffset: Int(projection.biasOffset),
                x: hidden, y: pleContext!,
                m: UInt32(plan.packedWidth), n: D)
        }

        func drainForDebug() throws {
            if let pending = pendingRoutedCommand {
                try finishPendingRoutedCommand(pending, waitIfNeeded: true)
                pendingRoutedCommand = nil
            }
            if let first = firstLayerCommandBuffer {
                first.commit()
                waitUntilCompleted(first)
                try checkCommandBufferError(first)
                firstLayerCommandBuffer = nil
            }
        }

        // Flint: a branch output (shared/routed FFN) is added to the FP32
        // residual inside the next norm instead of by its own dispatch. Tests
        // that read the residual at layer boundaries get the eager add.
        let flintDefersResidualAdd = flintResidual != nil && debugLayerSink == nil
        var flintPendingDelta: MTLBuffer?

        for L in 0..<cfg.numLayers {
            if model.stagesDenseLayers {
                try drainForDebug()
                try model.prepareDenseLayer(L)
            }
            if let sink = debugLayerSink {
                try drainForDebug()
                sink(L)
            }
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim     = UInt32(cfg.numHeads * headDimL)
            let kvDim    = UInt32(numKVL * headDimL)
            let seqLen   = UInt32(position + 1)

            // A hyper-connection architecture has neither of these: its two
            // hyper-connection modules carry the only norms in the layer.
            let usesHyperConnections = cfg.hyperConnection.isEnabled
            let inNorm   = usesHyperConnections
                ? nil : try model.inputNorm(layer: L)
            let postAttn = usesHyperConnections
                ? nil : try model.postAttnNorm(layer: L)
            let sharedProj = sharedExpertProjections[L]

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Everything up to and including the router runs in a single CB:
            // the only reason to break is the CPU readback of router indices
            // needed to issue I/O for the routed-expert blobs.
            let cb: MTLCommandBuffer
            if let first = firstLayerCommandBuffer {
                cb = first
                firstLayerCommandBuffer = nil
            } else {
                cb = ctx.queue.makeCommandBuffer()!
            }
            if usesHyperConnections {
                // The n-gram embedding joins the residual before the layer's
                // first hyper-connection reads it.
                if cfg.ngramEmbedding.isEnabled, L == cfg.ngramEmbedding.layer {
                    try encodeNgramPLE(cb, layer: L, token: token)
                }
                try encodeHyperConnectionRead(cb, slot: .attention, layer: L)

            } else if let inNorm {
                if let prism {
                    prism.encodeNorm(commandBuffer: cb, weight: inNorm, input: hidden,
                                     output: normed, width: cfg.hiddenSize, rows: 1, eps: eps)
                } else if let flintResidual {
                    if let delta = flintPendingDelta {
                        flintResidual.addNorm(cb, delta: delta, mirror: hidden,
                                              weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                              output: normed, count: cfg.hiddenSize, eps: eps)
                        flintPendingDelta = nil
                    } else {
                        flintResidual.norm(cb, weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                           output: normed, count: cfg.hiddenSize, eps: eps)
                    }
                } else {
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                out: normed,
                                d: D, eps: eps)
                }
            }

            if isLinear {
                // Gated-DeltaNet linear attention: no KV slots, no RoPE — a
                // fixed-size recurrent state updated in place.
                try encodeLinearAttentionDecode(cb, layer: L)
            } else if cfg.attnOutputGate {
                // Qwen full attention: packed [query ; gate] q_proj, real
                // v_proj, no V norm, NeoX sub-dim RoPE, sigmoid output gate.
                try encodeGatedFullAttentionDecode(cb, layer: L,
                                                   position: position,
                                                   seqLen: seqLen)
            } else if let kvSourceLayer = cfg.kvSourceLayer(for: L) {
                guard let kv, let rope else {
                    preconditionFailure("shared-KV attention requires KV and RoPE kernels")
                }
                let q = try model.qProj(layer: L)
                let o = try model.oProj(layer: L)
                let qNorm = try model.qNorm(layer: L)
                int4.encode(commandBuffer: cb,
                            weights: q.buffer, weightsOffset: Int(q.offset),
                            scales: q.buffer, scalesOffset: Int(q.scaleOffset),
                            biases: q.buffer, biasesOffset: Int(q.biasOffset),
                            x: normed, y: qScratch, m: qDim, n: D)
                rms.encodeBF16WPerHead(commandBuffer: cb,
                                       x: qScratch,
                                       weight: qNorm.buffer,
                                       weightOffset: Int(qNorm.offset),
                                       out: qScratch,
                                       headDim: UInt32(headDimL),
                                       numHeads: cfg.numHeads,
                                       eps: eps)
                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                if isFull {
                    rope.encodeProportionalNeox(
                        commandBuffer: cb, data: qScratch,
                        position: UInt32(position), headDim: UInt32(headDimL),
                        numHeads: UInt32(cfg.numHeads), rotatedPairs: rotated,
                        theta: Float(cfg.fullRopeTheta))
                } else {
                    rope.encodeDefaultNeox(
                        commandBuffer: cb, data: qScratch,
                        position: UInt32(position), headDim: UInt32(headDimL),
                        numHeads: UInt32(cfg.numHeads), theta: Float(cfg.ropeTheta))
                }
                let keyBuffer = kv.keyBuffer(layer: kvSourceLayer,
                                             validTokenCount: position + 1)
                let valueBuffer = kv.valueBuffer(layer: kvSourceLayer,
                                                 validTokenCount: position + 1)
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: keyBuffer, kOffset: 0,
                                         v: valueBuffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: Float(cfg.attentionScale))
                } else {
                    let ringCapacity = kv.ringCapacity(layer: kvSourceLayer)
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity) : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: keyBuffer, kOffset: 0,
                                        v: valueBuffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: Float(cfg.attentionScale),
                                        ringCapacity: activeRingCapacity)
                }
                int4.encode(commandBuffer: cb,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                            biases: o.buffer, biasesOffset: Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim)
            } else {
                let kSlot = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
                let vSlot = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
                let q     = try model.qProj(layer: L)
                let k     = try model.kProj(layer: L)
                // Under the K=V quirk full layers reuse k_proj; otherwise
                // v_proj is a real tensor.
                let vProj = (isFull && cfg.attentionKEqV) ? k : (try model.vProj(layer: L))
                let o     = try model.oProj(layer: L)
                let qNorm = try model.qNorm(layer: L)
                let kNorm = try model.kNorm(layer: L)

                if cfg.family == .flint &&
                    (flintProjectionBits(q, rows: qDim, columns: D) == 8 ||
                     flintProjectionBits(k, rows: kvDim, columns: D) == 8 ||
                     flintProjectionBits(vProj, rows: kvDim, columns: D) == 8) {
                    encodeFlintProjection(cb, view: q, x: blockInput, y: qScratch, rows: qDim, columns: D)
                    encodeFlintProjection(cb, view: k, x: blockInput, y: kSlot.buffer,
                                          yOffset: kSlot.offset, rows: kvDim, columns: D)
                    encodeFlintProjection(cb, view: vProj, x: blockInput, y: vSlot.buffer,
                                          yOffset: vSlot.offset, rows: kvDim, columns: D)
                } else {
                fusedQKVGEMV.encode(commandBuffer: cb,
                                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                    x: blockInput,
                                    qOut: qScratch,
                                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                    qRows: qDim,
                                    kvRows: kvDim,
                                    n: D)
                }

                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                if cfg.family == .flint {
                    guard let rope else { preconditionFailure("Flint requires RoPE kernels") }
                    rms.encodeBF16WPerHead(commandBuffer: cb, x: qScratch,
                        weight: qNorm.buffer, weightOffset: Int(qNorm.offset), out: qScratch,
                        headDim: UInt32(headDimL), numHeads: cfg.numHeads, eps: eps)
                    rms.encodeBF16WPerHead(commandBuffer: cb, x: kSlot.buffer, xOffset: kSlot.offset,
                        weight: kNorm.buffer, weightOffset: Int(kNorm.offset), out: kSlot.buffer,
                        outOffset: kSlot.offset, headDim: UInt32(headDimL), numHeads: numKVL, eps: eps)
                    if !isFull {
                        rope.encodeDefaultNeox(commandBuffer: cb, data: qScratch, position: UInt32(position),
                            headDim: UInt32(headDimL), numHeads: UInt32(cfg.numHeads), theta: Float(cfg.ropeTheta))
                        rope.encodeDefaultNeox(commandBuffer: cb, data: kSlot.buffer, dataOffset: kSlot.offset,
                            position: UInt32(position), headDim: UInt32(headDimL),
                            numHeads: UInt32(numKVL), theta: Float(cfg.ropeTheta))
                    }
                } else if cfg.usesProjectionWideQKNorm {
                    guard let rope else {
                        preconditionFailure("MiniMax attention requires RoPE kernels")
                    }
                    rms.encodeBF16W(commandBuffer: cb,
                                    x: qScratch,
                                    weight: qNorm.buffer,
                                    weightOffset: Int(qNorm.offset),
                                    out: qScratch,
                                    d: qDim, eps: eps)
                    rms.encodeBF16W(commandBuffer: cb,
                                    x: kSlot.buffer,
                                    xOffset: kSlot.offset,
                                    weight: kNorm.buffer,
                                    weightOffset: Int(kNorm.offset),
                                    out: kSlot.buffer,
                                    outOffset: kSlot.offset,
                                    d: kvDim, eps: eps)
                    let rotaryDim = UInt32(Double(headDimL) * cfg.partialRotaryFactor)
                    rope.encodeNeoxSubdim(
                        commandBuffer: cb, data: qScratch,
                        position: UInt32(position), headDim: UInt32(headDimL),
                        numHeads: UInt32(cfg.numHeads), rotaryDim: rotaryDim,
                        theta: Float(cfg.fullRopeTheta))
                    rope.encodeNeoxSubdim(
                        commandBuffer: cb, data: kSlot.buffer, dataOffset: kSlot.offset,
                        position: UInt32(position), headDim: UInt32(headDimL),
                        numHeads: UInt32(numKVL), rotaryDim: rotaryDim,
                        theta: Float(cfg.fullRopeTheta))
                } else {
                fusedQKVEpilogue.encode(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer,
                                        kOffset: kSlot.offset,
                                        v: vSlot.buffer,
                                        vOffset: vSlot.offset,
                                        qWeight: qNorm.buffer,
                                        qWeightOffset: Int(qNorm.offset),
                                        kWeight: kNorm.buffer,
                                        kWeightOffset: Int(kNorm.offset),
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        position: UInt32(position),
                                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                        rotatedPairs: rotated,
                                        eps: eps)
                }

                guard kv != nil else {
                    preconditionFailure("FP16 attention requires an FP16 KV cache")
                }
                let sinkView = cfg.family == .flint ? try model.attentionSinks(layer: L) : nil
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: Float(cfg.attentionScale),
                                         sinks: sinkView?.buffer, sinksOffset: Int(sinkView?.offset ?? 0))
                } else {
                    let ringCapacity = kv?.ringCapacity(layer: L) ?? 0
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer, kOffset: 0,
                                        v: vSlot.buffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: Float(cfg.attentionScale),
                                        sinks: sinkView?.buffer, sinksOffset: Int(sinkView?.offset ?? 0),
                                        ringCapacity: activeRingCapacity)
                }
                if cfg.family == .flint {
                    encodeFlintProjection(cb, view: o, x: attnOut, y: oOut, rows: D, columns: qDim)
                } else {
                    int4.encode(commandBuffer: cb,
                                weights: o.buffer, weightsOffset: Int(o.offset),
                                scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                                biases: o.buffer, biasesOffset: Int(o.biasOffset),
                                x: attnOut, y: oOut, m: D, n: qDim)
                }
            }

            if cfg.feedForwardKind == .dense || (cfg.family == .flint && L < 2) {
                if cfg.family == .qwen36 || cfg.family == .flint {
                    // Flint folds this add into the post-attention norm below.
                    if flintResidual == nil {
                        elementwise!.encodeResidualAdd(commandBuffer: cb, hidden: hidden,
                                                       delta: oOut, count: cfg.hiddenSize)
                    }
                    if let prism {
                        prism.encodeNorm(commandBuffer: cb, weight: postAttn!, input: hidden,
                                         output: denseX, width: cfg.hiddenSize, rows: 1, eps: eps)
                        let gate = try model.sharedExpertGate(layer: L)
                        let up = try model.sharedExpertUp(layer: L)
                        let down = try model.sharedExpertDown(layer: L)
                        let intermediate = Int(gate.shape.0)
                        if prism.supportsDual(columns: cfg.hiddenSize, outputsA: intermediate) {
                            try prism.encodeDual(commandBuffer: cb, model: model, weightA: gate, weightB: up,
                                                 input: denseX, outputA: denseScratchGate, outputB: denseScratchUp,
                                                 columns: cfg.hiddenSize, outputsA: intermediate,
                                                 outputsB: Int(up.shape.0))
                        } else {
                            try prism.encode(commandBuffer: cb, model: model, weight: gate,
                                input: denseX, output: denseScratchGate, columns: cfg.hiddenSize,
                                outputs: intermediate, rows: 1, inputStride: cfg.hiddenSize, outputStride: intermediate)
                            try prism.encode(commandBuffer: cb, model: model, weight: up,
                                input: denseX, output: denseScratchUp, columns: cfg.hiddenSize,
                                outputs: intermediate, rows: 1, inputStride: cfg.hiddenSize, outputStride: intermediate)
                        }
                        prism.encodeSiluMul(commandBuffer: cb, gate: denseScratchGate,
                                            up: denseScratchUp, output: denseScratchAct, count: intermediate)
                        try prism.encode(commandBuffer: cb, model: model, weight: down,
                            input: denseScratchAct, output: h1Buf, columns: intermediate,
                            outputs: cfg.hiddenSize, rows: 1, inputStride: intermediate, outputStride: cfg.hiddenSize)
                    } else {
                    if let flintResidual {
                        flintResidual.addNorm(cb, delta: oOut, mirror: hidden,
                                              weight: postAttn!.buffer, weightOffset: Int(postAttn!.offset),
                                              output: denseX, count: cfg.hiddenSize, eps: eps)
                    } else {
                        rms.encodeBF16W(commandBuffer: cb, x: hidden,
                            weight: postAttn!.buffer, weightOffset: Int(postAttn!.offset),
                            out: denseX, d: D, eps: eps)
                    }
                    let projections = sharedProj!
                    try shared.encode(commandBuffer: cb, x: denseX,
                        gate: projections.gate, up: projections.up, down: projections.down,
                        y: h1Buf, scratchGate: denseScratchGate, scratchUp: denseScratchUp,
                        scratchAct: denseScratchAct)
                    }
                    if let flintResidual {
                        if flintDefersResidualAdd {
                            flintPendingDelta = h1Buf
                        } else {
                            flintResidual.add(cb, delta: h1Buf, mirror: hidden, count: cfg.hiddenSize)
                        }
                    } else {
                        elementwise!.encodeResidualAdd(commandBuffer: cb, hidden: hidden,
                                                       delta: h1Buf, count: cfg.hiddenSize)
                    }
                } else {
                    try encodeDenseGemmaLayerDecode(
                        cb, layer: L, postAttention: postAttn!,
                        projections: sharedProj!, d: D, eps: eps)
                }
                // A dense layer produces nothing the CPU reads, so the whole
                // token stays on one command buffer: the layer's work is
                // handed to the next iteration instead of being committed and
                // waited on here. Each round trip cost ~130 us of wall time
                // against ~500 us of GPU work, so 35 of them per token were
                // most of the gap between GPU time and tokens per second.
                firstLayerCommandBuffer = cb
                totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start
                continue
            }

            if cfg.ffnSandwichNorms {
                let preFFN   = try model.preFFN(layer: L)
                let preFFN2  = try model.preFFN2(layer: L)
                fusedPostAttentionSetup.encode(commandBuffer: cb,
                                               hidden: hidden,
                                               attn: oOut,
                                               denseX: denseX,
                                               routedX: routedX,
                                               routerX: routerInput,
                                               postAttentionWeight: postAttn!.buffer,
                                               postAttentionWeightOffset: Int(postAttn!.offset),
                                               preFFNWeight: preFFN.buffer,
                                               preFFNWeightOffset: Int(preFFN.offset),
                                               preFFN2Weight: preFFN2.buffer,
                                               preFFN2WeightOffset: Int(preFFN2.offset),
                                               d: D,
                                               eps: eps)
            } else if usesHyperConnections {
                // The attention branch is injected into every residual stream
                // through its own gate, then the feed-forward hyper-connection
                // reads the updated streams and leaves the block's input in
                // `hcMixed`, which `ffnInput` resolves to. There is no
                // post-attention norm: this pair of modules is the only
                // normalization the layer has.
                encodeHyperConnectionWrite(cb, branch: oOut)
                try encodeHyperConnectionRead(cb, slot: .feedForward, layer: L)
            } else {
                // Plain pre-norm residual block: hidden += attention branch,
                // then one post-attention norm feeds router, shared expert,
                // and routed phase 1 (routedX doubles as moeX).
                if let flintResidual {
                    flintResidual.addNorm(cb, delta: oOut, mirror: hidden,
                                          weight: postAttn!.buffer, weightOffset: Int(postAttn!.offset),
                                          output: routedX, count: cfg.hiddenSize, eps: eps)
                } else {
                    elementwise!.encodeResidualAdd(commandBuffer: cb, hidden: hidden,
                                                   delta: oOut, count: cfg.hiddenSize)
                    rms.encodeBF16W(commandBuffer: cb, x: hidden, weight: postAttn!.buffer,
                                    weightOffset: Int(postAttn!.offset), out: routedX, d: D, eps: eps)
                }
            }

            let moe = self.moe!
            let routerW = try model.router(layer: L)
            let perExpertScale: (buffer: MTLBuffer, offset: Int)
            if cfg.routerScaled {
                let view = try model.routerPerExpertScale(layer: L)
                perExpertScale = (view.buffer, Int(view.offset))
            } else {
                perExpertScale = (onesPerExpertScale!, 0)
            }

            if let flintRouter {
                flintRouter.encode(commandBuffer: cb, weight: routerW.buffer,
                    weightOffset: Int(routerW.offset), hidden: flintResidual!.normalized,
                    indices: outIndices, weights: outWeights, hiddenSize: cfg.hiddenSize,
                    experts: cfg.numExperts, topK: cfg.topKExperts, floatInput: true,
                    affineScalesOffset: routerW.dtype == 0 ? Int(routerW.scaleOffset) : nil,
                    affineBiasesOffset: routerW.dtype == 0 ? Int(routerW.biasOffset) : nil)
            } else if cfg.usesSigmoidCorrectionRouter {
                let correction = try model.routerCorrectionBias(layer: L)
                moe.encodeRouterMiniMax(commandBuffer: cb,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                    biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                    hidden: ffnInput,
                    effectiveScale: effectiveScaleBuffers[L],
                    correctionBias: correction.buffer,
                    correctionBiasOffset: Int(correction.offset),
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D,
                    topK: UInt32(cfg.topKExperts))
            } else {
            moe.encodeRouterGemma4(commandBuffer: cb,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                hidden: cfg.ffnSandwichNorms ? routerInput : ffnInput,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale.buffer,
                perExpertScaleOffset: perExpertScale.offset,
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
            }
            // Lookahead: layer L+1's router over this layer's feed-forward
            // input predicts L+1's experts, so their reads overlap this
            // layer's work. See `ExpertLookahead`.
            var lookaheadEncoded = false
            var lookaheadDispatch: (() -> Void)?
            func fireLookahead() {
                lookaheadDispatch?()
                lookaheadDispatch = nil
            }
            // Whatever path leaves this layer, including a throw, releases a
            // read that was registered but not yet started, so no drain can
            // wait on it forever.
            defer { fireLookahead() }
            if lookahead.enabled, !model.stagesDenseLayers, L + 1 < cfg.numLayers,
               let nextRouter = try? model.router(layer: L + 1) {
                if let flintRouter {
                    // Flint routes on the FP32 normalized residual: this
                    // layer's post-attention norm is the predictor input.
                    flintRouter.encode(commandBuffer: cb, weight: nextRouter.buffer,
                        weightOffset: Int(nextRouter.offset), hidden: flintResidual!.normalized,
                        indices: lookaheadIndices, weights: lookaheadWeights,
                        hiddenSize: cfg.hiddenSize, experts: cfg.numExperts,
                        topK: cfg.topKExperts, floatInput: true,
                        affineScalesOffset: nextRouter.dtype == 0 ? Int(nextRouter.scaleOffset) : nil,
                        affineBiasesOffset: nextRouter.dtype == 0 ? Int(nextRouter.biasOffset) : nil)
                } else if cfg.usesSigmoidCorrectionRouter {
                    let correction = try model.routerCorrectionBias(layer: L + 1)
                    moe.encodeRouterMiniMax(commandBuffer: cb,
                        weights: nextRouter.buffer, weightsOffset: Int(nextRouter.offset),
                        scales: nextRouter.buffer, scalesOffset: Int(nextRouter.scaleOffset),
                        biases: nextRouter.buffer, biasesOffset: Int(nextRouter.biasOffset),
                        hidden: ffnInput,
                        effectiveScale: effectiveScaleBuffers[L + 1],
                        correctionBias: correction.buffer,
                        correctionBiasOffset: Int(correction.offset),
                        outIndices: lookaheadIndices, outWeights: lookaheadWeights,
                        numExperts: UInt32(cfg.numExperts), d: D,
                        topK: UInt32(cfg.topKExperts))
                } else {
                    let nextScale: (buffer: MTLBuffer, offset: Int)
                    if cfg.routerScaled {
                        let view = try model.routerPerExpertScale(layer: L + 1)
                        nextScale = (view.buffer, Int(view.offset))
                    } else {
                        nextScale = (onesPerExpertScale!, 0)
                    }
                    moe.encodeRouterGemma4(commandBuffer: cb,
                        weights: nextRouter.buffer, weightsOffset: Int(nextRouter.offset),
                        scales: nextRouter.buffer, scalesOffset: Int(nextRouter.scaleOffset),
                        biases: nextRouter.buffer, biasesOffset: Int(nextRouter.biasOffset),
                        hidden: cfg.ffnSandwichNorms ? routerInput : ffnInput,
                        effectiveScale: effectiveScaleBuffers[L + 1],
                        perExpertScale: nextScale.buffer,
                        perExpertScaleOffset: nextScale.offset,
                        outIndices: lookaheadIndices, outWeights: lookaheadWeights,
                        numExperts: UInt32(cfg.numExperts), d: D,
                        topK: UInt32(cfg.topKExperts))
                }
                lookaheadEncoded = true
            }
            cb.commit()
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            waitUntilCompleted(cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            if let pending = pendingRoutedCommand {
                try finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            try checkCommandBufferError(cb)
            totalGPULayerNanos &+= UInt64(max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1e9))
            totalGPULayerCommandBuffers &+= 1
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            // CPU readback to fetch routed-expert blobs from disk.
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }
            debugRouterSink?(L, experts)
            // The read issued for this layer during the previous one must land
            // before this layer plans its own fetch.
            if let finished = lookahead.drain(), finished.layer == L {
                lookahead.record(predicted: finished.experts, actual: experts)
            }
            if lookaheadEncoded {
                let predictedPtr = lookaheadIndices.contents()
                    .bindMemory(to: UInt32.self, capacity: cfg.topKExperts)
                let predicted = (0..<cfg.topKExperts).map {
                    min(Int(predictedPtr[$0]), cfg.numExperts - 1)
                }
                var reads: DispatchGroup?
                if let plan = try model.planRoutedExperts(layer: L + 1, experts: predicted, purpose: .prefetch),
                   !plan.misses.isEmpty {
                    // Entered now, dispatched after this layer's own fetch:
                    // started together, the two reads split the SSD's
                    // bandwidth and gained nothing, while after it the read
                    // overlaps this layer's routed work and the next
                    // layer's attention on the GPU.
                    let prefetch = try model.expertPrefetchOperation(plan: plan)
                    let group = DispatchGroup()
                    group.enter()
                    lookaheadDispatch = {
                        DispatchQueue.global(qos: .userInitiated).async {
                            // A failed read only loses the head start: the
                            // layer's own fetch reads whatever is missing.
                            try? prefetch()
                            group.leave()
                        }
                    }
                    reads = group
                    lookahead.noteRead()
                }
                lookahead.pending = ExpertLookahead.Pending(layer: L + 1, experts: predicted,
                                                            reads: reads)
                if let reads { try decodeLookaheadObserver?(reads) }
            }

            let routedOffsets = model.routedExpertOffsets(layer: L)
            let routedBitWidths = model.routedExpertBitWidths(layer: L)
            let topK = UInt32(cfg.topKExperts)
            let canPlanPhase1HitSplit =
                cfg.topKExperts <= MoE.maxStreamedExperts
            let plannedFetch = canPlanPhase1HitSplit
                ? try model.planRoutedExperts(layer: L, experts: experts)
                : nil
            recordDecodeExpertFetch(layer: L, plan: plannedFetch)
            var phase1HitCB: MTLCommandBuffer?
            var phase1HitSplitArgBuf: MTLBuffer?
            var phase1HitSplitRoutedBufs: [MTLBuffer] = []
            var phase1HitSlots: [UInt32] = []
            var phase1MissSlots: [UInt32] = []

            if let plan = plannedFetch {
                phase1HitSlots.reserveCapacity(cfg.topKExperts - plan.misses.count)
                for slot in 0..<cfg.topKExperts where !plan.misses.contains(slot) {
                    phase1HitSlots.append(UInt32(slot))
                }
                phase1MissSlots = plan.misses.map { UInt32($0) }
            }
            func encodeRoutedPhase1Full(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer]
            ) {
                moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                        routedArgBuffer: argBuf,
                                                        routedBlobs: routedBufs,
                                                        routedOffsets: routedOffsets,
                                                        x: ffnInput,
                                                        acts: moeActs,
                                                        d: D,
                                                        f: FmoE,
                                                        topK: topK, bitWidths: routedBitWidths)
            }

            func encodeRoutedPhase1Subset(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer],
                activeSlots: MTLBuffer,
                activeSlotIndices: [UInt32],
                activeCount: UInt32
            ) {
                moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: cb,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: ffnInput,
                    acts: moeActs,
                    activeSlots: activeSlots,
                    activeSlotIndices: activeSlotIndices,
                    activeCount: activeCount,
                    d: D,
                    f: FmoE,
                    topK: topK, bitWidths: routedBitWidths)
            }

            // Cache-hit work and the shared expert use the same input and
            // can share a submission while the missing experts are read.
            let sharedCB = sharedProj.map { _ in ctx.queue.makeCommandBuffer()! }
            if let plan = plannedFetch,
               plan.hits > 0,
               !plan.misses.isEmpty {
                let plannedBlobs = try model.routedExpertBuffers(for: plan)
                phase1HitSplitRoutedBufs = plannedBlobs.map { $0.buffer }
                // The router readback above has drained the previous layer,
                // so its argument storage is safe to reuse for both subsets.
                phase1HitSplitArgBuf = moe.makeReusedRoutedArgumentBuffer(
                    routedBlobs: phase1HitSplitRoutedBufs,
                    topK: topK)
                if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                    writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                    let cb = sharedCB ?? ctx.queue.makeCommandBuffer()!
                    encodeRoutedPhase1Subset(
                        cb,
                        argBuf: argBuf,
                        routedBufs: phase1HitSplitRoutedBufs,
                        activeSlots: moeHitActiveSlots,
                        activeSlotIndices: phase1HitSlots,
                        activeCount: UInt32(phase1HitSlots.count))
                    if sharedCB == nil { phase1HitCB = cb }
                }
            }

            // The shared dense MLP depends only on its normed input, not on
            // the routed experts. Commit it without waiting so its GPU work
            // overlaps the routed-expert pread. The routed CB follows it on
            // the same queue, so the combine sees h1Buf.
            if let sharedCB, let sharedProj {
            if sharedGateBeforeHalf {
                let view = sharedProj.scalarGate!
                int8ScalarGate!.encode(commandBuffer: sharedCB,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: ffnInput, y: sharedScalarGateBuf!, m: 1, n: D)
            }
            try! shared.encode(commandBuffer: sharedCB,
                               x: cfg.ffnSandwichNorms ? denseX : ffnInput,
                               gate: sharedProj.gate,
                               up: sharedProj.up,
                               down: sharedProj.down,
                               y: h1Buf,
                               scratchGate: denseScratchGate,
                               scratchUp: denseScratchUp,
                               scratchAct: denseScratchAct,
                               outputGate: sharedGateBeforeHalf ? sharedScalarGateBuf : nil)
            if cfg.ffnSandwichNorms {
                let postF1 = sharedProj.postF1!
                rms.encodeBF16W(commandBuffer: sharedCB, x: h1Buf,
                                weight: postF1.buffer,
                                weightOffset: Int(postF1.offset),
                                out: h1Buf, d: D, eps: eps)
            } else if cfg.sharedExpertGated && !sharedGateBeforeHalf {
                // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
                let gateView = sharedProj.scalarGate!
                int8ScalarGate!.encode(commandBuffer: sharedCB,
                                       weights: gateView.buffer,
                                       weightsOffset: Int(gateView.offset),
                                       scales: gateView.buffer,
                                       scalesOffset: Int(gateView.scaleOffset),
                                       biases: gateView.buffer,
                                       biasesOffset: Int(gateView.biasOffset),
                                       x: ffnInput,
                                       y: sharedScalarGateBuf!,
                                       m: 1, n: D)
                elementwise!.encodeSigmoidScalarMul(commandBuffer: sharedCB,
                                                    y: h1Buf,
                                                    gate: sharedScalarGateBuf!,
                                                    count: cfg.hiddenSize)
            }
            sharedCB.commit()
            }
            if let cb = phase1HitCB {
                cb.commit()
            }
            if rdadviseEnabled && rdadvisePolicyMode != .off {
                let requestedMisses = plannedFetch?.misses.count ?? experts.count
                let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                    layer: L,
                    missCount: requestedMisses)
                if let skipped = shouldSkipRDAdvice(position: position,
                                                    requestedMisses: requestedMisses,
                                                    estimatedBytes: estimatedAdviceBytes,
                                                    canOverlapUsefulGPUWork: true) {
                    recordRDAdvice(skipped, wallNanos: 0)
                } else {
                    let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let result: ExpertIOAdviceResult
                    if let plannedFetch {
                        result = try model.adviseRoutedExperts(plan: plannedFetch)
                    } else {
                        result = try model.adviseRoutedExperts(layer: L, experts: experts)
                    }
                    let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                    recordRDAdvice(result, wallNanos: wallNanos)
                    updateRDAdvicePolicy(after: result, position: position)
                }
            }

            // Routed-expert pread — overlaps the shared MLP GPU work above.
            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs: [TensorView]
            if let plannedFetch {
                blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
            } else {
                blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
            }
            fireLookahead()
            let layerIo = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
            totalIoNanos &+= layerIo
            let routedBufs = blobs.map { $0.buffer }
            let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let gTail: (MTLCommandBuffer) -> Void
            if cfg.ffnSandwichNorms {
                let postF2 = try model.postFFN2(layer: L)
                let postF = try model.postFFN(layer: L)
                let layerScalarView = try model.layerScalar(layer: L)
                let scalarPtr = layerScalarView.buffer.contents()
                    .advanced(by: Int(layerScalarView.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let layerScalar = Quantization.bf16ToFloat(scalarPtr[0])
                gTail = { [self] cb in
                    fusedTail.encode(commandBuffer: cb,
                                     h2: h2Buf,
                                     h1: h1Buf,
                                     hidden: hidden,
                                     postFFN2Weight: postF2.buffer,
                                     postFFN2WeightOffset: Int(postF2.offset),
                                     postFFNWeight: postF.buffer,
                                     postFFNWeightOffset: Int(postF.offset),
                                     d: D,
                                     eps: eps,
                                     layerScalar: layerScalar)
                }
            } else if usesHyperConnections {
                // Same shape as the plain tail, but the feed-forward output
                // goes into all four streams through the gate the layer's
                // second hyper-connection produced.
                gTail = { [self] cb in
                    encodeHyperConnectionWrite(cb, branch: h2Buf)
                }
            } else {
                // The phase-2 reduce already folded the shared branch (h1Buf
                // as its residual); the tail is a plain residual add.
                gTail = { [self] cb in
                    if let flintResidual {
                        // The add joins the next norm (layer input or final head).
                        if flintDefersResidualAdd {
                            flintPendingDelta = h2Buf
                        } else {
                            flintResidual.add(cb, delta: h2Buf, mirror: hidden, count: cfg.hiddenSize)
                        }
                    } else {
                        elementwise!.encodeResidualAdd(commandBuffer: cb, hidden: hidden,
                                                       delta: h2Buf, count: cfg.hiddenSize)
                    }
                }
            }
            let routedCB = ctx.queue.makeCommandBuffer()!
            let splitArgBuf = phase1HitSplitArgBuf != nil && !phase1MissSlots.isEmpty
                ? phase1HitSplitArgBuf
                : nil
            let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs,
                topK: topK)
            if splitArgBuf != nil {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                encodeRoutedPhase1Subset(
                    routedCB,
                    argBuf: argBuf,
                    routedBufs: routedBufs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count))
            } else {
                encodeRoutedPhase1Full(routedCB,
                                       argBuf: argBuf,
                                       routedBufs: routedBufs)
            }
            moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: routedBufs,
                                                   routedOffsets: routedOffsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: (!cfg.ffnSandwichNorms && cfg.hasSharedExpert)
                                                       ? h1Buf : zeroResidual,
                                                   y: h2Buf,
                                                   d: D,
                                                   f: FmoE,
                                                   topK: topK, bitWidths: routedBitWidths)
            gTail(routedCB)
            routedCB.commit()
            precondition(pendingRoutedCommand == nil,
                         "routed command-buffer pipeline drained before queuing the next layer")
            pendingRoutedCommand = PendingRoutedCommand(
                cb: routedCB,
                sharedCB: sharedCB,
                phase1HitCB: phase1HitCB,
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
            continue
        }
        if let pending = pendingRoutedCommand {
            try finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        // A hyper-connection architecture has no `model.norm`: the mixer that
        // collapses its streams is the last thing before the head, and it
        // carries the only normalization left.
        let usesMixer = cfg.hyperConnection.isEnabled
        let fNorm = usesMixer ? nil : model.finalNorm
        let lm    = model.lmHead
        let headInput = usesMixer ? (hcMixed ?? normed) : normed
        let gFinalNorm: (MTLCommandBuffer) -> Void = { cb in
            if usesMixer {
                // `layer: nil` selects the model-level mixer, which has no
                // injection gate and only collapses.
                try? self.encodeHyperConnectionRead(cb, slot: .attention, layer: nil)
            } else if let fNorm {
                if let prism = self.prism {
                    prism.encodeNorm(commandBuffer: cb, weight: fNorm, input: self.hidden,
                                     output: self.normed, width: self.cfg.hiddenSize, rows: 1, eps: eps)
                } else if let flintResidual = self.flintResidual {
                    if let delta = flintPendingDelta {
                        flintResidual.addNorm(cb, delta: delta, mirror: self.hidden,
                                              weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                              output: self.normed, count: self.cfg.hiddenSize, eps: eps)
                        flintPendingDelta = nil
                    } else {
                        flintResidual.norm(cb, weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                           output: self.normed, count: self.cfg.hiddenSize, eps: eps)
                    }
                } else {
                self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                     weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                     out: self.normed, d: D, eps: eps)
                }
            }
        }
        let gLmHead: (MTLCommandBuffer) throws -> Void = { cb in
            if let prism = self.prism {
                try prism.encode(commandBuffer: cb, model: self.model, weight: lm,
                                 input: headInput, output: logits, columns: self.cfg.hiddenSize,
                                 outputs: self.cfg.vocabSize, rows: 1,
                                 inputStride: self.cfg.hiddenSize, outputStride: self.cfg.vocabSize)
            } else if self.cfg.family == .flint {
                encodeFlintProjection(cb, view: lm, x: headInput, y: logits,
                                      rows: UInt32(self.cfg.vocabSize), columns: D)
            } else {
            self.int4.encode(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: headInput, y: logits,
                             m: UInt32(self.cfg.vocabSize), n: D)
            }
        }
        let gFusionHead: (MTLCommandBuffer) -> Void = { cb in
            // Flint norms its FP32 residual first; the fused kernel then takes
            // the already-normalized row instead of the half mirror.
            let flintNormed: MTLBuffer?
            if self.cfg.family == .flint {
                gFinalNorm(cb)
                flintNormed = self.normed
            } else {
                flintNormed = nil
            }
            self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normedInput: flintNormed,
                normWeight: fNorm!.buffer, normOffset: Int(fNorm!.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        if let sink = debugLayerSink {
            try drainForDebug()
            sink(cfg.numLayers)
        }
        // A dense token carries its layers on `firstLayerCommandBuffer`; the
        // head joins them so the token costs exactly one commit and one wait.
        let pendingTokenCommandBuffer = firstLayerCommandBuffer
        firstLayerCommandBuffer = nil
        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if !useFusedHeadForThisToken { flintFiniteLogits?.prepare() }
            let head: (MTLCommandBuffer) throws -> Void = { cb in
                if useFusedHeadForThisToken {
                    if self.prism != nil {
                        // Prism's head is a rotated Q2 projection; take the GPU argmax of its logits.
                        gFinalNorm(cb)
                        try gLmHead(cb)
                        self.greedyArgmax.encode(commandBuffer: cb, values: logits,
                                                 count: self.cfg.vocabSize, output: self.greedyTokenBuf)
                    } else {
                        gFusionHead(cb)
                    }
                } else { gFinalNorm(cb); try gLmHead(cb) }
                // The fused head never writes logits, so there is nothing to check.
                if !useFusedHeadForThisToken {
                    try self.flintFiniteLogits?.encode(commandBuffer: cb, logits: logits, count: self.cfg.vocabSize)
                }
            }
            if let cb = pendingTokenCommandBuffer {
                try head(cb)
                epilogue?(cb)
                cb.commit()
                waitUntilCompleted(cb)
                try checkCommandBufferError(cb)
                totalGPULayerNanos &+= UInt64(max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1e9))
                totalGPULayerCommandBuffers &+= 1
            } else {
                try runSync(head)
            }
            if !useFusedHeadForThisToken { try flintFiniteLogits?.validate() }
            if useFusedHeadForThisToken {
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        } else if let cb = pendingTokenCommandBuffer {
            cb.commit()
            waitUntilCompleted(cb)
            try checkCommandBufferError(cb)
            totalGPULayerNanos &+= UInt64(max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1e9))
            totalGPULayerCommandBuffers &+= 1
        }

        if emitHead { GPUOpProfiler.shared?.endStep(kind: "decode") }
        kv?.advance()
    }

    /// Dense Gemma 4 tail, including the per-layer embedding mapping:
    /// post-attention sandwich, dense gated MLP, PLE residual, layer scale.
    private func encodeDenseGemmaLayerDecode(
        _ cb: MTLCommandBuffer,
        layer: Int,
        postAttention: TensorView,
        projections: LayerSharedExpertProjections,
        d: UInt32,
        eps: Float
    ) throws {
        guard let elementwise else {
            preconditionFailure("dense Gemma layer requires elementwise kernels")
        }
        let preFFN = try model.preFFN(layer: layer)
        rms.encodeBF16W(commandBuffer: cb,
                        x: oOut,
                        weight: postAttention.buffer,
                        weightOffset: Int(postAttention.offset),
                        out: h1Buf, d: d, eps: eps)
        elementwise.encodeResidualAdd(commandBuffer: cb,
                                      hidden: hidden, delta: h1Buf,
                                      count: cfg.hiddenSize)
        rms.encodeBF16W(commandBuffer: cb,
                        x: hidden,
                        weight: preFFN.buffer,
                        weightOffset: Int(preFFN.offset),
                        out: denseX, d: d, eps: eps)
        try shared.encode(commandBuffer: cb,
                          x: denseX,
                          gate: projections.gate,
                          up: projections.up,
                          down: projections.down,
                          y: h1Buf,
                          scratchGate: denseScratchGate,
                          scratchUp: denseScratchUp,
                          scratchAct: denseScratchAct)
        let postFFN = projections.postF1!
        rms.encodeBF16W(commandBuffer: cb,
                        x: h1Buf,
                        weight: postFFN.buffer,
                        weightOffset: Int(postFFN.offset),
                        out: h1Buf, d: d, eps: eps)
        elementwise.encodeResidualAdd(commandBuffer: cb,
                                      hidden: hidden, delta: h1Buf,
                                      count: cfg.hiddenSize)

        let layerScalar = try model.layerScalar(layer: layer)
        let scalarBits = layerScalar.buffer.contents()
            .advanced(by: Int(layerScalar.offset))
            .assumingMemoryBound(to: UInt16.self)[0]
        let layerScale = Quantization.bf16ToFloat(scalarBits)
        guard cfg.hasPerLayerInputs else {
            elementwise.encodeScale(commandBuffer: cb, values: hidden,
                                    count: cfg.hiddenSize, scale: layerScale)
            return
        }
        guard let perLayerEmbeddingKernel,
              let pleIdentity, let pleContext, let pleLayer, let pleGate else {
            preconditionFailure("PLE architecture requires PLE kernels and scratch")
        }

        let plan = PerLayerEmbeddingPlan(config: cfg)
        let pleNorm = model.perLayerProjectionNorm
        perLayerEmbeddingKernel.encodePrepareLayer(
            commandBuffer: cb,
            identity: pleIdentity,
            context: pleContext,
            normWeight: pleNorm.buffer,
            normWeightOffset: Int(pleNorm.offset),
            out: pleLayer,
            rows: 1,
            packedWidth: plan.packedWidth,
            pleWidth: plan.perLayerHiddenSize,
            layer: layer,
            contextScale: plan.contextProjectionScale,
            combinedScale: plan.combinedScale,
            eps: eps)
        let inputGate = try model.perLayerInputGate(layer: layer)
        int4.encode(commandBuffer: cb,
                    weights: inputGate.buffer, weightsOffset: Int(inputGate.offset),
                    scales: inputGate.buffer, scalesOffset: Int(inputGate.scaleOffset),
                    biases: inputGate.buffer, biasesOffset: Int(inputGate.biasOffset),
                    x: hidden, y: pleGate,
                    m: UInt32(plan.perLayerHiddenSize), n: d)
        elementwise.encodeGELUMul(commandBuffer: cb,
                                  gate: pleGate, value: pleLayer, out: pleGate,
                                  count: plan.perLayerHiddenSize)
        let projection = try model.perLayerProjection(layer: layer)
        int4.encode(commandBuffer: cb,
                    weights: projection.buffer, weightsOffset: Int(projection.offset),
                    scales: projection.buffer, scalesOffset: Int(projection.scaleOffset),
                    biases: projection.buffer, biasesOffset: Int(projection.biasOffset),
                    x: pleGate, y: h2Buf,
                    m: d, n: UInt32(plan.perLayerHiddenSize))
        let postPLE = try model.postPerLayerInputNorm(layer: layer)
        rms.encodeBF16W(commandBuffer: cb,
                        x: h2Buf,
                        weight: postPLE.buffer,
                        weightOffset: Int(postPLE.offset),
                        out: h2Buf, d: d, eps: eps)
        elementwise.encodeResidualAddScale(
            commandBuffer: cb,
            hidden: hidden,
            delta: h2Buf,
            count: cfg.hiddenSize,
            scale: layerScale)
    }

    /// Gated-DeltaNet linear attention (layer mask 2), one decode step.
    /// Reads `normed`, updates the layer's recurrent state + conv tail in
    /// place, and leaves the attention-branch output in `oOut`.
    /// Encode the n-gram per-layer embedding for one token and add it to the
    /// residual streams.
    ///
    ///     embedding = table[hash(recent tokens)]        // read on the host
    ///     keys      = norm_key(key_proj(embedding))
    ///     values    = value_proj(embedding)
    ///     query     = norm_query(hidden)
    ///     gated     = sigmoid(signedSqrt(keys·query / sqrt(D))) * values
    ///     hidden   += gated + silu(conv(norm_conv(gated)))
    ///
    /// The rows are resolved and read on the host: sixteen rows of a hundred
    /// bytes, scattered by a hash across 320 million, is a latency problem
    /// rather than a bandwidth one.
    /// One token's n-gram per-layer embedding, folded into `residual` at
    /// `residualOffset`.
    ///
    /// The chunk path drives this per token rather than batching it: only the
    /// dilated convolution is sequential across tokens, the layer carrying it
    /// is one of forty-eight, and its projections are small — so a loop costs
    /// little and avoids a second, differently-shaped implementation of the
    /// history window.
    private func encodeNgramPLE(_ cb: MTLCommandBuffer,
                                layer L: Int,
                                token: Int32,
                                residual: MTLBuffer? = nil,
                                residualOffset: Int = 0,
                                embeddingStaging: MTLBuffer? = nil,
                                embeddingRow: Int = 0) throws {
        guard let ple = ngramPLE, let rowIndex = ngramRows,
              let table = ngramTable,
              let embedding = embeddingStaging ?? pleEmbedding, let keys = pleKeys,
              let values = pleValues, let queries = pleQueries,
              let gated = pleGated, let normedConv = pleNormed,
              let convOut = pleConvOut, let history = pleConvHistory else {
            preconditionFailure("n-gram encode without its scratch")
        }
        let ngram = cfg.ngramEmbedding
        let streams = cfg.hyperConnection.streamCount
        let width = UInt32(cfg.hiddenSize)
        let stacked = UInt32(streams * cfg.hiddenSize)

        let (rows, nextContext) = rowIndex.rows(for: [token],
                                                context: ngramContext)
        ngramContext = nextContext
        let gathered = try table.embedding(rows: rows)
        guard gathered.count == ngram.embedDim else {
            throw ModelError.indexCorrupt(detail:
                "n-gram lookup produced \(gathered.count) values, expected "
                    + "\(ngram.embedDim)")
        }
        let embeddingOffset = embeddingRow * ngram.embedDim
        // A chunk stages one row per token, and an image span is a chunk of
        // its own pooled length rather than `maxChunkTokens`. Sizing this from
        // a constant wrote past the buffer and took the decode service down
        // with a bus error on the first photo large enough to matter.
        let capacity = embedding.length / MemoryLayout<Float16>.stride
        guard embeddingOffset + gathered.count <= capacity else {
            throw PrefillError.chunkedUnsupported(
                "n-gram staging holds \(capacity / max(ngram.embedDim, 1)) rows, "
                    + "needed row \(embeddingRow)")
        }
        embedding.contents().withMemoryRebound(
            to: Float16.self,
            capacity: capacity
        ) { destination in
            for index in 0..<gathered.count {
                destination[embeddingOffset + index] = gathered[index]
            }
        }
        let embeddingByteOffset = embeddingOffset * MemoryLayout<Float16>.stride

        let keyProjection = try model.ngramKeyProjection(layer: L)
        int4.encode(commandBuffer: cb,
                    weights: keyProjection.buffer,
                    weightsOffset: Int(keyProjection.offset),
                    scales: keyProjection.buffer,
                    scalesOffset: Int(keyProjection.scaleOffset),
                    biases: keyProjection.buffer,
                    biasesOffset: Int(keyProjection.biasOffset),
                    x: embedding, xOffset: embeddingByteOffset, y: keys,
                    m: stacked, n: UInt32(ngram.embedDim))
        let keyNorm = try model.ngramKeyNorm(layer: L)
        rms.encodeBF16WGroupedCentered(
            commandBuffer: cb, x: keys,
            weight: keyNorm.buffer, weightOffset: Int(keyNorm.offset),
            out: keys, d: width, groups: streams, eps: 1e-6)

        let valueProjection = try model.ngramValueProjection(layer: L)
        int4.encode(commandBuffer: cb,
                    weights: valueProjection.buffer,
                    weightsOffset: Int(valueProjection.offset),
                    scales: valueProjection.buffer,
                    scalesOffset: Int(valueProjection.scaleOffset),
                    biases: valueProjection.buffer,
                    biasesOffset: Int(valueProjection.biasOffset),
                    x: embedding, xOffset: embeddingByteOffset, y: values,
                    m: width, n: UInt32(ngram.embedDim))

        let target = residual ?? hidden
        let queryNorm = try model.ngramQueryNorm(layer: L)
        rms.encodeBF16WGroupedCentered(
            commandBuffer: cb, x: target, xOffset: residualOffset,
            weight: queryNorm.buffer, weightOffset: Int(queryNorm.offset),
            out: queries, d: width, groups: streams, eps: 1e-6)

        ple.encodeGateValues(commandBuffer: cb, keys: keys, queries: queries,
                             values: values, out: gated,
                             hiddenSize: cfg.hiddenSize, streamCount: streams)

        let convNorm = try model.ngramConvNorm(layer: L)
        rms.encodeBF16WGroupedCentered(
            commandBuffer: cb, x: gated,
            weight: convNorm.buffer, weightOffset: Int(convNorm.offset),
            out: normedConv, d: width, groups: streams, eps: 1e-6)

        // The convolution is dilated by the n-gram size, so its taps read
        // every `ngramSize`-th row of the history.
        let historyRows = (ngram.convKernelSize - 1) * ngram.ngramSize + 1
        ple.encodeHistoryPush(commandBuffer: cb, history: history,
                              row: normedConv,
                              width: Int(stacked), length: historyRows)
        let convWeight = try model.ngramConv1D(layer: L)
        ple.encodeDepthwiseConv(commandBuffer: cb, history: history,
                                weight: convWeight.buffer,
                                weightOffset: Int(convWeight.offset),
                                out: convOut,
                                width: Int(stacked),
                                taps: ngram.convKernelSize,
                                dilation: ngram.ngramSize)

        elementwise!.encodeResidualAdd(commandBuffer: cb,
                                       hidden: target, hiddenOffset: residualOffset,
                                       delta: gated, count: Int(stacked))
        elementwise!.encodeResidualAdd(commandBuffer: cb,
                                       hidden: target, hiddenOffset: residualOffset,
                                       delta: convOut, count: Int(stacked))
    }

    /// Encode a hyper-connection's read side, leaving the block's input in
    /// `hcMixed` and the per-stream injection gate in `hcInject`.
    ///
    ///     normed = hc_norm(hidden)                 // grouped, centered
    ///     mixed  = mean_s sigmoid(W_up @ silu(W_down @ normed / S)) * normed
    ///     inject = W_inject @ normed               // gated in encodeInject
    ///
    /// `hidden` itself is untouched: the residual that survives the block is
    /// the unnormalized stream, and `normed` only feeds the mix and the gate.
    private func encodeHyperConnectionRead(
        _ cb: MTLCommandBuffer,
        slot: Model.HyperConnectionSlot,
        layer L: Int?
    ) throws {
        guard let hc = hyperConnection,
              let mixed = hcMixed, let lowRank = hcLowRank,
              let mixUp = hcMixUp else {
            preconditionFailure("hyper-connection encode without its scratch")
        }
        let config = cfg.hyperConnection
        let streams = config.streamCount
        let width = UInt32(cfg.hiddenSize)
        let stacked = UInt32(streams * cfg.hiddenSize)

        let norm = try model.hyperConnectionNorm(slot: slot, layer: L)
        rms.encodeBF16WGroupedCentered(
            commandBuffer: cb,
            x: hidden,
            weight: norm.buffer, weightOffset: Int(norm.offset),
            out: normed,
            d: width, groups: streams, eps: 1e-6)

        let down = try model.hyperConnectionMixDown(slot: slot, layer: L)
        int4.encode(commandBuffer: cb,
                    weights: down.buffer, weightsOffset: Int(down.offset),
                    scales: down.buffer, scalesOffset: Int(down.scaleOffset),
                    biases: down.buffer, biasesOffset: Int(down.biasOffset),
                    x: normed, y: lowRank,
                    m: UInt32(config.lowRank), n: stacked)
        hc.encodeLowRankSilu(commandBuffer: cb,
                             x: lowRank, out: lowRank,
                             count: config.lowRank,
                             streamCount: streams)

        let up = try model.hyperConnectionMixUp(slot: slot, layer: L)
        int4.encode(commandBuffer: cb,
                    weights: up.buffer, weightsOffset: Int(up.offset),
                    scales: up.buffer, scalesOffset: Int(up.scaleOffset),
                    biases: up.buffer, biasesOffset: Int(up.biasOffset),
                    x: lowRank, y: mixUp,
                    m: stacked, n: UInt32(config.lowRank))
        hc.encodeCombine(commandBuffer: cb,
                         up: mixUp, normed: normed, mixed: mixed,
                         hiddenSize: cfg.hiddenSize, streamCount: streams)

        // The model-level mixer only collapses; it has no injection gate.
        guard let L, let inject = hcInject else { return }
        let gate = try model.hyperConnectionInject(slot: slot, layer: L)
        int4.encode(commandBuffer: cb,
                    weights: gate.buffer, weightsOffset: Int(gate.offset),
                    scales: gate.buffer, scalesOffset: Int(gate.scaleOffset),
                    biases: gate.buffer, biasesOffset: Int(gate.biasOffset),
                    x: normed, y: inject,
                    m: UInt32(streams), n: stacked)
    }

    /// Accumulate a block's output into every residual stream through the gate
    /// `encodeHyperConnectionRead` left in `hcInject`.
    private func encodeHyperConnectionWrite(_ cb: MTLCommandBuffer,
                                            branch: MTLBuffer,
                                            branchOffset: Int = 0) {
        guard let hc = hyperConnection, let inject = hcInject else {
            preconditionFailure("hyper-connection write without its scratch")
        }
        hc.encodeInject(commandBuffer: cb,
                        hyper: hidden,
                        branch: branch, branchOffset: branchOffset,
                        injectionRaw: inject,
                        hiddenSize: cfg.hiddenSize,
                        streamCount: cfg.hyperConnection.streamCount)
    }

    private func encodeLinearAttentionDecode(_ cb: MTLCommandBuffer, layer L: Int) throws {
        guard let gdn, let gdnState, let gdnQKVRaw, let gdnConvOut,
              let gdnZ, let gdnA, let gdnB, let gdnY, let gdnOut else {
            preconditionFailure("linear-attention layer without GDN kernels")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)
        let qkvW = try model.linearInProjQKV(layer: L)
        let zW = try model.linearInProjZ(layer: L)
        let aW = try model.linearInProjA(layer: L)
        let bW = try model.linearInProjB(layer: L)
        let outW = try model.linearOutProj(layer: L)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.linearALog(layer: L)
        let dtBias = try model.linearDtBias(layer: L)
        let gatedNormW = try model.linearNorm(layer: L)

        if let prism, let prismGDN {
            if prism.supportsDual(columns: cfg.hiddenSize, outputsA: la.qkvDim) {
                try prism.encodeDual(commandBuffer: cb, model: model, weightA: qkvW, weightB: zW,
                                     input: blockInput, outputA: gdnQKVRaw, outputB: gdnZ,
                                     columns: cfg.hiddenSize, outputsA: la.qkvDim, outputsB: la.valueDim)
            } else {
                try prism.encode(commandBuffer: cb, model: model, weight: qkvW,
                                 input: blockInput, output: gdnQKVRaw, columns: cfg.hiddenSize,
                                 outputs: la.qkvDim, rows: 1, inputStride: cfg.hiddenSize, outputStride: la.qkvDim)
                try prism.encode(commandBuffer: cb, model: model, weight: zW,
                                 input: blockInput, output: gdnZ, columns: cfg.hiddenSize,
                                 outputs: la.valueDim, rows: 1, inputStride: cfg.hiddenSize, outputStride: la.valueDim)
            }
            prism.encodeFloatDual(commandBuffer: cb, weightA: aW, weightB: bW, input: blockInput,
                                  outputA: gdnA, outputB: gdnB, columns: cfg.hiddenSize, outputs: la.numVHeads)
            prismGDN.encodeConv(commandBuffer: cb, tail: gdnState.convTailBuffer(layer: L),
                                qkv: gdnQKVRaw, weight: convW, output: gdnConvOut)
            gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
            prismGDN.encodeDelta(commandBuffer: cb, convOut: gdnConvOut, a: gdnA, b: gdnB,
                                 aLog: aLog, dtBias: dtBias, state: gdnState.stateBuffer(layer: L), output: gdnY)
            prismGDN.encodeGatedNorm(commandBuffer: cb, y: gdnY, z: gdnZ, weight: gatedNormW, output: gdnOut)
            try prism.encode(commandBuffer: cb, model: model, weight: outW,
                             input: gdnOut, output: oOut, columns: la.valueDim,
                             outputs: cfg.hiddenSize, rows: 1, inputStride: la.valueDim, outputStride: cfg.hiddenSize)
            return
        }

        // One dispatch over the concatenated qkv/z/a/b row space instead of four
        // separate GEMVs (a and b were 4 threadgroups each).
        gdn.encodeInputProjections(commandBuffer: cb,
                                   x: blockInput,
                                   qkv: qkvW, qkvOut: gdnQKVRaw,
                                   z: zW, zOut: gdnZ,
                                   a: aW, aOut: gdnA,
                                   b: bW, bOut: gdnB,
                                   hiddenSize: cfg.hiddenSize)

        gdn.encodeConvDecode(commandBuffer: cb,
                             tail: gdnState.convTailBuffer(layer: L),
                             qkv: gdnQKVRaw,
                             convWeight: convW.buffer,
                             convWeightOffset: Int(convW.offset),
                             out: gdnConvOut)
        gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        gdn.encodeDeltaStepDecode(commandBuffer: cb,
                                  convOut: gdnConvOut,
                                  aProj: gdnA,
                                  bProj: gdnB,
                                  aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                  dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                                  state: gdnState.stateBuffer(layer: L),
                                  y: gdnY)
        gdn.encodeGatedNorm(commandBuffer: cb,
                            y: gdnY,
                            z: gdnZ,
                            weight: gatedNormW.buffer,
                            weightOffset: Int(gatedNormW.offset),
                            out: gdnOut)
        int4.encode(commandBuffer: cb,
                    weights: outW.buffer, weightsOffset: Int(outW.offset),
                    scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                    biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                    x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim))
    }

    /// Qwen full attention (attn_output_gate), one decode step: packed
    /// [query ; gate] q_proj split per head, weighted per-head q/k norms
    /// (no V norm), NeoX sub-dim RoPE, full attention with the configured
    /// scale, sigmoid output gate, then o_proj into `oOut`.
    /// Per-head Q/K RMSNorm, in whichever form the architecture stores its
    /// weights. Qwen3.6 centres them at one and scales by `w`; Qwen4-Exp
    /// centres them at zero and scales by `1 + w`, so the same tensor name
    /// means different arithmetic and using the wrong one leaves Q and K
    /// scaled by roughly zero.
    private func encodeHeadNorm(_ cb: MTLCommandBuffer,
                                x: MTLBuffer, xOffset: Int = 0,
                                weight: MTLBuffer, weightOffset: Int = 0,
                                out: MTLBuffer, outOffset: Int = 0,
                                headDim: UInt32, numHeads: Int,
                                eps: Float) {
        if cfg.normWeightsCenteredAtZero {
            rms.encodeBF16WPerHeadCentered(
                commandBuffer: cb, x: x, xOffset: xOffset,
                weight: weight, weightOffset: weightOffset,
                out: out, outOffset: outOffset,
                headDim: headDim, numHeads: numHeads, eps: eps)
        } else {
            rms.encodeBF16WPerHead(
                commandBuffer: cb, x: x, xOffset: xOffset,
                weight: weight, weightOffset: weightOffset,
                out: out, outOffset: outOffset,
                headDim: headDim, numHeads: numHeads, eps: eps)
        }
    }

    private func encodeGatedFullAttentionDecode(_ cb: MTLCommandBuffer,
                                                layer L: Int,
                                                position: Int,
                                                seqLen: UInt32) throws {
        guard let elementwise, let rope, let qPackedScratch, let attnGateScratch else {
            preconditionFailure("attn_output_gate layer without gate kernels")
        }
        guard let kv else {
            preconditionFailure("FP16 attention requires an FP16 KV cache")
        }
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot = kv.kSlot(layer: L, position: position)
        let vSlot = kv.vSlot(layer: L, position: position)
        let q = try model.qProj(layer: L)
        let k = try model.kProj(layer: L)
        let v = try model.vProj(layer: L)
        let o = try model.oProj(layer: L)
        let qNormW = try model.qNorm(layer: L)
        let kNormW = try model.kNorm(layer: L)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

        if let prism {
            for (weight, output, offset, count) in [
                (q, qPackedScratch, 0, Int(2 * qDim)),
                (k, kSlot.buffer, kSlot.offset, Int(kvDim)),
                (v, vSlot.buffer, vSlot.offset, Int(kvDim)),
            ] {
                try prism.encode(commandBuffer: cb, model: model, weight: weight,
                                 input: blockInput, output: output, columns: cfg.hiddenSize,
                                 outputs: count, rows: 1, outputOffset: offset,
                                 inputStride: cfg.hiddenSize, outputStride: count)
            }
        } else {
        fusedQKVGEMV.encode(commandBuffer: cb,
                            qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                            qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                            qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                            kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                            kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                            kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                            vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                            vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                            vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                            // With hyper-connections the block reads the
                            // stream mixture, not `normed` — which here holds
                            // all four normalized streams stacked, so a GEMV
                            // over the first D values would silently read
                            // stream 0 alone.
                            x: blockInput,
                            qOut: qPackedScratch,
                            kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                            vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                            qRows: 2 * qDim,
                            kvRows: kvDim,
                            n: D)
        }
        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: qPackedScratch,
                                     q: qScratch,
                                     gate: attnGateScratch,
                                     heads: cfg.numHeads,
                                     dim: headDim)
        if let prism {
            prism.encodeNorm(commandBuffer: cb, weight: qNormW, input: qScratch, output: qScratch,
                             width: headDim, rows: cfg.numHeads, eps: eps)
            prism.encodeNorm(commandBuffer: cb, weight: kNormW, input: kSlot.buffer, output: kSlot.buffer,
                             width: headDim, rows: numKV, eps: eps,
                             inputOffset: kSlot.offset, outputOffset: kSlot.offset)
        } else {
        encodeHeadNorm(cb,
                               x: qScratch,
                               weight: qNormW.buffer,
                               weightOffset: Int(qNormW.offset),
                               out: qScratch,
                               headDim: UInt32(headDim),
                               numHeads: cfg.numHeads,
                               eps: eps)
        encodeHeadNorm(cb,
                               x: kSlot.buffer, xOffset: kSlot.offset,
                               weight: kNormW.buffer,
                               weightOffset: Int(kNormW.offset),
                               out: kSlot.buffer, outOffset: kSlot.offset,
                               headDim: UInt32(headDim),
                               numHeads: numKV,
                               eps: eps)
        }
        let ropePosition = position + Int(qwenMultimodalRopeDelta)
        precondition(ropePosition >= 0, "Qwen multimodal RoPE position must be nonnegative")
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: qScratch,
                              position: UInt32(ropePosition),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(cfg.numHeads),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: kSlot.buffer,
                              dataOffset: kSlot.offset,
                              position: UInt32(ropePosition),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(numKV),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        if let qsa = qwenSparseAttention {
            let weights = try model.resident(name:
                "language_model.model.layers.\(L).self_attn.indexer.index_qk_proj.weight")
            let projected = try qsa.projectionBuffer(tokens: 1)
            int4.encode(commandBuffer: cb,
                weights: weights.buffer, weightsOffset: Int(weights.offset),
                scales: weights.buffer, scalesOffset: Int(weights.scaleOffset),
                biases: weights.buffer, biasesOffset: Int(weights.biasOffset),
                x: blockInput, y: projected,
                m: UInt32(cfg.attentionIndexer.projectionRows), n: D)
            try qsa.encode(commandBuffer: cb, model: model, layer: L,
                projected: projected, start: position, tokens: 1,
                positions: nil, ropeDelta: qwenMultimodalRopeDelta,
                q: qScratch, k: kSlot.buffer, v: vSlot.buffer, out: attnOut)
        }
        if qwenSparseAttention?.usesSparseAttention(endPosition: position + 1) != true {
        attention.encodeFull(commandBuffer: cb,
                             q: qScratch,
                             k: kSlot.buffer, kOffset: 0,
                             v: vSlot.buffer, vOffset: 0,
                             out: attnOut,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(cfg.numHeads),
                             numKVHeads: UInt32(numKV),
                             seqLen: seqLen,
                             scale: Float(cfg.attentionScale))
        }
        elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: attnOut,
                                         gate: attnGateScratch,
                                         count: Int(qDim))
        if let prism {
            try prism.encode(commandBuffer: cb, model: model, weight: o,
                             input: attnOut, output: oOut, columns: Int(qDim), outputs: cfg.hiddenSize,
                             rows: 1, inputStride: Int(qDim), outputStride: cfg.hiddenSize)
        } else {
        int4.encode(commandBuffer: cb,
                    weights: o.buffer, weightsOffset: Int(o.offset),
                    scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                    biases: o.buffer, biasesOffset: Int(o.biasOffset),
                    x: attnOut, y: oOut, m: D, n: qDim)
        }
    }

    private func runSync(_ body: (MTLCommandBuffer) throws -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        try body(cb)
        cb.commit()
        try waitForCompletion(cb)
        totalGPUHeadNanos &+= UInt64(max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1e9))
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        waitUntilCompleted(cb)
        prefillClock.waited(cb, wallNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
        try checkCommandBufferError(cb)
    }

    private nonisolated func waitUntilCompleted(_ cb: MTLCommandBuffer) {
        cb.waitUntilCompleted()
    }

}

// MARK: - Decode kernel benchmark (diagnostic)

extension RealForwardRunner {
    /// Times the decode projections against this model's own resident tensors
    /// and reports achieved memory bandwidth, then times command-buffer round
    /// trips. Decode is memory-bound, so those two numbers say whether a token
    /// is limited by the kernels or by submission latency. The CLI exposes it
    /// behind `TUFF_KERNEL_BENCH=1`.
    public func benchmarkDecodeKernels(repetitions: Int = 20) throws -> String {
        if prism != nil {
            return try benchmarkPrismAffine2(repetitions: repetitions)
        }
        var report = ""
        let D = UInt32(cfg.hiddenSize)
        guard let scratchLogits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        // Real activations: an all-zero input can be measured optimistically.
        for buffer in [normed, denseX, attnOut, hidden, denseScratchAct] {
            let count = buffer.length / MemoryLayout<Float16>.size
            let pointer = buffer.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<count { pointer[i] = Float16(sin(Float(i) * 0.37) * 0.5) }
        }
        func gemvBytes(_ view: TensorView) -> Double {
            Double(view.length + view.scaleLength + view.biasLength)
        }
        /// Best of several rounds. Each timed buffer is preceded by a burst of
        /// real work so nothing is measured at idle GPU clocks.
        func time(_ label: String, bytesPerRep: Double,
                  _ body: @escaping (MTLCommandBuffer) throws -> Void) throws {
            let cb = ctx.queue.makeCommandBuffer()!
            try body(cb)
            cb.commit()
            cb.waitUntilCompleted()
            let lm = model.lmHead
            var best = Double.infinity
            for _ in 0..<4 {
                let heater = ctx.queue.makeCommandBuffer()!
                for _ in 0..<2 {
                    int4.encode(commandBuffer: heater,
                                weights: lm.buffer, weightsOffset: Int(lm.offset),
                                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                x: normed, y: scratchLogits,
                                m: UInt32(cfg.vocabSize), n: D)
                }
                heater.commit()
                let timed = ctx.queue.makeCommandBuffer()!
                for _ in 0..<repetitions { try body(timed) }
                timed.commit()
                timed.waitUntilCompleted()
                try checkCommandBufferError(timed)
                best = min(best, timed.gpuEndTime - timed.gpuStartTime)
            }
            report += String(format: "%-34@ %9.1f us  %7.1f GB/s  (%.1f MB)\n",
                             label as NSString,
                             best / Double(repetitions) * 1e6,
                             bytesPerRep * Double(repetitions) / best / 1e9,
                             bytesPerRep / 1e6)
        }
        func gemv(_ label: String, _ view: TensorView, x: MTLBuffer, y: MTLBuffer) throws {
            try time("\(label) \(view.shape.0)x\(view.shape.1)",
                     bytesPerRep: gemvBytes(view)) { [self] cb in
                int4.encode(commandBuffer: cb,
                            weights: view.buffer, weightsOffset: Int(view.offset),
                            scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                            biases: view.buffer, biasesOffset: Int(view.biasOffset),
                            x: x, y: y, m: view.shape.0, n: view.shape.1)
            }
        }
        for L in Set([0, cfg.numLayers / 2, cfg.numLayers - 1]).sorted() {
            try gemv("L\(L) q_proj", try model.qProj(layer: L), x: normed, y: qScratch)
            try gemv("L\(L) o_proj", try model.oProj(layer: L), x: attnOut, y: oOut)
            if cfg.hasSharedExpert {
                try gemv("L\(L) gate_proj", try model.sharedExpertGate(layer: L),
                         x: denseX, y: denseScratchGate)
                try gemv("L\(L) down_proj", try model.sharedExpertDown(layer: L),
                         x: denseScratchAct, y: h1Buf)
            }
        }
        try gemv("lm_head", model.lmHead, x: normed, y: scratchLogits)

        // Wall-clock cost of a command-buffer round trip. Every one of these a
        // token pays is latency no kernel change can recover.
        let inputNorm = try model.inputNorm(layer: 0)
        func wall(_ label: String, iterations: Int, _ body: () -> Void) {
            for _ in 0..<3 { body() }
            var best = UInt64.max
            for _ in 0..<5 {
                let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                for _ in 0..<iterations { body() }
                best = min(best, clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
            }
            report += String(format: "%-34@ %9.1f us wall\n", label as NSString,
                             Double(best) / Double(iterations) / 1000.0)
        }
        func encodeNorms(_ cb: MTLCommandBuffer, count: Int) {
            for _ in 0..<count {
                rms.encodeBF16W(commandBuffer: cb, x: hidden,
                                weight: inputNorm.buffer,
                                weightOffset: Int(inputNorm.offset),
                                out: normed, d: D, eps: 1e-6)
            }
        }
        wall("empty CB commit+wait", iterations: 200) { [self] in
            let cb = ctx.queue.makeCommandBuffer()!
            cb.commit()
            cb.waitUntilCompleted()
        }
        wall("16 norms, one CB", iterations: 100) { [self] in
            let cb = ctx.queue.makeCommandBuffer()!
            encodeNorms(cb, count: 16)
            cb.commit()
            cb.waitUntilCompleted()
        }
        wall("16 norms, 16 CBs", iterations: 20) { [self] in
            for _ in 0..<16 {
                let cb = ctx.queue.makeCommandBuffer()!
                encodeNorms(cb, count: 1)
                cb.commit()
                cb.waitUntilCompleted()
            }
        }
        return report
    }

    /// Measure Q2 projections without Hadamard using actual resident pack tensors.
    private func benchmarkPrismAffine2(repetitions: Int) throws -> String {
        precondition(repetitions > 0)
        let affine = try PrismAffine2(context: ctx)
        let head = model.lmHead
        let layers = cfg.numLayers
        let repetitions = max(repetitions, layers * 2)
        let gates = try (0..<layers).map { try model.sharedExpertGate(layer: $0) }
        let ups = try (0..<layers).map { try model.sharedExpertUp(layer: $0) }
        let gateAndUp = (0..<layers).flatMap { [gates[$0], ups[$0]] }
        let downs = try (0..<layers).map { try model.sharedExpertDown(layer: $0) }
        let maxColumns = max(Int(head.shape.1), Int(downs[0].shape.1))
        guard let input = ctx.device.makeBuffer(length: maxColumns * 2, options: .storageModeShared),
              let output = ctx.device.makeBuffer(length: Int(head.shape.0) * 2, options: .storageModeShared),
              let reference = ctx.device.makeBuffer(length: Int(head.shape.0) * 2, options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let values = input.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<maxColumns { values[i] = Float16(sin(Float(i) * 0.37) * 0.5) }
        var report = "Prism Q2 GEMV, group=128, FP16 input/output; best GPU time per dispatch. "
            + "Per-layer tensors rotate; the gate/up case covers both projections in every layer.\n"
        // (label, views cycled per repetition, rows to use)
        let cases: [(String, [TensorView], Int)] = [
            ("gate_proj", gates, Int(gates[0].shape.0)),
            ("gate_up_all_layers", gateAndUp, Int(gates[0].shape.0)),
            ("gate_proj rows[0..<2048]", gates, 2048),
            ("down_proj", downs, Int(downs[0].shape.0)),
            ("lm_head", [head], Int(head.shape.0)),
        ]
        for (label, views, rows) in cases {
            let columns = Int(views[0].shape.1)
            func encode(_ cb: MTLCommandBuffer, _ index: Int, _ variant: PrismAffine2.Variant, _ out: MTLBuffer) {
                let view = views[index % views.count]
                affine.encode(commandBuffer: cb, input: input, weights: view.buffer,
                              scales: view.buffer, biases: view.buffer, output: out,
                              weightsOffset: Int(view.offset), scalesOffset: Int(view.scaleOffset),
                              biasesOffset: Int(view.biasOffset), columns: columns, outputs: rows, rows: 1,
                              variant: variant)
            }
            let check = ctx.queue.makeCommandBuffer()!
            encode(check, 0, .rows4, reference)
            check.commit()
            check.waitUntilCompleted()
            try checkCommandBufferError(check)
            let expected = Array(UnsafeBufferPointer(start: reference.contents().assumingMemoryBound(to: Float16.self), count: rows))
            let bytes = Double(rows * columns) * (0.25 + 4.0 / 128.0)
            for variant in PrismAffine2.Variant.allCases {
                for shared in [false, true] where variant == .rows4 || !shared {
                    let verify = ctx.queue.makeCommandBuffer()!
                    encode(verify, 0, variant, output)
                    verify.commit()
                    verify.waitUntilCompleted()
                    let actual = Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float16.self), count: rows))
                    var worst: Float = 0
                    for i in 0..<rows { worst = max(worst, abs(Float(actual[i]) - Float(expected[i]))) }
                    var best = Double.infinity
                    for _ in 0..<5 {
                        let cb = ctx.queue.makeCommandBuffer()!
                        if shared { SharedComputeEncoding.share(cb) }
                        for index in 0..<repetitions { encode(cb, index, variant, output) }
                        cb.commit()
                        cb.waitUntilCompleted()
                        try checkCommandBufferError(cb)
                        best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / Double(repetitions))
                    }
                    report += String(format: "%-26@ %5dx%-5d %-6@ %@ %9.3f us %7.2f GB/s  max|diff vs rows4| %.4f\n",
                                     label as NSString, rows, columns, variant.rawValue as NSString,
                                     (shared ? "shared " : "separate") as NSString,
                                     best * 1e6, bytes / best / 1e9, worst)
                }
            }
        }
        return report
    }
}

extension RealForwardRunner: PrefixCheckpointingRunner {
    /// Rows that later tokens only append after stay where they are: Gemma
    /// and Qwen's full-attention KV and the sparse indexer's keys and
    /// completed blocks. A checkpoint copies what is overwritten in place:
    /// sliding-window rings, Qwen's Gated DeltaNet recurrent and convolution
    /// state, and Flash Next's n-gram history. An image RoPE offset would
    /// also have to be undone, so a runner holding one declines.
    public var supportsPrefixCheckpoints: Bool {
        kv != nil && qwenMultimodalRopeDelta == 0
    }

    public var prefixCheckpointByteEstimate: Int {
        checkpointBuilder(position: continuationPosition).byteCount
    }

    private func checkpointBuilder(position: Int) -> RunnerStateSnapshotBuilder {
        var builder = RunnerStateSnapshotBuilder()
        if let kv {
            kv.addSnapshotRanges(to: &builder, position: position,
                                 skipLayer: { [self] in snapshotSkipsLayer($0) || !kv.isRingLayer($0) })
        }
        gdnState?.addSnapshotRanges(to: &builder)
        if let pleConvHistory {
            builder.add("ngram.conv", pleConvHistory, length: pleConvHistory.length)
        }
        return builder
    }

    public func capturePrefixCheckpoint() throws -> RunnerStateSnapshot {
        guard supportsPrefixCheckpoints, speculativeStartPosition == nil else {
            throw RunnerStateSnapshotError.unsupported("this runner cannot take prefix checkpoints")
        }
        let position = continuationPosition
        return try checkpointBuilder(position: position).capture(
            owner: self, queue: ctx.queue,
            host: .init(position: position, ngramContext: ngramContext, ropeDelta: 0))
    }

    public func rewind(to checkpoint: RunnerStateSnapshot) throws {
        guard checkpoint.owner == ObjectIdentifier(self) else {
            throw RunnerStateSnapshotError.foreignSnapshot
        }
        let position = checkpoint.host.position
        do {
            guard supportsPrefixCheckpoints, let kv, speculativeStartPosition == nil,
                  position > 0, position <= kv.position,
                  checkpoint.host.ngramContext.count == ngramContext.count else {
                throw RunnerStateSnapshotError.layoutMismatch(
                    "checkpoint at \(position) cannot be reached from \(continuationPosition)")
            }
            try prefillChunkState.requireClean(operation: "rewind")
            try checkpointBuilder(position: position).restore(checkpoint, queue: ctx.queue)
            kv.rewind(to: position)
            ngramContext = checkpoint.host.ngramContext
            resetTransientState()
        } catch {
            reset()
            throw error
        }
    }
}

extension RealForwardRunner: StateSnapshottingRunner {
    public var stateSnapshotByteEstimate: Int? {
        guard speculativeTransactionIsIdle else { return nil }
        let position = continuationPosition
        var total = kv?.snapshotBytes(position: position, skipLayer: snapshotSkipsLayerForEstimate) ?? 0
        total += gdnStateSnapshotBytes
        total += sparseIndexerSnapshotBytes(position: position)
        total += ngramHistorySnapshotBytes
        return total
    }

    public func captureState() throws -> RunnerStateSnapshot {
        guard speculativeTransactionIsIdle else {
            throw RunnerStateSnapshotError.unsupported("a speculative verification is in progress")
        }
        let position = continuationPosition
        return try snapshotBuilderForCapture(position: position).capture(
            owner: self, queue: commandQueueForSnapshots,
            host: .init(position: position, ngramContext: ngramContextForSnapshot,
                        ropeDelta: ropeDeltaForSnapshot))
    }

    public func restoreState(_ snapshot: RunnerStateSnapshot) throws {
        guard snapshot.owner == ObjectIdentifier(self) else {
            throw RunnerStateSnapshotError.foreignSnapshot
        }
        reset()
        do {
            try applySnapshot(snapshot)
        } catch {
            reset()
            throw error
        }
    }
}

// MARK: - Flint chunked prefill

/// Flint prefill batches a chunk of prompt rows per layer. Attention runs the
/// decode kernels row by row inside one command buffer, so a row's numerics
/// are a decode token's; the routed experts are grouped by expert so each is
/// read from SSD once per chunk rather than once per token. The last prompt
/// token goes through the ordinary decode path, which writes the logits and
/// any fused-greedy seed.
extension RealForwardRunner {
    func prefillFlintBatched(tokens: ArraySlice<Int32>,
                             startPosition: Int,
                             outputMode: PrefillOutputMode,
                             config: PrefillRuntimeConfig,
                             into logits: MTLBuffer,
                             onProgress: (Int) -> Void) async throws -> PrefillResult {
        let total = tokens.count
        let batched = total - 1
        if batched > 0 {
            let capacity = min(config.chunkTokens, prefillTextChunkCapacity)
            let state: FlintBatchedPrefillState
            if let existing = flintPrefill, existing.rows >= capacity {
                state = existing
            } else {
                state = try FlintBatchedPrefillState(context: ctx, config: cfg, rows: capacity,
                                                     silu: cfg.hiddenActivation == "silu")
                flintPrefill = state
            }
            var offset = 0
            while offset < batched {
                let count = min(capacity, batched - offset)
                let lower = tokens.index(tokens.startIndex, offsetBy: offset)
                let upper = tokens.index(lower, offsetBy: count)
                try Task.checkCancellation()
                try await encodeFlintPrefillChunk(tokens: tokens[lower..<upper],
                                                  startPosition: startPosition + offset,
                                                  state: state)
                offset += count
                onProgress(offset)
            }
        }
        try await produceToken(token: tokens[tokens.index(before: tokens.endIndex)],
                               position: startPosition + total - 1, into: logits,
                               emitHead: true, outputMode: outputMode)
        onProgress(total)
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + total, seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + total, seed: .logitsWritten)
    }

    private func encodeFlintPrefillChunk(tokens: ArraySlice<Int32>,
                                         startPosition: Int,
                                         state: FlintBatchedPrefillState) async throws {
        guard let kv, let rope, let flintEmbedding else {
            throw PrefillError.chunkedUnsupported("Flint chunked prefill requires KV, RoPE and embedding kernels")
        }
        let T = tokens.count
        precondition(T > 0 && T <= state.rows)
        let D = cfg.hiddenSize
        let K = cfg.topKExperts
        let F = cfg.moeIntermediateSize
        let half = MemoryLayout<Float16>.stride
        let eps: Float = 1e-6
        try kv.ensureCapacity(through: startPosition + T, on: ctx.queue)
        _ = lookahead.drain()
        if let ringLayer = (0..<cfg.numLayers).first(where: { kv.ringCapacity(layer: $0) > 0 }) {
            let required = min(maxContext, cfg.slidingWindow + T)
            guard required <= kv.ringCapacity(layer: ringLayer) else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(kv.ringCapacity(layer: ringLayer)) cannot hold \(required) rows for a Flint prefill chunk")
            }
        }
        let tokenPointer = state.tokens.contents().assumingMemoryBound(to: UInt32.self)
        for (row, token) in tokens.enumerated() { tokenPointer[row] = UInt32(bitPattern: token) }
        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: T)

        func makeBuffer(_ label: String) throws -> MTLCommandBuffer {
            guard let cb = ctx.queue.makeCommandBuffer() else { throw ModelError.residentBufferWrapFailed }
            SharedComputeEncoding.share(cb)
            cb.label = "flint prefill start=\(startPosition) rows=\(T) \(label)"
            return cb
        }

        func bits(_ view: TensorView, rows: UInt32, columns: UInt32) -> Int {
            view.length == UInt64(rows) * UInt64(columns) ? 8 : 4
        }

        func project(_ cb: MTLCommandBuffer, _ view: TensorView,
                     x: MTLBuffer, xOffset: Int, y: MTLBuffer, yOffset: Int,
                     rows: UInt32, columns: UInt32) {
            if bits(view, rows: rows, columns: columns) == 8 {
                flintInt8!.encode(commandBuffer: cb,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: rows, n: columns)
            } else {
                int4.encode(commandBuffer: cb,
                    weights: view.buffer, weightsOffset: Int(view.offset),
                    scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                    biases: view.buffer, biasesOffset: Int(view.biasOffset),
                    x: x, xOffset: xOffset, y: y, yOffset: yOffset, m: rows, n: columns)
            }
        }

        var cb = try makeBuffer("layer=0")
        flintEmbedding.encodeRows(commandBuffer: cb, table: model.embedding,
                                  tokens: state.tokens, residual: state.residual.values,
                                  rows: T, width: UInt32(D))
        var pendingDelta: MTLBuffer?

        for L in 0..<cfg.numLayers {
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isLast = L == cfg.numLayers - 1
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKV = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = UInt32(cfg.numHeads * headDim)
            let kvDim = UInt32(numKV * headDim)
            let inNorm = try model.inputNorm(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            if let delta = pendingDelta {
                state.residual.addNorm(cb, delta: delta, mirror: state.attentionOut,
                                       weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                       output: state.normed, count: D, eps: eps, rows: T)
                pendingDelta = nil
            } else {
                state.residual.norm(cb, weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                    output: state.normed, count: D, eps: eps, rows: T)
            }

            let q = try model.qProj(layer: L)
            let k = try model.kProj(layer: L)
            let v = try model.vProj(layer: L)
            let o = try model.oProj(layer: L)
            let qNorm = try model.qNorm(layer: L)
            let kNorm = try model.kNorm(layer: L)
            let sinks = try model.attentionSinks(layer: L)
            let fusedProjection = bits(q, rows: qDim, columns: UInt32(D)) == 4
                && bits(k, rows: kvDim, columns: UInt32(D)) == 4
                && bits(v, rows: kvDim, columns: UInt32(D)) == 4
            for row in 0..<T {
                let position = startPosition + row
                let kSlot = kv.kSlot(layer: L, position: position)
                let vSlot = kv.vSlot(layer: L, position: position)
                let xOffset = row * D * half
                let qOffset = row * state.attentionStride * half
                if isLast {
                    // Only this layer's K/V rows are needed: nothing reads its output.
                    project(cb, k, x: state.normed, xOffset: xOffset, y: kSlot.buffer, yOffset: kSlot.offset,
                            rows: kvDim, columns: UInt32(D))
                    project(cb, v, x: state.normed, xOffset: xOffset, y: vSlot.buffer, yOffset: vSlot.offset,
                            rows: kvDim, columns: UInt32(D))
                } else if fusedProjection {
                    fusedQKVGEMV.encode(commandBuffer: cb,
                        qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                        qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                        qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                        kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                        kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                        kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                        vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                        vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                        vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                        x: state.normed, xOffset: xOffset,
                        qOut: state.queries, qOutOffset: qOffset,
                        kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                        vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                        qRows: qDim, kvRows: kvDim, n: UInt32(D))
                } else {
                    project(cb, q, x: state.normed, xOffset: xOffset, y: state.queries, yOffset: qOffset,
                            rows: qDim, columns: UInt32(D))
                    project(cb, k, x: state.normed, xOffset: xOffset, y: kSlot.buffer, yOffset: kSlot.offset,
                            rows: kvDim, columns: UInt32(D))
                    project(cb, v, x: state.normed, xOffset: xOffset, y: vSlot.buffer, yOffset: vSlot.offset,
                            rows: kvDim, columns: UInt32(D))
                }
                if !isLast {
                    rms.encodeBF16WPerHead(commandBuffer: cb, x: state.queries, xOffset: qOffset,
                        weight: qNorm.buffer, weightOffset: Int(qNorm.offset),
                        out: state.queries, outOffset: qOffset,
                        headDim: UInt32(headDim), numHeads: cfg.numHeads, eps: eps)
                }
                rms.encodeBF16WPerHead(commandBuffer: cb, x: kSlot.buffer, xOffset: kSlot.offset,
                    weight: kNorm.buffer, weightOffset: Int(kNorm.offset),
                    out: kSlot.buffer, outOffset: kSlot.offset,
                    headDim: UInt32(headDim), numHeads: numKV, eps: eps)
                if !isFull {
                    if !isLast {
                        rope.encodeDefaultNeox(commandBuffer: cb, data: state.queries, dataOffset: qOffset,
                            position: UInt32(position), headDim: UInt32(headDim),
                            numHeads: UInt32(cfg.numHeads), theta: Float(cfg.ropeTheta))
                    }
                    rope.encodeDefaultNeox(commandBuffer: cb, data: kSlot.buffer, dataOffset: kSlot.offset,
                        position: UInt32(position), headDim: UInt32(headDim),
                        numHeads: UInt32(numKV), theta: Float(cfg.ropeTheta))
                }
                if isLast { continue }
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                        q: state.queries, qOffset: qOffset,
                        k: kSlot.buffer, kOffset: 0, v: vSlot.buffer, vOffset: 0,
                        out: state.context, outOffset: qOffset,
                        headDim: UInt32(headDim), numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKV), seqLen: UInt32(position + 1),
                        scale: Float(cfg.attentionScale),
                        sinks: sinks.buffer, sinksOffset: Int(sinks.offset))
                } else {
                    let ringCapacity = kv.ringCapacity(layer: L)
                    let activeRing = ringCapacity > 0 && position + 1 > ringCapacity ? UInt32(ringCapacity) : 0
                    attention.encodeSWA(commandBuffer: cb,
                        q: state.queries, qOffset: qOffset,
                        k: kSlot.buffer, kOffset: 0, v: vSlot.buffer, vOffset: 0,
                        out: state.context, outOffset: qOffset,
                        headDim: UInt32(headDim), numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKV), seqLen: UInt32(position + 1),
                        window: UInt32(cfg.slidingWindow), scale: Float(cfg.attentionScale),
                        sinks: sinks.buffer, sinksOffset: Int(sinks.offset), ringCapacity: activeRing)
                }
                project(cb, o, x: state.context, xOffset: qOffset, y: state.attentionOut,
                        yOffset: row * D * half, rows: UInt32(D), columns: qDim)
            }
            if isLast {
                cb.commit()
                try waitForCompletion(cb)
                break
            }

            let sharedProjection = sharedExpertProjections[L]!
            func encodeSharedRows(_ target: MTLCommandBuffer, x: MTLBuffer) throws {
                for row in 0..<T {
                    try shared.encode(commandBuffer: target,
                        x: x, xOffset: row * D * half,
                        gate: sharedProjection.gate, up: sharedProjection.up, down: sharedProjection.down,
                        y: state.sharedOut, yOffset: row * D * half,
                        scratchGate: state.sharedGate, scratchGateOffset: row * state.sharedStride * half,
                        scratchUp: state.sharedUp, scratchUpOffset: row * state.sharedStride * half,
                        scratchAct: state.sharedAct, scratchActOffset: row * state.sharedStride * half)
                }
            }

            if L < 2 {
                state.residual.addNorm(cb, delta: state.attentionOut, mirror: state.denseX,
                                       weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                                       output: state.denseX, count: D, eps: eps, rows: T)
                try encodeSharedRows(cb, x: state.denseX)
                pendingDelta = state.sharedOut
                continue
            }

            // Moe layer: attention add + post-attention norm, route every row.
            state.residual.addNorm(cb, delta: state.attentionOut, mirror: state.routedX,
                                   weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                                   output: state.routedX, count: D, eps: eps, rows: T)
            let routerWeight = try model.router(layer: L)
            state.router.encode(commandBuffer: cb, weight: routerWeight.buffer,
                weightOffset: Int(routerWeight.offset), hidden: state.residual.normalized,
                indices: state.routeIDs, weights: state.routeWeights, hiddenSize: D,
                experts: cfg.numExperts, topK: K, floatInput: true,
                affineScalesOffset: routerWeight.dtype == 0 ? Int(routerWeight.scaleOffset) : nil,
                affineBiasesOffset: routerWeight.dtype == 0 ? Int(routerWeight.biasOffset) : nil,
                rows: T)
            cb.commit()
            try waitForCompletion(cb)

            let routeCount = T * K
            let idPointer = state.routeIDs.contents().bindMemory(to: UInt32.self, capacity: routeCount)
            let weightPointer = state.routeWeights.contents().bindMemory(to: Float16.self, capacity: routeCount)
            var routeIDs = [UInt32](), routeWeights = [Float16]()
            routeIDs.reserveCapacity(routeCount)
            routeWeights.reserveCapacity(routeCount)
            for i in 0..<routeCount {
                routeIDs.append(min(idPointer[i], UInt32(cfg.numExperts - 1)))
                routeWeights.append(weightPointer[i])
            }
            debugFlintPrefillRouteSink?(L, startPosition, routeIDs)
            let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs, weights: routeWeights,
                                                           queryCount: T, topK: K)
            let slotCount = model.routedExpertCacheSlotCount(layer: L) ?? 16
            let tileExperts = max(1, min(16, slotCount / 2))
            let schedulerConfig = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 1, tileExperts: tileExperts)
            guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                throw PrefillError.chunkedUnsupported(
                    "Flint prefill needs at least 2 expert-cache slots, has \(slotCount)")
            }
            let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                pairs, queryCount: T, topK: K, numExperts: cfg.numExperts,
                tileExpertCount: tileExperts,
                expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))
            let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(device: ctx.device, routes: routes)
            let routedOffsets = model.routedExpertOffsets(layer: L)
            let routedWidths = model.routedExpertBitWidths(layer: L)

            // The shared expert has no dependence on the routed experts, so it
            // runs while the first tile is read.
            let sharedCB = try makeBuffer("layer=\(L) shared")
            try encodeSharedRows(sharedCB, x: state.routedX)
            sharedCB.commit()

            struct PendingTile {
                let index: Int
                let commandBuffer: MTLCommandBuffer
                let fetch: PrefillStreamedTileFetchResult
                let argument: PrefillStreamedTileArgumentBuffer
            }
            var inFlight: [PendingTile] = []
            var lifetime = PrefillStreamedTileSlotLifetime()
            func drainOldest() throws {
                guard !inFlight.isEmpty else { return }
                let tile = inFlight.removeFirst()
                try withExtendedLifetime((tile.fetch, tile.argument)) {
                    try waitForCompletion(tile.commandBuffer)
                }
                if !tile.fetch.plannedMissSlots.isEmpty { try lifetime.complete(tileIndex: tile.index) }
            }
            do {
                for (tileIndex, tile) in routes.tiles.enumerated() {
                    let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes)
                    var plannedFetch: RoutedExpertFetchPlan?
                    var avoiding = Set<Int>()
                    if !inFlight.isEmpty {
                        avoiding = Set(inFlight.flatMap { $0.fetch.plannedAssignedSlots })
                        if !avoiding.isEmpty,
                           let plan = try model.planRoutedExpertsIfPossible(
                               layer: L, experts: expertIDs, avoidingSlots: avoiding) {
                            plannedFetch = plan
                        } else {
                            while !inFlight.isEmpty { try drainOldest() }
                            avoiding = []
                        }
                    }
                    try Task.checkCancellation()
                    let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                        model: model, layer: L, tileIndex: tileIndex, routes: routes,
                        plannedFetch: plannedFetch, avoidingSlots: avoiding)
                    try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                          pairStart: Int(tile.pairStart),
                                                          pairCount: Int(tile.pairCount))
                    if !fetch.plannedMissSlots.isEmpty {
                        try lifetime.begin(tileIndex: tileIndex, plannedSlots: fetch.plannedMissSlots)
                    }
                    let argument = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                        device: ctx.device, binding: fetch.binding)
                    let tileCB = try makeBuffer("layer=\(L) tile=\(tileIndex)")
                    state.experts.encodeTile(
                        commandBuffer: tileCB, argumentBuffer: argument.buffer,
                        views: fetch.binding.views, expertIDs: fetch.binding.expertIDs,
                        offsets: routedOffsets, widths: routedWidths,
                        x: state.routedX, sortedPairs: metadata.sortedPairs,
                        pairStart: Int(tile.pairStart), pairCount: Int(tile.pairCount),
                        d: D, f: F, topK: K,
                        acts: state.acts[tileIndex & 1], partials: state.partials)
                    tileCB.commit()
                    inFlight.append(PendingTile(index: tileIndex, commandBuffer: tileCB,
                                                fetch: fetch, argument: argument))
                    // Depth one: the next tile's read overlaps this tile's compute.
                    while inFlight.count > 1 { try drainOldest() }
                }
                while !inFlight.isEmpty { try drainOldest() }
            } catch {
                while !inFlight.isEmpty { try? drainOldest() }
                try? waitForCompletion(sharedCB)
                throw error
            }
            try waitForCompletion(sharedCB)
            let tailCB = try makeBuffer("layer=\(L) tail")
            state.experts.encodeReduce(commandBuffer: tailCB, partials: state.partials,
                                       sharedOut: state.sharedOut, y: state.routedOut,
                                       rows: T, d: D, topK: K)
            pendingDelta = state.routedOut
            cb = tailCB
        }
        kv.advance(by: T)
        prefillChunkState.markCommitted()
    }
}
