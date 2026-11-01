import Foundation
import Metal
import Darwin
import TUFFFormat

public struct ModelLoadStats: Sendable {
    public var manifestSha256Nanos: UInt64
    public var receiptValidationNanos: UInt64
    public var eagerSha256Nanos: UInt64

    public init(manifestSha256Nanos: UInt64 = 0,
                receiptValidationNanos: UInt64 = 0,
                eagerSha256Nanos: UInt64 = 0) {
        self.manifestSha256Nanos = manifestSha256Nanos
        self.receiptValidationNanos = receiptValidationNanos
        self.eagerSha256Nanos = eagerSha256Nanos
    }
}

/// Bounded routed-expert cache configuration.
public enum ExpertStreamingMode: Sendable {
    /// Read each expert into one of `slotCount` 2 MB-aligned cache slots.
    case pread(slotCount: Int)
}

/// Loaded `.gturbo/` model. Resident weights live behind tensor-safe mmap'd
/// `MTLBuffer` regions; routed expert weights live behind per-layer streaming
/// backends opened lazily on first touch.
public struct Model {
    public let device: MTLDevice
    public let config: ArchConfig
    public let streamingMode: ExpertStreamingMode
    public let expertCachePolicy: ExpertCachePolicy
    public let integrityPolicy: ModelIntegrityPolicy
    public var modelID: String { manifest.modelID }
    public var sourceSnapshotHash: String? { manifest.sourceSnapshotHash }
    public var sharedExpertWeightBits: Int { manifest.quant?.sharedExpert.weightBits ?? 8 }

    /// Start a cancellable, non-wiring GPU warm-up of the resident weights
    /// when they fit in half of memory alongside other allocations.
    public func startResidentWarmup() -> ResidentWarmup? {
        guard ResidentWarmup.eligible(
            payloadSize: residentIndex.header.residentSize,
            physicalMemory: ProcessInfo.processInfo.physicalMemory,
            stagesDenseLayers: residentBuffers.stagesDenseLayers) else { return nil }
        return ResidentWarmup.start(buffers: residentBuffers.metalBuffers, device: device)
    }

    let residentBuffers: ResidentBufferSet
    let residentIndex: ResidentIndex
    let packedExpertsLayout: PackedExpertsLayout
    let manifest: Manifest
    let directoryURL: URL
    let modelDirectory: GTurboModelDirectory

    /// Lazy state. Held inside a reference box so `Model` can stay a struct
    /// while still letting accessors mutate layer state via a serial queue.
    let streamersBox: StreamersBox
    let streamersQueue: DispatchQueue

    /// Cache slots that can provide useful residency or one cold-read tile of
    /// prefill overlap. The latter needs two top-8 tile banks even when a toy
    /// or future compact MoE owns fewer than 16 distinct experts.
    var effectiveExpertCacheSlotCount: Int {
        let configured: Int
        switch streamingMode {
        case .pread(let slotCount):
            configured = slotCount
        }
        let usefulMaximum = max(
            packedExpertsLayout.expertsPerLayer,
            RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill)
        return min(configured, usefulMaximum)
    }

    /// Ornith 1.5 122B's expert reads skip the file cache when its experts are
    /// more than twice this Mac's memory. On a 16 GB M2 its 65 GB of experts
    /// went through the cache at the same speed but pushed 0.9 to 1.2 GB to
    /// swap per 80-token answer, and that paging sent Auto to a plan 2.5x
    /// slower. Past the cache: 1.90 against 1.83 to 1.87 tokens per second, 0
    /// to 0.17 GB of swap writes.
    ///
    /// Size alone is not the rule. Qwen 3.8 Flash Next has 76 GB of experts,
    /// never paged with the cache, and decoded 10% slower without it.
    static func readsExpertsPastFileCache(variant: ModelVariant?, expertStride: UInt64,
                                          expertsPerLayer: Int, layers: Int,
                                          physicalMemory: UInt64) -> Bool {
        variant == .ornith15_122B && expertPayloadOutgrowsFileCache(expertStride: expertStride,
            expertsPerLayer: expertsPerLayer, layers: layers, physicalMemory: physicalMemory)
    }

    /// Whether every routed expert together is more than twice this Mac's memory.
    static func expertPayloadOutgrowsFileCache(expertStride: UInt64, expertsPerLayer: Int,
                                               layers: Int, physicalMemory: UInt64) -> Bool {
        guard physicalMemory > 0, expertsPerLayer > 0, layers > 0 else { return false }
        let perLayer = expertStride.multipliedReportingOverflow(by: UInt64(expertsPerLayer))
        let allLayers = perLayer.partialValue.multipliedReportingOverflow(by: UInt64(layers))
        let limit = physicalMemory.multipliedReportingOverflow(by: 2)
        return perLayer.overflow || allLayers.overflow
            || (!limit.overflow && allLayers.partialValue > limit.partialValue)
    }

    static func shouldBypassExpertFilesystemCache(expertStride: UInt64, minimumSlots: Int,
                                                 layers: Int, physicalMemory: UInt64) -> Bool {
        guard physicalMemory > 0, minimumSlots > 0, layers > 0 else { return false }
        let perLayer = expertStride.multipliedReportingOverflow(by: UInt64(minimumSlots))
        let allLayers = perLayer.partialValue.multipliedReportingOverflow(by: UInt64(layers))
        return perLayer.overflow || allLayers.overflow || allLayers.partialValue > physicalMemory / 4
    }

    final class StreamersBox: @unchecked Sendable {
        var streamers: [PreadExpertStreamer?]
        var retiredReadMetrics = ExpertReadMetrics()
        var layerVerified: [Bool]
        var pendingVerification: [Int: VerifiedExpertFile] = [:]
        init(numLayers: Int) {
            self.streamers = Array(repeating: nil, count: numLayers)
            self.layerVerified = Array(repeating: false, count: numLayers)
        }
    }

    init(device: MTLDevice,
         config: ArchConfig,
         streamingMode: ExpertStreamingMode,
         expertCachePolicy: ExpertCachePolicy,
         integrityPolicy: ModelIntegrityPolicy,
         residentBuffers: ResidentBufferSet,
         residentIndex: ResidentIndex,
         packedExpertsLayout: PackedExpertsLayout,
         manifest: Manifest,
         directoryURL: URL,
         modelDirectory: GTurboModelDirectory) {
        self.device = device
        self.config = config
        self.streamingMode = streamingMode
        self.expertCachePolicy = expertCachePolicy
        self.integrityPolicy = integrityPolicy
        self.residentBuffers = residentBuffers
        self.residentIndex = residentIndex
        self.packedExpertsLayout = packedExpertsLayout
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.modelDirectory = modelDirectory
        self.streamersBox = StreamersBox(numLayers: packedExpertsLayout.numLayers)
        self.streamersQueue = DispatchQueue(label: "turbo-fieldfare.expert-streamers")
    }

    public var stagesDenseLayers: Bool { residentBuffers.stagesDenseLayers }
    public var denseStagingBytes: UInt64 { residentBuffers.denseStagingBytes }
    public var residentProjectionLayers: Set<Int> {
        Set(0..<config.numLayers).subtracting(residentBuffers.streamedProjectionLayers)
    }
    func prepareDenseLayer(_ layer: Int) throws {
        // Staged projection callers drain prior GPU work before advancing a
        // layer. At the minimum slot count, keeping every previous layer's
        // experts buys no cache capacity beyond a single routed working set.
        if stagesDenseLayers,
           effectiveExpertCacheSlotCount == config.topKExperts,
           Self.shouldBypassExpertFilesystemCache(
               expertStride: packedExpertsLayout.expertStride, minimumSlots: config.topKExperts,
               layers: config.numLayers, physicalMemory: ProcessInfo.processInfo.physicalMemory) {
            releaseExpertResidency(except: layer)
        }
        try residentBuffers.prepareDenseLayer(layer)
    }

    /// Requires completed prior-layer GPU work and no cross-layer prefetch.
    /// Tensor views retain their buffers independently of the cache owner.
    func releaseExpertResidency(except layer: Int) {
        streamersQueue.sync {
            for index in streamersBox.streamers.indices where index != layer {
                guard let streamer = streamersBox.streamers[index] else { continue }
                streamersBox.retiredReadMetrics.add(streamer.readMetrics)
                streamersBox.streamers[index] = nil
            }
            // Verification results remain valid for these immutable files.
        }
    }

    // MARK: - Resident accessors

    public var embedding: TensorView {
        try! resident(name: "language_model.model.embed_tokens.weight")
    }

    /// Gemma 4 E2B/E4B token-identity PLE table. Its logical row width is
    /// `numLayers * hiddenSizePerLayerInput`.
    public var perLayerEmbedding: TensorView {
        try! resident(name: "language_model.model.embed_tokens_per_layer.weight")
    }

    /// Context projection from the main embedding into all packed PLE rows.
    public var perLayerModelProjection: TensorView {
        try! resident(name: "language_model.model.per_layer_model_projection.weight")
    }

    /// One shared gain vector applied independently to every packed PLE row.
    public var perLayerProjectionNorm: TensorView {
        try! resident(name: "language_model.model.per_layer_projection_norm.weight")
    }

    /// Gemma 4 ties lm_head to the embedding; Qwen 3.6 carries a separate
    /// `lm_head` tensor. The transpose for the lm_head GEMV path is the
    /// kernel's job, not the loader's.
    public var lmHead: TensorView {
        if config.tieWordEmbeddings { return embedding }
        return try! resident(name: "language_model.lm_head.weight")
    }

    public func qProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.q_proj.weight")
    }
    public func kProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.k_proj.weight")
    }
    public func vProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.v_proj.weight")
    }
    public func oProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.o_proj.weight")
    }
    public func qProjBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.q_proj.bias")
    }
    public func kProjBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.k_proj.bias")
    }
    public func vProjBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.v_proj.bias")
    }
    public func oProjBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.o_proj.bias")
    }
    public func attentionSinks(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.sinks")
    }
    /// Gemma's writer emits `.router.proj.weight` (no `.mlp.` segment);
    /// Qwen's router is the source-named `.mlp.gate.weight`.
    public func router(layer L: Int) throws -> TensorView {
        switch config.family {
        case .gemma4:
            return try resident(name: "language_model.model.layers.\(L).router.proj.weight")
        case .qwen36, .qwen4Exp, .flint:
            return try resident(name: "language_model.model.layers.\(L).mlp.gate.weight")
        case .gptOss:
            return try resident(name: "language_model.model.layers.\(L).mlp.router.weight")
        case .minimaxM2:
            return try resident(name: "language_model.model.layers.\(L).block_sparse_moe.gate.weight")
        }
    }
    public func routerBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.router.bias")
    }
    /// Shared-expert FFN. Gemma emits `.mlp.{gate,up,down}_proj.weight`
    /// without a `.shared_expert.` segment; Qwen keeps the source's
    /// `.mlp.shared_expert.{gate,up,down}_proj.weight` names.
    public func sharedExpertGate(layer L: Int) throws -> TensorView {
        try resident(name: sharedExpertName("gate_proj", layer: L))
    }
    public func sharedExpertUp(layer L: Int) throws -> TensorView {
        try resident(name: sharedExpertName("up_proj", layer: L))
    }
    public func sharedExpertDown(layer L: Int) throws -> TensorView {
        try resident(name: sharedExpertName("down_proj", layer: L))
    }
    private func sharedExpertName(_ proj: String, layer L: Int) -> String {
        if config.feedForwardKind == .dense || (config.family == .flint && L < 2) {
            return "language_model.model.layers.\(L).mlp.\(proj).weight"
        }
        switch config.family {
        case .gemma4:
            return "language_model.model.layers.\(L).mlp.\(proj).weight"
        case .qwen36, .qwen4Exp, .flint:
            return "language_model.model.layers.\(L).mlp.shared_expert.\(proj).weight"
        case .gptOss:
            return "language_model.model.layers.\(L).mlp.\(proj).weight"
        case .minimaxM2:
            return "language_model.model.layers.\(L).block_sparse_moe.\(proj).weight"
        }
    }

    public func routerCorrectionBias(layer L: Int) throws -> TensorView {
        try resident(name:
            "language_model.model.layers.\(L).block_sparse_moe.e_score_correction_bias")
    }
    /// Qwen-only scalar gate on the shared-expert branch: a `[1, hidden]`
    /// 8-bit projection whose sigmoid multiplies the shared FFN output.
    public func sharedExpertScalarGate(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.shared_expert_gate.weight")
    }
    public func inputNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).input_layernorm.weight")
    }
    public func postAttnNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_attention_layernorm.weight")
    }
    public var finalNorm: TensorView {
        try! resident(name: "language_model.model.norm.weight")
    }

    // MARK: - Per-head attention norms (Q/K only)
    //
    // `q_norm` and `k_norm` are RMSNorm with learnable scale, applied per head
    // before RoPE. `v_norm` has **no learnable weight** (no-scale RMSNorm) and
    // is therefore not stored as a tensor — the runtime uses an
    // explicit no-scale variant rather than consuming a unit-weight buffer.

    public func qNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.q_norm.weight")
    }
    public func kNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.k_norm.weight")
    }

    // MARK: - Feed-forward norms
    //
    // The Gemma 4 sandwich wraps two parallel FFN branches:
    //   pre_feedforward_layernorm        -> dense MLP input
    //   pre_feedforward_layernorm_2      -> routed expert input
    //   post_feedforward_layernorm_1     -> dense MLP output
    //   post_feedforward_layernorm_2     -> routed expert output
    //   post_feedforward_layernorm       -> combined (h1+h2) output

    public func preFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm.weight")
    }
    public func preFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm_2.weight")
    }
    public func postFFN1(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_1.weight")
    }
    public func postFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_2.weight")
    }
    public func postFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm.weight")
    }

    // MARK: - Per-layer embedding mapping (dense Gemma 4)

    public func perLayerInputGate(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).per_layer_input_gate.weight")
    }
    public func perLayerProjection(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).per_layer_projection.weight")
    }
    public func postPerLayerInputNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_per_layer_input_norm.weight")
    }

    // MARK: - Router auxiliaries
    //
    // `router.scale` is a per-feature multiplier on the router's input
    // (post-RMSNorm), fused with 1/sqrt(hidden_size). `per_expert_scale` is
    // applied to the top-k routing weights after softmax over top-k.

    public func routerScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.scale")
    }
    public func routerPerExpertScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.per_expert_scale")
    }

    /// Per-layer scalar gain applied to the entire residual stream at the end
    /// of the layer; shape `[1]`, BF16.
    public func layerScalar(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).layer_scalar")
    }

    // MARK: - Gated-DeltaNet linear attention (Qwen 3.6)
    //
    // Layers whose mask value is 2 replace full/sliding attention with the
    // gated delta rule. Projections are 4-bit affine; the depthwise conv
    // weight, A_log, dt_bias, and the gated output norm are BF16.

    public func linearInProjQKV(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_qkv.weight")
    }
    public func linearInProjZ(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_z.weight")
    }
    public func linearInProjA(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_a.weight")
    }
    public func linearInProjB(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_b.weight")
    }
    public func linearOutProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.out_proj.weight")
    }
    /// Depthwise causal conv weight, source shape `[convDim, kernel, 1]`, BF16.
    public func linearConv1d(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.conv1d.weight")
    }
    /// Per-value-head decay base, shape `[numVHeads]`, BF16.
    public func linearALog(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.A_log")
    }
    /// Per-value-head dt bias, shape `[numVHeads]`, BF16.
    public func linearDtBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.dt_bias")
    }
    /// Gated RMSNorm weight over the value head dim, shape `[valueHeadDim]`.
    public func linearNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.norm.weight")
    }

    // MARK: - n-gram table (Qwen4-Exp)

    /// Read an `[n]` int64 sidecar out of the resident image.
    ///
    /// The n-gram hash constants — the multipliers and each head's offset and
    /// vocabulary size — are stored rather than derived. The reference builds
    /// them from a seed and a prime search, but a stored table is what the
    /// weights were trained against, so these are read, not recomputed.
    func int64Sidecar(_ view: TensorView, count: Int) throws -> [Int64] {
        let bytes = count * MemoryLayout<Int64>.size
        guard Int(view.length) >= bytes else {
            throw ModelError.indexCorrupt(detail:
                "int64 sidecar holds \(view.length) bytes, expected \(bytes)")
        }
        let base = view.buffer.contents().advanced(by: Int(view.offset))
        return (0..<count).map { index in
            base.advanced(by: index * MemoryLayout<Int64>.size)
                .loadUnaligned(as: Int64.self)
        }
    }

    /// Row selection for the n-gram table, built from the checkpoint's own
    /// constants. Nil for every architecture without one.
    func makeNgramRowIndex(eosTokenID: Int32) throws -> NgramRowIndex? {
        let ngram = config.ngramEmbedding
        guard ngram.isEnabled else { return nil }
        let layer = ngram.layer
        return NgramRowIndex(
            multipliers: try int64Sidecar(
                try ngramLayerMultipliers(layer: layer), count: ngram.ngramSize),
            headOffsets: try int64Sidecar(
                try ngramHeadOffsets(layer: layer), count: ngram.heads),
            headVocabSizes: try int64Sidecar(
                try ngramHeadVocabSizes(layer: layer), count: ngram.heads),
            ngramSize: ngram.ngramSize,
            headsPerNgram: ngram.headsPerNgram,
            eosTokenID: eosTokenID)
    }

    /// Open the n-gram table beside the model. Nil when the architecture has
    /// none; throws when it should have one and the manifest disagrees.
    func makeNgramTableReader() throws -> NgramTableReader? {
        guard config.ngramEmbedding.isEnabled else { return nil }
        guard let table = manifest.ngramTable else {
            throw ModelError.indexCorrupt(detail:
                "the architecture has an n-gram table but the manifest has no layout")
        }
        guard table.rowWidth == config.ngramEmbedding.headDim else {
            throw ModelError.indexCorrupt(detail:
                "n-gram row width \(table.rowWidth) does not divide the "
                    + "embedding across \(config.ngramEmbedding.heads) heads")
        }
        return try NgramTableReader(
            fileURL: directoryURL.appendingPathComponent(table.file),
            rowWidth: table.rowWidth,
            groupSize: table.groupSize,
            rowCount: table.rowCount,
            shards: table.shards.map {
                NgramTableReader.Shard(
                    rowStart: $0.rowStart, rowCount: $0.rowCount,
                    weightOffset: $0.weightOffset,
                    scaleOffset: $0.scaleOffset,
                    biasOffset: $0.biasOffset)
            })
    }

    // MARK: - Hyper-connections (Qwen4-Exp)
    //
    // The multi-stream residual replaces both layer norms: a `qwen4_exp` layer
    // has no `input_layernorm` and no `post_attention_layernorm`, only the
    // `hc_norm` inside each of its two hyper-connection modules. A model-level
    // mixer collapses the streams before the LM head, and there is no separate
    // final norm either.

    /// Which of a layer's two hyper-connections a tensor belongs to.
    public enum HyperConnectionSlot: String, Sendable {
        case attention = "attn_hyper_connection"
        case feedForward = "mlp_hyper_connection"
    }

    private func hyperConnectionName(_ leaf: String,
                                     slot: HyperConnectionSlot,
                                     layer L: Int?) -> String {
        guard let L else {
            return "language_model.model.hyper_connection_mixer.\(leaf)"
        }
        return "language_model.model.layers.\(L).\(slot.rawValue).\(leaf)"
    }

    /// Grouped RMSNorm weight over the stacked streams, shape
    /// `[streamCount * hiddenSize]`. Centered at zero: scale by `1 + w`.
    public func hyperConnectionNorm(slot: HyperConnectionSlot,
                                    layer L: Int? = nil) throws -> TensorView {
        try resident(name: hyperConnectionName("hc_norm.weight", slot: slot, layer: L))
    }

    /// `[lowRank, streamCount * hiddenSize]` projection into the mix.
    public func hyperConnectionMixDown(slot: HyperConnectionSlot,
                                       layer L: Int? = nil) throws -> TensorView {
        try resident(name: hyperConnectionName(
            "input_mix_weight_down.weight", slot: slot, layer: L))
    }

    /// `[streamCount * hiddenSize, lowRank]` projection back out of the mix.
    public func hyperConnectionMixUp(slot: HyperConnectionSlot,
                                     layer L: Int? = nil) throws -> TensorView {
        try resident(name: hyperConnectionName(
            "input_mix_weight_up.weight", slot: slot, layer: L))
    }

    /// `[streamCount, streamCount * hiddenSize]` per-stream injection gate.
    /// Absent on the model-level mixer, which only collapses.
    public func hyperConnectionInject(slot: HyperConnectionSlot,
                                      layer L: Int) throws -> TensorView {
        try resident(name: hyperConnectionName(
            "block_inject_weight.weight", slot: slot, layer: L))
    }

    // MARK: - n-gram per-layer embeddings (Qwen4-Exp)

    private func pleName(_ leaf: String, layer L: Int) -> String {
        "language_model.model.layers.\(L).ple.\(leaf)"
    }

    public func ngramConv1D(layer L: Int) throws -> TensorView {
        try resident(name: pleName("conv1d.weight", layer: L))
    }
    public func ngramKeyProjection(layer L: Int) throws -> TensorView {
        try resident(name: pleName("key_proj.weight", layer: L))
    }
    public func ngramValueProjection(layer L: Int) throws -> TensorView {
        try resident(name: pleName("value_proj.weight", layer: L))
    }
    public func ngramConvNorm(layer L: Int) throws -> TensorView {
        try resident(name: pleName("norm_conv.weight", layer: L))
    }
    public func ngramKeyNorm(layer L: Int) throws -> TensorView {
        try resident(name: pleName("norm_key.weight", layer: L))
    }
    public func ngramQueryNorm(layer L: Int) throws -> TensorView {
        try resident(name: pleName("norm_query.weight", layer: L))
    }
    /// The three rolling-hash multipliers, `[ngramSize]` int64.
    public func ngramLayerMultipliers(layer L: Int) throws -> TensorView {
        try resident(name: pleName(
            "ple_embedding.layer_multipliers", layer: L))
    }
    /// First global row of each n-gram head, `[heads]` int64.
    public func ngramHeadOffsets(layer L: Int) throws -> TensorView {
        try resident(name: pleName(
            "ple_embedding.ngram_heads_offsets", layer: L))
    }
    /// Each head's own vocabulary size, `[heads]` int64.
    public func ngramHeadVocabSizes(layer L: Int) throws -> TensorView {
        try resident(name: pleName(
            "ple_embedding.ngram_heads_vocab_sizes", layer: L))
    }

    /// Resolve a tensor name to a `TensorView` against its resident region.
    /// Absolute file offsets are converted to offsets relative to that
    /// region's `MTLBuffer`.
    func resident(name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let resolved = try residentBuffers.resolve(tensorName: name)
        let residentFileOffset = resolved.regionFileOffset
        func checkedRelativeOffset(_ absolute: UInt64,
                                   size: UInt64,
                                   field: String) throws -> UInt64 {
            if size == 0 {
                guard absolute == 0 else {
                    throw ModelError.indexCorrupt(detail: "\(name).\(field) has an absent nonzero offset")
                }
                return 0
            }
            guard absolute >= residentFileOffset else {
                throw ModelError.indexCorrupt(detail: "\(name).\(field) precedes its resident region")
            }
            let sum = (absolute - residentFileOffset).addingReportingOverflow(resolved.bufferOffset)
            guard !sum.overflow else { throw ModelError.residentBufferWrapFailed }
            let relative = sum.partialValue
            guard relative <= UInt64(resolved.buffer.length),
                  size <= UInt64(resolved.buffer.length) - relative else {
                throw ModelError.indexCorrupt(detail: "\(name).\(field) exceeds its resident region")
            }
            return relative
        }
        let relativeOffset = try checkedRelativeOffset(
            entry.fileOffset, size: entry.sizeBytes, field: "weights")
        let scaleRel = try checkedRelativeOffset(
            entry.scaleOffset, size: entry.scaleSize, field: "scales")
        let biasRel = try checkedRelativeOffset(
            entry.biasOffset, size: entry.biasSize, field: "biases")
        return TensorView(
            buffer: resolved.buffer,
            offset: relativeOffset,
            length: entry.sizeBytes,
            scaleOffset: scaleRel, scaleLength: entry.scaleSize,
            biasOffset:  biasRel,  biasLength:  entry.biasSize,
            shape: entry.shape,
            dtype: entry.dtype, name: name)
    }

    /// Resolve a named Prism projection's explicit input sign vector. The
    /// source sidecar guarantees this is the same width as the logical input.
    func prismSigns(for weight: TensorView) throws -> TensorView {
        guard let name = weight.name, name.hasSuffix(".weight") else {
            throw ModelError.indexCorrupt(detail: "Prism weight has no resident tensor name")
        }
        return try resident(name: String(name.dropLast(".weight".count)) + ".signs")
    }

    // MARK: - Routed expert (lazy)

    /// First touch of layer L opens its backend + verifies SHA-256; subsequent
    /// touches reuse the open backend. The backend resolves the expert to an
    /// cache-slot `(MTLBuffer, offset)` pair.
    public func routedExpert(layer L: Int, expert E: Int) throws -> TensorView {
        try ensureLayerOpened(L)
        let backend = streamersQueue.sync { streamersBox.streamers[L]! }
        let r = try backend.loadExpert(layer: 0, expert: E)
        return TensorView(
            buffer: r.buffer,
            offset: r.offset,
            length: r.size,
            scaleOffset: 0, scaleLength: 0,
            biasOffset:  0, biasLength:  0,
            shape: (UInt32(L), UInt32(E), 0, 0),
            dtype: GTurboFormatV1.DType.u32.rawValue)
    }

    /// Open layer L's file + verify SHA, idempotent.
    func ensureLayerOpened(_ L: Int) throws {
        try streamersQueue.sync {
            try openLayerLocked(L)
        }
    }

    /// Best-effort overlap hook for prefill: starts the same lazy layer open on
    /// the model's streamer queue without waiting for the first expert fetch.
    public func beginOpeningRoutedExpertStreamer(layer L: Int) {
        nonisolated(unsafe) let model = self
        streamersQueue.async {
            try? model.openLayerLocked(L)
        }
    }

    private func openLayerLocked(_ L: Int) throws {
        if streamersBox.streamers[L] != nil {
            return
        }
        let basename = packedExpertsLayout.layers[L].file
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        let manifestRel = "packed_experts/\(basename)"
        let file = try streamersBox.pendingVerification.removeValue(forKey: L)
            ?? beginLayerVerificationLocked(L)
        // A single lookahead overlaps two full SHA-256 checks. Keep the
        // bound per model: arbitrary/out-of-order expert fetches must not
        // launch an unbounded set of hash jobs or retain many descriptors.
        let next = L + 1
        if integrityPolicy == .fullSha256,
           (manifest.files[manifestRel]?.size ?? 0) >= 64 * 1_024 * 1_024,
           streamersBox.pendingVerification.isEmpty,
           next < packedExpertsLayout.layers.count,
           !streamersBox.layerVerified[next],
           streamersBox.streamers[next] == nil {
            // Report lookahead failures only if that layer is actually used.
            streamersBox.pendingVerification[next] = try? beginLayerVerificationLocked(next)
        }
        try file.waitUntilVerified()
        let layerFD = file.descriptor
        let layerStride = packedExpertsLayout.layers[L].experts[0].size
        let streamSize = UInt64(packedExpertsLayout.expertsPerLayer) * layerStride
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: streamSize,
            expertsPerLayer: packedExpertsLayout.expertsPerLayer,
            expertStride: layerStride,
            expertOffsets: packedExpertsLayout.layers[L].experts.map(\.offset))
        streamersBox.streamers[L] = try withExtendedLifetime(file) {
            try PreadExpertStreamer(
                layout: layout,
                device: device,
                slotCount: effectiveExpertCacheSlotCount,
                cachePolicy: expertCachePolicy,
                fileDescriptor: layerFD,
                bypassFilesystemCache: Self.shouldBypassExpertFilesystemCache(
                    expertStride: layerStride, minimumSlots: config.topKExperts,
                    layers: config.numLayers, physicalMemory: ProcessInfo.processInfo.physicalMemory)
                    || Self.readsExpertsPastFileCache(variant: config.variant,
                        expertStride: layerStride,
                        expertsPerLayer: packedExpertsLayout.expertsPerLayer,
                        layers: config.numLayers, physicalMemory: ProcessInfo.processInfo.physicalMemory))
        }
        streamersBox.layerVerified[L] = true
    }

    private func beginLayerVerificationLocked(_ L: Int) throws -> VerifiedExpertFile {
        let manifestRel = "packed_experts/\(packedExpertsLayout.layers[L].file)"
        let layerFD = try modelDirectory.openFile(manifestRel)
        var transferred = false
        defer { if !transferred { close(layerFD) } }
        var expectedSHA256: String?
        if !streamersBox.layerVerified[L] {
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            let actualSize = try modelDirectory.fileSize(
                fileDescriptor: layerFD, relativePath: manifestRel)
            guard actualSize == entry.size else {
                throw ModelError.tensorSizeMismatch(
                    name: manifestRel, expected: entry.size, actual: actualSize)
            }
            switch integrityPolicy {
            case .fullSha256:
                expectedSHA256 = entry.sha256
            case .sizeCheckTrustedReceipt:
                break
            }
        }
        let file = VerifiedExpertFile(descriptor: layerFD, name: manifestRel,
                                     expectedSHA256: expectedSHA256)
        transferred = true
        return file
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        streamersQueue.sync { streamersBox.streamers.compactMap { $0 }.count }
    }

}

extension Model {

    /// Open a `.gturbo/` directory with the architecture auto-detected from
    /// `manifest.json -> arch.family` (absent means Gemma 4).
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
                            denseResidency: DenseResidencyPolicy = .resident,
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> Model {
        let baseline = try ManifestReader.resolveArchitecture(
            directoryURL: directoryURL)
        return try load(directoryURL: directoryURL,
                        device: device,
                        expecting: baseline,
                        streamingMode: streamingMode,
                        denseResidency: denseResidency,
                        expertCachePolicy: expertCachePolicy,
                        integrityPolicy: integrityPolicy,
                        loadStats: loadStats)
    }

    /// Open a `.gturbo/` directory and return a typed handle. Eagerly verifies
    /// SHA-256 of `model_weights.bin` and `packed_experts/layout.json`; layer
    /// files are verified lazily on first `routedExpert(...)` touch.
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            expecting: ArchConfig,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
                            denseResidency: DenseResidencyPolicy = .resident,
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> Model {
        var stats = ModelLoadStats()
        defer {
            loadStats?.pointee = stats
        }
        let resolvedIntegrityPolicy = integrityPolicy ?? .fullSha256
        let modelDirectory = try GTurboModelDirectory(rootURL: directoryURL)
        let manifestFD: Int32
        do { manifestFD = try modelDirectory.openFile("manifest.json") }
        catch ModelError.missingFile { throw ModelError.partialInstall(path: directoryURL.path) }
        defer { close(manifestFD) }
        let manifestData = try modelDirectory.readMetadata(
            fileDescriptor: manifestFD, relativePath: "manifest.json",
            maxBytes: ManifestReader.defaultMaxBytes)
        let manifestSize = UInt64(manifestData.count)
        let manifestShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let manifestSha = Sha256Verifier.hashData(manifestData)
        stats.manifestSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - manifestShaStart
        let receipt: VerifiedInstallReceipt?
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let receiptFD: Int32
            do {
                receiptFD = try modelDirectory.openFile(VerifiedInstallReceiptReader.fileName)
            } catch ModelError.missingFile {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(VerifiedInstallReceiptReader.fileName) is missing")
            }
            defer { close(receiptFD) }
            let receiptData = try modelDirectory.readMetadata(
                fileDescriptor: receiptFD,
                relativePath: VerifiedInstallReceiptReader.fileName,
                maxBytes: VerifiedInstallReceiptReader.defaultMaxBytes)
            let loadedReceipt = try VerifiedInstallReceiptReader.decode(data: receiptData)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                loadedReceipt,
                directoryURL: directoryURL,
                manifestSha256: manifestSha)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
            receipt = loadedReceipt
        } else {
            receipt = nil
        }

        let manifest = try ManifestReader.decode(
            data: manifestData, expecting: expecting)
        if let receipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try VerifiedInstallReceiptReader.validate(receipt,
                                                      directoryURL: directoryURL,
                                                      manifest: manifest,
                                                      manifestSha256: manifestSha,
                                                      manifestSize: manifestSize)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        // Verify the small, always-touched files before mapping model data.
        let weightsURL = directoryURL.appendingPathComponent("model_weights.bin")
        guard let weightsEntry = manifest.files["model_weights.bin"] else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        let weightsFD = try modelDirectory.openFile("model_weights.bin")
        defer { close(weightsFD) }
        let weightsSize = try modelDirectory.fileSize(
            fileDescriptor: weightsFD, relativePath: "model_weights.bin")
        guard weightsSize == weightsEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "model_weights.bin",
                expected: weightsEntry.size,
                actual: weightsSize)
        }
        let eagerShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try VerifiedFileMemo.standard.verifyFile(fileDescriptor: weightsFD,
                                                 named: "model_weights.bin",
                                                 expectedHex: weightsEntry.sha256)
        let layout: PackedExpertsLayout
        if expecting.feedForwardKind == .dense {
            layout = .dense(numLayers: expecting.numLayers)
        } else {
            guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
                throw ModelError.missingFile(name: "packed_experts/layout.json")
            }
            let layoutFD = try modelDirectory.openFile("packed_experts/layout.json")
            defer { close(layoutFD) }
            let layoutData = try modelDirectory.readMetadata(
                fileDescriptor: layoutFD, relativePath: "packed_experts/layout.json",
                maxBytes: PackedExpertsLayoutReader.defaultMaxBytes)
            guard UInt64(layoutData.count) == layoutEntry.size else {
                throw ModelError.tensorSizeMismatch(
                    name: "packed_experts/layout.json",
                    expected: layoutEntry.size,
                    actual: UInt64(layoutData.count))
            }
            guard Sha256Verifier.hashData(layoutData).lowercased()
                    == layoutEntry.sha256.lowercased() else {
                throw ModelError.checksumMismatch(file: "packed_experts/layout.json")
            }
            layout = try PackedExpertsLayoutReader.decode(data: layoutData,
                                                           manifest: manifest)
        }
        stats.eagerSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - eagerShaStart

        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt,
           expecting.feedForwardKind == .mixtureOfExperts {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try validateTrustedReceiptLayerLayout(modelDirectory: modelDirectory,
                                                  manifest: manifest,
                                                  layout: layout)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        let residentIndex = try ResidentIndexReader.load(
            fileDescriptor: weightsFD, displayPath: "model_weights.bin")
        try validateRuntimeSchema(residentIndex: residentIndex,
                                  layout: layout,
                                  manifest: manifest,
                                  config: expecting)

        // The resident index must account for the complete weights file.
        let fileSize = weightsSize
        let (expectedSize, overflow) = residentIndex.header.indexSize
            .addingReportingOverflow(residentIndex.header.residentSize)
        if overflow || fileSize != expectedSize {
            throw ModelError.indexCorrupt(detail: """
                model_weights.bin size \(fileSize) != indexSize \
                \(residentIndex.header.indexSize) + residentSize \
                \(residentIndex.header.residentSize) = \(expectedSize)
                """)
        }

        let residentBuffers = try ResidentBufferSet(
            fileURL: weightsURL,
            index: residentIndex,
            device: device,
            fileDescriptor: weightsFD,
            denseResidency: expecting.family == .gptOss ? .resident : denseResidency,
            numLayers: expecting.numLayers)

        return Model(
            device: device,
            config: expecting,
            streamingMode: streamingMode,
            expertCachePolicy: expertCachePolicy,
            integrityPolicy: resolvedIntegrityPolicy,
            residentBuffers: residentBuffers,
            residentIndex: residentIndex,
            packedExpertsLayout: layout,
            manifest: manifest,
            directoryURL: directoryURL,
            modelDirectory: modelDirectory)
    }

    private static func validateTrustedReceiptLayerLayout(modelDirectory: GTurboModelDirectory,
                                                          manifest: Manifest,
                                                          layout: PackedExpertsLayout) throws {
        for layer in layout.layers {
            if layer.experts.isEmpty { continue }
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw ModelError.trustedReceiptInvalid(detail: "manifest missing \(relativePath)")
            }
            let actualSize: UInt64
            do {
                let fd = try modelDirectory.openFile(relativePath)
                defer { close(fd) }
                actualSize = try modelDirectory.fileSize(
                    fileDescriptor: fd, relativePath: relativePath)
            }
            guard actualSize == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(manifestEntry.size)")
            }
        }
    }

    static func validateRuntimeSchema(residentIndex: ResidentIndex,
                                      layout: PackedExpertsLayout,
                                      manifest: Manifest,
                                      config: ArchConfig) throws {
        try ModelRuntimeSchemaValidator.validateRuntimeSchema(residentIndex: residentIndex,
            layout: layout, manifest: manifest, config: config)
    }

    /// Releases routed-expert streamers and their slot scratch before vision
    /// encoding, so the tower's bounded scratch does not stack on top of the
    /// expert cache on an 8 GB machine.
    ///
    /// The caller must serialize this transition against forward execution.
    public func prepareExpertResidencyForVision(
        _ policy: VisionResidencyPolicy,
        gpuDrainNanoseconds: UInt64 = 0
    ) -> VisionExpertResidencyTransition {
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        return streamersQueue.sync {
            let releasedLayerCount = streamersBox.streamers.compactMap { $0 }.count
            let releasedBytes = streamersBox.streamers.compactMap { $0 }.reduce(UInt64(0)) {
                $0 + $1.diagnosticSlotScratchBytes
            }
            if policy == .onDemand {
                for streamer in streamersBox.streamers.compactMap({ $0 }) {
                    streamersBox.retiredReadMetrics.add(streamer.readMetrics)
                }
                var released = streamersBox.streamers
                streamersBox.streamers = Array(
                    repeating: nil, count: packedExpertsLayout.numLayers)
                // `layerVerified` deliberately survives the release. Clearing it
                // would make every image turn from the second on re-verify all
                // 30 packed expert files, ~12.9 GB, inside this queue.
                released.removeAll()
            }
            let remaining = streamersBox.streamers.compactMap { $0 }
            return VisionExpertResidencyTransition(
                policy: policy,
                gpuDrainNanoseconds: gpuDrainNanoseconds,
                releasedLayerCount: policy == .onDemand ? releasedLayerCount : 0,
                releasedSlotScratchBytes: policy == .onDemand ? releasedBytes : 0,
                remainingOpenLayerCount: remaining.count,
                remainingSlotScratchBytes: remaining.reduce(UInt64(0)) {
                    $0 + $1.diagnosticSlotScratchBytes
                },
                wallNanoseconds: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
        }
    }

}
