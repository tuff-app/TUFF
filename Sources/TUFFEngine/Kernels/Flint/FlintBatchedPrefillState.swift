import Metal

/// Scratch and kernels for one Flint prefill chunk. Rows are laid out
/// contiguously: FP32 residual rows, half normalized rows, per-row query and
/// attention-context rows, and per-(token, rank) FP32 down-projection partials.
final class FlintBatchedPrefillState {
    let rows: Int
    let residual: FlintResidual
    let router: FlintRouter
    let experts: FlintBatchedExperts
    /// Row stride, in elements, of the query and attention-context rows.
    let attentionStride: Int
    /// Row stride, in elements, of the shared-expert intermediates.
    let sharedStride: Int
    let tokens: MTLBuffer
    let normed: MTLBuffer
    let routedX: MTLBuffer
    let queries: MTLBuffer
    let context: MTLBuffer
    let attentionOut: MTLBuffer
    let denseX: MTLBuffer
    let sharedOut: MTLBuffer
    let routedOut: MTLBuffer
    let sharedGate: MTLBuffer
    let sharedUp: MTLBuffer
    let sharedAct: MTLBuffer
    let routeIDs: MTLBuffer
    let routeWeights: MTLBuffer
    /// Two FP32 intermediate scratches so a tile can be encoded while the
    /// previous one is still running.
    let acts: [MTLBuffer]
    let partials: MTLBuffer

    init(context ctx: MetalContext, config cfg: ArchConfig, rows: Int, silu: Bool) throws {
        precondition(rows > 0)
        let D = cfg.hiddenSize
        let half = MemoryLayout<Float16>.stride
        func make(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = ctx.device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "flint.prefill.\(label)"
            return buffer
        }
        self.rows = rows
        residual = try FlintResidual(context: ctx, width: D, rows: rows, writesMirror: false)
        router = try FlintRouter(context: ctx, maximumExperts: cfg.numExperts, maximumRows: rows)
        experts = try FlintBatchedExperts(context: ctx, siluActivation: silu,
                                          groupSize: cfg.int4GroupSize)
        attentionStride = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)
        sharedStride = (0..<cfg.numLayers).map { cfg.ffnIntermediateSize(layer: $0) }.max() ?? 0
        tokens = try make(rows * MemoryLayout<UInt32>.stride, "tokens")
        normed = try make(rows * D * half, "normed")
        routedX = try make(rows * D * half, "routedX")
        queries = try make(rows * attentionStride * half, "queries")
        context = try make(rows * attentionStride * half, "context")
        attentionOut = try make(rows * D * half, "attentionOut")
        denseX = try make(rows * D * half, "denseX")
        sharedOut = try make(rows * D * half, "sharedOut")
        routedOut = try make(rows * D * half, "routedOut")
        sharedGate = try make(rows * sharedStride * half, "sharedGate")
        sharedUp = try make(rows * sharedStride * half, "sharedUp")
        sharedAct = try make(rows * sharedStride * half, "sharedAct")
        routeIDs = try make(rows * cfg.topKExperts * MemoryLayout<UInt32>.stride, "routeIDs")
        routeWeights = try make(rows * cfg.topKExperts * half, "routeWeights")
        let tileActivations = FlintBatchedExperts.activationBytes(pairs: 16 * rows, f: cfg.moeIntermediateSize)
        acts = [try make(tileActivations, "actsA"), try make(tileActivations, "actsB")]
        partials = try make(FlintBatchedExperts.partialBytes(rows: rows, topK: cfg.topKExperts, d: D), "partials")
    }
}
