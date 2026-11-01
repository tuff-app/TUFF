import Metal

/// Routed experts for a prefill chunk. A tile binds up to 16 streamed experts;
/// each (token, expert) pair is one row computation that uses the decode
/// kernels' helpers (FP32 SwiGLU intermediates, per-layer expert bit widths),
/// so a pair's numerics match a decode token. Partial outputs are reduced per
/// token in rank order, the decode accumulation order.
final class FlintBatchedExperts {
    private struct TileParams {
        var pairStart: UInt32
        var pairCount: UInt32
        var d: UInt32
        var f: UInt32
        var topK: UInt32
        var live: UInt32
        var localExperts: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                           UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    }

    private let phase1: MTLComputePipelineState
    private let phase2: MTLComputePipelineState
    private let reduce: MTLComputePipelineState

    init(context: MetalContext, siluActivation: Bool,
         groupSize: Int = Quantization.groupSize) throws {
        var constants: [MetalFunctionConstant] = []
        if siluActivation { constants.append(MetalFunctionConstant(index: 4, value: .bool(true))) }
        if groupSize != Quantization.groupSize {
            constants.append(MetalFunctionConstant(
                index: Quantization.groupSizeFunctionConstantIndex,
                value: .uint32(UInt32(groupSize))))
        }
        phase1 = try context.pipeline("flint_moe_pairs_phase1", constants: constants)
        phase2 = try context.pipeline("flint_moe_pairs_phase2", constants: constants)
        reduce = try context.pipeline("flint_moe_reduce_rows")
    }

    /// Bytes of FP32 SwiGLU intermediates a tile of `pairs` pairs needs.
    static func activationBytes(pairs: Int, f: Int) -> Int { pairs * f * MemoryLayout<Float>.stride }

    /// Bytes of FP32 per-(token, rank) down-projection partials for `rows` rows.
    static func partialBytes(rows: Int, topK: Int, d: Int) -> Int {
        rows * topK * d * MemoryLayout<Float>.stride
    }

    func encodeTile(commandBuffer: MTLCommandBuffer,
                    argumentBuffer: MTLBuffer, views: [TensorView], expertIDs: [Int],
                    offsets: MoEExpertOffsets, widths: MoEExpertBitWidths,
                    x: MTLBuffer, sortedPairs: MTLBuffer,
                    pairStart: Int, pairCount: Int,
                    d: Int, f: Int, topK: Int,
                    acts: MTLBuffer, partials: MTLBuffer) {
        precondition(pairCount > 0 && expertIDs.count == views.count && expertIDs.count <= 16)
        precondition(acts.length >= Self.activationBytes(pairs: pairCount, f: f),
                     "FP32 tile intermediates exceed scratch")
        var ids = [UInt32](repeating: UInt32.max, count: 16)
        for (index, expert) in expertIDs.enumerated() { ids[index] = UInt32(expert) }
        var params = TileParams(
            pairStart: UInt32(pairStart), pairCount: UInt32(pairCount),
            d: UInt32(d), f: UInt32(f), topK: UInt32(topK), live: UInt32(expertIDs.count),
            localExperts: (ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
                           ids[8], ids[9], ids[10], ids[11], ids[12], ids[13], ids[14], ids[15]))
        var offsets = offsets
        var widths = widths
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(phase1)
        encoder.setBuffer(argumentBuffer, offset: 0, index: 0)
        for view in views { encoder.useResource(view.buffer, usage: .read) }
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBuffer(sortedPairs, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<TileParams>.stride, index: 5)
        encoder.setBytes(&widths, length: MemoryLayout<MoEExpertBitWidths>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (pairCount * f + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.setComputePipelineState(phase2)
        encoder.setBuffer(argumentBuffer, offset: 0, index: 0)
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(sortedPairs, offset: 0, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<TileParams>.stride, index: 4)
        encoder.setBuffer(partials, offset: 0, index: 5)
        encoder.setBytes(&widths, length: MemoryLayout<MoEExpertBitWidths>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (pairCount * d + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// `y[row] = half(sharedOut[row] + sum over ranks of partials[row][rank])`.
    func encodeReduce(commandBuffer: MTLCommandBuffer, partials: MTLBuffer,
                      sharedOut: MTLBuffer, y: MTLBuffer, rows: Int, d: Int, topK: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(reduce)
        encoder.setBuffer(partials, offset: 0, index: 0)
        encoder.setBuffer(sharedOut, offset: 0, index: 1)
        encoder.setBuffer(y, offset: 0, index: 2)
        var dd = UInt32(d), k = UInt32(topK)
        encoder.setBytes(&dd, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreads(MTLSize(width: d, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(d, 256), height: 1, depth: 1))
        encoder.endEncoding()
    }
}
