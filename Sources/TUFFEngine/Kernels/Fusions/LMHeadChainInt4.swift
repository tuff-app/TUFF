import Metal

/// Final BF16 RMSNorm, INT4 affine lm-head projection, and greedy argmax.
/// The hot path writes one token ID without materializing vocab-sized logits.
final class LMHeadChainInt4 {
    static let rowsPerThreadgroup = 8

    private static let rowSummaryStride = 2

    /// Shape this instance compiles a specialized head pipeline for. Taken
    /// from the loaded model so both Gemma 4 (2816/262144) and Qwen 3.6
    /// (2048/248320) get constant-folded loop bounds.
    private let specializedD: UInt32
    private let specializedVocab: UInt32

    private let rms: RMSNorm
    private let rowGreedy: MTLComputePipelineState
    private let rowGreedySpecialized: MTLComputePipelineState
    private let rowReducer: MTLComputePipelineState
    private let blockRowGreedy: MTLComputePipelineState
    private let blockRowGreedySpecialized: MTLComputePipelineState
    private let blockRowReducer: MTLComputePipelineState
    private let xNormedBuffer: MTLBuffer
    private let rowSummariesBuffer: MTLBuffer
    private let blockXNormedBuffer: MTLBuffer
    private let blockRowSummariesBuffer: MTLBuffer
    private let maxD: Int
    private let maxVocab: Int
    private let maxRows: Int

    /// Affine group size of the LM head weights. Qwen3.8 Flash Next's are
    /// grouped at 32; every other checkpoint's at 64.
    let groupSize: Int

    init(context: MetalContext,
         maxD: Int = 2816,
         maxVocab: Int = 262144,
         maxRows: Int = 8,
         groupSize: Int = Quantization.groupSize) throws {
        precondition(Quantization.supportedGroupSizes.contains(groupSize),
                     "unsupported affine group size \(groupSize)")
        self.groupSize = groupSize
        // Unset at the default, so the shipping models keep their pipelines.
        let groupConstants: [MetalFunctionConstant] =
            groupSize == Quantization.groupSize
                ? []
                : [MetalFunctionConstant(
                    index: Quantization.groupSizeFunctionConstantIndex,
                    value: .uint32(UInt32(groupSize)))]
        self.rms = try RMSNorm(context: context)
        self.rowGreedy = try context.pipeline("lm_head_greedy_int4_rows_chunk_raw", constants: groupConstants)
        self.specializedD = UInt32(maxD)
        self.specializedVocab = UInt32(maxVocab)
        self.rowGreedySpecialized = try context.pipeline(
            "lm_head_greedy_int4_rows_chunk_raw",
            constants: groupConstants + [
                MetalFunctionConstant(index: 10, value: .uint32(UInt32(maxD))),
                MetalFunctionConstant(index: 11, value: .uint32(UInt32(maxVocab))),
                MetalFunctionConstant(index: 13, value: .bool(true)),
            ])
        self.rowReducer = try context.pipeline("lm_head_greedy_int4_rows_reduce")
        self.blockRowGreedy = try context.pipeline(
            "lm_head_greedy_int4_block_rows_raw", constants: groupConstants)
        self.blockRowGreedySpecialized = try context.pipeline(
            "lm_head_greedy_int4_block_rows_raw",
            constants: groupConstants + [
                MetalFunctionConstant(index: 10, value: .uint32(UInt32(maxD))),
                MetalFunctionConstant(index: 11, value: .uint32(UInt32(maxVocab))),
                MetalFunctionConstant(index: 13, value: .bool(true)),
            ])
        self.blockRowReducer = try context.pipeline(
            "lm_head_greedy_int4_block_rows_reduce")
        self.maxD = maxD
        self.maxVocab = maxVocab
        self.maxRows = maxRows

        let rowGroups = (maxVocab + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        let xLength = max(maxD, 1) * MemoryLayout<Float16>.size
        let summaryLength = rowGroups * Self.rowSummaryStride * MemoryLayout<Float>.size
        guard let xNormedBuffer = context.device.makeBuffer(
                  length: xLength,
                  options: .storageModePrivate),
              let rowSummariesBuffer = context.device.makeBuffer(
                  length: summaryLength,
                  options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.xNormedBuffer = xNormedBuffer
        self.rowSummariesBuffer = rowSummariesBuffer
        guard let blockXNormedBuffer = context.device.makeBuffer(
                  length: max(maxRows, 1) * xLength,
                  options: .storageModePrivate),
              let blockRowSummariesBuffer = context.device.makeBuffer(
                  length: max(maxRows, 1) * summaryLength,
                  options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.blockXNormedBuffer = blockXNormedBuffer
        self.blockRowSummariesBuffer = blockRowSummariesBuffer
    }

    func encodeGreedyDecode(commandBuffer: MTLCommandBuffer,
                            hidden: MTLBuffer,
                            hiddenOffset: Int = 0,
                            normedInput: MTLBuffer? = nil,
                            normWeight: MTLBuffer,
                            normOffset: Int = 0,
                            weights: MTLBuffer,
                            weightsOffset: Int = 0,
                            scales: MTLBuffer,
                            scalesOffset: Int = 0,
                            biases: MTLBuffer,
                            biasesOffset: Int = 0,
                            outToken: MTLBuffer,
                            d: UInt32,
                            vocab: UInt32,
                            rmsEps: Float = 1e-6) {
        precondition(Int(d) <= maxD, "d=\(d) exceeds wrapper maxD=\(maxD)")
        precondition(Int(vocab) <= maxVocab,
                     "vocab=\(vocab) exceeds wrapper maxVocab=\(maxVocab)")
        precondition(Int(d) % groupSize == 0,
                     "d must be a multiple of \(groupSize)")
        precondition(hiddenOffset >= 0, "hiddenOffset must be non-negative")
        precondition(weightsOffset % 2 == 0,
                     "lm_head_greedy_int4_rows_chunk_raw needs a 2-aligned weightsOffset")

        let rowGroups = (Int(vocab) + Self.rowsPerThreadgroup - 1)
            / Self.rowsPerThreadgroup
        // A caller that already normalized the row (Flint, from its FP32
        // residual) passes it in and the kernel reads it directly.
        if normedInput == nil {
            rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: hidden,
                            xOffset: hiddenOffset,
                            weight: normWeight,
                            weightOffset: normOffset,
                            out: xNormedBuffer,
                            d: d,
                            eps: rmsEps)
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            let specialized = d == specializedD && vocab == specializedVocab
            encoder.setComputePipelineState(specialized ? rowGreedySpecialized : rowGreedy)
            encoder.setBuffer(normedInput ?? xNormedBuffer, offset: 0, index: 0)
            encoder.setBuffer(weights, offset: weightsOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            encoder.setBuffer(biases, offset: biasesOffset, index: 3)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 4)
            var dValue = d
            var vocabValue = vocab
            encoder.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)

            let threadgroupSize = MTLSize(
                width: 32 * Self.rowsPerThreadgroup,
                height: 1,
                depth: 1)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowGroups, height: 1, depth: 1),
                threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(rowReducer)
            encoder.setBuffer(rowSummariesBuffer, offset: 0, index: 0)
            encoder.setBuffer(outToken, offset: 0, index: 1)
            var rowGroupCount = UInt32(rowGroups)
            encoder.setBytes(&rowGroupCount, length: MemoryLayout<UInt32>.size, index: 2)

            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            encoder.dispatchThreads(threadgroupSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }

    /// Fused final norm plus one argmax per hidden row. The projections are
    /// ordered in one command buffer and the only shared result is one token ID
    /// per row. This is the affine-target verifier's bounded N+1 prediction
    /// representation; it does not materialize block-sized vocab logits.
    func encodeGreedyDecodeRows(commandBuffer: MTLCommandBuffer,
                                hidden: MTLBuffer,
                                hiddenOffset: Int = 0,
                                hiddenStride: Int,
                                rowCount: Int,
                                normWeight: MTLBuffer,
                                normOffset: Int = 0,
                                weights: MTLBuffer,
                                weightsOffset: Int = 0,
                                scales: MTLBuffer,
                                scalesOffset: Int = 0,
                                biases: MTLBuffer,
                                biasesOffset: Int = 0,
                                outTokens: MTLBuffer,
                                d: UInt32,
                                vocab: UInt32,
                                rmsEps: Float = 1e-6) {
        precondition(rowCount > 0 && rowCount <= maxRows,
                     "rowCount exceeds speculative head capacity")
        precondition(Int(d) <= maxD, "d exceeds wrapper maxD")
        precondition(Int(vocab) <= maxVocab, "vocab exceeds wrapper maxVocab")
        precondition(hiddenOffset >= 0 && hiddenStride >= Int(d))
        precondition(Int(d) % groupSize == 0)
        precondition(weightsOffset % 2 == 0)

        for row in 0..<rowCount {
            rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: hidden,
                            xOffset: hiddenOffset
                                + row * hiddenStride * MemoryLayout<Float16>.stride,
                            weight: normWeight,
                            weightOffset: normOffset,
                            out: blockXNormedBuffer,
                            outOffset: row * maxD * MemoryLayout<Float16>.stride,
                            d: d,
                            eps: rmsEps)
        }

        let rowGroups = (Int(vocab) + Self.rowsPerThreadgroup - 1)
            / Self.rowsPerThreadgroup
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            let specialized = d == specializedD && vocab == specializedVocab
            encoder.setComputePipelineState(
                specialized ? blockRowGreedySpecialized : blockRowGreedy)
            encoder.setBuffer(blockXNormedBuffer, offset: 0, index: 0)
            encoder.setBuffer(weights, offset: weightsOffset, index: 1)
            encoder.setBuffer(scales, offset: scalesOffset, index: 2)
            encoder.setBuffer(biases, offset: biasesOffset, index: 3)
            encoder.setBuffer(blockRowSummariesBuffer, offset: 0, index: 4)
            var dValue = d
            var vocabValue = vocab
            var rowCountValue = UInt32(rowCount)
            encoder.setBytes(&dValue, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&vocabValue, length: MemoryLayout<UInt32>.size, index: 6)
            encoder.setBytes(&rowCountValue,
                             length: MemoryLayout<UInt32>.size, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowGroups, height: rowCount, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: 32 * Self.rowsPerThreadgroup, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(blockRowReducer)
            encoder.setBuffer(blockRowSummariesBuffer, offset: 0, index: 0)
            encoder.setBuffer(outTokens, offset: 0, index: 1)
            var rowGroupCount = UInt32(rowGroups)
            encoder.setBytes(&rowGroupCount, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.dispatchThreadgroups(
                MTLSize(width: rowCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }
}
