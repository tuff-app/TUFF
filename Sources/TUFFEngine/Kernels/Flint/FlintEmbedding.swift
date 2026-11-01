import Metal

/// Flint token embeddings are group-64 INT8, separate from its INT4 LM head.
final class FlintEmbedding {
    private let pipeline: MTLComputePipelineState
    private let rowsPipeline: MTLComputePipelineState
    init(context: MetalContext) throws {
        pipeline = try context.pipeline("flint_embed_int8")
        rowsPipeline = try context.pipeline("flint_embed_int8_rows")
    }

    /// Embed `rows` tokens into `residual[row * width + i]` (FP32 only).
    func encodeRows(commandBuffer: MTLCommandBuffer, table: TensorView,
                    tokens: MTLBuffer, residual: MTLBuffer, rows: Int, width: UInt32) {
        precondition(width > 0 && width.isMultiple(of: 64) && rows > 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(rowsPipeline)
        encoder.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        encoder.setBuffer(table.buffer, offset: Int(table.scaleOffset), index: 1)
        encoder.setBuffer(table.buffer, offset: Int(table.biasOffset), index: 2)
        encoder.setBuffer(tokens, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        var w = width
        encoder.setBytes(&w, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.dispatchThreads(MTLSize(width: Int(width), height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(width), 256), height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encode(commandBuffer: MTLCommandBuffer, table: TensorView,
                output: MTLBuffer, residual: MTLBuffer, token: UInt32, width: UInt32) {
        precondition(width > 0 && width.isMultiple(of: 64))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        encoder.setBuffer(table.buffer, offset: Int(table.scaleOffset), index: 1)
        encoder.setBuffer(table.buffer, offset: Int(table.biasOffset), index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 6)
        var token = token, width = width
        encoder.setBytes(&token, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.dispatchThreads(MTLSize(width: Int(width), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(width), pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }
}

/// FP32 residual stream and its RMS-normalized views. One row for decode;
/// `rows` rows for a prefill chunk (which skips the half mirror).
final class FlintResidual {
    private let addPipeline: MTLComputePipelineState
    private let addNormPipeline: MTLComputePipelineState
    private let normPipeline: MTLComputePipelineState
    let values: MTLBuffer
    let normalized: MTLBuffer
    private let width: Int
    private let rows: Int
    private let writesMirror: Bool

    init(context: MetalContext, width: Int, rows: Int = 1, writesMirror: Bool = true) throws {
        let mirror = [MetalFunctionConstant(index: 120, value: .bool(writesMirror))]
        addPipeline = try context.pipeline("flint_residual_add", constants: mirror)
        addNormPipeline = try context.pipeline("flint_residual_add_norm", constants: mirror)
        normPipeline = try context.pipeline("flint_residual_norm")
        guard let values = context.device.makeBuffer(length: rows * width * MemoryLayout<Float>.stride, options: .storageModeShared),
              let normalized = context.device.makeBuffer(length: rows * width * MemoryLayout<Float>.stride, options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        self.values = values
        self.normalized = normalized
        self.width = width
        self.rows = rows
        self.writesMirror = writesMirror
    }

    /// `residual += delta` over `count` contiguous elements.
    func add(_ cb: MTLCommandBuffer, delta: MTLBuffer, mirror: MTLBuffer, count: Int) {
        guard let encoder = cb.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(addPipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(delta, offset: 0, index: 1)
        encoder.setBuffer(mirror, offset: 0, index: 2)
        var n = UInt32(count)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(count, addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// `residual += delta` then the RMS norm, in one dispatch (one threadgroup
    /// per row). Identical results to `add` followed by `norm`.
    func addNorm(_ cb: MTLCommandBuffer, delta: MTLBuffer, mirror: MTLBuffer,
                 weight: MTLBuffer, weightOffset: Int, output: MTLBuffer,
                 count: Int, eps: Float, rows: Int = 1) {
        precondition(rows <= self.rows && count == width)
        guard let encoder = cb.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(addNormPipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(delta, offset: 0, index: 1)
        encoder.setBuffer(mirror, offset: 0, index: 2)
        encoder.setBuffer(weight, offset: weightOffset, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBuffer(normalized, offset: 0, index: 5)
        var n = UInt32(count), epsilon = eps
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&epsilon, length: MemoryLayout<Float>.stride, index: 7)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func norm(_ cb: MTLCommandBuffer, weight: MTLBuffer, weightOffset: Int,
              output: MTLBuffer, count: Int, eps: Float, rows: Int = 1) {
        precondition(rows <= self.rows && count == width)
        guard let encoder = cb.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(normPipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: weightOffset, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBuffer(normalized, offset: 0, index: 3)
        var n = UInt32(count), epsilon = eps
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&epsilon, length: MemoryLayout<Float>.stride, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
