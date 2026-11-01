import Metal

/// Flint's BF16 router uses independent sigmoid probabilities, normalized only
/// after top-k. The two kernels keep score calculation and selection in FP32.
final class FlintRouter {
    private let projection: MTLComputePipelineState
    private let floatProjection: MTLComputePipelineState
    private let int8FloatProjection: MTLComputePipelineState
    private let selection: MTLComputePipelineState
    private let logits: MTLBuffer

    /// `maximumRows` sizes the score buffer for prefill chunks; decode uses one.
    init(context: MetalContext, maximumExperts: Int, maximumRows: Int = 1) throws {
        precondition(maximumExperts > 0 && maximumRows > 0)
        projection = try context.pipeline("flint_router_bf16")
        floatProjection = try context.pipeline("flint_router_bf16_float")
        int8FloatProjection = try context.pipeline("flint_router_int8_float")
        selection = try context.pipeline("flint_router_select")
        guard let buffer = context.device.makeBuffer(length: maximumRows * maximumExperts * MemoryLayout<Float>.stride,
                                                      options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        logits = buffer
    }

    func encode(commandBuffer: MTLCommandBuffer, weight: MTLBuffer, weightOffset: Int = 0,
                hidden: MTLBuffer, hiddenOffset: Int = 0, indices: MTLBuffer,
                weights: MTLBuffer, hiddenSize: Int, experts: Int, topK: Int,
                floatInput: Bool = false, affineScalesOffset: Int? = nil,
                affineBiasesOffset: Int? = nil, rows: Int = 1) {
        precondition(hiddenSize > 0 && experts > 0 && rows > 0
                     && (1...min(8, experts)).contains(topK))
        precondition(rows == 1 || floatInput, "multi-row routing needs FP32 input")
        precondition(logits.length >= rows * experts * MemoryLayout<Float>.stride)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        if affineScalesOffset != nil {
            precondition(floatInput && affineBiasesOffset != nil && hiddenSize.isMultiple(of: 64))
            encoder.setComputePipelineState(int8FloatProjection)
            encoder.setBuffer(weight, offset: affineScalesOffset!, index: 5)
            encoder.setBuffer(weight, offset: affineBiasesOffset!, index: 6)
        } else {
            precondition(affineBiasesOffset == nil)
            encoder.setComputePipelineState(floatInput ? floatProjection : projection)
        }
        encoder.setBuffer(weight, offset: weightOffset, index: 0)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 1)
        encoder.setBuffer(logits, offset: 0, index: 2)
        var d = UInt32(hiddenSize), count = UInt32(experts), k = UInt32(topK)
        encoder.setBytes(&d, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: experts, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(selection)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(indices, offset: 0, index: 1)
        encoder.setBuffer(weights, offset: 0, index: 2)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
