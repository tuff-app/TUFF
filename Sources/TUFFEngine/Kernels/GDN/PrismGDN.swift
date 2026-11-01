import Metal

/// Decode variants for Prism's FP32 convolution, recurrence scalars and
/// gated-normalization weights. Q/K normalization and state storage are shared
/// with the regular GDN implementation.
final class PrismGDN {
    private let config: LinearAttentionConfig
    private let conv: MTLComputePipelineState
    private let delta: MTLComputePipelineState
    private let gatedNorm: MTLComputePipelineState

    init(context: MetalContext, config: LinearAttentionConfig) throws {
        self.config = config
        conv = try context.pipeline("prism_gdn_conv_mix_decode")
        delta = try context.pipeline("prism_gdn_delta_step_decode")
        gatedNorm = try context.pipeline("prism_gdn_gated_norm")
    }

    func encodeConv(commandBuffer: MTLCommandBuffer, tail: MTLBuffer, qkv: MTLBuffer,
                    weight: TensorView, output: MTLBuffer,
                    qkvOffset: Int = 0, outputOffset: Int = 0) {
        precondition(weight.dtype == 3)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(conv)
        encoder.setBuffer(tail, offset: 0, index: 0)
        encoder.setBuffer(qkv, offset: qkvOffset, index: 1)
        encoder.setBuffer(weight.buffer, offset: Int(weight.offset), index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        var channels = UInt32(config.qkvDim), taps = UInt32(config.convKernelSize)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 5)
        let threads = min(256, conv.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: config.qkvDim, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeDelta(commandBuffer: MTLCommandBuffer, convOut: MTLBuffer,
                     a: MTLBuffer, b: MTLBuffer, aLog: TensorView, dtBias: TensorView,
                     state: MTLBuffer, output: MTLBuffer, convOutOffset: Int = 0,
                     aOffset: Int = 0, bOffset: Int = 0, outputOffset: Int = 0) {
        precondition(aLog.dtype == 3 && dtBias.dtype == 3)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(delta)
        encoder.setBuffer(convOut, offset: convOutOffset, index: 0)
        encoder.setBuffer(a, offset: aOffset, index: 1)
        encoder.setBuffer(b, offset: bOffset, index: 2)
        encoder.setBuffer(aLog.buffer, offset: Int(aLog.offset), index: 3)
        encoder.setBuffer(dtBias.buffer, offset: Int(dtBias.offset), index: 4)
        encoder.setBuffer(state, offset: 0, index: 5)
        encoder.setBuffer(output, offset: outputOffset, index: 6)
        var heads = UInt32(config.numKHeads), valueHeads = UInt32(config.numVHeads)
        var keyDim = UInt32(config.keyHeadDim), valueDim = UInt32(config.valueHeadDim)
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&valueHeads, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.size, index: 10)
        encoder.dispatchThreadgroups(MTLSize(width: config.numVHeads, height: config.valueHeadDim / 4, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        encoder.endEncoding()
    }

    func encodeGatedNorm(commandBuffer: MTLCommandBuffer, y: MTLBuffer, z: MTLBuffer,
                         weight: TensorView, output: MTLBuffer, yOffset: Int = 0,
                         zOffset: Int = 0, outputOffset: Int = 0) {
        precondition(weight.dtype == 3)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(gatedNorm)
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(z, offset: zOffset, index: 1)
        encoder.setBuffer(weight.buffer, offset: Int(weight.offset), index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        var heads = UInt32(config.numVHeads), width = UInt32(config.valueHeadDim), gate: UInt32 = 0
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&gate, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: config.numVHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
