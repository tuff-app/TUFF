import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

@Suite struct PrismGDNTests {
    @Test func floatWeightsMatchBFloatReferenceWhenExactlyRepresentable() throws {
        let config = LinearAttentionConfig(numKHeads: 1, numVHeads: 1, keyHeadDim: 32,
                                           valueHeadDim: 32, convKernelSize: 4)
        let context = try MetalContext()
        let standard = try GDN(context: context, config: config)
        let prism = try PrismGDN(context: context, config: config)
        func half(_ values: [Float]) throws -> MTLBuffer {
            try #require(Fp16Buffer.make(context.device, halves: values.map { Float16($0) }))
        }
        func float(_ values: [Float]) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared))
        }
        func bf16(_ values: [Float]) throws -> MTLBuffer {
            let bits = values.map(Quantization.bf16Bits)
            return try #require(context.device.makeBuffer(bytes: bits, length: bits.count * 2, options: .storageModeShared))
        }
        func zero(_ length: Int) throws -> MTLBuffer {
            let buffer = try #require(context.device.makeBuffer(length: length, options: .storageModeShared))
            buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: length)
            return buffer
        }
        func view(_ buffer: MTLBuffer, count: Int) -> TensorView {
            TensorView(buffer: buffer, offset: 0, length: UInt64(count * 4),
                       scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                       shape: (UInt32(count), 0, 0, 0), dtype: 3)
        }
        let channels = config.qkvDim, width = config.valueHeadDim
        let qkv = try half((0..<channels).map { Float($0 % 13 - 6) / 8 })
        let z = try half((0..<width).map { Float($0 % 7 - 3) / 4 })
        let convValues = (0..<(channels * 4)).map { Float($0 % 9 - 4) / 16 }
        let convF32 = try float(convValues), convBF16 = try bf16(convValues)
        let normValues = (0..<width).map { 0.5 + Float($0 % 8) / 16 }
        let normF32 = try float(normValues), normBF16 = try bf16(normValues)
        let aF32 = try float([0.25]), bF32 = try float([-0.5])
        let aHalf = try half([0.25]), bHalf = try half([-0.5])
        let aLogF32 = try float([0.5]), biasF32 = try float([-0.25])
        let aLogBF16 = try bf16([0.5]), biasBF16 = try bf16([-0.25])
        let tail1 = try zero(channels * 3 * 2), tail2 = try zero(channels * 3 * 2)
        let conv1 = try zero(channels * 2), conv2 = try zero(channels * 2)
        let state1 = try zero(width * 32 * 4), state2 = try zero(width * 32 * 4)
        let y1 = try zero(width * 2), y2 = try zero(width * 2)
        let out1 = try zero(width * 2), out2 = try zero(width * 2)
        for _ in 0..<2 {
            let cb = try #require(context.queue.makeCommandBuffer())
            standard.encodeConvDecode(commandBuffer: cb, tail: tail1, qkv: qkv,
                                      convWeight: convBF16, convWeightOffset: 0, out: conv1)
            prism.encodeConv(commandBuffer: cb, tail: tail2, qkv: qkv,
                             weight: view(convF32, count: convValues.count), output: conv2)
            standard.encodeQKNorm(commandBuffer: cb, convOut: conv1)
            standard.encodeQKNorm(commandBuffer: cb, convOut: conv2)
            standard.encodeDeltaStepDecode(commandBuffer: cb, convOut: conv1, aProj: aHalf,
                bProj: bHalf, aLog: aLogBF16, aLogOffset: 0, dtBias: biasBF16, dtBiasOffset: 0,
                state: state1, y: y1)
            prism.encodeDelta(commandBuffer: cb, convOut: conv2, a: aF32, b: bF32,
                              aLog: view(aLogF32, count: 1), dtBias: view(biasF32, count: 1),
                              state: state2, output: y2)
            standard.encodeGatedNorm(commandBuffer: cb, y: y1, z: z, weight: normBF16, weightOffset: 0, out: out1)
            prism.encodeGatedNorm(commandBuffer: cb, y: y2, z: z,
                                  weight: view(normF32, count: width), output: out2)
            cb.commit(); cb.waitUntilCompleted()
            try checkCommandBufferError(cb)
            #expect(Fp16Buffer.read(out1, count: width) == Fp16Buffer.read(out2, count: width))
        }
    }
}
