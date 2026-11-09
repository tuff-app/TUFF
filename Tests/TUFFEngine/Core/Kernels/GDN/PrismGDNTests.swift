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

    /// The chunk kernels (10.4) against the per-token ones they replaced, row
    /// by row on the same inputs: outputs, final state and conv tail must be
    /// bit-identical, because that is what keeps Bonsai's answers unchanged.
    /// Small shapes at a few lengths, then Bonsai 2B's GDN shape (16 key and
    /// value heads of 128) and a 27B-style one with more value heads, at a
    /// full 300-row chunk.
    @Test(arguments: [(1, 2, 4, 64), (5, 2, 4, 64), (37, 2, 4, 64),
                      (300, 16, 16, 128), (300, 16, 48, 128)])
    func chunkKernelsMatchRowByRowDecode(shape: (rows: Int, kHeads: Int, vHeads: Int, dim: Int)) throws {
        let rows = shape.rows
        let config = LinearAttentionConfig(numKHeads: shape.kHeads, numVHeads: shape.vHeads,
                                           keyHeadDim: shape.dim, valueHeadDim: shape.dim,
                                           convKernelSize: 4)
        let context = try MetalContext()
        let standard = try GDN(context: context, config: config)
        let prism = try PrismGDN(context: context, config: config)
        var generator = SystemRandomNumberGenerator()
        func random(_ count: Int, _ scale: Float = 1) -> [Float] {
            (0..<count).map { _ in Float.random(in: -scale...scale, using: &generator) }
        }
        func half(_ values: [Float]) throws -> MTLBuffer {
            try #require(Fp16Buffer.make(context.device, halves: values.map { Float16($0) }))
        }
        func float(_ values: [Float]) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared))
        }
        func copy(_ buffer: MTLBuffer) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(bytes: buffer.contents(), length: buffer.length,
                                                   options: .storageModeShared))
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
        func bytes(_ buffer: MTLBuffer, _ length: Int) -> [UInt8] {
            Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt8.self),
                                      count: length))
        }
        let C = config.qkvDim, Hv = config.numVHeads, V = config.valueDim
        let qkv = try half(random(rows * C))
        let z = try half(random(rows * V))
        let conv = view(try float(random(C * 4, 0.5)), count: C * 4)
        let a = try float(random(rows * Hv)), b = try float(random(rows * Hv))
        let aLog = view(try float(random(Hv, 0.5)), count: Hv)
        let bias = view(try float(random(Hv, 0.5)), count: Hv)
        let norm = view(try float(random(config.valueHeadDim, 1)), count: config.valueHeadDim)
        let stateBytes = Hv * config.valueHeadDim * config.keyHeadDim * 4
        let tailStart = try half(random(3 * C)), stateStart = try float(random(stateBytes / 4, 0.1))

        // Row by row, as through 10.3.
        let tail1 = try copy(tailStart), state1 = try copy(stateStart)
        let conv1 = try zero(rows * C * 2), y1 = try zero(rows * V * 2), out1 = try zero(rows * V * 2)
        let cb1 = try #require(context.queue.makeCommandBuffer())
        for row in 0..<rows {
            prism.encodeConv(commandBuffer: cb1, tail: tail1, qkv: qkv, weight: conv, output: conv1,
                             qkvOffset: row * C * 2, outputOffset: row * C * 2)
            standard.encodeQKNorm(commandBuffer: cb1, convOut: conv1, convOutOffset: row * C * 2)
            prism.encodeDelta(commandBuffer: cb1, convOut: conv1, a: a, b: b, aLog: aLog, dtBias: bias,
                              state: state1, output: y1, convOutOffset: row * C * 2,
                              aOffset: row * Hv * 4, bOffset: row * Hv * 4, outputOffset: row * V * 2)
            prism.encodeGatedNorm(commandBuffer: cb1, y: y1, z: z, weight: norm, output: out1,
                                  yOffset: row * V * 2, zOffset: row * V * 2, outputOffset: row * V * 2)
        }
        cb1.commit(); cb1.waitUntilCompleted()
        try checkCommandBufferError(cb1)

        // The whole chunk at once.
        let tail2 = try copy(tailStart), state2 = try copy(stateStart)
        let conv2 = try zero(rows * C * 2), y2 = try zero(rows * V * 2), out2 = try zero(rows * V * 2)
        let cb2 = try #require(context.queue.makeCommandBuffer())
        prism.encodeConvPrefill(commandBuffer: cb2, tail: tail2, qkv: qkv, weight: conv, output: conv2, rows: rows)
        standard.encodeQKNorm(commandBuffer: cb2, convOut: conv2, rows: rows)
        prism.encodeDeltaPrefill(commandBuffer: cb2, convOut: conv2, a: a, b: b, aLog: aLog, dtBias: bias,
                                 state: state2, output: y2, rows: rows)
        prism.encodeGatedNormPrefill(commandBuffer: cb2, y: y2, z: z, weight: norm, output: out2, rows: rows)
        cb2.commit(); cb2.waitUntilCompleted()
        try checkCommandBufferError(cb2)

        #expect(bytes(conv1, rows * C * 2) == bytes(conv2, rows * C * 2), "conv")
        #expect(bytes(out1, rows * V * 2) == bytes(out2, rows * V * 2), "output")
        #expect(bytes(state1, stateBytes) == bytes(state2, stateBytes), "state")
        #expect(bytes(tail1, 3 * C * 2) == bytes(tail2, 3 * C * 2), "tail")
    }

    /// The chunked alpha/beta projection against one dispatch per token.
    @Test func floatRowProjectionMatchesPerTokenProjection() throws {
        let context = try MetalContext()
        let projection = try PrismFloatProjection(context: context)
        let columns = 2048, outputs = 16, rows = 40
        var generator = SystemRandomNumberGenerator()
        let input = try #require(Fp16Buffer.make(context.device, halves: (0..<(rows * columns)).map { _ in
            Float16(Float.random(in: -1...1, using: &generator)) }))
        let weights = (0..<(outputs * columns)).map { _ in Float.random(in: -0.1...0.1, using: &generator) }
        let weightBuffer = try #require(context.device.makeBuffer(bytes: weights, length: weights.count * 4,
                                                                  options: .storageModeShared))
        let view = TensorView(buffer: weightBuffer, offset: 0, length: UInt64(weights.count * 4),
                              scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                              shape: (UInt32(outputs), UInt32(columns), 0, 0), dtype: 3)
        let one = try #require(context.device.makeBuffer(length: rows * outputs * 4, options: .storageModeShared))
        let many = try #require(context.device.makeBuffer(length: rows * outputs * 4, options: .storageModeShared))
        let cb = try #require(context.queue.makeCommandBuffer())
        for row in 0..<rows {
            projection.encode(commandBuffer: cb, input: input, weight: view, output: one,
                              columns: columns, outputs: outputs,
                              inputOffset: row * columns * 2, outputOffset: row * outputs * 4)
        }
        projection.encodeRows(commandBuffer: cb, input: input, weight: view, output: many,
                              columns: columns, outputs: outputs, rows: rows, inputStride: columns)
        cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        #expect(memcmp(one.contents(), many.contents(), rows * outputs * 4) == 0)
    }
}
