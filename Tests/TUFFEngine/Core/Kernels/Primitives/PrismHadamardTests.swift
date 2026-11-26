import Foundation
import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

@Suite struct PrismHadamardTests {
    @Test func signedForwardAndInverseMatchWalshMatrix() throws {
        let block = 512
        let width = block * 2
        let rows = 2
        let signs: [Float] = (0..<width).map { $0 % 3 == 0 ? -1 : 1 }
        let input: [Float16] = (0..<(width * rows)).map {
            Float16(Float(($0 * 17) % 53 - 26) / 32)
        }
        let context = try MetalContext()
        let kernel = try PrismHadamard(context: context)
        let x = try #require(Fp16Buffer.make(context.device, halves: input))
        let y = try #require(Fp16Buffer.make(context.device, count: input.count))
        let restored = try #require(Fp16Buffer.make(context.device, count: input.count))
        let signBuffer = try #require(context.device.makeBuffer(
            bytes: signs, length: signs.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let cb = try #require(context.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: cb, input: x, signs: signBuffer, output: y,
                      width: width, rows: rows, block: block, inverse: false)
        kernel.encode(commandBuffer: cb, input: y, signs: signBuffer, output: restored,
                      width: width, rows: rows, block: block, inverse: true)
        cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb)

        let actual = Fp16Buffer.read(y, count: input.count)
        for row in 0..<rows {
            for part in 0..<(width / block) {
                let base = row * width + part * block
                for output in 0..<block {
                    var sum: Float = 0
                    for index in 0..<block {
                        let parity: Float = ((output & index).nonzeroBitCount & 1) == 0 ? 1 : -1
                        sum += Float(input[base + index]) * signs[part * block + index] * parity
                    }
                    let expected = sum / sqrt(Float(block))
                    #expect(abs(actual[base + output] - expected) < 0.003)
                }
            }
        }
        let roundTrip = Fp16Buffer.read(restored, count: input.count)
        for i in input.indices { #expect(abs(roundTrip[i] - Float(input[i])) < 0.002) }
    }

    @Test func affine2ProjectionUsesLowFirstCodesAndPerGroupBias() throws {
        let columns = 256, outputs = 3, rows = 2
        let input: [Float16] = (0..<(columns * rows)).map { Float16(Float(($0 % 11) - 5) / 8) }
        let codes = (0..<(columns * outputs)).map { UInt32(($0 * 7 + $0 / 9) % 4) }
        var words = [UInt32](repeating: 0, count: codes.count / 16)
        for i in codes.indices { words[i / 16] |= codes[i] << (2 * (i % 16)) }
        let scales: [Float16] = [0.5, 0.25, 1, 0.125, 0.75, 0.0625]
        let biases: [Float16] = [-0.5, -0.25, -1, -0.125, -0.75, -0.0625]
        let context = try MetalContext()
        let kernel = try PrismAffine2(context: context)
        let x = try #require(Fp16Buffer.make(context.device, halves: input))
        let scaleBuffer = try #require(Fp16Buffer.make(context.device, halves: scales))
        let biasBuffer = try #require(Fp16Buffer.make(context.device, halves: biases))
        let weightBuffer = try #require(context.device.makeBuffer(
            bytes: words, length: words.count * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let out = try #require(Fp16Buffer.make(context.device, count: rows * outputs))
        let cb = try #require(context.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: cb, input: x, weights: weightBuffer,
                      scales: scaleBuffer, biases: biasBuffer, output: out,
                      columns: columns, outputs: outputs, rows: rows)
        cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        let actual = Fp16Buffer.read(out, count: rows * outputs)
        for row in 0..<rows {
            for outIndex in 0..<outputs {
                var expected: Float = 0
                for col in 0..<columns {
                    let g = outIndex * (columns / 128) + col / 128
                    expected += Float(input[row * columns + col]) *
                        (Float(codes[outIndex * columns + col]) * Float(scales[g]) + Float(biases[g]))
                }
                #expect(abs(actual[row * outputs + outIndex] - expected) < 0.02)
            }
        }
    }

    /// The wide kernel reads codes against inputs prescaled by powers of
    /// four; that must give exactly the bits of shifting the codes down.
    @Test(arguments: [(2048, 37), (5120, 72)])
    func affine2WidePrescaledMatchesShiftedCodesBitwise(shape: (columns: Int, outputs: Int)) throws {
        let (columns, outputs) = shape
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> UInt64 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return seed >> 33 }
        // Inputs span tiny to large magnitudes, including half subnormals.
        let input: [Float16] = (0..<columns).map { i in
            let magnitude = Float(next() % 2000) / 100 - 10
            return Float16(Float(i % 7 == 0 ? 0.00001 : 1) * magnitude * Float(1 << Int(next() % 6)))
        }
        let words: [UInt32] = (0..<(outputs * columns / 16)).map { _ in UInt32(truncatingIfNeeded: next() ^ (next() << 16)) }
        let scales: [Float16] = (0..<(outputs * columns / 128)).map { _ in Float16(Float(next() % 1000) / 4000 + 0.001) }
        let biases: [Float16] = (0..<(outputs * columns / 128)).map { _ in Float16(Float(next() % 1000) / 1000 - 0.5) }
        let context = try MetalContext()
        let kernel = try PrismAffine2(context: context)
        let x = try #require(Fp16Buffer.make(context.device, halves: input))
        let w = try #require(context.device.makeBuffer(bytes: words,
            length: words.count * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let sBuf = try #require(Fp16Buffer.make(context.device, halves: scales))
        let bBuf = try #require(Fp16Buffer.make(context.device, halves: biases))
        let prescaled = try #require(Fp16Buffer.make(context.device, count: outputs))
        let shifted = try #require(Fp16Buffer.make(context.device, count: outputs))
        let cb = try #require(context.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: cb, input: x, weights: w, scales: sBuf, biases: bBuf, output: prescaled,
                      columns: columns, outputs: outputs, rows: 1, variant: .wide)
        kernel.encode(commandBuffer: cb, input: x, weights: w, scales: sBuf, biases: bBuf, output: shifted,
                      columns: columns, outputs: outputs, rows: 1, variant: .wideShift)
        cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        let actual = prescaled.contents().assumingMemoryBound(to: UInt16.self)
        let expected = shifted.contents().assumingMemoryBound(to: UInt16.self)
        var mismatches = 0
        for i in 0..<outputs where actual[i] != expected[i] { mismatches += 1 }
        #expect(mismatches == 0, "\(mismatches) of \(outputs) outputs differ")
    }

    /// Prompt reading runs Bonsai's projections on the tiled tensor-op
    /// matmul. It must agree with the simple 32 x 32 kernel up to FP32
    /// summation order, across the 128-row tile, the smaller tiles and a
    /// partial output tile.
    @Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple8) == true
                      && ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 26,
                   "Metal tensor operations are unavailable on this device"),
          arguments: [8, 40, 128, 200])
    func affine2TiledPrefillMatchesReferenceKernel(rows: Int) throws {
        let columns = 2048, outputs = 100
        var seed: UInt64 = 0x1234_5678_9ABC_DEF0 &+ UInt64(rows)
        func next() -> UInt64 { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return seed >> 33 }
        let input: [Float16] = (0..<(rows * columns)).map { _ in Float16(Float(next() % 4000) / 1000 - 2) }
        let words: [UInt32] = (0..<(outputs * columns / 16)).map { _ in UInt32(truncatingIfNeeded: next() ^ (next() << 16)) }
        let scales: [Float16] = (0..<(outputs * columns / 128)).map { _ in Float16(Float(next() % 1000) / 20000 + 0.005) }
        let biases: [Float16] = (0..<(outputs * columns / 128)).map { _ in Float16(-Float(next() % 1000) / 20000 - 0.005) }
        let context = try MetalContext()
        let kernel = try PrismAffine2(context: context)
        let tiledPrefill = try #require(kernel.tiledPrefill)
        let x = try #require(Fp16Buffer.make(context.device, halves: input))
        let w = try #require(context.device.makeBuffer(bytes: words,
            length: words.count * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let sBuf = try #require(Fp16Buffer.make(context.device, halves: scales))
        let bBuf = try #require(Fp16Buffer.make(context.device, halves: biases))
        let tiled = try #require(Fp16Buffer.make(context.device, count: rows * outputs))
        let reference = try #require(Fp16Buffer.make(context.device, count: rows * outputs))
        let cb = try #require(context.queue.makeCommandBuffer())
        let before = tiledPrefill.encodedProjections
        kernel.encode(commandBuffer: cb, input: x, weights: w, scales: sBuf, biases: bBuf, output: tiled,
                      columns: columns, outputs: outputs, rows: rows)
        kernel.encode(commandBuffer: cb, input: x, weights: w, scales: sBuf, biases: bBuf, output: reference,
                      columns: columns, outputs: outputs, rows: rows, allowTiledPrefill: false)
        cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        #expect(tiledPrefill.encodedProjections == before + 1)
        let a = Fp16Buffer.read(tiled, count: rows * outputs)
        let b = Fp16Buffer.read(reference, count: rows * outputs)
        var worst: Float = 0
        for i in a.indices { worst = max(worst, abs(a[i] - b[i]) / max(1, abs(b[i]))) }
        // Both round each dequantized weight to FP16 and accumulate in FP32;
        // only the order of the sums differs.
        #expect(worst < 0.01, "largest relative difference \(worst)")
    }

    @Test func affine2BatchedMatchesSingleRowsBitwise() throws {
        let columns = 256, outputs = 11, rows = 7, stride = 17
        let input: [Float16] = (0..<(columns * rows)).map {
            Float16(Float(($0 * 13) % 97 - 48) / 37)
        }
        let codes = (0..<(columns * outputs)).map { UInt32(($0 * 7 + $0 / 9) % 4) }
        var words = [UInt32](repeating: 0, count: codes.count / 16)
        for i in codes.indices { words[i / 16] |= codes[i] << (2 * (i % 16)) }
        let scales: [Float16] = (0..<(outputs * 2)).map { Float16(Float($0 + 1) / 31) }
        let biases: [Float16] = (0..<(outputs * 2)).map { Float16(Float($0 - 11) / 29) }
        let context = try MetalContext()
        let kernel = try PrismAffine2(context: context)
        let x = try #require(Fp16Buffer.make(context.device, halves: input))
        let w = try #require(context.device.makeBuffer(bytes: words,
            length: words.count * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let s = try #require(Fp16Buffer.make(context.device, halves: scales))
        let b = try #require(Fp16Buffer.make(context.device, halves: biases))
        let sentinel = Float16(-123)
        let initial = [Float16](repeating: sentinel, count: rows * stride + 4)
        let batched = try #require(Fp16Buffer.make(context.device, halves: initial))
        let reference = try #require(Fp16Buffer.make(context.device, halves: initial))
        let cb = try #require(context.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: cb, input: x, weights: w, scales: s, biases: b,
                      output: batched, outputOffset: 4, columns: columns, outputs: outputs,
                      rows: rows, outputStride: stride)
        for row in 0..<rows {
            kernel.encode(commandBuffer: cb, input: x, weights: w, scales: s, biases: b,
                          output: reference, inputOffset: row * columns * 2,
                          outputOffset: 4 + row * stride * 2,
                          columns: columns, outputs: outputs, rows: 1)
        }
        cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        let actual = batched.contents().assumingMemoryBound(to: UInt16.self)
        let expected = reference.contents().assumingMemoryBound(to: UInt16.self)
        for i in initial.indices { #expect(actual[i] == expected[i]) }
    }

    @Test func stridedHadamardMatchesSingleRowsBitwise() throws {
        let width = 1024, block = 512, rows = 3, inputStride = 1031, outputStride = 1037
        let input: [Float16] = (0..<(rows * inputStride + 4)).map {
            Float16(Float(($0 * 11) % 101 - 50) / 32)
        }
        let signs: [Float] = (0..<width).map { $0 % 3 == 0 ? -1 : 1 }
        let context = try MetalContext()
        let kernel = try PrismHadamard(context: context)
        let x = try #require(Fp16Buffer.make(context.device, halves: input))
        let signBuffer = try #require(context.device.makeBuffer(bytes: signs,
            length: signs.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let initial = [Float16](repeating: -123, count: rows * outputStride + 4)
        for inverse in [false, true] {
            let batched = try #require(Fp16Buffer.make(context.device, halves: initial))
            let reference = try #require(Fp16Buffer.make(context.device, halves: initial))
            let cb = try #require(context.queue.makeCommandBuffer())
            kernel.encode(commandBuffer: cb, input: x, signs: signBuffer, output: batched,
                          inputOffset: 4, outputOffset: 4, width: width, rows: rows,
                          block: block, inverse: inverse, inputStride: inputStride, outputStride: outputStride)
            for row in 0..<rows {
                kernel.encode(commandBuffer: cb, input: x, signs: signBuffer, output: reference,
                              inputOffset: 4 + row * inputStride * 2,
                              outputOffset: 4 + row * outputStride * 2,
                              width: width, rows: 1, block: block, inverse: inverse)
            }
            cb.commit(); cb.waitUntilCompleted()
            try checkCommandBufferError(cb)
            let actual = batched.contents().assumingMemoryBound(to: UInt16.self)
            let expected = reference.contents().assumingMemoryBound(to: UInt16.self)
            for i in initial.indices { #expect(actual[i] == expected[i]) }
        }
    }

    @Test func affine2EmbeddingSelectsRows() throws {
        let columns = 128, vocabulary = 3
        let codes = (0..<(columns * vocabulary)).map { UInt32(($0 + $0 / 7) % 4) }
        var words = [UInt32](repeating: 0, count: codes.count / 16)
        for i in codes.indices { words[i / 16] |= codes[i] << (2 * (i % 16)) }
        let scales: [Float16] = [0.25, 0.5, 1]
        let biases: [Float16] = [-0.25, -0.5, -1]
        let tokens: [Int32] = [2, 0, -1]
        let context = try MetalContext()
        let kernel = try PrismAffine2(context: context)
        let weights = try #require(context.device.makeBuffer(bytes: words,
            length: words.count * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let ids = try #require(context.device.makeBuffer(bytes: tokens,
            length: tokens.count * MemoryLayout<Int32>.stride, options: .storageModeShared))
        let s = try #require(Fp16Buffer.make(context.device, halves: scales))
        let b = try #require(Fp16Buffer.make(context.device, halves: biases))
        let out = try #require(Fp16Buffer.make(context.device, count: columns * tokens.count))
        let cb = try #require(context.queue.makeCommandBuffer())
        kernel.encodeEmbedding(commandBuffer: cb, tokens: ids, weights: weights,
                               scales: s, biases: b, output: out,
                               columns: columns, vocabulary: vocabulary, count: tokens.count)
        cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        let actual = Fp16Buffer.read(out, count: columns * tokens.count)
        for i in tokens.indices {
            for column in 0..<columns {
                let token = Int(tokens[i])
                let expected: Float = token < 0 ? 0 :
                    Float(codes[token * columns + column]) * Float(scales[token]) + Float(biases[token])
                #expect(actual[i * columns + column] == expected)
            }
        }
    }

    @Test func floatProjectionPreservesSmallWeights() throws {
        let columns = 64, outputs = 3
        let input: [Float16] = (0..<columns).map { Float16(Float($0 % 9 - 4) / 8) }
        let weights: [Float] = (0..<(columns * outputs)).map { Float($0 % 17 - 8) / 100_000 }
        let context = try MetalContext()
        let kernel = try PrismFloatProjection(context: context)
        let x = try #require(Fp16Buffer.make(context.device, halves: input))
        let w = try #require(context.device.makeBuffer(bytes: weights,
            length: weights.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let y = try #require(context.device.makeBuffer(length: outputs * MemoryLayout<Float>.stride, options: .storageModeShared))
        let view = TensorView(buffer: w, offset: 0, length: UInt64(weights.count * 4),
                              scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                              shape: (UInt32(outputs), UInt32(columns), 0, 0), dtype: 3)
        let cb = try #require(context.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: cb, input: x, weight: view, output: y,
                      columns: columns, outputs: outputs)
        cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        let actual = Array(UnsafeBufferPointer(start: y.contents().assumingMemoryBound(to: Float.self), count: outputs))
        for row in 0..<outputs {
            let expected = (0..<columns).reduce(Float(0)) {
                $0 + Float(input[$1]) * weights[row * columns + $1]
            }
            #expect(abs(actual[row] - expected) < 0.000001)
        }
    }

    @Test func floatRMSNormMatchesIndependentRows() throws {
        let width = 128, rows = 3, eps: Float = 1e-6
        let input: [Float16] = (0..<(width * rows)).map { Float16(Float($0 % 19 - 9) / 8) }
        let weights: [Float] = (0..<width).map { 0.9 + Float($0) / 1024 }
        let context = try MetalContext()
        let kernel = try PrismFloatRMSNorm(context: context)
        let x = try #require(Fp16Buffer.make(context.device, halves: input))
        let w = try #require(context.device.makeBuffer(bytes: weights,
            length: weights.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let y = try #require(Fp16Buffer.make(context.device, count: width * rows))
        let view = TensorView(buffer: w, offset: 0, length: UInt64(weights.count * 4),
                              scaleOffset: 0, scaleLength: 0, biasOffset: 0, biasLength: 0,
                              shape: (UInt32(width), 0, 0, 0), dtype: 3)
        let cb = try #require(context.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: cb, input: x, weight: view, output: y,
                      width: width, rows: rows, eps: eps)
        cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        let actual = Fp16Buffer.read(y, count: width * rows)
        for row in 0..<rows {
            let square = (0..<width).reduce(Float(0)) {
                let value = Float(input[row * width + $1]); return $0 + value * value
            }
            let inverse = 1 / sqrt(square / Float(width) + eps)
            for col in 0..<width {
                let expected = Float(input[row * width + col]) * inverse * weights[col]
                #expect(abs(actual[row * width + col] - expected) < 0.002)
            }
        }
    }
}
