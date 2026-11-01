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
