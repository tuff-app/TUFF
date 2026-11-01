import Foundation
import Testing
@testable import TUFFEngine

@Suite struct FlintRouterTests {
    @Test func sigmoidTopSixUsesBF16ProjectionAndNormalizesSelectedScores() throws {
        let context = try MetalContext()
        let rows = 12, width = 32, topK = 6
        let desired: [Float] = [-3, 2, -1, 0, 4, 1, 3, -2, 0.5, 1.5, -4, 2.5]
        // One nonzero feature makes the CPU reference independent of a
        // reduction-order assumption while covering unequal sigmoid scores.
        var matrix = Array(repeating: UInt16(0), count: rows * width)
        for i in 0..<rows { matrix[i * width] = Quantization.bf16Bits(desired[i]) }
        var input = Array(repeating: Float16(0), count: width); input[0] = 1
        let w = try #require(context.device.makeBuffer(bytes: matrix,
            length: matrix.count * MemoryLayout<UInt16>.stride, options: .storageModeShared))
        let x = try #require(context.device.makeBuffer(bytes: input,
            length: input.count * MemoryLayout<Float16>.stride, options: .storageModeShared))
        let ids = try #require(context.device.makeBuffer(length: topK * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let probs = try #require(context.device.makeBuffer(length: topK * MemoryLayout<Float16>.stride, options: .storageModeShared))
        let cb = try #require(context.queue.makeCommandBuffer())
        try FlintRouter(context: context, maximumExperts: rows).encode(
            commandBuffer: cb, weight: w, hidden: x, indices: ids, weights: probs,
            hiddenSize: width, experts: rows, topK: topK)
        cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb)
        let order = desired.indices.sorted { desired[$0] > desired[$1] }.prefix(topK)
        let raw = order.map { 1 / (1 + exp(-desired[$0])) }
        let total = raw.reduce(0, +)
        let idPtr = ids.contents().bindMemory(to: UInt32.self, capacity: topK)
        let pPtr = probs.contents().bindMemory(to: Float16.self, capacity: topK)
        for (i, id) in order.enumerated() {
            #expect(idPtr[i] == UInt32(id))
            #expect(abs(Float(pPtr[i]) - raw[i] / total) < 0.001)
        }
    }
    @Test func floatInputPreservesRouterOrderingBelowHalfPrecision() throws {
        let context = try MetalContext()
        let width = 32
        var matrix = Array(repeating: UInt16(0), count: 2 * width)
        matrix[0] = Quantization.bf16Bits(1)
        matrix[width + 1] = Quantization.bf16Bits(1)
        var input = Array(repeating: Float(0), count: width)
        input[0] = 1.0001
        input[1] = 1.0002
        let w = try #require(context.device.makeBuffer(bytes: matrix,
            length: matrix.count * MemoryLayout<UInt16>.stride, options: .storageModeShared))
        let x = try #require(context.device.makeBuffer(bytes: input,
            length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let ids = try #require(context.device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let weights = try #require(context.device.makeBuffer(length: MemoryLayout<Float16>.stride, options: .storageModeShared))
        let cb = try #require(context.queue.makeCommandBuffer())
        try FlintRouter(context: context, maximumExperts: 2).encode(
            commandBuffer: cb, weight: w, hidden: x, indices: ids, weights: weights,
            hiddenSize: width, experts: 2, topK: 1, floatInput: true)
        cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb)
        #expect(ids.contents().load(as: UInt32.self) == 1)
    }

    @Test func affineInt8RouterUsesGroupBiasAndFloatInput() throws {
        let context = try MetalContext()
        let width = 128, rows = 2
        var bytes = [UInt8](repeating: 16, count: width)
            + [UInt8](repeating: 17, count: width)
        bytes[0] = 17
        bytes[width + 65] = 18
        let scalesOffset = bytes.count
        func append(_ bits: UInt16) {
            bytes.append(UInt8(truncatingIfNeeded: bits))
            bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
        }
        for _ in 0..<4 { append(Quantization.bf16Bits(1)) }
        let biasesOffset = bytes.count
        for value in [Float(-16), -16, -17, -17] { append(Quantization.bf16Bits(value)) }
        var input = Array(repeating: Float(0), count: width)
        input[0] = 1.0001
        input[65] = 1.0002
        let w = try #require(context.device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared))
        let x = try #require(context.device.makeBuffer(bytes: input,
            length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let ids = try #require(context.device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let weights = try #require(context.device.makeBuffer(length: MemoryLayout<Float16>.stride, options: .storageModeShared))
        let cb = try #require(context.queue.makeCommandBuffer())
        try FlintRouter(context: context, maximumExperts: rows).encode(commandBuffer: cb,
            weight: w, hidden: x, indices: ids, weights: weights, hiddenSize: width,
            experts: rows, topK: 1, floatInput: true,
            affineScalesOffset: scalesOffset, affineBiasesOffset: biasesOffset)
        cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb)
        #expect(ids.contents().load(as: UInt32.self) == 1)
        #expect(weights.contents().load(as: Float16.self) == 1)
    }

    @Test func residualAccumulatesSubHalfPrecisionDeltas() throws {
        let context = try MetalContext()
        let residual = try FlintResidual(context: context, width: 32)
        let values = residual.values.contents().bindMemory(to: Float.self, capacity: 32)
        for i in 0..<32 { values[i] = 1 }
        let delta = Array(repeating: Float16(0.0001), count: 32)
        let d = try #require(context.device.makeBuffer(bytes: delta,
            length: delta.count * MemoryLayout<Float16>.stride, options: .storageModeShared))
        let mirror = try #require(context.device.makeBuffer(length: 32 * MemoryLayout<Float16>.stride, options: .storageModeShared))
        let cb = try #require(context.queue.makeCommandBuffer())
        for _ in 0..<10 { residual.add(cb, delta: d, mirror: mirror, count: 32) }
        cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb)
        #expect(abs(values[0] - (1 + 10 * Float(delta[0]))) < 0.000001)
    }
}
