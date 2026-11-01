import Foundation
import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

@Suite struct FlintAttentionTests {
    @Test("virtual sink affects full and sliding attention", arguments: [
        (Int(4), Int(4), Float(0)), (Int(7), Int(3), Float(0)),
        (Int(4), Int(4), Float(8)), (Int(7), Int(3), Float(8)),
    ])
    func sinkAddsZeroValuedSoftmaxLogit(sequence: Int, window: Int, sink: Float) throws {
        let context = try MetalContext(), kernel = try Attention(context: context)
        let heads = 4, kvHeads = 1, dim = 128
        let q = try #require(Fp16Buffer.make(context.device, halves: Array(repeating: Float16(0), count: heads * dim)))
        let k = try #require(Fp16Buffer.make(context.device, halves: Array(repeating: Float16(0), count: sequence * kvHeads * dim)))
        let v = try #require(Fp16Buffer.make(context.device, halves: Array(repeating: Float16(1), count: sequence * kvHeads * dim)))
        let out = try #require(Fp16Buffer.make(context.device, count: heads * dim))
        let bits = Array(repeating: Quantization.bf16Bits(sink), count: heads)
        let sinks = try #require(context.device.makeBuffer(bytes: bits, length: bits.count * 2, options: .storageModeShared))
        let cb = try #require(context.queue.makeCommandBuffer())
        if window == sequence {
            kernel.encodeFull(commandBuffer: cb, q: q, k: k, v: v, out: out,
                headDim: UInt32(dim), numQHeads: UInt32(heads), numKVHeads: UInt32(kvHeads),
                seqLen: UInt32(sequence), sinks: sinks)
        } else {
            kernel.encodeSWA(commandBuffer: cb, q: q, k: k, v: v, out: out,
                headDim: UInt32(dim), numQHeads: UInt32(heads), numKVHeads: UInt32(kvHeads),
                seqLen: UInt32(sequence), window: UInt32(window), sinks: sinks)
        }
        cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb)
        let expected = Float(window) / (Float(window) + exp(sink))
        for actual in Fp16Buffer.read(out, count: heads * dim) {
            #expect(abs(actual - expected) < 0.001)
        }
    }
}
