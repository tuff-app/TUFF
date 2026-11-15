import Foundation
import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

/// The Dk = 128 prefill recurrence must reproduce the general kernel bit for
/// bit. Each case runs the whole prefill chain (conv + tail update, q/k norm,
/// delta recurrence, gated norm) twice from the same carried state, once per
/// recurrence kernel, and compares every output, the final state and the conv
/// tail exactly.
@Suite struct GDNDeltaPrefillParityTests {
    private static let chunkPlans: [[Int]] = [
        [1], [2], [31], [32], [255], [256], [256, 31, 1, 256, 2],
    ]

    /// Qwen 3.6, Ornith and Qwen 3.8: Hk 16, Dk 128, Dv 128 with 32, 48 or
    /// 64 value heads.
    @Test(arguments: [32, 48, 64])
    func qwenShapesMatchGeneralKernel(valueHeads: Int) throws {
        let config = LinearAttentionConfig(numKHeads: 16, numVHeads: valueHeads,
                                           keyHeadDim: 128, valueHeadDim: 128,
                                           convKernelSize: 4)
        try Self.checkParity(config: config, seed: UInt64(valueHeads))
    }

    @Test func smallShapeMatchesGeneralKernel() throws {
        let config = LinearAttentionConfig(numKHeads: 2, numVHeads: 6,
                                           keyHeadDim: 128, valueHeadDim: 64,
                                           convKernelSize: 4,
                                           outputGate: .sigmoid)
        try Self.checkParity(config: config, seed: 0x6D)
    }

    /// Dv 32 is the narrowest admitted shape: one threadgroup row per head.
    @Test func minimumValueDimMatchesGeneralKernel() throws {
        let config = LinearAttentionConfig(numKHeads: 2, numVHeads: 4,
                                           keyHeadDim: 128, valueHeadDim: 32,
                                           convKernelSize: 4)
        try Self.checkParity(config: config, seed: 0x20)
    }

    @Test func otherShapesKeepTheGeneralKernel() throws {
        let context = try MetalContext()
        let small = LinearAttentionConfig(numKHeads: 2, numVHeads: 4, keyHeadDim: 32,
                                          valueHeadDim: 32, convKernelSize: 4)
        let narrow = LinearAttentionConfig(numKHeads: 2, numVHeads: 4, keyHeadDim: 128,
                                           valueHeadDim: 20, convKernelSize: 4)
        let qwen = LinearAttentionConfig(numKHeads: 16, numVHeads: 32, keyHeadDim: 128,
                                         valueHeadDim: 128, convKernelSize: 4)
        #expect(try !GDN(context: context, config: small).usesDk128DeltaPrefill)
        #expect(try !GDN(context: context, config: narrow).usesDk128DeltaPrefill)
        #expect(try GDN(context: context, config: qwen).usesDk128DeltaPrefill)
    }

    private struct Run {
        let tail: MTLBuffer
        let state: MTLBuffer
        let convOut: MTLBuffer
        let y: MTLBuffer
        let out: MTLBuffer
    }

    private static func checkParity(config: LinearAttentionConfig, seed: UInt64) throws {
        let context = try MetalContext()
        let gdn = try GDN(context: context, config: config)
        #expect(gdn.usesDk128DeltaPrefill)
        let device = context.device
        var rng = SplitMix64(seed: 0x6D0_0000 ^ seed)
        let C = config.qkvDim
        let Hv = config.numVHeads
        let valueDim = config.valueDim
        let taps = config.convKernelSize
        let maxRows = chunkPlans.map { $0.reduce(0, +) }.max()!

        func halves(_ count: Int, _ lo: Float, _ hi: Float) throws -> MTLBuffer {
            try #require(Fp16Buffer.make(device, halves: (0..<count).map { _ in
                Float16(rng.uniform(lo, hi))
            }))
        }
        func bf16(_ count: Int, _ lo: Float, _ hi: Float) throws -> MTLBuffer {
            let bits = (0..<count).map { _ in Quantization.bf16Bits(rng.uniform(lo, hi)) }
            return try #require(device.makeBuffer(bytes: bits, length: bits.count * 2,
                                                  options: .storageModeShared))
        }
        func copy(_ buffer: MTLBuffer) throws -> MTLBuffer {
            try #require(device.makeBuffer(bytes: buffer.contents(), length: buffer.length,
                                           options: .storageModeShared))
        }
        func zeros(_ length: Int) throws -> MTLBuffer {
            let buffer = try #require(device.makeBuffer(length: length, options: .storageModeShared))
            memset(buffer.contents(), 0, length)
            return buffer
        }

        let qkv = try halves(maxRows * C, -2, 2)
        let z = try halves(maxRows * valueDim, -3, 3)
        // a spans the softplus shortcut at 20; b spans both sigmoid tails.
        let a = try halves(maxRows * Hv, -6, 24)
        let b = try halves(maxRows * Hv, -8, 8)
        let convW = try bf16(C * taps, -0.6, 0.6)
        let aLog = try bf16(Hv, -3, 2)
        let dtBias = try bf16(Hv, -3, 3)
        let normW = try bf16(config.valueHeadDim, 0.5, 1.5)
        // State and tail as a previous chunk would leave them.
        let initialTail = try halves((taps - 1) * C, -2, 2)
        let stateCount = Hv * config.valueHeadDim * config.keyHeadDim
        let initialState = try #require(device.makeBuffer(
            bytes: (0..<stateCount).map { _ in rng.uniform(-0.5, 0.5) },
            length: stateCount * 4, options: .storageModeShared))

        for plan in chunkPlans {
            func run(generalKernel: Bool) throws -> Run {
                let total = plan.reduce(0, +)
                let result = Run(tail: try copy(initialTail), state: try copy(initialState),
                                 convOut: try zeros(total * C * 2), y: try zeros(total * valueDim * 2),
                                 out: try zeros(total * valueDim * 2))
                var start = 0
                for rows in plan {
                    let cb = try #require(context.queue.makeCommandBuffer())
                    gdn.encodeConvPrefill(commandBuffer: cb, tail: result.tail,
                                          qkvRows: qkv, qkvRowsOffset: start * C * 2,
                                          convWeight: convW, convWeightOffset: 0,
                                          out: result.convOut, outOffset: start * C * 2,
                                          rows: rows)
                    gdn.encodeConvTailUpdate(commandBuffer: cb, tail: result.tail,
                                             qkvRows: qkv, qkvRowsOffset: start * C * 2,
                                             rows: rows)
                    gdn.encodeQKNorm(commandBuffer: cb, convOut: result.convOut,
                                     convOutOffset: start * C * 2, rows: rows)
                    gdn.encodeDeltaStepPrefill(commandBuffer: cb,
                                               convOut: result.convOut, convOutOffset: start * C * 2,
                                               aProj: a, aProjOffset: start * Hv * 2,
                                               bProj: b, bProjOffset: start * Hv * 2,
                                               aLog: aLog, aLogOffset: 0,
                                               dtBias: dtBias, dtBiasOffset: 0,
                                               state: result.state,
                                               y: result.y, yOffset: start * valueDim * 2,
                                               rows: rows, generalKernel: generalKernel)
                    gdn.encodeGatedNorm(commandBuffer: cb,
                                        y: result.y, yOffset: start * valueDim * 2,
                                        z: z, zOffset: start * valueDim * 2,
                                        weight: normW, weightOffset: 0,
                                        out: result.out, outOffset: start * valueDim * 2,
                                        rows: rows)
                    cb.commit()
                    cb.waitUntilCompleted()
                    #expect(cb.status == .completed)
                    start += rows
                }
                return result
            }
            let reference = try run(generalKernel: true)
            let candidate = try run(generalKernel: false)
            for (name, want, got) in [("y", reference.y, candidate.y),
                                      ("gated output", reference.out, candidate.out),
                                      ("state", reference.state, candidate.state),
                                      ("conv tail", reference.tail, candidate.tail)] {
                let mismatch = memcmp(want.contents(), got.contents(), want.length) != 0
                #expect(!mismatch, "\(name) differs for chunks \(plan), Hv \(Hv)")
            }
            // Guard against a vacuous pass: the recurrence must have moved
            // the state and produced non-zero outputs.
            #expect(memcmp(initialState.contents(), reference.state.contents(),
                           initialState.length) != 0)
            let y = reference.y.contents().bindMemory(to: UInt16.self,
                                                      capacity: reference.y.length / 2)
            #expect((0..<(reference.y.length / 2)).contains { y[$0] & 0x7FFF != 0 })
        }
    }
}
