import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// Compares the Metal `logit_softcap_softmax` kernel against
/// `LogitSoftcapSoftmaxRef`, an Accelerate-based two-pass reference (apply
/// softcap → find max → subtract → exp → divide by sum). The kernel does a
/// single-pass online safe softmax with the softcap fused into the (m, d)
/// running update. Different code paths through the same math.
@Suite struct LogitSoftcapSoftmaxTests {

    private static let softcap: Float = 30.0

    private static func runKernel(logitsFp16: [Float16], v: Int, softcap: Float) throws -> [Float] {
        let ctx = try MetalContext()
        let kernel = try LogitSoftcapSoftmax(context: ctx)

        guard let inBuf = Fp16Buffer.make(ctx.device, halves: logitsFp16),
              let outBuf = Fp16Buffer.make(ctx.device, count: v),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate Metal resources")
            return []
        }
        kernel.encode(commandBuffer: cmd,
                      logits: inBuf, probs: outBuf,
                      v: UInt32(v), softcap: softcap)
        cmd.commit()
        cmd.waitUntilCompleted()
        return Fp16Buffer.read(outBuf, count: v)
    }

    /// The three-dispatch path runs the two per-element passes one thread
    /// per logit and keeps the sum on the single kernel's 256 chains. Its
    /// probabilities must be the single kernel's, byte for byte, for every
    /// vocabulary size the models use and the awkward ones around a chain.
    @Test func splitPathMatchesTheSingleKernelBitForBit() throws {
        let ctx = try MetalContext()
        let sizes = [1, 7, 255, 256, 257, 1_000, 2_048, 4_133, 65_537, 151_936, 200_064, 262_144]
        var rng = SeedTree(0x5F7).key("softmax-split-parity")
        var compared = 0
        for softcap in [Float(30.0), 0.0] {
            for v in sizes {
                let single = try LogitSoftcapSoftmax(context: ctx, vocab: v, softcap: softcap,
                                                     path: .single)
                let split = try LogitSoftcapSoftmax(context: ctx, vocab: v, softcap: softcap,
                                                    path: .split)
                #expect(!single.usesSplitPath && split.usesSplitPath)
                for shape in 0..<3 {
                    var logits = (0..<v).map { index -> Float16 in
                        switch shape {
                        case 0: return Float16(rng.uniform(-12.0, 12.0))
                        // Raw Gemma logits reach the hundreds, deep in the cap.
                        case 1: return Float16(rng.uniform(-600.0, 600.0))
                        // One confident token over a flat tail.
                        default: return index == v / 3 ? 40 : Float16(rng.uniform(-3.0, 0.0))
                        }
                    }
                    // With a cap, minus infinity is an ordinary capped value.
                    // Without one it can turn a chain's sum into NaN in
                    // either path, and equal NaN would prove nothing.
                    if softcap > 0, v > 4 { logits[v - 3] = -.infinity }
                    let bytes = v * MemoryLayout<Float16>.stride
                    guard let input = Fp16Buffer.make(ctx.device, halves: logits),
                          let fromSingle = Fp16Buffer.make(ctx.device, count: v),
                          let fromSplit = Fp16Buffer.make(ctx.device, count: v),
                          let command = ctx.queue.makeCommandBuffer() else {
                        Issue.record("Failed to allocate Metal resources")
                        return
                    }
                    memset(fromSingle.contents(), 0x00, bytes)
                    memset(fromSplit.contents(), 0xFF, bytes)
                    single.encode(commandBuffer: command, logits: input, probs: fromSingle,
                                  v: UInt32(v), softcap: softcap)
                    split.encode(commandBuffer: command, logits: input, probs: fromSplit,
                                 v: UInt32(v), softcap: softcap)
                    command.commit()
                    command.waitUntilCompleted()
                    #expect(command.status == .completed)
                    #expect(memcmp(fromSingle.contents(), fromSplit.contents(), bytes) == 0,
                            "softcap \(softcap), V \(v), shape \(shape)")
                    let probabilities = Fp16Buffer.read(fromSplit, count: v)
                    #expect(probabilities.allSatisfy { $0.isFinite && $0 >= 0 },
                            "softcap \(softcap), V \(v), shape \(shape)")
                    #expect(abs(probabilities.reduce(0, +) - 1) < 0.02,
                            "softcap \(softcap), V \(v), shape \(shape)")
                    compared += 1
                }
            }
        }
        #expect(compared == 2 * sizes.count * 3)
    }

    /// Whichever path the sampler ends up with (the split where the
    /// load-time comparison passes, the single kernel where it does not or
    /// with `TUFF_SOFTMAX_SPLIT=off`), the probabilities are the single
    /// kernel's, including for a vocabulary larger than the scratch it was
    /// built for.
    @Test func automaticPathGivesTheSingleKernelsProbabilities() throws {
        let ctx = try MetalContext()
        let automatic = try LogitSoftcapSoftmax(context: ctx, vocab: 2_048, softcap: Self.softcap)
        if ProcessInfo.processInfo.environment["TUFF_SOFTMAX_SPLIT"] == "off" {
            #expect(!automatic.usesSplitPath)
        }
        #expect(try !LogitSoftcapSoftmax(context: ctx).usesSplitPath)
        #expect(try !LogitSoftcapSoftmax(context: ctx, vocab: 2_048, path: .single).usesSplitPath)

        var rng = SeedTree(0x5F8).key("softmax-split-capacity")
        // Inside the scratch, beyond it, and with a cap other than the one
        // the path was checked with: the last two use the single kernel.
        for (v, softcap) in [(2_000, Self.softcap), (4_096, Self.softcap), (2_000, Float(0))] {
            let logits = (0..<v).map { _ in Float16(rng.uniform(-20.0, 20.0)) }
            let reference = try Self.runKernel(logitsFp16: logits, v: v, softcap: softcap)
            guard let input = Fp16Buffer.make(ctx.device, halves: logits),
                  let output = Fp16Buffer.make(ctx.device, count: v),
                  let command = ctx.queue.makeCommandBuffer() else {
                Issue.record("Failed to allocate Metal resources")
                return
            }
            automatic.encode(commandBuffer: command, logits: input, probs: output,
                             v: UInt32(v), softcap: softcap)
            command.commit()
            command.waitUntilCompleted()
            #expect(Fp16Buffer.read(output, count: v) == reference, "V \(v), softcap \(softcap)")
        }
    }

    @Test func randomLogits_matchesReference() throws {
        let v = 2048
        var rng = SeedTree(0x131).key("softcap-softmax-random")
        let logitsFp32 = (0..<v).map { _ in rng.uniform(-50.0, 50.0) }
        let logitsFp16 = logitsFp32.map { Float16($0) }

        let gpu = try Self.runKernel(logitsFp16: logitsFp16, v: v, softcap: Self.softcap)
        let cpu = LogitSoftcapSoftmaxRef.apply(
            x: logitsFp16.map { Float($0) }, softcap: Self.softcap
        )

        // Probabilities at the extreme tail are near FP16 subnormal; use the
        // bounded-relative form so we don't blow up on c~0.
        let rel = RelError.boundedRel(actual: gpu, reference: cpu, absFloor: 1e-4)
        #expect(rel < Tolerance.fp16Reduction, "rel=\(rel)")

        // Probability axioms.
        let sum = gpu.reduce(0, +)
        #expect(abs(sum - 1.0) < Tolerance.fp16Reduction, "sum=\(sum)")
    }

    /// All logits identical → uniform 1/V distribution.
    @Test func uniformLogits_producesUniformProbs() throws {
        let v = 2048
        let logitsFp16 = [Float16](repeating: Float16(0.1234), count: v)

        let gpu = try Self.runKernel(logitsFp16: logitsFp16, v: v, softcap: Self.softcap)
        let expected: Float = 1.0 / Float(v)
        for i in 0..<v {
            #expect(abs(gpu[i] - expected) < 5e-4,
                    "i=\(i) g=\(gpu[i]) expected=\(expected)")
        }
    }

    /// One logit far above the softcap, rest at zero. Softcap must pin the
    /// outlier — kernel must not let the un-capped 1000.0 leak through and
    /// overflow exp. Small softcap (5.0) so saturated prob is < 1.0.
    @Test func singleHugeLogit_isSoftcapped() throws {
        let v = 2048
        let smallSoftcap: Float = 5.0
        var logits = [Float](repeating: 0, count: v)
        logits[42] = 1000.0
        let logitsFp16 = logits.map { Float16($0) }

        let gpu = try Self.runKernel(logitsFp16: logitsFp16, v: v, softcap: smallSoftcap)
        let cpu = LogitSoftcapSoftmaxRef.apply(
            x: logitsFp16.map { Float($0) }, softcap: smallSoftcap
        )
        let g = gpu[42]
        let c = cpu[42]
        #expect(c < 0.5, "softcap reference should not saturate: c=\(c)")
        #expect(abs(g - c) / c < Tolerance.fp16Reduction, "g=\(g) c=\(c)")
    }
}
