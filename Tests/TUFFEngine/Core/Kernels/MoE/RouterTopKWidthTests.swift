import Foundation
import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

/// The routed-expert kernels were written when every checkpoint selected
/// exactly eight experts, and said so: `precondition(topK == 8)`, a top-k
/// search over `top_idx[8]`, and a phase-2 reduction unrolled as
/// `partial[0] … partial[7]`. Qwen3.8 Flash Next routes ten of its 512.
///
/// The failure that mattered was silent: with the count raised but the
/// reduction left unrolled, experts eight and nine would have been selected,
/// fetched from SSD, multiplied by their routing weights — and then dropped on
/// the floor, producing a model that ran at full cost and answered slightly
/// wrongly. These tests check the count is honoured end to end.
@Suite struct RouterTopKWidthTests {
    private static let experts = 32
    private static let dimension = 128

    private static func selectTopK(_ k: Int,
                                   groupSize: Int = Quantization.groupSize)
        throws -> (indices: [UInt32], weights: [Float])? {
        var rng = SplitMix64(seed: 0x4B57_4454 &+ UInt64(k))
        // A clean ranking, so the expected selection is unambiguous: expert e
        // gets a gain that decreases with e, making the top-k exactly 0..<k.
        let weights = (0..<Self.experts).map { expert in
            (0..<Self.dimension).map { _ in
                Float(Self.experts - expert) * 0.02 + rng.uniform(-0.0005, 0.0005)
            }
        }
        let hidden = [Float](repeating: 1.0, count: Self.dimension)
        let invSqrtD = 1.0 / Float(Self.dimension).squareRoot()
        let effectiveScale = [Float](repeating: invSqrtD, count: Self.dimension)
        let expertScale = [Float](repeating: 1.0, count: Self.experts)

        // INT8 at the default group of 64: what the checkpoint's `mlp.gate`
        // is, regardless of how its INT4 tensors are grouped.
        let rows = weights.map { Quantization.quantizeInt8Affine($0) }
        let packed = rows.flatMap(\.packed)
        let scales = rows.flatMap(\.scales)
        let biases = rows.flatMap(\.biases)

        let context = try MetalContext()
        let kernel = try MoE(context: context,
                             specializedNumExperts: UInt32(Self.experts),
                             topKExperts: k,
                             groupSize: groupSize)
        guard let weightBuffer = context.device.makeBuffer(
                  bytes: packed, length: packed.count, options: .storageModeShared),
              let scaleBuffer = context.device.makeBuffer(
                  bytes: scales, length: scales.count * 2, options: .storageModeShared),
              let biasBuffer = context.device.makeBuffer(
                  bytes: biases, length: biases.count * 2, options: .storageModeShared),
              let hiddenBuffer = Fp16Buffer.make(context.device, values: hidden),
              let effectiveBuffer = context.device.makeBuffer(
                  bytes: effectiveScale.map(Quantization.bf16Bits),
                  length: effectiveScale.count * 2, options: .storageModeShared),
              let expertScaleBuffer = context.device.makeBuffer(
                  bytes: expertScale.map(Quantization.bf16Bits),
                  length: expertScale.count * 2, options: .storageModeShared),
              let indexBuffer = context.device.makeBuffer(
                  length: k * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let outputWeightBuffer = Fp16Buffer.make(context.device, count: k),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("alloc failed"); return nil
        }
        kernel.encodeRouterGemma4(
            commandBuffer: commandBuffer,
            weights: weightBuffer, scales: scaleBuffer, biases: biasBuffer,
            hidden: hiddenBuffer, effectiveScale: effectiveBuffer,
            perExpertScale: expertScaleBuffer,
            outIndices: indexBuffer, outWeights: outputWeightBuffer,
            numExperts: UInt32(Self.experts),
            d: UInt32(Self.dimension),
            topK: UInt32(k))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let indexPointer = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: k)
        return ((0..<k).map { indexPointer[$0] },
                Fp16Buffer.read(outputWeightBuffer, count: k))
    }

    /// Eight is what every shipping model uses, ten is Qwen3.8 Flash Next.
    @Test(arguments: [8, 10])
    func selectionReturnsExactlyTheRequestedExperts(k: Int) throws {
        guard let result = try Self.selectTopK(k) else { return }
        #expect(result.indices.count == k)
        // The gains decrease with expert index, so the top-k is 0..<k in order.
        #expect(result.indices == (0..<UInt32(k)).map { $0 })
    }

    /// The softmax runs over the selected set, so the weights must sum to one
    /// across all `k`. Normalizing over eight while selecting ten would leave
    /// this short.
    @Test(arguments: [8, 10])
    func routingWeightsNormalizeOverTheFullSelection(k: Int) throws {
        guard let result = try Self.selectTopK(k) else { return }
        let total = result.weights.reduce(0, +)
        #expect(abs(total - 1.0) < 0.01, "k=\(k) weights sum to \(total)")
        for (i, weight) in result.weights.enumerated() {
            #expect(weight > 0, "k=\(k) expert \(i) got no weight")
        }
        // Ranked selection means monotonically decreasing weights.
        for i in 1..<k {
            #expect(result.weights[i] <= result.weights[i - 1] + 1e-3)
        }
    }

    /// The combination the checkpoint actually presents: ten routed experts
    /// selected by an INT8 router that is still grouped at 64, while the INT4
    /// expert weights around it are grouped at 32.
    ///
    /// The first version of this test asked for `groupSize: 32` and then fed
    /// the router weights quantized at 64, which is what the checkpoint does —
    /// and the router decoded them at 32 and chose ten arbitrary experts. The
    /// wrapper now scopes the group size to the INT4 kernels, so the router is
    /// unaffected by it and this asks for the real arrangement.
    @Test func tenExpertsSelectCorrectlyWhileExpertsAreGroupedAt32() throws {
        guard let result = try Self.selectTopK(10, groupSize: 32) else { return }
        #expect(result.indices == (0..<UInt32(10)).map { $0 },
                "the INT4 group size reached the INT8 router")
        let total = result.weights.reduce(0, +)
        #expect(abs(total - 1.0) < 0.01, "weights sum to \(total)")
    }

    /// And the same selection with the group size left alone, so a regression
    /// in the scoping shows up as a difference between these two rather than
    /// as both being equally wrong.
    @Test func theInt4GroupSizeDoesNotChangeRouterSelection() throws {
        guard let grouped32 = try Self.selectTopK(10, groupSize: 32),
              let grouped64 = try Self.selectTopK(10, groupSize: 64) else { return }
        #expect(grouped32.indices == grouped64.indices)
        for (a, b) in zip(grouped32.weights, grouped64.weights) {
            #expect(abs(a - b) < 1e-3)
        }
    }

    /// A count past the blob array has to be refused rather than corrupting
    /// memory.
    @Test(arguments: [0, MoE.maxStreamedExperts + 1])
    func invalidRoutedExpertCountsAreRejected(count: Int) throws {
        let context = try MetalContext()
        #expect(throws: MoE.ConfigurationError.invalidRoutedExpertCount(count)) {
            try MoE(context: context, topKExperts: count)
        }
    }
}
