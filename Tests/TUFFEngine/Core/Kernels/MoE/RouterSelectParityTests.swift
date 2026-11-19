import Foundation
import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

/// `router_topk_select_k8_ranked` finds the router's top-K by rank on a whole
/// threadgroup. It has to pick the experts the single-thread scan picks, in
/// the scan's order, and give them the scan's weights, byte for byte.
@Suite struct RouterSelectParityTests {
    private struct Picked: Equatable {
        let indices: [UInt32]
        let weights: [UInt16]
    }

    private static func select(_ moe: MoE, context: MetalContext, logits: [Float],
                               scales: [UInt16], topK: Int, specialized: Bool) throws -> Picked {
        let slots = MoE.maxStreamedExperts
        let device = context.device
        let input = try #require(device.makeBuffer(
            bytes: logits, length: logits.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let scale = try #require(device.makeBuffer(
            bytes: scales, length: scales.count * 2, options: .storageModeShared))
        let indices = try #require(device.makeBuffer(
            length: slots * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let weights = try #require(device.makeBuffer(
            length: slots * MemoryLayout<UInt16>.stride, options: .storageModeShared))
        // A kernel that writes nothing leaves this pattern behind.
        memset(indices.contents(), 0xA5, indices.length)
        memset(weights.contents(), 0xA5, weights.length)
        let command = try #require(context.queue.makeCommandBuffer())
        moe.encodeRouterSelectForTesting(
            commandBuffer: command, logits: input, perExpertScale: scale,
            outIndices: indices, outWeights: weights,
            numExperts: UInt32(logits.count), specialized: specialized)
        command.commit()
        command.waitUntilCompleted()
        #expect(command.status == .completed)
        return Picked(
            indices: Array(UnsafeBufferPointer(
                start: indices.contents().bindMemory(to: UInt32.self, capacity: topK), count: topK)),
            weights: Array(UnsafeBufferPointer(
                start: weights.contents().bindMemory(to: UInt16.self, capacity: topK), count: topK)))
    }

    /// What the scan is defined to pick: the highest scores, ties to the
    /// lower index, never minus infinity or NaN; unfilled slots are expert 0.
    private static func expectedIndices(_ logits: [Float], topK: Int) -> [UInt32] {
        let eligible = logits.enumerated().filter { $0.element > -.infinity }
            .sorted { $0.element == $1.element ? $0.offset < $1.offset : $0.element > $1.element }
        return (0..<topK).map { $0 < eligible.count ? UInt32(eligible[$0].offset) : 0 }
    }

    @Test func rankedSelectionMatchesTheScanBitForBit() throws {
        let context = try MetalContext()
        // Expert counts below, at and above the 256 threads of the group, and
        // not multiples of it; K at the default and at other specialized sizes.
        let shapes = [(5, 8), (16, 8), (128, 8), (256, 8), (300, 10), (512, 16), (128, 4)]
        var rng = SplitMix64(seed: 0x5E1E_C7ED)
        var compared = 0
        for (experts, topK) in shapes {
            let scan = try MoE(context: context, specializedNumExperts: UInt32(experts),
                               topKExperts: topK, routerSelection: .serial)
            let ranked = try MoE(context: context, specializedNumExperts: UInt32(experts),
                                 topKExperts: topK, routerSelection: .ranked)
            #expect(!scan.usesRankedRouterSelection && ranked.usesRankedRouterSelection)
            let scales = (0..<experts).map { _ -> UInt16 in
                UInt16(truncatingIfNeeded: rng.uniform(0.6, 1.4).bitPattern >> 16)
            }
            for trial in 0..<72 {
                let logits: [Float] = (0..<experts).map { index in
                    switch trial % 9 {
                    // Zeros of both signs tie; the lower index wins.
                    case 6: return index % 3 == 0 ? -0.0
                        : (index % 3 == 1 ? 0.0 : rng.uniform(-2.0, 0.2))
                    // No usable score at all: both kernels leave expert 0 in
                    // every slot and write NaN weights. The bytes still match.
                    case 7: return index % 2 == 0 ? -.infinity : .nan
                    // Plus infinity is picked first and also ends in NaN weights.
                    case 8: return index == experts / 3 ? .infinity : rng.uniform(-4.0, 4.0)
                    case 0: return rng.uniform(-4.0, 4.0)
                    // Many exact ties, including across the K-th place.
                    case 1: return (rng.uniform(0.0, 6.0)).rounded() * 0.25
                    // Scores the scan never takes, among ordinary ones.
                    case 2: return index % 5 == 1 ? -.infinity
                        : (index % 7 == 3 ? .nan : rng.uniform(-4.0, 4.0))
                    // Fewer real scores than slots.
                    case 3: return index % 37 == 2 ? rng.uniform(0.0, 1.0) : -.infinity
                    case 4: return 0.5
                    // One dominant expert and a wide spread.
                    default: return index == experts / 2 ? 40 : rng.uniform(-30.0, 10.0)
                    }
                }
                for specialized in [true, false] {
                    let fromScan = try Self.select(scan, context: context, logits: logits,
                                                   scales: scales, topK: topK, specialized: specialized)
                    let fromRanked = try Self.select(ranked, context: context, logits: logits,
                                                     scales: scales, topK: topK, specialized: specialized)
                    #expect(fromScan == fromRanked,
                            "experts \(experts), K \(topK), trial \(trial), specialized \(specialized)")
                    #expect(fromRanked.indices == Self.expectedIndices(logits, topK: topK),
                            "experts \(experts), K \(topK), trial \(trial)")
                    compared += 1
                }
            }
        }
        #expect(compared == shapes.count * 72 * 2)
    }

    /// Whichever kernel the load-time comparison leaves in place, the result
    /// is the scan's. With `TUFF_ROUTER_SELECT=serial` it is the scan itself.
    @Test func automaticSelectionGivesTheScansResult() throws {
        let context = try MetalContext()
        let scan = try MoE(context: context, routerSelection: .serial)
        let automatic = try MoE(context: context)
        if ProcessInfo.processInfo.environment["TUFF_ROUTER_SELECT"] == "serial" {
            #expect(!automatic.usesRankedRouterSelection)
        }
        var rng = SplitMix64(seed: 0x5E1E_C7EE)
        let scales = (0..<128).map { _ -> UInt16 in
            UInt16(truncatingIfNeeded: rng.uniform(0.6, 1.4).bitPattern >> 16)
        }
        for _ in 0..<16 {
            let logits = (0..<128).map { _ in rng.uniform(-6.0, 6.0) }
            for specialized in [true, false] {
                #expect(try Self.select(automatic, context: context, logits: logits, scales: scales,
                                        topK: 8, specialized: specialized)
                    == Self.select(scan, context: context, logits: logits, scales: scales,
                                   topK: 8, specialized: specialized))
            }
        }
    }
}
