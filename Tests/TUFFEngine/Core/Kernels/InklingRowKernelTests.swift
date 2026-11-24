import Foundation
import Metal
import Testing
import TUFFRuntimeDescriptions
@testable import TUFFEngine

/// The several-rows expert kernels against the one-row kernels, byte for
/// byte, and the visit order decode lookahead predicts along. Needs no model.
@Suite struct InklingRowKernelTests {
    /// The load-time comparison, repeated on fresh random weights, and once
    /// more at Inkling 2 Mini's own shapes (hidden 2304, intermediate 1152)
    /// with eight distinct experts.
    @Test func rowKernelsGiveTheOneRowKernelsBytes() throws {
        let context = try MetalContext()
        try context.prepareKernelGroups([.core, .inkling])
        let kernels = try InklingKernels(context: context)
        #expect(kernels.hasRowKernels)
        for _ in 0..<4 {
            #expect(InklingForwardRunner.rowKernelsMatch(kernels, device: context.device))
        }
        let queue = try #require(context.device.makeCommandQueue())
        var random = SystemRandomNumberGenerator()
        let mini = RowKernelProbe(kernels: kernels, device: context.device, queue: queue,
                                  columns: 2304, intermediate: 1152)
        #expect(try mini.run(using: &random))
    }

    /// Every visit once, in the order `run` makes them; each visit's
    /// successor is the next one's layer, and the token's last has none.
    @Test func visitOrderCoversTheLoop() {
        let loop = ArchConfig.inkling2Mini.inklingLoop
        let layers = ArchConfig.inkling2Mini.numLayers
        let order = InklingForwardRunner.visitOrder(numLayers: layers, loop: loop)
        #expect(order.count == loop.visits(numLayers: layers))
        #expect(Set(order.map(\.visit)).count == order.count)
        for entry in order { #expect(loop.layer(ofVisit: entry.visit, numLayers: layers) == entry.layer) }
        #expect(order.map(\.layer) == Array(0..<18) + Array(6..<18) + Array(6..<18) + Array(18..<24))
        let next = InklingForwardRunner.nextLayers(numLayers: layers, loop: loop)
        for index in order.indices.dropLast() { #expect(next[order[index].visit] == order[index + 1].layer) }
        #expect(next[order.last!.visit] == nil)
    }
}
