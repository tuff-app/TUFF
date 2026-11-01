import Metal
import Testing
import TUFFValidationSupport
@testable import TUFFEngine

@Suite struct FlintFiniteLogitsTests {
    @Test func finiteValuesPassAndNonFiniteValuesFailClosed() throws {
        let context = try MetalContext()
        let check = try FlintFiniteLogits(context: context)
        for values: [Float16] in [[0, -1, 2], [0, .nan, 1], [0, .infinity, -1]] {
            let logits = try #require(Fp16Buffer.make(context.device, halves: values))
            let cb = try #require(context.queue.makeCommandBuffer())
            check.prepare()
            try check.encode(commandBuffer: cb, logits: logits, count: values.count)
            cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb)
            if values.allSatisfy(\.isFinite) {
                try check.validate()
            } else {
                #expect(throws: FlintNumericalError.self) { try check.validate() }
            }
        }
    }
}
