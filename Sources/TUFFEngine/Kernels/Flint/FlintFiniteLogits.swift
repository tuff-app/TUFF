import Foundation
import Metal

enum FlintNumericalError: LocalizedError {
    case nonFiniteLogits
    case validationUnavailable

    var errorDescription: String? {
        switch self {
        case .nonFiniteLogits: "Flint inference produced non-finite logits; generation stopped."
        case .validationUnavailable: "Flint could not validate its logits; generation stopped."
        }
    }
}

/// One GPU reduction per emitted token avoids copying the vocabulary to the CPU.
final class FlintFiniteLogits {
    private let pipeline: MTLComputePipelineState
    private let invalid: MTLBuffer

    init(context: MetalContext) throws {
        pipeline = try context.pipeline("flint_check_finite_logits")
        guard let flag = context.device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw FlintNumericalError.validationUnavailable
        }
        invalid = flag
    }

    func prepare() { invalid.contents().storeBytes(of: UInt32(0), as: UInt32.self) }

    func encode(commandBuffer: MTLCommandBuffer, logits: MTLBuffer, count: Int) throws {
        guard count > 0, count <= Int(UInt32.max),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw FlintNumericalError.validationUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(invalid, offset: 0, index: 1)
        var count = UInt32(count)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(count), pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }

    func validate() throws {
        guard invalid.contents().load(as: UInt32.self) == 0 else { throw FlintNumericalError.nonFiniteLogits }
    }
}
