import Foundation
import Metal

/// Blockwise signed Walsh-Hadamard transform for Prism's rotated weights.
/// This primitive alone does not enable loading a Prism model.
final class PrismHadamard {
    private let pipeline: MTLComputePipelineState
    private let pipeline512: MTLComputePipelineState
    private let dual512: MTLComputePipelineState

    init(context: MetalContext) throws {
        pipeline = try context.pipeline("prism_hadamard_fp16")
        pipeline512 = try context.pipeline("prism_hadamard512_fp16")
        dual512 = try context.pipeline("prism_hadamard512_dual")
        precondition(pipeline512.maxTotalThreadsPerThreadgroup >= 64 && dual512.maxTotalThreadsPerThreadgroup >= 64)
    }

    /// Forward transform of two same-width single-row inputs in one dispatch.
    /// Only the 512-wide block has a fused kernel; callers check `supportsDual`.
    static func supportsDual(block: Int) -> Bool { block == 512 }

    func encodeDual(commandBuffer: MTLCommandBuffer, width: Int, block: Int,
                    inputA: MTLBuffer, signsA: MTLBuffer, signsAOffset: Int, outputA: MTLBuffer,
                    inputB: MTLBuffer, signsB: MTLBuffer, signsBOffset: Int, outputB: MTLBuffer) {
        precondition(Self.supportsDual(block: block) && width > 0 && width % block == 0)
        precondition(inputA.length >= width * 2 && inputB.length >= width * 2)
        precondition(outputA.length >= width * 2 && outputB.length >= width * 2)
        precondition(signsAOffset >= 0 && signsBOffset >= 0)
        precondition(signsA.length - signsAOffset >= width * MemoryLayout<Float>.stride)
        precondition(signsB.length - signsBOffset >= width * MemoryLayout<Float>.stride)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(dual512)
        encoder.setBuffer(inputA, offset: 0, index: 0)
        encoder.setBuffer(inputB, offset: 0, index: 1)
        encoder.setBuffer(signsA, offset: signsAOffset, index: 2)
        encoder.setBuffer(signsB, offset: signsBOffset, index: 3)
        encoder.setBuffer(outputA, offset: 0, index: 4)
        encoder.setBuffer(outputB, offset: 0, index: 5)
        var widthValue = UInt32(width)
        encoder.setBytes(&widthValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: 2 * width / block, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encode(commandBuffer: MTLCommandBuffer,
                input: MTLBuffer, signs: MTLBuffer, output: MTLBuffer,
                inputOffset: Int = 0, signsOffset: Int = 0, outputOffset: Int = 0,
                width: Int, rows: Int, block: Int, inverse: Bool,
                inputStride: Int? = nil, outputStride: Int? = nil) {
        let inputStride = inputStride ?? width
        let outputStride = outputStride ?? width
        precondition(width > 0 && rows > 0 && [512, 1024, 2048, 4096].contains(block))
        precondition(width % block == 0 && width <= Int(UInt32.max))
        precondition(rows <= Int(UInt32.max) / width)
        precondition(inputOffset >= 0 && outputOffset >= 0 && signsOffset >= 0)
        precondition(signsOffset <= signs.length && signs.length - signsOffset >= width * MemoryLayout<Float>.stride)
        precondition(inputStride >= width && outputStride >= width)
        precondition(inputStride <= Int(UInt32.max) && outputStride <= Int(UInt32.max))
        precondition(rows - 1 <= (Int(UInt32.max) - width) / inputStride)
        precondition(rows - 1 <= (Int(UInt32.max) - width) / outputStride)
        precondition(inputOffset <= input.length && input.length - inputOffset >= ((rows - 1) * inputStride + width) * 2)
        precondition(outputOffset <= output.length && output.length - outputOffset >= ((rows - 1) * outputStride + width) * 2)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(block == 512 ? pipeline512 : pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(signs, offset: signsOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        var widthValue = UInt32(width)
        var blockValue = UInt32(block)
        var inverseValue: UInt32 = inverse ? 1 : 0
        encoder.setBytes(&widthValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&blockValue, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&inverseValue, length: MemoryLayout<UInt32>.size, index: 5)
        var inputStrideValue = UInt32(inputStride), outputStrideValue = UInt32(outputStride)
        encoder.setBytes(&inputStrideValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&outputStrideValue, length: MemoryLayout<UInt32>.size, index: 7)
        let threads = block == 512 ? 64 : min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(MTLSize(width: rows * width / block, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

/// Initial Q2 affine projection path; model integration must also retain the
/// per-tensor signs and invoke the transform before each rotated projection.
final class PrismAffine2 {
    enum Variant: String, CaseIterable {
        case rows4, wide, split
    }

    private let pipeline: MTLComputePipelineState
    private let widePipeline: MTLComputePipelineState
    private let splitPipeline: MTLComputePipelineState
    private let batchPipeline: MTLComputePipelineState
    private let gemmPipeline: MTLComputePipelineState
    private let embeddingPipeline: MTLComputePipelineState
    private let dualPipeline: MTLComputePipelineState
    /// Single-row kernel choice; `TUFF_PRISM_GEMV` selects one for benchmarks.
    static let defaultVariant = Variant(rawValue: ProcessInfo.processInfo.environment["TUFF_PRISM_GEMV"] ?? "") ?? .wide

    init(context: MetalContext) throws {
        pipeline = try context.pipeline("prism_affine2_gemv")
        widePipeline = try context.pipeline("prism_affine2_gemv_wide")
        splitPipeline = try context.pipeline("prism_affine2_gemv_split")
        batchPipeline = try context.pipeline("prism_affine2_batched")
        gemmPipeline = try context.pipeline("prism_affine2_gemm")
        embeddingPipeline = try context.pipeline("prism_affine2_embedding")
        dualPipeline = try context.pipeline("prism_affine2_gemv_dual")
        precondition(pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= 256,
                     "Prism affine2 currently requires 32-lane SIMD groups")
        precondition(batchPipeline.threadExecutionWidth == 32 && batchPipeline.maxTotalThreadsPerThreadgroup >= 256,
                     "Prism affine2 batching requires 32-lane SIMD groups")
    }

    func encode(commandBuffer: MTLCommandBuffer, input: MTLBuffer,
                weights: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                output: MTLBuffer, inputOffset: Int = 0, weightsOffset: Int = 0,
                scalesOffset: Int = 0, biasesOffset: Int = 0, outputOffset: Int = 0,
                columns: Int, outputs: Int, rows: Int, outputStride: Int? = nil,
                variant: Variant = PrismAffine2.defaultVariant) {
        let outputStride = outputStride ?? outputs
        let variant = variant == .wide && columns % 1024 != 0 ? Variant.rows4 : variant
        precondition(columns > 0 && columns % 128 == 0 && outputs > 0 && rows > 0)
        precondition(columns <= Int(UInt32.max) && outputs <= Int(UInt32.max))
        precondition(rows <= Int(UInt32.max) / outputs)
        precondition(outputs <= Int(UInt32.max) / columns)
        precondition(rows <= Int(UInt32.max) / columns)
        precondition([inputOffset, weightsOffset, scalesOffset, biasesOffset, outputOffset].allSatisfy { $0 >= 0 })
        precondition(inputOffset <= input.length && input.length - inputOffset >= columns * rows * 2)
        precondition(outputStride >= outputs && outputStride <= Int(UInt32.max))
        precondition(rows - 1 <= (Int(UInt32.max) - outputs) / outputStride)
        precondition(outputOffset <= output.length && output.length - outputOffset >= ((rows - 1) * outputStride + outputs) * 2)
        precondition(weightsOffset <= weights.length && weights.length - weightsOffset >= outputs * columns / 4)
        precondition(scalesOffset <= scales.length && scales.length - scalesOffset >= outputs * columns / 128 * 2)
        precondition(biasesOffset <= biases.length && biases.length - biasesOffset >= outputs * columns / 128 * 2)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(rows == 1
            ? (variant == .wide ? widePipeline : variant == .split ? splitPipeline : pipeline)
            : (rows >= 8 ? gemmPipeline : batchPipeline))
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(scales, offset: scalesOffset, index: 2)
        encoder.setBuffer(biases, offset: biasesOffset, index: 3)
        encoder.setBuffer(output, offset: outputOffset, index: 4)
        var columnCount = UInt32(columns)
        var outputCount = UInt32(outputs)
        encoder.setBytes(&columnCount, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&outputCount, length: MemoryLayout<UInt32>.size, index: 6)
        if rows == 1 {
            if variant == .split {
                encoder.dispatchThreadgroups(MTLSize(width: (outputs + 3) / 4, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            } else {
                encoder.dispatchThreadgroups(MTLSize(width: (outputs + 7) / 8, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
        } else {
            var rowCount = UInt32(rows), stride = UInt32(outputStride)
            encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 7)
            encoder.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 8)
            if rows >= 8 {
                encoder.dispatchThreadgroups(MTLSize(width: (outputs + 31) / 32, height: (rows + 31) / 32, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            } else {
                encoder.dispatchThreadgroups(MTLSize(width: (outputs + 7) / 8, height: (rows + 3) / 4, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        encoder.endEncoding()
    }

    /// Two single-row projections that share `columns` in one dispatch. Callers
    /// check `supportsDual` and otherwise encode the projections separately.
    static func supportsDual(columns: Int, outputsA: Int) -> Bool {
        columns % 1024 == 0 && outputsA % 8 == 0
    }

    func encodeDual(commandBuffer: MTLCommandBuffer, columns: Int,
                    inputA: MTLBuffer, weightsA: TensorView, outputA: MTLBuffer, outputsA: Int,
                    inputB: MTLBuffer, weightsB: TensorView, outputB: MTLBuffer, outputsB: Int) {
        precondition(Self.supportsDual(columns: columns, outputsA: outputsA) && outputsB > 0)
        precondition(inputA.length >= columns * 2 && inputB.length >= columns * 2)
        precondition(outputA.length >= outputsA * 2 && outputB.length >= outputsB * 2)
        for (weights, outputs) in [(weightsA, outputsA), (weightsB, outputsB)] {
            precondition(Int(weights.offset) >= 0 && Int(weights.offset) <= weights.buffer.length
                         && weights.buffer.length - Int(weights.offset) >= outputs * columns / 4)
            precondition(Int(weights.scaleOffset) <= weights.buffer.length
                         && weights.buffer.length - Int(weights.scaleOffset) >= outputs * columns / 128 * 2)
            precondition(Int(weights.biasOffset) <= weights.buffer.length
                         && weights.buffer.length - Int(weights.biasOffset) >= outputs * columns / 128 * 2)
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(dualPipeline)
        encoder.setBuffer(inputA, offset: 0, index: 0)
        encoder.setBuffer(inputB, offset: 0, index: 1)
        encoder.setBuffer(weightsA.buffer, offset: Int(weightsA.offset), index: 2)
        encoder.setBuffer(weightsB.buffer, offset: Int(weightsB.offset), index: 3)
        encoder.setBuffer(weightsA.buffer, offset: Int(weightsA.scaleOffset), index: 4)
        encoder.setBuffer(weightsB.buffer, offset: Int(weightsB.scaleOffset), index: 5)
        encoder.setBuffer(weightsA.buffer, offset: Int(weightsA.biasOffset), index: 6)
        encoder.setBuffer(weightsB.buffer, offset: Int(weightsB.biasOffset), index: 7)
        encoder.setBuffer(outputA, offset: 0, index: 8)
        encoder.setBuffer(outputB, offset: 0, index: 9)
        var columnCount = UInt32(columns), countA = UInt32(outputsA), countB = UInt32(outputsB)
        encoder.setBytes(&columnCount, length: MemoryLayout<UInt32>.size, index: 10)
        encoder.setBytes(&countA, length: MemoryLayout<UInt32>.size, index: 11)
        encoder.setBytes(&countB, length: MemoryLayout<UInt32>.size, index: 12)
        encoder.dispatchThreadgroups(MTLSize(width: (outputsA + outputsB + 7) / 8, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeEmbedding(commandBuffer: MTLCommandBuffer, tokens: MTLBuffer,
                         weights: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                         output: MTLBuffer, tokensOffset: Int = 0, weightsOffset: Int = 0,
                         scalesOffset: Int = 0, biasesOffset: Int = 0, outputOffset: Int = 0,
                         columns: Int, vocabulary: Int, count: Int) {
        precondition(columns > 0 && columns % 128 == 0 && vocabulary > 0 && count > 0)
        precondition(columns <= Int(UInt32.max) && vocabulary <= Int(UInt32.max) && count <= Int(UInt32.max))
        precondition(count <= Int(UInt32.max) / columns)
        precondition(vocabulary <= Int(UInt32.max) / columns)
        precondition([tokensOffset, weightsOffset, scalesOffset, biasesOffset, outputOffset].allSatisfy { $0 >= 0 })
        precondition(tokensOffset <= tokens.length && tokens.length - tokensOffset >= count * MemoryLayout<Int32>.stride)
        precondition(outputOffset <= output.length && output.length - outputOffset >= count * columns * 2)
        precondition(weightsOffset <= weights.length && weights.length - weightsOffset >= vocabulary * columns / 4)
        precondition(scalesOffset <= scales.length && scales.length - scalesOffset >= vocabulary * columns / 128 * 2)
        precondition(biasesOffset <= biases.length && biases.length - biasesOffset >= vocabulary * columns / 128 * 2)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(embeddingPipeline)
        encoder.setBuffer(tokens, offset: tokensOffset, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(scales, offset: scalesOffset, index: 2)
        encoder.setBuffer(biases, offset: biasesOffset, index: 3)
        encoder.setBuffer(output, offset: outputOffset, index: 4)
        var c = UInt32(columns), v = UInt32(vocabulary), n = UInt32(count)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&v, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 7)
        let threads = min(256, embeddingPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count * columns, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

final class PrismFloatProjection {
    private let pipeline: MTLComputePipelineState
    private let dualPipeline: MTLComputePipelineState
    private let rowsPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        pipeline = try context.pipeline("prism_f32_gemv")
        dualPipeline = try context.pipeline("prism_f32_gemv_dual")
        rowsPipeline = try context.pipeline("prism_f32_gemv_rows")
        precondition(pipeline.threadExecutionWidth == 32)
    }

    /// Two FP32 matrices with the same shape applied to one input row.
    func encodeDual(commandBuffer: MTLCommandBuffer, input: MTLBuffer,
                    weightA: TensorView, outputA: MTLBuffer,
                    weightB: TensorView, outputB: MTLBuffer, columns: Int, outputs: Int) {
        precondition(columns > 0 && columns % 128 == 0 && outputs > 0)
        for weight in [weightA, weightB] {
            precondition(weight.dtype == 3 && Int(weight.shape.0) == outputs && Int(weight.shape.1) == columns)
            precondition(weight.length == UInt64(columns * outputs * MemoryLayout<Float>.stride))
            precondition(Int(weight.offset) % 16 == 0)
        }
        precondition(input.length >= columns * 2 && input.length % 8 == 0)
        precondition(outputA.length >= outputs * 4 && outputB.length >= outputs * 4)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(dualPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weightA.buffer, offset: Int(weightA.offset), index: 1)
        encoder.setBuffer(weightB.buffer, offset: Int(weightB.offset), index: 2)
        encoder.setBuffer(outputA, offset: 0, index: 3)
        encoder.setBuffer(outputB, offset: 0, index: 4)
        var c = UInt32(columns), n = UInt32(outputs)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: 2 * outputs, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// `encode` for `rows` token rows of `input` (stride `inputStride`
    /// halves) in one dispatch; row `t` writes `outputs` floats at `t * outputs`.
    func encodeRows(commandBuffer: MTLCommandBuffer, input: MTLBuffer, weight: TensorView,
                    output: MTLBuffer, columns: Int, outputs: Int, rows: Int, inputStride: Int) {
        precondition(columns > 0 && outputs > 0 && rows > 0 && inputStride >= columns)
        precondition(weight.dtype == 3 && Int(weight.shape.0) == outputs && Int(weight.shape.1) == columns)
        precondition(input.length >= ((rows - 1) * inputStride + columns) * 2)
        precondition(output.length >= rows * outputs * 4)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(rowsPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight.buffer, offset: Int(weight.offset), index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var c = UInt32(columns), o = UInt32(outputs), stride = UInt32(inputStride)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&o, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: outputs, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encode(commandBuffer: MTLCommandBuffer, input: MTLBuffer, weight: TensorView,
                output: MTLBuffer, columns: Int, outputs: Int,
                inputOffset: Int = 0, outputOffset: Int = 0) {
        precondition(columns > 0 && outputs > 0 && columns <= Int(UInt32.max) / outputs)
        precondition(weight.dtype == 3 && Int(weight.shape.0) == outputs && Int(weight.shape.1) == columns)
        precondition(weight.length == UInt64(columns * outputs * MemoryLayout<Float>.stride))
        precondition(inputOffset >= 0 && inputOffset <= input.length && input.length - inputOffset >= columns * 2)
        precondition(outputOffset >= 0 && outputOffset <= output.length && output.length - outputOffset >= outputs * 4)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(weight.buffer, offset: Int(weight.offset), index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        var c = UInt32(columns)
        encoder.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: outputs, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

final class PrismFloatRMSNorm {
    private let pipeline: MTLComputePipelineState
    init(context: MetalContext) throws {
        pipeline = try context.pipeline("prism_f32_rmsnorm")
        precondition(pipeline.threadExecutionWidth == 32)
    }

    func encode(commandBuffer: MTLCommandBuffer, input: MTLBuffer, weight: TensorView,
                output: MTLBuffer, width: Int, rows: Int, eps: Float,
                inputOffset: Int = 0, outputOffset: Int = 0) {
        precondition(width > 0 && rows > 0 && width <= Int(UInt32.max) / rows)
        precondition(weight.dtype == 3 && Int(weight.shape.0) == width && weight.length == UInt64(width * 4))
        precondition(inputOffset >= 0 && inputOffset <= input.length && input.length - inputOffset >= width * rows * 2)
        precondition(outputOffset >= 0 && outputOffset <= output.length && output.length - outputOffset >= width * rows * 2)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(weight.buffer, offset: Int(weight.offset), index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        var d = UInt32(width), epsilon = eps
        encoder.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&epsilon, length: MemoryLayout<Float>.size, index: 4)
        let threads = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
