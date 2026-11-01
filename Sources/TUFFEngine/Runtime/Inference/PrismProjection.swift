import Foundation
import Metal

/// Rotation and Q2 projection primitives for a Prism language graph. The
/// runner owns one workspace, used in command-buffer order; callers must not
/// encode overlapping command buffers against the same instance.
final class PrismProjection {
    private let transform: PrismHadamard
    private let affine: PrismAffine2
    private let floatProjection: PrismFloatProjection
    private let floatNorm: PrismFloatRMSNorm
    private let silu: MTLComputePipelineState
    private let workspace: MTLBuffer
    private let secondWorkspace: MTLBuffer
    private let block: Int

    init(context: MetalContext, block: Int, maximumWidth: Int) throws {
        precondition([512, 1024, 2048, 4096].contains(block))
        precondition(maximumWidth > 0 && maximumWidth % block == 0)
        self.block = block
        transform = try PrismHadamard(context: context)
        affine = try PrismAffine2(context: context)
        floatProjection = try PrismFloatProjection(context: context)
        floatNorm = try PrismFloatRMSNorm(context: context)
        silu = try context.pipeline("silu_mul_fp16")
        guard let buffer = context.device.makeBuffer(length: maximumWidth * MemoryLayout<Float16>.stride * 16,
                                                      options: .storageModeShared),
              let second = context.device.makeBuffer(length: maximumWidth * MemoryLayout<Float16>.stride,
                                                     options: .storageModeShared) else {
            throw ModelError.indexCorrupt(detail: "unable to allocate Prism projection scratch")
        }
        workspace = buffer
        secondWorkspace = second
    }

    /// Whether `encodeDual` can fuse two rotated projections with this shape.
    func supportsDual(columns: Int, outputsA: Int) -> Bool {
        PrismHadamard.supportsDual(block: block) && columns % block == 0
            && columns <= secondWorkspace.length / 2
            && PrismAffine2.supportsDual(columns: columns, outputsA: outputsA)
    }

    /// Two single-row projections of the same input in two dispatches (one
    /// rotation, one matrix multiply) instead of four. A and B keep their own
    /// sign vectors, so the rotations differ.
    func encodeDual(commandBuffer: MTLCommandBuffer, model: Model,
                    weightA: TensorView, weightB: TensorView, input: MTLBuffer,
                    outputA: MTLBuffer, outputB: MTLBuffer,
                    columns: Int, outputsA: Int, outputsB: Int) throws {
        guard supportsDual(columns: columns, outputsA: outputsA),
              weightA.dtype == 0, weightB.dtype == 0,
              Int(weightA.shape.0) == outputsA, Int(weightA.shape.1) == columns,
              Int(weightB.shape.0) == outputsB, Int(weightB.shape.1) == columns,
              input.length >= columns * 2 else {
            throw ModelError.indexCorrupt(detail: "invalid Prism dual projection shape")
        }
        let signsA = try model.prismSigns(for: weightA)
        let signsB = try model.prismSigns(for: weightB)
        guard signsA.dtype == 3, Int(signsA.shape.0) == columns,
              signsA.length == UInt64(columns * MemoryLayout<Float>.stride),
              signsB.dtype == 3, Int(signsB.shape.0) == columns,
              signsB.length == UInt64(columns * MemoryLayout<Float>.stride) else {
            throw ModelError.indexCorrupt(detail: "invalid Prism projection signs")
        }
        transform.encodeDual(commandBuffer: commandBuffer, width: columns, block: block,
                             inputA: input, signsA: signsA.buffer, signsAOffset: Int(signsA.offset),
                             outputA: workspace,
                             inputB: input, signsB: signsB.buffer, signsBOffset: Int(signsB.offset),
                             outputB: secondWorkspace)
        affine.encodeDual(commandBuffer: commandBuffer, columns: columns,
                          inputA: workspace, weightsA: weightA, outputA: outputA, outputsA: outputsA,
                          inputB: secondWorkspace, weightsB: weightB, outputB: outputB, outputsB: outputsB)
    }

    /// GDN alpha and beta (unrotated F32) in one dispatch.
    func encodeFloatDual(commandBuffer: MTLCommandBuffer, weightA: TensorView, weightB: TensorView,
                         input: MTLBuffer, outputA: MTLBuffer, outputB: MTLBuffer,
                         columns: Int, outputs: Int) {
        floatProjection.encodeDual(commandBuffer: commandBuffer, input: input,
                                   weightA: weightA, outputA: outputA,
                                   weightB: weightB, outputB: outputB,
                                   columns: columns, outputs: outputs)
    }

    /// GDN alpha and beta projections are unrotated F32 matrices.
    func encodeFloat(commandBuffer: MTLCommandBuffer, weight: TensorView,
                     input: MTLBuffer, output: MTLBuffer, columns: Int, outputs: Int,
                     inputOffset: Int = 0, outputOffset: Int = 0) {
        floatProjection.encode(commandBuffer: commandBuffer, input: input, weight: weight,
                               output: output, columns: columns, outputs: outputs,
                               inputOffset: inputOffset, outputOffset: outputOffset)
    }

    func encodeNorm(commandBuffer: MTLCommandBuffer, weight: TensorView,
                    input: MTLBuffer, output: MTLBuffer, width: Int, rows: Int,
                    eps: Float, inputOffset: Int = 0, outputOffset: Int = 0) {
        floatNorm.encode(commandBuffer: commandBuffer, input: input, weight: weight,
                         output: output, width: width, rows: rows, eps: eps,
                         inputOffset: inputOffset, outputOffset: outputOffset)
    }

    func encodeSiluMul(commandBuffer: MTLCommandBuffer, gate: MTLBuffer, up: MTLBuffer,
                       output: MTLBuffer, count: Int) {
        precondition(count > 0 && count <= Int(UInt32.max))
        precondition(gate.length >= count * 2 && up.length >= count * 2 && output.length >= count * 2)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(silu)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var n = UInt32(count)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(256, silu.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Apply H(Dx), then project bounded batches of strided token rows.
    func encode(commandBuffer: MTLCommandBuffer, model: Model,
                weight: TensorView, input: MTLBuffer, output: MTLBuffer,
                columns: Int, outputs: Int, rows: Int,
                inputOffset: Int = 0, outputOffset: Int = 0,
                inputStride: Int, outputStride: Int) throws {
        guard weight.dtype == 0, Int(weight.shape.0) == outputs,
              Int(weight.shape.1) == columns, columns > 0, outputs > 0, columns % block == 0,
              columns <= Int(UInt32.max), outputs <= Int(UInt32.max) / columns,
              inputStride >= columns, outputStride >= outputs, rows > 0,
              inputOffset >= 0, outputOffset >= 0,
              inputOffset <= input.length, outputOffset <= output.length,
              columns <= (input.length - inputOffset) / 2,
              outputs <= (output.length - outputOffset) / 2,
              rows - 1 <= ((input.length - inputOffset) / 2 - columns) / inputStride,
              rows - 1 <= ((output.length - outputOffset) / 2 - outputs) / outputStride,
              columns * MemoryLayout<Float16>.stride <= workspace.length else {
            throw ModelError.indexCorrupt(detail: "invalid Prism projection shape")
        }
        let signs = try model.prismSigns(for: weight)
        guard signs.dtype == 3, Int(signs.shape.0) == columns,
              signs.length == UInt64(columns * MemoryLayout<Float>.stride) else {
            throw ModelError.indexCorrupt(detail: "invalid Prism projection signs")
        }
        let capacity = min(16, workspace.length / (columns * 2))
        var row = 0
        while row < rows {
            // Keep every per-dispatch element index representable as UInt32.
            let inputRows = inputStride > Int(UInt32.max) ? 1
                : 1 + (Int(UInt32.max) - columns) / inputStride
            let outputRows = outputStride > Int(UInt32.max) ? 1
                : 1 + (Int(UInt32.max) - outputs) / outputStride
            let batch = min(capacity, rows - row, inputRows, outputRows,
                            Int(UInt32.max) / columns, Int(UInt32.max) / outputs)
            transform.encode(commandBuffer: commandBuffer, input: input, signs: signs.buffer,
                             output: workspace, inputOffset: inputOffset + row * inputStride * 2,
                             signsOffset: Int(signs.offset), width: columns, rows: batch,
                             block: block, inverse: false, inputStride: batch == 1 ? columns : inputStride)
            affine.encode(commandBuffer: commandBuffer, input: workspace, weights: weight.buffer,
                          scales: weight.buffer, biases: weight.buffer, output: output,
                          weightsOffset: Int(weight.offset), scalesOffset: Int(weight.scaleOffset),
                          biasesOffset: Int(weight.biasOffset), outputOffset: outputOffset + row * outputStride * 2,
                          columns: columns, outputs: outputs, rows: batch,
                          outputStride: batch == 1 ? outputs : outputStride)
            row += batch
        }
    }

    /// Dequantize each token's row and restore the embedding with D(Hx).
    func encodeEmbedding(commandBuffer: MTLCommandBuffer, model: Model,
                         weight: TensorView, tokens: MTLBuffer, output: MTLBuffer,
                         width: Int, vocabulary: Int, count: Int) throws {
        guard weight.dtype == 0, Int(weight.shape.0) == vocabulary,
              Int(weight.shape.1) == width, width > 0, width % block == 0, count >= 0,
              width <= Int(UInt32.max), vocabulary > 0, vocabulary <= Int(UInt32.max) / width,
              count <= tokens.length / MemoryLayout<Int32>.stride,
              count <= output.length / 2 / width,
              width * MemoryLayout<Float16>.stride <= workspace.length else {
            throw ModelError.indexCorrupt(detail: "invalid Prism embedding shape")
        }
        let signs = try model.prismSigns(for: weight)
        guard signs.dtype == 3, Int(signs.shape.0) == width,
              signs.length == UInt64(width * MemoryLayout<Float>.stride) else {
            throw ModelError.indexCorrupt(detail: "invalid Prism embedding signs")
        }
        let capacity = min(16, workspace.length / (width * 2))
        var row = 0
        while row < count {
            let batch = min(capacity, count - row, Int(UInt32.max) / width)
            affine.encodeEmbedding(commandBuffer: commandBuffer, tokens: tokens, weights: weight.buffer,
                scales: weight.buffer, biases: weight.buffer, output: workspace,
                tokensOffset: row * MemoryLayout<Int32>.stride, weightsOffset: Int(weight.offset),
                scalesOffset: Int(weight.scaleOffset), biasesOffset: Int(weight.biasOffset),
                columns: width, vocabulary: vocabulary, count: batch)
            transform.encode(commandBuffer: commandBuffer, input: workspace, signs: signs.buffer,
                output: output, signsOffset: Int(signs.offset), outputOffset: row * width * 2,
                width: width, rows: batch, block: block, inverse: true)
            row += batch
        }
    }
}
