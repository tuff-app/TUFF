import Foundation
import Metal

final class FusedQKVGEMV {
    private struct Shape: Hashable {
        var qRows: UInt32
        var kvRows: UInt32
        var n: UInt32
    }

    let groupSize: Int
    private let pso: MTLComputePipelineState
    private let specializedPSOs: [Shape: MTLComputePipelineState]

    private static let realDecodeShapes: [Shape] = [
        Shape(qRows: 4096, kvRows: 2048, n: 2816),
        Shape(qRows: 8192, kvRows: 1024, n: 2816),
    ]

    /// `groupSize` reaches the packed q/k/v GEMV, which dequantizes INT4.
    /// Function constants are per-pipeline: omitting it decodes at 64.
    init(context: MetalContext,
         additionalShapes: [(qRows: Int, kvRows: Int, n: Int)] = [],
         groupSize: Int = Quantization.groupSize) throws {
        precondition(Quantization.supportedGroupSizes.contains(groupSize),
                     "unsupported affine group size \(groupSize)")
        self.groupSize = groupSize
        let groupConstants: [MetalFunctionConstant] =
            groupSize == Quantization.groupSize
                ? []
                : [MetalFunctionConstant(
                    index: Quantization.groupSizeFunctionConstantIndex,
                    value: .uint32(UInt32(groupSize)))]
        self.pso = try context.pipeline("dequant_int4_qkv_gemv_simd",
                                        constants: groupConstants,
                                        maxTotalThreadsPerThreadgroup: 512)
        var variants: [Shape: MTLComputePipelineState] = [:]
        let shapes = Self.realDecodeShapes + additionalShapes.map {
            Shape(qRows: UInt32($0.qRows), kvRows: UInt32($0.kvRows), n: UInt32($0.n))
        }
        for shape in Set(shapes) {
            variants[shape] = try context.pipeline(
                "dequant_int4_qkv_gemv_simd",
                constants: groupConstants + [
                    MetalFunctionConstant(index: 23, value: .uint32(shape.qRows)),
                    MetalFunctionConstant(index: 24, value: .uint32(shape.kvRows)),
                    MetalFunctionConstant(index: 25, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 26, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPSOs = variants
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       qWeights: MTLBuffer, qWeightsOffset: Int = 0,
                       qScales: MTLBuffer, qScalesOffset: Int = 0,
                       qBiases: MTLBuffer, qBiasesOffset: Int = 0,
                       kWeights: MTLBuffer, kWeightsOffset: Int = 0,
                       kScales: MTLBuffer, kScalesOffset: Int = 0,
                       kBiases: MTLBuffer, kBiasesOffset: Int = 0,
                       vWeights: MTLBuffer, vWeightsOffset: Int = 0,
                       vScales: MTLBuffer, vScalesOffset: Int = 0,
                       vBiases: MTLBuffer, vBiasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       qOut: MTLBuffer, qOutOffset: Int = 0,
                       kOut: MTLBuffer, kOutOffset: Int = 0,
                       vOut: MTLBuffer, vOutOffset: Int = 0,
                       qRows: UInt32,
                       kvRows: UInt32,
                       n: UInt32) {
        precondition(n % UInt32(groupSize) == 0,
                     "N must be a multiple of \(groupSize)")
        precondition(qWeightsOffset % 2 == 0 &&
                     kWeightsOffset % 2 == 0 &&
                     vWeightsOffset % 2 == 0,
                     "FusedQKVGEMV needs 2-aligned weights offsets")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        let shape = Shape(qRows: qRows, kvRows: kvRows, n: n)
        enc.setComputePipelineState(specializedPSOs[shape] ?? pso)
        enc.setBuffer(qWeights, offset: qWeightsOffset, index: 0)
        enc.setBuffer(qScales, offset: qScalesOffset, index: 1)
        enc.setBuffer(qBiases, offset: qBiasesOffset, index: 2)
        enc.setBuffer(kWeights, offset: kWeightsOffset, index: 3)
        enc.setBuffer(kScales, offset: kScalesOffset, index: 4)
        enc.setBuffer(kBiases, offset: kBiasesOffset, index: 5)
        enc.setBuffer(vWeights, offset: vWeightsOffset, index: 6)
        enc.setBuffer(vScales, offset: vScalesOffset, index: 7)
        enc.setBuffer(vBiases, offset: vBiasesOffset, index: 8)
        enc.setBuffer(x, offset: xOffset, index: 9)
        enc.setBuffer(qOut, offset: qOutOffset, index: 10)
        enc.setBuffer(kOut, offset: kOutOffset, index: 11)
        enc.setBuffer(vOut, offset: vOutOffset, index: 12)
        var qVar = qRows
        var kvVar = kvRows
        var nVar = n
        enc.setBytes(&qVar, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&kvVar, length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 15)
        let totalRows = Int(qRows + 2 * kvRows)
        enc.dispatchThreadgroups(MTLSize(width: (totalRows + 7) / 8,
                                         height: 1,
                                         depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256,
                                                                 height: 1,
                                                                 depth: 1))
        enc.endEncoding()
    }
}
