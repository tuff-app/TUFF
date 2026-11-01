import Foundation
import Metal

@frozen
public struct MoEExpertOffsets {
    public var gateWOff: UInt32
    public var gateSOff: UInt32
    public var gateBOff: UInt32
    public var upWOff: UInt32
    public var upSOff: UInt32
    public var upBOff: UInt32
    public var downWOff: UInt32
    public var downSOff: UInt32
    public var downBOff: UInt32

    public init(gateWOff: UInt32, gateSOff: UInt32, gateBOff: UInt32,
                upWOff: UInt32, upSOff: UInt32, upBOff: UInt32,
                downWOff: UInt32, downSOff: UInt32, downBOff: UInt32) {
        self.gateWOff = gateWOff
        self.gateSOff = gateSOff
        self.gateBOff = gateBOff
        self.upWOff = upWOff
        self.upSOff = upSOff
        self.upBOff = upBOff
        self.downWOff = downWOff
        self.downSOff = downSOff
        self.downBOff = downBOff
    }
}

/// Affine packed widths for the three projections in one routed layer.
struct MoEExpertBitWidths {
    let gate: UInt32
    let up: UInt32
    let down: UInt32

    init(gate: UInt32 = 4, up: UInt32 = 4, down: UInt32 = 4) {
        precondition([4, 8].contains(gate) && [4, 8].contains(up) && [4, 8].contains(down))
        self.gate = gate
        self.up = up
        self.down = down
    }
}


final class MoE {
    /// Routed experts the streamed kernels can hold blobs for. Matches
    /// `kMaxStreamedExperts` in `quant_group.metal`.
    static let maxStreamedExperts = 16
    /// The count every checkpoint routed until Qwen3.8 Flash Next, which
    /// selects ten of its 512 experts.
    static let defaultStreamedExperts = 8

    private let realDecodeD: UInt32
    private let realDecodeF: UInt32
    private let realDecodeTopK: UInt32
    private let realDecodeNumExperts: UInt32
    /// Affine group size these pipelines decode.
    let groupSize: Int

    private let routerGemvPSO: MTLComputePipelineState
    private let routerGemvSpecializedPSO: MTLComputePipelineState
    private let routerSelectK8PSO: MTLComputePipelineState
    private let routerSelectK8SpecializedPSO: MTLComputePipelineState
    private let routerSelectMiniMaxK8PSO: MTLComputePipelineState
    private let routerSelectMiniMaxK8SpecializedPSO: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    /// Experts `routerLogits` can hold.
    private let routerLogitCapacity: Int
    private let phase1U16PSO: MTLComputePipelineState
    private let phase1U16SpecializedPSO: MTLComputePipelineState
    private let phase1SubsetU16PSO: MTLComputePipelineState
    private let phase1SubsetU16SpecializedPSO: MTLComputePipelineState
    private let phase2ReduceK8PSO: MTLComputePipelineState
    private let phase2ReduceK8SpecializedPSO: MTLComputePipelineState
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer

    /// `specializedD`/`specializedF`/`specializedNumExperts` describe the
    /// production shape this instance specializes for (Gemma 4 by default;
    /// Qwen 3.6 passes 2048/512/256). `siluActivation` selects the expert
    /// FFN activation (false = gelu_pytorch_tanh, true = silu).
    init(context: MetalContext,
         siluActivation: Bool = false,
         specializedD: UInt32 = 2816,
         specializedF: UInt32 = 704,
         specializedNumExperts: UInt32 = 128,
         topKExperts: Int = MoE.defaultStreamedExperts,
         groupSize: Int = Quantization.groupSize,
         wideIntermediate: Bool = false,
         floatIntermediate: Bool = false) throws {
        precondition(Quantization.supportedGroupSizes.contains(groupSize),
                     "unsupported affine group size \(groupSize)")
        precondition(!floatIntermediate || wideIntermediate)
        precondition(topKExperts > 0 && topKExperts <= Self.maxStreamedExperts,
                     "routed top-k \(topKExperts) exceeds the streamed blob array")
        self.realDecodeD = specializedD
        self.realDecodeF = specializedF
        self.realDecodeNumExperts = specializedNumExperts
        self.realDecodeTopK = UInt32(topKExperts)
        self.groupSize = groupSize
        // Two independent knobs, and conflating them is a real bug rather
        // than untidiness: in the Qwen3.8 Flash Next checkpoint the group size
        // tracks the bit width, not the model. Its 4-bit tensors — attention,
        // shared expert, routed experts, the PLE table — are grouped at 32,
        // while its 8-bit router and shared-expert gate are grouped at 64,
        // exactly as every earlier checkpoint's are. So the group size belongs
        // only to the INT4 expert kernels; the INT8 router keeps the default.
        //
        // Both default to the historical values when left unset, so the
        // shipping models keep the exact pipelines they had.
        let int4GroupConstants: [MetalFunctionConstant] =
            groupSize == Quantization.groupSize
                ? []
                : [MetalFunctionConstant(
                    index: Quantization.groupSizeFunctionConstantIndex,
                    value: .uint32(UInt32(groupSize)))]
        // The routed-expert count reaches the top-k search and the phase-2
        // reduction, neither of which decodes anything.
        let expertCountConstants: [MetalFunctionConstant] =
            topKExperts == Self.defaultStreamedExperts
                ? []
                : [MetalFunctionConstant(
                    index: Quantization.streamedExpertsFunctionConstantIndex,
                    value: .uint32(UInt32(topKExperts)))]
        let expertConstants = int4GroupConstants + expertCountConstants
        let activationConstants: [MetalFunctionConstant] = siluActivation
            ? [MetalFunctionConstant(index: 4, value: .bool(true))]
            : []
        let moeConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 0, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 1, value: .uint32(specializedF)),
            MetalFunctionConstant(index: 2, value: .uint32(UInt32(topKExperts))),
            MetalFunctionConstant(index: 3, value: .bool(true)),
        ] + activationConstants + expertConstants
        let routerConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 40, value: .uint32(specializedNumExperts)),
            MetalFunctionConstant(index: 41, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 42, value: .uint32(UInt32(topKExperts))),
            MetalFunctionConstant(index: 43, value: .bool(true)),
        ] + expertCountConstants
        let routerName = "router_gemv_gemma4_r4"
        self.routerGemvPSO = try context.pipeline(
            routerName,
            constants: expertCountConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.routerGemvSpecializedPSO = try context.pipeline(
            routerName,
            constants: routerConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.routerSelectK8PSO = try context.pipeline(
            "router_topk_select_k8", constants: expertCountConstants)
        self.routerSelectK8SpecializedPSO = try context.pipeline(
            "router_topk_select_k8",
            constants: routerConstants)
        self.routerSelectMiniMaxK8PSO = try context.pipeline(
            "router_topk_select_minimax_k8", constants: expertCountConstants)
        self.routerSelectMiniMaxK8SpecializedPSO = try context.pipeline(
            "router_topk_select_minimax_k8",
            constants: routerConstants)
        let phase1Name = floatIntermediate ? "flint_moe_phase1_float"
            : (wideIntermediate ? "flint_moe_phase1_bf16" : "moe_phase1_gate_up_act_u16load")
        let phase1SubsetName = floatIntermediate ? "flint_moe_phase1_subset_float"
            : (wideIntermediate ? "flint_moe_phase1_subset_bf16" : "moe_phase1_gate_up_act_subset_u16load")
        let phase2Name = floatIntermediate ? "flint_moe_phase2_float"
            : (wideIntermediate ? "flint_moe_phase2_bf16" : "moe_phase2_down_reduce_k8")
        self.phase1U16PSO = try context.pipeline(
            phase1Name,
            constants: activationConstants + expertConstants)
        self.phase1U16SpecializedPSO = try context.pipeline(
            phase1Name,
            constants: moeConstants)
        self.phase1SubsetU16PSO = try context.pipeline(
            phase1SubsetName,
            constants: activationConstants + expertConstants)
        self.phase1SubsetU16SpecializedPSO = try context.pipeline(
            phase1SubsetName,
            constants: moeConstants)
        self.phase2ReduceK8PSO = try context.pipeline(
            phase2Name, constants: expertConstants)
        self.phase2ReduceK8SpecializedPSO = try context.pipeline(
            phase2Name,
            constants: moeConstants)

        // One float per expert. Fixed at 256 while that was the most any
        // checkpoint routed over; Qwen3.8 Flash Next has 512 per layer, and a
        // buffer sized for 256 would have the router writing past its end.
        let routerLogitCapacity = max(256, Int(specializedNumExperts))
        guard let logits = context.device.makeBuffer(
            length: routerLogitCapacity * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let phase1Function = try? context.function(
                named: phase1Name) else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        self.routerLogitCapacity = routerLogitCapacity
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    func encodeRouterGemma4(commandBuffer: MTLCommandBuffer,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   scales: MTLBuffer, scalesOffset: Int = 0,
                                   biases: MTLBuffer, biasesOffset: Int = 0,
                                   hidden: MTLBuffer,
                                   effectiveScale: MTLBuffer, effectiveScaleOffset: Int = 0,
                                   perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                                   outIndices: MTLBuffer,
                                   outWeights: MTLBuffer,
                                   numExperts: UInt32,
                                   d: UInt32,
                                   topK: UInt32) {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(Int(numExperts) <= routerLogitCapacity,
                     "router logits hold \(routerLogitCapacity) experts, "
                        + "asked for \(numExperts)")
        precondition(topK == realDecodeTopK)

        var expertCount = numExperts
        var dimension = d
        let useSpecialized = numExperts == realDecodeNumExperts
            && d == realDecodeD
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            encoder.setBuffer(biases, offset: biasesOffset, index: 2)
            encoder.setBuffer(hidden, offset: 0, index: 3)
            encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
            encoder.setBuffer(routerLogits, offset: 0, index: 5)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerSelectK8SpecializedPSO : routerSelectK8PSO)
            encoder.setBuffer(routerLogits, offset: 0, index: 0)
            encoder.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
            encoder.setBuffer(outIndices, offset: 0, index: 2)
            encoder.setBuffer(outWeights, offset: 0, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    func encodeRouterMiniMax(commandBuffer: MTLCommandBuffer,
                             weights: MTLBuffer, weightsOffset: Int = 0,
                             scales: MTLBuffer, scalesOffset: Int = 0,
                             biases: MTLBuffer, biasesOffset: Int = 0,
                             hidden: MTLBuffer,
                             effectiveScale: MTLBuffer,
                             correctionBias: MTLBuffer,
                             correctionBiasOffset: Int = 0,
                             outIndices: MTLBuffer,
                             outWeights: MTLBuffer,
                             numExperts: UInt32,
                             d: UInt32,
                             topK: UInt32) {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(Int(numExperts) <= routerLogitCapacity,
                     "router logits hold \(routerLogitCapacity) experts, "
                        + "asked for \(numExperts)")
        precondition(topK == realDecodeTopK)
        var expertCount = numExperts
        var dimension = d
        let useSpecialized = numExperts == realDecodeNumExperts && d == realDecodeD
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            encoder.setBuffer(biases, offset: biasesOffset, index: 2)
            encoder.setBuffer(hidden, offset: 0, index: 3)
            encoder.setBuffer(effectiveScale, offset: 0, index: 4)
            encoder.setBuffer(routerLogits, offset: 0, index: 5)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerSelectMiniMaxK8SpecializedPSO
                    : routerSelectMiniMaxK8PSO)
            encoder.setBuffer(routerLogits, offset: 0, index: 0)
            encoder.setBuffer(correctionBias, offset: correctionBiasOffset, index: 1)
            encoder.setBuffer(outIndices, offset: 0, index: 2)
            encoder.setBuffer(outWeights, offset: 0, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    func makeRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                         topK: UInt32) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs, topK: topK)
        guard let buffer = routedBlobs.first?.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            return nil
        }
        encodeRoutedArgumentBuffer(buffer, routedBlobs: routedBlobs)
        return buffer
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                               topK: UInt32) -> MTLBuffer {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer, routedBlobs: routedBlobs)
        return reusableRoutedArgBuffer
    }

    func encodeRoutedPersistentPhase1U16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        bitWidths: MoEExpertBitWidths = MoEExpertBitWidths()
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f)
                ? phase1U16SpecializedPSO
                : phase1U16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        var widths = bitWidths
        encoder.setBytes(&widths, length: MemoryLayout<MoEExpertBitWidths>.stride, index: 9)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase1SubsetU16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        activeSlots: MTLBuffer,
        activeSlotIndices: [UInt32],
        activeCount: UInt32,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        bitWidths: MoEExpertBitWidths = MoEExpertBitWidths()
    ) {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs, topK: topK)
        precondition(activeSlotIndices.count == Int(activeCount))
        var dimension = d
        var intermediate = f
        var expertCount = topK
        var active = activeCount
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f)
                ? phase1SubsetU16SpecializedPSO
                : phase1SubsetU16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for slot in activeSlotIndices {
            encoder.useResource(routedBlobs[Int(slot)], usage: .read)
        }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(activeSlots, offset: 0, index: 7)
        encoder.setBytes(&active, length: MemoryLayout<UInt32>.stride, index: 8)
        var widths = bitWidths
        encoder.setBytes(&widths, length: MemoryLayout<MoEExpertBitWidths>.stride, index: 9)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase2Reduce(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        routingWeights: MTLBuffer,
        residual: MTLBuffer,
        y: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        bitWidths: MoEExpertBitWidths = MoEExpertBitWidths()
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f)
                ? phase2ReduceK8SpecializedPSO
                : phase2ReduceK8PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        var widths = bitWidths
        encoder.setBytes(&widths, length: MemoryLayout<MoEExpertBitWidths>.stride, index: 8)
        // One SIMD group per routed expert: the kernel indexes `partial` and
        // the blob array by `sg_idx`, so the threadgroup has to be exactly as
        // wide as the selection. This was 256 — eight SIMD groups — for as
        // long as eight was the only routed count, and at ten it left
        // `partial[8]` and `partial[9]` holding whatever the threadgroup
        // memory happened to contain while experts eight and nine were never
        // computed at all.
        let simdWidth = 32
        let threads = Int(topKExpertCount) * simdWidth
        precondition(threads <= phase2ReduceK8PSO.maxTotalThreadsPerThreadgroup,
                     "routed top-k \(topKExpertCount) needs \(threads) threads")
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Routed experts these pipelines were specialized for.
    var topKExpertCount: UInt32 { realDecodeTopK }

    private func validate(routedBlobs: [MTLBuffer], topK: UInt32) {
        precondition(topK == realDecodeTopK)
        precondition(routedBlobs.count == Int(topK))
    }

    private func encodeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                            routedBlobs: [MTLBuffer]) {
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(blob, offset: 0, index: index)
        }
    }

    private func useRealDecodeConstants(d: UInt32, f: UInt32) -> Bool {
        d == realDecodeD && f == realDecodeF
    }
}
