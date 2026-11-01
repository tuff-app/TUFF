import Foundation
import Testing
import TUFFFormat
@testable import TUFFRepackCore

@Suite struct FlintRepackPlannerTests {
    @Test func bytePackedExpertsAreSlicedWithoutMaterializingBanks() throws {
        let arch = ArchInfo(hiddenSize: 64, intermediateSize: 64, moeIntermediateSize: 64,
            numHeads: 2, numKVHeads: 1, numFullKVHeads: 1, headDim: 32, fullHeadDim: 32,
            vocabSize: 64, slidingWindow: 4, finalLogitSoftcap: 0, ropeTheta: 10000,
            fullRopeTheta: 10000, partialRotaryFactor: 1, numLayers: 3, numExperts: 2,
            topKExperts: 1, tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: [1, 1, 0], hiddenActivation: "silu")
        var tensors: [SourceTensor] = []
        var sourceOffset: UInt64 = 100
        func append(_ name: String, _ dtype: SourceTensor.Dtype, _ shape: [UInt64]) {
            let bytes = shape.reduce(UInt64(dtype.elementBytes), *)
            tensors.append(SourceTensor(name: name, shardPath: "fixture.safetensors", dtype: dtype,
                                        shape: shape, absoluteOffset: sourceOffset, sizeBytes: bytes))
            sourceOffset += bytes
        }
        append("model.layers.2.mlp.gate.weight", .bf16, [2, 64])
        append("model.layers.2.self_attn.softmax_offset", .bf16, [2])
        for role in ["gate", "up", "down"] {
            let prefix = "model.layers.2.mlp.switch_mlp.\(role)_proj"
            append(prefix + ".weight", .u8, [2, 64, 32])
            append(prefix + ".scales", .bf16, [2, 64, 1])
            append(prefix + ".biases", .bf16, [2, 64, 1])
        }
        let header = Safetensors.Header(path: "fixture.safetensors", payloadBaseOffset: 0, tensors: tensors)
        let plan = try FlintRepackPlanner.plan(arch: arch, headers: [header], outputDir: "/tmp/flint")
        #expect(plan.layers.count == 3)
        #expect(plan.layers[0].expertsPerLayer == 0 && plan.layers[1].expertsPerLayer == 0)
        #expect(plan.layers[2].expertsPerLayer == 2)
        #expect(plan.layers[2].subTensors.count == 9)
        #expect(plan.layers[2].subTensors[0].sizeInExpertBlob == 64 * 32)
        #expect(plan.layers[2].subTensors[0].sourceOffsetPerExpert == 64 * 32)
        #expect(plan.resident.entries.contains { $0.name == "language_model.model.layers.2.self_attn.sinks" })
        #expect(plan.resident.entries.first { $0.name.hasSuffix("mlp.gate.weight") }?.quantSpec == nil)
        let data = try GTurboJSON.encodeLayout(plan: plan, expertStride: plan.layers[2].expertStride)
        let layout = try GTurboPackedExpertsLayoutCodec.decode(data)
        let sizes = ["packed_experts/layer_02.bin": plan.layers[2].fileSize]
        try GTurboV1StructuralValidator.crossValidate(manifestNumLayers: 3, manifestExpertsPerLayer: 2,
            manifestExpertStride: plan.layers[2].expertStride, manifestFileSizes: sizes,
            layout: layout, densePrefixLayers: 2)
        #expect(throws: (any Error).self) {
            try GTurboV1StructuralValidator.crossValidate(manifestNumLayers: 3, manifestExpertsPerLayer: 2,
                manifestExpertStride: plan.layers[2].expertStride, manifestFileSizes: sizes, layout: layout)
        }
    }

    @Test func mixedWordPackedExpertsRetainTheirWidthsAndLayerStrides() throws {
        let arch = ArchInfo(hiddenSize: 128, intermediateSize: 128, moeIntermediateSize: 128,
            numHeads: 2, numKVHeads: 1, numFullKVHeads: 1, headDim: 32, fullHeadDim: 32,
            vocabSize: 64, slidingWindow: 4, finalLogitSoftcap: 0, ropeTheta: 10000,
            fullRopeTheta: 10000, partialRotaryFactor: 1, numLayers: 4, numExperts: 2,
            topKExperts: 1, tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: [1, 1, 0, 0], hiddenActivation: "silu")
        var tensors: [SourceTensor] = []
        var offset: UInt64 = 0
        func append(_ name: String, _ dtype: SourceTensor.Dtype, _ shape: [UInt64]) {
            let size = shape.reduce(UInt64(dtype.elementBytes), *)
            tensors.append(SourceTensor(name: name, shardPath: "fixture", dtype: dtype,
                                        shape: shape, absoluteOffset: offset, sizeBytes: size))
            offset += size
        }
        append("model.embed_tokens.weight", .u32, [64, 16])
        append("model.embed_tokens.scales", .bf16, [64, 1])
        append("model.embed_tokens.biases", .bf16, [64, 1])
        for layer in 2...3 {
            for role in ["gate", "up", "down"] {
                let stem = "model.layers.\(layer).mlp.switch_mlp.\(role)_proj"
                append(stem + ".weight", .u32, [2, 128, layer == 2 ? 32 : 16])
                append(stem + ".scales", .bf16, [2, 128, 2])
                append(stem + ".biases", .bf16, [2, 128, 2])
            }
        }
        let header = Safetensors.Header(path: "fixture", payloadBaseOffset: 0, tensors: tensors)
        let plan = try FlintRepackPlanner.plan(arch: arch, headers: [header], outputDir: "/tmp/flint")
        #expect(throws: (any Error).self) {
            try FlintRepackPlanner.plan(arch: arch, headers: [header], outputDir: "/tmp/flint",
                                        declaredQuantization: [:])
        }
        #expect(plan.resident.entries[0].quantSpec?.bits == 8)
        #expect(plan.layers[2].subTensors[0].bitsForWeights == 8)
        #expect(plan.layers[3].subTensors[0].bitsForWeights == 4)
        #expect(plan.layers[2].expertStride > plan.layers[3].expertStride)
        let layout = try GTurboPackedExpertsLayoutCodec.decode(
            GTurboJSON.encodeLayout(plan: plan, expertStride: plan.layers[2].expertStride))
        try GTurboV1StructuralValidator.crossValidate(manifestNumLayers: 4, manifestExpertsPerLayer: 2,
            manifestExpertStride: plan.layers[2].expertStride,
            manifestFileSizes: ["packed_experts/layer_02.bin": plan.layers[2].fileSize,
                                "packed_experts/layer_03.bin": plan.layers[3].fileSize],
            layout: layout, densePrefixLayers: 2)
        #expect(throws: (any Error).self) {
            try GTurboV1StructuralValidator.crossValidate(manifestNumLayers: 4, manifestExpertsPerLayer: 2,
                manifestExpertStride: plan.layers[2].expertStride,
                manifestFileSizes: ["packed_experts/layer_02.bin": plan.layers[2].fileSize,
                                    "packed_experts/layer_03.bin": plan.layers[3].fileSize],
                layout: layout)
        }
    }
}
