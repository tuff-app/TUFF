import Foundation
import TUFFFormat
import TUFFRuntimeDescriptions

/// The on-disk contract of an Inkling 2 install: every tensor the runner
/// binds, with the type, shape and byte count its kernels assume. Checked
/// before any weight is mapped, so a damaged or altered install is refused
/// instead of being read out of bounds.
extension ModelRuntimeSchemaValidator {
    static func validateInklingSchema(residentIndex: ResidentIndex,
                                      layout: PackedExpertsLayout,
                                      config: ArchConfig) throws {
        let loop = config.inklingLoop
        guard loop.isEnabled, loop.convKernelSize == 4, config.hiddenSize % 64 == 0,
              config.moeIntermediateSize % 32 == 0 else {
            throw ModelError.indexCorrupt(detail: "Inkling architecture is not one this runtime supports")
        }
        let hidden = config.hiddenSize
        let qDim = config.numHeads * config.headDim
        let kvDim = config.numKVHeads * config.headDim
        let layerPrefix = "language_model.model.layers."

        func entry(_ name: String) throws -> ResidentIndexEntry {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            return entry
        }
        func shapeMatches(_ entry: ResidentIndexEntry, _ shape: [Int]) -> Bool {
            let padded = shape.map(UInt32.init) + Array(repeating: UInt32(0), count: 4 - shape.count)
            return entry.shape.0 == padded[0] && entry.shape.1 == padded[1]
                && entry.shape.2 == padded[2] && entry.shape.3 == padded[3]
        }
        func plain(_ name: String, dtype: GTurboFormatV1.DType, shape: [Int]) throws {
            let found = try entry(name)
            let width: UInt64 = dtype == .fp32 ? 4 : 2
            let elements = shape.reduce(UInt64(1)) { $0 * UInt64($1) }
            guard found.dtype == dtype.rawValue, shapeMatches(found, shape),
                  found.sizeBytes == elements * width,
                  found.scaleOffset == 0, found.scaleSize == 0,
                  found.biasOffset == 0, found.biasSize == 0,
                  found.fileOffset % width == 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required Inkling schema")
            }
        }
        func bf16(_ name: String, _ shape: [Int]) throws { try plain(name, dtype: .bf16, shape: shape) }
        func fp32(_ name: String, _ shape: [Int]) throws { try plain(name, dtype: .fp32, shape: shape) }
        /// Group-64 affine 4-bit: eight codes to a word, BF16 scale and bias.
        func affine(_ name: String, rows: Int, columns: Int) throws {
            let found = try entry(name)
            let groups = UInt64(rows) * UInt64(columns / 64)
            guard columns % 64 == 0, found.dtype == GTurboFormatV1.DType.u32.rawValue,
                  shapeMatches(found, [rows, columns]),
                  found.sizeBytes == UInt64(rows) * UInt64(columns) / 2,
                  found.scaleSize == groups * 2, found.biasSize == groups * 2,
                  found.fileOffset % 4 == 0, found.scaleOffset % 2 == 0, found.biasOffset % 2 == 0,
                  found.scaleOffset != 0, found.biasOffset != 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required Inkling 4-bit schema")
            }
        }

        try affine("language_model.model.embed_tokens.weight", rows: loop.paddedVocabSize, columns: hidden)
        try affine("language_model.lm_head.weight", rows: loop.paddedVocabSize, columns: hidden)
        try bf16("language_model.model.embed_norm.weight", [hidden])
        try bf16("language_model.model.norm.weight", [hidden])
        try fp32("language_model.model.final_input.query", [hidden])
        try fp32("language_model.model.message_input.query", [loop.loopLayers.count, 4, hidden])

        guard layout.layers.count == config.numLayers,
              layout.expertsPerLayer == config.numExperts else {
            throw ModelError.indexCorrupt(detail: "Inkling expert layout does not cover every layer")
        }
        for layer in 0..<config.numLayers {
            let prefix = "\(layerPrefix)\(layer)."
            let attention = prefix + "self_attn."
            let full = config.layerIsFull(layer)
            for (leaf, rows, columns) in [("q_proj", qDim, hidden), ("gate_proj", qDim, hidden),
                                          ("k_proj", kvDim, hidden), ("v_proj", kvDim, hidden),
                                          ("r_proj", config.numHeads * loop.relativeDim, hidden),
                                          ("sink_proj", config.numHeads, hidden), ("o_proj", hidden, qDim)] {
                try affine(attention + leaf + ".weight", rows: rows, columns: columns)
            }
            try bf16(attention + "q_l2_clip.scale", [config.numHeads, config.headDim])
            try bf16(attention + "rel_l2_clip.scale", [config.numHeads, loop.relativeDim])
            try bf16(attention + "rel_logits_proj.proj",
                     [loop.relativeDim, full ? loop.relativeExtent : config.slidingWindow])
            if full {
                for name in ["q", "k", "v"] { try bf16(prefix + "\(name)_input_layernorm.weight", [hidden]) }
                for name in ["k", "v"] {
                    try bf16(attention + "\(name)_sconv.static_weight", [loop.convKernelSize, kvDim])
                    try affine(attention + "\(name)_sconv.weight_proj.weight",
                               rows: config.numKVHeads * loop.convKernelSize, columns: hidden)
                }
            } else {
                try bf16(prefix + "input_layernorm.weight", [hidden])
                for name in ["k", "v"] {
                    try bf16(attention + "\(name)_sconv.conv1d.weight", [kvDim, loop.convKernelSize, 1])
                }
            }
            try bf16(prefix + "post_attention_layernorm.weight", [hidden])
            try bf16(prefix + "post_attention_convolution.conv1d.weight", [hidden, loop.convKernelSize, 1])
            try bf16(prefix + "post_mlp_convolution.conv1d.weight", [hidden, loop.convKernelSize, 1])
            try fp32(prefix + "mlp_input.query", [hidden])
            if layer > 0 {
                try fp32(prefix + "attention_input.query", full ? [3, hidden] : [hidden])
            }
            try fp32(prefix + "mlp.gate.weight", [config.numExperts, hidden])
            try fp32(prefix + "mlp.gate.e_score_correction_bias", [config.numExperts])
            if loop.loopLayers.contains(layer) {
                for name in ["q", "k", "v", "mlp"] {
                    let gate = prefix + "\(name)_message_gate."
                    try bf16(gate + "down_proj.weight", [loop.gateRank, 2 * hidden])
                    try fp32(gate + "forget_gate_proj.weight", [hidden, loop.gateRank])
                    try fp32(gate + "message_gate_proj.weight", [hidden, loop.gateRank])
                }
            }

            // Experts: an interleaved gate/up matrix and a down matrix, in
            // NVFP4 (with a resident factor per expert) or MXFP8 (without).
            guard let found = layout.layers.first(where: { $0.layer == layer }),
                  found.experts.count == config.numExperts,
                  let reference = found.experts.first else {
                throw ModelError.indexCorrupt(detail: "Inkling layer \(layer) has no routed experts")
            }
            let bits = reference.subTensors["gate_up"]?.bits ?? 0
            guard bits == 4 || bits == 8 else {
                throw ModelError.indexCorrupt(detail: "Inkling layer \(layer) has an unsupported expert format")
            }
            let group = bits == 4 ? 16 : 32
            let factors = ["gate_up_proj", "down_proj"].map { prefix + "mlp.experts.\($0).global_scale" }
            if bits == 4 {
                for name in factors { try fp32(name, [config.numExperts]) }
            } else {
                guard factors.allSatisfy({ residentIndex.entries[$0] == nil }) else {
                    throw ModelError.indexCorrupt(detail: "Inkling layer \(layer) has factors for an MXFP8 expert")
                }
            }
            let intermediate = config.moeIntermediateSize
            let expected: [(String, String, [Int], Int?, Int, UInt64)] = [
                ("gate_up", "U32", [2 * intermediate, hidden], bits, 2 * intermediate * hidden * bits / 8, 4),
                ("gate_up_scales", "U8", [2 * intermediate, hidden / group], nil, 2 * intermediate * hidden / group, 1),
                ("down", "U32", [hidden, intermediate], bits, hidden * intermediate * bits / 8, 4),
                ("down_scales", "U8", [hidden, intermediate / group], nil, hidden * intermediate / group, 1),
            ]
            guard reference.subTensors.count == expected.count else {
                throw ModelError.indexCorrupt(detail: "Inkling layer \(layer) has unexpected expert tensors")
            }
            var ranges: [Range<UInt64>] = []
            for (name, dtype, shape, tensorBits, size, alignment) in expected {
                guard let tensor = reference.subTensors[name] else {
                    throw ModelError.indexCorrupt(detail: "Inkling layer \(layer) is missing expert tensor \(name)")
                }
                let (end, overflow) = tensor.offset.addingReportingOverflow(tensor.size)
                guard tensor.dtype == dtype, tensor.shape == shape.map(UInt32.init), tensor.bits == tensorBits,
                      tensor.size == UInt64(size), tensor.offset % alignment == 0, !overflow,
                      end <= reference.size, end <= UInt64(UInt32.max) + 1 else {
                    throw ModelError.indexCorrupt(
                        detail: "Inkling layer \(layer) expert tensor \(name) does not match the required schema")
                }
                ranges.append(tensor.offset..<end)
                for expert in found.experts.dropFirst() where expert.subTensors[name] != tensor {
                    throw ModelError.indexCorrupt(
                        detail: "Inkling layer \(layer) expert tensor \(name) differs across experts")
                }
            }
            ranges.sort { $0.lowerBound < $1.lowerBound }
            for (a, b) in zip(ranges, ranges.dropFirst()) where a.upperBound > b.lowerBound {
                throw ModelError.indexCorrupt(detail: "Inkling layer \(layer) expert tensors overlap")
            }
            guard found.experts.allSatisfy({ $0.size == reference.size }),
                  reference.size <= layout.expertStride else {
                throw ModelError.indexCorrupt(detail: "Inkling layer \(layer) experts are not one size")
            }
        }
    }
}
