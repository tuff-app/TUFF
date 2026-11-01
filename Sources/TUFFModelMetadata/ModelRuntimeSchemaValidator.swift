import Foundation
import Darwin
import TUFFRuntimeDescriptions
import TUFFFormat

/// The same metadata contract enforced by model loading, without allocating GPU weights.
public enum ModelRuntimeSchemaValidator {
    public static func validate(directory: URL, expecting config: ArchConfig) throws {
        let access = try GTurboModelDirectory(rootURL: directory)
        let manifest = try ManifestReader.load(directoryURL: directory, expecting: config)
        let fd = try access.openFile("model_weights.bin")
        defer { close(fd) }
        let index = try ResidentIndexReader.load(fileDescriptor: fd, displayPath: "model_weights.bin")
        let layout = config.feedForwardKind == .dense ? PackedExpertsLayout.dense(numLayers: config.numLayers)
            : try PackedExpertsLayoutReader.load(directoryURL: directory, manifest: manifest)
        try validateRuntimeSchema(residentIndex: index, layout: layout, manifest: manifest, config: config)
    }

    package static func validateRuntimeSchema(residentIndex: ResidentIndex,
                                      layout: PackedExpertsLayout,
                                      manifest: Manifest,
                                      config: ArchConfig) throws {
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required by the executable runtime schema")
        }
        let isPrism = config.variant == .bonsai2_2B || config.variant == .bonsai2_27B

        func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) byte count overflows UInt64")
            }
            return value
        }

        func checkedIntMultiply(_ lhs: Int, _ rhs: Int, field: String) throws -> Int {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) dimension overflows Int")
            }
            return value
        }

        func dimensions(_ rows: Int, _ columns: Int, field: String) throws -> (UInt32, UInt32) {
            guard let r = UInt32(exactly: rows), let c = UInt32(exactly: columns),
                  r > 0, c > 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) has invalid dimensions")
            }
            return (r, c)
        }

        /// The n-gram hash constants: small int64 sidecars read as values
        /// rather than decoded as weights.
        func requireInt64(_ name: String, _ count: Int) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(
                    detail: "missing required resident tensor \(name)")
            }
            guard let logicalCount = UInt32(exactly: count), logicalCount > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            let expectedBytes = try checkedMultiply(
                UInt64(logicalCount), UInt64(MemoryLayout<Int64>.size), field: name)
            guard entry.dtype == GTurboFormatV1.DType.i64.rawValue,
                  entry.shape.0 == logicalCount,
                  entry.sizeBytes == expectedBytes,
                  entry.scaleSize == 0, entry.biasSize == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) does not match the required int64 schema")
            }
        }

        func requireBF16(_ name: String, count: Int) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            guard let logicalCount = UInt32(exactly: count), logicalCount > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            let expectedBytes = try checkedMultiply(
                UInt64(logicalCount), UInt64(MemoryLayout<UInt16>.size), field: name)
            guard entry.dtype == GTurboFormatV1.DType.bf16.rawValue,
                  entry.shape.0 == logicalCount,
                  entry.shape.1 == 0, entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required BF16 schema")
            }
        }

        func requireFP32(_ name: String, count: Int) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            guard let logicalCount = UInt32(exactly: count), logicalCount > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            let expectedBytes = try checkedMultiply(
                UInt64(logicalCount), UInt64(MemoryLayout<Float>.size), field: name)
            guard entry.dtype == GTurboFormatV1.DType.fp32.rawValue,
                  entry.shape.0 == logicalCount,
                  entry.shape.1 == 0, entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(MemoryLayout<Float>.alignment) == 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required FP32 schema")
            }
        }

        func affineSizes(rows: Int,
                         columns: Int,
                         slot: ManifestQuantSlot,
                         field: String) throws -> (shape: (UInt32, UInt32), weight: UInt64, aux: UInt64) {
            let shape = try dimensions(rows, columns, field: field)
            guard slot.weightBits == 4 || slot.weightBits == 8 || (isPrism && slot.weightBits == 2),
                  slot.effectiveGroupSize > 0,
                  columns % slot.effectiveGroupSize == 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) has unsupported affine quantization")
            }
            let elements = try checkedMultiply(UInt64(rows), UInt64(columns), field: field)
            let bitCount = try checkedMultiply(elements, UInt64(slot.weightBits), field: field)
            guard bitCount % 8 == 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) packed byte count is fractional")
            }
            let groups = UInt64(columns / slot.effectiveGroupSize)
            let auxElements = try checkedMultiply(UInt64(shape.0), groups, field: field)
            let auxBytes = try checkedMultiply(
                auxElements, UInt64(MemoryLayout<UInt16>.size), field: field)
            return (shape, bitCount / 8, auxBytes)
        }

        func requireAffine(_ name: String,
                           rows: Int,
                           columns: Int,
                           slot: ManifestQuantSlot) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            let effectiveSlot: ManifestQuantSlot
            if config.family == .flint {
                let elements = try checkedMultiply(UInt64(rows), UInt64(columns), field: name)
                let bits = entry.sizeBytes == elements ? 8 : 4
                effectiveSlot = ManifestQuantSlot(weightBits: bits, scheme: slot.scheme,
                    scaleType: slot.scaleType, biasType: slot.biasType, groupSize: slot.groupSize)
            } else {
                effectiveSlot = slot
            }
            let expected = try affineSizes(
                rows: rows, columns: columns, slot: effectiveSlot, field: name)
            let primaryAlignment: UInt64 = effectiveSlot.weightBits == 2 ? 4 :
                (effectiveSlot.weightBits == 4 ? UInt64(MemoryLayout<UInt16>.alignment) : 1)
            guard entry.dtype == GTurboFormatV1.DType.u32.rawValue,
                  entry.shape.0 == expected.shape.0,
                  entry.shape.1 == expected.shape.1,
                  entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expected.weight,
                  entry.scaleSize == expected.aux,
                  entry.biasSize == expected.aux,
                  entry.fileOffset % primaryAlignment == 0,
                  entry.scaleOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0,
                  entry.biasOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) affine metadata mismatch: dtype=\(entry.dtype), shape=[\(entry.shape.0),\(entry.shape.1),\(entry.shape.2),\(entry.shape.3)], bytes=\(entry.sizeBytes), scales=\(entry.scaleSize), biases=\(entry.biasSize), expected shape=[\(expected.shape.0),\(expected.shape.1),0,0], bytes=\(expected.weight), aux=\(expected.aux)")
            }
            if isPrism {
                guard name.hasSuffix(".weight") else {
                    throw ModelError.indexCorrupt(detail: "invalid Prism projection name")
                }
                try requireFP32(String(name.dropLast(".weight".count)) + ".signs", count: columns)
            }
        }

        func requireFP32Shaped(_ name: String, shape: [UInt32]) throws {
            guard shape.count <= 4, !shape.isEmpty, shape.allSatisfy({ $0 > 0 }),
                  let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing or invalid FP32 tensor \(name)")
            }
            let padded = shape + Array(repeating: UInt32(0), count: 4 - shape.count)
            var elements: UInt64 = 1
            for dimension in shape { elements = try checkedMultiply(elements, UInt64(dimension), field: name) }
            let bytes = try checkedMultiply(elements, 4, field: name)
            guard entry.dtype == GTurboFormatV1.DType.fp32.rawValue,
                  entry.shape.0 == padded[0], entry.shape.1 == padded[1],
                  entry.shape.2 == padded[2], entry.shape.3 == padded[3],
                  entry.sizeBytes == bytes, entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0, entry.fileOffset % 4 == 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required FP32 schema")
            }
        }

        /// BF16 tensor with a non-vector logical shape (Qwen's depthwise conv
        /// kernel is the only one). `requireBF16` insists on `[n, 0, 0, 0]`.
        func requireBF16Shaped(_ name: String, shape: [UInt32]) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            let padded = shape + Array(repeating: UInt32(0), count: max(0, 4 - shape.count))
            let elements = shape.reduce(UInt64(1)) { $0 * UInt64(max($1, 1)) }
            let expectedBytes = try checkedMultiply(
                elements, UInt64(MemoryLayout<UInt16>.size), field: name)
            guard entry.dtype == GTurboFormatV1.DType.bf16.rawValue,
                  entry.shape.0 == padded[0], entry.shape.1 == padded[1],
                  entry.shape.2 == padded[2], entry.shape.3 == padded[3],
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) does not match the required BF16 schema")
            }
        }

        if config.family == .gptOss {
            try requireBF16Shaped(
                "language_model.model.embed_tokens.weight",
                shape: [UInt32(config.vocabSize), UInt32(config.hiddenSize)])
        } else {
            try requireAffine(
                "language_model.model.embed_tokens.weight",
                rows: config.vocabSize,
                columns: config.hiddenSize,
                slot: quant.embedding)
        }
        if config.hyperConnection.isEnabled {
            // No `model.norm`: the mixer that collapses the residual streams
            // sits in that position and carries the last normalization. It has
            // no injection gate, so only three of the four tensors.
            let stacked = config.hyperConnection.stackedWidth(
                hiddenSize: config.hiddenSize)
            let mixer = "language_model.model.hyper_connection_mixer"
            try requireBF16("\(mixer).hc_norm.weight", count: stacked)
            try requireAffine("\(mixer).input_mix_weight_down.weight",
                              rows: config.hyperConnection.lowRank,
                              columns: stacked,
                              slot: quant.attention)
            try requireAffine("\(mixer).input_mix_weight_up.weight",
                              rows: stacked,
                              columns: config.hyperConnection.lowRank,
                              slot: quant.attention)
        } else if isPrism {
            try requireFP32("language_model.model.norm.weight", count: config.hiddenSize)
        } else {
            try requireBF16("language_model.model.norm.weight",
                            count: config.hiddenSize)
        }
        // Gemma ties lm_head to the embedding; an untied family carries its own.
        if !config.tieWordEmbeddings {
            if config.family == .gptOss {
                try requireBF16Shaped(
                    "language_model.lm_head.weight",
                    shape: [UInt32(config.vocabSize), UInt32(config.hiddenSize)])
            } else {
                try requireAffine("language_model.lm_head.weight",
                                  rows: config.vocabSize,
                                  columns: config.hiddenSize,
                                  slot: config.family == .flint ? quant.attention : quant.embedding)
            }
        }

        // Dense Gemma has no router or packed-expert files. Its resident
        // contract also includes the packed PLE tables and per-layer mapping.
        if config.feedForwardKind == .dense && config.family == .gemma4 {
            try validateDenseGemma4LayerSchema(
                config: config, quant: quant,
                requireBF16: requireBF16,
                requireAffine: requireAffine,
                checkedIntMultiply: checkedIntMultiply)
        } else if config.family == .gptOss {
            for layer in 0..<config.numLayers {
                let prefix = "language_model.model.layers.\(layer)"
                try requireBF16("\(prefix).input_layernorm.weight",
                                count: config.hiddenSize)
                try requireBF16("\(prefix).post_attention_layernorm.weight",
                                count: config.hiddenSize)
                try requireBF16Shaped(
                    "\(prefix).self_attn.q_proj.weight",
                    shape: [UInt32(config.numHeads * config.headDim),
                            UInt32(config.hiddenSize)])
                try requireBF16("\(prefix).self_attn.q_proj.bias",
                                count: config.numHeads * config.headDim)
                for projection in ["k_proj", "v_proj"] {
                    try requireBF16Shaped(
                        "\(prefix).self_attn.\(projection).weight",
                        shape: [UInt32(config.numKVHeads * config.headDim),
                                UInt32(config.hiddenSize)])
                    try requireBF16("\(prefix).self_attn.\(projection).bias",
                                    count: config.numKVHeads * config.headDim)
                }
                try requireBF16Shaped(
                    "\(prefix).self_attn.o_proj.weight",
                    shape: [UInt32(config.hiddenSize),
                            UInt32(config.numHeads * config.headDim)])
                try requireBF16("\(prefix).self_attn.o_proj.bias",
                                count: config.hiddenSize)
                try requireBF16("\(prefix).self_attn.sinks",
                                count: config.numHeads)
                try requireBF16Shaped(
                    "\(prefix).mlp.router.weight",
                    shape: [UInt32(config.numExperts), UInt32(config.hiddenSize)])
                try requireBF16("\(prefix).mlp.router.bias",
                                count: config.numExperts)
            }
        } else if config.family == .flint {
            for layer in 0..<config.numLayers {
                let prefix = "language_model.model.layers.\(layer)"
                try requireBF16("\(prefix).input_layernorm.weight", count: config.hiddenSize)
                try requireBF16("\(prefix).post_attention_layernorm.weight", count: config.hiddenSize)
                try requireBF16("\(prefix).self_attn.q_norm.weight", count: config.headDim)
                try requireBF16("\(prefix).self_attn.k_norm.weight", count: config.headDim)
                try requireBF16("\(prefix).self_attn.sinks", count: config.numHeads)
                for (name, rows) in [("q_proj", config.numHeads * config.headDim),
                                     ("k_proj", config.numKVHeads * config.headDim),
                                     ("v_proj", config.numKVHeads * config.headDim)] {
                    try requireAffine("\(prefix).self_attn.\(name).weight", rows: rows,
                                      columns: config.hiddenSize, slot: quant.attention)
                }
                try requireAffine("\(prefix).self_attn.o_proj.weight", rows: config.hiddenSize,
                                  columns: config.numHeads * config.headDim, slot: quant.attention)
                let stem = layer < 2 ? "\(prefix).mlp" : "\(prefix).mlp.shared_expert"
                let width = config.ffnIntermediateSize(layer: layer)
                for name in ["gate_proj", "up_proj"] {
                    try requireAffine("\(stem).\(name).weight", rows: width,
                                      columns: config.hiddenSize, slot: quant.sharedExpert)
                }
                try requireAffine("\(stem).down_proj.weight", rows: config.hiddenSize,
                                  columns: width, slot: quant.sharedExpert)
                if layer >= 2 {
                    let name = "\(prefix).mlp.gate.weight"
                    if residentIndex.entries[name]?.dtype == GTurboFormatV1.DType.bf16.rawValue {
                        try requireBF16Shaped(name,
                            shape: [UInt32(config.numExperts), UInt32(config.hiddenSize)])
                    } else {
                        try requireAffine(name, rows: config.numExperts,
                                          columns: config.hiddenSize, slot: quant.router)
                    }
                }
            }
        } else if isPrism {
            let linear = config.linearAttention
            for layer in 0..<config.numLayers {
                let p = "language_model.model.layers.\(layer)"
                try requireFP32("\(p).input_layernorm.weight", count: config.hiddenSize)
                try requireFP32("\(p).post_attention_layernorm.weight", count: config.hiddenSize)
                for name in ["gate_proj", "up_proj"] {
                    try requireAffine("\(p).mlp.\(name).weight", rows: config.intermediateSize,
                                      columns: config.hiddenSize, slot: quant.sharedExpert)
                }
                try requireAffine("\(p).mlp.down_proj.weight", rows: config.hiddenSize,
                                  columns: config.intermediateSize, slot: quant.sharedExpert)
                if config.layerIsLinear(layer) {
                    try requireAffine("\(p).linear_attn.in_proj_qkv.weight", rows: linear.qkvDim,
                                      columns: config.hiddenSize, slot: quant.attention)
                    try requireAffine("\(p).linear_attn.in_proj_z.weight", rows: linear.valueDim,
                                      columns: config.hiddenSize, slot: quant.attention)
                    try requireAffine("\(p).linear_attn.out_proj.weight", rows: config.hiddenSize,
                                      columns: linear.valueDim, slot: quant.attention)
                    for name in ["in_proj_a", "in_proj_b"] {
                        try requireFP32Shaped("\(p).linear_attn.\(name).weight",
                            shape: [UInt32(linear.numVHeads), UInt32(config.hiddenSize)])
                    }
                    try requireFP32Shaped("\(p).linear_attn.conv1d.weight",
                        shape: [UInt32(linear.qkvDim), UInt32(linear.convKernelSize), 1])
                    try requireFP32("\(p).linear_attn.A_log", count: linear.numVHeads)
                    try requireFP32("\(p).linear_attn.dt_bias", count: linear.numVHeads)
                    try requireFP32("\(p).linear_attn.norm.weight", count: linear.valueHeadDim)
                } else {
                    let q = config.numHeads * config.fullHeadDim
                    let kv = config.numFullKVHeads * config.fullHeadDim
                    try requireAffine("\(p).self_attn.q_proj.weight", rows: 2 * q,
                                      columns: config.hiddenSize, slot: quant.attention)
                    for name in ["k_proj", "v_proj"] {
                        try requireAffine("\(p).self_attn.\(name).weight", rows: kv,
                                          columns: config.hiddenSize, slot: quant.attention)
                    }
                    try requireAffine("\(p).self_attn.o_proj.weight", rows: config.hiddenSize,
                                      columns: q, slot: quant.attention)
                    try requireFP32("\(p).self_attn.q_norm.weight", count: config.fullHeadDim)
                    try requireFP32("\(p).self_attn.k_norm.weight", count: config.fullHeadDim)
                }
            }
        } else if config.family == .qwen36 || config.family == .qwen4Exp {
            // Qwen4-Exp shares Qwen3.6's per-layer tensor contract; what it
            // replaces is the pair of layer norms, which the schema below
            // swaps for the two hyper-connection modules.
            try validateQwen36LayerSchema(
                config: config, quant: quant,
                requireInt64: requireInt64,
                requireBF16: requireBF16,
                requireBF16Shaped: requireBF16Shaped,
                requireAffine: requireAffine,
                checkedIntMultiply: checkedIntMultiply)
        } else if config.family == .minimaxM2 {
            try validateMiniMaxM27LayerSchema(
                config: config, quant: quant,
                requireBF16: requireBF16,
                requireFP32: requireFP32,
                requireAffine: requireAffine,
                checkedIntMultiply: checkedIntMultiply)
        } else {
        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let headDimension = isFull ? config.fullHeadDim : config.headDim
            let kvHeads = isFull ? config.numFullKVHeads : config.numKVHeads
            let queryDimension = try checkedIntMultiply(
                config.numHeads, headDimension, field: "layer \(layer) query")
            let kvDimension = try checkedIntMultiply(
                kvHeads, headDimension, field: "layer \(layer) key/value")

            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "pre_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm.weight",
                "router.scale",
            ] {
                try requireBF16("\(prefix).\(name)", count: config.hiddenSize)
            }
            try requireBF16("\(prefix).self_attn.q_norm.weight", count: headDimension)
            try requireBF16("\(prefix).self_attn.k_norm.weight", count: headDimension)
            try requireBF16("\(prefix).router.per_expert_scale", count: config.numExperts)
            try requireBF16("\(prefix).layer_scalar", count: 1)

            try requireAffine("\(prefix).self_attn.q_proj.weight",
                              rows: queryDimension, columns: config.hiddenSize,
                              slot: quant.attention)
            try requireAffine("\(prefix).self_attn.k_proj.weight",
                              rows: kvDimension, columns: config.hiddenSize,
                              slot: quant.attention)
            if !isFull {
                try requireAffine("\(prefix).self_attn.v_proj.weight",
                                  rows: kvDimension, columns: config.hiddenSize,
                                  slot: quant.attention)
            }
            try requireAffine("\(prefix).self_attn.o_proj.weight",
                              rows: config.hiddenSize, columns: queryDimension,
                              slot: quant.attention)
            let ffn = config.ffnIntermediateSize(layer: layer)
            try requireAffine("\(prefix).mlp.gate_proj.weight",
                              rows: ffn, columns: config.hiddenSize,
                              slot: quant.sharedExpert)
            try requireAffine("\(prefix).mlp.up_proj.weight",
                              rows: ffn, columns: config.hiddenSize,
                              slot: quant.sharedExpert)
            try requireAffine("\(prefix).mlp.down_proj.weight",
                              rows: config.hiddenSize, columns: ffn,
                              slot: quant.sharedExpert)
            try requireAffine("\(prefix).router.proj.weight",
                              rows: config.numExperts, columns: config.hiddenSize,
                              slot: quant.router)
        }
        }

        guard config.feedForwardKind == .mixtureOfExperts else {
            guard layout.layers.isEmpty,
                  layout.expertsPerLayer == 0,
                  layout.expertStride == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "dense architecture must not carry routed experts")
            }
            return
        }

        if config.family == .gptOss {
            try validateGPTOSSRoutedLayout(config: config, layout: layout)
            return
        }

        let routedShapes: [(String, Int, Int)] = [
            ("gate", config.moeIntermediateSize, config.hiddenSize),
            ("up", config.moeIntermediateSize, config.hiddenSize),
            ("down", config.hiddenSize, config.moeIntermediateSize),
        ]
        for layer in layout.layers {
            if config.family == .flint && layer.layer < 2 {
                guard layer.experts.isEmpty else {
                    throw ModelError.indexCorrupt(detail: "Flint dense layer has routed experts")
                }
                continue
            }
            guard let reference = layer.experts.first else {
                throw ModelError.indexCorrupt(
                    detail: "routed layer \(layer.layer) has no experts")
            }
            for (role, rows, columns) in routedShapes {
                let bits = config.family == .flint
                    ? reference.subTensors[role]?.bits ?? quant.routedExpert.weightBits
                    : quant.routedExpert.weightBits
                let routedSlot = ManifestQuantSlot(weightBits: bits,
                    scheme: quant.routedExpert.scheme, scaleType: quant.routedExpert.scaleType,
                    biasType: quant.routedExpert.biasType, groupSize: quant.routedExpert.groupSize)
                let sizes = try affineSizes(
                    rows: rows, columns: columns, slot: routedSlot,
                    field: "routed layer \(layer.layer) \(role)")
                let expectedRoles: [(String, String, [UInt32], Int?, UInt64, UInt64)] = [
                    (role, "U32", [sizes.shape.0, sizes.shape.1],
                     bits, sizes.weight,
                     UInt64(MemoryLayout<UInt32>.alignment)),
                    ("\(role)_scales", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                    ("\(role)_biases", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                ]
                for (name, dtype, shape, bits, size, alignment) in expectedRoles {
                    guard let expected = reference.subTensors[name] else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) is missing role \(name)")
                    }
                    let (end, overflow) = expected.offset.addingReportingOverflow(expected.size)
                    guard expected.dtype == dtype,
                          expected.shape == shape,
                          expected.bits == bits,
                          expected.size == size,
                          expected.offset % alignment == 0,
                          !overflow,
                          end <= reference.size,
                          end <= UInt64(UInt32.max) + 1 else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) does not match the required schema")
                    }
                    for expert in layer.experts.dropFirst()
                        where expert.subTensors[name] != expected {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) metadata differs across experts")
                    }
                }
            }
        }
    }

    private static func validateGPTOSSRoutedLayout(
        config: ArchConfig,
        layout: PackedExpertsLayout
    ) throws {
        let hidden = config.hiddenSize
        let intermediate = config.moeIntermediateSize
        let expected: [(String, String, [UInt32], Int?, UInt64, UInt64)] = [
            ("mlp1", "U8", [UInt32(2 * intermediate), UInt32(hidden)], 4,
             UInt64(2 * intermediate * hidden / 2), 1),
            ("mlp1_scales", "U8",
             [UInt32(2 * intermediate), UInt32(hidden / 32)], nil,
             UInt64(2 * intermediate * hidden / 32), 1),
            ("mlp1_bias", "BF16", [UInt32(2 * intermediate)], nil,
             UInt64(2 * intermediate * MemoryLayout<UInt16>.stride), 2),
            ("mlp2", "U8", [UInt32(hidden), UInt32(intermediate)], 4,
             UInt64(hidden * intermediate / 2), 1),
            ("mlp2_scales", "U8",
             [UInt32(hidden), UInt32(intermediate / 32)], nil,
             UInt64(hidden * intermediate / 32), 1),
            ("mlp2_bias", "BF16", [UInt32(hidden)], nil,
             UInt64(hidden * MemoryLayout<UInt16>.stride), 2),
        ]
        guard layout.numLayers == config.numLayers,
              layout.expertsPerLayer == config.numExperts else {
            throw ModelError.indexCorrupt(
                detail: "GPT-OSS routed layout dimensions do not match the architecture")
        }
        for layer in layout.layers {
            guard let reference = layer.experts.first else {
                throw ModelError.indexCorrupt(
                    detail: "GPT-OSS routed layer \(layer.layer) has no experts")
            }
            for (name, dtype, shape, bits, size, alignment) in expected {
                guard let tensor = reference.subTensors[name] else {
                    throw ModelError.indexCorrupt(
                        detail: "GPT-OSS routed layer \(layer.layer) is missing \(name)")
                }
                let (end, overflow) = tensor.offset.addingReportingOverflow(tensor.size)
                guard tensor.dtype == dtype,
                      tensor.shape == shape,
                      tensor.bits == bits,
                      tensor.size == size,
                      tensor.offset % alignment == 0,
                      !overflow,
                      end <= reference.size else {
                    throw ModelError.indexCorrupt(
                        detail: "GPT-OSS routed layer \(layer.layer) role \(name) does not match the required schema")
                }
                for expert in layer.experts.dropFirst()
                    where expert.subTensors[name] != tensor {
                    throw ModelError.indexCorrupt(
                        detail: "GPT-OSS routed layer \(layer.layer) role \(name) differs across experts")
                }
            }
        }
    }

    private static func validateDenseGemma4LayerSchema(
        config: ArchConfig,
        quant: ManifestQuant,
        requireBF16: (String, Int) throws -> Void,
        requireAffine: (String, Int, Int, ManifestQuantSlot) throws -> Void,
        checkedIntMultiply: (Int, Int, String) throws -> Int
    ) throws {
        if config.hasPerLayerInputs {
            let packedPLE = try checkedIntMultiply(
                config.numLayers, config.hiddenSizePerLayerInput, "packed PLE width")
            try requireAffine("language_model.model.embed_tokens_per_layer.weight",
                              config.vocabSizePerLayerInput, packedPLE, quant.embedding)
            try requireAffine("language_model.model.per_layer_model_projection.weight",
                              packedPLE, config.hiddenSize, quant.sharedExpert)
            try requireBF16("language_model.model.per_layer_projection_norm.weight",
                            config.hiddenSizePerLayerInput)
        }

        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let isFull = config.layerIsFull(layer)
            let headDimension = isFull ? config.fullHeadDim : config.headDim
            let kvHeads = isFull ? config.numFullKVHeads : config.numKVHeads
            let queryDimension = try checkedIntMultiply(
                config.numHeads, headDimension, "layer \(layer) query")
            let kvDimension = try checkedIntMultiply(
                kvHeads, headDimension, "layer \(layer) key/value")

            var normNames = [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ]
            if config.hasPerLayerInputs {
                normNames.append("post_per_layer_input_norm.weight")
            }
            for name in normNames {
                try requireBF16("\(prefix).\(name)", config.hiddenSize)
            }
            try requireBF16("\(prefix).self_attn.q_norm.weight", headDimension)
            try requireBF16("\(prefix).layer_scalar", 1)
            try requireAffine("\(prefix).self_attn.q_proj.weight",
                              queryDimension, config.hiddenSize, quant.attention)

            if !config.layerSharesKV(layer) {
                try requireAffine("\(prefix).self_attn.k_proj.weight",
                                  kvDimension, config.hiddenSize, quant.attention)
                if !isFull || !config.attentionKEqV {
                    try requireAffine("\(prefix).self_attn.v_proj.weight",
                                      kvDimension, config.hiddenSize, quant.attention)
                }
                try requireBF16("\(prefix).self_attn.k_norm.weight", headDimension)
            }
            try requireAffine("\(prefix).self_attn.o_proj.weight",
                              config.hiddenSize, queryDimension, quant.attention)
            let ffn = config.ffnIntermediateSize(layer: layer)
            try requireAffine("\(prefix).mlp.gate_proj.weight",
                              ffn, config.hiddenSize, quant.sharedExpert)
            try requireAffine("\(prefix).mlp.up_proj.weight",
                              ffn, config.hiddenSize, quant.sharedExpert)
            try requireAffine("\(prefix).mlp.down_proj.weight",
                              config.hiddenSize, ffn, quant.sharedExpert)
            if config.hasPerLayerInputs {
                try requireAffine("\(prefix).per_layer_input_gate.weight",
                                  config.hiddenSizePerLayerInput, config.hiddenSize,
                                  quant.sharedExpert)
                try requireAffine("\(prefix).per_layer_projection.weight",
                                  config.hiddenSize, config.hiddenSizePerLayerInput,
                                  quant.sharedExpert)
            }
        }
    }

    /// Per-layer resident contract for Qwen 3.6.
    ///
    /// The layer graph is hybrid, so the required tensors depend on the layer
    /// kind: mask-2 layers carry the gated-DeltaNet block (`linear_attn.*`) and
    /// no KV projections, mask-1 layers carry gated full attention with a
    /// packed `[query ; gate]` q_proj. Neither kind has Gemma's sandwich norms,
    /// router scales, or per-layer scalar.
    private static func validateQwen36LayerSchema(
        config: ArchConfig,
        quant: ManifestQuant,
        requireInt64: (String, Int) throws -> Void,
        requireBF16: (String, Int) throws -> Void,
        requireBF16Shaped: (String, [UInt32]) throws -> Void,
        requireAffine: (String, Int, Int, ManifestQuantSlot) throws -> Void,
        checkedIntMultiply: (Int, Int, String) throws -> Int
    ) throws {
        let linear = config.linearAttention
        let hyper = config.hyperConnection
        let stacked = hyper.stackedWidth(hiddenSize: config.hiddenSize)
        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            if hyper.isEnabled {
                // Two hyper-connections stand where the two layer norms were,
                // each with its own grouped norm, low-rank mix and per-stream
                // injection gate.
                for slot in ["attn_hyper_connection", "mlp_hyper_connection"] {
                    let module = "\(prefix).\(slot)"
                    try requireBF16("\(module).hc_norm.weight", stacked)
                    try requireAffine("\(module).input_mix_weight_down.weight",
                                      hyper.lowRank, stacked, quant.attention)
                    try requireAffine("\(module).input_mix_weight_up.weight",
                                      stacked, hyper.lowRank, quant.attention)
                    try requireAffine("\(module).block_inject_weight.weight",
                                      hyper.streamCount, stacked, quant.attention)
                }
            } else {
                try requireBF16("\(prefix).input_layernorm.weight", config.hiddenSize)
                try requireBF16("\(prefix).post_attention_layernorm.weight",
                                config.hiddenSize)
            }

            // The n-gram module, on the one layer that carries it. Checked
            // here so a checkpoint missing a piece of it fails at load rather
            // than partway through the first token.
            let ngram = config.ngramEmbedding
            if ngram.isEnabled && layer == ngram.layer {
                let ple = "\(prefix).ple"
                try requireAffine("\(ple).key_proj.weight",
                                  stacked, ngram.embedDim, quant.attention)
                try requireAffine("\(ple).value_proj.weight",
                                  config.hiddenSize, ngram.embedDim,
                                  quant.attention)
                for norm in ["norm_key", "norm_query", "norm_conv"] {
                    try requireBF16("\(ple).\(norm).weight", stacked)
                }
                // Depthwise: one kernel per stacked channel, shaped
                // [channels, taps, 1] in the checkpoint.
                try requireBF16Shaped("\(ple).conv1d.weight",
                                      [UInt32(stacked),
                                       UInt32(ngram.convKernelSize), 1])
                // The hash constants are stored, not derived.
                let embedding = "\(ple).ple_embedding"
                try requireInt64("\(embedding).layer_multipliers", ngram.ngramSize)
                try requireInt64("\(embedding).ngram_heads_offsets", ngram.heads)
                try requireInt64("\(embedding).ngram_heads_vocab_sizes", ngram.heads)
            }

            if config.feedForwardKind == .dense {
                for projection in ["gate_proj", "up_proj"] {
                    try requireAffine("\(prefix).mlp.\(projection).weight",
                                      config.intermediateSize, config.hiddenSize, quant.sharedExpert)
                }
                try requireAffine("\(prefix).mlp.down_proj.weight",
                                  config.hiddenSize, config.intermediateSize, quant.sharedExpert)
            } else {
            try requireAffine("\(prefix).mlp.gate.weight",
                              config.numExperts, config.hiddenSize, quant.router)
            try requireAffine("\(prefix).mlp.shared_expert_gate.weight",
                              1, config.hiddenSize, quant.router)
            try requireAffine("\(prefix).mlp.shared_expert.gate_proj.weight",
                              config.intermediateSize, config.hiddenSize, quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.up_proj.weight",
                              config.intermediateSize, config.hiddenSize, quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.down_proj.weight",
                              config.hiddenSize, config.intermediateSize, quant.sharedExpert)

            }

            if config.layerIsLinear(layer) {
                try requireAffine("\(prefix).linear_attn.in_proj_qkv.weight",
                                  linear.qkvDim, config.hiddenSize, quant.attention)
                try requireAffine("\(prefix).linear_attn.in_proj_z.weight",
                                  linear.valueDim, config.hiddenSize, quant.attention)
                try requireAffine("\(prefix).linear_attn.in_proj_a.weight",
                                  linear.numVHeads, config.hiddenSize, quant.attention)
                try requireAffine("\(prefix).linear_attn.in_proj_b.weight",
                                  linear.numVHeads, config.hiddenSize, quant.attention)
                try requireAffine("\(prefix).linear_attn.out_proj.weight",
                                  config.hiddenSize, linear.valueDim, quant.attention)
                try requireBF16Shaped(
                    "\(prefix).linear_attn.conv1d.weight",
                    [UInt32(linear.qkvDim), UInt32(linear.convKernelSize), 1])
                try requireBF16("\(prefix).linear_attn.A_log", linear.numVHeads)
                try requireBF16("\(prefix).linear_attn.dt_bias", linear.numVHeads)
                try requireBF16("\(prefix).linear_attn.norm.weight", linear.valueHeadDim)
                continue
            }

            let queryDimension = try checkedIntMultiply(
                config.numHeads, config.fullHeadDim, "layer \(layer) query")
            let kvDimension = try checkedIntMultiply(
                config.numFullKVHeads, config.fullHeadDim, "layer \(layer) key/value")
            // q_proj is packed `[query ; gate]`, so it has twice the query rows.
            let qProjRows = try checkedIntMultiply(2, queryDimension, "layer \(layer) q_proj")
            try requireAffine("\(prefix).self_attn.q_proj.weight",
                              qProjRows, config.hiddenSize, quant.attention)
            try requireAffine("\(prefix).self_attn.k_proj.weight",
                              kvDimension, config.hiddenSize, quant.attention)
            try requireAffine("\(prefix).self_attn.v_proj.weight",
                              kvDimension, config.hiddenSize, quant.attention)
            try requireAffine("\(prefix).self_attn.o_proj.weight",
                              config.hiddenSize, queryDimension, quant.attention)
            try requireBF16("\(prefix).self_attn.q_norm.weight", config.fullHeadDim)
            try requireBF16("\(prefix).self_attn.k_norm.weight", config.fullHeadDim)
        }
    }

    private static func validateMiniMaxM27LayerSchema(
        config: ArchConfig,
        quant: ManifestQuant,
        requireBF16: (String, Int) throws -> Void,
        requireFP32: (String, Int) throws -> Void,
        requireAffine: (String, Int, Int, ManifestQuantSlot) throws -> Void,
        checkedIntMultiply: (Int, Int, String) throws -> Int
    ) throws {
        let qDimension = try checkedIntMultiply(
            config.numHeads, config.headDim, "MiniMax query")
        let kvDimension = try checkedIntMultiply(
            config.numKVHeads, config.headDim, "MiniMax key/value")
        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            try requireBF16("\(prefix).input_layernorm.weight", config.hiddenSize)
            try requireBF16("\(prefix).post_attention_layernorm.weight", config.hiddenSize)
            try requireBF16("\(prefix).self_attn.q_norm.weight", qDimension)
            try requireBF16("\(prefix).self_attn.k_norm.weight", kvDimension)
            try requireFP32(
                "\(prefix).block_sparse_moe.e_score_correction_bias",
                config.numExperts)
            try requireAffine("\(prefix).self_attn.q_proj.weight",
                              qDimension, config.hiddenSize, quant.attention)
            try requireAffine("\(prefix).self_attn.k_proj.weight",
                              kvDimension, config.hiddenSize, quant.attention)
            try requireAffine("\(prefix).self_attn.v_proj.weight",
                              kvDimension, config.hiddenSize, quant.attention)
            try requireAffine("\(prefix).self_attn.o_proj.weight",
                              config.hiddenSize, qDimension, quant.attention)
            try requireAffine("\(prefix).block_sparse_moe.gate.weight",
                              config.numExperts, config.hiddenSize, quant.router)
        }
    }

}
