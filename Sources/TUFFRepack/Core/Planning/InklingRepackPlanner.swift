import Foundation
import TUFFFormat

/// Inkling 2 Mini, from the MLX 4-bit conversion. Projections, the embedding
/// and the head are U32-packed group-64 affine with BF16 companions. Norm
/// gains, convolution taps and clip scales are BF16; depth-attention queries,
/// routers and the loop's gate projections are FP32. All of those are
/// resident.
///
/// Routed experts keep the publisher's own formats and are copied as
/// per-expert ranges: NVFP4 in layers 0 to 19 (4-bit codes with one E4M3
/// scale per 16 values) and MXFP8 in layers 20 to 23 (8-bit codes with one
/// exponent per 32). An expert is two matrices, the interleaved gate/up
/// projection and the down projection. The per-expert NVFP4 factors are 128
/// FP32 values a layer and stay resident, so an expert blob holds only what
/// the GPU reads. The vision tower is not installed.
enum InklingRepackPlanner {
    static let sourcePrefix = "model.language_model."
    static let visionPrefix = "model.vision_tower."

    static func canonical(_ name: String) -> String {
        if name.hasPrefix("lm_head.") { return "language_model." + name }
        return "language_model.model." + name.dropFirst(sourcePrefix.count)
    }

    static func plan(arch: ArchInfo, headers: [Safetensors.Header], outputDir: String,
                     sourceIdentity: String? = nil) throws -> RepackPlan {
        guard arch.family == .inkling2, arch.numExperts > 0, arch.hiddenSize % 64 == 0 else {
            throw RepackError.configurationInvalid(detail: "invalid Inkling architecture")
        }
        var registry: [String: SourceTensor] = [:]
        for header in headers {
            for tensor in header.tensors {
                guard registry.updateValue(tensor, forKey: tensor.name) == nil else {
                    throw RepackError.configurationInvalid(detail: "duplicate Inkling tensor \(tensor.name)")
                }
            }
        }
        func tensor(_ name: String) throws -> SourceTensor {
            guard let value = registry[name] else { throw RepackError.missingTensor(name: name) }
            return value
        }
        func page(_ value: UInt64) -> UInt64 {
            (value + Layout.pageBytes - 1) / Layout.pageBytes * Layout.pageBytes
        }
        func shape4(_ shape: [UInt64]) throws -> [UInt32] {
            guard (1...4).contains(shape.count), shape.allSatisfy({ $0 <= UInt64(UInt32.max) }) else {
                throw RepackError.configurationInvalid(detail: "Inkling tensor shape cannot be represented")
            }
            return shape.map(UInt32.init) + Array(repeating: 0, count: 4 - shape.count)
        }
        let vision = registry.keys.filter { $0.hasPrefix(visionPrefix) }.sorted()
        guard registry.keys.allSatisfy({
            $0.hasPrefix(sourcePrefix) || $0.hasPrefix("lm_head.") || $0.hasPrefix(visionPrefix)
        }) else {
            throw RepackError.configurationInvalid(detail: "unexpected Inkling tensor prefix")
        }
        func isExpertPayload(_ name: String) -> Bool {
            name.contains(".mlp.experts.") && !name.hasSuffix(".global_scale")
        }
        let bases = registry.keys.filter {
            !$0.hasPrefix(visionPrefix) && !isExpertPayload($0)
                && !$0.hasSuffix(".scales") && !$0.hasSuffix(".biases")
        }.sorted { canonical($0) < canonical($1) }

        var names: [UInt8] = [], nameOffsets: [UInt32] = []
        for base in bases {
            guard names.count <= Int(UInt32.max) else {
                throw RepackError.configurationInvalid(detail: "Inkling index too large")
            }
            nameOffsets.append(UInt32(names.count))
            names.append(contentsOf: canonical(base).utf8)
        }
        let indexSize = page(UInt64(GTurboBinary.indexHeaderBytes
            + bases.count * GTurboBinary.indexEntryBytes + names.count))
        var offset = indexSize, entries: [ResidentEntry] = []
        // FP32 and packed tensors start on a four-byte boundary.
        func aligned(_ value: UInt64) -> UInt64 { (value + 3) / 4 * 4 }
        for name in bases {
            let weight = try tensor(name)
            if weight.dtype == .u32, name.hasSuffix(".weight") {
                let stem = String(name.dropLast(".weight".count))
                let scale = try tensor(stem + ".scales"), bias = try tensor(stem + ".biases")
                guard weight.shape.count == 2, scale.dtype == .bf16, bias.dtype == .bf16,
                      scale.shape == bias.shape, scale.shape.count == 2,
                      scale.shape[0] == weight.shape[0], scale.shape[1] > 0,
                      weight.shape[1] * 32 == scale.shape[1] * 64 * 4 else {
                    throw RepackError.shapeMismatch(name: name, detail: "invalid Inkling 4-bit affine companions")
                }
                offset = aligned(offset)
                let weightOffset = offset; offset += weight.sizeBytes
                let scaleOffset = offset; offset += scale.sizeBytes
                let biasOffset = offset; offset += bias.sizeBytes
                entries.append(ResidentEntry(name: canonical(name), dtype: GTurboFormatV1.DType.u32.rawValue,
                    logicalShape4: try shape4([weight.shape[0], scale.shape[1] * 64]), fileOffset: weightOffset,
                    sizeBytes: weight.sizeBytes, scaleOffset: scaleOffset, scaleSize: scale.sizeBytes,
                    biasOffset: biasOffset, biasSize: bias.sizeBytes, quantSpec: QuantSpec(bits: 4, groupSize: 64),
                    sourceWeight: weight, sourceScales: scale, sourceBiases: bias))
            } else {
                let dtype: GTurboFormatV1.DType
                switch weight.dtype {
                case .bf16: dtype = .bf16
                case .fp32: dtype = .fp32
                default:
                    throw RepackError.dtypeMismatch(name: name, detail: "expected BF16, FP32 or packed U32")
                }
                offset = aligned(offset)
                let weightOffset = offset; offset += weight.sizeBytes
                entries.append(ResidentEntry(name: canonical(name), dtype: dtype.rawValue,
                    logicalShape4: try shape4(weight.shape), fileOffset: weightOffset, sizeBytes: weight.sizeBytes,
                    scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0, quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }
        let resident = ResidentFilePlan(path: (outputDir as NSString).appendingPathComponent("model_weights.bin"),
            entries: entries, stringTable: names, stringTableOffsets: nameOffsets,
            indexSize: indexSize, residentSize: offset - indexSize)

        let experts = UInt64(arch.numExperts)
        let hidden = UInt64(arch.hiddenSize), intermediate = UInt64(arch.moeIntermediateSize)
        var layers: [LayerFilePlan] = []
        var eightBitMatrices = 0
        for layer in 0..<arch.numLayers {
            let path = (outputDir as NSString).appendingPathComponent(
                "packed_experts/layer_\(String(format: "%02d", layer)).bin")
            let base = "\(sourcePrefix)layers.\(layer).mlp.experts."
            let nvfp4 = registry[base + "gate_up_proj.global_scale"] != nil
            guard nvfp4 == (registry[base + "down_proj.global_scale"] != nil) else {
                throw RepackError.shapeMismatch(name: base, detail: "Inkling expert matrices disagree on their format")
            }
            var slices: [PerExpertTensorSlice] = [], cursor: UInt64 = 0
            for (role, leaf, rows, columns) in [("gate_up", "gate_up_proj", 2 * intermediate, hidden),
                                                ("down", "down_proj", hidden, intermediate)] {
                let w = try tensor(base + leaf + ".weight"), s = try tensor(base + leaf + ".scales")
                let bits: UInt64 = nvfp4 ? 4 : 8, group: UInt64 = nvfp4 ? 16 : 32
                guard w.dtype == .u32, s.dtype == .u8, columns % 32 == 0,
                      w.shape == [experts, rows, columns * bits / 32],
                      s.shape == [experts, rows, columns / group] else {
                    throw RepackError.shapeMismatch(name: base + leaf, detail: "invalid Inkling expert dimensions")
                }
                if nvfp4 {
                    let factor = try tensor(base + leaf + ".global_scale")
                    guard factor.dtype == .fp32, factor.shape == [experts] else {
                        throw RepackError.shapeMismatch(name: factor.name, detail: "expected one FP32 factor per expert")
                    }
                } else {
                    eightBitMatrices += 1
                }
                for (component, source, logical, dtype, weightBits) in [
                    ("weights", w, [rows, columns], GTurboFormatV1.DType.u32.rawValue, Optional(Int(bits))),
                    ("scales", s, [rows, columns / group], SourceTensor.Dtype.u8.rawValue, nil),
                ] {
                    let size = source.sizeBytes / experts
                    guard size * experts == source.sizeBytes else {
                        throw RepackError.shapeMismatch(name: source.name, detail: "expert tensor does not divide evenly")
                    }
                    // Codes are read as 32-bit words.
                    cursor = aligned(cursor)
                    slices.append(PerExpertTensorSlice(role: role, component: component, dtype: dtype,
                        logicalShape: logical, offsetInExpertBlob: cursor, sizeInExpertBlob: size,
                        sourceOffsetPerExpert: size, sourceTensor: source, bitsForWeights: weightBits))
                    cursor += size
                }
            }
            layers.append(LayerFilePlan(layerIndex: layer, path: path, expertsPerLayer: arch.numExperts,
                                        expertStride: page(cursor), subTensors: slices))
        }
        return RepackPlan(arch: arch, baseMode: "affine", baseGroupSize: 64,
                          bitsOverrideCount: eightBitMatrices,
                          resident: resident, layers: layers, ngramTable: nil,
                          matchedModelID: sourceIdentity, excludedMultimodalTensorNames: vision)
    }
}
