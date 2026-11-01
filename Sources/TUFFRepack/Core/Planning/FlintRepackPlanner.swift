import Foundation
import TUFFFormat

/// Flint checkpoints use U8 or U32-packed affine matrices with BF16
/// companions. Expert banks are copied as per-expert ranges without
/// materializing a bank or the full checkpoint in RAM.
enum FlintRepackPlanner {
    static func plan(arch: ArchInfo, headers: [Safetensors.Header], outputDir: String,
                     sourceIdentity: String? = nil,
                     declaredQuantization: [String: QuantSpec]? = nil) throws -> RepackPlan {
        guard arch.numLayers >= 3, arch.numExperts > 0, arch.topKExperts > 0,
              arch.topKExperts <= arch.numExperts else {
            throw RepackError.configurationInvalid(detail: "invalid Flint layer or expert count")
        }
        var registry: [String: SourceTensor] = [:]
        for header in headers {
            for tensor in header.tensors {
                guard registry.updateValue(tensor, forKey: tensor.name) == nil else {
                    throw RepackError.configurationInvalid(detail: "duplicate Flint tensor \(tensor.name)")
                }
            }
        }
        func tensor(_ name: String) throws -> SourceTensor {
            guard let value = registry[name] else { throw RepackError.missingTensor(name: name) }
            return value
        }
        func canonical(_ name: String) -> String {
            if name == "lm_head.weight" { return "language_model.lm_head.weight" }
            let full = "language_model." + name
            return full.hasSuffix(".self_attn.softmax_offset")
                ? String(full.dropLast("softmax_offset".count)) + "sinks" : full
        }
        func validateBits(_ bits: Int, stem: String) throws {
            guard let declaredQuantization else { return }
            let spec = declaredQuantization[stem] ?? QuantSpec(bits: 4, groupSize: 64)
            guard spec.bits == bits, spec.groupSize == 64 else {
                throw RepackError.configurationInvalid(detail: "Flint quantization disagrees with tensor \(stem)")
            }
        }
        func page(_ value: UInt64) -> UInt64 {
            (value + Layout.pageBytes - 1) / Layout.pageBytes * Layout.pageBytes
        }
        func shape4(_ shape: [UInt64]) throws -> [UInt32] {
            guard shape.count <= 4, shape.allSatisfy({ $0 <= UInt64(UInt32.max) }) else {
                throw RepackError.configurationInvalid(detail: "Flint tensor shape cannot be represented")
            }
            return shape.map(UInt32.init) + Array(repeating: 0, count: 4 - shape.count)
        }
        let bases = registry.keys.filter {
            !$0.hasSuffix(".scales") && !$0.hasSuffix(".biases") && !$0.contains(".mlp.switch_mlp.")
        }.sorted { canonical($0) < canonical($1) }
        guard bases.allSatisfy({ $0 == "lm_head.weight" || $0.hasPrefix("model.") }) else {
            throw RepackError.configurationInvalid(detail: "unexpected Flint tensor prefix")
        }
        var names: [UInt8] = [], nameOffsets: [UInt32] = []
        for base in bases {
            guard names.count <= Int(UInt32.max) else { throw RepackError.configurationInvalid(detail: "Flint index too large") }
            nameOffsets.append(UInt32(names.count)); names.append(contentsOf: canonical(base).utf8)
        }
        let indexSize = page(UInt64(GTurboBinary.indexHeaderBytes + bases.count * GTurboBinary.indexEntryBytes + names.count))
        var offset = indexSize, entries: [ResidentEntry] = []
        for name in bases {
            let weight = try tensor(name)
            let isPacked = (weight.dtype == .u8 || weight.dtype == .u32) && name.hasSuffix(".weight")
            if isPacked {
                let stem = String(name.dropLast(".weight".count))
                let scale = try tensor(stem + ".scales"), bias = try tensor(stem + ".biases")
                guard weight.shape.count == 2, scale.dtype == .bf16, bias.dtype == .bf16,
                      scale.shape == bias.shape, scale.shape.count == 2,
                      scale.shape[0] == weight.shape[0], scale.shape[1] > 0 else {
                    throw RepackError.shapeMismatch(name: name, detail: "invalid Flint affine companions")
                }
                let logicalColumns = scale.shape[1] * 64
                let packedBits = weight.shape[1] * UInt64(weight.dtype.elementBytes * 8)
                guard packedBits % logicalColumns == 0 else {
                    throw RepackError.shapeMismatch(name: name, detail: "packed width does not match group-64 scales")
                }
                let bits = Int(packedBits / logicalColumns)
                guard bits == 4 || bits == 8 else { throw RepackError.configurationInvalid(detail: "unsupported Flint bit width") }
                try validateBits(bits, stem: stem)
                let weightOffset = offset; offset += weight.sizeBytes
                let scaleOffset = offset; offset += scale.sizeBytes
                let biasOffset = offset; offset += bias.sizeBytes
                entries.append(ResidentEntry(name: canonical(name), dtype: GTurboFormatV1.DType.u32.rawValue,
                    logicalShape4: try shape4([weight.shape[0], logicalColumns]), fileOffset: weightOffset,
                    sizeBytes: weight.sizeBytes, scaleOffset: scaleOffset, scaleSize: scale.sizeBytes,
                    biasOffset: biasOffset, biasSize: bias.sizeBytes, quantSpec: QuantSpec(bits: bits, groupSize: 64),
                    sourceWeight: weight, sourceScales: scale, sourceBiases: bias))
            } else {
                guard weight.dtype == .bf16 else {
                    throw RepackError.dtypeMismatch(name: name, detail: "expected BF16 or packed U8")
                }
                let weightOffset = offset; offset += weight.sizeBytes
                entries.append(ResidentEntry(name: canonical(name), dtype: GTurboFormatV1.DType.bf16.rawValue,
                    logicalShape4: try shape4(weight.shape), fileOffset: weightOffset, sizeBytes: weight.sizeBytes,
                    scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0, quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }
        let resident = ResidentFilePlan(path: (outputDir as NSString).appendingPathComponent("model_weights.bin"),
            entries: entries, stringTable: names, stringTableOffsets: nameOffsets,
            indexSize: indexSize, residentSize: offset - indexSize)
        var layers: [LayerFilePlan] = []
        for layer in 0..<arch.numLayers {
            let path = (outputDir as NSString).appendingPathComponent("packed_experts/layer_\(String(format: "%02d", layer)).bin")
            if layer < 2 {
                layers.append(LayerFilePlan(layerIndex: layer, path: path, expertsPerLayer: 0,
                                            expertStride: 0, subTensors: []))
                continue
            }
            var slices: [PerExpertTensorSlice] = [], cursor: UInt64 = 0
            for role in ["gate", "up", "down"] {
                let stem = "model.layers.\(layer).mlp.switch_mlp.\(role)_proj"
                let w = try tensor(stem + ".weight"), s = try tensor(stem + ".scales"), b = try tensor(stem + ".biases")
                let rows = UInt64(role == "down" ? arch.hiddenSize : arch.moeIntermediateSize)
                let cols = UInt64(role == "down" ? arch.moeIntermediateSize : arch.hiddenSize)
                let experts = UInt64(arch.numExperts)
                guard cols % 64 == 0, (w.dtype == .u8 || w.dtype == .u32),
                      w.shape.count == 3, w.shape[0] == experts, w.shape[1] == rows,
                      s.dtype == .bf16, b.dtype == .bf16,
                      s.shape == [experts, rows, cols / 64], b.shape == s.shape else {
                    throw RepackError.shapeMismatch(name: stem, detail: "invalid Flint packed expert dimensions")
                }
                let packedBits = w.shape[2] * UInt64(w.dtype.elementBytes * 8)
                guard packedBits % cols == 0, [4, 8].contains(Int(packedBits / cols)) else {
                    throw RepackError.shapeMismatch(name: stem, detail: "unsupported Flint expert bit width")
                }
                let bits = Int(packedBits / cols)
                try validateBits(bits, stem: stem)
                for (component, source, logical, dtype, weightBits) in [
                    ("weights", w, [rows, cols], GTurboFormatV1.DType.u32.rawValue, Optional(bits)),
                    ("scales", s, [rows, cols / 64], GTurboFormatV1.DType.bf16.rawValue, nil),
                    ("biases", b, [rows, cols / 64], GTurboFormatV1.DType.bf16.rawValue, nil),
                ] {
                    let size = source.sizeBytes / experts
                    guard size * experts == source.sizeBytes else {
                        throw RepackError.shapeMismatch(name: source.name, detail: "expert tensor does not divide evenly")
                    }
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
                          bitsOverrideCount: entries.filter { $0.quantSpec?.bits == 8 }.count
                              + layers.flatMap(\.subTensors).filter { $0.bitsForWeights == 8 }.count,
                          resident: resident, layers: layers, ngramTable: nil,
                          matchedModelID: sourceIdentity, excludedMultimodalTensorNames: [])
    }
}
