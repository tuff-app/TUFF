import Foundation

/// Validated view of a publisher MLX Prism snapshot. This inspects the source;
/// it does not yet make a TUFF model or grant catalog support.
struct PrismSnapshot {
    let configPath: String
    let header: Safetensors.Header
    let hadamard: PrismHadamard
    let tensors: [String: SourceTensor]

    static func inspect(directory: URL) throws -> PrismSnapshot {
        let configPath = directory.appendingPathComponent("config.json").path
        let sidecarPath = directory.appendingPathComponent("hadamard.json").path
        let root: [String: Any]
        do {
            guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: configPath))) as? [String: Any] else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
            }
            root = value
        } catch let error as RepackError { throw error }
        catch { throw RepackError.configJsonInvalid(path: configPath, detail: "\(error)") }
        guard root["model_type"] as? String == "prism_hadamard_qwen35",
              root["base_model_type"] as? String == "qwen3_5",
              let quant = root["quantization"] as? [String: Any],
              quant["bits"] as? Int == 2, quant["group_size"] as? Int == 128,
              quant["mode"] as? String == "affine",
              let modules = root["modules"] as? [[String: Any]], !modules.isEmpty else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "unsupported Prism source configuration")
        }
        let hadamard = try PrismHadamard.load(path: sidecarPath)
        let header = try Safetensors.parseHeader(path: directory.appendingPathComponent("model.safetensors").path)
        let tensors = Dictionary(uniqueKeysWithValues: header.tensors.map { ($0.name, $0) })
        var names = Set<String>()
        var embeddings = Set<String>()
        for module in modules {
            guard let path = module["path"] as? String, !path.isEmpty,
                  module["block"] as? Int == hadamard.blockSize,
                  module["dtype"] as? String == "float16",
                  let embedding = module["embedding"] as? Bool else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "invalid Prism module record")
            }
            let prefix = "language_model." + path
            let weightName = prefix + ".weight"
            guard names.insert(weightName).inserted,
                  let weight = tensors[weightName], weight.dtype == .u32, weight.shape.count == 2,
                  weight.shape[1] <= UInt64(Int.max / 16) else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "invalid Prism weight \(weightName)")
            }
            let columns = Int(weight.shape[1]) * 16
            guard columns > 0, columns % 128 == 0,
                  let scales = tensors[prefix + ".scales"], let biases = tensors[prefix + ".biases"],
                  scales.dtype == .fp16, biases.dtype == .fp16,
                  scales.shape == [weight.shape[0], UInt64(columns / 128)], biases.shape == scales.shape else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "invalid Prism affine companions for \(weightName)")
            }
            if embedding { embeddings.insert(weightName) }
        }
        guard names == Set(hadamard.weightNames + hadamard.inverseWeightNames),
              embeddings == Set(hadamard.inverseWeightNames) else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "Prism modules and transform metadata disagree")
        }
        try hadamard.validateTensors(tensors, sidecarPath: sidecarPath)
        return PrismSnapshot(configPath: configPath, header: header, hadamard: hadamard, tensors: tensors)
    }
}
