import Foundation

/// Exact public RecursiveSI/flint-int4 architecture contract. Keep this
/// separate from Qwen: Flint's sink attention and sigmoid router differ.
extension ArchInfo {
    static func loadValidatedFlint(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "Flint configuration is not an object")
        }
        let expectedFull = Set([0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48])
        let mask = (0..<49).map { expectedFull.contains($0) ? UInt8(1) : UInt8(0) }
        let layerTypes = (0..<49).map { expectedFull.contains($0) ? "full_attention" : "sliding_attention" }
        let exact: [(String, Int)] = [("hidden_size", 2048), ("intermediate_size", 6144),
            ("shared_expert_intermediate_size", 1024), ("moe_intermediate_size", 512),
            ("num_hidden_layers", 49), ("num_attention_heads", 32), ("num_key_value_heads", 4),
            ("head_dim", 128), ("num_experts", 256), ("num_experts_per_tok", 6),
            ("vocab_size", 150144), ("sliding_window", 513), ("max_position_embeddings", 262144)]
        guard root["model_type"] as? String == "flint",
              exact.allSatisfy({ (root[$0.0] as? NSNumber)?.intValue == $0.1 }),
              root["mlp_only_layers"] as? [Int] == [0, 1],
              root["layer_types"] as? [String] == layerTypes,
              root["no_rope_layer_indices"] as? [Int] == expectedFull.sorted(),
              root["norm_topk_prob"] as? Bool == true,
              (root["attention_sink"] as? Bool == true || root["softmax_type"] as? String == "learnable"),
              root["attention_bias"] as? Bool == false,
              root["tie_word_embeddings"] as? Bool == false,
              root["hidden_act"] as? String == "silu",
              (root["rms_norm_eps"] as? NSNumber)?.doubleValue == 1e-6,
              (root["rope_theta"] as? NSNumber)?.doubleValue == 10000,
              let quant = root["quantization"] as? [String: Any],
              (quant["bits"] as? NSNumber)?.intValue == 4,
              (quant["group_size"] as? NSNumber)?.intValue == 64 else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "Flint configuration differs from the pinned architecture")
        }
        return ArchInfo(hiddenSize: 2048, intermediateSize: 1024, moeIntermediateSize: 512,
            numHeads: 32, numKVHeads: 4, numFullKVHeads: 4, headDim: 128, fullHeadDim: 128,
            vocabSize: 150144, slidingWindow: 513, finalLogitSoftcap: 0, ropeTheta: 10000,
            fullRopeTheta: 10000, partialRotaryFactor: 1, numLayers: 49, numExperts: 256,
            topKExperts: 6, tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: mask, hiddenActivation: "silu", family: .flint, variant: .flint40B,
            feedForwardKind: .mixtureOfExperts, attnOutputGate: false,
            attentionScale: 1 / Double(128).squareRoot(), embeddingScaledBySqrtHidden: false,
            routerScaled: false, ffnSandwichNorms: false, sharedExpertGated: false, ropeNeoxSubdim: true)
    }
}
