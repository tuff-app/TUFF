import Foundation

/// The pinned Inkling 2 Mini architecture. The runtime is written for exactly
/// this shape, so a checkpoint whose configuration differs in any field the
/// runtime depends on is refused instead of being repacked into something
/// that loads and produces wrong text.
extension ArchInfo {
    static func loadValidatedInkling(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = root["text_config"] as? [String: Any],
              let loop = text["loop_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "Inkling configuration has no text or loop section")
        }
        let local = Set((0..<24).filter { $0 % 6 != 5 })
        let exact: [(String, Int)] = [
            ("hidden_size", 2304), ("num_hidden_layers", 24), ("vocab_size", 201_024),
            ("unpadded_vocab_size", 200_058), ("num_attention_heads", 24), ("num_key_value_heads", 8),
            ("head_dim", 128), ("swa_num_attention_heads", 24), ("swa_num_key_value_heads", 8),
            ("swa_head_dim", 128), ("sliding_window_size", 1024), ("d_rel", 16), ("rel_extent", 1024),
            ("log_scaling_n_floor", 8192), ("sconv_kernel_size", 4), ("n_layers_per_attn_res_block", 6),
            ("n_routed_experts", 128), ("num_experts_per_tok", 8), ("n_shared_experts", 0),
            ("intermediate_size", 1152)]
        let exactNumbers: [(String, Double)] = [
            ("log_scaling_alpha", 0.1), ("rms_norm_eps", 1e-6), ("qk_l2_clip_threshold", 22.627416997969522),
            ("rel_l2_clip_threshold", 8), ("mlp_gate_softcap", 4), ("mlp_up_softcap", 25), ("route_scale", 8),
            ("final_logit_softcapping", 30)]
        let exactLoop: [(String, Int)] = [("core_start", 6), ("core_end", 18), ("num_passes", 3), ("gate_rank", 64)]
        func close(_ value: Any?, _ expected: Double) -> Bool {
            guard let number = (value as? NSNumber)?.doubleValue else { return false }
            return abs(number - expected) <= 1e-9 * max(1, abs(expected))
        }
        guard root["model_type"] as? String == "inkling2_loop",
              exact.allSatisfy({ (text[$0.0] as? NSNumber)?.intValue == $0.1 }),
              exactNumbers.allSatisfy({ close(text[$0.0], $0.1) }),
              exactLoop.allSatisfy({ (loop[$0.0] as? NSNumber)?.intValue == $0.1 }),
              loop["is_varloop"] as? Bool == false,
              Set((text["local_layer_ids"] as? [Int]) ?? []) == local,
              text["use_gate_bias"] as? Bool == true,
              text["use_value_norm_no_gamma"] as? Bool == true,
              text["attn_res_normalize_values"] as? Bool == true,
              let quant = root["quantization"] as? [String: Any],
              (quant["bits"] as? NSNumber)?.intValue == 4,
              (quant["group_size"] as? NSNumber)?.intValue == 64,
              quant["mode"] as? String == "affine" else {
            throw RepackError.configJsonInvalid(path: configPath,
                detail: "Inkling configuration differs from the pinned Inkling 2 Mini architecture")
        }
        return ArchInfo(hiddenSize: 2304, intermediateSize: 1152, moeIntermediateSize: 1152,
            numHeads: 24, numKVHeads: 8, numFullKVHeads: 8, headDim: 128, fullHeadDim: 128,
            // No RoPE: positions enter through a relative bias. The manifest
            // format wants positive bases, so both are 1 and nothing reads them.
            vocabSize: 200_058, slidingWindow: 1024, finalLogitSoftcap: 30, ropeTheta: 1,
            fullRopeTheta: 1, partialRotaryFactor: 0, numLayers: 24, numExperts: 128,
            topKExperts: 8, tieWordEmbeddings: false, attentionKEqV: false,
            fullAttentionLayerMask: (0..<24).map { local.contains($0) ? UInt8(0) : UInt8(1) },
            hiddenActivation: "silu", family: .inkling2, variant: .inkling2Mini,
            feedForwardKind: .mixtureOfExperts, attnOutputGate: true,
            attentionScale: 1 / Double(128).squareRoot(), embeddingScaledBySqrtHidden: false,
            routerScaled: false, ffnSandwichNorms: false, sharedExpertGated: false, ropeNeoxSubdim: false)
    }
}
