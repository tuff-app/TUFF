import TUFFRuntimeDescriptions
import Foundation

/// Model family discriminator, mirrored into `manifest.json -> arch.family`
/// for non-Gemma families (Gemma manifests omit it — the format's original
/// architecture). Raw values match the runtime's `ModelFamily`.
enum RepackModelFamily: String, Sendable, Equatable {
    case gemma4 = "gemma4"
    case qwen36 = "qwen36"
    case gptOss = "gpt-oss"
    case minimaxM2 = "minimax-m2"
    case qwen4Exp = "qwen4-exp"
    case flint = "flint"
}

enum RepackModelVariant: String, Sendable, Equatable {
    case gemma4_E2B = "gemma4-e2b"
    case gemma4_E4B = "gemma4-e4b"
    case gemma4_12B_QAT = "gemma4-12b-qat"
    case gemma4_26B_A4B = "gemma4-26b-a4b"
    case qwen36_35B_A3B = "qwen36-35b-a3b"
    case gptOss_20B = "gpt-oss-20b"
    case gptOss_120B = "gpt-oss-120b"
    case minimaxM27 = "minimax-m2.7"
    case qwen38FlashNext = "qwen3.8-flash-next"
    case ornith15_9B = "ornith-1.5-9b"
    case ornith15_35B_A3B = "ornith-1.5-35b-a3b"
    case ornith15_122B = "ornith-1.5-122b"
    case ornith15_397B = "ornith-1.5-397b"
    case qwen38_27B = "qwen3.8-27b"
    case bonsai2_2B = "ternary-bonsai-2-2b"
    case bonsai2_27B = "ternary-bonsai-2-27b"
    case flint40B = "flint-40b-a2.5b"
}

/// Which dense Gemma a checkpoint is, from its own shape.
///
/// Every dense Gemma used to be labelled E4B, which was true while E4B was the
/// only one. E2B repacked correctly — every shape in the manifest was its
/// own — and then carried E4B's variant name, so the installed model was
/// validated against E4B's architecture and a complete, correct download was
/// reported as "completed install did not pass metadata validation".
///
/// Keyed on two dimensions rather than one so a future dense Gemma that
/// happens to share a hidden size cannot silently inherit a name.
func denseGemmaVariant(hiddenSize: Int, numLayers: Int) throws -> RepackModelVariant {
    switch (hiddenSize, numLayers) {
    case (1_536, 35): .gemma4_E2B
    case (2_560, 42): .gemma4_E4B
    case (3_840, 48): .gemma4_12B_QAT
#if DEBUG
    // The repacker's bounded synthetic fixture uses these dimensions so its
    // complete dense-install path stays testable without materializing a
    // multi-gigabyte checkpoint. Release builds still fail closed on it.
    case (128, 4): .gemma4_E4B
#endif
    default:
        throw RepackError.configurationInvalid(
            detail: "unsupported dense Gemma architecture \(hiddenSize)x\(numLayers)")
    }
}

enum RepackFeedForwardKind: String, Sendable, Equatable {
    case dense
    case mixtureOfExperts = "moe"
}

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`, 2 if `linear_attention`.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String

    // Family-dependent extensions. Defaults describe Gemma 4 so the Gemma
    // path (and its manifest output) is unchanged, and so a caller that
    // predates the family split still builds the architecture it meant to.
    var family: RepackModelFamily = .gemma4
    var variant: RepackModelVariant = .gemma4_26B_A4B
    /// First layer whose MLP is twice `intermediateSize` wide; -1 for none.
    var ffnDoubleWideFromLayer: Int = -1
    var feedForwardKind: RepackFeedForwardKind = .mixtureOfExperts
    var hiddenSizePerLayerInput: Int = 0
    var vocabSizePerLayerInput: Int = 0
    var numKVSharedLayers: Int = 0
    var attnOutputGate: Bool = false
    var attentionScale: Double = 1.0
    var embeddingScaledBySqrtHidden: Bool = true
    var routerScaled: Bool = true
    var ffnSandwichNorms: Bool = true
    var sharedExpertGated: Bool = false
    var ropeNeoxSubdim: Bool = false
    var linearNumKHeads: Int = 0
    var linearNumVHeads: Int = 0
    var linearKeyHeadDim: Int = 0
    var linearValueHeadDim: Int = 0
    var linearConvKernelSize: Int = 0
    // `qwen4_exp` extensions. Zero everywhere else, and omitted from those
    // manifests, so no other family's output changes.
    var hyperConnectionStreamCount: Int = 0
    var hyperConnectionLowRank: Int = 0
    var ngramLayer: Int = -1
    var ngramSize: Int = 0
    var ngramHeads: Int = 0
    var ngramHeadsPerNgram: Int = 0
    var ngramVocabSizeBase: Int = 0
    var ngramShardCount: Int = 0
    var ngramEmbedDim: Int = 0
    var ngramConvKernelSize: Int = 0
    var ngramEosTokenID: Int = 0
    var indexerBudget: Int = 0
    var indexerCompressRatio: Int = 0
    var indexerHeadDim: Int = 0
    var indexerNumHeads: Int = 0
    var indexerNumKVHeads: Int = 0

    static func load(configPath: String, variant: RepackModelVariant? = nil) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        // These architectures cannot be interpreted as an ordinary Qwen or
        // Gemma checkpoint. In particular, Prism's projection and embedding
        // weights are in a rotated basis and require explicit transforms.
        // Fail before any generic text_config fallback can misidentify them.
        switch root["model_type"] as? String {
        case "prism_hadamard_qwen35":
            throw RepackError.configJsonInvalid(path: configPath,
                detail: "Prism Hadamard packs require a dedicated 2-bit and Hadamard runtime; this build cannot repack them")
        case "flint":
            throw RepackError.configJsonInvalid(path: configPath,
                detail: "Flint requires its own attention and expert runtime; this build cannot repack it")
        default:
            break
        }
        if (root["model_type"] as? String) == "gpt_oss" {
            return try loadGPTOSS(configPath: configPath, config: root)
        }
        if (root["model_type"] as? String) == "minimax_m2" {
            return try loadMiniMaxM27(configPath: configPath, config: root)
        }
        guard let tc = (root["text_config"] as? [String: Any])
                ?? ((root["model_type"] as? String)?.hasPrefix("qwen3_5") == true ? root : nil) else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        if ["qwen3_5_moe", "qwen3_5_moe_text"].contains(root["model_type"] as? String ?? "") {
            return try loadQwen36(configPath: configPath, tc: tc, variant: variant)
        }
        if ["qwen3_5", "qwen3_5_text"].contains(root["model_type"] as? String ?? "") {
            return try loadQwen36(configPath: configPath, tc: tc, dense: true, variant: variant)
        }
        if (root["model_type"] as? String) == "qwen4_exp" {
            return try loadQwen4Exp(configPath: configPath, tc: tc)
        }
        return try loadGemma4(configPath: configPath, tc: tc)
    }

    // MARK: - MiniMax M2.7

    private static func loadMiniMaxM27(
        configPath: String,
        config: [String: Any]
    ) throws -> ArchInfo {
        func i(_ key: String) throws -> Int {
            guard let value = (config[key] as? Int)
                    ?? (config[key] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(key)")
            }
            return value
        }
        func d(_ key: String) throws -> Double {
            guard let value = (config[key] as? Double)
                    ?? (config[key] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(key)")
            }
            return value
        }
        let headDim = try i("head_dim")
        let rotaryDim = try i("rotary_dim")
        let layers = try i("num_hidden_layers")
        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0,
            ropeTheta: try d("rope_theta"),
            fullRopeTheta: try d("rope_theta"),
            partialRotaryFactor: Double(rotaryDim) / Double(headDim),
            numLayers: layers,
            numExperts: try i("num_local_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: (config["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: [UInt8](repeating: 1, count: layers),
            hiddenActivation: (config["hidden_act"] as? String) ?? "silu",
            family: .minimaxM2,
            variant: .minimaxM27,
            feedForwardKind: .mixtureOfExperts,
            attentionScale: 1 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true)
        guard arch.hiddenSize == 3_072,
              arch.intermediateSize == 1_536,
              arch.numLayers == 62,
              arch.numHeads == 48,
              arch.numKVHeads == 8,
              arch.headDim == 128,
              arch.vocabSize == 200_064,
              arch.ropeTheta == 5_000_000,
              arch.numExperts == 256,
              arch.topKExperts == 8,
              arch.partialRotaryFactor == 0.5,
              !arch.tieWordEmbeddings,
              arch.hiddenActivation == "silu",
              config["use_qk_norm"] as? Bool == true else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "config does not match the pinned MiniMax M2.7 architecture baseline")
        }
        return arch
    }

    // MARK: - GPT-OSS

    private static func loadGPTOSS(
        configPath: String,
        config: [String: Any]
    ) throws -> ArchInfo {
        func i(_ key: String) throws -> Int {
            guard let value = (config[key] as? Int)
                    ?? (config[key] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(key)")
            }
            return value
        }
        func d(_ key: String) throws -> Double {
            guard let value = (config[key] as? Double)
                    ?? (config[key] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing \(key)")
            }
            return value
        }
        guard let layerTypes = config["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing layer_types")
        }
        let mask: [UInt8] = try layerTypes.map { value in
            switch value {
            case "sliding_attention": return 0
            case "full_attention": return 1
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath,
                    detail: "unknown layer_types entry \"\(value)\"")
            }
        }
        guard let quant = config["quantization_config"] as? [String: Any],
              (quant["quant_method"] as? String)?.lowercased() == "mxfp4" else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "GPT-OSS requires quantization_config.quant_method=mxfp4")
        }
        let headDim = try i("head_dim")
        let experts = (config["num_local_experts"] as? Int)
            ?? (config["num_local_experts"] as? NSNumber)?.intValue
        let topK = (config["experts_per_token"] as? Int)
            ?? (config["experts_per_token"] as? NSNumber)?.intValue
            ?? (config["num_experts_per_tok"] as? Int)
            ?? (config["num_experts_per_tok"] as? NSNumber)?.intValue
        guard let experts, let topK else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing GPT-OSS expert dimensions")
        }
        let hiddenSize = try i("hidden_size")
        let numLayers = try i("num_hidden_layers")
        let variant: RepackModelVariant = hiddenSize == 2_880 && numLayers == 36
            ? .gptOss_120B : .gptOss_20B
        let arch = ArchInfo(
            hiddenSize: hiddenSize,
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: 0,
            ropeTheta: try d("rope_theta"),
            fullRopeTheta: try d("rope_theta"),
            partialRotaryFactor: 1,
            numLayers: numLayers,
            numExperts: experts,
            topKExperts: topK,
            tieWordEmbeddings: (config["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: "swiglu_capped",
            family: .gptOss,
            variant: variant,
            feedForwardKind: .mixtureOfExperts,
            attnOutputGate: false,
            attentionScale: 1 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true)
        try crossCheckProductionGPTOSS(
            arch, config: config, configPath: configPath)
        return arch
    }

    private static func crossCheckProductionGPTOSS(
        _ arch: ArchInfo,
        config: [String: Any],
        configPath: String
    ) throws {
        guard arch.hiddenSize == 2_880 else { return }
        let expectedLayers: Int
        let expectedExperts: Int
        let expectedVariant: RepackModelVariant
        switch arch.numLayers {
        case 24:
            expectedLayers = 24
            expectedExperts = 32
            expectedVariant = .gptOss_20B
        case 36:
            expectedLayers = 36
            expectedExperts = 128
            expectedVariant = .gptOss_120B
        default:
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "unsupported production GPT-OSS layer count \(arch.numLayers)")
        }
        let expectedMask = (0..<expectedLayers).map {
            UInt8($0.isMultiple(of: 2) ? 0 : 1)
        }
        let rope = config["rope_scaling"] as? [String: Any]
        func number(_ object: Any?) -> Double? {
            (object as? Double) ?? (object as? NSNumber)?.doubleValue
        }
        guard arch.intermediateSize == 2_880,
              arch.moeIntermediateSize == 2_880,
              arch.numHeads == 64,
              arch.numKVHeads == 8,
              arch.headDim == 64,
              arch.vocabSize == 201_088,
              arch.slidingWindow == 128,
              arch.ropeTheta == 150_000,
              arch.numExperts == expectedExperts,
              arch.topKExperts == 4,
              arch.variant == expectedVariant,
              arch.fullAttentionLayerMask == expectedMask,
              arch.attentionScale == 0.125,
              arch.hiddenActivation == "swiglu_capped",
              !arch.tieWordEmbeddings,
              config["attention_bias"] as? Bool == true,
              number(config["swiglu_limit"]) == 7,
              (config["initial_context_length"] as? Int) == 4_096,
              (config["max_position_embeddings"] as? Int) == 131_072,
              (rope?["rope_type"] as? String) == "yarn",
              number(rope?["factor"]) == 32,
              number(rope?["beta_fast"]) == 32,
              number(rope?["beta_slow"]) == 1,
              (rope?["original_max_position_embeddings"] as? Int) == 4_096 else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "GPT-OSS config does not match the pinned \(expectedLayers == 24 ? "20B" : "120B") architecture baseline")
        }
    }

    // MARK: - Gemma 4

    private static func loadGemma4(configPath: String,
                                   tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func optionalInt(_ k: String) -> Int? {
            (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue
        }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = (ropeFull["partial_rotary_factor"] as? Double)
            ?? (ropeFull["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let fullTheta = (ropeFull["rope_theta"] as? Double)
            ?? (ropeFull["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let swaTheta = (ropeSWA["rope_theta"] as? Double)
            ?? (ropeSWA["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000.0
        let kEqV = (tc["attention_k_eq_v"] as? Bool) ?? false
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let act = (tc["hidden_activation"] as? String) ?? "gelu_pytorch_tanh"
        let dense = (tc["enable_moe_block"] as? Bool) == false
            || optionalInt("num_experts") == nil
        // `use_double_wide_mlp` is not a whole-model property: on Gemma 4 E2B
        // the first layers hold a normal MLP and the last `num_kv_shared_layers`
        // hold one twice as wide. Sizing every layer from `intermediate_size`
        // rejected the checkpoint at the first wide layer.
        let doubleWideMLP = (tc["use_double_wide_mlp"] as? Bool) ?? false
        let sharedKV = optionalInt("num_kv_shared_layers") ?? 0
        let totalLayers = try i("num_hidden_layers")
        let doubleWideFrom = (doubleWideMLP && sharedKV > 0 && sharedKV < totalLayers)
            ? totalLayers - sharedKV
            : -1
        let numKVHeads = try i("num_key_value_heads")
        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: dense ? 0 : try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: numKVHeads,
            numFullKVHeads: optionalInt("num_global_key_value_heads") ?? numKVHeads,
            headDim: try i("head_dim"),
            fullHeadDim: try i("global_head_dim"),
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: try d("final_logit_softcapping"),
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: dense ? 0 : try i("num_experts"),
            topKExperts: dense ? 0 : try i("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .gemma4,
            variant: dense ? try denseGemmaVariant(
                hiddenSize: try i("hidden_size"),
                numLayers: try i("num_hidden_layers")) : .gemma4_26B_A4B,
            ffnDoubleWideFromLayer: doubleWideFrom,
            feedForwardKind: dense ? .dense : .mixtureOfExperts,
            hiddenSizePerLayerInput: optionalInt("hidden_size_per_layer_input") ?? 0,
            vocabSizePerLayerInput: optionalInt("vocab_size_per_layer_input") ?? 0,
            numKVSharedLayers: optionalInt("num_kv_shared_layers") ?? 0,
            attnOutputGate: false,
            attentionScale: 1.0,
            embeddingScaledBySqrtHidden: true,
            routerScaled: true,
            ffnSandwichNorms: true,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0)
        try crossCheckProductionGemma4(arch, configPath: configPath)
        return arch
    }

    private static func crossCheckProductionGemma4(_ arch: ArchInfo,
                                                    configPath: String) throws {
        if arch.variant == .gemma4_12B_QAT {
            var expectedMask = [UInt8](repeating: 0, count: 48)
            for layer in stride(from: 5, to: 48, by: 6) { expectedMask[layer] = 1 }
            guard arch.hiddenSize == 3_840,
                  arch.intermediateSize == 15_360,
                  arch.moeIntermediateSize == 0,
                  arch.numLayers == 48,
                  arch.numHeads == 16,
                  arch.numKVHeads == 8,
                  arch.numFullKVHeads == 1,
                  arch.headDim == 256,
                  arch.fullHeadDim == 512,
                  arch.vocabSize == 262_144,
                  arch.slidingWindow == 1_024,
                  arch.fullAttentionLayerMask == expectedMask,
                  arch.hiddenSizePerLayerInput == 0,
                  arch.numKVSharedLayers == 0,
                  arch.tieWordEmbeddings,
                  arch.attentionKEqV else {
                throw RepackError.configJsonInvalid(
                    path: configPath,
                    detail: "gemma4 dense config does not match the pinned 12B QAT architecture baseline")
            }
            return
        }
        guard arch.variant == .gemma4_E4B,
              arch.hiddenSize == 2_560,
              arch.numLayers == 42 else { return }
        var expectedMask = [UInt8](repeating: 0, count: 42)
        for layer in stride(from: 5, to: 42, by: 6) { expectedMask[layer] = 1 }
        guard arch.intermediateSize == 10_240,
              arch.moeIntermediateSize == 0,
              arch.numHeads == 8,
              arch.numKVHeads == 2,
              arch.numFullKVHeads == 2,
              arch.headDim == 256,
              arch.fullHeadDim == 512,
              arch.vocabSize == 262_144,
              arch.slidingWindow == 512,
              arch.numExperts == 0,
              arch.topKExperts == 0,
              arch.fullAttentionLayerMask == expectedMask,
              arch.hiddenSizePerLayerInput == 256,
              arch.vocabSizePerLayerInput == 262_144,
              arch.numKVSharedLayers == 18,
              arch.tieWordEmbeddings,
              !arch.attentionKEqV else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "gemma4 dense config does not match the pinned E4B architecture baseline")
        }
    }

    /// Read a pinned Prism backbone after its embedded transforms are checked.
    /// This does not enable generic dispatch.
    static func loadValidatedPrism(_ snapshot: PrismSnapshot) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: snapshot.configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: snapshot.configPath, detail: "missing Prism text_config")
        }
        let hidden = (text["hidden_size"] as? NSNumber)?.intValue
        let layers = (text["num_hidden_layers"] as? NSNumber)?.intValue
        let tied = (text["tie_word_embeddings"] as? Bool) ?? false
        let selected: RepackModelVariant
        switch (hidden, layers, snapshot.hadamard.blockSize, tied) {
        case (2048, 24, 512, true): selected = .bonsai2_2B
        case (5120, 64, 1024, false): selected = .bonsai2_27B
        default:
            throw RepackError.configJsonInvalid(path: snapshot.configPath, detail: "unsupported Prism architecture")
        }
        guard (snapshot.hadamard.tiedOutput ?? false) == tied else {
            throw RepackError.configJsonInvalid(path: snapshot.configPath, detail: "Prism tied-output metadata disagrees")
        }
        return try loadQwen36(configPath: snapshot.configPath, tc: text, dense: true, variant: selected)
    }

    // MARK: - Qwen 3.6 MoE (`model_type == "qwen3_5_moe"`)

    private static func loadQwen36(configPath: String,
                                   tc: [String: Any], dense: Bool = false,
                                   variant: RepackModelVariant? = nil) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "linear_attention": mask.append(2)
            case "full_attention":   mask.append(1)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        guard let theta = (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.rope_theta")
        }
        guard let prf = (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.partial_rotary_factor")
        }
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let gate = (tc["attn_output_gate"] as? Bool) ?? false
        let act = (tc["hidden_act"] as? String) ?? "silu"
        let headDim = try i("head_dim")

        let selectedVariant: RepackModelVariant
        if let variant { selectedVariant = variant }
        else if dense {
            switch (try i("hidden_size"), try i("num_hidden_layers")) {
            case (4096, 32): selectedVariant = .ornith15_9B
            case (5120, 64): selectedVariant = .qwen38_27B
            default: throw RepackError.configJsonInvalid(path: configPath, detail: "unsupported dense Qwen dimensions")
            }
        } else if try i("hidden_size") == 4096 && i("num_hidden_layers") == 60 {
            selectedVariant = .ornith15_397B
        } else if try i("hidden_size") == 3072 && i("num_hidden_layers") == 48 {
            selectedVariant = .ornith15_122B
        } else { selectedVariant = .qwen36_35B_A3B }
        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i(dense ? "intermediate_size" : "shared_expert_intermediate_size"),
            moeIntermediateSize: dense ? 0 : try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: dense ? 0 : try i("num_experts"),
            topKExperts: dense ? 0 : try i("num_experts_per_tok"),
            tieWordEmbeddings: tie,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .qwen36,
            variant: selectedVariant,
            feedForwardKind: dense ? .dense : .mixtureOfExperts,
            attnOutputGate: gate,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: !dense,
            ropeNeoxSubdim: true,
            linearNumKHeads: try i("linear_num_key_heads"),
            linearNumVHeads: try i("linear_num_value_heads"),
            linearKeyHeadDim: try i("linear_key_head_dim"),
            linearValueHeadDim: try i("linear_value_head_dim"),
            linearConvKernelSize: try i("linear_conv_kernel_dim"))
        if selectedVariant == .qwen36_35B_A3B {
            try crossCheckProductionQwen36(arch, configPath: configPath)
        } else {
            try crossCheckQwen35Contract(arch, configPath: configPath)
        }
        return arch
    }

    private static func crossCheckQwen35Contract(_ a: ArchInfo, configPath: String) throws {
        guard let variant = TUFFRuntimeDescriptions.ModelVariant(rawValue: a.variant.rawValue),
              let b = TUFFRuntimeDescriptions.ArchConfig.registeredArchitectures[variant],
              a.hiddenSize == b.hiddenSize, a.intermediateSize == b.intermediateSize,
              a.moeIntermediateSize == b.moeIntermediateSize, a.numLayers == b.numLayers,
              a.numHeads == b.numHeads, a.numKVHeads == b.numKVHeads,
              a.headDim == b.headDim, a.vocabSize == b.vocabSize,
              a.numExperts == b.numExperts, a.topKExperts == b.topKExperts,
              a.tieWordEmbeddings == b.tieWordEmbeddings,
              a.fullAttentionLayerMask == b.fullAttentionLayerMask,
              a.hiddenActivation == b.hiddenActivation, a.attnOutputGate == b.attnOutputGate,
              a.ropeTheta == b.ropeTheta, a.partialRotaryFactor == b.partialRotaryFactor,
              a.linearNumKHeads == b.linearAttention.numKHeads,
              a.linearNumVHeads == b.linearAttention.numVHeads,
              a.linearKeyHeadDim == b.linearAttention.keyHeadDim,
              a.linearValueHeadDim == b.linearAttention.valueHeadDim,
              a.linearConvKernelSize == b.linearAttention.convKernelSize else {
            throw RepackError.configJsonInvalid(path: configPath,
                detail: "config does not match pinned \(a.variant.rawValue) block contract")
        }
    }

    // MARK: - Qwen3.8 Flash Next (qwen4_exp)

    /// Parse the `qwen4_exp` text config.
    ///
    /// The blocks it shares with Qwen3.6 are read the same way. What it adds:
    /// four-stream hyper-connections, an n-gram per-layer-embedding table, and
    /// a sparse attention indexer.
    private static func loadQwen4Exp(configPath: String,
                                     tc: [String: Any]) throws -> ArchInfo {
        func i(_ key: String) throws -> Int {
            guard let value = (tc[key] as? Int)
                    ?? (tc[key] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing text_config.\(key)")
            }
            return value
        }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing text_config.layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "linear_attention": mask.append(2)
            case "full_attention":   mask.append(1)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        guard let theta = (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.rope_theta")
        }
        guard let prf = (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue
            ?? (tc["partial_rotary_factor"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.partial_rotary_factor")
        }
        let headDim = try i("head_dim")

        // The layer the n-gram module is attached to. `ple_layer_ids` names
        // layer 2 while the checkpoint emits the tensors under `layers.1`, and
        // the loader resolves tensor names, so take the smallest declared id
        // minus one and let the planner reject a checkpoint that disagrees.
        guard let pleLayerIDs = tc["ple_layer_ids"] as? [Int],
              let declaredPLELayer = pleLayerIDs.min(), pleLayerIDs.count == 1 else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "expected exactly one text_config.ple_layer_ids entry")
        }
        let shardCount = try i("split_ngram_parts")
        let headsPerNgram = try i("heads_per_ngram")
        let ngramSize = try i("ngram_size")
        guard headsPerNgram > 0, ngramSize > 1 else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "heads_per_ngram \(headsPerNgram) and ngram_size "
                    + "\(ngramSize) must both be positive")
        }
        // One block of heads per n-gram order from 2 up to `ngram_size`, which
        // is `ngram_size - 1` blocks of `heads_per_ngram`. The pinned
        // checkpoint's `ngram_heads_offsets` carries exactly this many entries.
        //
        // Not derived from the shard count: 128 shards over 8 heads per n-gram
        // also lands on 16 for this checkpoint, and would be wrong for any
        // other split.
        let ngramHeads = (ngramSize - 1) * headsPerNgram

        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("shared_expert_intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: (tc["hidden_act"] as? String) ?? "silu",
            family: .qwen4Exp,
            variant: .qwen38FlashNext,
            // Not declared in this config the way Qwen3.6 declares
            // `attn_output_gate`, but the checkpoint's q_proj emits
            // `2 * num_attention_heads * head_dim` rows — the per-head
            // [query ; gate] pair — so the gate is present.
            attnOutputGate: true,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: try i("linear_num_key_heads"),
            linearNumVHeads: try i("linear_num_value_heads"),
            linearKeyHeadDim: try i("linear_key_head_dim"),
            linearValueHeadDim: try i("linear_value_head_dim"),
            linearConvKernelSize: try i("linear_conv_kernel_dim"),
            hyperConnectionStreamCount: try i("hc_count"),
            hyperConnectionLowRank: try i("hc_lowrank"),
            ngramLayer: declaredPLELayer - 1,
            ngramSize: ngramSize,
            ngramHeads: ngramHeads,
            ngramHeadsPerNgram: headsPerNgram,
            ngramVocabSizeBase: try i("ngram_vocab_size_base"),
            ngramShardCount: shardCount,
            ngramEmbedDim: try i("ple_embed_dim"),
            ngramConvKernelSize: try i("ple_conv_kernel_size"),
            ngramEosTokenID: try i("eos_token_id"),
            indexerBudget: try i("indexer_budget"),
            indexerCompressRatio: try i("indexer_compress_ratio"),
            indexerHeadDim: try i("indexer_head_dim"),
            indexerNumHeads: try i("indexer_n_heads"),
            indexerNumKVHeads: try i("indexer_kv_heads"))
        try crossCheckProductionQwen4Exp(arch, configPath: configPath)
        return arch
    }

    /// Production Qwen3.8 Flash Next baseline, mirroring the runtime's
    /// `ArchConfig.qwen38FlashNext`. A config that matches the production
    /// shape must agree on every field a repack depends on; synthetic configs
    /// with other dimensions are exempt.
    private static func crossCheckProductionQwen4Exp(_ a: ArchInfo,
                                                     configPath: String) throws {
        guard a.hiddenSize == 2_560, a.numLayers == 48 else { return }
        var expectedMask = [UInt8](repeating: 2, count: 48)
        for i in stride(from: 3, to: 48, by: 4) { expectedMask[i] = 1 }
        guard a.intermediateSize == 640,
              a.moeIntermediateSize == 640,
              a.numHeads == 24,
              a.numKVHeads == 2,
              a.headDim == 256,
              a.vocabSize == 248_320,
              a.ropeTheta == 10_000_000,
              a.partialRotaryFactor == 0.25,
              a.numExperts == 512,
              a.topKExperts == 10,
              !a.tieWordEmbeddings,
              a.hiddenActivation == "silu",
              a.fullAttentionLayerMask == expectedMask,
              a.linearNumKHeads == 16,
              a.linearNumVHeads == 48,
              a.linearKeyHeadDim == 128,
              a.linearValueHeadDim == 128,
              a.linearConvKernelSize == 4,
              a.hyperConnectionStreamCount == 4,
              a.hyperConnectionLowRank == 320,
              a.ngramLayer == 1,
              a.ngramSize == 3,
              a.ngramHeads == 16,
              a.ngramHeadsPerNgram == 8,
              a.ngramVocabSizeBase == 20_000_000,
              a.ngramShardCount == 128,
              a.ngramEmbedDim == 2_560,
              a.ngramConvKernelSize == 4,
              a.ngramEosTokenID == 248_044,
              a.indexerBudget == 2_048,
              a.indexerCompressRatio == 4,
              a.indexerHeadDim == 128,
              a.indexerNumHeads == 4,
              a.indexerNumKVHeads == 1 else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "config.json does not match the pinned Qwen3.8 Flash "
                    + "Next architecture")
        }
    }

    /// Production Qwen3.6-35B-A3B baseline (mirrors the runtime's
    /// `ArchConfig.qwen36_35B_A3B`; the repack target has no dependency on the
    /// runtime module). A config that matches the production shape
    /// (hidden 2048, 40 layers) must agree on every field; toy/synthetic
    /// configs are exempt.
    private static func crossCheckProductionQwen36(_ a: ArchInfo,
                                                   configPath: String) throws {
        guard a.hiddenSize == 2048, a.numLayers == 40 else { return }
        var expectedMask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { expectedMask[i] = 1 }
        let expected = ArchInfo(
            hiddenSize: 2048,
            intermediateSize: 512,
            moeIntermediateSize: 512,
            numHeads: 16,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 256,
            fullHeadDim: 256,
            vocabSize: 248_320,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 40,
            numExperts: 256,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .qwen36,
            variant: .qwen36_35B_A3B,
            attnOutputGate: true,
            attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 16,
            linearNumVHeads: 32,
            linearKeyHeadDim: 128,
            linearValueHeadDim: 128,
            linearConvKernelSize: 4)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen3_5_moe config does not match the pinned "
                    + "Qwen3.6-35B-A3B architecture baseline")
        }
    }
}
