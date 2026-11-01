import Foundation
import TUFFFormat

/// On-disk page alignment unit for `.gturbo` files. Fixed at 16 KB regardless
/// of host page size — the format is the contract, not the kernel.
enum Layout {
    static let pageBytes = GTurboFormatV1.alignmentBytes
}

// MARK: - Plan data types

struct ResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    /// File offset where the (packed) weight bytes start.
    let fileOffset: UInt64
    /// Size in bytes of the weight bytes.
    let sizeBytes: UInt64
    /// Offset where BF16 scales start (0 if none).
    let scaleOffset: UInt64
    let scaleSize: UInt64
    /// Offset where BF16 biases start (0 if none).
    let biasOffset: UInt64
    let biasSize: UInt64
    /// Quantization spec (nil for unquantized scalars/norms).
    let quantSpec: QuantSpec?

    /// Source tensors that supply this entry's bytes.
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor?
    let sourceBiases: SourceTensor?
    /// MiniMax publishes the quantized router's affine companions as FP32.
    /// TUFF's resident affine format is BF16, so those two small tensors are
    /// converted while streaming instead of being staged as a second model.
    let companionTransform: RangeCopyTransform
    let weightTransform: RangeCopyTransform
    let scaleTransform: RangeCopyTransform
    let biasTransform: RangeCopyTransform

    init(name: String,
         dtype: UInt8,
         logicalShape4: [UInt32],
         fileOffset: UInt64,
         sizeBytes: UInt64,
         scaleOffset: UInt64,
         scaleSize: UInt64,
         biasOffset: UInt64,
         biasSize: UInt64,
         quantSpec: QuantSpec?,
         sourceWeight: SourceTensor,
         sourceScales: SourceTensor?,
         sourceBiases: SourceTensor?,
         companionTransform: RangeCopyTransform = .identity,
         weightTransform: RangeCopyTransform = .identity,
         scaleTransform: RangeCopyTransform? = nil, biasTransform: RangeCopyTransform? = nil) {
        self.name = name
        self.dtype = dtype
        self.logicalShape4 = logicalShape4
        self.fileOffset = fileOffset
        self.sizeBytes = sizeBytes
        self.scaleOffset = scaleOffset
        self.scaleSize = scaleSize
        self.biasOffset = biasOffset
        self.biasSize = biasSize
        self.quantSpec = quantSpec
        self.sourceWeight = sourceWeight
        self.sourceScales = sourceScales
        self.sourceBiases = sourceBiases
        self.companionTransform = companionTransform
        self.weightTransform = weightTransform
        self.scaleTransform = scaleTransform ?? companionTransform
        self.biasTransform = biasTransform ?? companionTransform
    }
}

struct ResidentFilePlan: Sendable {
    let path: String
    let entries: [ResidentEntry]
    let stringTable: [UInt8]
    let stringTableOffsets: [UInt32]   // per-entry offsets into the table
    let indexSize: UInt64              // header + entries + table + padding
    let residentSize: UInt64           // tensor payload region
    var totalSize: UInt64 { indexSize + residentSize }
}

struct PerExpertTensorSlice: Sendable {
    let role: String                   // affine roles or GPT-OSS "mlp1"/"mlp2"
    let component: String              // weights | scales | bias/biases
    let dtype: UInt8                   // 0=U32, 1=BF16, 4=U8
    let logicalShape: [UInt64]         // per-expert logical shape
    let offsetInExpertBlob: UInt64     // within each expert blob
    let sizeInExpertBlob: UInt64
    /// For each expert e (0..<expertsPerLayer): source byte offset & size.
    let sourceOffsetPerExpert: UInt64  // stride per expert in source
    let sourceTensor: SourceTensor
    let bitsForWeights: Int?           // 4 for routed expert weight; nil for scales/biases
}

struct LayerFilePlan: Sendable {
    let layerIndex: Int
    let path: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    let subTensors: [PerExpertTensorSlice]  // 9 entries: gate/up/down × {weights, scales, biases}
    var fileSize: UInt64 { UInt64(expertsPerLayer) * expertStride }

    func physicalRank(for logicalExpert: Int) -> Int {
        logicalExpert
    }

    init(layerIndex: Int,
                path: String,
                expertsPerLayer: Int,
                expertStride: UInt64,
                subTensors: [PerExpertTensorSlice]) {
        self.layerIndex = layerIndex
        self.path = path
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.subTensors = subTensors
    }
}

/// One shard of the n-gram PLE table. The three source tensors are copied
/// verbatim into three contiguous regions, keeping the shard's own row order.
struct NgramShardPlan: Sendable {
    let shardIndex: Int
    /// First global row this shard holds. Shards are contiguous and ordered.
    let rowStart: UInt64
    let rowCount: UInt64
    /// Byte offsets of the three regions within the table file.
    let weightOffset: UInt64
    let scaleOffset: UInt64
    let biasOffset: UInt64
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor
    let sourceBiases: SourceTensor

    var weightSize: UInt64 { sourceWeight.sizeBytes }
    var scaleSize: UInt64 { sourceScales.sizeBytes }
    var biasSize: UInt64 { sourceBiases.sizeBytes }
}

/// The n-gram per-layer-embedding table, held in one file outside the resident
/// image.
///
/// The table is 29.80 GiB in the pinned checkpoint while a token reads only a
/// handful of its 100-byte rows, so it is neither resident nor a slot-cached
/// stream: it is mapped and rows are faulted in on demand. Rounding rows up to
/// a page the way routed experts are rounded would turn 100 bytes into 16 KiB
/// and the table into terabytes, which is why it has a class of its own.
///
/// Regions stay separate rather than interleaving weight, scales and biases
/// into one record per row. Interleaving would be one page fault per row
/// instead of three, but expressing it needs a strided scatter that the range
/// copier has no vocabulary for, and against the 1.47 GB of routed-expert
/// traffic a token already pays, the difference does not register.
struct NgramTablePlan: Sendable {
    let path: String
    /// Decoder layer the module belongs to.
    let layerIndex: Int
    /// Quantized values per row, before packing.
    let rowWidth: Int
    let groupSize: Int
    let rowCount: UInt64
    let shards: [NgramShardPlan]
    let fileSize: UInt64
}

struct RepackPlan: Sendable {
    let arch: ArchInfo
    let baseMode: String                  // "affine"
    let baseGroupSize: Int                // 64
    let bitsOverrideCount: Int
    let resident: ResidentFilePlan
    let layers: [LayerFilePlan]
    /// Nil for every architecture without an n-gram embedding table.
    let ngramTable: NgramTablePlan?
    let matchedModelID: String?
    let excludedMultimodalTensorNames: [String]
    var sourceCopies: [RangeCopy]? = nil
}

// MARK: - Planner

struct VisionPackPlan: Sendable {
    struct Entry: Sendable {
        let source: SourceTensor
        let executionPosition: Int
        let fileOffset: UInt64
        let quantSpec: QuantSpec?
        /// Affine group size, pinned to 64 by `planVisionCompanion`.
        let groupSize: Int
        /// The pack's name and shape: MLX's, whatever the source called it.
        let name: String
        let shape: [UInt64]
        /// A Transformers Conv3D kernel, `[out, channels, t, h, w]`, that has
        /// to be reordered channels-last after it is copied.
        let reordersChannelsLast: Bool

        init(source: SourceTensor, executionPosition: Int, fileOffset: UInt64,
             quantSpec: QuantSpec?, groupSize: Int, name: String? = nil,
             shape: [UInt64]? = nil, reordersChannelsLast: Bool = false) {
            self.source = source
            self.executionPosition = executionPosition
            self.fileOffset = fileOffset
            self.quantSpec = quantSpec
            self.groupSize = groupSize
            self.name = name ?? source.name
            self.shape = shape ?? source.shape
            self.reordersChannelsLast = reordersChannelsLast
        }
    }

    let entries: [Entry]
    let weightsFileSize: UInt64
    let sourcePayloadBytes: UInt64
}

enum RepackPlanner {

    static func planVisionCompanion(
        meta: IndexLoader.SourceMetadata,
        shardHeaders: [Safetensors.Header]
    ) throws -> VisionPackPlan {
        // The vision tensors are BF16 in every supported checkpoint, so the
        // base quantization only has to be one this repacker understands —
        // not one particular group size. Qwen3.8 Flash Next is grouped at 32
        // because its 160-wide PLE rows cannot be grouped at 64, and rejecting
        // it here would have refused a tower that is bit-identical to one
        // already shipping.
        // A BF16 release or the compressed 397B checkpoint qualifies too: the
        // tower is copied, never dequantized, and every tensor is checked to
        // be BF16 below.
        let affine = meta.baseBits == 4
            && (meta.baseGroupSize == 64 || meta.baseGroupSize == 32)
            && meta.baseMode.lowercased() == "affine"
        guard affine || meta.baseMode == "bf16" || meta.baseMode == "compressed-affine" else {
            throw RepackError.configurationInvalid(
                detail: "vision companion requires MLX affine 4-bit source metadata")
        }

        guard let modelID = SourceFingerprint.modelID(
            forIndexSha256: meta.indexSha256Hex) else {
            throw RepackError.sourceFingerprintRejected(
                path: meta.indexPath, sha256: meta.indexSha256Hex)
        }
        let tensors = shardHeaders.flatMap(\.tensors).filter {
            isVisionTensorName(canonicalVisionName($0.name))
        }
        let expected: (count: Int, bytes: UInt64)
        switch modelID {
        case SupportedModelSource.qwen36.modelID:
            expected = (333, 893_142_496)
        case SupportedModelSource.qwen38FlashNext.modelID:
            // The same 333 tensors as Qwen 3.6, and exactly 4,719,616 bytes
            // more: (2560 - 2048) * 4608 * 2 for the merger's second linear
            // and (2560 - 2048) * 2 for its bias. Nothing else in the tower
            // differs.
            expected = (333, 897_862_112)
        case SupportedModelSource.qwen38_27B.modelID:
            // Qwen 3.6's tower again; only the merger widens, to the 5,120-wide
            // residual: (5120 - 2048) * 4608 * 2 + (5120 - 2048) * 2 bytes more.
            expected = (333, 921_460_192)
        case "ornith-ai/Ornith-1.5-35B-A3B":
            // Ornith's BF16 release of the 35B: Qwen 3.6's tower and width.
            expected = (333, 893_142_496)
        case "ornith-ai/Ornith-1.5-122B":
            // Into a 3,072-wide residual: (3072 - 2048) * 4608 * 2 + (3072 -
            // 2048) * 2 bytes more than Qwen 3.6.
            expected = (333, 902_581_728)
        case "ornith-ai/Ornith-1.5-9B", SupportedModelSource.ornith15_397B.modelID:
            // The same tower into a 4,096-wide residual: (4096 - 2048) * 4608
            // * 2 + (4096 - 2048) * 2 bytes more than Qwen 3.6.
            expected = (333, 912_020_960)
        case SupportedModelSource.gemma4E2B.modelID:
            expected = (213, 335_392_768)
        case SupportedModelSource.gemma4E4B.modelID:
            // E4B uses the same 768-wide, 16-layer Gemma vision tower as
            // E2B. Its only vision-side shape difference is the wider final
            // projection into the 2,560-wide language model.
            expected = (213, 335_835_136)
        case SupportedModelSource.gemma4_12B_QAT.modelID:
            expected = (14, 40_493_568)
        default:
            expected = (358, 1_140_925_536)
        }
        let expectedTensorCount = expected.count
        let expectedSourceBytes = expected.bytes
        guard tensors.count == expectedTensorCount else {
            throw RepackError.configurationInvalid(
                detail: "expected \(expectedTensorCount) vision tensors for \(modelID), found \(tensors.count)")
        }
        let sourceBytes = tensors.reduce(UInt64(0)) { $0 + $1.sizeBytes }
        guard sourceBytes == expectedSourceBytes else {
            throw RepackError.configurationInvalid(
                detail: "expected \(expectedSourceBytes) vision bytes for \(modelID), found \(sourceBytes)")
        }

        let ordered = tensors.sorted {
            visionExecutionKey(canonicalVisionName($0.name)) < visionExecutionKey(canonicalVisionName($1.name))
        }
        var offset: UInt64 = 0
        var entries: [VisionPackPlan.Entry] = []
        entries.reserveCapacity(ordered.count)
        for (position, tensor) in ordered.enumerated() {
            offset = visionAlignUp(offset, to: GTurboVisionFormatV1.alignmentBytes)
            let quantSpec: QuantSpec?
            if tensor.dtype == .u32 {
                guard affine else {
                    throw RepackError.configurationInvalid(
                        detail: "unsupported vision quantization for \(tensor.name)")
                }
                let spec = IndexLoader.quantSpec(forTensor: tensor.name, meta: meta)
                // Group size is not carried per tensor in this tree; the base
                // group was already pinned to 64 above.
                guard spec.bits == 4 else {
                    throw RepackError.configurationInvalid(
                        detail: "unsupported vision quantization for \(tensor.name)")
                }
                quantSpec = spec
            } else {
                guard tensor.dtype == .bf16 else {
                    throw RepackError.configurationInvalid(
                        detail: "unsupported vision dtype for \(tensor.name)")
                }
                quantSpec = nil
            }
            let name = canonicalVisionName(tensor.name)
            let reorders = name == "vision_tower.patch_embed.proj.weight"
                && tensor.shape.count == 5 && tensor.shape[1] == 3
            let shape = reorders
                ? [tensor.shape[0], tensor.shape[2], tensor.shape[3], tensor.shape[4], tensor.shape[1]]
                : tensor.shape
            entries.append(.init(source: tensor,
                                 executionPosition: position,
                                 fileOffset: offset,
                                 quantSpec: quantSpec,
                                 groupSize: meta.baseGroupSize,
                                 name: name, shape: shape, reordersChannelsLast: reorders))
            offset += tensor.sizeBytes
        }
        return VisionPackPlan(entries: entries,
                              weightsFileSize: offset,
                              sourcePayloadBytes: sourceBytes)
    }

    /// Transformers names the Qwen tower `model.visual.*`; MLX, and the pack,
    /// `vision_tower.*`.
    static func canonicalVisionName(_ name: String) -> String {
        name.hasPrefix("model.visual.") ? "vision_tower." + name.dropFirst("model.visual.".count) : name
    }

    private static func visionExecutionKey(_ name: String) -> String {
        if name.hasPrefix("vision_tower.patch_embed.") {
            return "0000/\(name)"
        }
        if name.hasPrefix("vision_tower.patch_embedder.") {
            return "0000/\(name)"
        }
        if let layer = layerIndex(in: name), name.hasPrefix("vision_tower.blocks.") {
            return String(format: "1000/%03d/%@", layer, name)
        }
        if let layer = layerIndex(in: name), name.hasPrefix("vision_tower.encoder.layers.") {
            return String(format: "1000/%03d/%@", layer, name)
        }
        if name.hasPrefix("vision_tower.merger.") || name == "vision_tower.pos_embed.weight" {
            return "3000/\(name)"
        }
        if name == "vision_tower.std_bias" || name == "vision_tower.std_scale" {
            return "2000/\(name)"
        }
        if name.hasPrefix("embed_vision.") {
            return "3000/\(name)"
        }
        if name.hasPrefix("vision_embedder.patch_ln1.") {
            return "0000/\(name)"
        }
        if name.hasPrefix("vision_embedder.patch_dense.") {
            return "1000/\(name)"
        }
        if name.hasPrefix("vision_embedder.patch_ln2.") {
            return "2000/\(name)"
        }
        if name.hasPrefix("vision_embedder.pos_embedding") {
            return "3000/\(name)"
        }
        if name.hasPrefix("vision_embedder.pos_norm.") {
            return "4000/\(name)"
        }
        return "9999/\(name)"
    }

    private static func visionAlignUp(_ value: UInt64, to alignment: UInt64) -> UInt64 {
        ((value + alignment - 1) / alignment) * alignment
    }

    /// Classify a tensor name. Routed-expert tensors split off the LM bucket.
    enum Bucket: Equatable {
        case lmResident
        case routedExpert(role: String, layer: Int)   // role = "gate"|"up"|"down"
        /// One shard of a `qwen4_exp` n-gram per-layer-embedding table.
        case ngramEmbeddingShard(shard: Int, layer: Int)
        case excludedMultimodal
        case unknown
    }

    static func classify(_ name: String, numLayers: Int,
                         family: RepackModelFamily) -> Bucket {
        if family == .gptOss,
           (name == "lm_head.weight" || name.hasPrefix("model.")) {
            if name.contains(".mlp.experts."),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(
                    role: name.contains("gate_up_proj") ? "mlp1" : "mlp2",
                    layer: layer)
            }
            return .lmResident
        }
        if family == .minimaxM2,
           (name == "lm_head.weight" || name.hasPrefix("model.")) {
            if let role = routedExpertRole(in: name, family: family),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(role: role, layer: layer)
            }
            return .lmResident
        }
        if name.hasPrefix("language_model.") {
            // Routed expert?
            if let role = routedExpertRole(in: name, family: family),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(role: role, layer: layer)
            }
            // The n-gram table is orders of magnitude larger than every other
            // resident tensor combined, so it never joins the resident bucket.
            if let shard = ngramShardIndex(in: name),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .ngramEmbeddingShard(shard: shard, layer: layer)
            }
            return .lmResident
        }
        if isMultimodalTensorName(name) {
            return .excludedMultimodal
        }
        return .unknown
    }

    private static func routedExpertRole(in name: String,
                                         family: RepackModelFamily) -> String? {
        let routedContainer: String
        switch family {
        case .gemma4: routedContainer = ".experts.switch_glu."
        case .qwen36: routedContainer = ".mlp.switch_mlp."
        case .gptOss: routedContainer = ".mlp.experts."
        case .minimaxM2: routedContainer = ".block_sparse_moe.switch_mlp."
        case .qwen4Exp, .flint: routedContainer = ".mlp.switch_mlp."
        }
        guard name.contains(routedContainer) else { return nil }
        if name.contains(".gate_proj.") { return "gate" }
        if name.contains(".up_proj.")   { return "up" }
        if name.contains(".down_proj.") { return "down" }
        return nil
    }

    /// Shard index of a `...ple_embedding.ngram_embedding.shards.<N>...`
    /// tensor, or nil when the name is not one.
    static func ngramShardIndex(in name: String) -> Int? {
        guard let r = name.range(of: ".ngram_embedding.shards.") else { return nil }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    private static func layerIndex(in name: String) -> Int? {
        // matches either language/legacy vision "...layers.<N>..." or the
        // Qwen vision tower's "...blocks.<N>...".
        guard let r = name.range(of: ".layers.") ?? name.range(of: ".blocks.") else {
            return nil
        }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// Build the plan from parsed shard headers + source metadata.
    /// - throws: classification + companion + override count failures.
    static func plan(meta: IndexLoader.SourceMetadata,
                            arch: ArchInfo,
                            shardHeaders: [Safetensors.Header],
                            outputDir: String) throws -> RepackPlan {

        if meta.baseMode == "compressed-affine" {
            return try CompressedQwenRepackPlanner.plan(meta: meta, arch: arch, headers: shardHeaders, outputDir: outputDir)
        }

        // Companion tensors may live in different shards, so resolve them
        // through one global registry.
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        // Source allowlisting owns exact fingerprint validation. Preserve the
        // declared override count for the output manifest audit.
        let bitsOverrideCount = meta.bitsOverrides.count

        if arch.family == .gptOss {
            return try planGPTOSS(
                meta: meta,
                arch: arch,
                registry: registry,
                outputDir: outputDir,
                bitsOverrideCount: bitsOverrideCount)
        }

        var lmResidentBases: [String] = []
        var excludedMultimodalNames: [String] = []
        var routedByLayerAndRole: [Int: [String: String]] = [:]
        var ngramShardBases: [Int: (Int, String)] = [:]
        for (name, _) in registry {
            if isMultimodalTensorName(name) {
                excludedMultimodalNames.append(name)
            }
            if name.hasSuffix(".scales") || name.hasSuffix(".biases") { continue }
            let b = classify(name, numLayers: arch.numLayers, family: arch.family)
            switch b {
            case .lmResident:                   lmResidentBases.append(name)
            case .routedExpert(let role, let layer):
                var byRole = routedByLayerAndRole[layer] ?? [:]
                if byRole[role] != nil {
                    throw RepackError.configurationInvalid(detail:
                        "two routed-expert tensors for layer \(layer) role \(role)")
                }
                byRole[role] = name
                routedByLayerAndRole[layer] = byRole
            case .ngramEmbeddingShard(let shard, let layer):
                if let existing = ngramShardBases[shard] {
                    throw RepackError.configurationInvalid(detail:
                        "two n-gram shards numbered \(shard): \(existing.1), \(name)")
                }
                ngramShardBases[shard] = (layer, name)
            case .excludedMultimodal:           continue
            case .unknown:                      throw RepackError.unknownTensorPrefix(name: name)
            }
        }

        // Sort deterministically. The LM order follows a fixed template.
        lmResidentBases.sort(by: lmResidentOrdering(family: arch.family))
        excludedMultimodalNames.sort()

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath,
                                            baseNames: lmResidentBases,
                                            registry: registry, meta: meta,
                                            family: arch.family)

        var layerPlans: [LayerFilePlan] = []
        if arch.feedForwardKind == .mixtureOfExperts {
            let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
            layerPlans.reserveCapacity(arch.numLayers)
            for layer in 0..<arch.numLayers {
                let bundle = routedByLayerAndRole[layer] ?? [:]
                // Synthetic snapshots may legitimately have no routed experts.
                guard let gName = bundle["gate"], let uName = bundle["up"],
                      let dName = bundle["down"] else {
                    if bundle.isEmpty {
                        let name = "layer_\(String(format: "%02d", layer)).bin"
                        layerPlans.append(LayerFilePlan(
                            layerIndex: layer,
                            path: (layersDir as NSString).appendingPathComponent(name),
                            expertsPerLayer: 0,
                            expertStride: 0,
                            subTensors: []))
                        continue
                    }
                    throw RepackError.configurationInvalid(detail:
                        "layer \(layer) routed-expert bundle incomplete: \(bundle)")
                }
                let path = (layersDir as NSString)
                    .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
                let lp = try planLayerFile(
                    path: path, layer: layer,
                    gateName: gName, upName: uName, downName: dName,
                    registry: registry, meta: meta, arch: arch)
                layerPlans.append(lp)
            }
        } else if !routedByLayerAndRole.isEmpty {
            throw RepackError.configurationInvalid(
                detail: "dense architecture contains routed-expert tensors")
        }

        let ngramTable = try planNgramTable(
            shardBases: ngramShardBases,
            registry: registry,
            arch: arch,
            outputDir: outputDir)

        let matched = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex)

        return RepackPlan(arch: arch,
                          baseMode: meta.baseMode,
                          baseGroupSize: meta.baseGroupSize,
                          bitsOverrideCount: bitsOverrideCount,
                          resident: resident,
                          layers: layerPlans,
                          ngramTable: ngramTable,
                          matchedModelID: matched,
                          excludedMultimodalTensorNames: excludedMultimodalNames)
    }

    private static func isMultimodalTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") ||
            name.hasPrefix("vision_embedder.") ||
            name.hasPrefix("embed_vision.") ||
            name.hasPrefix("audio_tower.") ||
            name.hasPrefix("embed_audio.") ||
            name.hasPrefix("embed_video.")
    }

    private static func isVisionTensorName(_ name: String) -> Bool {
        guard name.hasPrefix("vision_tower.")
                || name.hasPrefix("vision_embedder.")
                || name.hasPrefix("embed_vision.") else { return false }
        // E2B's clipped BF16 linears carry calibration extrema that are not
        // weights and are never consumed at inference time.
        return !name.hasSuffix(".input_min")
            && !name.hasSuffix(".input_max")
            && !name.hasSuffix(".output_min")
            && !name.hasSuffix(".output_max")
    }

    // MARK: - Resident planning

    private static func planResidentFile(path: String,
                                         baseNames: [String],
                                         registry: [String: SourceTensor],
                                         meta: IndexLoader.SourceMetadata,
                                         family: RepackModelFamily) throws
                                        -> ResidentFilePlan {
        let entryCount = baseNames.count

        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(entryCount)
        for sourceName in baseNames {
            let n = canonicalResidentName(sourceName, family: family)
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: n.utf8)
        }

        // Index size includes the fixed header, fixed-width entries, and the
        // string table, padded to a 16 KB page boundary.
        let rawIdx = UInt64(GTurboBinary.indexHeaderBytes
            + entryCount * GTurboBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        var fileCursor = indexSize
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(entryCount)

        for sourceName in baseNames {
            let name = canonicalResidentName(sourceName, family: family)
            guard let weight = registry[sourceName] else {
                throw RepackError.missingTensor(name: sourceName)
            }
            let dtype = ietnyDtype(weight.dtype)
            let isQuantizedPacked = (weight.dtype == .u32)
                && sourceName.hasSuffix(".weight")

            if isQuantizedPacked {
                let base = String(sourceName.dropLast(".weight".count))
                guard let scales = registry[base + ".scales"] else {
                    throw RepackError.missingScalesCompanion(name: name)
                }
                guard let biases = registry[base + ".biases"] else {
                    throw RepackError.missingBiasesCompanion(name: name)
                }
                let companionTransform: RangeCopyTransform
                if scales.dtype == .bf16, biases.dtype == .bf16 {
                    companionTransform = .identity
                } else if family == .minimaxM2,
                          sourceName.hasSuffix(".block_sparse_moe.gate.weight"),
                          scales.dtype == .fp32, biases.dtype == .fp32 {
                    companionTransform = .fp32ToBF16
                } else {
                    throw RepackError.dtypeMismatch(name: name,
                        detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
                }
                let spec = IndexLoader.quantSpec(forTensor: sourceName, meta: meta)
                let logical = logicalShape(forPackedSource: weight.shape, bits: spec.bits)

                let wOff = fileCursor
                let wSize = weight.sizeBytes
                let sOff = wOff + wSize
                let sSize = try companionTransform.destinationSize(
                    sourceBytes: scales.sizeBytes, name: scales.name)
                let bOff = sOff + sSize
                let bSize = try companionTransform.destinationSize(
                    sourceBytes: biases.sizeBytes, name: biases.name)
                fileCursor = bOff + bSize

                entries.append(ResidentEntry(
                    name: name, dtype: GTurboFormatV1.DType.u32.rawValue,
                    logicalShape4: padTo4(logical),
                    fileOffset: wOff, sizeBytes: wSize,
                    scaleOffset: sOff, scaleSize: sSize,
                    biasOffset: bOff, biasSize: bSize,
                    quantSpec: spec,
                    sourceWeight: weight, sourceScales: scales, sourceBiases: biases,
                    companionTransform: companionTransform))
            } else {
                // Unquantized (BF16 norm / scalar) — no companions.
                let off = fileCursor
                let size = weight.sizeBytes
                fileCursor = off + size

                entries.append(ResidentEntry(
                    name: name, dtype: dtype,
                    logicalShape4: padTo4(weight.shape),
                    fileOffset: off, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil,
                    companionTransform: .identity))
            }
        }

        let residentSize = fileCursor - indexSize

        return ResidentFilePlan(path: path,
                                entries: entries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: residentSize)
    }

    /// Lay the n-gram shards out in one file, shard by shard in index order,
    /// three contiguous regions each.
    ///
    /// Shards must be complete and consecutively numbered: a gap would make
    /// global row ids meaningless, since a row's shard is found by dividing
    /// through a uniform row count.
    private static func planNgramTable(
        shardBases: [Int: (Int, String)],
        registry: [String: SourceTensor],
        arch: ArchInfo,
        outputDir: String
    ) throws -> NgramTablePlan? {
        guard !shardBases.isEmpty else {
            guard arch.ngramShardCount == 0 else {
                throw RepackError.configurationInvalid(detail:
                    "architecture declares \(arch.ngramShardCount) n-gram shards "
                        + "but the checkpoint has none")
            }
            return nil
        }
        guard arch.ngramShardCount == shardBases.count else {
            throw RepackError.configurationInvalid(detail:
                "architecture declares \(arch.ngramShardCount) n-gram shards, "
                    + "checkpoint has \(shardBases.count)")
        }
        let layers = Set(shardBases.values.map(\.0))
        guard layers.count == 1, let layerIndex = layers.first,
              layerIndex == arch.ngramLayer else {
            throw RepackError.configurationInvalid(detail:
                "n-gram shards span layers \(layers.sorted()), expected "
                    + "layer \(arch.ngramLayer)")
        }

        let path = (outputDir as NSString)
            .appendingPathComponent(ngramTableFileName)
        var shards: [NgramShardPlan] = []
        shards.reserveCapacity(shardBases.count)
        var cursor: UInt64 = 0
        var rowStart: UInt64 = 0
        var rowWidth = 0

        for index in 0..<shardBases.count {
            guard let (_, base) = shardBases[index] else {
                throw RepackError.configurationInvalid(detail:
                    "n-gram shard \(index) is missing")
            }
            let stem = String(base.dropLast(".weight".count))
            guard let weight = registry[base] else {
                throw RepackError.missingTensor(name: base)
            }
            guard let scales = registry[stem + ".scales"] else {
                throw RepackError.missingScalesCompanion(name: base)
            }
            guard let biases = registry[stem + ".biases"] else {
                throw RepackError.missingBiasesCompanion(name: base)
            }
            guard weight.dtype == .u32, scales.dtype == .bf16,
                  biases.dtype == .bf16 else {
                throw RepackError.dtypeMismatch(name: base, detail:
                    "expected U32 weight with BF16 scales and biases, got "
                        + "\(weight.dtype)/\(scales.dtype)/\(biases.dtype)")
            }
            guard weight.shape.count == 2, scales.shape.count == 2,
                  biases.shape.count == 2 else {
                throw RepackError.shapeMismatch(name: base,
                    detail: "n-gram shard tensors must be two-dimensional")
            }
            let rows = weight.shape[0]
            guard rows > 0, scales.shape[0] == rows, biases.shape[0] == rows else {
                throw RepackError.shapeMismatch(name: base, detail:
                    "n-gram shard row counts disagree: \(weight.shape[0])/"
                        + "\(scales.shape[0])/\(biases.shape[0])")
            }
            // Eight 4-bit values per packed U32 word.
            let width = Int(weight.shape[1]) * 8
            if rowWidth == 0 { rowWidth = width }
            guard width == rowWidth else {
                throw RepackError.shapeMismatch(name: base, detail:
                    "n-gram shard row width \(width) disagrees with \(rowWidth)")
            }
            guard width == arch.ngramEmbedDim / max(1, arch.ngramHeads) else {
                throw RepackError.shapeMismatch(name: base, detail:
                    "n-gram row width \(width) does not divide the embedding "
                        + "across \(arch.ngramHeads) heads")
            }
            let groups = UInt64(width / ngramGroupSize)
            guard width % ngramGroupSize == 0,
                  scales.shape[1] == groups, biases.shape[1] == groups else {
                throw RepackError.shapeMismatch(name: base, detail:
                    "n-gram scales and biases must carry one value per "
                        + "\(ngramGroupSize)-wide group")
            }

            let weightOffset = cursor
            let scaleOffset = weightOffset + weight.sizeBytes
            let biasOffset = scaleOffset + scales.sizeBytes
            cursor = biasOffset + biases.sizeBytes
            shards.append(NgramShardPlan(
                shardIndex: index,
                rowStart: rowStart,
                rowCount: rows,
                weightOffset: weightOffset,
                scaleOffset: scaleOffset,
                biasOffset: biasOffset,
                sourceWeight: weight,
                sourceScales: scales,
                sourceBiases: biases))
            rowStart += rows
        }

        return NgramTablePlan(
            path: path,
            layerIndex: layerIndex,
            rowWidth: rowWidth,
            groupSize: ngramGroupSize,
            rowCount: rowStart,
            shards: shards,
            fileSize: cursor)
    }

    /// The n-gram table's own file, beside `model_weights.bin`.
    static let ngramTableFileName = "ngram_ple.bin"
    /// The checkpoint quantizes the table at group size 32 rather than the
    /// 64 the rest of the affine weights use; the conversion notes call that
    /// out as what lets the PLE dimensions be quantized at all.
    static let ngramGroupSize = 32

    private static func canonicalResidentName(
        _ sourceName: String,
        family: RepackModelFamily
    ) -> String {
        guard family == .gptOss || family == .minimaxM2 else { return sourceName }
        if sourceName == "lm_head.weight" {
            return "language_model.lm_head.weight"
        }
        return "language_model." + sourceName
    }

    // MARK: - GPT-OSS planning

    private static func planGPTOSS(
        meta: IndexLoader.SourceMetadata,
        arch: ArchInfo,
        registry: [String: SourceTensor],
        outputDir: String,
        bitsOverrideCount: Int
    ) throws -> RepackPlan {
        guard meta.baseMode.lowercased() == "mxfp4",
              meta.baseBits == 4,
              meta.baseGroupSize == 32,
              bitsOverrideCount == 0 else {
            throw RepackError.configurationInvalid(
                detail: "GPT-OSS requires MXFP4 group-32 source metadata without overrides")
        }

        let expertMarker = ".mlp.experts."
        var residentNames: [String] = []
        for name in registry.keys {
            if name.contains(expertMarker) { continue }
            guard name == "lm_head.weight" || name.hasPrefix("model.") else {
                throw RepackError.unknownTensorPrefix(name: name)
            }
            residentNames.append(name)
        }
        residentNames.sort(by: lmResidentOrdering(family: .gptOss))
        let residentPath = (outputDir as NSString)
            .appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(
            path: residentPath,
            baseNames: residentNames,
            registry: registry,
            meta: meta,
            family: .gptOss)
        guard resident.entries.allSatisfy({ $0.dtype == GTurboFormatV1.DType.bf16.rawValue }) else {
            throw RepackError.configurationInvalid(
                detail: "GPT-OSS resident tensors must be BF16")
        }

        let layersDir = (outputDir as NSString)
            .appendingPathComponent("packed_experts")
        var layers: [LayerFilePlan] = []
        layers.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let prefix = "model.layers.\(layer).mlp.experts."
            let path = (layersDir as NSString).appendingPathComponent(
                "layer_\(String(format: "%02d", layer)).bin")
            layers.append(try planGPTOSSLayer(
                path: path,
                layer: layer,
                prefix: prefix,
                registry: registry,
                arch: arch))
        }

        return RepackPlan(
            arch: arch,
            baseMode: meta.baseMode,
            baseGroupSize: meta.baseGroupSize,
            bitsOverrideCount: bitsOverrideCount,
            resident: resident,
            layers: layers,
            // GPT-OSS has no n-gram table.
            ngramTable: nil,
            matchedModelID: SourceFingerprint.modelID(
                forIndexSha256: meta.indexSha256Hex),
            excludedMultimodalTensorNames: [])
    }

    private static func planGPTOSSLayer(
        path: String,
        layer: Int,
        prefix: String,
        registry: [String: SourceTensor],
        arch: ArchInfo
    ) throws -> LayerFilePlan {
        let sources: [(role: String, component: String, suffix: String)] = [
            ("mlp1", "weights", "gate_up_proj_blocks"),
            ("mlp1", "scales", "gate_up_proj_scales"),
            ("mlp1", "bias", "gate_up_proj_bias"),
            ("mlp2", "weights", "down_proj_blocks"),
            ("mlp2", "scales", "down_proj_scales"),
            ("mlp2", "bias", "down_proj_bias"),
        ]
        var slices: [PerExpertTensorSlice] = []
        var cursor: UInt64 = 0
        for source in sources {
            let name = prefix + source.suffix
            guard let tensor = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            let expectedDtype: SourceTensor.Dtype = source.component == "bias"
                ? .bf16 : .u8
            guard tensor.dtype == expectedDtype,
                  tensor.shape.first == UInt64(arch.numExperts) else {
                throw RepackError.shapeMismatch(
                    name: name,
                    detail: "expected \(expectedDtype) with leading \(arch.numExperts), got \(tensor.dtype) \(tensor.shape)")
            }
            let expectedShape: [UInt64]
            switch (source.role, source.component) {
            case ("mlp1", "weights"):
                expectedShape = [UInt64(arch.numExperts),
                                 UInt64(2 * arch.moeIntermediateSize),
                                 UInt64(arch.hiddenSize / 32), 16]
            case ("mlp1", "scales"):
                expectedShape = [UInt64(arch.numExperts),
                                 UInt64(2 * arch.moeIntermediateSize),
                                 UInt64(arch.hiddenSize / 32)]
            case ("mlp1", "bias"):
                expectedShape = [UInt64(arch.numExperts),
                                 UInt64(2 * arch.moeIntermediateSize)]
            case ("mlp2", "weights"):
                expectedShape = [UInt64(arch.numExperts),
                                 UInt64(arch.hiddenSize),
                                 UInt64(arch.moeIntermediateSize / 32), 16]
            case ("mlp2", "scales"):
                expectedShape = [UInt64(arch.numExperts),
                                 UInt64(arch.hiddenSize),
                                 UInt64(arch.moeIntermediateSize / 32)]
            default:
                expectedShape = [UInt64(arch.numExperts), UInt64(arch.hiddenSize)]
            }
            guard tensor.shape == expectedShape,
                  tensor.sizeBytes % UInt64(arch.numExperts) == 0 else {
                throw RepackError.shapeMismatch(
                    name: name,
                    detail: "expected shape \(expectedShape), got \(tensor.shape)")
            }
            let perExpert = tensor.sizeBytes / UInt64(arch.numExperts)
            let logicalShape: [UInt64]
            switch (source.role, source.component) {
            case ("mlp1", "weights"):
                logicalShape = [UInt64(2 * arch.moeIntermediateSize),
                                UInt64(arch.hiddenSize)]
            case ("mlp1", "scales"):
                logicalShape = [UInt64(2 * arch.moeIntermediateSize),
                                UInt64(arch.hiddenSize / 32)]
            case ("mlp1", "bias"):
                logicalShape = [UInt64(2 * arch.moeIntermediateSize)]
            case ("mlp2", "weights"):
                logicalShape = [UInt64(arch.hiddenSize),
                                UInt64(arch.moeIntermediateSize)]
            case ("mlp2", "scales"):
                logicalShape = [UInt64(arch.hiddenSize),
                                UInt64(arch.moeIntermediateSize / 32)]
            default:
                logicalShape = [UInt64(arch.hiddenSize)]
            }
            slices.append(PerExpertTensorSlice(
                role: source.role,
                component: source.component,
                dtype: tensor.dtype == .u8 ? SourceTensor.Dtype.u8.rawValue
                    : GTurboFormatV1.DType.bf16.rawValue,
                logicalShape: logicalShape,
                offsetInExpertBlob: cursor,
                sizeInExpertBlob: perExpert,
                sourceOffsetPerExpert: perExpert,
                sourceTensor: tensor,
                bitsForWeights: source.component == "weights" ? 4 : nil))
            cursor += perExpert
        }
        return LayerFilePlan(
            layerIndex: layer,
            path: path,
            expertsPerLayer: arch.numExperts,
            expertStride: roundUpToPage(cursor),
            subTensors: slices)
    }

    // MARK: - Layer planning

    private static func planLayerFile(path: String, layer: Int,
                                      gateName: String, upName: String, downName: String,
                                      registry: [String: SourceTensor],
                                      meta: IndexLoader.SourceMetadata,
                                      arch: ArchInfo) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        let roles: [(role: String, name: String)] = [
            ("gate", gateName), ("up", upName), ("down", downName)
        ]
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(9)
        var blobCursor: UInt64 = 0

        for (role, name) in roles {
            guard let w = registry[name] else { throw RepackError.missingTensor(name: name) }
            if w.dtype != .u32 || w.shape.count != 3 || Int(w.shape[0]) != expertCount {
                throw RepackError.shapeMismatch(name: name,
                    detail: "expected U32 rank-3 with leading \(expertCount), got \(w.dtype) \(w.shape)")
            }
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            guard let s = registry[base + ".scales"] else { throw RepackError.missingScalesCompanion(name: name) }
            guard let b = registry[base + ".biases"] else { throw RepackError.missingBiasesCompanion(name: name) }
            if s.dtype != .bf16 || b.dtype != .bf16 {
                throw RepackError.dtypeMismatch(name: name,
                    detail: "expected BF16 scales/biases, got \(s.dtype)/\(b.dtype)")
            }

            let perExpertWeightSize = w.sizeBytes / UInt64(expertCount)
            let perExpertScaleSize  = s.sizeBytes / UInt64(expertCount)
            let perExpertBiasSize   = b.sizeBytes / UInt64(expertCount)
            if perExpertWeightSize * UInt64(expertCount) != w.sizeBytes ||
               perExpertScaleSize  * UInt64(expertCount) != s.sizeBytes ||
               perExpertBiasSize   * UInt64(expertCount) != b.sizeBytes {
                throw RepackError.shapeMismatch(name: name,
                    detail: "source bytes not evenly divisible by \(expertCount) experts")
            }

            let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
            let perExpertSourceShape = Array(w.shape.dropFirst())
            let logicalPerExpert = logicalShape(forPackedSource: perExpertSourceShape, bits: spec.bits)
            let scalesLogical = Array(s.shape.dropFirst())
            let biasesLogical = Array(b.shape.dropFirst())

            let wSlice = PerExpertTensorSlice(
                role: role, component: "weights", dtype: GTurboFormatV1.DType.u32.rawValue,
                logicalShape: logicalPerExpert,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertWeightSize,
                sourceOffsetPerExpert: perExpertWeightSize, sourceTensor: w,
                bitsForWeights: spec.bits)
            blobCursor += perExpertWeightSize
            let sSlice = PerExpertTensorSlice(
                role: role, component: "scales", dtype: GTurboFormatV1.DType.bf16.rawValue,
                logicalShape: scalesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertScaleSize,
                sourceOffsetPerExpert: perExpertScaleSize, sourceTensor: s,
                bitsForWeights: nil)
            blobCursor += perExpertScaleSize
            let bSlice = PerExpertTensorSlice(
                role: role, component: "biases", dtype: GTurboFormatV1.DType.bf16.rawValue,
                logicalShape: biasesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertBiasSize,
                sourceOffsetPerExpert: perExpertBiasSize, sourceTensor: b,
                bitsForWeights: nil)
            blobCursor += perExpertBiasSize

            subs.append(wSlice); subs.append(sSlice); subs.append(bSlice)
        }

        let expertStride = roundUpToPage(blobCursor)
        return LayerFilePlan(layerIndex: layer, path: path,
                             expertsPerLayer: expertCount,
                             expertStride: expertStride,
                             subTensors: subs)
    }

    // MARK: - Helpers

    private static func ietnyDtype(_ d: SourceTensor.Dtype) -> UInt8 {
        switch d {
        case .u32: 0
        case .bf16: 1
        case .fp16: 2
        case .fp32: 3
        case .u8: 4
        case .i64: 5
        case .i32: 0
        }
    }

    private static func roundUpToPage(_ v: UInt64) -> UInt64 {
        let p = Layout.pageBytes
        return ((v + p - 1) / p) * p
    }

    private static func padTo4(_ s: [UInt64]) -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(4)
        for v in s.prefix(4) { out.append(UInt32(v)) }
        while out.count < 4 { out.append(0) }
        return out
    }

    /// Logical shape of a packed quantized tensor whose source is `[D0,..,Dn-1, Dn/factor]`.
    private static func logicalShape(forPackedSource source: [UInt64], bits: Int) -> [UInt64] {
        let factor = UInt64(32 / bits)
        guard !source.isEmpty else { return source }
        var out = source
        out[out.count - 1] = source[source.count - 1] * factor
        return out
    }

    /// Stable order for the resident LM tensor list. Embedding first, then
    /// per-layer groups in layer index order, then the final norm (and, for
    /// families with an untied head, `lm_head` last).
    private static func lmResidentOrdering(family: RepackModelFamily)
        -> (String, String) -> Bool {
        // Compute a sort key per name; we order by (group rank, layer, slot rank, name).
        func key(_ sourceName: String) -> (Int, Int, Int, String) {
            let n = canonicalResidentName(sourceName, family: family)
            if n == "language_model.model.embed_tokens.weight" { return (0, 0, 0, n) }
            if n == "language_model.model.embed_tokens_per_layer.weight" { return (0, 0, 1, n) }
            if n == "language_model.model.norm.weight"          { return (3, 0, 0, n) }
            if n == "language_model.lm_head.weight"             { return (4, 0, 0, n) }
            if let li = layerIndex(in: n) {
                let slot: Int
                switch family {
                case .gemma4: slot = slotRank(in: n)
                case .qwen36: slot = qwenSlotRank(in: n)
                case .gptOss: slot = gptOssSlotRank(in: n)
                case .minimaxM2: slot = minimaxSlotRank(in: n)
                case .qwen4Exp: slot = qwen4ExpSlotRank(in: n)
                case .flint: slot = qwenSlotRank(in: n)
                }
                return (1, li, slot, n)
            }
            return (2, 0, 0, n)
        }
        return { a, b in
            let ka = key(a), kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    private static func gptOssSlotRank(in name: String) -> Int {
        if name.hasSuffix(".input_layernorm.weight") { return 0 }
        if name.contains(".self_attn.q_proj.weight") { return 1 }
        if name.contains(".self_attn.q_proj.bias") { return 2 }
        if name.contains(".self_attn.k_proj.weight") { return 3 }
        if name.contains(".self_attn.k_proj.bias") { return 4 }
        if name.contains(".self_attn.v_proj.weight") { return 5 }
        if name.contains(".self_attn.v_proj.bias") { return 6 }
        if name.contains(".self_attn.o_proj.weight") { return 7 }
        if name.contains(".self_attn.o_proj.bias") { return 8 }
        if name.hasSuffix(".self_attn.sinks") { return 9 }
        if name.hasSuffix(".post_attention_layernorm.weight") { return 10 }
        if name.contains(".mlp.router.weight") { return 11 }
        if name.contains(".mlp.router.bias") { return 12 }
        return 100
    }

    private static func minimaxSlotRank(in n: String) -> Int {
        if n.hasSuffix(".input_layernorm.weight") { return 0 }
        if n.contains(".self_attn.q_proj.weight") { return 1 }
        if n.contains(".self_attn.k_proj.weight") { return 2 }
        if n.contains(".self_attn.v_proj.weight") { return 3 }
        if n.contains(".self_attn.o_proj.weight") { return 4 }
        if n.contains(".self_attn.q_norm.weight") { return 5 }
        if n.contains(".self_attn.k_norm.weight") { return 6 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 7 }
        if n.contains(".block_sparse_moe.gate.weight") { return 8 }
        if n.hasSuffix(".block_sparse_moe.e_score_correction_bias") { return 9 }
        return 100
    }

    /// Within-layer slot order for the Qwen 3.6 family: full-attention
    /// projections/norms, then the gated-DeltaNet linear-attention bundle,
    /// then router, shared-expert gate and MLP, then the two layer norms.
    private static func qwenSlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight")   { return 0 }
        if n.contains(".self_attn.k_proj.weight")   { return 1 }
        if n.contains(".self_attn.v_proj.weight")   { return 2 }
        if n.contains(".self_attn.o_proj.weight")   { return 3 }
        if n.contains(".self_attn.q_norm.weight")   { return 4 }
        if n.contains(".self_attn.k_norm.weight")   { return 5 }
        if n.contains(".linear_attn.in_proj_qkv.weight") { return 6 }
        if n.contains(".linear_attn.in_proj_z.weight")   { return 7 }
        if n.contains(".linear_attn.in_proj_a.weight")   { return 8 }
        if n.contains(".linear_attn.in_proj_b.weight")   { return 9 }
        if n.contains(".linear_attn.conv1d.weight")      { return 10 }
        if n.hasSuffix(".linear_attn.A_log")             { return 11 }
        if n.hasSuffix(".linear_attn.dt_bias")           { return 12 }
        if n.contains(".linear_attn.norm.weight")        { return 13 }
        if n.contains(".linear_attn.out_proj.weight")    { return 14 }
        if n.contains(".mlp.gate.weight")                { return 15 }
        if n.contains(".mlp.shared_expert_gate.weight")  { return 16 }
        if n.contains(".mlp.shared_expert.gate_proj.weight") { return 17 }
        if n.contains(".mlp.shared_expert.up_proj.weight")   { return 18 }
        if n.contains(".mlp.shared_expert.down_proj.weight") { return 19 }
        if n.hasSuffix(".input_layernorm.weight")        { return 20 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 21 }
        return 100
    }

    /// Within-layer slot order for `qwen4_exp`. The blocks it shares with
    /// Qwen 3.6 keep that order; the hyper-connection modules, which stand in
    /// for the two layer norms Qwen 3.6 has, bracket it, and the sparse
    /// indexer follows the full-attention projections it feeds.
    private static func qwen4ExpSlotRank(in n: String) -> Int {
        if n.contains(".attn_hyper_connection.hc_norm.weight") { return 0 }
        if n.contains(".attn_hyper_connection.input_mix_weight_down.weight") { return 1 }
        if n.contains(".attn_hyper_connection.input_mix_weight_up.weight") { return 2 }
        if n.contains(".attn_hyper_connection.block_inject_weight.weight") { return 3 }
        if n.contains(".self_attn.q_proj.weight")   { return 4 }
        if n.contains(".self_attn.k_proj.weight")   { return 5 }
        if n.contains(".self_attn.v_proj.weight")   { return 6 }
        if n.contains(".self_attn.o_proj.weight")   { return 7 }
        if n.contains(".self_attn.q_norm.weight")   { return 8 }
        if n.contains(".self_attn.k_norm.weight")   { return 9 }
        if n.contains(".self_attn.indexer.index_qk_proj.weight") { return 10 }
        if n.contains(".self_attn.indexer.q_layernorm.weight")   { return 11 }
        if n.contains(".self_attn.indexer.k_layernorm.weight")   { return 12 }
        if n.contains(".linear_attn.in_proj_qkv.weight") { return 13 }
        if n.contains(".linear_attn.in_proj_z.weight")   { return 14 }
        if n.contains(".linear_attn.in_proj_a.weight")   { return 15 }
        if n.contains(".linear_attn.in_proj_b.weight")   { return 16 }
        if n.contains(".linear_attn.conv1d.weight")      { return 17 }
        if n.hasSuffix(".linear_attn.A_log")             { return 18 }
        if n.hasSuffix(".linear_attn.dt_bias")           { return 19 }
        if n.contains(".linear_attn.norm.weight")        { return 20 }
        if n.contains(".linear_attn.out_proj.weight")    { return 21 }
        if n.contains(".mlp_hyper_connection.hc_norm.weight") { return 22 }
        if n.contains(".mlp_hyper_connection.input_mix_weight_down.weight") { return 23 }
        if n.contains(".mlp_hyper_connection.input_mix_weight_up.weight") { return 24 }
        if n.contains(".mlp_hyper_connection.block_inject_weight.weight") { return 25 }
        if n.contains(".mlp.gate.weight")                { return 26 }
        if n.contains(".mlp.shared_expert_gate.weight")  { return 27 }
        if n.contains(".mlp.shared_expert.gate_proj.weight") { return 28 }
        if n.contains(".mlp.shared_expert.up_proj.weight")   { return 29 }
        if n.contains(".mlp.shared_expert.down_proj.weight") { return 30 }
        if n.contains(".ple.")                           { return 31 }
        return 100
    }

    /// Within-layer slot order. Mirrors the per-layer description in the
    /// architecture doc.
    private static func slotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight") { return 0 }
        if n.contains(".self_attn.k_proj.weight") { return 1 }
        if n.contains(".self_attn.v_proj.weight") { return 2 }
        if n.contains(".self_attn.o_proj.weight") { return 3 }
        if n.contains(".self_attn.q_norm.weight") { return 4 }
        if n.contains(".self_attn.k_norm.weight") { return 5 }
        if n.contains(".router.proj.weight")      { return 6 }
        if n.contains(".router.scale")            { return 7 }
        if n.contains(".router.per_expert_scale") { return 8 }
        if n.contains(".mlp.gate_proj.weight")    { return 9 }
        if n.contains(".mlp.up_proj.weight")      { return 10 }
        if n.contains(".mlp.down_proj.weight")    { return 11 }
        if n.contains(".per_layer_input_gate.weight") { return 12 }
        if n.contains(".per_layer_projection.weight") { return 13 }
        if n.hasSuffix(".input_layernorm.weight") { return 14 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 15 }
        if n.hasSuffix(".pre_feedforward_layernorm.weight") { return 16 }
        if n.hasSuffix(".pre_feedforward_layernorm_2.weight") { return 17 }
        if n.hasSuffix(".post_feedforward_layernorm.weight") { return 18 }
        if n.hasSuffix(".post_feedforward_layernorm_1.weight") { return 19 }
        if n.hasSuffix(".post_feedforward_layernorm_2.weight") { return 20 }
        if n.hasSuffix(".post_per_layer_input_norm.weight") { return 21 }
        if n.hasSuffix(".layer_scalar")           { return 22 }
        return 100
    }
}
