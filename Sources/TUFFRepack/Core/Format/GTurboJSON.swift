import Foundation
import TUFFFormat

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum GTurboJSON {

    static let magic = GTurboFormatV1.magic
    static let versionMajor = GTurboFormatV1.versionMajor
    static let versionMinor = GTurboFormatV1.versionMinor

    static func versionMinor(for plan: RepackPlan) -> Int {
        plan.arch.feedForwardKind == .dense || plan.arch.family == .gptOss
            ? GTurboFormatV1.featureVersionMinor : versionMinor
    }

    struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    struct QuantBitWidths {
        var embedding: Int
        var attention: Int
        var router: Int
        var sharedExpert: Int
        var routedExpert: Int
    }

    static func encodeManifest(plan: RepackPlan,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        let arch = plan.arch
        let isGemma = arch.family == .gemma4
        let isDense = arch.feedForwardKind == .dense
        let isGPTOSS = arch.family == .gptOss
        let bitWidthsByQuantSlot = [
            "embedding": bitWidths.embedding,
            "attention": bitWidths.attention,
            "router": bitWidths.router,
            "sharedExpert": bitWidths.sharedExpert,
            "routedExpert": bitWidths.routedExpert,
        ]
        let wireArch = GTurboManifestArchV1(
            hiddenSize: arch.hiddenSize,
            ffnIntermediate: arch.intermediateSize,
            moeIntermediateSize: arch.moeIntermediateSize,
            numHeads: arch.numHeads,
            numKVHeads: arch.numKVHeads,
            numFullKVHeads: arch.numFullKVHeads,
            headDim: arch.headDim,
            fullHeadDim: arch.fullHeadDim,
            vocabSize: arch.vocabSize,
            slidingWindow: arch.slidingWindow,
            finalLogitSoftcap: arch.finalLogitSoftcap,
            ropeTheta: arch.ropeTheta,
            fullRopeTheta: arch.fullRopeTheta,
            partialRotaryFactor: arch.partialRotaryFactor,
            numLayers: arch.numLayers,
            numExperts: arch.numExperts,
            topKExperts: arch.topKExperts,
            tieWordEmbeddings: arch.tieWordEmbeddings,
            attentionKEqV: arch.attentionKEqV,
            hiddenActivation: arch.hiddenActivation,
            fullAttentionLayerMask: arch.fullAttentionLayerMask.map(Int.init),
            hiddenSizePerLayerInput: arch.hiddenSizePerLayerInput == 0
                ? nil : arch.hiddenSizePerLayerInput,
            vocabSizePerLayerInput: arch.vocabSizePerLayerInput == 0
                ? nil : arch.vocabSizePerLayerInput,
            numKVSharedLayers: arch.numKVSharedLayers == 0
                ? nil : arch.numKVSharedLayers,
            ffnDoubleWideFromLayer: arch.ffnDoubleWideFromLayer < 0
                ? nil : arch.ffnDoubleWideFromLayer,
            // Gemma 4 omits every family extension field, so its manifests stay
            // byte-identical to the pre-family format.
            family: isGemma ? nil : arch.family.rawValue,
            // Named explicitly rather than left to the reader's
            // family-to-variant fallback: `qwen4_exp` is a family that can
            // hold more than one checkpoint shape.
            variant: (isDense || !isGemma)
                ? arch.variant.rawValue : nil,
            feedForwardKind: (isDense || isGPTOSS)
                ? arch.feedForwardKind.rawValue : nil,
            attnOutputGate: isGemma ? nil : arch.attnOutputGate,
            attentionScale: isGemma ? nil : arch.attentionScale,
            embeddingScaledBySqrtHidden: isGemma ? nil : arch.embeddingScaledBySqrtHidden,
            routerScaled: isGemma ? nil : arch.routerScaled,
            ffnSandwichNorms: isGemma ? nil : arch.ffnSandwichNorms,
            sharedExpertGated: isGemma ? nil : arch.sharedExpertGated,
            ropeNeoxSubdim: isGemma ? nil : arch.ropeNeoxSubdim,
            linearNumKHeads: isGemma ? nil : arch.linearNumKHeads,
            linearNumVHeads: isGemma ? nil : arch.linearNumVHeads,
            linearKeyHeadDim: isGemma ? nil : arch.linearKeyHeadDim,
            linearValueHeadDim: isGemma ? nil : arch.linearValueHeadDim,
            linearConvKernelSize: isGemma ? nil : arch.linearConvKernelSize,
            // Absent for every family but `qwen4_exp`, whose loader is the
            // only one that sets them.
            hyperConnection: arch.hyperConnectionStreamCount == 0 ? nil
                : GTurboManifestHyperConnectionV1(
                    streamCount: arch.hyperConnectionStreamCount,
                    lowRank: arch.hyperConnectionLowRank),
            ngramEmbedding: arch.ngramShardCount == 0 ? nil
                : GTurboManifestNgramEmbeddingV1(
                    layer: arch.ngramLayer,
                    ngramSize: arch.ngramSize,
                    heads: arch.ngramHeads,
                    headsPerNgram: arch.ngramHeadsPerNgram,
                    vocabSizeBase: arch.ngramVocabSizeBase,
                    shardCount: arch.ngramShardCount,
                    embedDim: arch.ngramEmbedDim,
                    convKernelSize: arch.ngramConvKernelSize,
                    eosTokenID: arch.ngramEosTokenID),
            attentionIndexer: arch.indexerBudget == 0 ? nil
                : GTurboManifestAttentionIndexerV1(
                    budget: arch.indexerBudget,
                    compressRatio: arch.indexerCompressRatio,
                    headDim: arch.indexerHeadDim,
                    numHeads: arch.indexerNumHeads,
                    numKVHeads: arch.indexerNumKVHeads))
        func slot(_ name: String) throws -> GTurboManifestQuantSlotV1 {
            guard let weightBits = bitWidthsByQuantSlot[name] else {
                throw RepackError.configurationInvalid(
                    detail: "missing manifest quant slot bit width for \(name)")
            }
            if isGPTOSS, name == "routedExpert" {
                return GTurboManifestQuantSlotV1(
                    weightBits: 4,
                    scheme: "mxfp4",
                    scaleType: "UE8M0",
                    biasType: "none",
                    groupSize: 32)
            }
            if isGPTOSS {
                return GTurboManifestQuantSlotV1(
                    weightBits: 16,
                    scheme: "bf16",
                    scaleType: "none",
                    biasType: "none",
                    groupSize: 1)
            }
            if arch.family == .flint, name == "router", weightBits == 16 {
                return GTurboManifestQuantSlotV1(weightBits: 16, scheme: "bf16",
                    scaleType: "none", biasType: "none", groupSize: 1)
            }
            if [RepackModelVariant.bonsai2_2B, .bonsai2_27B].contains(arch.variant), name == "routedExpert" {
                return GTurboManifestQuantSlotV1(weightBits: weightBits, scheme: "affine",
                    scaleType: "BF16", biasType: "BF16", groupSize: 64)
            }
            // Prism's packed language matrices retain the source's FP16
            // affine companions. Router and routed-expert slots remain the
            // ordinary unused dense placeholders for schema compatibility.
            if [RepackModelVariant.bonsai2_2B, .bonsai2_27B].contains(arch.variant),
               ["embedding", "attention", "sharedExpert"].contains(name) {
                guard weightBits == 2 else {
                    throw RepackError.configurationInvalid(detail: "Prism \(name) must use 2-bit weights")
                }
                return GTurboManifestQuantSlotV1(weightBits: 2, scheme: "affine",
                                                scaleType: "FP16", biasType: "FP16", groupSize: 128)
            }
            // The group size follows the bit width, not the checkpoint: a
            // 4-bit checkpoint can carry an 8-bit router grouped at 64 beside
            // 4-bit tensors grouped at 32, and recording the base for both
            // would describe scale regions of the wrong size.
            return GTurboManifestQuantSlotV1(
                weightBits: weightBits,
                scheme: plan.baseMode,
                scaleType: "BF16",
                biasType: "BF16",
                groupSize: weightBits == 8 ? 64 : plan.baseGroupSize)
        }
        let quant = GTurboManifestQuantV1(
            embedding: try slot("embedding"),
            attention: try slot("attention"),
            router: try slot("router"),
            sharedExpert: try slot("sharedExpert"),
            routedExpert: try slot("routedExpert"))
        var wireFiles: [String: GTurboManifestFileV1] = [:]
        wireFiles.reserveCapacity(files.count)
        for file in files {
            guard wireFiles.updateValue(
                GTurboManifestFileV1(size: file.info.size, sha256: file.info.sha256),
                forKey: file.relativePath) == nil else {
                throw RepackError.configurationInvalid(
                    detail: "duplicate manifest file entry \(file.relativePath)")
            }
        }
        var flags = [
            "streamingPresent": !isDense,
            "turboQuantKV": false,
            "aneSharedExpert": false,
        ]
        if isDense { flags["denseFFN"] = true }
        if isGPTOSS { flags["mxfp4Weights"] = true }
        return try GTurboManifestCodec.encode(GTurboManifestV1(
            versionMinor: versionMinor(for: plan),
            flags: flags,
            modelID: modelID,
            sourceSnapshotHash: sourceSnapshotHash,
            arch: wireArch,
            quant: quant,
            files: wireFiles,
            expertsPerLayer: expertsPerLayer,
            numLayers: numLayers,
            expertStride: expertStride,
            bitWidthOverridesHonored: plan.bitsOverrideCount,
            // Row addressing for the n-gram table. Absent for every
            // architecture without one, so no other manifest changes.
            ngramTable: plan.ngramTable.map { table in
                GTurboManifestNgramTableV1(
                    file: (table.path as NSString).lastPathComponent,
                    layerIndex: table.layerIndex,
                    rowWidth: table.rowWidth,
                    groupSize: table.groupSize,
                    rowCount: table.rowCount,
                    shards: table.shards.map {
                        GTurboManifestNgramShardV1(
                            rowStart: $0.rowStart,
                            rowCount: $0.rowCount,
                            weightOffset: $0.weightOffset,
                            scaleOffset: $0.scaleOffset,
                            biasOffset: $0.biasOffset)
                    })
            }))
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        let arch = plan.arch
        var layers: [GTurboLayerV1] = []
        layers.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [GTurboExpertV1] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let physicalRank = lp.physicalRank(for: e)
                let base = UInt64(physicalRank) * lp.expertStride
                var tensors: [String: GTurboSubTensorV1] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    case "bias":    key = slice.role + "_bias"
                    default:        key = slice.role + "_" + slice.component
                    }
                    guard slice.dtype == GTurboFormatV1.DType.u32.rawValue
                            || slice.dtype == GTurboFormatV1.DType.bf16.rawValue
                            || slice.dtype == SourceTensor.Dtype.u8.rawValue else {
                        throw RepackError.configurationInvalid(
                            detail: "unsupported packed expert dtype \(slice.dtype) for \(key)")
                    }
                    let shape = try slice.logicalShape.enumerated().map { index, value in
                        guard value <= UInt64(UInt32.max) else {
                            throw RepackError.configurationInvalid(
                                detail: "packed expert shape[\(index)] exceeds UInt32")
                        }
                        return UInt32(value)
                    }
                    let dtype: String
                    switch slice.dtype {
                    case GTurboFormatV1.DType.u32.rawValue: dtype = "U32"
                    case GTurboFormatV1.DType.bf16.rawValue: dtype = "BF16"
                    default: dtype = "U8"
                    }
                    let previous = tensors.updateValue(GTurboSubTensorV1(
                        offset: slice.offsetInExpertBlob,
                        size: slice.sizeInExpertBlob,
                        dtype: dtype,
                        shape: shape,
                        bits: slice.bitsForWeights), forKey: key)
                    guard previous == nil else {
                        throw RepackError.configurationInvalid(
                            detail: "duplicate packed expert tensor key \(key)")
                    }
                }
                experts.append(GTurboExpertV1(
                    expert: e,
                    physicalRank: nil,
                    offset: base,
                    size: lp.expertStride,
                    tensors: tensors))
            }
            layers.append(GTurboLayerV1(layer: lp.layerIndex,
                                        file: layerFile,
                                        experts: experts))
        }
        return try GTurboPackedExpertsLayoutCodec.encode(
            GTurboPackedExpertsLayoutV1(
                expertStride: expertStride,
                numLayers: arch.numLayers,
                expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
                layers: layers))
    }
}
