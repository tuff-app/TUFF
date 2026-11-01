import TUFFRuntimeDescriptions

import Foundation
import TUFFFormat

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct ManifestArch: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let ffnIntermediate: Int
    public let moeIntermediateSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let hiddenActivation: String
    public let fullAttentionLayerMask: [Int]
    public let hiddenSizePerLayerInput: Int?
    public let vocabSizePerLayerInput: Int?
    public let numKVSharedLayers: Int?
    public let ffnDoubleWideFromLayer: Int?

    // Family extensions. Optional so legacy Gemma manifests decode unchanged;
    // absent values validate against the Gemma defaults in `ArchConfig`.
    public let family: String?
    public let variant: String?
    public let feedForwardKind: String?
    public let attnOutputGate: Bool?
    public let attentionScale: Double?
    public let embeddingScaledBySqrtHidden: Bool?
    public let routerScaled: Bool?
    public let ffnSandwichNorms: Bool?
    public let sharedExpertGated: Bool?
    public let ropeNeoxSubdim: Bool?
    public let linearNumKHeads: Int?
    public let linearNumVHeads: Int?
    public let linearKeyHeadDim: Int?
    public let linearValueHeadDim: Int?
    public let linearConvKernelSize: Int?
    public let hyperConnection: ManifestHyperConnection?
    public let ngramEmbedding: ManifestNgramEmbedding?
    public let attentionIndexer: ManifestAttentionIndexer?
}

public struct ManifestHyperConnection: Decodable, Equatable, Sendable {
    public let streamCount: Int
    public let lowRank: Int
}

public struct ManifestNgramEmbedding: Decodable, Equatable, Sendable {
    public let layer: Int
    public let ngramSize: Int
    public let heads: Int
    public let headsPerNgram: Int
    public let vocabSizeBase: Int
    public let shardCount: Int
    public let embedDim: Int
    public let convKernelSize: Int
    public let eosTokenID: Int?
}

public struct ManifestAttentionIndexer: Decodable, Equatable, Sendable {
    public let budget: Int
    public let compressRatio: Int
    public let headDim: Int
    public let numHeads: Int
    public let numKVHeads: Int
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
}

public extension ManifestQuantSlot {
    /// The group size the runtime actually decodes this slot at.
    ///
    /// The INT8 kernels take no group-size parameter — they are compiled at
    /// 64 — so an 8-bit slot is decoded at 64 whatever the manifest records.
    /// This matters because a checkpoint can mix the two: Qwen3.8 Flash Next
    /// groups its 4-bit tensors at 32 while its 8-bit router stays at 64, and
    /// a manifest that recorded one base group size for both would otherwise
    /// have the loader expecting scale regions of the wrong size.
    var effectiveGroupSize: Int {
        weightBits == 8 ? AffineQuantizationDefaults.groupSize : groupSize
    }
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot
}

/// Where one n-gram shard's three regions sit in the table file, and the span
/// of global rows it holds.
public struct ManifestNgramShard: Decodable, Equatable, Sendable {
    public let rowStart: UInt64
    public let rowCount: UInt64
    public let weightOffset: UInt64
    public let scaleOffset: UInt64
    public let biasOffset: UInt64
}

/// Layout of the n-gram PLE table file. Absent for every architecture without
/// one, which is all of them but `qwen4_exp`.
public struct ManifestNgramTable: Decodable, Equatable, Sendable {
    public let file: String
    public let layerIndex: Int
    public let rowWidth: Int
    public let groupSize: Int
    public let rowCount: UInt64
    public let shards: [ManifestNgramShard]
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
    public let ngramTable: ManifestNgramTable?
}

public enum ManifestReader {
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = GTurboFormatV1.knownFlags

    /// Fixed required entries. Packed-layer filenames come from layout.json and
    /// are cross-validated only after that document is decoded.
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(directoryURL: URL,
                            expecting: ArchConfig,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        return try decode(data: data, expecting: expecting)
    }

    package static func decode(data: Data,
                               expecting: ArchConfig) throws -> Manifest {
        let manifest: Manifest
        do {
            let wire = try GTurboManifestCodec.decodeUnchecked(data)
            guard wire.magic == GTurboFormatV1.magic else {
                throw ModelError.notAGTurboDirectory
            }
            guard wire.versionMajor == GTurboFormatV1.versionMajor,
                  wire.versionMinor >= 0 else {
                throw ModelError.unsupportedVersion(major: wire.versionMajor,
                                                    minor: wire.versionMinor)
            }
            for key in wire.flags.keys where !GTurboFormatV1.knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
            if wire.expertStride % GTurboFormatV1.alignmentBytes != 0 {
                throw ModelError.expertStrideNotPageAligned(
                    stride: wire.expertStride,
                    pageSize: Int(GTurboFormatV1.alignmentBytes))
            }
            try GTurboManifestCodec.validate(wire)
            manifest = Manifest(wire: wire)
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        try validate(manifest, against: expecting)
        return manifest
    }

    package static func validate(_ m: Manifest,
                         against expected: ArchConfig) throws {
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArch(m.arch, expected: expected)
        if let quant = m.quant {
            try validateQuant(
                quant,
                mxfp4Weights: m.flags["mxfp4Weights"] == true,
                int4GroupSize: expected.int4GroupSize, family: expected.family, variant: expected.variant)
        } else if isProductionArch(expected) {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the production architecture")
        }
        let required = expected.feedForwardKind == .dense
            ? ["model_weights.bin"] : requiredFiles
        for f in required {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
    }

    /// A manifest matching one of the shipped production baselines must carry
    /// quantization metadata; toy/synthetic manifests may omit it.
    private static func isProductionArch(_ expected: ArchConfig) -> Bool {
        for baseline in ArchConfig.registeredArchitectures.values {
            if expected.numLayers == baseline.numLayers,
               expected.hiddenSize == baseline.hiddenSize {
                return true
            }
        }
        return false
    }

    /// `int4GroupSize` is the architecture's, not a constant: Qwen3.8 Flash
    /// Next's 4-bit tensors are grouped at 32. Its 8-bit router and
    /// shared-expert gate stay at 64, as every architecture's do, so the two
    /// widths are checked against different expectations.
    private static func validateQuant(
        _ quant: ManifestQuant,
        mxfp4Weights: Bool,
        int4GroupSize: Int,
        family: ModelFamily,
        variant: ModelVariant
    ) throws {
        if variant == .bonsai2_2B || variant == .bonsai2_27B {
            for (name, slot) in [("embedding", quant.embedding), ("attention", quant.attention),
                                 ("sharedExpert", quant.sharedExpert)] {
                guard slot.weightBits == 2, slot.groupSize == 128,
                      slot.scheme.lowercased() == "affine", slot.scaleType.lowercased() == "fp16",
                      slot.biasType.lowercased() == "fp16" else {
                    throw ModelError.indexCorrupt(detail: "unsupported Prism quantization for \(name)")
                }
            }
            return
        }
        if mxfp4Weights {
            for (name, slot) in [
                ("embedding", quant.embedding),
                ("attention", quant.attention),
                ("router", quant.router),
                ("sharedExpert", quant.sharedExpert),
            ] {
                guard slot.weightBits == 16,
                      slot.scheme.lowercased() == "bf16",
                      slot.scaleType.lowercased() == "none",
                      slot.biasType.lowercased() == "none",
                      slot.groupSize == 1 else {
                    throw ModelError.indexCorrupt(
                        detail: "unsupported quantization for \(name)")
                }
            }
            let routed = quant.routedExpert
            guard routed.weightBits == 4,
                  routed.scheme.lowercased() == "mxfp4",
                  routed.scaleType.lowercased() == "ue8m0",
                  routed.biasType.lowercased() == "none",
                  routed.groupSize == 32 else {
                throw ModelError.indexCorrupt(
                    detail: "unsupported quantization for routedExpert")
            }
            return
        }
        if family == .flint {
            let router = quant.router
            if router.weightBits == 16 {
                guard router.scheme.lowercased() == "bf16",
                      router.scaleType.lowercased() == "none", router.biasType.lowercased() == "none",
                      router.groupSize == 1 else {
                    throw ModelError.indexCorrupt(detail: "unsupported Flint router quantization")
                }
            }
        }
        let slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("embedding", quant.embedding, family == .flint ? [8] : [4]),
            ("attention", quant.attention, family == .flint ? [4, 8] : [4]),
            ("router", quant.router, [8]),
            ("sharedExpert", quant.sharedExpert, [4, 8]),
        ]
        for (name, slot, allowedBits) in slots {
            if family == .flint && name == "router" && slot.weightBits == 16 { continue }
            // The group size is only checked for 4-bit slots. The INT8 kernels
            // take no group-size parameter — they are compiled at 64 — so for
            // an 8-bit slot the manifest's value describes nothing the runtime
            // reads, and enforcing it would reject a correct install over a
            // field with no effect. (A checkpoint whose 8-bit tensors were
            // genuinely grouped differently would be mis-decoded either way;
            // that is a kernel limitation, not something this check catches.)
            let groupMatches = slot.weightBits != 4
                || slot.groupSize == int4GroupSize
            guard allowedBits.contains(slot.weightBits),
                  slot.scheme.lowercased() == "affine",
                  slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16",
                  groupMatches else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
        let routed = quant.routedExpert
        guard (routed.weightBits == 4 || (family == .flint && routed.weightBits == 8)),
              routed.scheme.lowercased() == "affine",
              routed.scaleType.lowercased() == "bf16",
              routed.biasType.lowercased() == "bf16",
              routed.groupSize == int4GroupSize else {
            throw ModelError.indexCorrupt(
                detail: "unsupported quantization for routedExpert")
        }
    }

    private static func validateArch(_ a: ManifestArch,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        try check("hiddenSizePerLayerInput",
                  a.hiddenSizePerLayerInput ?? 0, e.hiddenSizePerLayerInput)
        // The production 26B MoE config carries `vocab_size_per_layer_input`
        // equal to the ordinary vocabulary even though that model has no PLE
        // stream, and GTurboManifestV1.validate deliberately accepts that
        // spelling. PLE is switched on by a non-zero *hidden* width, so when it
        // is off this value is informational and both spellings describe the
        // same architecture. Comparing it strictly rejected every completed 26B
        // install with `vocabSizePerLayerInput = 262144; expected 0`.
        let actualPerLayerVocab = a.vocabSizePerLayerInput ?? 0
        if e.hiddenSizePerLayerInput > 0 || actualPerLayerVocab != e.vocabSize {
            try check("vocabSizePerLayerInput",
                      actualPerLayerVocab, e.vocabSizePerLayerInput)
        }
        try check("ffnDoubleWideFromLayer",
                  a.ffnDoubleWideFromLayer ?? -1, e.ffnDoubleWideFromLayer)
        try check("numKVSharedLayers",
                  a.numKVSharedLayers ?? 0, e.numKVSharedLayers)
        let actualMask = a.fullAttentionLayerMask.map { UInt8($0) }
        try check("fullAttentionLayerMask",
                  actualMask.description,
                  e.fullAttentionLayerMask.description)

        // Family extensions: absent fields mean the Gemma defaults.
        let gemmaDefaults = ArchConfig.gemma4_26B_A4B
        try check("family",
                  a.family ?? ModelFamily.gemma4.rawValue,
                  e.family.rawValue)
        let actualFamily = ModelFamily(rawValue: a.family ?? ModelFamily.gemma4.rawValue)
            ?? .gemma4
        try check("variant",
                  a.variant ?? ModelVariant.legacyDefault(for: actualFamily).rawValue,
                  e.variant.rawValue)
        try check("feedForwardKind",
                  a.feedForwardKind ?? FeedForwardKind.mixtureOfExperts.rawValue,
                  e.feedForwardKind.rawValue)
        try check("attnOutputGate",
                  a.attnOutputGate ?? gemmaDefaults.attnOutputGate,
                  e.attnOutputGate)
        try check("attentionScale",
                  a.attentionScale ?? gemmaDefaults.attentionScale,
                  e.attentionScale)
        try check("embeddingScaledBySqrtHidden",
                  a.embeddingScaledBySqrtHidden ?? gemmaDefaults.embeddingScaledBySqrtHidden,
                  e.embeddingScaledBySqrtHidden)
        try check("routerScaled",
                  a.routerScaled ?? gemmaDefaults.routerScaled,
                  e.routerScaled)
        try check("ffnSandwichNorms",
                  a.ffnSandwichNorms ?? gemmaDefaults.ffnSandwichNorms,
                  e.ffnSandwichNorms)
        try check("sharedExpertGated",
                  a.sharedExpertGated ?? gemmaDefaults.sharedExpertGated,
                  e.sharedExpertGated)
        try check("ropeNeoxSubdim",
                  a.ropeNeoxSubdim ?? gemmaDefaults.ropeNeoxSubdim,
                  e.ropeNeoxSubdim)
        try check("linearNumKHeads",
                  a.linearNumKHeads ?? 0, e.linearAttention.numKHeads)
        try check("linearNumVHeads",
                  a.linearNumVHeads ?? 0, e.linearAttention.numVHeads)
        try check("linearKeyHeadDim",
                  a.linearKeyHeadDim ?? 0, e.linearAttention.keyHeadDim)
        try check("linearValueHeadDim",
                  a.linearValueHeadDim ?? 0, e.linearAttention.valueHeadDim)
        try check("linearConvKernelSize",
                  a.linearConvKernelSize ?? 0, e.linearAttention.convKernelSize)

        // The three `qwen4_exp` mechanisms. Absent means the ordinary
        // single-stream residual, no n-gram table and dense attention, which
        // is what `.none` describes, so every other family checks unchanged.
        let hyper = a.hyperConnection
        try check("hyperConnection.streamCount",
                  hyper?.streamCount ?? 0, e.hyperConnection.streamCount)
        try check("hyperConnection.lowRank",
                  hyper?.lowRank ?? 0, e.hyperConnection.lowRank)
        let ngram = a.ngramEmbedding
        try check("ngramEmbedding.layer",
                  ngram?.layer ?? -1, e.ngramEmbedding.layer)
        try check("ngramEmbedding.ngramSize",
                  ngram?.ngramSize ?? 0, e.ngramEmbedding.ngramSize)
        try check("ngramEmbedding.heads",
                  ngram?.heads ?? 0, e.ngramEmbedding.heads)
        try check("ngramEmbedding.headsPerNgram",
                  ngram?.headsPerNgram ?? 0, e.ngramEmbedding.headsPerNgram)
        try check("ngramEmbedding.vocabSizeBase",
                  ngram?.vocabSizeBase ?? 0, e.ngramEmbedding.vocabSizeBase)
        try check("ngramEmbedding.shardCount",
                  ngram?.shardCount ?? 0, e.ngramEmbedding.shardCount)
        try check("ngramEmbedding.embedDim",
                  ngram?.embedDim ?? 0, e.ngramEmbedding.embedDim)
        try check("ngramEmbedding.convKernelSize",
                  ngram?.convKernelSize ?? 0, e.ngramEmbedding.convKernelSize)
        // Unlike the fields around it, this one describes tokenizer semantics
        // rather than the table's layout, and it cannot vary for a given
        // variant. Absent means "the architecture's own", the way every other
        // family-extension field defaults — so a manifest written before the
        // field existed still validates.
        try check("ngramEmbedding.eosTokenID",
                  (ngram?.eosTokenID as Int??) .flatMap { $0 }
                      ?? Int(e.ngramEmbedding.eosTokenID),
                  Int(e.ngramEmbedding.eosTokenID))
        let indexer = a.attentionIndexer
        try check("attentionIndexer.budget",
                  indexer?.budget ?? 0, e.attentionIndexer.budget)
        try check("attentionIndexer.compressRatio",
                  indexer?.compressRatio ?? 0, e.attentionIndexer.compressRatio)
        try check("attentionIndexer.headDim",
                  indexer?.headDim ?? 0, e.attentionIndexer.headDim)
        try check("attentionIndexer.numHeads",
                  indexer?.numHeads ?? 0, e.attentionIndexer.numHeads)
        try check("attentionIndexer.numKVHeads",
                  indexer?.numKVHeads ?? 0, e.attentionIndexer.numKVHeads)
    }

    /// Decode just enough of `manifest.json` to identify the model family,
    /// without arch validation. Used by `Model.load` auto-detection.
    public static func peekFamily(directoryURL: URL,
                                  maxBytes: UInt64 = defaultMaxBytes) throws -> ModelFamily {
        try resolveArchitecture(
            directoryURL: directoryURL, maxBytes: maxBytes).family
    }

    /// Resolve the exact architecture variant. A legacy v1.0 manifest without
    /// `arch.variant` uses the one historical baseline for its family; v1.1
    /// manifests select directly from the variant registry.
    public static func resolveArchitecture(
        directoryURL: URL,
        maxBytes: UInt64 = defaultMaxBytes
    ) throws -> ArchConfig {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        let familyRaw = manifest.arch.family ?? ModelFamily.gemma4.rawValue
        guard let family = ModelFamily(rawValue: familyRaw) else {
            throw ModelError.indexCorrupt(detail: "unknown arch.family \"\(familyRaw)\"")
        }
        let variant: ModelVariant
        if let raw = manifest.arch.variant {
            guard let decoded = ModelVariant(rawValue: raw) else {
                throw ModelError.indexCorrupt(detail: "unknown arch.variant \"\(raw)\"")
            }
            variant = decoded
        } else {
            variant = ModelVariant.legacyDefault(for: family)
        }
        guard let baseline = ArchConfig.registeredArchitectures[variant] else {
            throw ModelError.indexCorrupt(
                detail: "no baseline for arch.variant \"\(variant.rawValue)\"")
        }
        guard baseline.family == family else {
            throw ModelError.indexCorrupt(
                detail: "arch.variant \"\(variant.rawValue)\" is not in family \"\(family.rawValue)\"")
        }
        return baseline
    }
}

private extension ManifestFileEntry {
    init(wire: GTurboManifestFileV1) {
        self.init(size: wire.size, sha256: wire.sha256)
    }
}

private extension ManifestArch {
    init(wire: GTurboManifestArchV1) {
        self.init(hiddenSize: wire.hiddenSize,
                  ffnIntermediate: wire.ffnIntermediate,
                  moeIntermediateSize: wire.moeIntermediateSize,
                  numHeads: wire.numHeads,
                  numKVHeads: wire.numKVHeads,
                  numFullKVHeads: wire.numFullKVHeads,
                  headDim: wire.headDim,
                  fullHeadDim: wire.fullHeadDim,
                  vocabSize: wire.vocabSize,
                  slidingWindow: wire.slidingWindow,
                  finalLogitSoftcap: wire.finalLogitSoftcap,
                  ropeTheta: wire.ropeTheta,
                  fullRopeTheta: wire.fullRopeTheta,
                  partialRotaryFactor: wire.partialRotaryFactor,
                  numLayers: wire.numLayers,
                  numExperts: wire.numExperts,
                  topKExperts: wire.topKExperts,
                  tieWordEmbeddings: wire.tieWordEmbeddings,
                  attentionKEqV: wire.attentionKEqV,
                  hiddenActivation: wire.hiddenActivation,
                  fullAttentionLayerMask: wire.fullAttentionLayerMask,
                  hiddenSizePerLayerInput: wire.hiddenSizePerLayerInput,
                  vocabSizePerLayerInput: wire.vocabSizePerLayerInput,
                  numKVSharedLayers: wire.numKVSharedLayers,
                  ffnDoubleWideFromLayer: wire.ffnDoubleWideFromLayer,
                  family: wire.family,
                  variant: wire.variant,
                  feedForwardKind: wire.feedForwardKind,
                  attnOutputGate: wire.attnOutputGate,
                  attentionScale: wire.attentionScale,
                  embeddingScaledBySqrtHidden: wire.embeddingScaledBySqrtHidden,
                  routerScaled: wire.routerScaled,
                  ffnSandwichNorms: wire.ffnSandwichNorms,
                  sharedExpertGated: wire.sharedExpertGated,
                  ropeNeoxSubdim: wire.ropeNeoxSubdim,
                  linearNumKHeads: wire.linearNumKHeads,
                  linearNumVHeads: wire.linearNumVHeads,
                  linearKeyHeadDim: wire.linearKeyHeadDim,
                  linearValueHeadDim: wire.linearValueHeadDim,
                  linearConvKernelSize: wire.linearConvKernelSize,
                  hyperConnection: wire.hyperConnection.map(ManifestHyperConnection.init(wire:)),
                  ngramEmbedding: wire.ngramEmbedding.map(ManifestNgramEmbedding.init(wire:)),
                  attentionIndexer:
                    wire.attentionIndexer.map(ManifestAttentionIndexer.init(wire:)))
    }
}

private extension ManifestHyperConnection {
    init(wire: GTurboManifestHyperConnectionV1) {
        self.init(streamCount: wire.streamCount, lowRank: wire.lowRank)
    }
}

private extension ManifestNgramEmbedding {
    init(wire: GTurboManifestNgramEmbeddingV1) {
        self.init(layer: wire.layer, ngramSize: wire.ngramSize,
                  heads: wire.heads, headsPerNgram: wire.headsPerNgram,
                  vocabSizeBase: wire.vocabSizeBase, shardCount: wire.shardCount,
                  embedDim: wire.embedDim, convKernelSize: wire.convKernelSize,
                  eosTokenID: wire.eosTokenID)
    }
}

private extension ManifestAttentionIndexer {
    init(wire: GTurboManifestAttentionIndexerV1) {
        self.init(budget: wire.budget, compressRatio: wire.compressRatio,
                  headDim: wire.headDim, numHeads: wire.numHeads,
                  numKVHeads: wire.numKVHeads)
    }
}

private extension ManifestQuantSlot {
    init(wire: GTurboManifestQuantSlotV1) {
        self.init(weightBits: wire.weightBits, scheme: wire.scheme,
                  scaleType: wire.scaleType, biasType: wire.biasType,
                  groupSize: wire.groupSize)
    }
}

private extension ManifestQuant {
    init(wire: GTurboManifestQuantV1) {
        self.init(embedding: ManifestQuantSlot(wire: wire.embedding),
                  attention: ManifestQuantSlot(wire: wire.attention),
                  router: ManifestQuantSlot(wire: wire.router),
                  sharedExpert: ManifestQuantSlot(wire: wire.sharedExpert),
                  routedExpert: ManifestQuantSlot(wire: wire.routedExpert))
    }
}

private extension Manifest {
    init(wire: GTurboManifestV1) {
        self.init(magic: wire.magic,
                  versionMajor: wire.versionMajor,
                  versionMinor: wire.versionMinor,
                  flags: wire.flags,
                  modelID: wire.modelID,
                  sourceSnapshotHash: wire.sourceSnapshotHash,
                  arch: ManifestArch(wire: wire.arch),
                  quant: wire.quant.map(ManifestQuant.init(wire:)),
                  files: wire.files.mapValues(ManifestFileEntry.init(wire:)),
                  expertsPerLayer: wire.expertsPerLayer,
                  numLayers: wire.numLayers,
                  expertStride: wire.expertStride,
                  ngramTable: wire.ngramTable.map(ManifestNgramTable.init(wire:)))
    }
}

private extension ManifestNgramTable {
    init(wire: GTurboManifestNgramTableV1) {
        self.init(file: wire.file,
                  layerIndex: wire.layerIndex,
                  rowWidth: wire.rowWidth,
                  groupSize: wire.groupSize,
                  rowCount: wire.rowCount,
                  shards: wire.shards.map(ManifestNgramShard.init(wire:)))
    }
}

private extension ManifestNgramShard {
    init(wire: GTurboManifestNgramShardV1) {
        self.init(rowStart: wire.rowStart,
                  rowCount: wire.rowCount,
                  weightOffset: wire.weightOffset,
                  scaleOffset: wire.scaleOffset,
                  biasOffset: wire.biasOffset)
    }
}
