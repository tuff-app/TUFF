import Foundation
import TUFFFormat

package struct SubTensorEntry: Sendable, Equatable {
    package let offset: UInt64    // relative to the expert blob's start
    package let size: UInt64
    package let dtype: String     // "U32" | "BF16" | "U8"
    package let shape: [UInt32]
    package let bits: Int?        // weight bit width override (4 or 8), if applicable
    package init(offset: UInt64, size: UInt64, dtype: String, shape: [UInt32], bits: Int?) {
        self.offset = offset; self.size = size; self.dtype = dtype; self.shape = shape; self.bits = bits
    }
}

package struct ExpertEntry: Sendable {
    /// Logical routed-expert id used by the model/router.
    package let expert: Int
    /// Physical rank inside the layer file. Old identity layouts use
    /// `physicalRank == expert`.
    package let physicalRank: Int
    /// Absolute byte offset of this expert blob's start inside its layer file.
    package let offset: UInt64
    /// Total bytes consumed by this expert blob (== `expertStride`).
    package let size: UInt64
    /// Sub-tensors keyed by role (gate / up / down / shared) and component
    /// (raw, `_scales`, `_biases`).
    package let subTensors: [String: SubTensorEntry]

    package init(expert: Int,
                physicalRank: Int? = nil,
                offset: UInt64,
                size: UInt64,
                subTensors: [String: SubTensorEntry]) {
        self.expert = expert
        self.physicalRank = physicalRank ?? expert
        self.offset = offset
        self.size = size
        self.subTensors = subTensors
    }
}

package struct LayerLayout: Sendable {
    package let layer: Int
    package let file: String          // basename, e.g. "layer_00.bin"
    package let experts: [ExpertEntry]
    package init(layer: Int, file: String, experts: [ExpertEntry]) {
        self.layer = layer; self.file = file; self.experts = experts
    }
}

package struct PackedExpertsLayout: Sendable {
    package let expertStride: UInt64
    package let numLayers: Int
    package let expertsPerLayer: Int
    package let layers: [LayerLayout]
    package init(expertStride: UInt64, numLayers: Int, expertsPerLayer: Int, layers: [LayerLayout]) {
        self.expertStride = expertStride; self.numLayers = numLayers
        self.expertsPerLayer = expertsPerLayer; self.layers = layers
    }

    /// Resolve `(layer, expert)` -> `ExpertEntry`. O(1).
    package func expert(layer: Int, expert: Int) -> ExpertEntry {
        return layers[layer].experts[expert]
    }

    package static func dense(numLayers: Int) -> PackedExpertsLayout {
        PackedExpertsLayout(expertStride: 0,
                            numLayers: numLayers,
                            expertsPerLayer: 0,
                            layers: [])
    }
}

package enum PackedExpertsLayoutReader {
    // Ornith 397B's 60 layers x 512 experts produce a 68,280,566-byte layout.
    // Keep a bounded limit with room for every supported model.
    package static let defaultMaxBytes: UInt64 = 128 * 1024 * 1024

    package static func load(directoryURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        try load(directoryURL: directoryURL, manifest: nil, maxBytes: maxBytes)
    }

    package static func load(directoryURL: URL,
                             manifest: Manifest,
                             maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        try load(directoryURL: directoryURL, manifest: Optional(manifest), maxBytes: maxBytes)
    }

    private static func load(directoryURL: URL,
                             manifest: Manifest?,
                             maxBytes: UInt64) throws -> PackedExpertsLayout {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data = try directory.readMetadata(
            "packed_experts/layout.json", maxBytes: maxBytes)
        return try decode(data: data, manifest: manifest)
    }

    package static func decode(data: Data,
                               manifest: Manifest?) throws -> PackedExpertsLayout {
        let wire: GTurboPackedExpertsLayoutV1
        do {
            wire = try GTurboPackedExpertsLayoutCodec.decode(data)
            if let manifest {
                try GTurboV1StructuralValidator.crossValidate(
                    manifestNumLayers: manifest.numLayers,
                    manifestExpertsPerLayer: manifest.expertsPerLayer,
                    manifestExpertStride: manifest.expertStride,
                    manifestFileSizes: manifest.files.mapValues(\.size),
                    layout: wire, densePrefixLayers: manifest.arch.family == "flint" ? 2 : 0)
            }
        } catch {
            throw ModelError.indexCorrupt(detail: "layout.json: \(error)")
        }
        let layers = wire.layers.sorted { $0.layer < $1.layer }.map { layer -> LayerLayout in
            let experts = layer.experts.enumerated().map { position, expert -> ExpertEntry in
                let logical = expert.expert ?? position
                return ExpertEntry(
                    expert: logical,
                    physicalRank: expert.physicalRank,
                    offset: expert.offset,
                    size: expert.size,
                    subTensors: expert.tensors.mapValues {
                        SubTensorEntry(offset: $0.offset, size: $0.size,
                                       dtype: $0.dtype, shape: $0.shape,
                                       bits: $0.bits)
                    })
            }.sorted { $0.expert < $1.expert }
            return LayerLayout(layer: layer.layer, file: layer.file, experts: experts)
        }
        return PackedExpertsLayout(expertStride: wire.expertStride,
                                   numLayers: wire.numLayers,
                                   expertsPerLayer: wire.expertsPerLayer,
                                   layers: layers)
    }

}
