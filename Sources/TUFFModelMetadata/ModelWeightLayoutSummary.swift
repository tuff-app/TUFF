import Foundation
import Darwin
import TUFFInferenceTypes
import TUFFRuntimeDescriptions

/// Calculated from the installed pack's bounded index and manifest. This is
/// allocation input, not a measurement of occupied pages or device speed.
public struct ModelWeightLayoutSummary: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let modelID: String
    public let architectureID: String
    public let manifestSHA256: String
    public let weightsSHA256: String
    public let layout: InferenceWeightLayout
    public let packedResidentPayloadBytes: UInt64
    public let minimumDenseStagingBytes: UInt64

    package init(schemaVersion: Int, modelID: String, architectureID: String,
                  manifestSHA256: String, weightsSHA256: String, layout: InferenceWeightLayout,
                  packedResidentPayloadBytes: UInt64, minimumDenseStagingBytes: UInt64) {
        self.schemaVersion = schemaVersion
        self.modelID = modelID
        self.architectureID = architectureID
        self.manifestSHA256 = manifestSHA256
        self.weightsSHA256 = weightsSHA256
        self.layout = layout
        self.packedResidentPayloadBytes = packedResidentPayloadBytes
        self.minimumDenseStagingBytes = minimumDenseStagingBytes
    }

    public static func read(directory: URL, expecting: ArchConfig? = nil) throws -> Self {
        let pack = try GTurboModelDirectory(rootURL: directory)
        let data = try pack.readMetadata("manifest.json", maxBytes: ManifestReader.defaultMaxBytes)
        let architecture = try expecting ?? ManifestReader.resolveArchitecture(directoryURL: directory)
        let manifest = try ManifestReader.decode(data: data, expecting: architecture)
        let fd = try pack.openFile("model_weights.bin")
        defer { close(fd) }
        let index = try ResidentIndexReader.load(fileDescriptor: fd, displayPath: "model_weights.bin")
        guard let weights = manifest.files["model_weights.bin"],
              try pack.fileSize(fileDescriptor: fd, relativePath: "model_weights.bin") == weights.size else {
            throw ModelError.indexCorrupt(detail: "weights file size disagrees with manifest")
        }
        func add(_ left: UInt64, _ right: UInt64) throws -> UInt64 {
            let result = left.addingReportingOverflow(right)
            guard !result.overflow else { throw InferencePlanningError.arithmeticOverflow }
            return result.partialValue
        }
        var fixed: UInt64 = 0
        var layerBytes = Array(repeating: UInt64(0), count: architecture.numLayers)
        var staging = Array(repeating: UInt64(0), count: architecture.numLayers)
        let supportsProjectionStaging = architecture.family != .gptOss && architecture.family != .inkling2
        let payloadEnd = try add(index.header.indexSize, index.header.residentSize)
        guard payloadEnd == weights.size else { throw InferencePlanningError.invalidLayout }
        for entry in index.entries.values.sorted(by: { $0.fileOffset < $1.fileOffset }) {
            var starts: [UInt64] = [], ends: [UInt64] = []
            for (offset, bytes) in [(entry.fileOffset, entry.sizeBytes),
                                   (entry.scaleOffset, entry.scaleSize),
                                   (entry.biasOffset, entry.biasSize)] where bytes > 0 {
                let end = try add(offset, bytes)
                guard offset >= index.header.indexSize, end <= payloadEnd else {
                    throw InferencePlanningError.invalidLayout
                }
                starts.append(offset); ends.append(end)
            }
            guard let start = starts.min(), let end = ends.max() else {
                throw InferencePlanningError.invalidLayout
            }
            let bytes = end - start
            let prefix = "language_model.model.layers."
            let layer = entry.name.hasPrefix(prefix)
                ? Int(entry.name.dropFirst(prefix.count).split(separator: ".").first ?? "") : nil
            if supportsProjectionStaging, let layer {
                guard layer >= 0 && layer < layerBytes.count else { throw InferencePlanningError.invalidLayout }
                // Account for each retained mmap's whole pages. One shared
                // staging allocation packs complete tensors at 256-byte offsets.
                let page = UInt64(getpagesize())
                let mapped = try add(try add(bytes, start % page), page - 1) / page * page
                layerBytes[layer] = try add(layerBytes[layer], mapped)
                let aligned = try add(staging[layer], 255) / 256 * 256
                staging[layer] = try add(aligned, bytes)
            } else {
                let page = UInt64(getpagesize())
                fixed = try add(fixed, try add(try add(bytes, start % page), page - 1) / page * page)
            }
        }
        let experts: [InferenceWeightLayout.ExpertLayer]
        if architecture.feedForwardKind == .mixtureOfExperts {
            guard manifest.expertStride > 0 else { throw InferencePlanningError.invalidLayout }
            experts = (0..<architecture.numLayers).map { _ in
                .init(bytesPerExpert: manifest.expertStride, expertCount: architecture.numExperts,
                      minimumSlots: architecture.topKExperts)
            }
        } else { experts = [] }
        var rows: [InferenceWeightLayout.RowAccessTable] = []
        if let table = manifest.ngramTable {
            guard table.rowWidth > 0, table.groupSize > 0,
                  table.rowWidth.isMultiple(of: table.groupSize), let count = Int(exactly: table.rowCount) else {
                throw InferencePlanningError.invalidLayout
            }
            let rowBytes = UInt64(table.rowWidth / 2 + table.rowWidth / table.groupSize * 4)
            rows.append(.init(bytesPerRow: rowBytes, rowCount: count,
                              maximumRowsPerToken: architecture.ngramEmbedding.heads))
        }
        let layout = InferenceWeightLayout(fixedResidentBytes: fixed,
            denseGroupBytes: supportsProjectionStaging ? layerBytes : [], expertLayers: experts, rowAccessTables: rows)
        try layout.validate()
        return Self(schemaVersion: 1, modelID: manifest.modelID,
            architectureID: architecture.variant.rawValue,
            manifestSHA256: Sha256Verifier.hashData(data), weightsSHA256: weights.sha256,
            layout: layout, packedResidentPayloadBytes: index.header.residentSize,
            minimumDenseStagingBytes: staging.max() ?? 0)
    }
}
