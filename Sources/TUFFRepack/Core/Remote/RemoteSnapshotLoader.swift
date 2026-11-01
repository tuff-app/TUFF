import Foundation
import TUFFModelCatalog

struct RemoteSnapshot {
    let metadata: IndexLoader.SourceMetadata
    let arch: ArchInfo
    let shardHeaders: [Safetensors.Header]
    let remoteFiles: [String: RemoteFileInfo]
    let resolvedCommit: String
    let metadataDirectory: String
}

enum RemoteSnapshotLoader {
    static func load(remote: HuggingFaceRemoteSource,
                     requireKnownSource: Bool,
                     metadataDirectory: String,
                     audit: RepackAudit? = nil) async throws -> RemoteSnapshot {
        try Posix.mkdirP(metadataDirectory)
        if remote.repoID == FlintInt4RemoteInstaller.repoID {
            return try await loadFlint(remote: remote, metadataDirectory: metadataDirectory, audit: audit)
        }

        let isOptiQ = remote.repoID == OptiQRemoteInstaller.repoID
        if isOptiQ && remote.requestedRevision != OptiQRemoteInstaller.revision {
            throw RepackError.remoteProtocolInvalid(detail: "OptiQ revision differs from the pinned checkpoint")
        }

        let indexInfo = try await remote.resolveFileInfo(filename: "model.safetensors.index.json",
                                                         audit: audit)
        let pinned = remote.pinned(commit: indexInfo.resolvedCommit)
        if isOptiQ && indexInfo.resolvedCommit != OptiQRemoteInstaller.revision {
            throw RepackError.remoteProtocolInvalid(detail: "OptiQ resolved commit differs from the pinned checkpoint")
        }
        let configInfo = try await pinned.resolveFileInfo(filename: "config.json",
                                                          audit: audit)
        guard configInfo.resolvedCommit == indexInfo.resolvedCommit else {
            throw RepackError.remoteProtocolInvalid(detail: "config commit differs from index commit")
        }

        try await pinned.fetchSmallFile(filename: "model.safetensors.index.json",
                                        info: indexInfo,
                                        capBytes: remote.repoID == TUFFModelCatalog.ornith15_397B.source.repoID
                                            && indexInfo.resolvedCommit == TUFFModelCatalog.ornith15_397B.source.revision
                                            ? 64 * 1024 * 1024 : 4 * 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("model.safetensors.index.json"),
                                        audit: audit)
        try await pinned.fetchSmallFile(filename: "config.json",
                                        info: configInfo,
                                        capBytes: 1024 * 1024,
                                        outputPath: (metadataDirectory as NSString)
                                            .appendingPathComponent("config.json"),
                                        audit: audit)

        let metadata = try IndexLoader.load(snapshotDir: metadataDirectory)
        if isOptiQ {
            guard metadata.indexSha256Hex == "54bd24dfdb836f9154da82f2273cc52274a230b74c9996379357285d72330fd0" else {
                throw RepackError.sourceFingerprintRejected(path: metadata.indexPath, sha256: metadata.indexSha256Hex)
            }
            let configSHA = try Sha256Stream.hashFile(path: metadata.configPath)
            guard configSHA == "5d90b91771da21a6f98d3c4f6f2fc6bcd4cfe58de3d994a8291b776ff76a9afa" else {
                throw RepackError.sourceFingerprintRejected(path: metadata.configPath, sha256: configSHA)
            }
            guard Set(metadata.shardFilenames) == Set(optiqShards.keys) else {
                throw RepackError.remoteProtocolInvalid(detail: "OptiQ index shard list differs from the pinned checkpoint")
            }
        }
        if requireKnownSource && !isOptiQ && SourceFingerprint.modelID(forIndexSha256: metadata.indexSha256Hex) == nil {
            throw RepackError.sourceFingerprintRejected(path: metadata.indexPath,
                                                        sha256: metadata.indexSha256Hex)
        }
        let matched = SupportedModelSource.all.first { $0.sourceIndexSHA256 == metadata.indexSha256Hex && $0.repoID == remote.repoID }
        let variant = matched.flatMap { source in
            TUFFModelCatalog.all.first { $0.source.manifestModelID == source.modelID }
                .flatMap { RepackModelVariant(rawValue: $0.architecture.id.rawValue) }
        }
        let arch = try isOptiQ ? ArchInfo.loadValidatedFlint(configPath: metadata.configPath)
            : ArchInfo.load(configPath: metadata.configPath, variant: variant)

        var files: [String: RemoteFileInfo] = [
            indexInfo.filename: indexInfo,
            configInfo.filename: configInfo,
        ]
        var headers: [Safetensors.Header] = []
        headers.reserveCapacity(metadata.shardFilenames.count)
        for shard in metadata.shardFilenames {
            let info = try await pinned.resolveFileInfo(filename: shard, audit: audit)
            guard info.resolvedCommit == indexInfo.resolvedCommit else {
                throw RepackError.remoteProtocolInvalid(detail: "shard \(shard) commit differs from index commit")
            }
            guard info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(detail: "shard \(shard) does not advertise byte ranges")
            }
            if isOptiQ {
                let tag = info.etag?.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                guard let pin = optiqShards[shard], info.size == pin.size, tag == pin.sha256 else {
                    throw RepackError.remoteProtocolInvalid(detail: "OptiQ shard \(shard) differs from its pinned content hash or size")
                }
            }
            files[shard] = info
            let prefix = try await pinned.downloadRangeToTempFile(filename: shard,
                                                                 info: info,
                                                                 offset: 0,
                                                                 length: 8,
                                                                 audit: audit)
            defer { try? FileManager.default.removeItem(atPath: prefix.path) }
            let prefixData = try Data(contentsOf: URL(fileURLWithPath: prefix.path))
            guard prefixData.count == 8 else {
                throw RepackError.safetensorsHeaderInvalid(path: shard, detail: "short header prefix")
            }
            let headerSize = prefixData.withUnsafeBytes { raw -> UInt64 in
                var value: UInt64 = 0
                for i in 0..<8 {
                    value |= UInt64(raw[i]) << UInt64(i * 8)
                }
                return value
            }
            // The remote reports `info.size`; a value under 8 would underflow
            // the bound below into ~2^64 and admit any header at all.
            guard info.size >= 8 else {
                throw RepackError.safetensorsHeaderInvalid(
                    path: shard, detail: "remote reports \(info.size) bytes")
            }
            if headerSize > Safetensors.maxHeaderBytes || headerSize > info.size - 8 {
                throw RepackError.safetensorsHeaderTooLarge(path: shard, size: headerSize)
            }
            let headerFile = try await pinned.downloadRangeToTempFile(filename: shard,
                                                                     info: info,
                                                                     offset: 8,
                                                                     length: Int(headerSize),
                                                                     audit: audit)
            defer { try? FileManager.default.removeItem(atPath: headerFile.path) }
            let headerData = try Data(contentsOf: URL(fileURLWithPath: headerFile.path))
            headers.append(try Safetensors.parseHeaderBytes(path: shard,
                                                            fileSize: info.size,
                                                            headerBytes: headerData))
        }

        return RemoteSnapshot(metadata: metadata,
                              arch: arch,
                              shardHeaders: headers,
                              remoteFiles: files,
                              resolvedCommit: indexInfo.resolvedCommit,
                              metadataDirectory: metadataDirectory)
    }
    private static let optiqShards: [String: (size: UInt64, sha256: String)] = [
        "model-00001-of-00006.safetensors": (5_167_622_992, "14730927f084314e179dde68e97964a6bffb1d5fa63067661128f0a46c3c6ad3"),
        "model-00002-of-00006.safetensors": (5_366_934_229, "92bfbc7ce18c8c4abd16859b6317ed63453e0f465a9dcb0aaef8f52b027a8ad3"),
        "model-00003-of-00006.safetensors": (5_291_621_482, "af6f26bc269fa123e49b2a7e499cdb4a6dd06624e59b078ca64f4cdd06901ba6"),
        "model-00004-of-00006.safetensors": (5_283_232_687, "3ad99cdd9014af1b79804fb8f55c5fa915628330e8f8f3aa88b500dc57cd2b01"),
        "model-00005-of-00006.safetensors": (5_258_066_881, "d7a282c039f10645a9aa5e8ae4c482cb00827a6e1bdb1f43474754d09c328646"),
        "model-00006-of-00006.safetensors": (1_115_696_513, "03ca3d43cc47e518fd72051d94505c52434c2fd1704c45a6528c40aa21665e85"),
    ]

    /// Flint does not publish model.safetensors.index.json. Its pinned
    /// publisher manifest provides full shard hashes, and the range headers
    /// construct the tensor map without reading the shard payloads.
    private static func loadFlint(remote: HuggingFaceRemoteSource,
                                  metadataDirectory: String,
                                  audit: RepackAudit?) async throws -> RemoteSnapshot {
        let revision = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
        let manifestSHA = "319822f10e32a5916b57e23c7f05f9e92b24d02b3a136d02a3846b5edab78109"
        let info = try await remote.resolveFileInfo(filename: "flint-int4.json", audit: audit)
        guard info.resolvedCommit == revision else {
            throw RepackError.remoteProtocolInvalid(detail: "Flint revision differs from the pinned checkpoint")
        }
        let pinned = remote.pinned(commit: revision)
        let manifestPath = (metadataDirectory as NSString).appendingPathComponent("flint-int4.json")
        try await pinned.fetchSmallFile(filename: info.filename, info: info, capBytes: 1024 * 1024,
                                        outputPath: manifestPath, audit: audit)
        guard try Sha256Stream.hashFile(path: manifestPath) == manifestSHA,
              let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: manifestPath))) as? [String: Any],
              let hashes = manifest["checkpoint_files"] as? [String: String] else {
            throw RepackError.sourceFingerprintRejected(path: manifestPath, sha256: try Sha256Stream.hashFile(path: manifestPath))
        }
        let configInfo = try await pinned.resolveFileInfo(filename: "config.json", audit: audit)
        let configPath = (metadataDirectory as NSString).appendingPathComponent("config.json")
        try await pinned.fetchSmallFile(filename: "config.json", info: configInfo, capBytes: 1024 * 1024,
                                        outputPath: configPath, audit: audit)
        guard configInfo.resolvedCommit == revision,
              try Sha256Stream.hashFile(path: configPath) == hashes["config.json"] else {
            throw RepackError.remoteProtocolInvalid(detail: "Flint config differs from its pinned digest")
        }
        let arch = try ArchInfo.loadValidatedFlint(configPath: configPath)
        let names = (0..<5).map { String(format: "model-%05d.safetensors", $0) }
        var files: [String: RemoteFileInfo] = [info.filename: info, configInfo.filename: configInfo]
        var headers: [Safetensors.Header] = [], weightMap: [String: String] = [:]
        for name in names {
            let file = try await pinned.resolveFileInfo(filename: name, audit: audit)
            let tag = file.etag?.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
            guard file.resolvedCommit == revision, file.acceptsRanges, file.size >= 8,
                  let expected = hashes[name], tag == expected else {
                throw RepackError.remoteProtocolInvalid(detail: "Flint shard \(name) does not match its pinned content hash")
            }
            files[name] = file
            let prefix = try await pinned.downloadRangeToTempFile(filename: name, info: file, offset: 0, length: 8, audit: audit)
            defer { try? FileManager.default.removeItem(atPath: prefix.path) }
            let raw = try Data(contentsOf: URL(fileURLWithPath: prefix.path))
            guard raw.count == 8 else { throw RepackError.safetensorsHeaderInvalid(path: name, detail: "short header prefix") }
            let length = raw.enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << UInt64($1.offset * 8)) }
            guard length <= Safetensors.maxHeaderBytes, length <= file.size - 8 else {
                throw RepackError.safetensorsHeaderTooLarge(path: name, size: length)
            }
            let headerFile = try await pinned.downloadRangeToTempFile(filename: name, info: file, offset: 8,
                                                                       length: Int(length), audit: audit)
            defer { try? FileManager.default.removeItem(atPath: headerFile.path) }
            let header = try Safetensors.parseHeaderBytes(path: name, fileSize: file.size,
                headerBytes: Data(contentsOf: URL(fileURLWithPath: headerFile.path)))
            for tensor in header.tensors {
                guard weightMap.updateValue(name, forKey: tensor.name) == nil else {
                    throw RepackError.configurationInvalid(detail: "duplicate Flint tensor \(tensor.name)")
                }
            }
            headers.append(header)
        }
        let metadata = IndexLoader.SourceMetadata(indexPath: manifestPath, configPath: configPath,
            indexSha256Hex: manifestSHA, weightMap: weightMap, baseBits: 4, baseGroupSize: 64,
            baseMode: "affine", bitsOverrides: ["model.embed_tokens": QuantSpec(bits: 8, groupSize: 64)],
            shardFilenames: names)
        return RemoteSnapshot(metadata: metadata, arch: arch, shardHeaders: headers,
                              remoteFiles: files, resolvedCommit: revision, metadataDirectory: metadataDirectory)
    }

}
