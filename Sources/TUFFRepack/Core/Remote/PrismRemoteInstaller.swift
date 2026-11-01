import Foundation
import TUFFModelCatalog

/// Downloads only the pinned text checkpoint files, then uses the same
/// hash-validating local repacker as the reviewed reference packs. The source
/// snapshot is temporary, so peak disk includes both it and the final pack.
public enum PrismRemoteInstaller {
    public enum Size: Sendable {
        case twoB, twentySevenB

        /// The installer for a catalog model, or `nil` for models that use the
        /// generic streaming repacker.
        public init?(modelID: TUFFModelID) {
            switch modelID {
            case .bonsai2_2B: self = .twoB
            case .bonsai2_27B: self = .twentySevenB
            default: return nil
            }
        }

        var repoID: String {
            switch self {
            case .twoB: "prism-ml/Ternary-Bonsai-2-2B-mlx-2bit"
            case .twentySevenB: "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
            }
        }
        var revision: String {
            switch self {
            case .twoB: "1c6cbae21f8c38830ce337972b46fdae7b18eaaa"
            case .twentySevenB: "fcba37d2117a7077eac6b613b2668d14d9779edd"
            }
        }
        var fileSizes: [UInt64] {
            switch self {
            case .twoB: [1_200_284_784, 22_998, 87_679, 12_807_982, 16_709, 7_755]
            case .twentySevenB: [8_595_477_990, 58_145, 297_903, 12_809_320, 17_928, 8_952]
            }
        }
        var pins: [(String, String)] {
            switch self {
            case .twoB: PrismLocalRepacker.pins2B
            case .twentySevenB: PrismLocalRepacker.pins27B
            }
        }
        /// Conservative final pack reservation, including tokenizer and index.
        public var outputReservationBytes: UInt64 {
            switch self {
            case .twoB: 600_000_000
            case .twentySevenB: 7_800_000_000
            }
        }
        public var snapshotBytes: UInt64 { fileSizes.reduce(0, +) }
    }

    public static func discardPartial(outputDirectory: URL) throws {
        let lock = try InstallLock.acquire(outputDirectory: outputDirectory.standardizedFileURL.path)
        defer { withExtendedLifetime(lock) {} }
        guard try Posix.entryKind(lock.paths.checkpointFile) == .absent else {
            throw RepackError.installStateIncompatible(detail: "A different download owns the saved checkpoint")
        }
        if try Posix.entryKind(lock.paths.partialDirectory) == .directory {
            try FileManager.default.removeItem(atPath: lock.paths.partialDirectory)
        }
    }

    public static func install(size: Size, outputDirectory: URL, token: String? = nil,
                               baseURL: URL = URL(string: "https://huggingface.co")!,
                               downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
                               reserveBytes: UInt64 = 1_073_741_824,
                               progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws -> URL {
        let requested = outputDirectory.standardizedFileURL
        guard requested.isFileURL else {
            throw RepackError.configurationInvalid(detail: "Bonsai output must be a local directory")
        }
        let lock = try InstallLock.acquire(outputDirectory: requested.path)
        defer { withExtendedLifetime(lock) {} }
        let destination = URL(fileURLWithPath: lock.paths.finalDirectory, isDirectory: true)
        let fm = FileManager.default
        guard !fm.fileExists(atPath: destination.path) else {
            throw RepackError.configurationInvalid(detail: "Bonsai output already exists")
        }
        let staging = URL(fileURLWithPath: lock.paths.partialDirectory, isDirectory: true)
        guard try Posix.entryKind(staging.path) == .absent,
              try Posix.entryKind(lock.paths.checkpointFile) == .absent else {
            throw RepackError.installStateIncompatible(detail: "A partial Bonsai download exists; discard it before restarting")
        }
        try fm.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? fm.removeItem(at: staging) }
        let remote = HuggingFaceRemoteSource(repoID: size.repoID, requestedRevision: size.revision,
            token: token, downloadSession: downloadSession, baseURL: baseURL, tempDirectory: staging.path)
        progress(.downloadingMetadata)
        let pairs = Array(zip(size.pins, size.fileSizes))
        var infos: [RemoteFileInfo] = []
        for ((name, _), expectedSize) in pairs {
            try Task.checkCancellation()
            let info = try await remote.resolveFileInfo(filename: name)
            guard info.resolvedCommit == size.revision, info.size == expectedSize, info.acceptsRanges else {
                throw RepackError.remoteProtocolInvalid(detail: "Bonsai pinned file metadata differs for \(name)")
            }
            infos.append(info)
        }
        let planned = size.snapshotBytes + size.outputReservationBytes + 8 * 1024 * 1024
        progress(.planning(downloadBytes: size.snapshotBytes, outputBytes: size.outputReservationBytes))
        let space = try DiskSpaceChecker.requireAvailable(path: destination.path, bytes: planned, reserveBytes: reserveBytes)
        progress(.checkingDisk(space))
        var totalDownloaded: UInt64 = 0
        let chunkSize: UInt64 = 8 * 1024 * 1024
        for (index, info) in infos.enumerated() {
            try Task.checkCancellation()
            let path = staging.appendingPathComponent(info.filename)
            guard fm.createFile(atPath: path.path, contents: nil) else {
                throw RepackError.configurationInvalid(detail: "Could not create Bonsai staging file")
            }
            let handle = try FileHandle(forWritingTo: path)
            defer { try? handle.close() }
            var offset: UInt64 = 0
            while offset < info.size {
                try Task.checkCancellation()
                let length = Int(min(chunkSize, info.size - offset))
                let temp = try await remote.downloadRangeToTempFile(filename: info.filename, info: info,
                    offset: offset, length: length)
                do {
                    let data = try Data(contentsOf: URL(fileURLWithPath: temp.path))
                    guard data.count == length else {
                        throw RepackError.remoteProtocolInvalid(detail: "Incomplete Bonsai download range")
                    }
                    try handle.write(contentsOf: data)
                    try fm.removeItem(atPath: temp.path)
                } catch {
                    try? fm.removeItem(atPath: temp.path)
                    throw error
                }
                offset += UInt64(length)
                totalDownloaded += UInt64(length)
                progress(.copyingPayload(reusedBytes: 0, downloadedThisRunBytes: totalDownloaded,
                                         totalBytes: size.snapshotBytes))
            }
            try handle.close()
            progress(.hashingOutput(info.filename))
            let actual = try Sha256Stream.hashFile(path: path.path, noFollow: true)
            guard actual == pairs[index].0.1 else {
                throw RepackError.sourceFingerprintRejected(path: path.path, sha256: actual)
            }
        }
        try Task.checkCancellation()
        progress(.finalizing)
        try PrismLocalRepacker.write(snapshotDirectory: staging, outputDirectory: destination)
        return destination
    }
}
