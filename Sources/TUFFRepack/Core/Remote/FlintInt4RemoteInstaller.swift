import Foundation

/// Installs the pinned publisher Flint INT4 checkpoint with the streaming
/// repacker. The shard digests, manifest digest and tokenizer sidecars are all
/// pinned: see `RemoteSnapshotLoader.loadFlint` and
/// `RemoteStreamingRepacker.flintInt4SidecarHashes`.
public enum FlintInt4RemoteInstaller {
    public static let repoID = "RecursiveSI/flint-int4"
    public static let revision = "3b93f9053b530488405ae9fca51d4900d6ce21f0"

    public static func install(outputDirectory: URL, token: String? = nil,
                               baseURL: URL = URL(string: "https://huggingface.co")!,
                               downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
                               reserveBytes: UInt64 = 1_073_741_824,
                               resume: Bool = false,
                               progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws -> URL {
        guard outputDirectory.isFileURL else {
            throw RepackError.configurationInvalid(detail: "Flint INT4 output must be a local directory")
        }
        let options = RemoteStreamingRepackOptions(
            repoID: repoID, revision: revision, outputDir: outputDirectory.standardizedFileURL.path,
            token: token, minFreeReserveBytes: reserveBytes, resume: resume,
            downloadSession: downloadSession, baseURL: baseURL)
        let result = try await RemoteStreamingRepacker(options: options).run(progress: progress)
        return URL(fileURLWithPath: result.outputDir, isDirectory: true)
    }

    public static func discardPartial(outputDirectory: URL) throws {
        guard outputDirectory.isFileURL else {
            throw RepackError.configurationInvalid(detail: "Flint INT4 output must be a local directory")
        }
        try RemoteStreamingRepacker.discardPartial(outputDirectory: outputDirectory.standardizedFileURL.path)
    }
}
