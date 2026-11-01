import Foundation

/// Installs the pinned Flint OptiQ checkpoint with the streaming repacker.
public enum OptiQRemoteInstaller {
    public static let repoID = "mlx-community/flint-OptiQ-4bit"
    public static let revision = "d5efe414b30ee0cf2042ec712321a6fa060c2a53"

    public static func install(outputDirectory: URL, token: String? = nil,
                               baseURL: URL = URL(string: "https://huggingface.co")!,
                               downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
                               reserveBytes: UInt64 = 1_073_741_824,
                               resume: Bool = false,
                               progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws -> URL {
        guard outputDirectory.isFileURL else {
            throw RepackError.configurationInvalid(detail: "OptiQ output must be a local directory")
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
            throw RepackError.configurationInvalid(detail: "OptiQ output must be a local directory")
        }
        try RemoteStreamingRepacker.discardPartial(outputDirectory: outputDirectory.standardizedFileURL.path)
    }
}
