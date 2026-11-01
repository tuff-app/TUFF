import Foundation
import Testing
@testable import TUFFRepackCore

/// Explicitly gated: this copies and verifies the full 22.6 GB pinned checkpoint.
@Suite struct FlintCheckpointIntegrationTests {
    @Test func repacksPinnedLocalCheckpoint() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["TUFF_FLINT_LOCAL_URL"],
              let output = env["TUFF_FLINT_OUTPUT"] else { return }
        let url = try #require(URL(string: address))
        #expect(url.scheme == "http" && url.host == "127.0.0.1")
        guard url.scheme == "http", url.host == "127.0.0.1" else { return }
        let revision = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
        let options = RemoteStreamingRepackOptions(
            repoID: "RecursiveSI/flint-int4", revision: revision, outputDir: output,
            baseURL: url, verifiedFileMemo: .init(directory: nil))
        let result = try await RemoteStreamingRepacker(options: options).run()
        #expect(result.resolvedCommit == revision)
        #expect(!result.dryRun)
        #expect(FileManager.default.fileExists(atPath: result.outputDir))
    }
}
