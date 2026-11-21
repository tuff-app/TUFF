import Foundation
import Testing
@testable import TUFFRepackCore

/// Explicitly gated: this copies and verifies the full 16.7 GB pinned checkpoint
/// from a loopback server that serves a local copy of it.
@Suite struct InklingCheckpointIntegrationTests {
    @Test func repacksPinnedLocalCheckpoint() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["TUFF_INKLING_LOCAL_URL"],
              let output = env["TUFF_INKLING_OUTPUT"] else { return }
        let url = try #require(URL(string: address))
        #expect(url.scheme == "http" && url.host == "127.0.0.1")
        guard url.scheme == "http", url.host == "127.0.0.1" else { return }
        let revision = "9856dc53edead71405f8348930937c3638ff21a2"
        let options = RemoteStreamingRepackOptions(
            repoID: "mlx-community/Inkling-2-Mini-4bit", revision: revision, outputDir: output,
            baseURL: url, verifiedFileMemo: .init(directory: nil))
        let result = try await RemoteStreamingRepacker(options: options).run()
        #expect(result.resolvedCommit == revision)
        #expect(!result.dryRun)
        #expect(FileManager.default.fileExists(atPath: result.outputDir))
    }
}
