import Foundation
import Testing
@testable import TUFFRepackCore

/// Opt-in local loopback fixture covering the real pinned 2B snapshot installer.
@Suite struct PrismCheckpointIntegrationTests {
    @Test func installsPinnedSnapshot() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["TUFF_BONSAI_LOCAL_URL"],
              let output = env["TUFF_BONSAI_INSTALL_OUTPUT"] else { return }
        let url = try #require(URL(string: address))
        guard url.scheme == "http", url.host == "127.0.0.1" else {
            Issue.record("The checkpoint integration fixture must use loopback")
            return
        }
        let destination = URL(fileURLWithPath: output)
        let result = try await PrismRemoteInstaller.install(size: .twoB, outputDirectory: destination,
            baseURL: url)
        #expect(FileManager.default.fileExists(atPath: result.appendingPathComponent("manifest.json").path))
        let receipt = try #require(JSONSerialization.jsonObject(with: Data(contentsOf:
            result.appendingPathComponent(VerifiedInstallReceiptWriter.fileName))) as? [String: Any])
        #expect(receipt["modelDirectoryPath"] as? String == result.standardizedFileURL.path)
        #expect(receipt["manifestSha256"] as? String
            == (try Sha256Stream.hashFile(path: result.appendingPathComponent("manifest.json").path)))
        let files = try #require(receipt["files"] as? [String: Any])
        #expect(files.keys.contains("model_weights.bin") && files.keys.contains("tokenizer/tokenizer.json"))
        #expect(!FileManager.default.fileExists(atPath: result.path + ".partial"))
    }
}
