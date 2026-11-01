import Foundation
import Testing
@testable import TUFFRepackCore

@Suite(.serialized) struct OptiQRemoteInstallerTests {
    @Test func rejectsDifferentResolvedCommitBeforeDownloading() async throws {
        resetFakeHF()
        FakeHFURLProtocol.commit = String(repeating: "a", count: 40)
        FakeHFURLProtocol.files["model.safetensors.index.json"] = Data("{}".utf8)
        let output = URL(fileURLWithPath: tmpPathForRemote("optiq-wrong-commit"))
        defer { cleanUpRemote([output.path, output.path + ".partial", output.path + ".install.lock"]) }
        await #expect(throws: RepackError.self) {
            _ = try await OptiQRemoteInstaller.install(outputDirectory: output,
                baseURL: URL(string: "https://hf.test")!, downloadSession: fakeHFSession())
        }
        #expect(FakeHFURLProtocol.requestCounts["GET:model.safetensors.index.json"] == nil)
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    @Test func rejectsChangedIndexBeforeFetchingShards() async throws {
        let snapshotDir = tmpDirForRemote("optiq-changed-index")
        let output = URL(fileURLWithPath: tmpPathForRemote("optiq-changed-output"))
        defer { cleanUpRemote([snapshotDir, output.path, output.path + ".partial", output.path + ".install.lock"]) }
        let snapshot = try SyntheticSnapshot.buildDenseGemmaE4B(at: snapshotDir)
        resetFakeHF()
        FakeHFURLProtocol.commit = OptiQRemoteInstaller.revision
        FakeHFURLProtocol.files = try remoteFiles(snapshotDir: snapshotDir, snap: snapshot,
            includeRequiredTokenizer: true, includeOptionalTokenizer: true)
        await #expect(throws: RepackError.self) {
            _ = try await OptiQRemoteInstaller.install(outputDirectory: output,
                baseURL: URL(string: "https://hf.test")!, downloadSession: fakeHFSession())
        }
        #expect(!FakeHFURLProtocol.requestCounts.keys.contains { $0.hasPrefix("HEAD:model-") })
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    @Test func verifiesSidecarContentsAgainstPinnedDigest() throws {
        let path = tmpPathForRemote("optiq-sidecar")
        defer { cleanUpRemote([path]) }
        try Data().write(to: URL(fileURLWithPath: path))
        let emptySHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        try RemoteStreamingRepacker.verifyPinnedSidecar(path: path, sha256: emptySHA)
        for digest in RemoteStreamingRepacker.optiqSidecarHashes.values {
            #expect(throws: RepackError.self) {
                try RemoteStreamingRepacker.verifyPinnedSidecar(path: path, sha256: digest)
            }
        }
        try Data("changed".utf8).write(to: URL(fileURLWithPath: path))
        #expect(throws: RepackError.self) {
            try RemoteStreamingRepacker.verifyPinnedSidecar(path: path, sha256: emptySHA)
        }
        #expect(Set(RemoteStreamingRepacker.optiqSidecarHashes.keys) ==
                Set(["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]))
    }

    /// Optional loopback fixture serving the complete pinned checkpoint.
    @Test func installsPinnedLoopbackCheckpoint() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["TUFF_OPTIQ_LOCAL_URL"], let output = env["TUFF_OPTIQ_OUTPUT"] else { return }
        let url = try #require(URL(string: address))
        #expect(url.scheme == "http" && url.host == "127.0.0.1")
        guard url.scheme == "http", url.host == "127.0.0.1" else { return }
        let installed = try await OptiQRemoteInstaller.install(
            outputDirectory: URL(fileURLWithPath: output), baseURL: url)
        #expect(FileManager.default.fileExists(atPath: installed.path))
    }
}
