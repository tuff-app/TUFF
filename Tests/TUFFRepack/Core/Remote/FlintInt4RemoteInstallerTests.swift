import Foundation
import Testing
@testable import TUFFRepackCore

@Suite(.serialized) struct FlintInt4RemoteInstallerTests {
    @Test func pinsPublisherRepositoryAndRevision() {
        #expect(FlintInt4RemoteInstaller.repoID == "RecursiveSI/flint-int4")
        #expect(FlintInt4RemoteInstaller.revision == "3b93f9053b530488405ae9fca51d4900d6ce21f0")
    }

    @Test func rejectsDifferentResolvedCommitBeforeDownloading() async throws {
        resetFakeHF()
        FakeHFURLProtocol.commit = String(repeating: "b", count: 40)
        FakeHFURLProtocol.files["flint-int4.json"] = Data("{}".utf8)
        let output = URL(fileURLWithPath: tmpPathForRemote("flint-int4-wrong-commit"))
        defer { cleanUpRemote([output.path, output.path + ".partial", output.path + ".install.lock"]) }
        await #expect(throws: RepackError.self) {
            _ = try await FlintInt4RemoteInstaller.install(outputDirectory: output,
                baseURL: URL(string: "https://hf.test")!, downloadSession: fakeHFSession())
        }
        #expect(FakeHFURLProtocol.requestCounts["GET:flint-int4.json"] == nil)
        #expect(FakeHFURLProtocol.requestCounts.keys.allSatisfy { !$0.contains("model-0000") })
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    @Test func rejectsUnpinnedManifestContent() async throws {
        resetFakeHF()
        FakeHFURLProtocol.commit = FlintInt4RemoteInstaller.revision
        FakeHFURLProtocol.files["flint-int4.json"] = Data("{\"checkpoint_files\":{}}".utf8)
        let output = URL(fileURLWithPath: tmpPathForRemote("flint-int4-wrong-manifest"))
        defer { cleanUpRemote([output.path, output.path + ".partial", output.path + ".install.lock"]) }
        await #expect(throws: RepackError.self) {
            _ = try await FlintInt4RemoteInstaller.install(outputDirectory: output,
                baseURL: URL(string: "https://hf.test")!, downloadSession: fakeHFSession())
        }
        #expect(FakeHFURLProtocol.requestCounts.keys.allSatisfy { !$0.contains("model-0000") })
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    @Test func sidecarPinsRejectChangedContentAndCoverAllFiles() throws {
        let path = tmpPathForRemote("flint-int4-sidecar")
        defer { cleanUpRemote([path]) }
        try Data("changed".utf8).write(to: URL(fileURLWithPath: path))
        for digest in RemoteStreamingRepacker.flintInt4SidecarHashes.values {
            #expect(throws: RepackError.self) {
                try RemoteStreamingRepacker.verifyPinnedSidecar(path: path, sha256: digest)
            }
        }
        #expect(Set(RemoteStreamingRepacker.flintInt4SidecarHashes.keys) ==
                Set(["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]))
        #expect(RemoteStreamingRepacker.pinnedSidecars[FlintInt4RemoteInstaller.repoID] != nil)
    }

    /// Optional loopback fixture serving the complete pinned checkpoint from
    /// 127.0.0.1. Needs about the pack size (22 GB) of free space.
    @Test func installsPinnedLoopbackCheckpoint() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let address = env["TUFF_FLINT_INT4_LOCAL_URL"], let output = env["TUFF_FLINT_INT4_OUTPUT"] else { return }
        let url = try #require(URL(string: address))
        #expect(url.scheme == "http" && url.host == "127.0.0.1")
        guard url.scheme == "http", url.host == "127.0.0.1" else { return }
        let installed = try await FlintInt4RemoteInstaller.install(
            outputDirectory: URL(fileURLWithPath: output), baseURL: url)
        #expect(FileManager.default.fileExists(atPath: installed.path))
    }
}
