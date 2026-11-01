import Foundation
import Testing
@testable import TUFFRepackCore

@Suite(.serialized) struct PrismRemoteInstallerTests {
    @Test func rejectsChangedPublisherFileBeforeDownloading() async throws {
        resetFakeHF()
        FakeHFURLProtocol.commit = PrismRemoteInstaller.Size.twoB.revision
        FakeHFURLProtocol.files["model.safetensors"] = Data([0])
        let output = URL(fileURLWithPath: tmpPathForRemote("bonsai-wrong-size"))
        defer { cleanUpRemote([output.path, output.path + ".partial", output.path + ".install.lock"]) }
        await #expect(throws: RepackError.self) {
            _ = try await PrismRemoteInstaller.install(size: .twoB, outputDirectory: output,
                baseURL: URL(string: "https://hf.test")!, downloadSession: fakeHFSession())
        }
        #expect(!FileManager.default.fileExists(atPath: output.path))
        #expect(!FileManager.default.fileExists(atPath: output.path + ".partial"))
        #expect(FakeHFURLProtocol.requestCounts["GET:model.safetensors"] == nil)
    }

    @Test func conservativeSpaceCoversSnapshotAndPack() {
        for size in [PrismRemoteInstaller.Size.twoB, .twentySevenB] {
            #expect(size.snapshotBytes > 0)
            #expect(size.outputReservationBytes > 0)
            #expect(size.pins.count == size.fileSizes.count)
            #expect(size.pins.allSatisfy { $0.1.count == 64 })
        }
    }
}
