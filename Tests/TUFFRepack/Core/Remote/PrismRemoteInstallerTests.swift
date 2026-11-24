import Foundation
import Testing
import TUFFModelCatalog
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
        for size in [PrismRemoteInstaller.Size.twoB, .twentySevenB, .twentySevenB25] {
            #expect(size.snapshotBytes > 0)
            #expect(size.outputReservationBytes > 0)
            #expect(size.pins.count == size.fileSizes.count)
            #expect(size.pins.allSatisfy { $0.1.count == 64 })
        }
    }

    @Test func bonsai25IsInstalledFromItsOwnPinnedRevision() {
        #expect(PrismRemoteInstaller.Size(modelID: .bonsai25_27B) == .twentySevenB25)
        #expect(PrismRemoteInstaller.Size(modelID: .bonsai2_27B) == .twentySevenB)
        let catalog = TUFFModelCatalog.bonsai25_27B.source
        let size = PrismRemoteInstaller.Size.twentySevenB25
        #expect(size.repoID == catalog.repoID && size.revision == catalog.revision)
        #expect(size.pins[0].1 == catalog.sourceIndexSHA256)
        #expect(size.pins.map(\.0) == PrismRemoteInstaller.Size.twentySevenB.pins.map(\.0))
    }

    /// Bonsai 2 27B and 2.5 27B ship the same config.json, so the repacker
    /// must tell them apart by their weights and never mislabel one as the other.
    @Test func sharedConfigIsResolvedByTheWeights() throws {
        let config = PrismLocalRepacker.pins27B[1].1
        #expect(config == PrismLocalRepacker.pins25_27B[1].1)
        let v2 = try PrismLocalRepacker.source(configSHA256: config,
            weightsSHA256: { PrismLocalRepacker.pins27B[0].1 }, modelID: nil)
        #expect(v2?.modelID == "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit")
        let v25 = try PrismLocalRepacker.source(configSHA256: config,
            weightsSHA256: { PrismLocalRepacker.pins25_27B[0].1 }, modelID: nil)
        #expect(v25?.modelID == "prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit")
        // A download for one model never installs as the other.
        let mismatch = try PrismLocalRepacker.source(configSHA256: config,
            weightsSHA256: { PrismLocalRepacker.pins27B[0].1 },
            modelID: "prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit")
        #expect(mismatch == nil)
        var hashed = false
        let unknown = try PrismLocalRepacker.source(configSHA256: String(repeating: "0", count: 64),
            weightsSHA256: { hashed = true; return "" }, modelID: nil)
        #expect(unknown == nil && !hashed)
    }
}
