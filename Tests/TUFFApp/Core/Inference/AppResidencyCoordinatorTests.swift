import Foundation
import Testing
import TUFFModelCatalog
@testable import TUFFAppCore

@Suite(.serialized) struct AppResidencyCoordinatorTests {
    @Test func imageWorkspaceBlocksOtherModelsOnlyWhileInUse() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TUFFVisionAdmission-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = TUFFResidencyRegistry(directory: root)
        let coordinator = AppResidencyCoordinator(registry: registry, budgetBytes: 100, reclaim: { _ in false })
        let owner = try registry.reserve(.init(owner: .app, modelID: "owner", estimatedBytes: 30), budgetBytes: 100)
        let other = try registry.reserve(.init(owner: .app, modelID: "other", estimatedBytes: 20), budgetBytes: 100)
        #expect(throws: AppInferenceError.self) { try coordinator.reserveVision(allowing: [owner]) }
        other.release()
        let image = try coordinator.reserveVision(allowing: [owner])
        #expect(throws: TUFFResidencyRegistry.MemoryBusy.self) {
            try registry.reserve(.init(owner: .app, modelID: "new", estimatedBytes: 1), budgetBytes: 100)
        }
        image.release()
        let new = try registry.reserve(.init(owner: .app, modelID: "new", estimatedBytes: 20), budgetBytes: 100)
        new.release()
        owner.release()
    }
    @Test func installedVisionCompanionDoesNotReserveWholeTextBudget() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TUFFTextAdmission-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let pack = root.appendingPathComponent("model.gturbo")
        let vision = root.appendingPathComponent("model.vision.gturbo")
        try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: vision, withIntermediateDirectories: true)
        try JSONEncoder().encode(["modelID": TUFFModelCatalog.gemma4_E2B.source.manifestModelID])
            .write(to: pack.appendingPathComponent("manifest.json"))
        let registry = TUFFResidencyRegistry(directory: root.appendingPathComponent("leases"))
        let budget = UInt64(16) * 1_073_741_824
        let coordinator = AppResidencyCoordinator(registry: registry, budgetBytes: budget, reclaim: { _ in false })
        let text = try await coordinator.reserve(directory: pack, context: 1024, slots: 16)
        #expect(text.record.estimatedBytes < budget)
        let peer = try registry.reserve(.init(owner: .app, modelID: "other-text", estimatedBytes: 1),
                                        budgetBytes: budget)
        peer.release()
        text.release()
    }
    @Test func renamedAndCustomPacksAlwaysHoldLeases() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TUFFAppResidency-\(UUID())")
        let pack = root.appendingPathComponent("private-name")
        try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = TUFFResidencyRegistry(directory: root.appendingPathComponent("leases"))
        let coordinator = AppResidencyCoordinator(registry: registry, budgetBytes: 123,
            reclaim: { _ in false })
        for identity in [TUFFModelCatalog.gemma4_E2B.source.manifestModelID, "custom"] {
            try JSONEncoder().encode(["modelID": identity]).write(to: pack.appendingPathComponent("manifest.json"))
            let lease = try await coordinator.reserve(directory: pack, context: 4096, slots: 16)
            #expect(registry.activeRecords().count == 1)
            #expect(registry.activeRecords()[0].modelID == (identity == "custom" ? "custom" : TUFFModelCatalog.gemma4_E2B.apiModelID))
            lease.release()
        }
    }
    @Test func idleServerYieldsBeforeAppReservation() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TUFFAppResidency-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = TUFFResidencyRegistry(directory: root)
        let model = TUFFModelCatalog.gemma4_E2B
        let needed = model.memory.estimatedWorkingSetBytes(contextTokens: 4096, expertCacheSlots: 16)
        let server = try registry.reserve(.init(owner: .backgroundServer, modelID: model.apiModelID,
            estimatedBytes: needed, controlPort: 9999), budgetBytes: needed)
        let coordinator = AppResidencyCoordinator(registry: registry, budgetBytes: needed, reclaim: { port in
            #expect(port == 9999)
            server.release()
            return true
        })
        let app = try await coordinator.reserve(model: model, context: 4096, slots: 16)
        #expect(registry.activeRecords().map(\.owner) == [.app])
        app.release()
        #expect(registry.activeRecords().isEmpty)
    }
    @Test func busyServerIsNotInterrupted() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TUFFAppResidency-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = TUFFResidencyRegistry(directory: root)
        let server = try registry.reserve(.init(owner: .backgroundServer, modelID: "other",
            estimatedBytes: 100, controlPort: 9999), budgetBytes: 100)
        defer { server.release() }
        let coordinator = AppResidencyCoordinator(registry: registry, budgetBytes: 100, reclaim: { _ in false })
        await #expect(throws: AppInferenceError.self) {
            try await coordinator.reserve(model: TUFFModelCatalog.gemma4_E2B, context: 4096, slots: 16)
        }
        #expect(registry.activeRecords().map(\.owner) == [.backgroundServer])
    }
}
