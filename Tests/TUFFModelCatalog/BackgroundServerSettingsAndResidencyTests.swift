import Foundation
import Testing
@testable import TUFFModelCatalog

@Suite(.serialized) struct BackgroundServerSettingsAndResidencyTests {
    private func root() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("TUFFResidencyTests-\(UUID())")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    @Test func settingsRoundTripAndNewerFilePreservation() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("settings.json")
        #expect(TUFFBackgroundServerSettingsStore.load(from: url) == .missing)
        for delay in TUFFModelUnloadDelay.choices {
            let s = TUFFBackgroundServerSettings(enabled: true, defaultModel: "gemma4-e2b", unloadDelay: delay)
            try TUFFBackgroundServerSettingsStore.save(s, to: url)
            #expect(TUFFBackgroundServerSettingsStore.load(from: url) == .loaded(s))
        }
        let newer = Data(#"{"version":2,"future":"untouched"}"#.utf8)
        try newer.write(to: url)
        #expect(throws: TUFFBackgroundServerSettingsStore.SaveError.newerVersionOnDisk(2)) {
            try TUFFBackgroundServerSettingsStore.save(.init(), to: url)
        }
        #expect(try Data(contentsOf: url) == newer)
        #expect(throws: TUFFBackgroundServerSettingsStore.SaveError.invalidSettings) {
            try TUFFBackgroundServerSettingsStore.save(.init(port: 0), to: url)
        }
    }
    @Test func aFileWithoutHostStaysOnLoopback() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("settings.json")
        // As 10.2.0 wrote it.
        try Data(#"{"defaultModel":"gemma4-e2b","enabled":true,"port":8080,"queueLimit":4,"unloadDelay":300,"version":1}"#.utf8).write(to: url)
        guard case .loaded(let settings) = TUFFBackgroundServerSettingsStore.load(from: url) else {
            Issue.record("not loaded"); return
        }
        #expect(settings.host == "127.0.0.1" && !settings.listensBeyondLoopback && settings.enabled)
        let network = TUFFBackgroundServerSettings(enabled: true, host: "0.0.0.0")
        try TUFFBackgroundServerSettingsStore.save(network, to: url)
        #expect(TUFFBackgroundServerSettingsStore.load(from: url) == .loaded(network))
        #expect(network.listensBeyondLoopback)
        #expect(throws: TUFFBackgroundServerSettingsStore.SaveError.invalidSettings) {
            try TUFFBackgroundServerSettingsStore.save(.init(host: "example.com"), to: url)
        }
    }
    @Test func loopbackAddresses() {
        for host in ["127.0.0.1", "127.8.9.10", "::1", "0:0:0:0:0:0:0:1"] {
            #expect(TUFFBackgroundServerSettings.isLoopback(host), "\(host)")
        }
        for host in ["0.0.0.0", "::", "192.168.1.20", "128.0.0.1", "::2", "fe80::1", "localhost"] {
            #expect(!TUFFBackgroundServerSettings.isLoopback(host), "\(host)")
        }
    }
    @Test func liveLeaseReleaseAndStaleCleanup() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let registry = TUFFResidencyRegistry(directory: root)
        let record = TUFFResidencyRecord(owner: .app, modelID: "test", estimatedBytes: 70)
        let lease = try registry.reserve(record, budgetBytes: 100)
        #expect(registry.activeRecords() == [record])
        #expect(registry.activeRecords(excluding: [lease]).isEmpty)
        #expect(throws: TUFFResidencyRegistry.MemoryBusy.self) {
            try registry.reserve(record, budgetBytes: 100)
        }
        lease.release(); lease.release()
        #expect(registry.activeRecords().isEmpty)
        let stale = root.appendingPathComponent("dead.json")
        try JSONEncoder().encode(record).write(to: stale)
        #expect(registry.activeRecords().isEmpty)
        #expect(!FileManager.default.fileExists(atPath: stale.path))
    }
    @Test func simultaneousReservationsCannotOvercommit() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let registry = TUFFResidencyRegistry(directory: root)
        let leases = await withTaskGroup(of: TUFFResidencyLease?.self, returning: [TUFFResidencyLease].self) { group in
            for _ in 0..<12 {
                group.addTask { try? registry.reserve(.init(owner: .app, modelID: "test", estimatedBytes: 60), budgetBytes: 100) }
            }
            var result: [TUFFResidencyLease] = []
            for await lease in group { if let lease { result.append(lease) } }
            return result
        }
        #expect(leases.count == 1)
        leases.forEach { $0.release() }
    }
    @Test func overflowCannotAdmitAlongsideAnotherModel() {
        #expect(TUFFResidencyAdmission.evaluate(neededBytes: .max,
            others: [.init(owner: .app, modelID: "a", estimatedBytes: 1)], budgetBytes: 100)
            != .admitted)
    }
}
