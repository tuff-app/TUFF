import Foundation
import ServiceManagement
import Testing
import TUFFModelCatalog
@testable import TUFFAppServer

@MainActor private final class FixtureAgent: BackgroundAgentService {
    var status: SMAppService.Status = .notRegistered
    var calls: [String] = []
    var fail = false
    func register() throws {
        calls.append("register")
        if fail { throw NSError(domain: "FixtureRegistration", code: 1) }
        status = .enabled
    }
    func unregister() async throws { calls.append("unregister"); status = .notRegistered }
}

@Suite(.serialized) struct AppBackgroundAPIControllerTests {
    @MainActor private func settle(_ controller: AppBackgroundAPIController) async throws {
        for _ in 0..<1000 {
            if !controller.isChanging { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        throw NSError(domain: "FixtureTimeout", code: 1)
    }
    @MainActor @Test func enablePortRestartAndDisablePersistInOrder() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let agent = FixtureAgent(), file = root.appendingPathComponent("settings.json")
        let controller = AppBackgroundAPIController(service: agent, settingsURL: file)
        #expect(controller.defaultModelID == TUFFModelCatalog.default.apiModelID)
        controller.update { $0.enabled = true; $0.port = 65431 }
        try await settle(controller)
        #expect(agent.calls == ["register"])
        controller.updatePort("65432")
        try await settle(controller)
        #expect(agent.calls == ["register", "unregister", "register"])
        controller.update { $0.enabled = false }
        try await settle(controller)
        #expect(agent.calls == ["register", "unregister", "register", "unregister"])
        #expect(TUFFBackgroundServerSettingsStore.load(from: file) == .loaded(controller.settings))
    }
    @MainActor @Test func registrationFailureRestoresSavedSettingAndCloneCannotEnable() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let agent = FixtureAgent(), file = root.appendingPathComponent("settings.json")
        agent.fail = true
        let controller = AppBackgroundAPIController(service: agent, settingsURL: file)
        agent.status = .notFound
        controller.update { $0.defaultModel = "gemma4-e2b" }
        try await settle(controller)
        #expect(agent.calls.isEmpty)
        controller.updatePort("65536")
        #expect(controller.message == "Port must be a number from 1 to 65535.")
        controller.update { $0.enabled = true }
        try await settle(controller)
        #expect(!controller.settings.enabled)
        #expect(controller.message?.contains("Could not change") == true)
        let clone = AppBackgroundAPIController(service: nil, settingsURL: root.appendingPathComponent("clone.json"))
        clone.update { $0.enabled = true }
        #expect(!clone.isAvailable && !clone.settings.enabled)
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("clone.json").path))
    }

    /// Allowing the network restarts the login item on every interface, with
    /// the key already on disk so the server reads the one the screen shows.
    @MainActor @Test func networkAccessCreatesTheKeyAndRestarts() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let agent = FixtureAgent(), file = root.appendingPathComponent("settings.json")
        let controller = AppBackgroundAPIController(service: agent, settingsURL: file)
        controller.update { $0.enabled = true; $0.port = 65434 }
        try await settle(controller)
        #expect(controller.apiKey == nil && controller.networkEndpoints.isEmpty)

        controller.setNetworkAccess(true)
        try await settle(controller)
        #expect(agent.calls == ["register", "unregister", "register"])
        #expect(controller.settings.host == "0.0.0.0")
        let key = try #require(controller.apiKey)
        #expect(key.count == 64)
        let keyFile = root.appendingPathComponent("api-key")
        let mode = try FileManager.default.attributesOfItem(atPath: keyFile.path)[.posixPermissions] as? Int
        #expect(mode == 0o600)
        #expect(controller.networkEndpoints.allSatisfy { $0.hasSuffix(":65434/v1") })

        controller.setNetworkAccess(false)
        try await settle(controller)
        #expect(controller.settings.host == "127.0.0.1" && controller.apiKey == nil)
        #expect(agent.calls.count == 5)
        // The key is kept, so a client configured once keeps working.
        controller.setNetworkAccess(true)
        try await settle(controller)
        #expect(controller.apiKey == key)
    }

    /// An update changes the binary the login item was pinned to, so the
    /// first launch of a new build registers it again, once.
    @MainActor @Test func aNewBuildRegistersTheLoginItemAgainOnce() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let agent = FixtureAgent(), file = root.appendingPathComponent("settings.json")
        let controller = AppBackgroundAPIController(service: agent, settingsURL: file)
        controller.update { $0.enabled = true; $0.port = 65433 }
        try await settle(controller)
        #expect(agent.calls == ["register"])

        controller.refreshRegistrationAfterUpdate(build: TUFFVersion.current)
        try await settle(controller)
        #expect(agent.calls == ["register"])

        controller.refreshRegistrationAfterUpdate(build: "99.0.0")
        try await settle(controller)
        #expect(agent.calls == ["register", "unregister", "register"])
        #expect(controller.registeredBuild == "99.0.0")
        controller.refreshRegistrationAfterUpdate(build: "99.0.0")
        try await settle(controller)
        #expect(agent.calls.count == 3)

        // Off stays off.
        controller.update { $0.enabled = false }
        try await settle(controller)
        controller.refreshRegistrationAfterUpdate(build: "100.0.0")
        try await settle(controller)
        #expect(agent.calls.last == "unregister")
    }
}
