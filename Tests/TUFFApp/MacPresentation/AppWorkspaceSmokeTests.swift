import AppKit
import ServiceManagement
import Testing
import TUFFModelCatalog
import TUFFAppCore
@testable import TUFFAppServer
import TUFFAppUpdater
@testable import TUFFMac
import TUFFMacPresentation
import SwiftUI

@Suite(.serialized) @MainActor struct AppWorkspaceSmokeTests {
    @Test func everyDestinationRendersAtTheMinimumWindowSize() throws {
        let client = WorkspaceSmokeLifecycleClient()
        let model = AppModel(
            modelDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "TUFFWorkspaceSmoke-\(UUID().uuidString).gturbo",
                    isDirectory: true),
            client: client,
            otherInstalls: [],
            deviceCapabilities: TUFFDeviceCapabilities(
                unifiedMemoryBytes: 16 * 1_024 * 1_024 * 1_024,
                macOSMajorVersion: 15,
                appleSiliconGeneration: 2))
        let backgroundAPI = AppBackgroundAPIController(
            service: nil,
            settingsURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("TUFFWorkspaceSmoke-\(UUID().uuidString).json"))
        let updateController = AppUpdateController(infoDictionary: nil)
        for destination in AppDestination.allCases {
            let content = AppWorkspaceView(
                destination: destination,
                model: model,
                backgroundAPI: backgroundAPI,
                updateController: updateController,
                benchmarks: BenchmarkController())
                .frame(
                    width: AppWindowLayout.detailMinimumWidth,
                    height: AppWindowLayout.minimumHeight)
                .transaction { $0.disablesAnimations = true }
            let renderer = ImageRenderer(content: content)
            renderer.scale = 1
            let image = try #require(renderer.nsImage)
            let data = try #require(image.tiffRepresentation)

            #expect(image.size == NSSize(
                width: AppWindowLayout.detailMinimumWidth,
                height: AppWindowLayout.minimumHeight))
            #expect(!data.isEmpty)
        }
    }
}

@MainActor private final class WorkspaceSmokeAgent: BackgroundAgentService {
    var status: SMAppService.Status = .enabled
    func register() throws {}
    func unregister() async throws {}
}

extension AppWorkspaceSmokeTests {
    /// The packaged app's Server screen, which a clone build never shows.
    @Test func packagedServerScreenRendersAtTheMinimumWindowSize() throws {
        let model = AppModel(
            modelDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "TUFFWorkspaceSmoke-\(UUID().uuidString).gturbo",
                    isDirectory: true),
            client: WorkspaceSmokeLifecycleClient(),
            otherInstalls: [])
        let backgroundAPI = AppBackgroundAPIController(
            service: WorkspaceSmokeAgent(),
            settingsURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("TUFFWorkspaceSmoke-\(UUID().uuidString).json"))
        #expect(backgroundAPI.isAvailable)
        let content = AppWorkspaceView(
            destination: .server,
            model: model,
            backgroundAPI: backgroundAPI,
            updateController: AppUpdateController(infoDictionary: nil),
            benchmarks: BenchmarkController())
            .frame(
                width: AppWindowLayout.detailMinimumWidth,
                height: AppWindowLayout.minimumHeight)
            .transaction { $0.disablesAnimations = true }
        let renderer = ImageRenderer(content: content)
        renderer.scale = 1
        let image = try #require(renderer.nsImage)
        #expect(image.size == NSSize(
            width: AppWindowLayout.detailMinimumWidth,
            height: AppWindowLayout.minimumHeight))
    }
}

private final class WorkspaceSmokeLifecycleClient: AppModelLifecycleClient,
    @unchecked Sendable {
    func generate(
        _ request: AppGenerationRequest
    ) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }

    func cancel() {}

    func ensureLoaded(
        modelDirectory: URL,
        maxContextTokens: Int,
        options: AppRuntimeOptions,
        forceLogitsHead: Bool,
        onState: @escaping @Sendable (AppModelLoadState) -> Void
    ) async throws {}

    func unload() async {}
}
