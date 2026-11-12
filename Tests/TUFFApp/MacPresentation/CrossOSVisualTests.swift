import AppKit
import SwiftUI
import Testing
@testable import TUFFAppCore
@testable import TUFFAppServer
import TUFFAppUpdater
import TUFFMacPresentation
@testable import TUFFMac

/// Opt-in render artifacts. These supplement, rather than replace, native-window QA.
@Suite(.serialized) @MainActor struct CrossOSVisualTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TUFF_VISUAL_OUTPUT"] != nil))
    func captureRoot() async throws {
        let env = ProcessInfo.processInfo.environment
        let output = URL(fileURLWithPath: try #require(env["TUFF_VISUAL_OUTPUT"]))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let dark = env["TUFF_VISUAL_APPEARANCE"] == "dark"
        let width = Double(env["TUFF_VISUAL_WIDTH"] ?? "1080") ?? 1080
        let height = 720.0
        let model = AppModel(modelDirectory: output.appendingPathComponent("missing.gturbo"),
            client: VisualClient(), otherInstalls: [], settingsPersistenceEnabled: false)
        model.appleAvailabilityProvider = { .available }
        model.selectAppleModel()
        model.appearanceMode = dark ? .dark : .light
        model.windowTransparency = Double(env["TUFF_VISUAL_TRANSPARENCY"] ?? "0") ?? 0
        let root = RootView(model: model,
            backgroundAPI: AppBackgroundAPIController(service: nil,
                settingsURL: output.appendingPathComponent("server.json")),
            updateController: AppUpdateController(infoDictionary: nil))
            .environment(\.colorScheme, dark ? .dark : .light)
            .transaction { $0.disablesAnimations = true }
            .frame(width: width, height: height)
        let host = NSHostingView(rootView: root)
        host.frame = NSRect(x: 0, y: 0, width: width, height: height)
        let window = NSWindow(contentRect: host.frame, styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = host
        defer { window.contentView = nil; window.close() }
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        try await Task.sleep(for: .milliseconds(500))
        host.layoutSubtreeIfNeeded()
        let bitmap = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let png = try #require(bitmap.representation(using: .png, properties: [:]))
        try png.write(to: output.appendingPathComponent("render.png"))
        let metadata: [String: Any] = ["os": ProcessInfo.processInfo.operatingSystemVersionString,
            "appearance": dark ? "dark" : "light", "width": width, "height": height,
            "pixelWidth": bitmap.pixelsWide, "pixelHeight": bitmap.pixelsHigh,
            "transparency": model.windowTransparency,
            "screen": env["TUFF_OPEN_SCREEN"] ?? "chat",
            "sidebar": env["TUFF_DEBUG_SIDEBAR"] ?? "expanded",
            "capture": "offscreen-content; excludes native frame and desktop compositing"]
        try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("metadata.json"))
        #expect(bitmap.pixelsWide > 0 && bitmap.pixelsHigh > 0)
    }
}
private final class VisualClient: AppInferenceClient, @unchecked Sendable {
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
    func cancel() {}
}
