import AppKit
import Testing
@testable import TUFFMac
import TUFFMacPresentation

@Suite(.serialized) @MainActor struct WindowChromeTests {
    @Test func buttonsDispatchEachWindowActionOnceWithoutRepressingWidgets() throws {
        _ = NSApplication.shared
        let window = WindowActionProbe(contentRect: NSRect(x: 0, y: 0, width: 640, height: 440),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.collectionBehavior.insert(.fullScreenNone)
        let controls = WindowChromeConfiguration.ChromeView(frame: NSRect(x: 18, y: 18, width: 62, height: 16))
        try #require(window.contentView).addSubview(controls)
        controls.layoutSubtreeIfNeeded()
        let buttons = [NSWindow.ButtonType.closeButton, .miniaturizeButton, .zoomButton]
            .compactMap { window.standardWindowButton($0) }
        try #require(buttons.count == 3)
        #expect(buttons.allSatisfy { !$0.isHidden })
        for button in buttons { button.performClick(nil) }
        #expect(window.closeCount == 1)
        #expect(window.minimizeCount == 1)
        #expect(window.zoomCount == 1)
        #expect(window.titlebarAppearsTransparent)

        let veto = CloseVeto()
        window.delegate = veto
        buttons[0].performClick(nil)
        #expect(window.closeCount == 1)
    }
}

@MainActor private final class WindowActionProbe: NSWindow {
    var closeCount = 0
    var minimizeCount = 0
    var zoomCount = 0
    override func close() { closeCount += 1 }
    override func miniaturize(_ sender: Any?) { minimizeCount += 1 }
    override func zoom(_ sender: Any?) { zoomCount += 1 }
}

@MainActor private final class CloseVeto: NSObject, NSWindowDelegate {
    func windowShouldClose(_ sender: NSWindow) -> Bool { false }
}
