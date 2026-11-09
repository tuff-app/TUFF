import AppKit
import SwiftUI
import TUFFAppCore

/// The app's live accent color, driven by the user's appearance setting
/// (default purple, synced with the system accent color, or a custom hex
/// value). `TUFFMacTheme` reads through this singleton so every call site
/// that already references `TUFFMacTheme.accentColor` stays reactive without
/// threading `AppModel` through the view hierarchy.
///
/// `RootView` pushes the persisted setting in with `update(mode:customHex:)`
/// whenever it changes. While the mode is `.system`, this also listens for
/// `NSColor.systemColorsDidChangeNotification` so a live change to the
/// macOS accent color in System Settings is picked up immediately.
@MainActor
@Observable
public final class TUFFAccentThemeStore {
    public static let shared = TUFFAccentThemeStore()

    public private(set) var mode: AppAccentColorMode = .appDefault
    public private(set) var customHex: String = AppHexColor.defaultPurple.hexString

    /// Bumped on `NSColor.systemColorsDidChangeNotification` purely so
    /// `nsColor` has a tracked stored property to read while in `.system`
    /// mode — the actual color comes from `NSColor.controlAccentColor`,
    /// which Observation cannot see change on its own.
    private var systemColorTick = 0

    // `shared` outlives the app, so the observer is never torn down — there is
    // no deinit that would need to remove it.
    private init() {
        NotificationCenter.default.addObserver(
            forName: NSColor.systemColorsDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.systemColorTick += 1
            }
        }
    }

    public func update(mode: AppAccentColorMode, customHex: String) {
        guard self.mode != mode || self.customHex != customHex else { return }
        self.mode = mode
        self.customHex = customHex
    }

    public var nsColor: NSColor {
        switch mode {
        case .appDefault:
            return Self.nsColor(for: .defaultPurple)
        case .system:
            _ = systemColorTick
            return NSColor.controlAccentColor
        case .custom:
            let hex = AppHexColor(hexString: customHex) ?? .defaultPurple
            return Self.nsColor(for: hex)
        }
    }

    public var markWingNSColor: NSColor {
        switch mode {
        case .appDefault:
            return Self.nsColor(for: AppHexColor(
                red: 167.0 / 255.0, green: 139.0 / 255.0, blue: 250.0 / 255.0))
        case .system, .custom:
            return Self.lightened(nsColor, by: 0.35)
        }
    }

    private static func nsColor(for hex: AppHexColor) -> NSColor {
        NSColor(srgbRed: hex.red, green: hex.green, blue: hex.blue, alpha: 1)
    }

    private static func lightened(_ color: NSColor, by fraction: CGFloat) -> NSColor {
        guard let rgb = color.usingColorSpace(.deviceRGB) else { return color }
        return NSColor(
            red: rgb.redComponent + (1 - rgb.redComponent) * fraction,
            green: rgb.greenComponent + (1 - rgb.greenComponent) * fraction,
            blue: rgb.blueComponent + (1 - rgb.blueComponent) * fraction,
            alpha: 1)
    }
}

public enum TUFFMacTheme {
    @MainActor
    public static var accentNSColor: NSColor { TUFFAccentThemeStore.shared.nsColor }

    @MainActor
    public static var accentColor: Color { Color(nsColor: accentNSColor) }

    /// The accent for text, icons and outlines: the chosen color, darkened in
    /// light mode or lightened in dark mode until it reads at 3:1 on the page.
    /// A pale custom accent otherwise made accent text invisible (white menu
    /// labels on white rows). Fills keep `accentColor` exactly.
    @MainActor
    public static var accentTextColor: Color {
        let accent = accentNSColor
        return Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            var resolved = NSColor.black
            appearance.performAsCurrentDrawingAppearance {
                resolved = accent.usingColorSpace(.sRGB) ?? accent
            }
            let page = TUFFDesign.page
            var pageResolved = NSColor.black
            appearance.performAsCurrentDrawingAppearance {
                pageResolved = page.usingColorSpace(.sRGB) ?? page
            }
            let target: NSColor = isDark ? .white : .black
            var color = resolved
            var step: CGFloat = 0
            while TUFFDesign.contrast(color, pageResolved) < 3, step < 1 {
                step += 0.1
                color = resolved.blended(withFraction: step, of: target) ?? color
            }
            return color
        })
    }

    /// Text and icons drawn on an accent fill: white, or black when the
    /// accent is light enough that white would not read (a custom white or
    /// pale accent made the New Chat plus and Send label disappear).
    @MainActor
    public static var onAccentColor: Color {
        let color = accentNSColor.usingColorSpace(.sRGB) ?? accentNSColor
        return TUFFDesign.contrast(color, .white) >= TUFFDesign.contrast(color, .black)
            ? .white : .black
    }

    @MainActor
    public static var markWingNSColor: NSColor { TUFFAccentThemeStore.shared.markWingNSColor }

    @MainActor
    public static var markWingColor: Color { Color(nsColor: markWingNSColor) }

    public static func surfaceStyle(
        reduceTransparency: Bool,
        material: Material = .regular
    ) -> AnyShapeStyle {
        if reduceTransparency {
            return AnyShapeStyle(Color(nsColor: .controlBackgroundColor))
        }
        return AnyShapeStyle(material)
    }
}
