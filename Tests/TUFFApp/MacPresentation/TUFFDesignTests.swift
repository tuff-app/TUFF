import AppKit
import Testing
@testable import TUFFMacPresentation

@MainActor
@Suite struct TUFFDesignTests {
    private func resolved(_ color: NSColor, dark: Bool) -> NSColor {
        let appearance = NSAppearance(named: dark ? .darkAqua : .aqua)!
        var result = color
        appearance.performAsCurrentDrawingAppearance {
            result = color.usingColorSpace(.sRGB) ?? color
        }
        return result
    }

    /// Every text color reads at WCAG AA (4.5:1) on every surface, in both
    /// modes. Light mode was where it fell short.
    @Test(arguments: [false, true])
    func textMeetsAAOnEverySurface(dark: Bool) {
        let surfaces = [("page", TUFFDesign.page), ("card", TUFFDesign.card),
                        ("inset", TUFFDesign.inset)]
        let texts = [("label", NSColor.labelColor), ("secondary", TUFFDesign.secondaryText),
                     ("tertiary", TUFFDesign.tertiaryText)]
        for (surfaceName, surface) in surfaces {
            for (textName, text) in texts {
                let ratio = TUFFDesign.contrast(resolved(text, dark: dark),
                                                resolved(surface, dark: dark))
                #expect(ratio >= 4.5, "\(textName) on \(surfaceName) dark=\(dark): \(ratio)")
            }
        }
    }

    /// Cards have to stand apart from the page, which is what light mode lost.
    @Test(arguments: [false, true])
    func cardsSeparateFromThePage(dark: Bool) {
        let ratio = TUFFDesign.contrast(resolved(TUFFDesign.card, dark: dark),
                                        resolved(TUFFDesign.page, dark: dark))
        #expect(ratio >= 1.08, "dark=\(dark): \(ratio)")
    }

    @Test func innerCornersAreConcentric() {
        #expect(TUFFDesign.Radius.inset
            == TUFFDesign.Radius.card - TUFFDesign.Radius.cardPadding)
        #expect(TUFFDesign.Radius.concentric(outer: 8, padding: 12) == 0)
    }
}
