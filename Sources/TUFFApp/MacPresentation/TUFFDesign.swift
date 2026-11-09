import AppKit
import SwiftUI

/// The app's design tokens: one place for surfaces, text colors, spacing and
/// corner radii, so screens stop picking their own.
///
/// Light mode needs its own values rather than the system's: on macOS 26 the
/// window background and a material card are both close to white, so cards
/// vanished. Here the page is a light grey and cards are white with a hairline
/// edge; dark mode lifts cards one step above the page instead.
public enum TUFFDesign {
    // MARK: Spacing

    public enum Space {
        public static let xxs: CGFloat = 4
        public static let xs: CGFloat = 8
        public static let sm: CGFloat = 12
        public static let md: CGFloat = 16
        public static let lg: CGFloat = 24
        public static let xl: CGFloat = 32
        /// The margin around a workspace's content.
        public static let page: CGFloat = 28
    }

    // MARK: Radii

    /// Nested shapes stay concentric: an inner corner is the outer corner minus
    /// the padding between them, so a tile inside a card never looks rounder
    /// than the card.
    public enum Radius {
        public static let card: CGFloat = 20
        public static let cardPadding: CGFloat = 14
        public static let inset: CGFloat = concentric(outer: card, padding: cardPadding)
        public static let insetPadding: CGFloat = 12
        /// Small controls and chips that sit on their own.
        public static let control: CGFloat = 8
        /// Tiles in the transcript: tool activity, thinking, attachments.
        public static let panel: CGFloat = 12
        /// Settings groups: grouped rows read better with a tighter corner
        /// than a content card.
        public static let form: CGFloat = 12

        public static func concentric(outer: CGFloat, padding: CGFloat) -> CGFloat {
            max(0, outer - padding)
        }
    }

    // MARK: Colors

    /// Behind every workspace.
    // Light mode sits a step darker than paper white, so white cards and
    // the composer stand off the page and the window is not glaring.
    public static let page = dynamic(light: 0xE9E9ED, dark: 0x1C1C1E)
    /// The windowed sidebar panel: a step off the page, as the native
    /// sidebar is, but opaque so it reads the same in a background window.
    public static let sidebar = dynamic(light: 0xF0F0F3, dark: 0x242427)
    /// A card on the page.
    /// Off-white in light mode: pure white cards glared against the page.
    public static let card = dynamic(light: 0xF8F8FA, dark: 0x2A2A2D)
    /// A tile or well inside a card.
    public static let inset = dynamic(light: 0xEDEDF1, dark: 0x343438)
    /// The edge of a card or inset. Barely there in dark mode, where the card
    /// tone already separates it from the page; a visible white line around
    /// every surface read as outlines rather than depth.
    public static let hairline = dynamicAlpha(light: (0x000000, 0.07), dark: (0xFFFFFF, 0.03))
    /// Body text is the system label color; these are checked at 4.5:1 or
    /// better against `page`, `card` and `inset` in both modes.
    public static let secondaryText = dynamic(light: 0x55555C, dark: 0xADADB4)
    public static let tertiaryText = dynamic(light: 0x5E5E65, dark: 0xA0A0A7)

    public static var pageColor: Color { Color(nsColor: page) }
    public static var sidebarColor: Color { Color(nsColor: sidebar) }
    public static var cardColor: Color { Color(nsColor: card) }
    public static var insetColor: Color { Color(nsColor: inset) }
    public static var hairlineColor: Color { Color(nsColor: hairline) }
    public static var secondaryTextColor: Color { Color(nsColor: secondaryText) }
    public static var tertiaryTextColor: Color { Color(nsColor: tertiaryText) }

    static func dynamic(light: UInt32, dark: UInt32) -> NSColor {
        NSColor(name: nil) { appearance in
            rgb(appearance.isDark ? dark : light)
        }
    }

    static func dynamicAlpha(light: (UInt32, CGFloat), dark: (UInt32, CGFloat)) -> NSColor {
        NSColor(name: nil) { appearance in
            let value = appearance.isDark ? dark : light
            return rgb(value.0).withAlphaComponent(value.1)
        }
    }

    static func rgb(_ hex: UInt32) -> NSColor {
        NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
                green: CGFloat((hex >> 8) & 0xFF) / 255,
                blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
    }

    // MARK: Contrast

    /// WCAG 2 contrast ratio between two opaque sRGB colors.
    public static func contrast(_ a: NSColor, _ b: NSColor) -> CGFloat {
        let la = luminance(a), lb = luminance(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    static func luminance(_ color: NSColor) -> CGFloat {
        guard let rgb = color.usingColorSpace(.sRGB) else { return 0 }
        func channel(_ value: CGFloat) -> CGFloat {
            value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
        }
        return 0.2126 * channel(rgb.redComponent) + 0.7152 * channel(rgb.greenComponent)
            + 0.0722 * channel(rgb.blueComponent)
    }
}

extension NSAppearance {
    var isDark: Bool {
        bestMatch(from: [.aqua, .darkAqua, .vibrantLight, .vibrantDark,
                         .accessibilityHighContrastAqua, .accessibilityHighContrastDarkAqua,
                         .accessibilityHighContrastVibrantLight,
                         .accessibilityHighContrastVibrantDark])
            .map { [.darkAqua, .vibrantDark, .accessibilityHighContrastDarkAqua,
                    .accessibilityHighContrastVibrantDark].contains($0) } ?? false
    }
}

// MARK: - Components

public extension View {
    /// A card: padding, the card surface, a hairline edge and, in light mode,
    /// a soft shadow so it lifts off the grey page.
    func tuffCard(padding: CGFloat = TUFFDesign.Radius.cardPadding) -> some View {
        modifier(TUFFCardModifier(padding: padding))
    }

    /// A tile or well inside a card, with the concentric inner radius.
    func tuffInset(padding: CGFloat = TUFFDesign.Radius.insetPadding) -> some View {
        self
            .padding(padding)
            .background(TUFFDesign.insetColor,
                        in: RoundedRectangle(cornerRadius: TUFFDesign.Radius.inset, style: .continuous))
    }

    /// A small standalone surface, such as a tool-activity or attachment
    /// tile: the card color and hairline at a smaller radius, no shadow.
    func tuffPanel(radius: CGFloat = TUFFDesign.Radius.panel) -> some View {
        let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
        return background(TUFFDesign.cardColor, in: shape)
            .overlay(shape.strokeBorder(TUFFDesign.hairlineColor, lineWidth: 1))
    }

    /// The page behind a workspace.
    func tuffPageBackground() -> some View {
        background(TUFFDesign.pageColor)
    }
}

struct TUFFCardModifier: ViewModifier {
    let padding: CGFloat
    @Environment(\.colorScheme) private var colorScheme

    func body(content: Content) -> some View {
        // The same card as a Settings section: 18 pt corners, no outline.
        let shape = RoundedRectangle(cornerRadius: 18, style: .continuous)
        content
            .padding(max(padding, 18))
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(TUFFDesign.cardColor, in: shape)
            .shadow(color: .black.opacity(colorScheme == .dark ? 0 : 0.05), radius: 6, y: 2)
    }
}

/// A short label beside a title, such as "Recommended". One style for every
/// badge in the app.
public struct TUFFBadge: View {
    public enum Tone { case accent, neutral, warning }
    let text: String
    let tone: Tone

    public init(_ text: String, tone: Tone = .accent) {
        self.text = text
        self.tone = tone
    }

    public var body: some View {
        Text(text)
            .appFont(.caption.weight(.semibold))
            .foregroundStyle(foreground)
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(background, in: Capsule())
    }

    private var foreground: Color {
        switch tone {
        case .accent: TUFFMacTheme.accentTextColor
        case .neutral: TUFFDesign.secondaryTextColor
        case .warning: TUFFMacTheme.accentTextColor
        }
    }

    private var background: Color {
        switch tone {
        case .accent: TUFFMacTheme.accentColor.opacity(0.14)
        case .neutral: Color.primary.opacity(0.07)
        case .warning: TUFFMacTheme.accentColor.opacity(0.14)
        }
    }
}

// MARK: - Forms

/// Settings drawn with the app's own cards. The system grouped form fills its
/// sections with a translucent grey that almost disappears on the page, and
/// it can't be recolored; this style keeps the same `Form { Section … }`
/// code and draws each section as a card, one row per control with the label
/// and control on a shared baseline.
public struct TUFFFormStyle: FormStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        TUFFFormBody(configuration: configuration, column: column)
    }

    fileprivate func column(_ sections: [SectionConfiguration]) -> some View {
        VStack(alignment: .leading, spacing: TUFFDesign.Space.md) {
            ForEach(sections, id: \.id) { section in
                card(section)
            }
        }
        .frame(maxWidth: .infinity, alignment: .top)
    }
}

/// The scrolling body of a Settings form: two columns of cards when the
/// window is wide enough, one otherwise. The choice comes from the measured
/// width. `ViewThatFits` laid the whole form out twice on every pass, and
/// with views that read model files from their bodies that froze the app.
private struct TUFFFormBody<Column: View>: View {
    let configuration: FormStyleConfiguration
    let column: ([SectionConfiguration]) -> Column
    @State private var isWide = false
    /// What the width asks for right now; applied once it has held still,
    /// so the layout does not flip between one and two columns partway
    /// through the sidebar sliding open or closed.
    @State private var wantsWide: Bool?

    var body: some View {
        ScrollView {
            Group(sections: configuration.content) { sections in
                let all = Array(sections)
                if isWide {
                    HStack(alignment: .top, spacing: TUFFDesign.Space.md) {
                        column(all.enumerated().filter { $0.offset.isMultiple(of: 2) }.map(\.element))
                        column(all.enumerated().filter { !$0.offset.isMultiple(of: 2) }.map(\.element))
                    }
                } else {
                    column(all)
                }
            }
            .padding(.horizontal, TUFFDesign.Space.page)
            .padding(.top, TUFFDesign.Space.sm)
            .padding(.bottom, TUFFDesign.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        // Judged by how far the page reaches across the window (its right
        // edge), not its own width: that stays the same when the sidebar
        // collapses, so the cards never change layout because of it.
        .onGeometryChange(for: Bool.self) { $0.frame(in: .global).maxX >= 1_200 } action: { wide in
            if wantsWide == nil { isWide = wide }
            wantsWide = wide
        }
        .task(id: wantsWide) {
            guard let wantsWide, wantsWide != isWide else { return }
            try? await Task.sleep(for: .milliseconds(380))
            guard !Task.isCancelled else { return }
            withAnimation(.easeInOut(duration: 0.22)) { isWide = wantsWide }
        }
        .labeledContentStyle(TUFFRowLabeledContentStyle())
        .toggleStyle(TUFFRowToggleStyle())
        // Regular-size controls, as in System Settings; large ones made every
        // row a different height.
        .controlSize(.regular)
    }
}

extension TUFFFormStyle {

    /// One section as a card with its title inside, rows under it, and any
    /// footer below the card.
    fileprivate func card(_ section: SectionConfiguration) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            VStack(alignment: .leading, spacing: 0) {
                if !section.header.isEmpty {
                    section.header
                        .appFont(.headline)
                        .foregroundStyle(.primary)
                        .padding(.top, 16)
                        .padding(.bottom, 4)
                        .accessibilityAddTraits(.isHeader)
                }
                Group(subviews: section.content) { rows in
                    ForEach(Array(rows.enumerated()), id: \.element.id) { index, row in
                        // A note tucks under the setting above it; first in
                        // its card, it is an ordinary row.
                        let isNote = row.containerValues.tuffSettingsNote && index > 0
                        if index > 0 && !isNote {
                            Divider().opacity(0.6)
                        }
                        row
                            .frame(maxWidth: .infinity, minHeight: isNote ? 0 : 32,
                                   alignment: .leading)
                            .padding(.top, isNote ? -4 : 10)
                            .padding(.bottom, 10)
                    }
                }
            }
            .padding(.horizontal, 18)
            .padding(.bottom, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(TUFFDesign.cardColor,
                        in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            if !section.footer.isEmpty {
                section.footer
                    .appFont(.caption)
                    .foregroundStyle(TUFFDesign.secondaryTextColor)
                    .padding(.horizontal, 18)
            }
        }
    }
}

/// A label on the leading edge and its value or control on the trailing
/// edge, aligned by the first line of text rather than by box height.
public struct TUFFRowLabeledContentStyle: LabeledContentStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        // Centred, not on the text baseline: controls without text (a color
        // well, a switch, a stepper) dropped the label below their middle.
        HStack(alignment: .center, spacing: TUFFDesign.Space.sm) {
            configuration.label
            Spacer(minLength: TUFFDesign.Space.sm)
            configuration.content
                .foregroundStyle(TUFFDesign.secondaryTextColor)
        }
    }
}

/// A switch on the trailing edge of the row, centred on the label's line.
public struct TUFFRowToggleStyle: ToggleStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        // The visible label is decoration for VoiceOver; the switch carries
        // the same label, hidden on screen, so it keeps its accessible name.
        HStack(alignment: .center, spacing: TUFFDesign.Space.sm) {
            configuration.label
                .accessibilityHidden(true)
            Spacer(minLength: TUFFDesign.Space.sm)
            TUFFSwitch(isOn: configuration.$isOn)
                .accessibilityLabel(Text("Toggle"))
        }
    }
}

/// The app's switch, drawn here so it looks the same in a background window.
/// The system switch turned grey whenever the window was not key.
public struct TUFFSwitch: View {
    @Binding var isOn: Bool
    @Environment(\.isEnabled) private var isEnabled
    @Environment(\.colorScheme) private var colorScheme
    @State private var isHovered = false

    public init(isOn: Binding<Bool>) { _isOn = isOn }

    public var body: some View {
        let dark = colorScheme == .dark
        Capsule()
            .fill(isOn ? AnyShapeStyle(TUFFMacTheme.accentColor)
                       : AnyShapeStyle(dark ? Color.white.opacity(0.16) : Color.black.opacity(0.12)))
            .frame(width: 38, height: 22)
            .overlay(alignment: isOn ? .trailing : .leading) {
                Circle()
                    .fill(.white)
                    .shadow(color: .black.opacity(0.22), radius: 1.5, y: 1)
                    .padding(2)
                    .scaleEffect(isHovered && isEnabled ? 1.05 : 1)
            }
            .opacity(isEnabled ? 1 : 0.45)
            .contentShape(Capsule())
            .onTapGesture { if isEnabled { isOn.toggle() } }
            .onHover { isHovered = $0 }
            .animation(TUFFMotion.standard, value: isOn)
            .animation(TUFFMotion.hover, value: isHovered)
            .accessibilityAddTraits(.isButton)
            .accessibilityValue(isOn ? "On" : "Off")
            .accessibilityAction { if isEnabled { isOn.toggle() } }
    }
}

// MARK: - Settings notes

extension ContainerValues {
    /// Marks a row as a note on the setting above it: the form style draws it
    /// under that setting, without a divider, in the caption style.
    @Entry var tuffSettingsNote = false
}

public enum TUFFSettingsNoteTone { case plain, warning }

public extension View {
    /// An explanation under a setting, drawn as part of that setting's row.
    func settingsNote(tone: TUFFSettingsNoteTone = .plain) -> some View {
        self
            .appFont(.caption)
            .foregroundStyle(tone == .warning ? AnyShapeStyle(TUFFMacTheme.accentTextColor)
                                              : AnyShapeStyle(TUFFDesign.secondaryTextColor))
            .fixedSize(horizontal: false, vertical: true)
            .containerValue(\.tuffSettingsNote, true)
    }
}

// MARK: - Buttons

/// The app's primary button: Liquid Glass tinted with the accent, a label
/// that stays readable on any accent, a plain look when disabled, and a
/// brighten-and-lift on hover.
public struct TUFFProminentButtonStyle: ButtonStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        TUFFGlassButtonBody(configuration: configuration, prominent: true)
    }
}

public extension ButtonStyle where Self == TUFFProminentButtonStyle {
    static var tuffProminent: TUFFProminentButtonStyle { TUFFProminentButtonStyle() }
}

/// Every other button: clear Liquid Glass with primary text, and the same
/// hover response as the primary button.
public struct TUFFSecondaryButtonStyle: ButtonStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        TUFFGlassButtonBody(configuration: configuration, prominent: false)
    }
}

/// The shared body of both button styles. A view rather than inline in the
/// style so it can hold its hover state.
struct TUFFGlassButtonBody: View {
    let configuration: ButtonStyleConfiguration
    let prominent: Bool
    @Environment(\.isEnabled) private var isEnabled
    @Environment(\.controlSize) private var controlSize
    @State private var isHovered = false

    var body: some View {
        let small = controlSize == .small || controlSize == .mini
        let shape = Capsule()
        let lit = isEnabled && isHovered
        configuration.label
            .appFont(small ? .callout.weight(.medium) : .body.weight(.medium))
            .lineLimit(1)
            .padding(.horizontal, small ? 12 : 16)
            .frame(minHeight: small ? 26 : 32)
            .foregroundStyle(!isEnabled ? AnyShapeStyle(TUFFDesign.tertiaryTextColor)
                             : prominent ? AnyShapeStyle(TUFFMacTheme.onAccentColor)
                             : AnyShapeStyle(.primary))
            .tuffGlass(shape, tint: prominent && isEnabled ? TUFFMacTheme.accentColor : nil,
                       highlighted: lit)
            .overlay(shape.fill(Color.black.opacity(configuration.isPressed ? 0.12 : 0)))
            .contentShape(shape)
            .scaleEffect(configuration.isPressed ? 0.97 : lit ? 1.03 : 1)
            .animation(TUFFMotion.hover, value: lit)
            .animation(TUFFMotion.hover, value: configuration.isPressed)
            .onHover { isHovered = $0 }
    }
}

// MARK: - Liquid Glass

public extension View {
    /// Puts Liquid Glass behind this view in `shape`: tinted when `tint` is
    /// given, brighter when `highlighted` (hovered). Before macOS 26, a
    /// solid fill of the tint or the card color.
    func tuffGlass<S: Shape>(_ shape: S, tint: Color? = nil, highlighted: Bool = false,
                             interactive: Bool = true) -> some View {
        modifier(TUFFGlassModifier(shape: shape, tint: tint, highlighted: highlighted,
                                   interactive: interactive))
    }
}

struct TUFFGlassModifier<S: Shape>: ViewModifier {
    let shape: S
    let tint: Color?
    let highlighted: Bool
    let interactive: Bool
    @Environment(\.colorScheme) private var colorScheme

    func body(content: Content) -> some View {
        let dark = colorScheme == .dark
        // Drawn entirely here, matching how Liquid Glass looks in the active
        // window: the tint (or a faint wash) and a fine light edge. The
        // system glass layer went flatter and lost its edge whenever the
        // window was not key, so the app changed when clicked away from.
        content
            .background {
                shape.fill(tint ?? (dark ? Color.white.opacity(0.07) : Color.black.opacity(0.035)))
                if highlighted { shape.fill(dark ? Color.white.opacity(0.07) : Color.black.opacity(0.04)) }
            }
            .overlay {
                shape.stroke(dark ? Color.white.opacity(0.14) : Color.black.opacity(0.08), lineWidth: 1)
                    .allowsHitTesting(false)
            }
    }
}

public extension ButtonStyle where Self == TUFFSecondaryButtonStyle {
    static var tuffSecondary: TUFFSecondaryButtonStyle { TUFFSecondaryButtonStyle() }
}

// MARK: - Disclosure

/// A disclosure whose whole header is the target: chevron and label share a
/// centre line, the chevron turns as it opens, and the content eases in.
/// The system style made only the small arrow clickable and set it above the
/// label's centre.
public struct TUFFDisclosureGroupStyle: DisclosureGroupStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(TUFFMotion.standard) {
                    configuration.isExpanded.toggle()
                }
            } label: {
                HStack(alignment: .center, spacing: 6) {
                    Image(systemName: "chevron.right")
                        .appFont(.caption.weight(.bold))
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                        .rotationEffect(.degrees(configuration.isExpanded ? 90 : 0))
                        .frame(width: 14, height: 14)
                    configuration.label
                    Spacer(minLength: 0)
                }
                .frame(minHeight: 24)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(.isButton)
            .accessibilityValue(configuration.isExpanded ? "Expanded" : "Collapsed")
            if configuration.isExpanded {
                configuration.content
                    .transition(.opacity)
            }
        }
        // As well as the `withAnimation` above: the window drops explicit
        // animations while a model is answering, which is when the Thinking
        // row is opened most.
        .animation(TUFFMotion.standard, value: configuration.isExpanded)
    }
}

public extension DisclosureGroupStyle where Self == TUFFDisclosureGroupStyle {
    static var tuff: TUFFDisclosureGroupStyle { TUFFDisclosureGroupStyle() }
}


// MARK: - Liquid Glass selection

/// The selected item's surface: a Liquid Glass capsule (or rounded rect) on
/// macOS 26, lightly tinted so it reads over flat surfaces, a raised fill
/// before it. Hovering brightens it and lifts it a touch.
public struct TUFFGlassPill: View {
    var cornerRadius: CGFloat?
    var isHovered: Bool

    /// `cornerRadius` nil is a capsule.
    public init(cornerRadius: CGFloat? = nil, isHovered: Bool = false) {
        self.cornerRadius = cornerRadius
        self.isHovered = isHovered
    }

    public var body: some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius ?? 999, style: .continuous)
        Color.clear
            .tuffGlass(shape, tint: nil, highlighted: isHovered)
            .scaleEffect(isHovered ? 1.03 : 1)
            .animation(TUFFMotion.hover, value: isHovered)
    }
}

/// Hover feedback for a bare icon button: a glass disc appears under the
/// pointer and the icon lifts slightly. Nothing is drawn at rest.
struct TUFFHoverGlass<S: Shape>: ViewModifier {
    let shape: S
    @State private var isHovered = false

    func body(content: Content) -> some View {
        content
            .background {
                if isHovered {
                    Color.clear
                        .tuffGlass(shape, highlighted: true)
                        .transition(.opacity)
                }
            }
            .scaleEffect(isHovered ? 1.06 : 1)
            .animation(TUFFMotion.hover, value: isHovered)
            .onHover { isHovered = $0 }
    }
}

public extension View {
    func tuffHoverGlass<S: Shape>(_ shape: S) -> some View {
        modifier(TUFFHoverGlass(shape: shape))
    }

    func tuffHoverGlass() -> some View {
        modifier(TUFFHoverGlass(shape: Circle()))
    }
}

// MARK: - Motion

/// The app's animation speeds, so every control moves at the same pace.
/// Springs throughout; some views used 0.12 s fades, which felt abrupt.
public enum TUFFMotion {
    /// Hover, press and other small responses to the pointer.
    public static let hover = Animation.spring(response: 0.32, dampingFraction: 0.78)
    /// Selection, toggles, showing and hiding, moving between tabs.
    public static let standard = Animation.spring(response: 0.42, dampingFraction: 0.86)
    /// Larger movements: the sidebar, the send flight, screens changing.
    public static let emphasized = Animation.spring(response: 0.55, dampingFraction: 0.84)
}

/// A fade with a little movement: the view grows from `scale` or rises
/// `offsetY` points as it appears, and the reverse as it goes. Under Reduce
/// Motion it only fades. The setting is read here, so a view using the
/// transition does not have to.
private struct TUFFArrivalModifier: ViewModifier {
    let isShown: Bool
    let scale: CGFloat
    let offsetY: CGFloat
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func body(content: Content) -> some View {
        content
            .opacity(isShown ? 1 : 0)
            .scaleEffect(isShown || reduceMotion ? 1 : scale)
            .offset(y: isShown || reduceMotion ? 0 : offsetY)
    }
}

public extension AnyTransition {
    static func tuffArrival(scale: CGFloat = 1, offsetY: CGFloat = 0) -> AnyTransition {
        .modifier(active: TUFFArrivalModifier(isShown: false, scale: scale, offsetY: offsetY),
                  identity: TUFFArrivalModifier(isShown: true, scale: scale, offsetY: offsetY))
    }
}

/// Just the app's switch, for a toggle whose label is shown elsewhere.
public struct TUFFSwitchToggleStyle: ToggleStyle {
    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        TUFFSwitch(isOn: configuration.$isOn)
    }
}

public extension View {
    /// The surface for chrome that floats over scrolling text (the status
    /// strip and the message box): the card color at 90%, so what scrolls
    /// beneath shows only faintly, with the glass's fine edge. Plain glass
    /// let the conversation show through clearly enough to clash with the
    /// strip's labels and the typed message.
    func tuffChrome<S: Shape>(_ shape: S) -> some View {
        modifier(TUFFChromeModifier(shape: shape))
    }
}

struct TUFFChromeModifier<S: Shape>: ViewModifier {
    let shape: S
    @Environment(\.colorScheme) private var colorScheme

    func body(content: Content) -> some View {
        let dark = colorScheme == .dark
        content
            .background(TUFFDesign.cardColor.opacity(0.9), in: shape)
            .overlay {
                shape.stroke(dark ? Color.white.opacity(0.14) : Color.black.opacity(0.08), lineWidth: 1)
                    .allowsHitTesting(false)
            }
    }
}

public extension TUFFMotion {
    /// Everything that moves with the sidebar opening or closing. One
    /// curve without overshoot for all of it, so the toggle, the strip and
    /// the column travel together; springs on top of the split view's own
    /// animation made them drift apart and wobble.
    static let sidebar = Animation.smooth(duration: 0.22)
}

/// The app's text field: a glass capsule like the dropdowns, with the text
/// inset from its ends. The system field was a dark sunken box that matched
/// nothing else. Use on a `TextField` in place of a field style.
public extension View {
    func tuffField() -> some View {
        self
            .textFieldStyle(.plain)
            .appFont(.callout)
            .padding(.horizontal, 14)
            .frame(minHeight: 30)
            .tuffGlass(Capsule())
    }
}
