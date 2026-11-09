/// Light or dark for TUFF alone, or whatever macOS is set to.
public enum AppAppearanceMode: String, CaseIterable, Codable, Sendable, Identifiable, Hashable {
    case system
    case light
    case dark

    public var id: String { rawValue }

    public var settingsLabel: String {
        switch self {
        case .system: return "System"
        case .light: return "Light"
        case .dark: return "Dark"
        }
    }
}
