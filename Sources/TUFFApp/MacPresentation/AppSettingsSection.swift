public enum AppSettingsSection: String, CaseIterable, Hashable, Identifiable, Sendable {
    case general
    case models
    case search
    case advanced

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .general: "General"
        // "Models" is the tab that downloads and manages them.
        case .models: "Generation"
        case .search: "Tools"
        case .advanced: "Advanced"
        }
    }

    public var systemImage: String {
        switch self {
        case .general: "switch.2"
        case .models: "memorychip"
        case .search: "magnifyingglass"
        case .advanced: "gearshape.2"
        }
    }
}
