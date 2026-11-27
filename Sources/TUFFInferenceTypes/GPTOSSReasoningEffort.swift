public enum ModelReasoningEffort: String, Codable, Sendable, Equatable, CaseIterable {
    case low
    case medium
    case high
}

public typealias GPTOSSReasoningEffort = ModelReasoningEffort
