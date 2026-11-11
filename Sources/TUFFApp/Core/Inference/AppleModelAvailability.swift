import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

/// Public Apple model endpoints, each admitted independently by this Mac.
public enum AppleModelKind: String, CaseIterable, Sendable, Identifiable {
    case onDevice = "apple.on-device"
    case privateCloudCompute = "apple.private-cloud-compute"
    public var id: String { rawValue }
    public var name: String {
        switch self {
        case .onDevice: "Apple Foundation Model"
        case .privateCloudCompute: "Apple · Private Cloud Compute"
        }
    }
}

/// Apple owns model installation and configuration. These are actions, never downloads.
public enum AppleModelAvailability: String, Sendable, CaseIterable {
    case available, updateRequired, notSupported, intelligenceDisabled, preparing, unknown

    public enum Action: Sendable, Equatable {
        case select, openSystemSettings, refresh, none
    }
    public var action: Action {
        switch self {
        case .available: .select
        case .updateRequired, .intelligenceDisabled: .openSystemSettings
        case .preparing, .unknown: .refresh
        case .notSupported: .none
        }
    }

    public var canGenerate: Bool { self == .available }
    public var label: String {
        switch self {
        case .available: "Ready"
        case .updateRequired: "Update macOS to Use"
        case .notSupported: "Not Supported on This Mac"
        case .intelligenceDisabled: "Set Up Apple Intelligence"
        case .preparing: "Preparing Apple Intelligence"
        case .unknown: "Apple Intelligence Unavailable"
        }
    }

    public static func current(for kind: AppleModelKind = .onDevice) -> Self {
        #if canImport(FoundationModels) && os(macOS)
        if kind == .privateCloudCompute {
            #if compiler(>=6.4)
            guard #available(macOS 27, *) else { return .updateRequired }
            switch PrivateCloudComputeLanguageModel().availability {
            case .available: return .available
            case .unavailable(.deviceNotEligible): return .notSupported
            case .unavailable(.systemNotReady): return .preparing
            @unknown default: return .unknown
            }
            #else
            return .unknown
            #endif
        }
        guard #available(macOS 26, *) else {
            #if arch(arm64)
            return .updateRequired
            #else
            return .notSupported
            #endif
        }
        switch SystemLanguageModel.default.availability {
        case .available: return .available
        case .unavailable(let reason):
            switch reason {
            case .deviceNotEligible: return .notSupported
            case .appleIntelligenceNotEnabled: return .intelligenceDisabled
            case .modelNotReady: return .preparing
            @unknown default: return .unknown
            }
        @unknown default: return .unknown
        }
        #else
        return .unknown
        #endif
    }
}
