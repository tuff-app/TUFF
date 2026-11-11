import Foundation
#if os(macOS)
import Security
#endif
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
    case appNotConfigured

    public enum Action: Sendable, Equatable {
        case select, openSystemSettings, refresh, none
    }
    public var action: Action {
        switch self {
        case .available: .select
        case .updateRequired, .intelligenceDisabled: .openSystemSettings
        case .preparing, .unknown: .refresh
        case .notSupported, .appNotConfigured: .none
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
        case .appNotConfigured: "Private Cloud Compute isn't available in this build of TUFF."
        }
    }

    public static func current(for kind: AppleModelKind = .onDevice) -> Self {
        #if canImport(FoundationModels) && os(macOS)
        if kind == .privateCloudCompute {
            #if compiler(>=6.4)
            guard #available(macOS 27, *) else { return .updateRequired }
            return privateCloudAvailability(hasEntitlement: hasPrivateCloudComputeEntitlement) {
                switch PrivateCloudComputeLanguageModel().availability {
                case .available: return .available
                case .unavailable(.deviceNotEligible): return .notSupported
                case .unavailable(.systemNotReady): return .preparing
                @unknown default: return .unknown
                }
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

    /// Device eligibility alone does not authorize the calling app to use PCC.
    /// Check signing before constructing the cloud model, which may otherwise
    /// advertise device availability even in an unentitled distribution.
    static func privateCloudAvailability(hasEntitlement: Bool,
                                         availability: () -> Self) -> Self {
        guard hasEntitlement else { return .appNotConfigured }
        return availability()
    }

    #if os(macOS)
    private static var hasPrivateCloudComputeEntitlement: Bool {
        var code: SecCode?
        var staticCode: SecStaticCode?
        var information: CFDictionary?
        guard SecCodeCopySelf([], &code) == errSecSuccess, let code,
              SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode,
              SecCodeCopySigningInformation(staticCode,
                  SecCSFlags(rawValue: kSecCSSigningInformation), &information) == errSecSuccess,
              let values = information as? [String: Any],
              let entitlements = values[kSecCodeInfoEntitlementsDict as String] as? [String: Any]
        else { return false }
        return entitlements["com.apple.developer.private-cloud-compute"] as? Bool == true
    }
    #endif
}
