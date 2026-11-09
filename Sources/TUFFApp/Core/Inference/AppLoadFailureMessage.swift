import Foundation
import TUFFRuntimeManagement

/// What to say when a load fails with an error from below the app.
///
/// Loading a model downloads its engine the first time, so a missing
/// connection or a refused download reaches the person as a load failure.
/// Printed as the Swift errors they are, those read
/// `unexpectedHTTPStatus(404)`.
enum AppLoadFailureMessage {
    static func text(for error: any Error) -> String {
        if let error = error as? InferenceEngineDownloadError {
            switch error {
            case .unexpectedHTTPStatus(let status):
                return "TUFF could not download the engine this model needs (the server answered \(status)). "
                    + "Try again in a moment, or check for a TUFF update."
            case .oversizedDownload, .unsupportedURL:
                return "TUFF could not download the engine this model needs. Check for a TUFF update."
            }
        }
        if let error = error as? URLError {
            switch error.code {
            case .notConnectedToInternet, .networkConnectionLost, .cannotFindHost, .cannotConnectToHost,
                 .dnsLookupFailed, .timedOut, .dataNotAllowed, .internationalRoamingOff:
                return "TUFF downloads the engine for this model once, and it could not reach the internet. "
                    + "Connect and try again."
            default:
                return "TUFF could not download the engine this model needs: \(error.localizedDescription)"
            }
        }
        return "\(error)"
    }
}
