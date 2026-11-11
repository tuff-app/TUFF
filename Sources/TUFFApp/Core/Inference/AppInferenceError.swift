import Foundation

public enum AppInferenceError: Error, Equatable, Sendable, CustomStringConvertible {
    case invalidRequest(String)
    case modelNotFound(String)
    case modelLoadFailed(String)
    case tokenizerUnavailable(String)
    case contextOverflow(prompt: Int, maxNew: Int, maxContext: Int)
    case generationInFlight
    case modelNotLoaded
    case reloadRequired
    case cancelled
    /// The model began a tool call the decoder could not read. Nothing from
    /// it was executed or shown.
    case malformedToolCall(String)
    case unknown(String)

    public var description: String { userMessage }

    /// The serving host has lost its family helper, so the cached ready state
    /// must be invalidated before the next submission can reload it.
    public var isEngineTransportFailure: Bool {
        switch self {
        case .unknown(let message), .modelLoadFailed(let message):
            return message.contains("ServingHelperTransportLost")
        default: return false
        }
    }

    public var userMessage: String {
        if isEngineTransportFailure {
            return "The model engine stopped unexpectedly. Load the model again to retry."
        }
        switch self {
        case .invalidRequest(let message):
            return message
        case .modelNotFound(let path):
            return "Model directory is not loadable: \(path)"
        case .modelLoadFailed(let message):
            return "Model load failed: \(message)"
        case .tokenizerUnavailable(let message):
            return "Tokenizer unavailable: \(message)"
        case .contextOverflow(let prompt, let maxNew, let maxContext):
            return "Prompt (\(prompt) tokens) plus max response (\(maxNew)) exceeds the \(maxContext)-token context."
        case .generationInFlight:
            return "A generation is already running."
        case .modelNotLoaded:
            return "Load the model before generating."
        case .reloadRequired:
            return "Model settings changed. Reload the model before generating."
        case .cancelled:
            return "Generation cancelled."
        case .malformedToolCall:
            return "The model produced a tool call TUFF could not read, so nothing was run."
        case .unknown(let message):
            return message
        }
    }

    public var technicalDetail: String {
        if isEngineTransportFailure {
            switch self {
            case .unknown(let message), .modelLoadFailed(let message): return message
            default: break
            }
        }
        switch self {
        case .tokenizerUnavailable:
            return "The installed tokenizer sidecar is missing or invalid. A Hugging Face fallback may require network access when no local sidecar is available."
        default:
            return userMessage
        }
    }
}
