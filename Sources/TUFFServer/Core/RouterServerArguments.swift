import Foundation
import TUFFModelCatalog

/// Arguments for the model-routing server: `TUFFServer` in the foreground, or
/// `TUFFServer --background` as the login item the app registers.
public struct RouterServerArguments: Equatable, Sendable {
    public var background: Bool
    public var modelsRoot: String?
    public var host: String?
    public var port: Int?
    public var defaultModel: String?
    public var unloadDelay: TUFFModelUnloadDelay?
    public var queueLimit: Int?

    public static let usage = """
    usage: TUFFServer [options]
           TUFFServer --background

    Serves installed models and the available Apple Foundation Model. A local
    model loads when a
    request names it and unloads after it has been idle. Requests that name
    `default` use the default model. Each model uses its memory settings from
    the app's Settings > Models, within this Mac's memory budget. /v1/models
    reports the serving limits.

      --background               Run as TUFF's background API: read settings from
                                 Application Support/TUFF/Server and log to
                                 ~/Library/Logs/TUFF. The app starts this mode.
      --models-root <dir>        Installed models (default Application
                                 Support/TUFF/Models).
      --host <address>           Address to listen on (default 127.0.0.1, or the
                                 saved setting). 0.0.0.0 serves other devices on
                                 your network, which must then send the API key
                                 in Application Support/TUFF/Server/api-key.
      --port <1...65535>         Port (default 8080, or the saved setting).
      --default-model <name>     Model for requests that name `default`.
      --unload-after <seconds|immediately>
                                 Idle time before the model unloads (default 300).
      --queue-limit <count>      Maximum queued requests, 1-16 (default 4).
      --help                     Show this help.
    """

    /// Fixed-model flags that 7.1.0 removed, with what to use instead.
    static let removedFlags: [String: String] = [
        "--model": "the server routes each request by its \"model\" field; use --default-model <name>",
        "--model-id": "each model is served under its catalog ID",
        "--max-context": "each model uses a memory-checked serving context",
        "--vision-pack": "image packs are found beside each installed model",
        "--vision-residency": "image packs load on demand",
        "--prompt-cache-mode": "prompt reuse is always on",
        "--expert-cache-slots": "each model uses its catalog expert-cache size",
        "--expert-cache-policy": "each model uses its catalog expert-cache size",
        "--prefill": "each model uses its catalog prefill setting",
        "--prefill-chunk-tokens": "each model uses the prefill chunk chosen for this Mac",
        "--rdadvise": "read advice is not configurable",
    ]

    public static func parse(_ input: [String]) throws -> RouterServerArguments {
        var arguments = RouterServerArguments(background: false)
        var index = 0
        while index < input.count {
            let flag = input[index]
            index += 1
            switch flag {
            case "--help", "-h":
                throw ServerArgumentError.help
            case "--all-models":
                // 7.0.0 needed this to choose routing; it is now the only mode.
                continue
            case "--background":
                arguments.background = true
                continue
            default:
                if let replacement = removedFlags[flag] {
                    throw ServerArgumentError.invalid("\(flag) was removed in TUFF 7.1: \(replacement)")
                }
            }
            guard index < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index]
            index += 1
            switch flag {
            case "--models-root":
                arguments.modelsRoot = value
            case "--host":
                let address = value == "localhost" ? TUFFBackgroundServerSettings.loopbackHost : value
                guard TUFFBackgroundServerSettings.isAddress(address) else {
                    throw ServerArgumentError.invalid(
                        "--host must be an IP address, such as 127.0.0.1 or 0.0.0.0")
                }
                arguments.host = address
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                arguments.port = parsed
            case "--default-model":
                guard ["apple", "apple.on-device"].contains(value)
                    || ServerInstalledModels.descriptor(named: value) != nil else {
                    throw ServerArgumentError.invalid("unknown model: \(value)")
                }
                arguments.defaultModel = value
            case "--unload-after":
                if value == "immediately" {
                    arguments.unloadDelay = .immediately
                } else if let seconds = Int(value), seconds >= 0 {
                    arguments.unloadDelay = seconds == 0 ? .immediately : .seconds(seconds)
                } else {
                    throw ServerArgumentError.invalid(
                        "--unload-after must be a number of seconds or immediately")
                }
            case "--queue-limit":
                guard let parsed = Int(value), (1...16).contains(parsed) else {
                    throw ServerArgumentError.invalid("--queue-limit must be between 1 and 16")
                }
                arguments.queueLimit = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        return arguments
    }

    /// Settings this run uses: the saved file in background mode, with any
    /// explicit flag taking precedence.
    public func resolvedSettings(saved: TUFFBackgroundServerSettings?) -> TUFFBackgroundServerSettings {
        var settings = saved ?? TUFFBackgroundServerSettings()
        if let host { settings.host = host }
        if let port { settings.port = port }
        if let defaultModel { settings.defaultModel = defaultModel }
        if let unloadDelay { settings.unloadDelay = unloadDelay }
        if let queueLimit { settings.queueLimit = queueLimit }
        return settings
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
