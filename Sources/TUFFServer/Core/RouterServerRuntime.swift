import Darwin
import Foundation
import Synchronization
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFModelMetadata
import TUFFModelCatalog

/// Runs the model-routing server until SIGINT or SIGTERM.
public enum RouterServerRuntime {
    public static func applicationSupportURL() -> URL {
        TUFFApplicationDirectories.applicationSupportURL()
    }

    public static func logURL() -> URL {
        if ProcessInfo.processInfo.environment["TUFF_APPLICATION_SUPPORT_ROOT"] != nil {
            return applicationSupportURL().appendingPathComponent("TUFF/Logs/background-server.log")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/TUFF/background-server.log")
    }

    /// Sends this process's stderr to the background log, keeping one older
    /// file once it grows past `rotateBytes`.
    public static func redirectLog(to url: URL, rotateBytes: UInt64 = 5 << 20) {
        let fileManager = FileManager.default
        try? fileManager.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let size = (try? fileManager.attributesOfItem(atPath: url.path))?[.size] as? UInt64,
           size > rotateBytes {
            let previous = url.appendingPathExtension("1")
            try? fileManager.removeItem(at: previous)
            try? fileManager.moveItem(at: url, to: previous)
        }
        freopen(url.path, "a", stderr)
        freopen(url.path, "a", stdout)
        setvbuf(stdout, nil, _IOLBF, 0)
    }

    /// Returns the exit status.
    public static func run(_ arguments: RouterServerArguments) async -> Int32 {
        let support = applicationSupportURL()
        let settingsURL = TUFFBackgroundServerSettingsStore.fileURL(applicationSupport: support)
        var saved: TUFFBackgroundServerSettings?
        if arguments.background {
            redirectLog(to: logURL())
            switch TUFFBackgroundServerSettingsStore.load(from: settingsURL) {
            case .loaded(let settings):
                guard settings.enabled else {
                    write("background API is turned off in TUFF; exiting")
                    return 0
                }
                saved = settings
            case .missing:
                write("no background API settings; exiting")
                return 0
            case .newer(let version):
                write("settings version \(version) is newer than this build; exiting without changing it")
                return 0
            case .unreadable:
                write("settings could not be read; exiting without starting the API")
                return 0
            }
        }
        let settings = arguments.resolvedSettings(saved: saved)
        let modelsRoot = arguments.modelsRoot.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? support.appendingPathComponent("TUFF/Models", isDirectory: true)
        let installed = ServerInstalledModels(modelsRoot: modelsRoot)
        let source = ServerSettingsSource(
            url: arguments.background ? settingsURL : nil, fallback: settings)
        let profile: ServingHostProfile
        do {
            profile = try ServingHostProfile.forCurrentProcess()
        } catch {
            write("error: could not open the serving host profile: \(error)")
            return 1
        }
        let tokenURL = ServerControlToken.fileURL(applicationSupport: support)
        let token: String
        do {
            token = try ServerControlToken.read(at: tokenURL)
                ?? ServerControlToken.create(at: tokenURL)
        } catch {
            write("error: could not create the control token: \(error)")
            return 1
        }

        let host = settings.host
        let port = settings.port
        var apiKey: String?
        if settings.listensBeyondLoopback {
            do {
                apiKey = try ServerAPIKey.readOrCreate(
                    at: ServerAPIKey.fileURL(applicationSupport: support))
            } catch {
                write("error: could not create the API key: \(error)")
                return 1
            }
        }
        let link = HostServingLink(profile: profile)
        let provider = HostRoutedModelProvider(installed: installed, settings: source, link: link)
        let control = ServerControl(
            token: token,
            status: { await ServerControl.status(provider: provider, version: TUFFVersion.current) },
            unloadIfIdle: { await provider.unloadIfIdle() })
        let server = TUFFHTTPServer(provider: provider, control: control, apiKey: apiKey)
        let signals = ServerTerminationSignals()
        do {
            _ = try await server.start(host: host, port: port)
        } catch {
            write("error: could not listen on \(Self.authority(host, port)): \(error)")
            await signals.cancel()
            return 1
        }
        // Attaching marks this process as the API adapter in the host's status
        // and starts the host when none runs. A failure is retried with the
        // first request that needs it.
        do {
            if !installed.available().isEmpty { _ = try await link.attach() }
        } catch {
            write("warning: the inference host is not reachable yet: \(error)")
        }
        let names = provider.modelList().data.map(\.id).joined(separator: ",")
        if apiKey != nil {
            print("Other devices on your network can connect. They must send the API key in "
                + ServerAPIKey.fileURL(applicationSupport: support).path)
        }
        print("TUFFServer \(TUFFVersion.current) routing at http://\(Self.authority(host, port)) "
            + "default=\(settings.defaultModel) unload_after=\(settings.unloadDelay.seconds)s "
            + "models=\(names.isEmpty ? "none" : names)")
        _ = await signals.wait()
        do {
            try await server.shutdown()
        } catch {
            write("error during shutdown: \(error)")
        }
        await signals.cancel()
        return 0
    }

    static func authority(_ host: String, _ port: Int) -> String {
        host.contains(":") ? "[\(host)]:\(port)" : "\(host):\(port)"
    }

    private static func write(_ message: String) {
        FileHandle.standardError.write(Data("[\(Date().formatted(.iso8601))] \(message)\n".utf8))
    }
}
