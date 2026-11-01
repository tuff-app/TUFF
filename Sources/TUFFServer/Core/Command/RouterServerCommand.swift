import Darwin
import Foundation
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFModelMetadata
import TUFFModelCatalog

/// The `TUFFServer` executable, callable from any binary that links the server.
public enum RouterServerCommand {
    public static func main(_ arguments: [String]) async -> Int32 {
        // Every request line goes to stderr, which is unbuffered, while the
        // ready line goes to stdout, which is fully buffered when it is not a
        // terminal. Line buffering puts the line where it is useful.
        setvbuf(stdout, nil, _IOLBF, 0)
        if arguments.contains("--private-host") {
            return await runPrivateHost(arguments)
        }
        do {
            let parsed = try RouterServerArguments.parse(arguments)
            return await RouterServerRuntime.run(parsed)
        } catch ServerArgumentError.help {
            print(RouterServerArguments.usage)
            return 0
        } catch {
            FileHandle.standardError.write(
                Data("error: \(error)\n\n\(RouterServerArguments.usage)\n".utf8))
            return 2
        }
    }

    /// The dedicated serving host. It is started by the GUI or API adapter,
    /// is never registered as a login item, and exits once every client has
    /// gone and its work has drained.
    private static func runPrivateHost(_ arguments: [String]) async -> Int32 {
        let support = TUFFApplicationDirectories.applicationSupportURL()
        RouterServerRuntime.redirectLog(
            to: support.appendingPathComponent("TUFF/Logs/serving-host.log"))
        let bundleID = arguments.firstIndex(of: "--serving-bundle-id").flatMap {
            $0 + 1 < arguments.count ? arguments[$0 + 1] : nil
        }
        do {
            let profile = try ServingHostProfile.forCurrentProcess(bundleIdentifier: bundleID)
            let installed = ServerInstalledModels(
                modelsRoot: support.appendingPathComponent("TUFF/Models", isDirectory: true))
            return await ServingHostRuntime.runPrivateHost(profile: profile, installed: installed)
        } catch {
            FileHandle.standardError.write(Data("serving host profile error: \(error)\n".utf8))
            return 1
        }
    }
}
