import Foundation
import TUFFModelCatalog

public enum TUFFCommandPlan: Equatable, Sendable {
    case help
    case version
    case run(executableURL: URL, arguments: [String])
    case load(
        launcherURL: URL,
        arguments: [String],
        selection: String,
        modelURL: URL)
}

public enum TUFFCommandError: Error, Equatable, CustomStringConvertible {
    case unknownCommand(String)
    case missingValue(String)
    case duplicateModel
    case missingPrompt
    case unexpectedArgument(String)
    case unknownModel(String)
    case modelNotInstalled(String)
    case missingBundledExecutable(String)
    case serveModelRemoved

    public var description: String {
        switch self {
        case .unknownCommand(let command):
            "unknown command: \(command)"
        case .missingValue(let flag):
            "missing value for \(flag)"
        case .duplicateModel:
            "--model may only be supplied once"
        case .missingPrompt:
            "prompt text is required, for example: tuff prompt \"Hello\" "
                + "(text that starts with -- goes after --chat-prompt)"
        case .unexpectedArgument(let argument):
            "unexpected argument: \(argument)"
        case .unknownModel(let model):
            "unknown model: \(model)"
        case .modelNotInstalled(let path):
            "model is not installed at \(path)"
        case .missingBundledExecutable(let name):
            "bundled executable is missing: \(name)"
        case .serveModelRemoved:
            "tuff serve serves every installed model, so --model was removed in "
                + "TUFF 7.1; name the model in each request or pass --default-model <name>"
        }
    }
}

public enum TUFFCommand {
    /// Every catalog selector, wrapped like the rest of the help.
    static let modelNames: String = {
        let names = TUFFModelCatalog.all.map(\.selector)
        let sentence = "model names: " + names.dropLast().joined(separator: ", ")
            + ", and " + names.last! + "."
        var lines: [String] = [], line = ""
        for word in sentence.split(separator: " ") {
            if !line.isEmpty, line.count + 1 + word.count > 72 {
                lines.append(line)
                line = ""
            }
            line += (line.isEmpty ? "" : " ") + word
        }
        return (lines + [line]).joined(separator: "\n")
    }()

    public static let usage = """
    TUFF — local model command line

    usage:
      tuff prompt <text> [--model <name|path>] [generation options]
      tuff load [model]
      tuff serve [--default-model <name>] [--unload-after <seconds>]
                 [--host <address>] [--port <port>]
      tuff bench (--models <name,...> | --all) [--quick] [--share]
      tuff --version

    commands:
      prompt   Run a one-shot chat prompt. Uses the selected app model by default.
      load     Select an installed model, open TUFF, and load it into the app.
      serve    Start the local OpenAI-compatible server in the foreground. Every
               installed model is served and loads when a request names it;
               `default` means the selected app model, or --default-model.
               (The app's Background API has its own default, set on the
               Server screen.)
      bench    Benchmark installed models and optionally share the result on
               GitHub Discussions.

    \(modelNames)

    Run `tuff prompt --help`, `tuff serve --help` or `tuff bench --help` for
    command-specific options.
    """

    public static func plan(
        arguments: [String],
        executableURL: URL,
        currentDirectoryURL: URL,
        applicationSupportURL: URL,
        environment: [String: String] = [:],
        selectedModel: String? = nil,
        device: TUFFDeviceCapabilities = .current(),
        fileExists: (String) -> Bool = FileManager.default.fileExists(atPath:)
    ) throws -> TUFFCommandPlan {
        guard let command = arguments.first else { return .help }
        if command == "help" || command == "--help" || command == "-h" {
            return .help
        }
        if command == "--version" || command == "version" {
            return .version
        }

        let remaining = Array(arguments.dropFirst())
        switch command {
        case "prompt":
            return try promptPlan(
                remaining,
                device: device,
                executableURL: executableURL,
                currentDirectoryURL: currentDirectoryURL,
                applicationSupportURL: applicationSupportURL,
                environment: environment,
                selectedModel: selectedModel,
                fileExists: fileExists)
        case "load":
            return try loadPlan(
                remaining,
                executableURL: executableURL,
                currentDirectoryURL: currentDirectoryURL,
                applicationSupportURL: applicationSupportURL,
                environment: environment,
                selectedModel: selectedModel,
                fileExists: fileExists)
        case "bench", "benchmark":
            return try benchPlan(
                remaining, executableURL: executableURL, fileExists: fileExists)
        case "serve":
            return try servePlan(
                remaining,
                executableURL: executableURL,
                currentDirectoryURL: currentDirectoryURL,
                applicationSupportURL: applicationSupportURL,
                selectedModel: selectedModel,
                fileExists: fileExists)
        default:
            throw TUFFCommandError.unknownCommand(command)
        }
    }

    private static func promptPlan(
        _ arguments: [String],
        device: TUFFDeviceCapabilities,
        executableURL: URL,
        currentDirectoryURL: URL,
        applicationSupportURL: URL,
        environment: [String: String],
        selectedModel: String?,
        fileExists: (String) -> Bool
    ) throws -> TUFFCommandPlan {
        let extracted = try extractModel(arguments)
        let model = try resolveModel(
            selection: extracted.selection,
            currentDirectoryURL: currentDirectoryURL,
            applicationSupportURL: applicationSupportURL,
            executableURL: executableURL,
            environment: environment,
            selectedModel: selectedModel,
            fileExists: fileExists)
        var forwarded = extracted.remaining
        if let first = forwarded.first, !first.hasPrefix("-") {
            guard !first.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { throw TUFFCommandError.missingPrompt }
            forwarded.removeFirst()
            guard !containsAny(["--prompt", "--chat-prompt", "--messages-file"], in: forwarded)
            else { throw TUFFCommandError.unexpectedArgument(first) }
            forwarded.insert(contentsOf: ["--chat-prompt", first], at: 0)
        }
        guard forwarded == ["--help"]
                || containsAny(["--prompt", "--chat-prompt", "--messages-file"], in: forwarded)
        else { throw TUFFCommandError.missingPrompt }

        // Explicit memory controls remain manual. The ordinary prompt command
        // uses the same bounded planner as the app, including MoE projections.
        if model.descriptor != nil, !containsAny(["--memory-auto", "--expert-cache-slots", "--dense-residency",
                         "--dense-weight-budget-mb", "--prefill", "--prefill-chunk-tokens",
                         "--prefill-chunk-max"], in: forwarded), forwarded != ["--help"] {
            forwarded += ["--memory-auto"]
        }

        if let descriptor = model.descriptor {
            addDefault("--max-context", value: descriptor.runtimeDefaults.contextTokens, to: &forwarded)
            addDefault(
                "--expert-cache-slots",
                value: descriptor.runtimeDefaults.expertCacheSlots,
                to: &forwarded)
            addDefault("--temperature", value: descriptor.runtimeDefaults.temperature, to: &forwarded)
            addDefault("--top-k", value: descriptor.runtimeDefaults.topK, to: &forwarded)
            addDefault("--top-p", value: descriptor.runtimeDefaults.topP, to: &forwarded)
            if descriptor.reasoningControl == .alwaysOn,
               !forwarded.contains("--thinking") {
                forwarded += ["--thinking", "on"]
            }
            if forwarded.contains("--chat-prompt") {
                addDefault(
                    "--system-prompt",
                    value: descriptor.defaultSystemPrompt,
                    to: &forwarded)
            }
        }
        if !forwarded.contains("--prefill-chunk-tokens") {
            forwarded += ["--prefill-chunk-tokens", "auto"]
            if let descriptor = model.descriptor,
               !forwarded.contains("--prefill-chunk-max") {
                forwarded += ["--prefill-chunk-max",
                              String(descriptor.recommendedPrefillChunkTokens(on: device))]
            }
        }
        let child = try bundledExecutable(
            named: "TUFFCLI", beside: executableURL, fileExists: fileExists)
        return .run(
            executableURL: child,
            arguments: ["--model", model.url.path] + forwarded)
    }

    /// The benchmark runs inside the app executable, which links the same
    /// inference client chat uses.
    private static func benchPlan(
        _ arguments: [String],
        executableURL: URL,
        fileExists: (String) -> Bool
    ) throws -> TUFFCommandPlan {
        let resolved = executableURL.resolvingSymlinksInPath().standardizedFileURL
        let bundle = resolved.deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let bundled = bundle.appendingPathComponent("Contents/MacOS/TUFF", isDirectory: false)
        let app = bundle.pathExtension == "app" && fileExists(bundled.path)
            ? bundled
            : try bundledExecutable(named: "TUFF", beside: resolved, fileExists: fileExists)
        return .run(executableURL: app, arguments: ["--benchmark"] + arguments)
    }

    private static func servePlan(
        _ arguments: [String],
        executableURL: URL,
        currentDirectoryURL: URL,
        applicationSupportURL: URL,
        selectedModel: String?,
        fileExists: (String) -> Bool
    ) throws -> TUFFCommandPlan {
        guard !arguments.contains("--model") else {
            throw TUFFCommandError.serveModelRemoved
        }
        // The router picks each model's settings itself; it only needs to know
        // where installed models live, which differs for clone builds.
        var forwarded = arguments
        if !forwarded.contains("--models-root") {
            let root = installURL(
                descriptor: TUFFModelCatalog.default,
                executableURL: executableURL,
                currentDirectoryURL: currentDirectoryURL,
                applicationSupportURL: applicationSupportURL,
                fileExists: fileExists).deletingLastPathComponent()
            forwarded += ["--models-root", root.path]
        }
        // `default` follows the app, as `tuff prompt` does.
        if !forwarded.contains("--default-model"),
           let selectedModel,
           let descriptor = descriptor(selection: selectedModel) {
            forwarded += ["--default-model", descriptor.selector]
        }
        let child = try bundledExecutable(
            named: "TUFFServer", beside: executableURL, fileExists: fileExists)
        return .run(executableURL: child, arguments: forwarded)
    }

    private static func loadPlan(
        _ arguments: [String],
        executableURL: URL,
        currentDirectoryURL: URL,
        applicationSupportURL: URL,
        environment: [String: String],
        selectedModel: String?,
        fileExists: (String) -> Bool
    ) throws -> TUFFCommandPlan {
        if arguments == ["--help"] || arguments == ["-h"] { return .help }
        let extracted = try extractModel(arguments)
        var selection = extracted.selection
        if let positional = extracted.remaining.first {
            guard !positional.hasPrefix("-"), extracted.remaining.count == 1 else {
                throw TUFFCommandError.unexpectedArgument(positional)
            }
            guard selection == nil else { throw TUFFCommandError.duplicateModel }
            selection = positional
        }
        let model = try resolveModel(
            selection: selection,
            currentDirectoryURL: currentDirectoryURL,
            applicationSupportURL: applicationSupportURL,
            executableURL: executableURL,
            environment: environment,
            selectedModel: selectedModel,
            fileExists: fileExists)
        guard let descriptor = model.descriptor else {
            throw TUFFCommandError.unknownModel(selection ?? model.url.path)
        }
        guard fileExists(model.url.appendingPathComponent("manifest.json").path),
              fileExists(model.url.appendingPathComponent("verified-install.json").path) else {
            throw TUFFCommandError.modelNotInstalled(model.url.path)
        }

        let resolvedExecutable = executableURL.resolvingSymlinksInPath().standardizedFileURL
        let possibleBundle = resolvedExecutable.deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let appExecutable = possibleBundle
            .appendingPathComponent("Contents/MacOS/TUFF", isDirectory: false)
        let launcher: URL
        let launcherArguments: [String]
        if possibleBundle.pathExtension == "app", fileExists(appExecutable.path) {
            launcher = URL(fileURLWithPath: "/usr/bin/open")
            launcherArguments = ["-n", possibleBundle.path, "--args", "--load-model"]
        } else {
            launcher = try bundledExecutable(
                named: "TUFF", beside: resolvedExecutable, fileExists: fileExists)
            launcherArguments = ["--load-model"]
        }
        return .load(
            launcherURL: launcher,
            arguments: launcherArguments,
            selection: descriptor.selector,
            modelURL: model.url)
    }

    private struct ResolvedModel {
        let descriptor: TUFFModelDescriptor?
        let url: URL
    }

    private static func resolveModel(
        selection explicitSelection: String?,
        currentDirectoryURL: URL,
        applicationSupportURL: URL,
        executableURL: URL,
        environment: [String: String],
        selectedModel: String?,
        fileExists: (String) -> Bool
    ) throws -> ResolvedModel {
        let selection = explicitSelection
            ?? environment["TUFF_MODEL"]
            ?? selectedModel
            ?? TUFFModelCatalog.default.selector
        if let descriptor = descriptor(selection: selection) {
            return ResolvedModel(
                descriptor: descriptor,
                url: installURL(
                    descriptor: descriptor,
                    executableURL: executableURL,
                    currentDirectoryURL: currentDirectoryURL,
                    applicationSupportURL: applicationSupportURL,
                    fileExists: fileExists))
        }
        guard selection.contains("/") || selection.hasSuffix(".gturbo")
                || fileExists(selection) else {
            throw TUFFCommandError.unknownModel(selection)
        }
        let pathURL = URL(fileURLWithPath: selection, isDirectory: true)
        let absolute = selection.hasPrefix("/")
            ? pathURL : currentDirectoryURL.appendingPathComponent(selection, isDirectory: true)
        return ResolvedModel(descriptor: nil, url: absolute.standardizedFileURL)
    }

    private static func descriptor(selection: String) -> TUFFModelDescriptor? {
        TUFFModelCatalog.all.first {
            $0.selector == selection
                || $0.aliases.contains(selection)
                || $0.id.rawValue == selection
                || $0.apiModelID == selection
                || $0.installDirectoryName == selection
        }
    }

    private static func installURL(
        descriptor: TUFFModelDescriptor,
        executableURL: URL,
        currentDirectoryURL: URL,
        applicationSupportURL: URL,
        fileExists: (String) -> Bool
    ) -> URL {
        for start in [
            executableURL.resolvingSymlinksInPath().deletingLastPathComponent(),
            currentDirectoryURL,
        ] {
            if let root = packageRoot(startingAt: start, fileExists: fileExists) {
                return root.appendingPathComponent(
                    "scratch/\(descriptor.installDirectoryName)", isDirectory: true)
                    .standardizedFileURL
            }
        }
        return applicationSupportURL
            .appendingPathComponent("TUFF/Models", isDirectory: true)
            .appendingPathComponent(descriptor.installDirectoryName, isDirectory: true)
            .standardizedFileURL
    }

    private static func packageRoot(
        startingAt start: URL,
        fileExists: (String) -> Bool
    ) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            if fileExists(candidate.appendingPathComponent("Package.swift").path),
               fileExists(candidate.appendingPathComponent("Sources/TUFFApp/Mac").path) {
                return candidate
            }
            let parent = (candidatePath as NSString).deletingLastPathComponent
            if parent.isEmpty || parent == candidatePath { return nil }
            candidatePath = parent
        }
    }

    private static func bundledExecutable(
        named name: String,
        beside executableURL: URL,
        fileExists: (String) -> Bool
    ) throws -> URL {
        let candidate = executableURL.resolvingSymlinksInPath()
            .deletingLastPathComponent()
            .appendingPathComponent(name, isDirectory: false)
        guard fileExists(candidate.path) else {
            throw TUFFCommandError.missingBundledExecutable(name)
        }
        return candidate
    }

    private static func extractModel(
        _ arguments: [String]
    ) throws -> (selection: String?, remaining: [String]) {
        var selection: String?
        var remaining: [String] = []
        var index = 0
        while index < arguments.count {
            if arguments[index] == "--model" {
                guard selection == nil else { throw TUFFCommandError.duplicateModel }
                guard index + 1 < arguments.count else {
                    throw TUFFCommandError.missingValue("--model")
                }
                selection = arguments[index + 1]
                index += 2
            } else {
                remaining.append(arguments[index])
                index += 1
            }
        }
        return (selection, remaining)
    }

    private static func containsAny(_ flags: [String], in arguments: [String]) -> Bool {
        flags.contains { arguments.contains($0) }
    }

    private static func addDefault<T>(
        _ flag: String,
        value: T,
        to arguments: inout [String]
    ) {
        if !arguments.contains(flag) {
            arguments += [flag, String(describing: value)]
        }
    }
}
