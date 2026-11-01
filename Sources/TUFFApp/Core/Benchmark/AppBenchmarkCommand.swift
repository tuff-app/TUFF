import Darwin
import Foundation
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFModelCatalog

/// `TUFF --benchmark ...`: what `tuff bench` runs, and what the app's
/// Benchmarks screen starts as a child process so a run never shares memory
/// or state with chat.
public enum AppBenchmarkCommand {
    public static let flag = "--benchmark"

    public static let usage = """
    usage: tuff bench (--models <name,...> | --all) [--quick] [--share] [--output <file>]
           tuff bench --list

      --models <names>  Models to run, comma separated (for example gemma4,qwen36).
      --all             Every installed model, smallest first.
      --quick           One trial of each workload instead of three.
      --share           Copy the result post and open a new GitHub Discussion for it.
      --output <file>   Also write the result JSON here.
      --list            Show installed models and exit.

    Runs TUFF's standard benchmark on installed models, one at a time, with the
    same settings chat uses. Results are saved in TUFF's Benchmarks folder.
    Nothing is uploaded unless you post it yourself.
    """

    struct Options: Equatable {
        var models: [String] = []
        var all = false
        var mode: AppBenchmarkMode = .standard
        var share = false
        var list = false
        var output: String?
        var progressJSON = false
    }

    enum UsageError: Error, CustomStringConvertible, Equatable {
        case missingValue(String)
        case unknownArgument(String)
        case noModels
        case unknownModel(String)
        case notInstalled(String)

        var description: String {
            switch self {
            case .missingValue(let flag): "missing value for \(flag)"
            case .unknownArgument(let argument): "unknown argument: \(argument)"
            case .noModels: "choose models with --models or use --all"
            case .unknownModel(let name): "unknown model: \(name)"
            case .notInstalled(let name): "\(name) is not installed"
            }
        }
    }

    /// Runs the command if `arguments` (without the executable) asks for it.
    public static func runIfRequested(arguments: [String]) -> Int32? {
        guard arguments.first == flag else { return nil }
        return drive(Array(arguments.dropFirst()))
    }

    static func parse(_ arguments: [String]) throws -> Options {
        var options = Options()
        var index = 0
        func value(_ flag: String) throws -> String {
            guard index + 1 < arguments.count else { throw UsageError.missingValue(flag) }
            index += 1
            return arguments[index]
        }
        while index < arguments.count {
            switch arguments[index] {
            case "--models":
                options.models += try value("--models").split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
            case "--all": options.all = true
            case "--quick": options.mode = .quick
            case "--standard": options.mode = .standard
            case "--share": options.share = true
            case "--list": options.list = true
            case "--output": options.output = try value("--output")
            case "--progress-json": options.progressJSON = true
            case let other: throw UsageError.unknownArgument(other)
            }
            index += 1
        }
        if !options.list, !options.all, options.models.isEmpty { throw UsageError.noModels }
        return options
    }

    /// The requested models, in the order given, or every installed model
    /// smallest first for `--all`.
    static func select(_ options: Options, installed: [AppBenchmarkModel]) throws -> [AppBenchmarkModel] {
        if options.all {
            return installed.sorted { $0.descriptor.source.installedBytes < $1.descriptor.source.installedBytes }
        }
        return try options.models.map { name in
            guard let descriptor = TUFFModelCatalog.all.first(where: {
                $0.selector == name || $0.aliases.contains(name)
                    || $0.id.rawValue == name || $0.apiModelID == name
            }) else { throw UsageError.unknownModel(name) }
            guard let model = installed.first(where: { $0.descriptor.id == descriptor.id })
            else { throw UsageError.notInstalled(descriptor.displayName) }
            return model
        }
    }

    /// Where results are kept: `Benchmarks` beside `Models`.
    public static func resultsDirectory() -> URL {
        let models = AppModelLocation.defaultURL(descriptor: .default).deletingLastPathComponent()
        let base = models.lastPathComponent == "Models" ? models.deletingLastPathComponent() : models
        return base.appendingPathComponent("Benchmarks", isDirectory: true)
    }

    public static func save(_ result: AppBenchmarkResult, to directory: URL = resultsDirectory()) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"
        let url = directory.appendingPathComponent(
            "tuff-benchmark-\(formatter.string(from: result.startedAt)).json")
        try AppBenchmarkResult.encoder(pretty: true).encode(result).write(to: url, options: .atomic)
        return url
    }

    /// Copies the post and opens the new-discussion page.
    public static func share(_ result: AppBenchmarkResult) throws -> (url: URL, prefilled: Bool) {
        let title = AppBenchmarkShare.title(for: result)
        let body = try AppBenchmarkShare.body(for: result)
        copyToPasteboard(body)
        let url = AppBenchmarkShare.discussionURL(title: title, body: body)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = [url.absoluteString]
        try process.run()
        process.waitUntilExit()
        return (url, url.absoluteString.contains("body="))
    }

    static func copyToPasteboard(_ text: String) {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pbcopy")
        process.standardInput = pipe
        guard (try? process.run()) != nil else { return }
        pipe.fileHandleForWriting.write(Data(text.utf8))
        try? pipe.fileHandleForWriting.close()
        process.waitUntilExit()
    }

    // MARK: - Running

    private final class Box: @unchecked Sendable {
        var code: Int32 = 0
        var task: Task<Void, Never>?
    }

    private static func drive(_ arguments: [String]) -> Int32 {
        if arguments == ["--help"] || arguments == ["-h"] {
            print(usage)
            return 0
        }
        let options: Options
        do {
            options = try parse(arguments)
        } catch {
            writeError("error: \(error)\n\n\(usage)")
            return 2
        }
        let box = Box()
        let done = DispatchSemaphore(value: 0)
        box.task = Task(priority: .userInitiated) {
            box.code = await run(options)
            done.signal()
        }
        signal(SIGINT, SIG_IGN)
        signal(SIGTERM, SIG_IGN)
        let sources = [SIGINT, SIGTERM].map { number in
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
            source.setEventHandler { box.task?.cancel() }
            source.resume()
            return source
        }
        done.wait()
        _ = sources
        return box.code
    }

    private static func run(_ options: Options) async -> Int32 {
        let installed = AppBenchmarkModel.installed()
        if options.list {
            if installed.isEmpty { print("No models are installed. Install one in TUFF > Models.") }
            for model in installed {
                print("\(model.descriptor.selector)\t\(model.descriptor.displayName)")
            }
            return 0
        }
        let models: [AppBenchmarkModel]
        do {
            models = try select(options, installed: installed)
        } catch {
            writeError("error: \(error)")
            return 2
        }
        guard !models.isEmpty else {
            writeError("No models are installed. Install one in TUFF > Models.")
            return 2
        }
        let emitJSON = options.progressJSON
        if !emitJSON {
            writeError("Benchmarking \(models.count) model\(models.count == 1 ? "" : "s") "
                + "(\(options.mode.rawValue)). Large models can take a long time. Ctrl-C stops.")
        }
        let runner = AppBenchmarkRunner(
            client: DecodeServiceInferenceClient())
        let result = await runner.run(models: models, mode: options.mode) { event in
            if emitJSON {
                emit(event)
            } else {
                writeError(describe(event))
            }
        }
        do {
            let saved = try save(result)
            if let output = options.output {
                try AppBenchmarkResult.encoder(pretty: true).encode(result)
                    .write(to: URL(fileURLWithPath: output), options: .atomic)
            }
            if emitJSON {
                emitLine(["kind": "result", "path": saved.path])
            } else {
                print(AppBenchmarkShare.table(for: result))
                print("\nSaved \(saved.path)")
            }
            if options.share, !result.completedRuns.isEmpty {
                let shared = try share(result)
                if !emitJSON {
                    print(shared.prefilled
                        ? "Opened a new Benchmarks discussion with the result filled in."
                        : "Opened a new Benchmarks discussion. The post is on your clipboard: paste it into the body.")
                }
            }
        } catch {
            writeError("error: \(error)")
            return 1
        }
        if Task.isCancelled { return 130 }
        return result.completedRuns.count == models.count ? 0 : 1
    }

    static func describe(_ event: AppBenchmarkProgress) -> String {
        let position = "[\(event.modelIndex + 1)/\(event.modelCount)] \(event.modelName)"
        switch event.kind {
        case .coolingDown: return "\(position): letting the Mac cool down"
        case .modelStarted: return "\(position): loading"
        case .loaded: return "\(position): loaded"
        case .stepStarted:
            return "\(position): \(event.workload?.rawValue ?? "step") \(event.trial ?? 1)"
        case .stepFinished: return "\(position): \(event.workload?.rawValue ?? "step") done"
        case .modelFinished:
            return "\(position): \(event.status?.rawValue ?? "finished")"
                + (event.message.map { ", \($0)" } ?? "")
        }
    }

    private static func emit(_ event: AppBenchmarkProgress) {
        guard let data = try? AppBenchmarkResult.encoder().encode(event) else { return }
        FileHandle.standardOutput.write(data + Data("\n".utf8))
    }

    private static func emitLine(_ object: [String: String]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        else { return }
        FileHandle.standardOutput.write(data + Data("\n".utf8))
    }

    private static func writeError(_ text: String) {
        FileHandle.standardError.write(Data((text + "\n").utf8))
    }
}
