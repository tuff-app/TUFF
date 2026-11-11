import Foundation
import Testing
import TUFFModelCatalog
@testable import TUFFCommandCore

@Suite("Unified tuff command")
struct TUFFCommandTests {
    @Test func explicitMemoryControlsRemainManual() throws {
        let plan = try TUFFCommand.plan(arguments: ["prompt", "hello", "--model", "ornith-397b",
                "--expert-cache-slots", "10", "--dense-weight-budget-mb", "2500"],
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/TUFFCommand"),
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { $0 == "/repo/.build/debug/TUFFCLI" })
        guard case .run(_, let arguments) = plan else { Issue.record("expected prompt command"); return }
        #expect(!arguments.contains("--memory-auto"))
        #expect(option("--expert-cache-slots", in: arguments) == "10")
        #expect(option("--dense-weight-budget-mb", in: arguments) == "2500")
    }
    private let repository = URL(fileURLWithPath: "/repo", isDirectory: true)
    private let appSupport = URL(
        fileURLWithPath: "/Users/test/Library/Application Support", isDirectory: true)

    @Test func promptUsesSelectedCatalogModelAndCatalogDefaults() throws {
        let executable = repository.appendingPathComponent(".build/debug/TUFFCommand")
        let existing: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/TUFFApp/Mac",
            "/repo/.build/debug/TUFFCLI",
        ]
        let plan = try TUFFCommand.plan(
            arguments: ["prompt", "What is 2 + 2?"],
            executableURL: executable,
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            selectedModel: "minimax-m2.7",
            device: Self.mac(memoryGiB: 16),
            fileExists: existing.contains)

        guard case .run(let child, let arguments) = plan else {
            Issue.record("expected a child process")
            return
        }
        #expect(child.path == "/repo/.build/debug/TUFFCLI")
        #expect(arguments.prefix(4) == [
            "--model", "/repo/scratch/minimax-m2.7.gturbo",
            "--chat-prompt", "What is 2 + 2?",
        ])
        #expect(option("--max-context", in: arguments) == "4096")
        #expect(option("--expert-cache-slots", in: arguments) == "16")
        #expect(arguments.contains("--memory-auto"))
        #expect(option("--thinking", in: arguments) == "on")
        #expect(option("--system-prompt", in: arguments)
            == TUFFModelCatalog.minimaxM27.defaultSystemPrompt)
        #expect(option("--prefill-chunk-tokens", in: arguments) == "auto")
        // MiniMax (128.7 GB) cannot stay in a 16 GB Mac's page cache.
        #expect(option("--prefill-chunk-max", in: arguments) == "2048")
    }

    private static func mac(memoryGiB: UInt64) -> TUFFDeviceCapabilities {
        TUFFDeviceCapabilities(unifiedMemoryBytes: memoryGiB * TUFFModelCatalog.oneGiB,
                               macOSMajorVersion: 26,
                               appleSiliconGeneration: 2)
    }

    @Test func promptSizesPrefillChunksForTheModelAndMac() throws {
        let executable = repository.appendingPathComponent(".build/debug/TUFFCommand")
        let existing: Set<String> = [
            "/repo/Package.swift", "/repo/Sources/TUFFApp/Mac", "/repo/.build/debug/TUFFCLI",
        ]
        guard case .run(_, let promptArguments) = try TUFFCommand.plan(
            arguments: ["prompt", "--model", "gemma4-e4b", "hi"],
            executableURL: executable,
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            device: Self.mac(memoryGiB: 16),
            fileExists: existing.contains) else {
            Issue.record("expected a child process")
            return
        }
        #expect(option("--prefill-chunk-max", in: promptArguments) == "256")
    }

    @Test func promptAcceptsAnExplicitModelPathAndRawCLIOptions() throws {
        let executable = repository.appendingPathComponent(".build/debug/TUFFCommand")
        let existing: Set<String> = ["/repo/.build/debug/TUFFCLI"]
        let plan = try TUFFCommand.plan(
            arguments: [
                "prompt", "--model", "/models/custom.gturbo",
                "--prompt", "raw", "--temperature", "0",
            ],
            executableURL: executable,
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            fileExists: existing.contains)
        guard case .run(_, let arguments) = plan else {
            Issue.record("expected a child process")
            return
        }
        #expect(arguments.prefix(4) == [
            "--model", "/models/custom.gturbo", "--prompt", "raw",
        ])
        #expect(option("--temperature", in: arguments) == "0")
    }

    @Test func loadSelectsInstalledModelAndLaunchesTheContainingApp() throws {
        let executable = URL(fileURLWithPath:
            "/Applications/TUFF.app/Contents/Resources/bin/tuff")
        let model = "/Users/test/Library/Application Support/TUFF/Models/minimax-m2.7.gturbo"
        let existing: Set<String> = [
            model + "/manifest.json",
            model + "/verified-install.json",
            "/Applications/TUFF.app/Contents/MacOS/TUFF",
        ]
        let plan = try TUFFCommand.plan(
            arguments: ["load", "minimax-m2.7"],
            executableURL: executable,
            currentDirectoryURL: URL(fileURLWithPath: "/tmp", isDirectory: true),
            applicationSupportURL: appSupport,
            fileExists: existing.contains)
        #expect(plan == .load(
            launcherURL: URL(fileURLWithPath: "/usr/bin/open"),
            arguments: [
                "-n", "/Applications/TUFF.app", "--args", "--load-model",
            ],
            selection: "minimax-m2.7",
            modelURL: URL(fileURLWithPath: model, isDirectory: true)))
    }

    @Test func loadRefusesASelectionThatIsNotInstalled() {
        #expect(throws: TUFFCommandError.modelNotInstalled(
            "/Users/test/Library/Application Support/TUFF/Models/minimax-m2.7.gturbo")) {
            _ = try TUFFCommand.plan(
                arguments: ["load", "minimax"],
                executableURL: URL(fileURLWithPath:
                    "/Applications/TUFF.app/Contents/Resources/bin/tuff"),
                currentDirectoryURL: URL(fileURLWithPath: "/tmp", isDirectory: true),
                applicationSupportURL: appSupport,
                fileExists: { _ in false })
        }
    }

    @Test func helpAndInvalidCommandsAreExplicit() throws {
        #expect(try TUFFCommand.plan(
            arguments: [],
            executableURL: repository.appendingPathComponent("tuff"),
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            fileExists: { _ in false }) == .help)
        #expect(throws: TUFFCommandError.unknownCommand("pull")) {
            _ = try TUFFCommand.plan(
                arguments: ["pull"],
                executableURL: repository.appendingPathComponent("tuff"),
                currentDirectoryURL: repository,
                applicationSupportURL: appSupport,
                fileExists: { _ in false })
        }
    }

    @Test func versionPrintsTheSharedReleaseVersion() throws {
        for flag in ["--version", "version"] {
            #expect(try TUFFCommand.plan(
                arguments: [flag],
                executableURL: repository.appendingPathComponent("tuff"),
                currentDirectoryURL: repository,
                applicationSupportURL: appSupport,
                fileExists: { _ in false }) == .version)
        }
        let parts = TUFFVersion.current.split(separator: ".")
        #expect(parts.count == 3 && parts.allSatisfy { Int($0) != nil })
    }

    @Test func benchRunsTheAppExecutable() throws {
        // Packaged: `tuff` lives in Resources/bin, the runner in MacOS/TUFF.
        let packaged = try TUFFCommand.plan(
            arguments: ["bench", "--models", "gemma4", "--quick"],
            executableURL: URL(fileURLWithPath: "/Applications/TUFF.app/Contents/Resources/bin/tuff"),
            currentDirectoryURL: URL(fileURLWithPath: "/tmp", isDirectory: true),
            applicationSupportURL: appSupport,
            fileExists: { $0 == "/Applications/TUFF.app/Contents/MacOS/TUFF" })
        #expect(packaged == .run(
            executableURL: URL(fileURLWithPath: "/Applications/TUFF.app/Contents/MacOS/TUFF"),
            arguments: ["--benchmark", "--models", "gemma4", "--quick"]))

        // A clone build runs the TUFF executable beside TUFFCommand.
        let clone = try TUFFCommand.plan(
            arguments: ["bench", "--list"],
            executableURL: repository.appendingPathComponent(".build/debug/TUFFCommand"),
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            fileExists: { $0 == "/repo/.build/debug/TUFF" })
        #expect(clone == .run(
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/TUFF"),
            arguments: ["--benchmark", "--list"]))
    }

    @Test func serveRoutesEveryInstalledModel() throws {
        // A packaged app serves the models Application Support holds, and
        // `default` follows the model selected in the app.
        let packaged = URL(fileURLWithPath: "/Applications/TUFF.app/Contents/Resources/bin/tuff")
        let packagedPlan = try TUFFCommand.plan(
            arguments: ["serve", "--unload-after", "60"],
            executableURL: packaged,
            currentDirectoryURL: URL(fileURLWithPath: "/tmp", isDirectory: true),
            applicationSupportURL: appSupport,
            selectedModel: "gemma-4-e2b-it",
            fileExists: { $0 == "/Applications/TUFF.app/Contents/Resources/bin/TUFFServer" })
        guard case .run(let child, let arguments) = packagedPlan else {
            Issue.record("expected the server")
            return
        }
        #expect(child.lastPathComponent == "TUFFServer")
        #expect(!arguments.contains("--model"))
        #expect(option("--unload-after", in: arguments) == "60")
        #expect(option("--default-model", in: arguments) == "gemma4-e2b")
        #expect(option("--models-root", in: arguments)
            == "/Users/test/Library/Application Support/TUFF/Models")

        // A clone build serves the repository's scratch installs, as `prompt`
        // does, and an explicit default wins over the app's selection.
        let clonePlan = try TUFFCommand.plan(
            arguments: ["serve", "--default-model", "qwen36"],
            executableURL: repository.appendingPathComponent(".build/debug/TUFFCommand"),
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            selectedModel: "gemma4-e2b",
            fileExists: [
                "/repo/Package.swift", "/repo/Sources/TUFFApp/Mac",
                "/repo/.build/debug/TUFFServer",
            ].contains)
        guard case .run(_, let cloneArguments) = clonePlan else {
            Issue.record("expected the server")
            return
        }
        #expect(option("--models-root", in: cloneArguments) == "/repo/scratch")
        #expect(cloneArguments.filter { $0 == "--default-model" }.count == 1)
        #expect(option("--default-model", in: cloneArguments) == "qwen36")

        // 7.0.0's spelling still works; the server ignores it.
        guard case .run(_, let legacy) = try TUFFCommand.plan(
            arguments: ["serve", "--all-models"],
            executableURL: packaged,
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            fileExists: { _ in true }) else {
            Issue.record("expected the server")
            return
        }
        #expect(legacy.first == "--all-models")
        #expect(option("--default-model", in: legacy) == nil)
    }

    @Test func serveExplainsThatFixedModelServingWasRemoved() {
        for arguments in [["serve", "--model", "gemma4"], ["serve", "--all-models", "--model", "gemma4"]] {
            #expect(throws: TUFFCommandError.serveModelRemoved) {
                _ = try TUFFCommand.plan(
                    arguments: arguments,
                    executableURL: URL(fileURLWithPath: "/Applications/TUFF.app/Contents/Resources/bin/tuff"),
                    currentDirectoryURL: repository,
                    applicationSupportURL: appSupport,
                    fileExists: { _ in true })
            }
        }
        #expect(TUFFCommandError.serveModelRemoved.description.contains("--default-model"))
    }

    private func option(_ flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else {
            return nil
        }
        return arguments[index + 1]
    }

    @Test func helpNamesEveryCatalogModel() {
        let help = TUFFCommand.usage.replacingOccurrences(of: "\n", with: " ")
        for model in TUFFModelCatalog.all {
            #expect(help.contains(model.selector), "\(model.selector) missing from tuff --help")
        }
        #expect(TUFFCommand.usage.split(separator: "\n").allSatisfy { $0.count <= 80 })
    }
}

extension TUFFCommandTests {
    @Test func applePromptBypassesModelFilesAndMemoryPlanner() throws {
        for selection in ["apple", "apple.on-device"] {
            let plan = try TUFFCommand.plan(arguments: ["prompt", "hello", "--model", selection],
                executableURL: URL(fileURLWithPath: "/repo/.build/debug/TUFFCommand"),
                currentDirectoryURL: URL(fileURLWithPath: "/repo"),
                applicationSupportURL: URL(fileURLWithPath: "/support"),
                fileExists: { $0 == "/repo/.build/debug/TUFFCLI" })
            guard case .run(_, let arguments) = plan else { Issue.record("expected CLI"); return }
            #expect(arguments == ["--model", "apple.on-device", "--chat-prompt", "hello"])
        }
    }
}

extension TUFFCommandTests {
    @Test func loadAppleSelectsProviderWithoutRequiringWeightFiles() throws {
        let plan = try TUFFCommand.plan(arguments: ["load", "apple"],
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/TUFFCommand"),
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { $0 == "/repo/.build/debug/TUFF" })
        guard case .load(_, _, let selection, _) = plan else { Issue.record("expected app selection"); return }
        #expect(selection == "apple.on-device")
    }
}
