import Foundation
import TUFFCommandCore
import TUFFModelCatalog

private func writeError(_ text: String) {
    FileHandle.standardError.write(Data((text + "\n").utf8))
}

private func run(_ executableURL: URL, arguments: [String]) throws -> Int32 {
    let process = Process()
    process.executableURL = executableURL
    process.arguments = arguments
    process.standardInput = FileHandle.standardInput
    process.standardOutput = FileHandle.standardOutput
    process.standardError = FileHandle.standardError
    // Passed on rather than left to kill only this wrapper: `kill <pid>` on
    // `tuff serve` used to exit here and leave the server holding its port.
    // Ctrl-C reaches both already; the server ignores a second signal.
    let forwarded = [SIGINT, SIGTERM, SIGHUP].map { signalNumber in
        signal(signalNumber, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .global())
        source.setEventHandler {
            if process.isRunning { kill(process.processIdentifier, signalNumber) }
        }
        source.resume()
        return source
    }
    defer { forwarded.forEach { $0.cancel() } }
    try process.run()
    process.waitUntilExit()
    return process.terminationStatus
}

let fileManager = FileManager.default
let currentDirectoryURL = URL(
    fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
let applicationSupportURL = TUFFApplicationDirectories.applicationSupportURL()
// A shell that finds `tuff` on the PATH passes the bare name as argv[0], so
// argv[0] can't locate the bundled executables. The path the kernel ran can.
let executableURL = Bundle.main.executableURL
    ?? (CommandLine.arguments[0].hasPrefix("/")
        ? URL(fileURLWithPath: CommandLine.arguments[0])
        : currentDirectoryURL.appendingPathComponent(CommandLine.arguments[0]))
let preferences = UserDefaults(suiteName: TUFFApplicationDirectories.preferencesSuiteName())

do {
    let plan = try TUFFCommand.plan(
        arguments: Array(CommandLine.arguments.dropFirst()),
        executableURL: executableURL,
        currentDirectoryURL: currentDirectoryURL,
        applicationSupportURL: applicationSupportURL,
        environment: ProcessInfo.processInfo.environment,
        selectedModel: preferences?.string(forKey: "model"))
    switch plan {
    case .help:
        print(TUFFCommand.usage)
        exit(0)
    case .version:
        print("tuff \(TUFFVersion.current)")
        exit(0)
    case .run(let child, let arguments):
        exit(try run(child, arguments: arguments))
    case .load(let launcher, let arguments, let selection, let modelURL):
        preferences?.set(selection, forKey: "model")
        let status = try run(launcher, arguments: arguments)
        if status == 0 {
            if modelURL.scheme == "apple" { print("Selected Apple Foundation Model") }
            else { print("Loading \(selection) from \(modelURL.path)") }
        }
        exit(status)
    }
} catch {
    writeError("error: \(error)")
    if case TUFFCommandError.unknownCommand = error {
        writeError("")
        writeError(TUFFCommand.usage)
    } else {
        writeError("Run `tuff --help` for the commands.")
    }
    exit(2)
}
