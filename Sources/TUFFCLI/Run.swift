import Foundation
import Synchronization
import TUFFCLIArguments
import TUFFDecodeProtocol
import TUFFRuntimeManagement
import TUFFAppCore
import TUFFModelCatalog
import TUFFModelMetadata

/// The host registers a process-backed runner; engine executables register the
/// native implementation. Argument parsing and command dispatch stay small.
public enum CLIRuntimeRegistry {
    public typealias Runner = @Sendable (Args, FileHandle, FileHandle) async -> RunResult
    private static let runner = Mutex<Runner?>(nil)

    public static func register(_ value: @escaping Runner) { runner.withLock { $0 = value } }
    static func current() -> Runner? { runner.withLock { $0 } }
}

public func run(args: Args, stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    if ["apple", AppleModelKind.onDevice.rawValue].contains(args.model) {
        return await runApple(args: args, stdout: stdout, stderr: stderr)
    }
    if let runner = CLIRuntimeRegistry.current() { return await runner(args, stdout, stderr) }
    do {
        var args = args
        var learning: CLIAutoLearning?
        if args.automaticMemory == true {
            let directory = URL(fileURLWithPath: args.model)
            // The model directory puts this run on the same records the app
            // keeps for this model, volume and engine, so the CLI uses what
            // the app has learned and adds to it.
            guard let summary = AppAutomaticMemoryPlanner.layoutSummary(at: directory) else {
                throw CocoaError(.fileReadNoSuchFile, userInfo: [NSLocalizedDescriptionKey:
                    "No complete model at \(directory.path)."])
            }
            guard let catalog = TUFFModelCatalog.model(manifestModelID: summary.modelID)
                    ?? TUFFModelCatalog.all.first(where: { $0.architecture.id.rawValue == summary.architectureID }),
                  let plan = AppAutomaticMemoryPlanner.plan(for: AppModelInstallDescriptor(catalog: catalog),
                    on: .current(), profile: .speed, layoutSummary: summary,
                    workloadKey: AppAutoPerformanceRecords.workloadKey(
                        promptTokens: 0, reusedPrefix: false, imageInput: !args.images.isEmpty,
                        sampled: args.temperature > 0),
                    modelDirectory: directory, contextTokens: args.maxContext,
                    imageInput: !args.images.isEmpty) else {
                throw CocoaError(.fileReadInvalidFileName, userInfo: [NSLocalizedDescriptionKey:
                    "No safe automatic allocation at this context. Lower --max-context."])
            }
            args.expertCacheSlots = plan.expertCacheSlots
            args.prefillPolicy = plan.expertCacheSlots >= 16 ? .chunked : .off
            args.prefillChunkTokens = plan.prefillChunkTokens
            args.prefillChunkTokensAuto = false
            args.denseResidencyPolicy = plan.denseResidencyPolicy
            args.automaticMemory = nil
            if let layers = AppAutoPerformanceRecords.residentDenseLayers(
                plan.denseResidencyPolicy, summary: summary) {
                let report = FileManager.default.temporaryDirectory
                    .appendingPathComponent("tuff-run-\(UUID().uuidString).json")
                args.runReportPath = report.path
                learning = CLIAutoLearning(
                    summary: summary, directory: directory, report: report,
                    candidateID: AppAutoPerformanceRecords.candidateID(
                        contextTokens: plan.contextTokens, expertCacheSlots: plan.expertCacheSlots,
                        residentDenseLayers: layers, prefillChunkTokens: plan.prefillChunkTokens),
                    imageInput: !args.images.isEmpty, sampled: args.temperature > 0)
            }
        }
        defer { learning.map { try? FileManager.default.removeItem(at: $0.report) } }
        let installation = try await InferenceEngineEnvironment.acquireModel(at: URL(fileURLWithPath: args.model))
        let window = AppAutoObservationWindow.begin()
        let child = CLIEngineProcess()
        do {
            let result = try await withTaskCancellationHandler {
                try Task.checkCancellation()
                try child.start(installation: installation, args: args, stdout: stdout, stderr: stderr)
                return await Task.detached { child.wait() }.value
            } onCancel: { child.cancel() }
            if result.exitCode == 0 {
                await InferenceEngineEnvironment.markKnownGood(installation, for: URL(fileURLWithPath: args.model))
                learning?.record(window: window, engineIdentity: installation.lease.manifest.packageID
                    + ":" + installation.lease.manifest.version.directoryName + ":" + installation.identity)
            }
            await installation.release()
            return result
        } catch {
            await installation.release()
            throw error
        }
    } catch {
        try? stderr.write(contentsOf: Data("error: \(error)\n".utf8))
        return RunResult(exitCode: 1)
    }
}

/// What the host needs to add an Auto-planned run to the shared history.
private struct CLIAutoLearning {
    let summary: ModelWeightLayoutSummary
    let directory: URL
    let report: URL
    let candidateID: String
    let imageInput: Bool
    let sampled: Bool

    func record(window: AppAutoObservationWindow, engineIdentity: String) {
        guard let data = try? Data(contentsOf: report),
              let run = try? JSONDecoder().decode(CLIRunReport.self, from: data) else { return }
        AppAutoPerformanceRecords.record(
            identity: AppAutoPerformanceRecords.identity(
                summary: summary, device: .current(), engineIdentity: engineIdentity,
                modelDirectory: directory),
            observation: window.observation(
                candidateID: candidateID,
                workloadKey: AppAutoPerformanceRecords.workloadKey(
                    promptTokens: run.prefillTokens, reusedPrefix: false,
                    imageInput: imageInput, sampled: sampled),
                decodeTokens: run.newTokens, decodeSeconds: run.decodeSeconds,
                timeToFirstTokenSeconds: run.prefillSeconds))
    }
}

private final class CLIEngineProcess: @unchecked Sendable {
    private let state = Mutex((process: Optional<Process>.none, cancelled: false))
    func start(installation: InferenceEngineInstallation, args: Args,
               stdout: FileHandle, stderr: FileHandle) throws {
        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
        let pipe = Pipe(), process = Process()
        process.executableURL = installation.executableURL
        process.arguments = ["--cli-json", "--engine-lock", installation.lockFileURL.path]
        process.standardInput = pipe
        process.standardOutput = stdout
        process.standardError = stderr
        try state.withLock { value in
            guard !value.cancelled else { throw CancellationError() }
            try process.run()
            value.process = process
        }
        do { try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(args)) }
        catch { process.terminate(); process.waitUntilExit(); throw error }
        try pipe.fileHandleForWriting.close()
    }
    func wait() -> RunResult {
        let process = state.withLock { $0.process! }
        process.waitUntilExit()
        return RunResult(exitCode: process.terminationStatus)
    }
    func cancel() {
        state.withLock { value in
            value.cancelled = true
            if let process = value.process, process.isRunning { process.interrupt() }
        }
    }
}
