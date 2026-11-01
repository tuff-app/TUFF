import AppKit
import Foundation
import Observation
import TUFFAppCore

/// Runs `TUFF --benchmark` as a child process and follows its progress. A
/// separate process keeps a run's memory and state apart from chat, and
/// stopping it is just ending the process.
@MainActor
@Observable
final class BenchmarkController {
    enum ModelState: Equatable {
        case waiting
        case running(String)
        case finished(AppBenchmarkResult.Status)
    }

    private(set) var installed: [AppBenchmarkModel] = []
    var selected: Set<String> = []
    var mode: AppBenchmarkMode = .standard
    private(set) var isRunning = false
    private(set) var progress: Double = 0
    var states: [String: ModelState] = [:]
    var result: AppBenchmarkResult?
    private(set) var resultURL: URL?
    private(set) var error: String?
    private(set) var shareNote: String?

    private var process: Process?
    private var order: [String] = []

    func refresh() {
        installed = AppBenchmarkModel.installed()
        selected.formIntersection(installed.map(\.id))
    }

    var canStart: Bool { !isRunning && !selected.isEmpty }

    /// Selected models in catalog order, which is also smallest-ish first.
    var selectedModels: [AppBenchmarkModel] { installed.filter { selected.contains($0.id) } }

    func start(prepare: () async -> Void) async {
        guard canStart, let executable = Bundle.main.executableURL else { return }
        isRunning = true
        error = nil
        shareNote = nil
        result = nil
        resultURL = nil
        progress = 0
        order = selectedModels.map(\.id)
        states = Dictionary(uniqueKeysWithValues: order.map { ($0, ModelState.waiting) })
        await prepare()

        let process = Process()
        process.executableURL = executable
        process.arguments = ["--benchmark", "--models", order.joined(separator: ","),
                             mode == .quick ? "--quick" : "--standard", "--progress-json"]
        let output = Pipe()
        let errors = Pipe()
        process.standardOutput = output
        process.standardError = errors
        let lines = LineBuffer()
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            for line in lines.append(data) {
                Task { @MainActor in self?.handle(line) }
            }
        }
        let stderr = ErrorTail()
        errors.fileHandleForReading.readabilityHandler = { handle in
            stderr.append(handle.availableData)
        }
        process.terminationHandler = { [weak self] finished in
            output.fileHandleForReading.readabilityHandler = nil
            errors.fileHandleForReading.readabilityHandler = nil
            let rest = lines.flush()
            let status = finished.terminationStatus
            let tail = stderr.text
            Task { @MainActor in
                for line in rest { self?.handle(line) }
                self?.finish(status: status, errorTail: tail)
            }
        }
        do {
            try process.run()
            self.process = process
        } catch {
            isRunning = false
            self.error = "Could not start the benchmark: \(error.localizedDescription)"
        }
    }

    /// Ends the run. The runner saves what it measured so far.
    func stop() {
        process?.terminate()
    }

    private func handle(_ line: Data) {
        if let event = try? AppBenchmarkResult.decoder().decode(AppBenchmarkProgress.self, from: line) {
            apply(event)
        } else if let object = try? JSONSerialization.jsonObject(with: line) as? [String: String],
                  object["kind"] == "result", let path = object["path"] {
            resultURL = URL(fileURLWithPath: path)
        }
    }

    private func apply(_ event: AppBenchmarkProgress) {
        let modelShare = 1 / Double(max(event.modelCount, 1))
        let stepShare = event.stepCount.map { Double((event.stepIndex ?? 0) + 1) / Double(max($0, 1)) } ?? 0
        switch event.kind {
        case .coolingDown:
            states[event.modelID] = .running("Letting the Mac cool down")
        case .modelStarted:
            states[event.modelID] = .running("Loading")
        case .loaded:
            states[event.modelID] = .running("Loaded")
        case .stepStarted:
            states[event.modelID] = .running(Self.describe(event))
        case .stepFinished:
            progress = max(progress, (Double(event.modelIndex) + stepShare) * modelShare)
        case .modelFinished:
            states[event.modelID] = .finished(event.status ?? .completed)
            progress = max(progress, Double(event.modelIndex + 1) * modelShare)
        }
    }

    static func describe(_ event: AppBenchmarkProgress) -> String {
        let trial = event.trial.map { " \($0)" } ?? ""
        switch event.workload {
        case .check?: return "Checking the answer"
        case .short?: return "Writing, trial\(trial)"
        case .long?: return "Reading a long prompt, trial\(trial)"
        case .followUp?: return "Follow-up with prompt reuse"
        case nil: return "Running"
        }
    }

    private func finish(status: Int32, errorTail: String) {
        isRunning = false
        process = nil
        for id in order where states[id] == .waiting || isActive(states[id]) {
            states[id] = .finished(.cancelled)
        }
        guard let resultURL,
              let data = try? Data(contentsOf: resultURL),
              let result = try? AppBenchmarkResult.decoder().decode(AppBenchmarkResult.self, from: data)
        else {
            error = errorTail.isEmpty
                ? "The benchmark stopped before it saved a result."
                : errorTail
            return
        }
        self.result = result
        progress = 1
    }

    private func isActive(_ state: ModelState?) -> Bool {
        if case .running = state { return true }
        return false
    }

    // MARK: - Sharing

    func share() {
        guard let result else { return }
        do {
            let title = AppBenchmarkShare.title(for: result)
            let body = try AppBenchmarkShare.body(for: result)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(body, forType: .string)
            let url = AppBenchmarkShare.discussionURL(title: title, body: body)
            NSWorkspace.shared.open(url)
            shareNote = url.absoluteString.contains("body=")
                ? "Opened GitHub with your result filled in. Check it and press Start discussion."
                : "Opened GitHub and copied your result. Paste it into the body, then press Start discussion."
        } catch {
            shareNote = "Could not prepare the post: \(error.localizedDescription)"
        }
    }

    func revealResult() {
        guard let resultURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([resultURL])
    }
}

/// Splits a byte stream into lines across reads.
private final class LineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()

    func append(_ data: Data) -> [Data] {
        lock.withLock {
            pending.append(data)
            var lines: [Data] = []
            while let newline = pending.firstIndex(of: 0x0A) {
                lines.append(pending[pending.startIndex..<newline])
                pending.removeSubrange(pending.startIndex...newline)
            }
            return lines
        }
    }

    func flush() -> [Data] {
        lock.withLock {
            defer { pending.removeAll() }
            return pending.isEmpty ? [] : [pending]
        }
    }
}

/// The last few lines the runner wrote to stderr, for an error message.
private final class ErrorTail: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()

    func append(_ chunk: Data) {
        lock.withLock {
            data.append(chunk)
            if data.count > 4_096 { data = data.suffix(4_096) }
        }
    }

    var text: String {
        lock.withLock {
            String(decoding: data, as: UTF8.self)
                .split(separator: "\n").suffix(3).joined(separator: "\n")
        }
    }
}
