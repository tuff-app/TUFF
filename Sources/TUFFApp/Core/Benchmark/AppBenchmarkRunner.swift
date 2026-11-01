import Foundation
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFModelCatalog

/// An installed model the benchmark can run.
public struct AppBenchmarkModel: Equatable, Sendable, Identifiable {
    public let descriptor: TUFFModelDescriptor
    public let directory: URL

    public var id: String { descriptor.id.rawValue }

    public init(descriptor: TUFFModelDescriptor, directory: URL) {
        self.descriptor = descriptor
        self.directory = directory
    }

    /// Every catalog model with a complete install, in catalog order.
    public static func installed() -> [AppBenchmarkModel] {
        installed { AppModelLocation.defaultURL(descriptor: $0) }
    }

    static func installed(locate: (AppModelInstallDescriptor) -> URL) -> [AppBenchmarkModel] {
        TUFFModelCatalog.all.compactMap { descriptor in
            let install = AppModelInstallDescriptor(catalog: descriptor)
            let directory = locate(install)
            guard AppModelInstallationProbe.status(at: directory, descriptor: install) == .complete
            else { return nil }
            return AppBenchmarkModel(descriptor: descriptor, directory: directory)
        }
    }
}

/// What the runner is doing, for progress displays and the CLI's
/// `--progress-json` stream.
public struct AppBenchmarkProgress: Codable, Equatable, Sendable {
    public enum Kind: String, Codable, Sendable {
        case coolingDown = "cooling-down"
        case modelStarted = "model-started"
        case loaded
        case stepStarted = "step-started"
        case stepFinished = "step-finished"
        case modelFinished = "model-finished"
    }

    public var kind: Kind
    public var modelIndex: Int
    public var modelCount: Int
    public var modelID: String
    public var modelName: String
    public var stepIndex: Int?
    public var stepCount: Int?
    public var workload: AppBenchmarkWorkload?
    public var trial: Int?
    public var status: AppBenchmarkResult.Status?
    public var message: String?

    // Spelled so the snake_case strategies round-trip `modelID` as
    // `model_id`. The default spelling decoded back as `modelId`, so the
    // Benchmarks screen dropped every event and sat at 0% for a whole run.
    private enum CodingKeys: String, CodingKey {
        case kind, modelIndex, modelCount
        case modelID = "modelId"
        case modelName, stepIndex, stepCount, workload, trial, status, message
    }
}

/// Runs the suite through the same inference client chat uses, one model at
/// a time. A model that fails is recorded and the next one starts.
public final class AppBenchmarkRunner: @unchecked Sendable {
    private let client: any AppModelLifecycleClient
    private let device: TUFFDeviceCapabilities
    private let machine: AppBenchmarkMachine
    private let appBuild: String
    private let sensors: AppBenchmarkSensors
    private let cooldown: AppBenchmarkCooldown
    private let sleep: @Sendable (Double) async throws -> Void
    private let now: @Sendable () -> Date
    private let clock = ContinuousClock()

    public init(client: any AppModelLifecycleClient,
                device: TUFFDeviceCapabilities = .current(),
                machine: AppBenchmarkMachine = .current(),
                appBuild: String = AppBenchmarkRunner.currentBuild(),
                sensors: AppBenchmarkSensors = .current,
                cooldown: AppBenchmarkCooldown = .standard,
                sleep: @escaping @Sendable (Double) async throws -> Void = {
                    try await Task.sleep(for: .seconds($0))
                },
                now: @escaping @Sendable () -> Date = Date.init) {
        self.client = client
        self.device = device
        self.machine = machine
        self.appBuild = appBuild
        self.sensors = sensors
        self.cooldown = cooldown
        self.sleep = sleep
        self.now = now
    }

    /// "release" inside a packaged app, "source" otherwise.
    public static func currentBuild() -> String {
        Bundle.main.bundleURL.pathExtension == "app" ? "release" : "source"
    }

    public func run(models: [AppBenchmarkModel],
                    mode: AppBenchmarkMode,
                    progress: @escaping @Sendable (AppBenchmarkProgress) -> Void = { _ in }
    ) async -> AppBenchmarkResult {
        let started = now()
        var runs: [AppBenchmarkResult.ModelRun] = []
        for (index, model) in models.enumerated() {
            var base = AppBenchmarkProgress(
                kind: .modelStarted, modelIndex: index, modelCount: models.count,
                modelID: model.id, modelName: model.descriptor.displayName)
            if Task.isCancelled {
                runs.append(Self.emptyRun(model, status: .cancelled, error: nil))
                continue
            }
            var waited: Double?
            if index > 0 {
                var event = base
                event.kind = .coolingDown
                progress(event)
                waited = await coolDown()
                if Task.isCancelled {
                    runs.append(Self.emptyRun(model, status: .cancelled, error: nil))
                    continue
                }
            }
            var conditions = sensors.sample(cooldownSeconds: waited)
            progress(base)
            var run = await runModel(model, mode: mode) { step, index, count, finished in
                var event = base
                event.kind = finished ? .stepFinished : .stepStarted
                event.stepIndex = index
                event.stepCount = count
                event.workload = step.workload
                event.trial = step.trial
                progress(event)
            } loaded: {
                var event = base
                event.kind = .loaded
                progress(event)
            }
            conditions.thermalStateAtEnd = sensors.thermalState()
            run.conditions = conditions
            runs.append(run)
            base.kind = .modelFinished
            base.status = run.status
            base.message = run.error
            progress(base)
        }
        return AppBenchmarkResult(
            id: UUID().uuidString.lowercased(),
            suite: .current(mode),
            app: .init(version: TUFFVersion.current, build: appBuild),
            machine: machine,
            startedAt: started,
            finishedAt: now(),
            runs: runs)
    }

    /// Waits the minimum, then while macOS reports the Mac warmer than
    /// nominal, up to the maximum. Returns the seconds waited.
    private func coolDown() async -> Double {
        var waited = 0.0
        do {
            if cooldown.minimumSeconds > 0 {
                try await sleep(cooldown.minimumSeconds)
                waited = cooldown.minimumSeconds
            }
            while cooldown.pollSeconds > 0, waited < cooldown.maximumSeconds,
                  sensors.thermalState() != "nominal" {
                let step = min(cooldown.pollSeconds, cooldown.maximumSeconds - waited)
                try await sleep(step)
                waited += step
            }
        } catch {}
        return waited
    }

    // MARK: - One model

    private func runModel(
        _ model: AppBenchmarkModel,
        mode: AppBenchmarkMode,
        stepProgress: (AppBenchmarkStep, Int, Int, Bool) -> Void,
        loaded: () -> Void
    ) async -> AppBenchmarkResult.ModelRun {
        let plan = Self.plan(for: model.descriptor, on: device, modelDirectory: model.directory)
        var run = Self.emptyRun(model, status: .completed, error: nil)
        run.settings = plan.settings

        let loadStarted = clock.now
        do {
            try await client.ensureLoaded(
                modelDirectory: model.directory,
                maxContextTokens: plan.settings.contextTokens,
                options: plan.options,
                forceLogitsHead: plan.settings.temperature != 0) { _ in }
            run.loadSeconds = seconds(since: loadStarted)
        } catch {
            await client.unload()
            run.status = Task.isCancelled ? .cancelled : .failed
            run.error = Task.isCancelled ? nil : "Could not load: \(error)"
            return run
        }
        loaded()

        let steps = AppBenchmarkSuite.steps(for: mode)
        var lastLong: (prompt: String, answer: String)?
        stepLoop: for (index, step) in steps.enumerated() {
            if Task.isCancelled {
                run.status = .cancelled
                break
            }
            // MiniMax always reasons, so a 96-token check is spent thinking
            // and would only measure that. Recorded as not run.
            if step.workload == .check, model.descriptor.reasoningControl == .alwaysOn {
                run.check = .init(passed: nil, answer: "")
                continue
            }
            stepProgress(step, index, steps.count, false)
            let prompt = AppBenchmarkSuite.prompt(for: step)
            var history: [AppChatTurn] = []
            if step.workload == .followUp {
                guard let lastLong else { continue }
                history = [AppChatTurn(prompt: lastLong.prompt, response: lastLong.answer)]
            }
            let request = plan.request(directory: model.directory, prompt: prompt,
                                       history: history, maxNewTokens: step.maxNewTokens,
                                       conversationKey: "benchmark-\(step.workload.rawValue)-\(step.trial)")
            do {
                let outcome = try await generate(request)
                run.trials.append(Self.trial(step, outcome.diagnostics))
                switch step.workload {
                case .check:
                    let answer = outcome.answer.trimmingCharacters(in: .whitespacesAndNewlines)
                    run.check = .init(
                        passed: answer.isEmpty ? nil
                            : answer.lowercased().contains(AppBenchmarkSuite.checkAnswer),
                        answer: String(answer.prefix(80)))
                case .long:
                    lastLong = (prompt, outcome.answer)
                default:
                    break
                }
            } catch is CancellationError {
                run.status = .cancelled
                break stepLoop
            } catch {
                run.status = Task.isCancelled ? .cancelled : .failed
                run.error = Task.isCancelled
                    ? nil : "\(step.workload.rawValue) trial \(step.trial): \(error)"
                break stepLoop
            }
            stepProgress(step, index, steps.count, true)
        }
        await client.unload()
        if run.status == .completed || !run.trials.isEmpty {
            run.summary = AppBenchmarkResult.ModelRun.summarize(run.trials)
        }
        return run
    }

    private struct Outcome {
        var answer: String
        var diagnostics: AppDiagnostics
    }

    private func generate(_ request: AppGenerationRequest) async throws -> Outcome {
        try await withTaskCancellationHandler {
            var answer = ""
            for try await event in client.generate(request) {
                switch event {
                case .token(let token):
                    answer += token.textDelta
                case .finished(let diagnostics):
                    return Outcome(answer: answer, diagnostics: diagnostics)
                case .cancelled:
                    throw CancellationError()
                case .failed(let error, _):
                    throw error
                default:
                    break
                }
            }
            throw AppInferenceError.invalidRequest("the generation ended without a result")
        } onCancel: { [client] in
            client.cancel()
        }
    }

    private func seconds(since start: ContinuousClock.Instant) -> Double {
        let elapsed = start.duration(to: clock.now).components
        return Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
    }

    static func trial(_ step: AppBenchmarkStep, _ diagnostics: AppDiagnostics)
        -> AppBenchmarkResult.Trial {
        // Dense models never look up an expert; leave the counts out rather
        // than report a misleading zero.
        let experts = diagnostics.runner?.expertReads.flatMap { $0.demandRequests > 0 ? $0 : nil }
        return AppBenchmarkResult.Trial(
            workload: step.workload,
            trial: step.trial,
            promptTokens: diagnostics.promptTokenCount ?? 0,
            cachedTokens: diagnostics.cachedPromptTokens ?? 0,
            prefillSeconds: diagnostics.prefillSeconds ?? 0,
            timeToFirstTokenSeconds: diagnostics.requestStartTimeToFirstTokenSeconds,
            generatedTokens: diagnostics.generatedTokens,
            decodeSeconds: diagnostics.decodeSeconds,
            stopReason: diagnostics.stopReason.rawValue,
            peakMemoryBytes: diagnostics.peakMemoryBytes,
            expertRequests: experts?.demandRequests,
            expertReads: experts?.demandReads)
    }

    static func emptyRun(_ model: AppBenchmarkModel,
                         status: AppBenchmarkResult.Status,
                         error: String?) -> AppBenchmarkResult.ModelRun {
        let descriptor = model.descriptor
        return AppBenchmarkResult.ModelRun(
            model: .init(id: descriptor.id.rawValue,
                         name: descriptor.displayName,
                         revision: descriptor.source.revision,
                         weights: descriptor.architecture.weightLayout.rawValue,
                         installedBytes: descriptor.source.installedBytes),
            settings: nil, status: status, error: error,
            loadSeconds: nil, check: nil, trials: [], summary: nil, conditions: nil)
    }

    // MARK: - Settings

    /// The settings a fresh install of TUFF uses for this model on this Mac:
    /// the catalog defaults resolved by Auto. Sampling uses the model's own
    /// defaults with a fixed seed, as chat does.
    struct Plan {
        var settings: AppBenchmarkResult.Settings
        var options: AppRuntimeOptions
        var reasoning: ChatReasoning
        var reasoningEffort: GPTOSSReasoningEffort?

        func request(directory: URL, prompt: String, history: [AppChatTurn],
                     maxNewTokens: Int, conversationKey: String) -> AppGenerationRequest {
            AppGenerationRequest(
                modelDirectory: directory,
                prompt: prompt,
                history: history,
                maxNewTokens: maxNewTokens,
                maxContextTokens: settings.contextTokens,
                reasoning: reasoning,
                reasoningEffort: reasoningEffort,
                temperature: Float(settings.temperature),
                topK: settings.topK,
                topP: settings.topP.map(Float.init),
                seed: AppBenchmarkSuite.seed,
                runtimeOptions: options,
                conversationKey: conversationKey)
        }
    }

    /// The settings a new chat would load the model with: the same Auto plan
    /// as the app, including which dense layers stay resident. Without the
    /// plan's residency the options default to keeping every dense layer,
    /// which a model larger than the Mac's memory cannot do (Qwen 3.8 27B
    /// failed with a Metal out-of-memory error on a 16 GB Mac in 9.2.0).
    static func plan(for descriptor: TUFFModelDescriptor,
                     on device: TUFFDeviceCapabilities,
                     modelDirectory: URL? = nil, useBundledLayout: Bool = false) -> Plan {
        let install = AppModelInstallDescriptor(catalog: descriptor)
        let defaults = AppModelSettingsProfile.defaults(for: descriptor.id.rawValue)
        let profile = AppAutomaticMemoryPlanner.applying(
            defaults, for: install, on: device, modelDirectory: modelDirectory,
            useBundledLayout: useBundledLayout)
        let autoPlan = defaults.automaticMemory ? AppAutomaticMemoryPlanner.plan(
            for: install, on: device, profile: defaults.automaticMemoryProfile,
            modelDirectory: modelDirectory, useBundledLayout: useBundledLayout) : nil
        let chunk = autoPlan?.prefillChunkTokens ?? descriptor.recommendedPrefillChunkTokens(on: device)
        let options = AppRuntimeOptions(
            expertCacheSlots: profile.expertCacheSlots,
            prefillEnabled: profile.prefillEnabled,
            prefillChunkTokens: chunk,
            rdadvisePolicy: profile.rdadvisePolicy,
            visionResidencyPolicy: .onDemand,
            denseResidencyPolicy: autoPlan?.denseResidencyPolicy
                ?? profile.manualDenseResidencyPolicy(for: install),
            retainedConversationBytes: autoPlan?.retainedConversationBytes)
        // Thinking off wherever the model allows it, so every model spends
        // its tokens on the answer. GPT-OSS uses its lowest effort.
        let reasoning: ChatReasoning = descriptor.reasoningControl == .alwaysOn ? .on : .off
        let effort: GPTOSSReasoningEffort? = descriptor.reasoningControl == .graded ? .low : nil
        let reasoningLabel = effort.map { "effort-\($0.rawValue)" } ?? reasoning.rawValue
        return Plan(
            settings: .init(
                contextTokens: profile.contextTokens,
                expertCacheSlots: profile.expertCacheSlots,
                prefillChunkTokens: chunk,
                batchedPrefill: profile.prefillEnabled,
                reasoning: reasoningLabel,
                temperature: profile.temperature,
                topK: profile.topKEnabled ? profile.topK : nil,
                topP: profile.topPEnabled ? profile.topP : nil),
            options: options,
            reasoning: reasoning,
            reasoningEffort: effort)
    }
}
