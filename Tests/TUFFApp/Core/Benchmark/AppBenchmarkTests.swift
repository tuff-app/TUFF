import Foundation
import Testing
import TUFFModelCatalog
@testable import TUFFAppCore

/// Answers every request from a script and records what it was asked.
private final class ScriptedBenchmarkClient: AppModelLifecycleClient, @unchecked Sendable {
    private let lock = NSLock()
    private var recordedRequests: [AppGenerationRequest] = []
    private var recordedLoads: [URL] = []
    private var unloadCount = 0
    var failLoads: Set<String> = []
    var answer = "Paris"
    var onGenerate: (@Sendable () -> Void)?

    var requests: [AppGenerationRequest] { lock.withLock { recordedRequests } }
    var loads: [URL] { lock.withLock { recordedLoads } }
    var unloads: Int { lock.withLock { unloadCount } }

    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions,
                      forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        lock.withLock { recordedLoads.append(modelDirectory) }
        if failLoads.contains(modelDirectory.lastPathComponent) {
            throw AppInferenceError.invalidRequest("not enough memory")
        }
    }

    func unload() async { lock.withLock { unloadCount += 1 } }

    func cancel() {}

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        lock.withLock { recordedRequests.append(request) }
        onGenerate?()
        let answer = self.answer
        let cached = request.history.isEmpty ? 0 : 1_000
        var diagnostics = AppDiagnostics(
            generatedTokens: request.maxNewTokens,
            stopReason: .maxTokens,
            promptTokenCount: 1_100,
            prefillSeconds: 2,
            timeToFirstTokenSeconds: 0.5,
            decodeSeconds: Double(request.maxNewTokens - 1) / 10,
            tokensPerSecond: 10,
            peakMemoryBytes: 1 << 30,
            runtimeOptions: request.runtimeOptions)
        diagnostics.cachedPromptTokens = cached
        let finished = diagnostics
        return AsyncThrowingStream { continuation in
            continuation.yield(.token(.init(index: 0, textDelta: answer, elapsedDecodeSeconds: 0)))
            continuation.yield(.finished(finished))
            continuation.finish()
        }
    }
}

private func model(_ descriptor: TUFFModelDescriptor) -> AppBenchmarkModel {
    AppBenchmarkModel(descriptor: descriptor,
                      directory: URL(fileURLWithPath: "/models/\(descriptor.installDirectoryName)"))
}

private let machine = AppBenchmarkMachine(
    chip: "Apple M2", modelIdentifier: "Mac14,2", memoryBytes: 16 << 30,
    performanceCores: 4, efficiencyCores: 4, gpuCores: 10, macOSVersion: "26.6.2")

private let device = TUFFDeviceCapabilities(
    unifiedMemoryBytes: 16 << 30, macOSMajorVersion: 26, appleSiliconGeneration: 2)

private let coolMac = AppBenchmarkSensors(
    thermalState: { "nominal" }, powerSource: { "ac" }, lowPowerMode: { false })

private func runner(_ client: ScriptedBenchmarkClient,
                    sensors: AppBenchmarkSensors = coolMac,
                    cooldown: AppBenchmarkCooldown = .none,
                    sleep: @escaping @Sendable (Double) async throws -> Void = { _ in }) -> AppBenchmarkRunner {
    AppBenchmarkRunner(client: client, device: device, machine: machine, appBuild: "source",
                       sensors: sensors, cooldown: cooldown, sleep: sleep,
                       now: { Date(timeIntervalSince1970: 1_800_000_000) })
}

/// Thermal readings handed out in order; the last one repeats.
private final class ThermalScript: @unchecked Sendable {
    private let lock = NSLock()
    private var states: [String]
    private(set) var sleeps: [Double] = []
    init(_ states: [String]) { self.states = states }
    func next() -> String {
        lock.lock(); defer { lock.unlock() }
        return states.count > 1 ? states.removeFirst() : states[0]
    }
    func slept(_ seconds: Double) { lock.lock(); sleeps.append(seconds); lock.unlock() }
}

private final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [AppBenchmarkProgress] = []
    func append(_ event: AppBenchmarkProgress) { lock.lock(); events.append(event); lock.unlock() }
    var kinds: [AppBenchmarkProgress.Kind] { lock.lock(); defer { lock.unlock() }; return events.map(\.kind) }
}

@Suite struct AppBenchmarkSuiteTests {
    @Test func standardRunsThreeTimedTrialsAndQuickRunsOne() {
        let standard = AppBenchmarkSuite.steps(for: .standard).map(\.workload)
        #expect(standard == [.check, .short, .short, .short, .long, .long, .long, .followUp])
        let quick = AppBenchmarkSuite.steps(for: .quick).map(\.workload)
        #expect(quick == [.check, .short, .long, .followUp])
    }

    @Test func everyTimedTrialStartsDifferentlySoReuseCannotCarryOver() {
        let prompts = AppBenchmarkSuite.steps(for: .standard)
            .filter { $0.workload == .short || $0.workload == .long }
            .map(AppBenchmarkSuite.prompt(for:))
        #expect(Set(prompts).count == prompts.count)
        #expect(prompts.allSatisfy { $0.hasPrefix("Benchmark trial ") })
    }

    @Test func theLongDocumentIsLongEnoughToMeasurePrefill() {
        let words = AppBenchmarkSuite.longDocument.split(whereSeparator: \.isWhitespace).count
        #expect((900...1_300).contains(words))
    }

    /// The leaderboard validator accepts this exact hash. Changing a prompt
    /// or limit must bump the suite version and the validator's list.
    @Test func workloadHashIsPinned() {
        #expect(AppBenchmarkSuite.version == 1)
        #expect(AppBenchmarkSuite.workloadSHA256
            == "610808a3c16ab35f09388fb46bb9b8f099ee6bef5811a5ef1d9b6e6b8a76f341")
    }

    @Test func statisticsUseTheMiddleValue() throws {
        let odd = try #require(AppBenchmarkStatistic([3, 1, 2]))
        #expect(odd.median == 2 && odd.min == 1 && odd.max == 3 && odd.count == 3)
        let even = try #require(AppBenchmarkStatistic([4, 1, 3, 2]))
        #expect(even.median == 2.5)
        #expect(AppBenchmarkStatistic([]) == nil)
        #expect(AppBenchmarkStatistic([.nan]) == nil)
    }

    @Test func decodeRateLeavesOutTheTokenPrefillProduced() {
        let trial = AppBenchmarkResult.Trial(
            workload: .short, trial: 1, promptTokens: 40, cachedTokens: 0, prefillSeconds: 0.5,
            timeToFirstTokenSeconds: 0.5, generatedTokens: 11, decodeSeconds: 2,
            stopReason: "maxTokens", peakMemoryBytes: nil, expertRequests: nil, expertReads: nil)
        #expect(trial.decodeTokensPerSecond == 5)
        #expect(trial.prefillTokensPerSecond == 80)
    }
}

@Suite struct AppBenchmarkProgressTests {
    /// The Benchmarks screen reads the runner's `--progress-json` lines with
    /// the result decoder. `modelID` used to come back as `modelId`, so every
    /// event failed to decode and the screen stayed at 0%.
    @Test func progressEventsRoundTripThroughTheStream() throws {
        let event = AppBenchmarkProgress(kind: .stepFinished, modelIndex: 9, modelCount: 12,
            modelID: "ornith-1.5-397b", modelName: "Ornith 1.5 397B Q4", stepIndex: 2,
            stepCount: 8, workload: .long, trial: 1, status: nil, message: nil)
        let data = try AppBenchmarkResult.encoder().encode(event)
        let json = String(decoding: data, as: UTF8.self)
        #expect(json.contains("\"model_id\":\"ornith-1.5-397b\""))
        #expect(try AppBenchmarkResult.decoder().decode(AppBenchmarkProgress.self, from: data) == event)
    }
}

@Suite struct AppBenchmarkRunnerTests {
    @Test func runsEachModelThenUnloadsAndKeepsGoingAfterAFailure() async {
        let client = ScriptedBenchmarkClient()
        client.failLoads = [TUFFModelCatalog.qwen36_35B_A3B.installDirectoryName]
        let result = await runner(client).run(
            models: [model(TUFFModelCatalog.gemma4_E4B), model(TUFFModelCatalog.qwen36_35B_A3B)],
            mode: .quick)

        #expect(result.runs.map(\.status) == [.completed, .failed])
        #expect(result.runs[1].error?.contains("not enough memory") == true)
        #expect(client.loads.count == 2)
        #expect(client.unloads == 2)
        #expect(client.requests.count == 4)
        let first = result.runs[0]
        #expect(first.check == .init(passed: true, answer: "Paris"))
        #expect(first.summary?.decodeTokensPerSecond?.median == 10)
        #expect(first.summary?.followUpCachedTokens == 1_000)
        #expect(result.app.build == "source")
        #expect(result.suite.workloadSHA256 == AppBenchmarkSuite.workloadSHA256)
    }

    @Test func theFollowUpContinuesTheLastLongConversation() async throws {
        let client = ScriptedBenchmarkClient()
        _ = await runner(client).run(models: [model(TUFFModelCatalog.gemma4_E4B)], mode: .standard)
        let followUp = try #require(client.requests.last)
        #expect(followUp.prompt == AppBenchmarkSuite.followUpPrompt)
        #expect(followUp.history.count == 1)
        #expect(followUp.history[0].prompt.hasPrefix("Benchmark trial 3."))
        #expect(client.requests.allSatisfy { $0.seed == AppBenchmarkSuite.seed })
    }

    @Test func aWrongAnswerFailsTheCheck() async {
        let client = ScriptedBenchmarkClient()
        client.answer = "Lyon"
        let result = await runner(client).run(models: [model(TUFFModelCatalog.gemma4_E4B)], mode: .quick)
        #expect(result.runs[0].check?.passed == false)
    }

    @Test func aModelThatAlwaysReasonsSkipsTheCheck() async {
        let client = ScriptedBenchmarkClient()
        let result = await runner(client).run(models: [model(TUFFModelCatalog.minimaxM27)], mode: .quick)
        #expect(result.runs[0].check?.passed == nil)
        #expect(!client.requests.contains { $0.prompt == AppBenchmarkSuite.checkPrompt })
        #expect(result.runs[0].settings?.reasoning == "on")
    }

    @Test func gptOSSUsesItsLowestReasoningEffort() {
        let plan = AppBenchmarkRunner.plan(for: TUFFModelCatalog.gptOss_20B, on: device)
        #expect(plan.reasoningEffort == .low)
        #expect(plan.settings.reasoning == "effort-low")
    }

    /// A model larger than memory is loaded the way a chat loads it: with
    /// Auto's streamed dense layers, not every layer resident.
    @Test func aDenseModelLargerThanMemoryStreamsLikeAChat() throws {
        let install = AppModelInstallDescriptor(catalog: TUFFModelCatalog.qwen38_27B)
        let plan = AppBenchmarkRunner.plan(for: TUFFModelCatalog.qwen38_27B, on: device,
                                           useBundledLayout: true)
        let chat = try #require(AppAutomaticMemoryPlanner.plan(
            for: install, on: device, profile: AppModelSettingsProfile.defaults(
                for: TUFFModelCatalog.qwen38_27B.id.rawValue).automaticMemoryProfile,
            useBundledLayout: true))
        #expect(plan.options.denseResidencyPolicy != .resident)
        #expect(plan.options.denseResidencyPolicy == chat.denseResidencyPolicy)
        #expect(plan.settings.contextTokens == chat.contextTokens)
    }

    @Test func everyModelAfterTheFirstWaitsForTheMacToCoolDown() async {
        // Model 1 start and end, two warm polls, a cool poll, then model 2
        // start and end.
        let script = ThermalScript(["nominal", "fair", "fair", "fair", "nominal", "fair", "serious"])
        let sensors = AppBenchmarkSensors(thermalState: { script.next() },
                                          powerSource: { "battery" }, lowPowerMode: { true })
        let events = ProgressLog()
        let result = await runner(ScriptedBenchmarkClient(), sensors: sensors,
                                  cooldown: .init(minimumSeconds: 20, maximumSeconds: 300, pollSeconds: 5),
                                  sleep: { script.slept($0) })
            .run(models: [model(TUFFModelCatalog.gemma4_E4B), model(TUFFModelCatalog.gemma4_E2B)],
                 mode: .quick) { events.append($0) }

        #expect(script.sleeps == [20, 5, 5])
        #expect(events.kinds.filter { $0 == .coolingDown }.count == 1)
        let first = result.runs[0].conditions, second = result.runs[1].conditions
        #expect(first == .init(thermalStateAtStart: "nominal", thermalStateAtEnd: "fair",
                               powerSource: "battery", lowPowerMode: true, cooldownSeconds: nil))
        #expect(second?.cooldownSeconds == 30)
        #expect(second?.thermalStateAtStart == "fair")
        #expect(second?.thermalStateAtEnd == "serious")
    }

    @Test func theCooldownGivesUpAtItsLimit() async {
        let script = ThermalScript(["critical"])
        let sensors = AppBenchmarkSensors(thermalState: { script.next() },
                                          powerSource: { "ac" }, lowPowerMode: { false })
        _ = await runner(ScriptedBenchmarkClient(), sensors: sensors,
                         cooldown: .init(minimumSeconds: 20, maximumSeconds: 32, pollSeconds: 5),
                         sleep: { script.slept($0) })
            .run(models: [model(TUFFModelCatalog.gemma4_E4B), model(TUFFModelCatalog.gemma4_E2B)],
                 mode: .quick)
        #expect(script.sleeps == [20, 5, 5, 2])
    }

    @Test func cancellingStopsTheRunAndMarksWhatIsLeft() async {
        let client = ScriptedBenchmarkClient()
        let task = Task {
            await runner(client).run(
                models: [model(TUFFModelCatalog.gemma4_E4B), model(TUFFModelCatalog.qwen36_35B_A3B)],
                mode: .standard)
        }
        client.onGenerate = { task.cancel() }
        let result = await task.value
        #expect(result.runs.map(\.status) == [.cancelled, .cancelled])
        #expect(client.requests.count == 1)
    }
}

@Suite struct AppBenchmarkShareTests {
    private func sampleResult() async -> AppBenchmarkResult {
        await runner(ScriptedBenchmarkClient()).run(
            models: [model(TUFFModelCatalog.gemma4_E4B)], mode: .quick)
    }

    @Test func theResultRoundTripsAndNamesNoLocalPaths() async throws {
        let result = await sampleResult()
        let data = try AppBenchmarkResult.encoder().encode(result)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("\"schema\":\"tuff-benchmark/1\""))
        #expect(text.contains("\"workload_sha256\""))
        #expect(!text.contains("/models/"))
        let decoded = try AppBenchmarkResult.decoder().decode(AppBenchmarkResult.self, from: data)
        #expect(decoded == result)
    }

    @Test func thePostHasATableAndTheDataTheLeaderboardReads() async throws {
        let result = await sampleResult()
        let body = try AppBenchmarkShare.body(for: result)
        #expect(body.contains("| Gemma 4 E4B Q4 | 10.0 tok/s |"))
        let start = try #require(body.range(of: AppBenchmarkShare.dataFence + "\n"))
        let end = try #require(body.range(of: "\n```", range: start.upperBound..<body.endIndex))
        let json = Data(body[start.upperBound..<end.lowerBound].utf8)
        let decoded = try AppBenchmarkResult.decoder().decode(AppBenchmarkResult.self, from: json)
        #expect(decoded.id == result.id)
        #expect(AppBenchmarkShare.title(for: result) == "M2, 16 GB: Gemma 4 E4B Q4 on TUFF \(TUFFVersion.current)")
    }

    @Test func thePostSaysWhenAModelMayHaveRunSlow() async throws {
        var result = await sampleResult()
        #expect(!(try AppBenchmarkShare.body(for: result)).contains("slower than usual"))
        result.runs[0].conditions = .init(thermalStateAtStart: "fair", thermalStateAtEnd: "serious",
                                          powerSource: "battery", lowPowerMode: false)
        let body = try AppBenchmarkShare.body(for: result)
        #expect(body.contains("- Gemma 4 E4B Q4: started warm (fair), ended hot (serious), on battery"))
    }

    @Test func aResultFromBeforeConditionsStillDecodes() async throws {
        let result = await sampleResult()
        var object = try #require(try JSONSerialization.jsonObject(
            with: AppBenchmarkResult.encoder().encode(result)) as? [String: Any])
        var runs = try #require(object["runs"] as? [[String: Any]])
        #expect(runs[0]["conditions"] != nil)
        runs[0].removeValue(forKey: "conditions")
        object["runs"] = runs
        let decoded = try AppBenchmarkResult.decoder().decode(
            AppBenchmarkResult.self, from: JSONSerialization.data(withJSONObject: object))
        #expect(decoded.runs[0].conditions == nil)
    }

    @Test func aPostTooLongForAURLOpensWithTheTitleOnly() {
        let short = AppBenchmarkShare.discussionURL(title: "M2", body: "hello")
        #expect(short.absoluteString.contains("category=benchmarks"))
        #expect(short.absoluteString.contains("body=hello"))
        let long = AppBenchmarkShare.discussionURL(title: "M2", body: String(repeating: "x", count: 10_000))
        #expect(!long.absoluteString.contains("body="))
        #expect(long.absoluteString.contains("title=M2"))
    }
}

@Suite struct AppBenchmarkCommandTests {
    @Test func parsesModelsModeAndSharing() throws {
        let options = try AppBenchmarkCommand.parse(["--models", "gemma4, qwen36", "--quick", "--share"])
        #expect(options.models == ["gemma4", "qwen36"])
        #expect(options.mode == .quick)
        #expect(options.share)
        #expect(throws: AppBenchmarkCommand.UsageError.noModels) {
            try AppBenchmarkCommand.parse([])
        }
        #expect(throws: AppBenchmarkCommand.UsageError.unknownArgument("--fast")) {
            try AppBenchmarkCommand.parse(["--all", "--fast"])
        }
        #expect(throws: AppBenchmarkCommand.UsageError.missingValue("--models")) {
            try AppBenchmarkCommand.parse(["--models"])
        }
        #expect(try AppBenchmarkCommand.parse(["--list"]).list)
    }

    @Test func selectsByAnyNameAndRunsEverythingSmallestFirst() throws {
        let installed = [model(TUFFModelCatalog.qwen36_35B_A3B), model(TUFFModelCatalog.gemma4_E4B)]
        var options = try AppBenchmarkCommand.parse(["--models", "gemma4-e4b,qwen36"])
        #expect(try AppBenchmarkCommand.select(options, installed: installed).map(\.id)
            == [TUFFModelCatalog.gemma4_E4B.id.rawValue, TUFFModelCatalog.qwen36_35B_A3B.id.rawValue])
        options = try AppBenchmarkCommand.parse(["--all"])
        #expect(try AppBenchmarkCommand.select(options, installed: installed).first?.id
            == TUFFModelCatalog.gemma4_E4B.id.rawValue)
        options = try AppBenchmarkCommand.parse(["--models", "gpt-oss-120b"])
        #expect(throws: AppBenchmarkCommand.UsageError.notInstalled(TUFFModelCatalog.gptOss_120B.displayName)) {
            try AppBenchmarkCommand.select(options, installed: installed)
        }
        options = try AppBenchmarkCommand.parse(["--models", "llama"])
        #expect(throws: AppBenchmarkCommand.UsageError.unknownModel("llama")) {
            try AppBenchmarkCommand.select(options, installed: installed)
        }
    }
}

/// The leaderboard validator reads `.github/benchmark-suite.json`. It must
/// know every catalog model and the current workload hash, or real results
/// would be rejected.
@Suite struct AppBenchmarkValidatorConfigTests {
    private struct Config: Decodable {
        let suites: [String: String]
        let models: [String: [String: AnyCodable]]
        struct AnyCodable: Decodable { init(from decoder: Decoder) throws {} }
    }

    @Test func validatorKnowsEveryModelAndTheCurrentSuite() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent(".github/benchmark-suite.json"))
        let config = try JSONDecoder().decode(Config.self, from: data)
        #expect(Set(config.models.keys) == Set(TUFFModelCatalog.all.map(\.id.rawValue)))
        #expect(config.suites["\(AppBenchmarkSuite.name)/\(AppBenchmarkSuite.version)"]
            == AppBenchmarkSuite.workloadSHA256)
    }
}
