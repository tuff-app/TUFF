import Foundation
import Testing
@testable import TUFFAppCore

@Suite struct AppModelLiveMetricsTests {
    @MainActor
    @Test func prefillProgressUpdatesPhaseAndCounts() {
        let model = makeAppModel()
        model.apply(.prefillProgress(done: 12, total: 50))
        #expect(model.phase == .prefill)
        #expect(model.livePrefillDone == 12)
        #expect(model.livePrefillTotal == 50)
    }

    @MainActor
    @Test func tokenEventSwitchesToDecodeAndComputesRate() {
        let model = makeAppModel()
        model.apply(.token(AppTokenEvent(
            index: 4, textDelta: " x", elapsedDecodeSeconds: 2.0)))
        #expect(model.phase == .decode)
        #expect(model.liveTokenCount == 5)
        #expect(model.liveElapsedDecodeSeconds == 2.0)
        #expect(model.liveTokensPerSecond == 2.5)
        #expect(model.liveMemoryBytes != nil)
    }

    @MainActor
    @Test func liveRateIsZeroBeforeAnyToken() {
        let model = makeAppModel()
        #expect(model.liveTokensPerSecond == 0)
    }

    @MainActor
    @Test func terminalEventsReturnPhaseToIdleKeepingLastValues() async {
        let client = MockInferenceClient(response: "one two", tokenDelayNanos: 1_000_000)
        let model = makeAppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "go"
        model.run()
        for _ in 0..<400 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(!model.isRunning)
        #expect(model.phase == .idle)
        #expect(model.liveTokenCount > 0)
    }

    @MainActor
    @Test func runResetsLiveMetricsAndEntersPrefill() async {
        let client = MockInferenceClient(response: "one two", tokenDelayNanos: 1_000_000_000)
        let model = makeAppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.apply(.token(AppTokenEvent(
            index: 9, textDelta: "z", elapsedDecodeSeconds: 5)))
        model.promptText = "go"
        model.run()
        #expect(model.phase == .prefill)
        #expect(model.liveTokenCount == 0)
        #expect(model.liveElapsedDecodeSeconds == 0)
        #expect(model.livePrefillDone == 0)
        model.cancel()
        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    @MainActor
    @Test func cancelKeepsPartialLiveMetrics() async {
        let client = MockInferenceClient(response: "one two three four", tokenDelayNanos: 50_000_000)
        let model = makeAppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "go"
        model.run()
        for _ in 0..<400 where model.liveTokenCount == 0 {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        let partial = model.liveTokenCount
        model.cancel()
        for _ in 0..<400 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(!model.isRunning)
        #expect(model.phase == .idle)
        #expect(model.liveTokenCount >= partial && partial > 0)
        #expect(model.error == .cancelled)
    }
}
