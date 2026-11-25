@testable import TUFFServerEngine
import Testing
import TUFFEngine
@testable import TUFFServerCore

@Suite struct ServerRequestTimingsTests {
    @Test func exclusiveSpansIncludeBodyAndQueue() {
        let t = ServerRequestTimings(arrival: 0, validated: 1_000_000_000)
        t.preparing(at: 3_000_000_000)
        t.prepared(at: 4_000_000_000)
        t.generating(at: 5_000_000_000)
        t.visibleEvent(at: 8_000_000_000)
        t.visibleEvent(at: 9_000_000_000)
        let s = t.snapshot()
        #expect(s["body_and_validation"] == 1)
        #expect(s["queue_and_model_load"] == 2)
        #expect(s["preparation"] == 1)
        #expect(s["stream_setup"] == 1)
        #expect(s["generation_to_first_event"] == 3)
        #expect(s["time_to_first_event"] == 8)
    }
    @Test func aDiscardedAttemptsOutputIsNotTheFirstEvent() {
        let t = ServerRequestTimings(arrival: 0, validated: 1_000_000_000)
        t.preparing(at: 2_000_000_000); t.prepared(at: 3_000_000_000); t.generating(at: 4_000_000_000)
        t.visibleEvent(at: 5_000_000_000)
        t.discardVisibleEvent()
        t.visibleEvent(at: 14_000_000_000)
        #expect(t.snapshot()["time_to_first_event"] == 14)
        #expect(t.snapshot()["generation_to_first_event"] == 10)
    }
    @Test func emptyAnswerDoesNotInventFirstToken() {
        let t = ServerRequestTimings(arrival: 1, validated: 2)
        t.preparing(at: 3); t.prepared(at: 4); t.generating(at: 5)
        #expect(t.snapshot()["time_to_first_event"] == nil)
    }

    @Test func backendSubspansAccumulateWithoutOverwritingExistingPhases() {
        var phases = ServerInferencePhaseTimings(["prefill": 3, "decode": 4])
        phases.record("prompt_render_and_tokenization", since: 1_000_000_000, through: 2_000_000_000)
        phases.record("prompt_render_and_tokenization", since: 3_000_000_000, through: 3_500_000_000)
        phases.record("cache_plan", since: 5_000_000_000, through: 5_100_000_000)
        #expect(phases.seconds["prompt_render_and_tokenization"] == 1.5)
        #expect(phases.seconds["cache_plan"] == 0.1)
        #expect(phases.seconds["prefill"] == 3)
        #expect(phases.seconds["decode"] == 4)
    }

    @Test func cacheSubspansNeverReuseAnEarlierRequestsCopyTime() {
        let store = ConversationStateStore(budgetBytes: 0)
        var statistics = store.statistics
        statistics.lastLookupSeconds = 0.01
        statistics.lastCaptureSeconds = 0.2
        statistics.lastRestoreSeconds = 0.3
        var phases = ServerInferencePhaseTimings()
        phases.recordCachePlan(statistics)
        #expect(phases.seconds["cache_lookup_and_bridge"] == 0.01)
        #expect(phases.seconds["cache_capture"] == 0.2)
        #expect(phases.seconds["cache_restore"] == 0.3)
        phases.recordCachePlan(store.statistics)
        #expect(phases.seconds["cache_lookup_and_bridge"] == 0)
        #expect(phases.seconds["cache_capture"] == 0)
        #expect(phases.seconds["cache_restore"] == 0)
    }
}
