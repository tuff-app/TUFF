import Foundation

/// Monotonic request boundaries. These spans describe server work through the
/// first visible event, not client network delivery or per-token GPU latency.
final class ServerRequestTimings: @unchecked Sendable {
    private let lock = NSLock()
    private let arrival: UInt64
    private let validated: UInt64
    private var preparationStart: UInt64?
    private var preparationEnd: UInt64?
    private var generationStart: UInt64?
    private var firstEvent: UInt64?

    init(arrival: UInt64, validated: UInt64) {
        self.arrival = arrival
        self.validated = validated
    }

    func preparing(at time: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        lock.withLock { preparationStart = time }
    }
    func prepared(at time: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        lock.withLock { preparationEnd = time }
    }
    func generating(at time: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        lock.withLock { generationStart = time }
    }
    func visibleEvent(at time: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        lock.withLock { if firstEvent == nil { firstEvent = time } }
    }
    /// A discarded attempt's output was never shown, so the next attempt's
    /// first output is the first visible event.
    func discardVisibleEvent() {
        lock.withLock { firstEvent = nil }
    }
    func snapshot() -> [String: Double] {
        lock.withLock {
            func seconds(_ start: UInt64, _ end: UInt64) -> Double {
                Double(end >= start ? end - start : 0) / 1e9
            }
            var values = ["body_and_validation": seconds(arrival, validated)]
            if let start = preparationStart, let end = preparationEnd {
                values["queue_and_model_load"] = seconds(validated, start)
                values["preparation"] = seconds(start, end)
                if let generationStart {
                    values["stream_setup"] = seconds(end, generationStart)
                    if let firstEvent {
                        values["generation_to_first_event"] = seconds(generationStart, firstEvent)
                        values["time_to_first_event"] = seconds(arrival, firstEvent)
                    }
                }
            }
            return values
        }
    }
}
