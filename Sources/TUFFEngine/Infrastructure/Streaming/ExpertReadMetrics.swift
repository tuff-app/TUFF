import TUFFInferenceTypes

public enum ExpertReadPurpose: Sendable, Equatable { case demand, prefetch }
public typealias ExpertReadMetrics = TUFFInferenceTypes.ExpertReadMetrics

extension TUFFInferenceTypes.ExpertReadMetrics {
    mutating func record(purpose: ExpertReadPurpose, bytes: UInt64, succeeded: Bool) {
        switch purpose {
        case .demand:
            demandBytes += bytes
            if succeeded { demandReads += 1 } else { demandFailures += 1 }
        case .prefetch:
            prefetchBytes += bytes
            if succeeded { prefetchReads += 1 } else { prefetchFailures += 1 }
        }
    }

    /// Bytes for a read already counted by `record`: one part of an expert
    /// read in several pieces.
    mutating func addBytes(purpose: ExpertReadPurpose, bytes: UInt64) {
        switch purpose {
        case .demand: demandBytes += bytes
        case .prefetch: prefetchBytes += bytes
        }
    }

    mutating func add(_ other: Self) {
        demandRequests += other.demandRequests
        predictionRequests += other.predictionRequests
        demandReads += other.demandReads
        demandBytes += other.demandBytes
        demandFailures += other.demandFailures
        prefetchReads += other.prefetchReads
        prefetchBytes += other.prefetchBytes
        prefetchFailures += other.prefetchFailures
        usefulPrefetchReads += other.usefulPrefetchReads
        unusedPrefetchEvictions += other.unusedPrefetchEvictions
        pollutedDemandMisses += other.pollutedDemandMisses
    }
}
