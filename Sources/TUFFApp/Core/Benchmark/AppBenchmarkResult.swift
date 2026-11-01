import Foundation

/// A complete benchmark run, as shared to GitHub Discussions. The JSON form
/// is versioned by `schema`; the leaderboard validator rejects anything else.
///
/// Holds only what a comparison needs. No serial number, host name, user
/// name, file path or prompt text from the user is ever recorded.
public struct AppBenchmarkResult: Codable, Equatable, Sendable {
    public static let schemaIdentifier = "tuff-benchmark/1"

    public var schema: String = AppBenchmarkResult.schemaIdentifier
    /// Random per run. Lets the validator spot the same run posted twice.
    public var id: String
    public var suite: Suite
    public var app: App
    public var machine: AppBenchmarkMachine
    public var startedAt: Date
    public var finishedAt: Date
    public var runs: [ModelRun]

    public struct Suite: Codable, Equatable, Sendable {
        public var name: String
        public var version: Int
        public var mode: AppBenchmarkMode
        public var workloadSHA256: String

        // Spelled so the snake_case strategies round-trip it as
        // `workload_sha256`; the default spelling would not decode.
        private enum CodingKeys: String, CodingKey {
            case name, version, mode
            case workloadSHA256 = "workloadSha256"
        }

        public static func current(_ mode: AppBenchmarkMode) -> Suite {
            Suite(name: AppBenchmarkSuite.name, version: AppBenchmarkSuite.version,
                  mode: mode, workloadSHA256: AppBenchmarkSuite.workloadSHA256)
        }
    }

    public struct App: Codable, Equatable, Sendable {
        public var version: String
        /// "release" for the packaged app, "source" for a clone build.
        public var build: String
    }

    public enum Status: String, Codable, Equatable, Sendable {
        case completed
        case failed
        case skipped
        case cancelled
    }

    public struct Model: Codable, Equatable, Sendable {
        public var id: String
        public var name: String
        public var revision: String
        public var weights: String
        public var installedBytes: UInt64
    }

    public struct Settings: Codable, Equatable, Sendable {
        public var contextTokens: Int
        public var expertCacheSlots: Int
        public var prefillChunkTokens: Int
        public var batchedPrefill: Bool
        public var reasoning: String
        public var temperature: Double
        public var topK: Int?
        public var topP: Double?
    }

    public struct Trial: Codable, Equatable, Sendable {
        public var workload: AppBenchmarkWorkload
        public var trial: Int
        public var promptTokens: Int
        public var cachedTokens: Int
        public var prefillSeconds: Double
        public var timeToFirstTokenSeconds: Double?
        public var generatedTokens: Int
        public var decodeSeconds: Double
        public var stopReason: String
        public var peakMemoryBytes: UInt64?
        /// Expert lookups and the ones that had to be read from disk. Nil for
        /// dense models.
        public var expertRequests: UInt64?
        public var expertReads: UInt64?

        public var decodeTokensPerSecond: Double? {
            // The first token comes out of prefill; decode time covers the rest.
            guard generatedTokens > 1, decodeSeconds > 0 else { return nil }
            return Double(generatedTokens - 1) / decodeSeconds
        }

        public var prefillTokensPerSecond: Double? {
            let processed = promptTokens - cachedTokens
            guard processed > 0, prefillSeconds > 0 else { return nil }
            return Double(processed) / prefillSeconds
        }
    }

    public struct Check: Codable, Equatable, Sendable {
        /// Nil when the check could not run, for example a model that spent
        /// its whole budget reasoning.
        public var passed: Bool?
        public var answer: String
    }

    public struct Summary: Codable, Equatable, Sendable {
        public var decodeTokensPerSecond: AppBenchmarkStatistic?
        public var prefillTokensPerSecond: AppBenchmarkStatistic?
        public var shortTimeToFirstTokenSeconds: AppBenchmarkStatistic?
        public var longTimeToFirstTokenSeconds: AppBenchmarkStatistic?
        public var followUpTimeToFirstTokenSeconds: Double?
        public var followUpCachedTokens: Int?
        public var peakMemoryBytes: UInt64?
    }

    public struct ModelRun: Codable, Equatable, Sendable {
        public var model: Model
        public var settings: Settings?
        public var status: Status
        public var error: String?
        public var loadSeconds: Double?
        public var check: Check?
        public var trials: [Trial]
        public var summary: Summary?
        /// Nil in results from before 10.0.2.
        public var conditions: AppBenchmarkConditions?
    }

    public var completedRuns: [ModelRun] { runs.filter { $0.status == .completed } }
}

/// Median, range and sample count for a set of trials.
public struct AppBenchmarkStatistic: Codable, Equatable, Sendable {
    public var median: Double
    public var min: Double
    public var max: Double
    public var count: Int

    public init?(_ values: [Double]) {
        let sorted = values.filter(\.isFinite).sorted()
        guard let first = sorted.first, let last = sorted.last else { return nil }
        let middle = sorted.count / 2
        median = sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
        min = first
        max = last
        count = sorted.count
    }
}

extension AppBenchmarkResult.ModelRun {
    /// The summary the leaderboard shows, recomputed from the trials.
    static func summarize(_ trials: [AppBenchmarkResult.Trial]) -> AppBenchmarkResult.Summary {
        let short = trials.filter { $0.workload == .short }
        let long = trials.filter { $0.workload == .long }
        let followUp = trials.first { $0.workload == .followUp }
        return AppBenchmarkResult.Summary(
            decodeTokensPerSecond: AppBenchmarkStatistic(short.compactMap(\.decodeTokensPerSecond)),
            prefillTokensPerSecond: AppBenchmarkStatistic(long.compactMap(\.prefillTokensPerSecond)),
            shortTimeToFirstTokenSeconds: AppBenchmarkStatistic(short.compactMap(\.timeToFirstTokenSeconds)),
            longTimeToFirstTokenSeconds: AppBenchmarkStatistic(long.compactMap(\.timeToFirstTokenSeconds)),
            followUpTimeToFirstTokenSeconds: followUp?.timeToFirstTokenSeconds,
            followUpCachedTokens: followUp?.cachedTokens,
            peakMemoryBytes: trials.compactMap(\.peakMemoryBytes).max())
    }
}

extension AppBenchmarkResult {
    /// The encoder used everywhere the result is written, so the shared JSON
    /// is stable: snake_case keys, sorted, ISO 8601 dates.
    public static func encoder(pretty: Bool = false) -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = pretty
            ? [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
