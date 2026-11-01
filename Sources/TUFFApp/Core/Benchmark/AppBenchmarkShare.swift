import Foundation

/// Turns a result into a GitHub Discussions post: a readable table for
/// people, and the raw JSON the leaderboard reads.
public enum AppBenchmarkShare {
    public static let repository = "tuff-app/TUFF"
    public static let categorySlug = "benchmarks"
    public static let leaderboardURL = URL(string: "https://tuff-app.github.io/benchmarks/")!
    /// The opening of the fenced block the validator looks for.
    public static let dataFence = "```json tuff-benchmark"
    /// GitHub turns long query strings away; past this the body goes on the
    /// clipboard only.
    public static let maximumURLLength = 7_500

    public static func title(for result: AppBenchmarkResult) -> String {
        let names = result.completedRuns.map(\.model.name)
        let models: String
        switch names.count {
        case 0: models = "no completed models"
        case 1: models = names[0]
        case 2: models = names.joined(separator: " and ")
        default: models = "\(names.count) models"
        }
        return "\(result.machine.shortDescription): \(models) on TUFF \(result.app.version)"
    }

    public static func body(for result: AppBenchmarkResult) throws -> String {
        let data = try AppBenchmarkResult.encoder().encode(result)
        let json = String(decoding: data, as: UTF8.self)
        var lines = [
            "Benchmarked with TUFF \(result.app.version) on \(machineLine(result.machine)).",
            "Suite: \(result.suite.name) v\(result.suite.version), \(result.suite.mode.rawValue).",
            "",
            table(for: result),
            "",
        ]
        let problems = result.runs.filter { $0.status != .completed }
        if !problems.isEmpty {
            lines.append(problems.map {
                "- \($0.model.name): \($0.status.rawValue)" + ($0.error.map { " (\($0))" } ?? "")
            }.joined(separator: "\n"))
            lines.append("")
        }
        let slowed = result.completedRuns.compactMap { run in
            run.conditions.flatMap { $0.mayBeSlowed ? "- \(run.model.name): \(describe($0))" : nil }
        }
        if !slowed.isEmpty {
            lines.append("These may have run slower than usual:")
            lines.append(slowed.joined(separator: "\n"))
            lines.append("")
        }
        lines += [
            "<!-- Add notes above this line if you like. Please leave the data below as it is. -->",
            "",
            "<details><summary>Result data</summary>",
            "",
            dataFence,
            json,
            "```",
            "",
            "</details>",
        ]
        return lines.joined(separator: "\n")
    }

    /// The new-discussion page, with as much prefilled as fits in a URL.
    public static func discussionURL(title: String, body: String?) -> URL {
        var components = URLComponents(string: "https://github.com/\(repository)/discussions/new")!
        var items = [URLQueryItem(name: "category", value: categorySlug),
                     URLQueryItem(name: "title", value: title)]
        if let body { items.append(URLQueryItem(name: "body", value: body)) }
        components.queryItems = items
        if let url = components.url, url.absoluteString.count <= maximumURLLength || body == nil {
            return url
        }
        return discussionURL(title: title, body: nil)
    }

    /// Whether the whole post fits in the URL, or has to be pasted.
    public static func fitsInURL(title: String, body: String) -> Bool {
        discussionURL(title: title, body: body).absoluteString.contains("body=")
    }

    /// "started warm (fair), on battery, Low Power Mode".
    static func describe(_ conditions: AppBenchmarkConditions) -> String {
        var parts: [String] = []
        if conditions.thermalStateAtStart != "nominal" {
            parts.append("started warm (\(conditions.thermalStateAtStart))")
        }
        if let end = conditions.thermalStateAtEnd, end == "serious" || end == "critical" {
            parts.append("ended hot (\(end))")
        }
        if conditions.powerSource == "battery" { parts.append("on battery") }
        if conditions.lowPowerMode { parts.append("Low Power Mode") }
        return parts.joined(separator: ", ")
    }

    static func machineLine(_ machine: AppBenchmarkMachine) -> String {
        var parts = ["\(machine.chip) with \(machine.memoryDescription)"]
        if let gpu = machine.gpuCores { parts.append("\(gpu)-core GPU") }
        parts.append("macOS \(machine.macOSVersion)")
        return parts.joined(separator: ", ")
    }

    public static func table(for result: AppBenchmarkResult) -> String {
        var rows = [
            "| Model | Writes | Reads prompt | First token, long prompt | First token, follow-up | App memory | Check |",
            "| --- | ---: | ---: | ---: | ---: | ---: | :-: |",
        ]
        for run in result.runs where run.status == .completed {
            let summary = run.summary
            let followUp = summary?.followUpTimeToFirstTokenSeconds.map {
                seconds($0) + (summary?.followUpCachedTokens.map { ", \($0) tokens reused" } ?? "")
            } ?? "n/a"
            rows.append("| \(run.model.name) "
                + "| \(rate(summary?.decodeTokensPerSecond?.median)) "
                + "| \(rate(summary?.prefillTokensPerSecond?.median)) "
                + "| \(summary?.longTimeToFirstTokenSeconds.map { seconds($0.median) } ?? "n/a") "
                + "| \(followUp) "
                + "| \(summary?.peakMemoryBytes.map(gigabytes) ?? "n/a") "
                + "| \(checkMark(run.check)) |")
        }
        return rows.joined(separator: "\n")
    }

    public static func rate(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return value >= 10
            ? String(format: "%.1f tok/s", value)
            : String(format: "%.2f tok/s", value)
    }

    public static func seconds(_ value: Double) -> String {
        value >= 10 ? String(format: "%.0f s", value) : String(format: "%.1f s", value)
    }

    public static func gigabytes(_ bytes: UInt64) -> String {
        String(format: "%.1f GB", Double(bytes) / Double(1 << 30))
    }

    public static func checkMark(_ check: AppBenchmarkResult.Check?) -> String {
        switch check?.passed {
        case true?: "Pass"
        case false?: "Fail"
        case nil: "n/a"
        }
    }
}
