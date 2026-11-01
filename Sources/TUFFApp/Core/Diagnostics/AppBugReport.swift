import Darwin
import Foundation
import TUFFModelCatalog

public struct AppDiagnosticSystem: Equatable, Sendable {
    public var macOS: String
    public var macModel: String
    public var chip: String
    public var memoryBytes: UInt64

    public init(macOS: String, macModel: String, chip: String, memoryBytes: UInt64) {
        self.macOS = macOS
        self.macModel = macModel
        self.chip = chip
        self.memoryBytes = memoryBytes
    }

    public static func current() -> Self {
        let os = ProcessInfo.processInfo.operatingSystemVersion
        return Self(macOS: "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)",
                    macModel: sysctlText("hw.model"), chip: sysctlText("machdep.cpu.brand_string"),
                    memoryBytes: ProcessInfo.processInfo.physicalMemory)
    }

    private static func sysctlText(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "Unknown" }
        var data = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &data, &size, nil, 0) == 0 else { return "Unknown" }
        return String(cString: data)
    }
}

/// Only public model identity, system facts and scalar settings enter this
/// summary. There is no input for chats, attachments, paths or credentials.
public struct AppBugReport: Sendable {
    public var system: AppDiagnosticSystem
    public var model: TUFFModelDescriptor?
    public var contextTokens: Int
    public var temperature: Double
    public var topK: Int
    public var topP: Double
    public var runtime: AppRuntimeOptions
    public var diagnostics: AppDiagnostics?

    public init(system: AppDiagnosticSystem, model: TUFFModelDescriptor?, contextTokens: Int,
                temperature: Double, topK: Int, topP: Double, runtime: AppRuntimeOptions,
                diagnostics: AppDiagnostics?) {
        self.system = system
        self.model = model
        self.contextTokens = contextTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.runtime = runtime
        self.diagnostics = diagnostics
    }

    private func safe(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " .,+()-"))
        return value.unicodeScalars.allSatisfy { allowed.contains($0) } ? value : "Redacted"
    }

    public var summary: String {
        var lines = ["TUFF \(TUFFVersion.current)",
            "macOS: \(safe(system.macOS))", "Mac: \(safe(system.macModel))",
            "Chip: \(safe(system.chip))", "Unified memory: \(system.memoryBytes) bytes",
            "Model: \(model.flatMap { supplied in TUFFModelCatalog.all.first { $0.id == supplied.id } }?.apiModelID ?? "Custom model")", "Context: \(contextTokens) tokens",
            "Sampling: temperature \(temperature), top-k \(topK), top-p \(topP)",
            "Expert cache: \(runtime.expertCacheSlots) slots (\(runtime.expertCachePolicy.rawValue))",
            "Prefill: \(runtime.prefillEnabled ? "on" : "off"), chunk \(runtime.prefillChunkTokens)",
            "Image residency: \(runtime.visionResidencyPolicy.rawValue)"]
        if let d = diagnostics {
            lines += ["Last generation: \(d.generatedTokens) tokens, stop \(d.stopReason.rawValue)",
                      "Decode: \(d.tokensPerSecond) tokens/s, \(d.decodeSeconds) seconds"]
            if let prefill = d.prefillSeconds { lines.append("Prefill: \(prefill) seconds") }
            if let ttft = d.requestStartTimeToFirstTokenSeconds { lines.append("First token: \(ttft) seconds") }
            if let peak = d.peakMemoryBytes { lines.append("Peak process memory: \(peak) bytes") }
        }
        return lines.joined(separator: "\n")
    }

    private static func formName(_ model: TUFFModelDescriptor) -> String {
        switch model.id {
        case .flint40B: "Flint 40B"
        case .bonsai2_2B: "Bonsai 2 2B"
        case .bonsai2_27B: "Bonsai 2 27B"
        case .gemma4_E2B: "Gemma 4 E2B"
        case .gemma4_E4B: "Gemma 4 E4B"
        case .gemma4_12B_QAT: "Gemma 4 12B QAT"
        case .gemma4_26B_A4B: "Gemma 4 26B-A4B"
        case .gemma4_26B_Heretic: "Gemma 4 26B Heretic"
        case .qwen36_35B_A3B: "Qwen3.6 35B-A3B"
        case .qwen38FlashNext: "Qwen3.8 Flash Next"
        case .gptOss_20B: "GPT-OSS 20B"
        case .gptOss_120B: "GPT-OSS 120B"
        case .minimaxM27: "MiniMax M2.7"
        case .ornith15_9B: "Ornith 1.5 9B"
        case .ornith15_35B_A3B: "Ornith 1.5 35B-A3B"
        case .ornith15_122B: "Ornith 1.5 122B"
        case .ornith15_397B: "Ornith 1.5 397B"
        case .qwen38_27B: "Qwen3.8 27B"
        }
    }

    public func issueURL(includeDiagnostics: Bool) -> URL {
        var components = URLComponents(string: "https://github.com/tuff-app/TUFF/issues/new")!
        components.queryItems = [URLQueryItem(name: "template", value: "bug.yml"),
            URLQueryItem(name: "version", value: TUFFVersion.current),
            URLQueryItem(name: "model", value: model.map { Self.formName($0) } ?? "Not sure"),
            URLQueryItem(name: "hardware", value: "\(safe(system.macModel)), \(safe(system.chip)), \(system.memoryBytes / (1 << 30)) GB"),
            URLQueryItem(name: "macos", value: safe(system.macOS))]
        if includeDiagnostics {
            components.queryItems?.append(URLQueryItem(name: "diagnostics", value: summary))
        }
        return components.url!
    }
}
