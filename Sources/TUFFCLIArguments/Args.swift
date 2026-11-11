import TUFFInferenceTypes
import TUFFRuntimeDescriptions

public struct Args: Codable, Equatable, Sendable {
    public var model: String
    public var prompt: String?
    public var chatPrompt: String?
    public var systemPrompt: String?
    public var messagesFile: String?
    public var images: [String]
    public var visionPack: String?
    public var visionResidency: VisionResidencyPolicy
    public var maxNew: Int
    public var maxContext: Int
    public var thinking: ChatReasoning
    public var reasoningEffort: GPTOSSReasoningEffort?
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var stops: [String]
    public var quiet: Bool
    public var expertCacheSlots: Int
    public var expertCachePolicy: RuntimeExpertCachePolicy
    public var prefillPolicy: RuntimePrefillPolicy
    public var prefillChunkTokens: Int
    /// `--prefill-chunk-tokens auto`: the size is decided once the prompt length
    /// is known, which needs the tokenizer and, for images, their geometry.
    public var prefillChunkTokensAuto: Bool
    /// Largest size `auto` may pick. `tuff prompt` passes the model's
    /// catalog recommendation; a direct CLI run keeps the historical 256.
    public var prefillChunkMax: Int
    public var rdadvisePolicy: RDAdvicePolicyMode
    public var speculativeMode: SpeculativeDecodeMode
    public var speculativeDraftTokens: Int
    public var denseResidencyPolicy: DenseResidencyPolicy?
    /// Resolved by the lightweight host before sending explicit options.
    /// Optional so older JSON argument payloads continue to decode.
    public var automaticMemory: Bool? = nil
    /// Set by the host when Auto chose the settings: the engine writes a
    /// `CLIRunReport` here so the host can record the run in Auto's history.
    /// Optional, so an engine that predates it simply writes nothing.
    public var runReportPath: String? = nil

    public init(model: String,
                prompt: String? = nil,
                chatPrompt: String? = nil,
                systemPrompt: String? = nil,
                messagesFile: String? = nil,
                images: [String] = [],
                visionPack: String? = nil,
                visionResidency: VisionResidencyPolicy = .defaultPolicy,
                maxNew: Int = 1_024,
                maxContext: Int = 8192,
                thinking: ChatReasoning = .off,
                reasoningEffort: GPTOSSReasoningEffort? = nil,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                expertCacheSlots: Int = RuntimeConfiguration.production.expertCacheSlots,
                expertCachePolicy: RuntimeExpertCachePolicy = RuntimeConfiguration.production.expertCachePolicy,
                prefillPolicy: RuntimePrefillPolicy = RuntimeConfiguration.production.prefillPolicy,
                prefillChunkTokens: Int = RuntimeConfiguration.production.prefillChunkTokens,
                prefillChunkTokensAuto: Bool = false,
                prefillChunkMax: Int = PrefillRuntimeConfig.baselineChunkTokens,
                rdadvisePolicy: RDAdvicePolicyMode = RuntimeConfiguration.production.rdadvisePolicy,
                speculativeMode: SpeculativeDecodeMode = .off,
                speculativeDraftTokens: Int = 4,
                denseResidencyPolicy: DenseResidencyPolicy? = nil) {
        self.model = model
        self.prompt = prompt
        self.chatPrompt = chatPrompt
        self.systemPrompt = systemPrompt
        self.messagesFile = messagesFile
        self.images = images
        self.visionPack = visionPack
        self.visionResidency = visionResidency
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.thinking = thinking
        self.reasoningEffort = reasoningEffort
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stops = stops
        self.quiet = quiet
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillPolicy = prefillPolicy
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillChunkTokensAuto = prefillChunkTokensAuto
        self.prefillChunkMax = prefillChunkMax
        self.rdadvisePolicy = rdadvisePolicy
        self.speculativeMode = speculativeMode
        self.speculativeDraftTokens = speculativeDraftTokens
        self.denseResidencyPolicy = denseResidencyPolicy
    }
}

public enum ArgsError: Error, Equatable, CustomStringConvertible {
    case helpRequested
    case unknownFlag(String)
    case missingValue(flag: String)
    case invalidValue(flag: String, value: String)
    case requiredMissing(String)
    case mutuallyExclusive(String, String)
    case unsupported(flag: String, context: String)
    case modeMissing
    case imagePromptNeedsExpertCacheSlots(have: Int, need: Int)

    public var description: String {
        switch self {
        case .helpRequested: return "help requested"
        case .unknownFlag(let flag): return "unknown flag: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .invalidValue(let flag, let value):
            if flag == "--expert-cache-slots", Int(value) != nil {
                let allowed = RuntimeConfiguration.allowedExpertCacheSlots.sorted()
                    .map(String.init).joined(separator: ", ")
                return "invalid value for \(flag): \(value) (use one of \(allowed))"
            }
            return "invalid value for \(flag): \(value)"
        case .requiredMissing(let flag): return "required flag missing: \(flag)"
        case .mutuallyExclusive(let a, let b): return "\(a) and \(b) are mutually exclusive"
        case .unsupported(let flag, let context):
            return "\(flag) is not supported for \(context)"
        case .modeMissing:
            return "one of --prompt, --chat-prompt or --messages-file is required"
        case .imagePromptNeedsExpertCacheSlots(let have, let need):
            return "--image requires chunked prefill, which needs at least \(need) "
                + "expert-cache slots; --expert-cache-slots \(have) cannot serve it: "
                + "raise --expert-cache-slots to \(need) or more, or drop --image"
        }
    }
}

extension Args {
    public static let usage = """
    TUFFCLI — local text generation

    usage: TUFFCLI --model <dir|apple.on-device>
           (--prompt <string> | --chat-prompt <string> | --messages-file <path>) [options]

    `tuff prompt <text> [--model <name|path>]` runs this with the text as the
    chat prompt, takes model names as well as paths, and fills in each
    model's own context, cache, temperature, top-k and top-p, so the defaults
    below apply only where neither you nor the model set them.

    Apple uses text, instructions, temperature and output limits only.
    Engine settings and explicit sampling filters are unavailable for Apple.

    required:
      --model <dir|apple>       A .gturbo directory or Apple Foundation Model.
      --prompt <string>         Raw-completion prompt.
      --chat-prompt <string>    Single-turn instruction chat; pairs with --image
      --system-prompt <string>  System message for a single-turn chat.
      --messages-file <path>    JSON chat messages with role and content fields.
      --image <path>            Attach an image; repeatable. Needs --chat-prompt
                                and an installed companion pack.
      --vision-pack <dir>       Companion pack (default beside the model).
      --vision-residency <on-demand|keep-ready>
                                Routed-expert residency during vision (default on-demand).

    options:
      --max-new <int>            Generated-token limit (default 1024).
      --max-context <int>        Context limit in tokens (default 8192).
      --memory-auto             Calculate a safe cache, projection and prefill allocation.
      --dense-residency <auto|resident|streamed>
                                Dense weights: retain what fits, all, or stage layers.
      --dense-weight-budget-mb <int>
                                Explicit dense weight allocation budget for Auto.
      --thinking <off|on>        Native chat reasoning (default off).
      --reasoning <low|medium|high>
                                GPT-OSS only: reasoning effort (default medium).
                                 Reasoning counts toward the --max-new limit
      --temperature <float>      Sampling temperature (default 0.2, or the model's
                                 setting through tuff prompt; 0 = greedy).
      --top-k <int>              Top-k truncation, 1...256 (default 64; 0 = off with --top-p 1 or --temperature 0).
      --top-p <float>            Nucleus truncation (default 0.95).
      --repetition-penalty <f>   Repetition penalty (default 1.0).
      --seed <uint64>            Deterministic sampling seed (default off).
      --stop <string>            Stop substring (repeatable).
      --quiet                    Suppress the timing footer.
      --expert-cache-slots <n>   Expert-cache slots: \(RuntimeConfiguration.allowedExpertCacheSlots.map(String.init).joined(separator: ", ")) (default 16).
      --expert-cache-policy <s>  Expert-cache policy: lfu or lru (default lfu).
      --prefill on|off           Enable or disable chunked prompt prefill (default on).
                                 Chunked prefill requires 16 or more cache slots.
      --prefill-chunk-tokens <n|auto>
                                 Prefill chunk size: 32, 64, 128, 256, 512, 1024,
                                 2048, or auto (default 128; tuff prompt uses auto
                                 up to the model's size for this Mac). Each chunk re-reads
                                 the routed expert pool, so larger chunks read
                                 less but hold more memory; auto picks the
                                 smallest size that covers the prompt.
      --prefill-chunk-max <n>    Largest size auto may pick (default 256).
      --rdadvise <s>             Read-advice policy: off, default, bounded, or adaptive (default off).
      --speculative <off|greedy|auto>
                                Opt-in greedy speculation; auto starts at two
                                tokens and adapts to measured acceptance/cost
                                (default off).
      --speculative-draft-tokens <n>
                                Candidate block size, 1...8 (default 4).
      --help                     Show this message.
    """

    /// `chunkedPrefillNeedsSlotFloor` is false for GPT-OSS, which reads each
    /// cache-sized expert group in turn and so prefills in chunks with any
    /// cache size, as `tuff serve` already runs it. Argument parsing doesn't
    /// know the model yet, so it passes false and the run checks again.
    public func resolvedRuntimeConfiguration(
        forceLogitsHead: Bool,
        imagePrompt: Bool = false,
        chunkedPrefillNeedsSlotFloor: Bool = true) throws -> RuntimeConfiguration {
        // Images reach the runtime two ways, `--image` and `image_file` parts
        // inside `--messages-file`, and only the first fills `images`. Asking
        // the caller keeps the second from surviving validation and dying in
        // the routed tile scheduler after the model load and every GPU encode.
        guard !imagePrompt
                || expertCacheSlots
                >= RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill else {
            throw ArgsError.imagePromptNeedsExpertCacheSlots(
                have: expertCacheSlots,
                need: RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill)
        }
        guard RuntimeConfiguration.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw ArgsError.invalidValue(
                flag: "--expert-cache-slots", value: "\(expertCacheSlots)")
        }
        guard RuntimeConfiguration.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw ArgsError.invalidValue(
                flag: "--prefill-chunk-tokens", value: "\(prefillChunkTokens)")
        }
        let chunkedPrefillSupported = expertCacheSlots >=
            RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        guard prefillPolicy == .off || chunkedPrefillSupported
                || !chunkedPrefillNeedsSlotFloor else {
            throw ArgsError.invalidValue(
                flag: "--expert-cache-slots",
                value: "\(expertCacheSlots) requires --prefill off")
        }
        return RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            rdadvisePolicy: rdadvisePolicy,
            prefillEnabled: prefillPolicy == .chunked,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead)
    }

    public static func parse(_ argv: [String]) throws -> Args {
        var model: String?
        var prompt: String?
        var chatPrompt: String?
        var systemPrompt: String?
        var messagesFile: String?
        var images: [String] = []
        var visionPack: String?
        var visionResidency: VisionResidencyPolicy = .defaultPolicy
        var maxNew = 1_024
        var maxContext = 8192
        var thinking: ChatReasoning = .off
        var reasoningEffort: GPTOSSReasoningEffort?
        var temperature: Float = 0.2
        var topK: Int? = 64
        var topP: Float? = 0.95
        var repetitionPenalty: Float = 1.0
        var seed: UInt64?
        var stops: [String] = []
        var quiet = false
        let runtimeDefaults = RuntimeConfiguration.production
        var expertCacheSlots = runtimeDefaults.expertCacheSlots
        var expertCachePolicy = runtimeDefaults.expertCachePolicy
        var prefillPolicy = runtimeDefaults.prefillPolicy
        var prefillChunkTokens = runtimeDefaults.prefillChunkTokens
        var prefillChunkTokensAuto = false
        var prefillChunkMax = PrefillRuntimeConfig.baselineChunkTokens
        var rdadvisePolicy = runtimeDefaults.rdadvisePolicy
        var speculativeMode: SpeculativeDecodeMode = .off
        var speculativeDraftTokens = 4
        var denseResidencyPolicy: DenseResidencyPolicy?
        var automaticMemory: Bool? = nil

        var index = 0
        var providedFlags = Set<String>()
        while index < argv.count {
            let flag = argv[index]
            providedFlags.insert(flag)
            switch flag {
            case "--memory-auto":
                automaticMemory = true
                index += 1
            case "--help":
                throw ArgsError.helpRequested
            case "--quiet":
                quiet = true
                index += 1
            case "--model":
                model = try takeValue(argv, &index, flag: flag)
            case "--dense-residency":
                let value = try takeValue(argv, &index, flag: flag)
                switch value {
                case "auto": denseResidencyPolicy = nil
                case "resident": denseResidencyPolicy = .resident
                case "streamed": denseResidencyPolicy = .bounded(residentLayers: [])
                default: throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--dense-weight-budget-mb":
                let value = try takeValue(argv, &index, flag: flag)
                guard let megabytes = UInt64(value), megabytes > 0,
                      megabytes <= UInt64.max / (1 << 20) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                denseResidencyPolicy = .automatic(weightBudgetBytes: megabytes * (1 << 20))
            case "--prompt":
                prompt = try takeValue(argv, &index, flag: flag)
            case "--chat-prompt":
                chatPrompt = try takeValue(argv, &index, flag: flag)
            case "--system-prompt":
                systemPrompt = try takeValue(argv, &index, flag: flag)
            case "--image":
                images.append(try takeValue(argv, &index, flag: flag))
            case "--vision-pack":
                visionPack = try takeValue(argv, &index, flag: flag)
            case "--vision-residency":
                let value = try takeValue(argv, &index, flag: flag)
                guard let policy = VisionResidencyPolicy(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                visionResidency = policy
            case "--messages-file":
                messagesFile = try takeValue(argv, &index, flag: flag)
            case "--max-new":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxNew = parsed
            case "--max-context":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxContext = parsed
            case "--thinking":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = ChatReasoning(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                thinking = parsed
            case "--reasoning":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = GPTOSSReasoningEffort(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                reasoningEffort = parsed
            case "--temperature":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed >= 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                temperature = parsed
            case "--top-k":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (0...256).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topK = parsed == 0 ? nil : parsed
            case "--top-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topP = parsed
            case "--repetition-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                repetitionPenalty = parsed
            case "--seed":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = UInt64(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                seed = parsed
            case "--stop":
                stops.append(try takeValue(argv, &index, flag: flag))
            case "--expert-cache-slots":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertCacheSlots = parsed
            case "--expert-cache-policy":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RuntimeExpertCachePolicy(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                expertCachePolicy = parsed
            case "--prefill":
                let value = try takeValue(argv, &index, flag: flag)
                switch value {
                case "on": prefillPolicy = .chunked
                case "off": prefillPolicy = .off
                default: throw ArgsError.invalidValue(flag: flag, value: value)
                }
            case "--prefill-chunk-tokens":
                let value = try takeValue(argv, &index, flag: flag)
                if value == "auto" {
                    prefillChunkTokensAuto = true
                    break
                }
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                prefillChunkTokensAuto = false
                prefillChunkTokens = parsed
            case "--prefill-chunk-max":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                prefillChunkMax = parsed
            case "--rdadvise":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = RDAdvicePolicyMode(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                rdadvisePolicy = parsed
            case "--speculative":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = SpeculativeDecodeMode(rawValue: value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                speculativeMode = parsed
            case "--speculative-draft-tokens":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (1...8).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                speculativeDraftTokens = parsed
            default:
                throw ArgsError.unknownFlag(flag)
            }
        }

        guard let model else { throw ArgsError.requiredMissing("--model") }
        if prompt != nil && messagesFile != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--messages-file")
        }
        if prompt != nil, chatPrompt != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--chat-prompt")
        }
        if messagesFile != nil, chatPrompt != nil {
            throw ArgsError.mutuallyExclusive("--chat-prompt", "--messages-file")
        }
        if prompt != nil, !images.isEmpty {
            throw ArgsError.mutuallyExclusive("--prompt", "--image")
        }
        if prompt != nil, thinking != .off {
            throw ArgsError.mutuallyExclusive("--prompt", "--thinking")
        }
        if prompt != nil, reasoningEffort != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--reasoning")
        }
        if systemPrompt != nil, chatPrompt == nil {
            throw ArgsError.requiredMissing("--chat-prompt")
        }
        if thinking != .off, reasoningEffort != nil {
            throw ArgsError.mutuallyExclusive("--thinking", "--reasoning")
        }
        let minimumSlots = RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        if !images.isEmpty, expertCacheSlots < minimumSlots {
            throw ArgsError.imagePromptNeedsExpertCacheSlots(
                have: expertCacheSlots, need: minimumSlots)
        }
        if !images.isEmpty, chatPrompt == nil {
            // Naming the flag the user actually left out. Reporting a conflict
            // with `--messages-file` sent them looking for a flag they never
            // passed, in the most likely first-run mistake there is.
            throw messagesFile == nil
                ? ArgsError.requiredMissing("--chat-prompt")
                : ArgsError.mutuallyExclusive("--image", "--messages-file")
        }
        if prompt == nil && chatPrompt == nil && messagesFile == nil {
            throw ArgsError.modeMissing
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw ArgsError.invalidValue(
                flag: "--top-p",
                value: "\(topP) requires --top-k between 1 and 256")
        }
        if ["apple", "apple.on-device"].contains(model) {
            let supported: Set<String> = ["--model", "--prompt", "--chat-prompt", "--system-prompt",
                "--messages-file", "--max-new", "--temperature", "--quiet"]
            if let flag = providedFlags.subtracting(supported).sorted().first {
                throw ArgsError.invalidValue(flag: flag, value: "unsupported by Apple Foundation Model")
            }
        }
        var arguments = Args(model: model,
                             prompt: prompt,
                             chatPrompt: chatPrompt,
                             systemPrompt: systemPrompt,
                             messagesFile: messagesFile,
                             images: images,
                             visionPack: visionPack,
                             visionResidency: visionResidency,
                             maxNew: maxNew,
                             maxContext: maxContext,
                             thinking: thinking,
                             reasoningEffort: reasoningEffort,
                             temperature: temperature,
                             topK: topK,
                             topP: topP,
                             repetitionPenalty: repetitionPenalty,
                             seed: seed,
                             stops: stops,
                             quiet: quiet,
                             expertCacheSlots: expertCacheSlots,
                             expertCachePolicy: expertCachePolicy,
                             prefillPolicy: prefillPolicy,
                             prefillChunkTokens: prefillChunkTokens,
                             prefillChunkTokensAuto: prefillChunkTokensAuto,
                             prefillChunkMax: prefillChunkMax,
                             rdadvisePolicy: rdadvisePolicy,
                             speculativeMode: speculativeMode,
                             speculativeDraftTokens: speculativeDraftTokens,
                             denseResidencyPolicy: denseResidencyPolicy)
        arguments.automaticMemory = automaticMemory
        _ = try arguments.resolvedRuntimeConfiguration(
            forceLogitsHead: false, chunkedPrefillNeedsSlotFloor: false)
        return arguments
    }

    private static func takeValue(_ argv: [String],
                                  _ index: inout Int,
                                  flag: String) throws -> String {
        guard index + 1 < argv.count else { throw ArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        // `--max-new --max-context 4096` meant a forgotten value, not a value
        // of "--max-context". Text flags may legitimately start with dashes.
        let takesText = ["--prompt", "--chat-prompt", "--system-prompt", "--stop"]
        if value.hasPrefix("--"), !takesText.contains(flag) {
            throw ArgsError.missingValue(flag: flag)
        }
        index += 2
        return value
    }
}
