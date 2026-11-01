import TUFFInferenceTypes
import Foundation
import Hub
import Tokenizers

public enum GFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate
    case missingTokenizerConfig
    case unsupportedDecoder(actual: String)
    case invalidTokenID(token: String, id: Int)
    case unsupportedForDialect(String)
    case specialTokenMismatch(token: String, expected: Int32, resolved: Int32)

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        case .missingTokenizerConfig:
            return "tokenizer_config.json is missing or unreadable"
        case .unsupportedDecoder(let actual):
            return "tokenizer decoder is not the pinned Gemma 4 sequence "
                + "Sequence[Replace(▁→␣), ByteFallback, Fuse]; found: \(actual)"
        case .invalidTokenID(let token, let id):
            return "tokenizer declares out-of-range ID \(id) for token \(token)"
        case .unsupportedForDialect(let operation):
            return "operation is not supported for this tokenizer's chat dialect: \(operation)"
        case .specialTokenMismatch(let token, let expected, let resolved):
            return "tokenizer resolves \(token) to \(resolved), but the runtime "
                + "expects \(expected); the tokenizer does not match this build"
        }
    }
}

/// Chat framing dialect, resolved from the loaded tokenizer's special tokens.
///
/// Harmony is detected first by `<|start|>`, then ChatML by `<|im_end|>`;
/// everything else uses the Gemma 4 contract.
public typealias ChatDialect = TUFFRuntimeDescriptions.ChatDialect

/// Model-supported reasoning switch used by chat prompt renderers.
///
/// GPT-OSS adds graded effort in its own family integration. Gemma 4 and Qwen
/// expose a true on/off template switch, so keeping this type narrow prevents a
/// caller from silently mapping `low` or `high` onto a model that cannot honor
/// that distinction.
public typealias ChatReasoning = TUFFInferenceTypes.ChatReasoning

/// Detokenization pipeline declared by `tokenizer.json`. Resolved from the
/// decoder declaration rather than from the chat dialect, because it is the
/// decoder — not the chat framing — that decides how token text becomes bytes.
public enum TokenDecoding: String, Sendable {
    /// `Sequence[Replace("▁" -> " "), ByteFallback, Fuse]` — see `GemmaDecoding`.
    case gemmaByteFallback
    /// GPT-2 `bytes_to_unicode` — see `ByteLevelDecoding`.
    case byteLevel
}

/// Tokenizer wrapper for Gemma 4, ChatML/Qwen, and Harmony/GPT-OSS.
///
/// Prefers tokenizer sidecars in a completed `.gturbo/tokenizer/` directory,
/// then falls back to the IT variant's Hugging Face Hub tokenizer cache. Exposes
/// typed accessors for the IDs the generator actually needs (BOS / EOS / pad /
/// end-of-turn) and adapts encode/decode to Int32 to match the buffer types
/// kernels consume.
///
/// TUFF owns the minimal chat framing because the upstream
/// `tokenizer_config.json` has no `chat_template`. Literal control-token text in
/// user content is accepted as a trusted-input research-runtime limitation.
public struct GFTokenizer: @unchecked Sendable {
    public static let modelID = "google/gemma-4-26B-A4B-it"
    public static let chatTemplateIdentity = "gemma4-it-text-no-tools-v1"
    public static let toolChatTemplateIdentity = "gemma4-it-tools-jinja-v2-schema-unions"
    public static let harmonyChatTemplateIdentity = "gpt-oss-harmony-v2-native-headers"
    public static let minimaxChatTemplateIdentity = "minimax-m2.7-v2-native-tool-xml"

    public let dialect: ChatDialect
    /// Decoder pipeline declared by the installed `tokenizer.json`.
    public let decoding: TokenDecoding
    /// Nominal BOS. For ChatML this is `<|endoftext|>` (the config's unused
    /// `bos_token_id`); it is never prepended — see `encode(_:addBOS:)`.
    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let endOfTurnID: Int32
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseID: Int32
    public let toolResponseEndID: Int32
    /// For ChatML these alias the `<think>` / `</think>` markers, the dialect's
    /// closest analog of Gemma's hidden-channel delimiters.
    public let channelStartID: Int32
    public let channelEndID: Int32
    /// ChatML `<think>` / `</think>` special-token IDs; nil for Gemma.
    public let thinkStartID: Int32?
    public let thinkEndID: Int32?
    public let harmonyTokenIDs: HarmonySpecialTokenIDs?
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int
    /// The channel/tool markers that structure assistant output. Streaming
    /// treats them as detokenizer barriers: a byte-fallback run must not span
    /// one, or its text would surface after the marker and be routed under the
    /// wrong channel state.
    public var structuralMarkerIDs: Set<Int32> {
        var ids: Set<Int32> = [
            toolCallStartID, toolCallEndID, toolResponseID, toolResponseEndID,
            channelStartID, channelEndID,
        ]
        if let harmonyTokenIDs {
            ids.formUnion(harmonyTokenIDs.structuralMarkerIDs)
        }
        return ids
    }
    /// IDs that `decode(skipSpecialTokens: true)` strips — the
    /// `added_tokens[special == true]` set from `tokenizer.json`, identical to
    /// the filter the library's own decode applies before its decoder chain.
    let specialTokenIDs: Set<Int32>

    /// BOS actually prepended by `encode(_:addBOS:)`; nil for dialects that
    /// never use a BOS prefix (ChatML).
    private let bosPrefixID: Int32?
    private let gemmaSchemaTemplate: String?
    private let memoizesToolTemplate: Bool
    private let encodingCache = TokenizerEncodingCache()

    var encodingCacheStatistics: TokenizerEncodingCache.Statistics { encodingCache.statistics }

    @usableFromInline
    let tokenizer: any Tokenizer

    public static func load() async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.pretrained(modelID))
    }

    public static func load(from folder: URL) async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.local(folder.standardizedFileURL.path))
    }

    public static func load(forModelDirectory modelDirectory: URL,
                            environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> GFTokenizer {
        if let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) {
            return try await load(from: folder)
        }
        return try await load()
    }

    public static func tokenizerFolder(forModelDirectory modelDirectory: URL,
                                       environment: [String: String] = ProcessInfo.processInfo.environment,
                                       fileManager: FileManager = .default) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }

        guard let override = environment["TUFF_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    static func loadUncached(pretrained modelID: String = Self.modelID) async throws -> GFTokenizer {
        try await make(from: LanguageModelConfigurationFromHub(modelName: modelID))
    }

    static func loadUncached(from folder: URL) async throws -> GFTokenizer {
        try await make(from: LanguageModelConfigurationFromHub(modelFolder: folder))
    }

    /// Build from the raw tokenizer configs so the decoder pipeline and the
    /// special-token set can be validated against `tokenizer.json` itself.
    /// Same fetch/caching path `AutoTokenizer.from(pretrained:)` uses internally.
    private static func make(from hub: LanguageModelConfigurationFromHub) async throws -> GFTokenizer {
        guard let declaredConfig = try await hub.tokenizerConfig else {
            throw GFTokenizerError.missingTokenizerConfig
        }
        let tokenizerConfig = registeredTokenizerClass(declaredConfig)
        let tokenizerData = try await hub.tokenizerData
        let underlying = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        return try GFTokenizer(tokenizer: underlying, tokenizerData: tokenizerData,
                               chatTemplate: tokenizerConfig.chatTemplate.string())
    }

    /// Ornith 1.5 122B names its tokenizer `Qwen3_5Tokenizer`, which the
    /// library doesn't register and refuses in strict mode. It is the same
    /// byte-level BPE as `Qwen2Tokenizer`, so load it under that name.
    static func registeredTokenizerClass(_ config: Config) -> Config {
        guard config.tokenizerClass.string() == "Qwen3_5Tokenizer",
              var values = config.dictionary() else { return config }
        values["tokenizer_class"] = Config("Qwen2Tokenizer")
        return Config(values)
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    /// Reject a tokenizer whose declared decoder is not the pinned Gemma 4
    /// sequence.
    ///
    /// `GemmaDecoding` reproduces `Sequence[Replace("▁" -> " "), ByteFallback,
    /// Fuse]` rather than calling `Tokenizers.decode`, so that decode stays
    /// lossless and per-token (see `GemmaDecoding`). The installed tokenizer is
    /// pinned and hash-validated, but `TUFF_TOKENIZER_DIR` can point
    /// at any directory. Without this check a tokenizer declaring a different
    /// decoder would decode subtly wrong text instead of failing.
    ///
    /// The check reads `tokenizer.json`'s decoder declaration structurally, so
    /// it rejects any foreign decoder — including one whose behavior happens to
    /// coincide on a handful of probe strings — without asserting the library's
    /// exact runtime output, which a benign dependency bump may change.
    /// Behavioral agreement with the library is pinned by the differential
    /// tests instead.
    @discardableResult
    static func verifyDecoderConfiguration(_ tokenizerData: Config) throws -> TokenDecoding {
        let decoder = tokenizerData["decoder"]
        // ChatML/Qwen declares a bare ByteLevel decoder, reproduced by
        // `ByteLevelDecoding` on the same terms.
        if decoder.type.string() == "ByteLevel" { return .byteLevel }
        let steps = decoder.decoders.array(or: [])
        guard decoder.type.string() == "Sequence",
              steps.count == 3,
              steps[0].type.string() == "Replace",
              steps[0].pattern.String.string() == "▁",
              steps[0].content.string() == " ",
              steps[1].type.string() == "ByteFallback",
              steps[2].type.string() == "Fuse"
        else {
            throw GFTokenizerError.unsupportedDecoder(actual: decoder.description)
        }
        return .gemmaByteFallback
    }

    /// Resolve a special token to its ID, rejecting `<unk>` substitution.
    ///
    /// BPE's `convertTokenToId` returns the unknown-token ID — not `nil` — for
    /// a token absent from the vocab, so a plain `guard let` never fires and a
    /// missing marker would silently bind to `<unk>`, colliding with every
    /// other missing marker. The round-trip through `convertIdToToken` detects
    /// any substitution.
    static func requireTokenID(_ tokenizer: any Tokenizer, _ token: String) throws -> Int32 {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else {
            throw GFTokenizerError.missingSpecialToken(token)
        }
        return try int32ID(token, id)
    }

    /// Config-supplied IDs are full-range `Int`; a trapping `Int32(_:)` would
    /// crash on a corrupt tokenizer instead of taking the typed error path
    /// every other malformed-config case uses.
    private static func int32ID(_ token: String, _ id: Int) throws -> Int32 {
        guard let value = Int32(exactly: id) else {
            throw GFTokenizerError.invalidTokenID(token: token, id: id)
        }
        return value
    }

    public init(tokenizer: any Tokenizer, tokenizerData: Config,
                chatTemplate: String? = nil) throws {
        self.tokenizer = tokenizer
        gemmaSchemaTemplate = chatTemplate.flatMap(GemmaSchemaTemplate.extending)
        memoizesToolTemplate = TokenizerEncodingCache.permitsTemplateMemoization(chatTemplate)
        self.decoding = try Self.verifyDecoderConfiguration(tokenizerData)

        // The same `added_tokens[special == true]` ID set the library's
        // `decode(skipSpecialTokens: true)` filters before running its decoder.
        var specials: Set<Int32> = []
        for added in tokenizerData["addedTokens"].array(or: []) {
            guard added["special"].boolean(or: false),
                  let id = added["id"].integer() else { continue }
            try specials.insert(Self.int32ID(added.content.string() ?? "added token", id))
        }
        self.specialTokenIDs = specials

        let dialect: ChatDialect
        let resolved: ResolvedSpecialTokens
        if Self.specialTokenID(tokenizer, "<|begin_of_text|>") != nil,
           Self.specialTokenID(tokenizer, "<|observation|>") != nil {
            dialect = .flint
            resolved = try Self.resolveFlintTokens(tokenizer)
        } else if Self.specialTokenID(tokenizer, Self.harmonyStartMark) != nil {
            dialect = .harmony
            resolved = try Self.resolveHarmonyTokens(tokenizer)
        } else if Self.specialTokenID(tokenizer, Self.minimaxBOSMark) != nil {
            dialect = .minimax
            resolved = try Self.resolveMiniMaxTokens(tokenizer)
        } else if Self.specialTokenID(tokenizer, Self.imEndMark) != nil {
            dialect = .chatml
            resolved = try Self.resolveChatMLTokens(tokenizer)
        } else {
            dialect = .gemma
            resolved = try Self.resolveGemmaTokens(tokenizer)
        }

        self.dialect = dialect
        self.bosID = resolved.bosID
        self.bosPrefixID = resolved.bosPrefixID
        self.eosID = resolved.eosID
        self.padID = resolved.padID
        self.endOfTurnID = resolved.endOfTurnID
        self.toolCallStartID = resolved.toolCallStartID
        self.toolCallEndID = resolved.toolCallEndID
        self.toolResponseID = resolved.toolResponseID
        self.toolResponseEndID = resolved.toolResponseEndID
        self.channelStartID = resolved.channelStartID
        self.channelEndID = resolved.channelEndID
        self.thinkStartID = resolved.thinkStartID
        self.thinkEndID = resolved.thinkEndID
        self.harmonyTokenIDs = resolved.harmonyTokenIDs
        self.stopTokenIDs = resolved.stopTokenIDs
        self.vocabSize = resolved.vocabSize
    }

    private struct ResolvedSpecialTokens {
        let bosID: Int32
        let bosPrefixID: Int32?
        let eosID: Int32
        let padID: Int32
        let endOfTurnID: Int32
        let toolCallStartID: Int32
        let toolCallEndID: Int32
        let toolResponseID: Int32
        let toolResponseEndID: Int32
        let channelStartID: Int32
        let channelEndID: Int32
        let thinkStartID: Int32?
        let thinkEndID: Int32?
        let harmonyTokenIDs: HarmonySpecialTokenIDs?
        let stopTokenIDs: Set<Int32>
        let vocabSize: Int
    }

    private static func resolveFlintTokens(_ tokenizer: any Tokenizer) throws -> ResolvedSpecialTokens {
        func id(_ value: String) throws -> Int32 { try requireTokenID(tokenizer, value) }
        let bos = try id("<|begin_of_text|>"), eos = try id("<|end_of_text|>")
        let user = try id("<|user|>"), observation = try id("<|observation|>")
        let developer = try id("<|developer|>")
        _ = try id("<|system|>"); _ = try id("<|assistant|>")
        let think = try id("<think>"), endThink = try id("</think>")
        return ResolvedSpecialTokens(bosID: bos, bosPrefixID: nil, eosID: eos, padID: eos,
            endOfTurnID: user, toolCallStartID: try id("<tool_call>"),
            toolCallEndID: try id("</tool_call>"), toolResponseID: try id("<tool_response>"),
            toolResponseEndID: try id("</tool_response>"), channelStartID: think, channelEndID: endThink,
            thinkStartID: think, thinkEndID: endThink, harmonyTokenIDs: nil,
            stopTokenIDs: [eos, user, observation, developer], vocabSize: 150144)
    }

    private static func resolveGemmaTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        guard let bos = tokenizer.bosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<bos>")
        }
        guard let eos = tokenizer.eosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<eos>")
        }
        // `requireTokenID` round-trips each marker through `convertIdToToken`;
        // a plain `convertTokenToId` would silently bind a missing marker to
        // `<unk>` and collide every missing marker onto the same ID.
        let bosID = try Self.int32ID("<bos>", bos)
        let eosID = try Self.int32ID("<eos>", eos)
        let pad = try Self.requireTokenID(tokenizer, "<pad>")
        let eot = try Self.requireTokenID(tokenizer, "<turn|>")
        let toolCallStart = try Self.requireTokenID(tokenizer, "<|tool_call>")
        let toolCallEnd = try Self.requireTokenID(tokenizer, "<tool_call|>")
        let toolResponse = try Self.requireTokenID(tokenizer, "<|tool_response>")
        let toolResponseEnd = try Self.requireTokenID(tokenizer, "<tool_response|>")
        let channelStart = try Self.requireTokenID(tokenizer, "<|channel>")
        let channelEnd = try Self.requireTokenID(tokenizer, "<channel|>")
        // Multimodal prompt layout uses compile-time marker IDs, so verify this
        // Gemma tokenizer resolves the same vocabulary positions before any
        // image prompt can be constructed.
        for (token, expected) in [
            ("<|image>", MultimodalPromptRenderer.beginImageTokenID),
            ("<|image|>", MultimodalPromptRenderer.imageTokenID),
            ("<image|>", MultimodalPromptRenderer.endImageTokenID),
        ] {
            let resolved = try Self.requireTokenID(tokenizer, token)
            guard resolved == expected else {
                throw GFTokenizerError.specialTokenMismatch(
                    token: token, expected: expected, resolved: resolved)
            }
        }
        return ResolvedSpecialTokens(
            bosID: bosID,
            bosPrefixID: bosID,
            eosID: eosID,
            padID: pad,
            endOfTurnID: eot,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: channelStart,
            channelEndID: channelEnd,
            thinkStartID: nil,
            thinkEndID: nil,
            harmonyTokenIDs: nil,
            stopTokenIDs: [eosID, eot, toolResponse],
            vocabSize: 262_144)
    }

    /// Resolves a token string to its ID, rejecting the unk-token fallback
    /// some tokenizers substitute for out-of-vocabulary strings.
    private static func specialTokenID(_ tokenizer: any Tokenizer, _ token: String) -> Int? {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else { return nil }
        return id
    }

    private static func resolveChatMLTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw GFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        // `<|im_start|>` is required even though no stored property holds it;
        // template rendering relies on the tokenizer recognizing its text.
        _ = try id(Self.imStartMark)
        let imEnd = try id(Self.imEndMark)
        let endOfText = try id("<|endoftext|>")
        let toolCallStart = try id("<tool_call>")
        let toolCallEnd = try id("</tool_call>")
        let toolResponse = try id("<tool_response>")
        let toolResponseEnd = try id("</tool_response>")
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        return ResolvedSpecialTokens(
            bosID: endOfText,
            bosPrefixID: nil,
            eosID: endOfText,
            padID: endOfText,
            endOfTurnID: imEnd,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            harmonyTokenIDs: nil,
            stopTokenIDs: [imEnd, endOfText],
            // The model's padded embedding/lm_head row count, not the
            // tokenizer's actual vocab (248 077) — logits buffers use this.
            vocabSize: 248_320)
    }

    private static func resolveHarmonyTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            try Self.requireTokenID(tokenizer, token)
        }
        let resolved = HarmonySpecialTokenIDs(
            startOfText: try id("<|startoftext|>"),
            endOfText: try id("<|endoftext|>"),
            return: try id("<|return|>"),
            constrain: try id("<|constrain|>"),
            channel: try id("<|channel|>"),
            start: try id(Self.harmonyStartMark),
            end: try id("<|end|>"),
            message: try id("<|message|>"),
            call: try id("<|call|>"))
        let expected = HarmonySpecialTokenIDs.official
        for (token, actual, pinned) in [
            ("<|startoftext|>", resolved.startOfText, expected.startOfText),
            ("<|endoftext|>", resolved.endOfText, expected.endOfText),
            ("<|return|>", resolved.return, expected.return),
            ("<|constrain|>", resolved.constrain, expected.constrain),
            ("<|channel|>", resolved.channel, expected.channel),
            (Self.harmonyStartMark, resolved.start, expected.start),
            ("<|end|>", resolved.end, expected.end),
            ("<|message|>", resolved.message, expected.message),
            ("<|call|>", resolved.call, expected.call),
        ] where actual != pinned {
            throw GFTokenizerError.specialTokenMismatch(
                token: token, expected: pinned, resolved: actual)
        }
        return ResolvedSpecialTokens(
            bosID: resolved.startOfText,
            bosPrefixID: nil,
            eosID: resolved.return,
            padID: resolved.endOfText,
            endOfTurnID: resolved.end,
            toolCallStartID: resolved.start,
            toolCallEndID: resolved.call,
            // The raw generation loop uses this generic slot to classify a
            // stop boundary as a tool call. Harmony's `<|call|>` both closes
            // the payload and stops generation.
            toolResponseID: resolved.call,
            toolResponseEndID: resolved.end,
            channelStartID: resolved.channel,
            channelEndID: resolved.message,
            thinkStartID: nil,
            thinkEndID: nil,
            harmonyTokenIDs: resolved,
            stopTokenIDs: [resolved.return, resolved.call, resolved.endOfText],
            // Both official checkpoints pad embedding and unembedding rows to
            // this size; the actual o200k Harmony vocabulary ends at 200018.
            vocabSize: 201_088)
    }

    private static func resolveMiniMaxTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            try Self.requireTokenID(tokenizer, token)
        }
        let bos = try id(Self.minimaxBOSMark)
        _ = try id(Self.minimaxTurnMark)
        let eos = try id(Self.minimaxEOSMark)
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        let toolStart = try id("<minimax:tool_call>")
        let toolEnd = try id("</minimax:tool_call>")
        return ResolvedSpecialTokens(
            bosID: bos,
            bosPrefixID: nil,
            eosID: eos,
            padID: eos,
            endOfTurnID: eos,
            toolCallStartID: toolStart,
            toolCallEndID: toolEnd,
            toolResponseID: eos,
            toolResponseEndID: eos,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            harmonyTokenIDs: nil,
            stopTokenIDs: [eos],
            vocabSize: 200_064)
    }

    /// Encode UTF-8 text to token IDs. `addBOS = true` prepends `<bos>`.
    ///
    /// The library's `addSpecialTokens: true` flag is a no-op for the Gemma 4 IT
    /// tokenizer (its config has `add_bos_token = false`; BOS is expected to come
    /// from the chat template). We prepend manually so the kernel-facing API stays
    /// the same regardless of upstream defaults. ChatML has no BOS, so `addBOS`
    /// is a no-op for that dialect.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        let base = encodingCache.value(for: TokenizerEncodingCache.textKey(text)) {
            tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        }
        guard addBOS, let bosPrefixID else { return base }
        return [bosPrefixID] + base
    }

    /// Decode token IDs to text. `skipSpecialTokens` strips BOS/EOS/turn markers from the output.
    ///
    /// Runs the pinned Gemma decoder sequence directly (see `GemmaDecoding`)
    /// rather than `Tokenizers.decode`, whose trailing
    /// `clean_up_tokenization_spaces` pass defaults to on for this tokenizer and
    /// rewrites model output (`"he said ' ok ' now"` -> `"he said'ok'now"`),
    /// breaking `decode(encode(x)) == x`. Batch decode is a push-loop over
    /// `GFDetokenizer`, so batch and streaming decode agree by construction.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        var detok = GFDetokenizer(tokenizer: self, skipSpecialTokens: skipSpecialTokens)
        var text = ""
        for id in ids {
            text += detok.push(id)
        }
        return text + detok.flush()
    }

    // MARK: - Chat template

    public typealias Role = TUFFInferenceTypes.InferenceChatRole
    public typealias Message = TUFFInferenceTypes.InferenceChatMessage
    public typealias HistoricalToolCall = InferenceToolCall
    public typealias FunctionDefinition = InferenceFunctionDefinition

    /// Text-only, no-tool rendering of the pinned checkpoints' bundled
    /// `chat_template.jinja`. Keeping this narrow makes unsupported tool/media
    /// behavior explicit instead of approximating it.
    private static let turnOpen    = "<|turn>"
    private static let turnClose   = "<turn|>"
    private static let bosMark     = "<bos>"
    private static let imStartMark = "<|im_start|>"
    private static let imEndMark   = "<|im_end|>"
    private static let harmonyStartMark = "<|start|>"
    private static let minimaxBOSMark = "]~!b["
    private static let minimaxTurnMark = "]~b]"
    private static let minimaxEOSMark = "[e~["
    /// Generation prompt with thinking disabled, matching the Jinja template's
    /// `add_generation_prompt` + `enable_thinking=false` branch.
    private static let chatMLThinkingOffSuffix =
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    private static let chatMLThinkingOnSuffix =
        "<|im_start|>assistant\n<think>\n"

    public func applyChatTemplate(
        _ messages: [Message],
        modelVariant: ModelVariant? = nil,
        reasoning: ChatReasoning = .off
    ) throws -> String {
        try applyChatTemplate(
            messages,
            modelVariant: modelVariant,
            reasoning: reasoning,
            preserveThinking: false)
    }

    public func applyChatTemplate(
        _ messages: [Message],
        modelVariant: ModelVariant? = nil,
        reasoning: ChatReasoning = .off,
        preserveThinking: Bool
    ) throws -> String {
        switch dialect {
        case .gemma:
            return try gemmaChatTemplate(
                messages, modelVariant: modelVariant, reasoning: reasoning)
        case .chatml:
            return try chatMLChatTemplate(
                messages,
                reasoning: reasoning,
                preserveThinking: preserveThinking)
        case .harmony:
            throw GFTokenizerError.unsupportedForDialect(
                "binary reasoning control for Harmony")
        case .minimax:
            return try minimaxChatTemplate(
                messages, preserveThinking: preserveThinking)
        case .flint:
            return try FlintPrompt.render(messages: messages, tools: [],
                preserveThinking: preserveThinking, addGenerationPrompt: true)
        }
    }

    public func applyHarmonyChatTemplate(
        _ messages: [Message],
        tools: [FunctionDefinition] = [],
        reasoningEffort: GPTOSSReasoningEffort = .medium,
        currentDate: String,
        modelIdentity: String = HarmonyPromptRenderer.defaultModelIdentity
    ) throws -> String {
        guard dialect == .harmony else {
            throw GFTokenizerError.unsupportedForDialect("Harmony prompt rendering")
        }
        return try HarmonyPromptRenderer().render(
            messages: messages,
            tools: tools,
            reasoningEffort: reasoningEffort,
            currentDate: currentDate,
            modelIdentity: modelIdentity)
    }

    public func encodeHarmonyChat(
        messages: [Message],
        tools: [FunctionDefinition] = [],
        reasoningEffort: GPTOSSReasoningEffort = .medium,
        currentDate: String,
        modelIdentity: String = HarmonyPromptRenderer.defaultModelIdentity
    ) throws -> [Int32] {
        encode(
            try applyHarmonyChatTemplate(
                messages,
                tools: tools,
                reasoningEffort: reasoningEffort,
                currentDate: currentDate,
                modelIdentity: modelIdentity),
            addBOS: false)
    }

    private func gemmaChatTemplate(
        _ messages: [Message],
        modelVariant: ModelVariant?,
        reasoning: ChatReasoning
    ) throws -> String {
        var s = Self.bosMark
        var loopMessages = messages[...]

        // Both pinned Gemma 4 templates enable native thinking by placing the
        // control token at the top of the first system turn. If the caller
        // supplied a system/developer message, the token and content share that
        // turn; otherwise the renderer creates a control-only system turn.
        if reasoning == .on {
            s += Self.turnOpen + "system\n<|think|>\n"
            if let first = messages.first,
               first.role == .system || first.role == .developer {
                guard let rawContent = first.content else {
                    throw GFTokenizerError.invalidChatTemplate(
                        "text-only messages require content")
                }
                s += rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
                loopMessages = messages.dropFirst()
            }
            s += Self.turnClose + "\n"
        }

        for (index, message) in loopMessages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if (message.role == .system || message.role == .developer)
                && (index != 0 || reasoning == .on) {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            let role = message.role == .assistant ? "model" : message.role.rawValue
            s += Self.turnOpen + role + "\n" + content + Self.turnClose + "\n"
        }
        s += Self.turnOpen + "model\n"
            + Self.gemmaGenerationSuffix(modelVariant: modelVariant, reasoning: reasoning)
        return s
    }

    /// What follows the open model turn. The 26B and 12B templates close an
    /// empty thought channel there when thinking is off; the E2B and E4B
    /// templates add nothing, and those models answer in plain text after an
    /// unexpected empty channel, narrating their reasoning as the answer. An
    /// absent variant keeps the 26B suffix, the v1 behavior.
    static func gemmaGenerationSuffix(modelVariant: ModelVariant?,
                                      reasoning: ChatReasoning) -> String {
        guard reasoning == .off else { return "" }
        switch modelVariant {
        case .gemma4_E2B, .gemma4_E4B: return ""
        default: return "<|channel>thought\n<channel|>"
        }
    }

    private func chatMLChatTemplate(
        _ messages: [Message],
        reasoning: ChatReasoning,
        preserveThinking: Bool
    ) throws -> String {
        var s = ""
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            var content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            if preserveThinking, message.role == .assistant,
               let thinking = message.thinking?
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !thinking.isEmpty {
                content = "<think>\n\(thinking)\n</think>\n\n" + content
            }
            s += Self.imStartMark + message.role.rawValue + "\n" + content + Self.imEndMark + "\n"
        }
        s += reasoning == .on
            ? Self.chatMLThinkingOnSuffix
            : Self.chatMLThinkingOffSuffix
        return s
    }

    private func minimaxChatTemplate(
        _ messages: [Message],
        preserveThinking: Bool
    ) throws -> String {
        var remaining = messages[...]
        let system: String
        if let first = messages.first, first.role == .system {
            guard let content = first.content else {
                throw GFTokenizerError.invalidChatTemplate("system message requires content")
            }
            system = content
            remaining = messages.dropFirst()
        } else {
            system = "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
        }
        let lastUser = remaining.lastIndex { $0.role == .user }
        var rendered = Self.minimaxBOSMark + Self.minimaxTurnMark
            + "system\n" + system + Self.minimaxEOSMark + "\n"
        for index in remaining.indices {
            let message = remaining[index]
            guard let content = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            switch message.role {
            case .user:
                rendered += Self.minimaxTurnMark + "user\n" + content
                    + Self.minimaxEOSMark + "\n"
            case .assistant:
                rendered += Self.minimaxTurnMark + "ai\n"
                if preserveThinking, index > (lastUser ?? remaining.startIndex),
                   let thinking = message.thinking, !thinking.isEmpty {
                    rendered += "<think>\n" + thinking + "\n</think>\n\n"
                }
                rendered += content + Self.minimaxEOSMark + "\n"
            case .system, .developer:
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            case .tool:
                throw GFTokenizerError.invalidChatTemplate(
                    "tool messages require the installed MiniMax chat template")
            }
        }
        return rendered + Self.minimaxTurnMark + "ai\n<think>\n"
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition],
                               reasoning: ChatReasoning = .off) throws -> [Int32] {
        try encodeToolChat(
            messages: messages,
            tools: tools,
            reasoning: reasoning,
            preserveThinking: false)
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition],
                               reasoning: ChatReasoning = .off,
                               preserveThinking: Bool,
                               addGenerationPrompt: Bool = true) throws -> [Int32] {
        if dialect == .flint {
            let prompt = try FlintPrompt.render(messages: messages, tools: tools,
                preserveThinking: preserveThinking, addGenerationPrompt: addGenerationPrompt)
            return encode(prompt, addBOS: false)
        }
        guard dialect != .harmony else {
            throw GFTokenizerError.unsupportedForDialect(
                "binary reasoning control for Harmony tool chat")
        }
        guard tokenizer.hasChatTemplate else {
            throw GFTokenizerError.missingToolTemplate
        }
        let extendsSchema = dialect == .gemma && tools.contains(where: {
            GemmaSchemaTemplate.needsExtension($0.parameters)
        })
        if extendsSchema, gemmaSchemaTemplate == nil {
            throw GFTokenizerError.invalidChatTemplate(
                "the installed Gemma template cannot render schema unions")
        }
        let upstreamMessages: [Tokenizers.Message] = try messages.map { message in
            var value: Tokenizers.Message = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            if !message.toolCalls.isEmpty {
                value["tool_calls"] = try message.toolCalls.map { call -> [String: any Sendable] in
                    [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": try call.arguments.jinjaSendableValue(),
                        ] as [String: any Sendable],
                    ]
                }
            }
            if let toolCallID = message.toolCallID { value["tool_call_id"] = toolCallID }
            if let name = message.name { value["name"] = name }
            if let thinking = message.thinking {
                value["reasoning_content"] = thinking
            }
            return value
        }
        let upstreamTools: [ToolSpec] = try tools.map { tool in
            [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": try tool.parameters.jinjaSendableValue(),
                ] as [String: any Sendable],
            ]
        }
        let identity = ToolChatEncodingIdentity(
            messages: messages, tools: tools, reasoning: reasoning,
            preserveThinking: preserveThinking, addGenerationPrompt: addGenerationPrompt,
            extendsSchema: extendsSchema)
        return try encodingCache.value(for: memoizesToolTemplate ? identity.key : nil) {
            try tokenizer.applyChatTemplate(
            messages: upstreamMessages,
            chatTemplate: extendsSchema ? gemmaSchemaTemplate.map(ChatTemplateArgument.literal) : nil,
            addGenerationPrompt: addGenerationPrompt,
            truncation: false,
            maxLength: nil,
            tools: upstreamTools,
            additionalContext: [
                "enable_thinking": reasoning == .on,
                "preserve_thinking": preserveThinking,
            ]
            ).map(Int32.init)
        }
    }

    public func encodeTextContinuation(
        userContent: String,
        modelVariant: ModelVariant? = nil,
        reasoning: ChatReasoning = .off
    ) -> [Int32] {
        let content = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        switch dialect {
        case .gemma:
            let suffix = Self.gemmaGenerationSuffix(modelVariant: modelVariant,
                                                    reasoning: reasoning)
            return [endOfTurnID] + encode(
                "\n\(Self.turnOpen)user\n\(content)\(Self.turnClose)\n"
                    + "\(Self.turnOpen)model\n" + suffix,
                addBOS: false)
        case .chatml:
            return [endOfTurnID] + encode(
                "\n\(Self.imStartMark)user\n\(content)\(Self.imEndMark)\n"
                    + (reasoning == .on
                        ? Self.chatMLThinkingOnSuffix
                        : Self.chatMLThinkingOffSuffix),
                addBOS: false)
        case .flint:
            return [endOfTurnID] + encode(FlintPrompt.safe(content) + "<|assistant|>", addBOS: false)
        case .harmony:
            return [eosID] + encode(
                "\(Self.harmonyStartMark)user<|message|>\(content)<|end|>"
                    + "\(Self.harmonyStartMark)assistant",
                addBOS: false)
        case .minimax:
            return [eosID] + encode(
                "\n\(Self.minimaxTurnMark)user\n\(content)\(Self.minimaxEOSMark)\n"
                    + "\(Self.minimaxTurnMark)ai\n<think>\n",
                addBOS: false)
        }
    }

    /// A user turn carrying images, encoded as a continuation onto an existing
    /// KV prefix. Mirrors `encodeTextContinuation` exactly and then expands each
    /// image marker the way `MultimodalPromptRenderer` does, so the result is
    /// the same tokens a full render of that turn would produce. Token counts
    /// come from image geometry, so no image has to be encoded to build this.
    /// `openingConversation` builds the turn as the *first* turn of a
    /// conversation instead of a continuation: the chat template's opening,
    /// including `<bos>`, rather than a continuation's leading end-of-turn.
    /// Without it a fresh conversation's first image turn started with a
    /// dangling end-of-turn and no BOS, which this model is sensitive to.
    public func encodeMultimodalUserContinuation(
        textAndImages: [MultimodalContinuationPart],
        imageTokenCounts: [Int],
        openingConversation: Bool = false,
        family: ModelFamily = .gemma4,
        modelVariant: ModelVariant? = nil,
        reasoning: ChatReasoning = .off
    ) throws -> MultimodalContinuationTokens {
        let policy = MultimodalPromptRenderer.FamilyPolicy.forFamily(family)
        var text = ""
        var expected = 0
        for part in textAndImages {
            switch part {
            case .text(let value):
                guard !value.contains(policy.placeholder),
                      !value.contains("<|image_pad|>"),
                      !value.contains("<|vision_start|>"),
                      !value.contains("<|vision_end|>") else {
                    throw MultimodalPromptRendererError.reservedImageMarker
                }
                text += value
            case .image:
                text += policy.placeholder
                expected += 1
            }
        }
        guard expected == imageTokenCounts.count, expected > 0 else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        guard text.components(
            separatedBy: policy.placeholder).count - 1 == expected else {
            throw MultimodalPromptRendererError.reservedImageMarker
        }
        let content = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let template: [Int32]
        if openingConversation {
            // The same rendering the text path uses for an empty KV, so the
            // first turn of a conversation is identical whether or not it
            // carries an image.
            template = encode(
                try applyChatTemplate(
                    [Message(role: .user, content: content)],
                    modelVariant: modelVariant,
                    reasoning: reasoning),
                addBOS: false)
        } else if dialect == .gemma {
            let suffix = Self.gemmaGenerationSuffix(modelVariant: modelVariant,
                                                    reasoning: reasoning)
            template = [endOfTurnID] + encode(
                "\n\(Self.turnOpen)user\n\(content)\(Self.turnClose)\n"
                    + "\(Self.turnOpen)model\n" + suffix,
                addBOS: false)
        } else {
            template = [endOfTurnID] + encode(
                "\n\(Self.imStartMark)user\n\(content)\(Self.imEndMark)\n"
                    + (reasoning == .on
                        ? Self.chatMLThinkingOnSuffix
                        : Self.chatMLThinkingOffSuffix),
                addBOS: false)
        }
        // Count the placeholders the tokenizer actually produced before indexing
        // anything by them. The per-part marker check above cannot see a marker
        // split across two text parts, which concatenation would reassemble into
        // a real image token and leave more placeholders than counts.
        let placeholders = template.reduce(into: 0) {
            if $1 == policy.imageTokenID { $0 += 1 }
        }
        guard placeholders == imageTokenCounts.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }

        var effective: [Int32] = []
        var embedding: [Int32] = []
        var ranges: [Range<Int>] = []
        var index = 0
        for token in template {
            guard token == policy.imageTokenID else {
                effective.append(token)
                embedding.append(token)
                continue
            }
            let count = imageTokenCounts[index]
            if policy.rendererAddsBoundaryTokens {
                effective.append(policy.beginImageTokenID)
                embedding.append(policy.beginImageTokenID)
            }
            let lower = effective.count
            effective.append(contentsOf: repeatElement(
                policy.imageTokenID, count: count))
            embedding.append(contentsOf: repeatElement(Int32(0), count: count))
            ranges.append(lower..<effective.count)
            if policy.rendererAddsBoundaryTokens {
                effective.append(policy.endImageTokenID)
                embedding.append(policy.endImageTokenID)
            }
            index += 1
        }
        guard index == imageTokenCounts.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        return MultimodalContinuationTokens(
            effectiveTokenIDs: effective,
            embeddingTokenIDs: embedding,
            imageTokenRanges: ranges)
    }

    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition],
        reasoning: ChatReasoning = .off,
        preserveThinking: Bool = false,
        reasoningEffort: GPTOSSReasoningEffort? = nil,
        harmonyCurrentDate: String? = nil
    ) throws -> [Int32] {
        switch dialect {
        case .gemma:
            break
        case .flint:
            // No assistant end marker is rendered before an observation;
            // rebuild rather than assume a potentially incorrect KV boundary.
            throw GFTokenizerError.unsupportedForDialect("Flint tool-result KV continuation")
        case .chatml, .minimax:
            return try encodeTaggedToolResultContinuation(
                cachedMessages: cachedMessages, assistant: assistant,
                incomingMessages: incomingMessages, tools: tools,
                reasoning: reasoning, preserveThinking: preserveThinking)
        case .harmony:
            guard let reasoningEffort, let harmonyCurrentDate else {
                throw GFTokenizerError.invalidChatTemplate(
                    "Harmony tool-result continuation needs its reasoning effort and date")
            }
            return try encodeHarmonyToolResultContinuation(
                cachedMessages: cachedMessages, assistant: assistant,
                incomingMessages: incomingMessages, tools: tools,
                reasoningEffort: reasoningEffort, currentDate: harmonyCurrentDate)
        }
        // Positional disambiguation is safe only when both renders share the
        // exact leading conversation. Otherwise the same token offset can name
        // an unrelated, byte-identical call in the incoming history.
        guard incomingMessages.count > cachedMessages.count,
              incomingMessages.prefix(cachedMessages.count)
                .elementsEqual(cachedMessages) else {
            throw GFTokenizerError.invalidChatTemplate(
                "incoming tool-result history does not extend the cached conversation")
        }
        let incomingAssistant = incomingMessages[cachedMessages.count]
        guard incomingAssistant.role == .assistant,
              assistant.role == .assistant,
              !assistant.toolCalls.isEmpty,
              incomingAssistant.toolCalls == assistant.toolCalls,
              incomingAssistant.toolCallID == assistant.toolCallID,
              incomingAssistant.name == assistant.name,
              (incomingAssistant.content ?? "") == (assistant.content ?? "") else {
            throw GFTokenizerError.invalidChatTemplate(
                "incoming tool-result history does not contain the cached assistant call")
        }
        let prefix = try encodeToolChat(
            messages: cachedMessages + [assistant],
            tools: tools,
            reasoning: reasoning,
            preserveThinking: preserveThinking)
        let full = try encodeToolChat(
            messages: incomingMessages,
            tools: tools,
            reasoning: reasoning,
            preserveThinking: preserveThinking)
        let callCount = assistant.toolCalls.count
        let starts = prefix.indices.filter { prefix[$0] == toolCallStartID }
        guard callCount > 0, starts.count >= callCount,
              let callEnd = prefix.lastIndex(of: toolCallEndID) else {
            throw GFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is missing")
        }
        let callStart = starts[starts.count - callCount]
        let callSequence = Array(prefix[callStart...callEnd])
        let matches = full.subsequenceStartIndices(matching: callSequence)
        // The boundary is unambiguous by POSITION even when the sequence repeats.
        // `prefix` and `full` render the same leading history, and the generation
        // prompt is appended past the boundary, so the occurrence at `callStart`
        // is the cached call itself rather than an identical earlier twin.
        //
        // Requiring a unique match cost agentic loops their cache: as soon as a
        // model repeated a call verbatim — `ls` of the same path, a re-read of
        // the same file — the sequence occurred twice, the bridge refused to
        // encode, and the cache stayed dead for the rest of the conversation,
        // because every later request carried both twins too.
        let boundary: Int
        if matches.contains(callStart) {
            boundary = callStart
        } else if matches.count == 1 {
            boundary = matches[0]
        } else {
            throw GFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is ambiguous")
        }
        let suffixStart = boundary + callSequence.count
        let suffix = Array(full[suffixStart...])
        guard suffix.first == toolResponseID else {
            throw GFTokenizerError.invalidChatTemplate(
                "tool-result continuation does not begin at the KV boundary")
        }
        return suffix
    }
}

extension GFTokenizer {
    /// The tool-result bridge for ChatML (Qwen) and MiniMax, whose assistant
    /// turn ends with an end-of-turn token rather than Gemma's in-turn tool
    /// response marker.
    ///
    /// The cached KV holds the tokens the model generated up to, but not
    /// including, that end-of-turn token. The bridge is the full render of
    /// the incoming conversation from that token on. It is located by
    /// rendering the cached conversation plus the assistant turn without a
    /// generation prompt and requiring the full render to agree with it up to
    /// and including the closing token. Inside one user turn the templates'
    /// last-query position does not move, so the assistant turn's reasoning
    /// and calls render the same way in both, and any disagreement means the
    /// boundary cannot be trusted.
    func encodeTaggedToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition],
        reasoning: ChatReasoning,
        preserveThinking: Bool
    ) throws -> [Int32] {
        guard incomingMessages.count > cachedMessages.count + 1,
              incomingMessages.prefix(cachedMessages.count).elementsEqual(cachedMessages) else {
            throw GFTokenizerError.invalidChatTemplate(
                "incoming tool-result history does not extend the cached conversation")
        }
        let incomingAssistant = incomingMessages[cachedMessages.count]
        guard incomingAssistant.role == .assistant, assistant.role == .assistant,
              !assistant.toolCalls.isEmpty,
              incomingAssistant.toolCalls == assistant.toolCalls,
              (incomingAssistant.content ?? "") == (assistant.content ?? ""),
              incomingMessages.dropFirst(cachedMessages.count + 1).allSatisfy({ $0.role == .tool }) else {
            throw GFTokenizerError.invalidChatTemplate(
                "incoming tool-result history does not contain the cached assistant call")
        }
        // The cached assistant's own reasoning is what the KV holds; render it
        // into both sides so they agree on what precedes the boundary.
        var renderedIncoming = incomingMessages
        renderedIncoming[cachedMessages.count] = assistant
        let prefix = try encodeToolChat(
            messages: cachedMessages + [assistant], tools: tools,
            reasoning: reasoning, preserveThinking: preserveThinking,
            addGenerationPrompt: false)
        let full = try encodeToolChat(
            messages: renderedIncoming, tools: tools,
            reasoning: reasoning, preserveThinking: preserveThinking)
        let boundaryID = dialect == .minimax ? eosID : endOfTurnID
        guard let end = prefix.lastIndex(of: boundaryID),
              full.count > end + 1,
              full[...end].elementsEqual(prefix[...end]) else {
            throw GFTokenizerError.invalidChatTemplate(
                "tool-result continuation does not begin at the KV boundary")
        }
        return Array(full[end...])
    }
}

extension GFTokenizer {
    /// The tool-result bridge for GPT-OSS. A Harmony tool call ends with
    /// `<|call|>`, which is also the stop token, so the cached KV holds what
    /// the model generated up to that token and the bridge starts with it.
    ///
    /// Harmony keeps a tool call's analysis until a final answer follows it,
    /// and a tool result never adds one, so the cached assistant renders the
    /// same way with and without the results after it. The bridge is located
    /// the same way as the tagged one: the full render must agree with the
    /// cached conversation plus the assistant turn up to and including
    /// `<|call|>`, and everything from there on is new.
    func encodeHarmonyToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition],
        reasoningEffort: GPTOSSReasoningEffort,
        currentDate: String
    ) throws -> [Int32] {
        guard incomingMessages.count == cachedMessages.count + 2,
              incomingMessages.prefix(cachedMessages.count).elementsEqual(cachedMessages) else {
            throw GFTokenizerError.invalidChatTemplate(
                "incoming tool-result history does not extend the cached conversation")
        }
        let incomingAssistant = incomingMessages[cachedMessages.count]
        guard incomingAssistant.role == .assistant, assistant.role == .assistant,
              assistant.toolCalls.count == 1,
              incomingAssistant.toolCalls == assistant.toolCalls,
              (incomingAssistant.content ?? "") == (assistant.content ?? ""),
              incomingMessages.last?.role == .tool else {
            throw GFTokenizerError.invalidChatTemplate(
                "incoming tool-result history does not contain the cached assistant call")
        }
        var renderedIncoming = incomingMessages
        renderedIncoming[cachedMessages.count] = assistant
        let renderer = HarmonyPromptRenderer()
        let prefix = encode(
            try renderer.render(
                messages: cachedMessages + [assistant], tools: tools,
                reasoningEffort: reasoningEffort, currentDate: currentDate,
                addGenerationPrompt: false),
            addBOS: false)
        let full = encode(
            try renderer.render(
                messages: renderedIncoming, tools: tools,
                reasoningEffort: reasoningEffort, currentDate: currentDate),
            addBOS: false)
        guard let call = harmonyTokenIDs?.call,
              let end = prefix.lastIndex(of: call),
              end == prefix.count - 1,
              full.count > end + 1,
              full[...end].elementsEqual(prefix[...end]) else {
            throw GFTokenizerError.invalidChatTemplate(
                "tool-result continuation does not begin at the KV boundary")
        }
        return Array(full[end...])
    }
}

private extension Array where Element: Equatable {
    func subsequenceStartIndices(matching needle: [Element]) -> [Int] {
        guard !needle.isEmpty, needle.count <= count else { return [] }
        return indices.dropLast(needle.count - 1).filter { start in
            self[start..<(start + needle.count)].elementsEqual(needle)
        }
    }
}

enum GFTokenizerLoadSource: Hashable {
    case pretrained(String)
    case local(String)

    /// Whether resolving this source has to reach the network. A folder on
    /// disk either has the files or does not; a hub model ID can fail for a
    /// reason that will not still be true a second later.
    var isNetwork: Bool {
        switch self {
        case .pretrained: return true
        case .local: return false
        }
    }
}

/// Retry policy for tokenizer loads.
///
/// The hub fetch is the one part of loading a tokenizer that can fail without
/// anything being wrong: a request timed out, a connection dropped. Roughly
/// seventy tests resolve a tokenizer this way, so a single timeout failed an
/// otherwise clean run. One transient failure should not be a verdict.
///
/// A machine with no network at all is a different case, and retrying there
/// only makes a doomed run slower. So the retries stop being offered once a
/// network load has exhausted them: the first caller pays for the attempts and
/// everything after it fails immediately.
enum GFTokenizerLoadRetry {
    static let networkAttempts = 3

    static func attemptCount(for source: GFTokenizerLoadSource,
                             networkExhausted: Bool) -> Int {
        source.isNetwork && !networkExhausted ? networkAttempts : 1
    }

    /// Backoff before the attempt at `index` (1 for the first retry).
    static func backoff(beforeAttempt index: Int) -> Duration {
        .milliseconds(index <= 1 ? 500 : 2_000)
    }

    /// Runs `operation`, retrying up to `attempts` times. Cancellation is not
    /// a transient failure and is never retried.
    static func run<T: Sendable>(
        attempts: Int,
        sleep: @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) },
        operation: @Sendable () async throws -> T
    ) async throws -> T {
        var lastError: Error?
        for attempt in 0..<max(1, attempts) {
            if attempt > 0 {
                try Task.checkCancellation()
                try await sleep(backoff(beforeAttempt: attempt))
            }
            do {
                return try await operation()
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = error
            }
        }
        throw lastError ?? CancellationError()
    }
}

private actor GFTokenizerLoadCoordinator {
    static let shared = GFTokenizerLoadCoordinator()

    private var tasks: [GFTokenizerLoadSource: Task<GFTokenizer, Error>] = [:]
    /// Set once a network load has used all its attempts and still failed.
    private var networkExhausted = false

    func load(_ source: GFTokenizerLoadSource) async throws -> GFTokenizer {
        if let task = tasks[source] {
            return try await task.value
        }

        let attempts = GFTokenizerLoadRetry.attemptCount(
            for: source, networkExhausted: networkExhausted)
        // Keep the CPU-heavy tokenizer build off the coordinator actor; callers
        // share the task result instead of owning its cancellation.
        let task = Task.detached(priority: .userInitiated) { () throws -> GFTokenizer in
            try await GFTokenizerLoadRetry.run(attempts: attempts) {
                switch source {
                case .pretrained(let modelID):
                    return try await GFTokenizer.loadUncached(pretrained: modelID)
                case .local(let path):
                    return try await GFTokenizer.loadUncached(
                        from: URL(fileURLWithPath: path))
                }
            }
        }
        tasks[source] = task

        do {
            return try await task.value
        } catch {
            tasks[source] = nil
            if source.isNetwork, attempts > 1, !(error is CancellationError) {
                networkExhausted = true
            }
            throw error
        }
    }
}
