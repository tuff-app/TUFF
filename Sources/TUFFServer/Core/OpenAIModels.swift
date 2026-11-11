import Foundation
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFModelMetadata

public struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String
    }

    public let error: Detail

    public init(message: String, param: String? = nil, code: String,
                type: String = "invalid_request_error") {
        error = Detail(message: message,
                       type: type,
                       param: param,
                       code: code)
    }
}

public struct OpenAIImageURL: Codable, Equatable, Sendable {
    public let url: String
    public let detail: String?
}

public struct OpenAIContentPart: Codable, Equatable, Sendable {
    public let type: String
    public let text: String?
    public let imageURL: OpenAIImageURL?

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageURL = "image_url"
    }
}

public typealias OpenAITextPart = OpenAIContentPart

public enum OpenAIMessageContent: Codable, Equatable, Sendable {
    case text(String)
    case parts([OpenAIContentPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAIContentPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .parts(let parts): try container.encode(parts)
        }
    }

}

public struct OpenAIFunctionCall: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: String
}

public struct OpenAIToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let function: OpenAIFunctionCall
}

public struct OpenAIChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: OpenAIMessageContent?
    public let toolCalls: [OpenAIToolCall]?
    public let toolCallID: String?
    public let name: String?
    public var reasoningContent: String? = nil

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
        case reasoningContent = "reasoning_content"
    }
}

public struct OpenAIFunctionDefinition: Codable, Equatable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue
}

public struct OpenAITool: Codable, Equatable, Sendable {
    public let type: String
    public let function: OpenAIFunctionDefinition
}

public enum OpenAIStop: Codable, Equatable, Sendable {
    case one(String)
    case many([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let one = try? container.decode(String.self) {
            self = .one(one)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .one(let value): try container.encode(value)
        case .many(let value): try container.encode(value)
        }
    }

    var values: [String] {
        switch self {
        case .one(let value): [value]
        case .many(let value): value
        }
    }
}

public struct OpenAIStreamOptions: Codable, Equatable, Sendable {
    public let includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

public struct OpenAIChatRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let stream: Bool?
    public let streamOptions: OpenAIStreamOptions?
    public let temperature: Float?
    public let topP: Float?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let stop: OpenAIStop?
    public let seed: UInt64?
    public let tools: [OpenAITool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let n: Int?
    public let logprobs: Bool?
    public let presencePenalty: Float?
    public let frequencyPenalty: Float?
    /// TUFF's model-aware on/off reasoning control for Gemma 4 and Qwen,
    /// from `enable_thinking` or `chat_template_kwargs.enable_thinking`.
    public let enableThinking: Bool?
    /// Qwen's request to render earlier assistant reasoning back into the
    /// prompt. Accepted directly or through chat_template_kwargs.
    public let preserveThinking: Bool?
    /// As sent. Only `enable_thinking` and `preserve_thinking` are accepted,
    /// and both are folded into the fields above.
    public let chatTemplateKwargs: [String: JSONValue]?
    /// GPT-OSS graded reasoning control. Harmony defaults to Medium when this
    /// field is absent; other model families reject it.
    public let reasoningEffort: GPTOSSReasoningEffort?
    /// Kept as raw JSON so a value of any shape reaches the validator as a
    /// request error rather than as malformed JSON. Only `type` is read.
    public let responseFormat: JSONValue?
    /// The caller's name for its conversation. It orders the server's search
    /// among retained conversations and replaces that conversation's older
    /// state; it never establishes a match by itself. Values that are not
    /// strings were tolerated before 8.0 and are still ignored.
    public let promptCacheKey: String?

    enum CodingKeys: String, CodingKey, CaseIterable {
        case model, messages, stream, temperature, stop, seed, tools, n, logprobs
        case streamOptions = "stream_options"
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case enableThinking = "enable_thinking"
        case preserveThinking = "preserve_thinking"
        case chatTemplateKwargs = "chat_template_kwargs"
        case reasoningEffort = "reasoning_effort"
        case responseFormat = "response_format"
        case promptCacheKey = "prompt_cache_key"
    }

    /// The `chat_template_kwargs` TUFF understands: the same switches it
    /// accepts at the top level. Any other template argument would change a
    /// template TUFF renders itself, so it is refused.
    static let chatTemplateKwargs: Set<String> = ["enable_thinking", "preserve_thinking"]

    /// Top-level keys accepted and ignored: caller-side bookkeeping that
    /// cannot change what the model generates. Every other undeclared key is
    /// a 400, so a misspelled option such as `max_token` cannot silently run
    /// under settings the caller did not ask for.
    static let toleratedKeys: Set<String> = [
        "user",
        "store",
        "metadata",
        "service_tier",
        "safety_identifier",
    ]

    /// Real OpenAI parameters TUFF cannot honour, refused as unsupported
    /// rather than unknown so a caller is not told a real parameter looks
    /// like a typo. `reasoning_effort` is not here: TUFF honours it for
    /// GPT-OSS and the validator refuses it per model family.
    static let unsupportedKeys: [String: String] = [
        "logit_bias": "logit_bias is not supported",
        "top_logprobs": "top_logprobs is not supported",
        "verbosity": "verbosity is not supported",
        "modalities": "only text output is supported",
        "audio": "audio output is not supported",
        "prediction": "predicted outputs are not supported",
        "web_search_options": "web search is not supported",
        "functions": "legacy functions are not supported; use tools",
        "function_call": "legacy function_call is not supported; use tools and tool_choice",
    ]

    /// Keys other local servers accept that have a TUFF equivalent. Refused
    /// like any unknown key, with the field to use instead.
    static let equivalentKeys: [String: String] = [
        "max_new_tokens": "max_tokens",
        "num_predict": "max_tokens",
        "reasoning": "enable_thinking or, for GPT-OSS, reasoning_effort",
    ]

    /// Reads the request object's keys as written. The `CodingKeys` container
    /// reports only the keys it declares, so an undeclared key would be gone
    /// before validation could see it.
    private struct AnyKey: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }

        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    private static let maximumNamedKeyLength = 64
    private static let maximumNamedKeys = 8

    public init(from decoder: any Decoder) throws {
        // Swept before any typed decode, so a misspelled key is named as
        // itself rather than answered with whatever DecodingError another
        // field raises first.
        let anyKeys = try decoder.container(keyedBy: AnyKey.self)
        // A key set to null asks for nothing: openai-python sends an unset
        // option as an explicit null, and declared keys read null as absent.
        let written = try anyKeys.allKeys
            .filter { try !anyKeys.decodeNil(forKey: $0) }
            .map(\.stringValue)
        // Sorted so the answer never depends on the order keys arrived in.
        if let unsupported = written.filter({ Self.unsupportedKeys[$0] != nil }).sorted().first {
            throw ServerRequestError.invalid(
                message: Self.unsupportedKeys[unsupported]!,
                param: unsupported,
                code: "unsupported_value")
        }
        let unknown = written
            .filter { CodingKeys(stringValue: $0) == nil && !Self.toleratedKeys.contains($0) }
            .sorted()
        if let first = unknown.first {
            throw ServerRequestError.invalid(
                message: Self.unknownKeyMessage(unknown),
                param: boundedForDisplay(first, maxLength: Self.maximumNamedKeyLength),
                code: "unknown_parameter")
        }

        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        messages = try container.decode([OpenAIChatMessage].self, forKey: .messages)
        stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        streamOptions = try container.decodeIfPresent(
            OpenAIStreamOptions.self, forKey: .streamOptions)
        temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        maxCompletionTokens = try container.decodeIfPresent(
            Int.self, forKey: .maxCompletionTokens)
        stop = try container.decodeIfPresent(OpenAIStop.self, forKey: .stop)
        seed = try container.decodeIfPresent(UInt64.self, forKey: .seed)
        tools = try container.decodeIfPresent([OpenAITool].self, forKey: .tools)
        toolChoice = try container.decodeIfPresent(JSONValue.self, forKey: .toolChoice)
        parallelToolCalls = try container.decodeIfPresent(
            Bool.self, forKey: .parallelToolCalls)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        repetitionPenalty = try container.decodeIfPresent(
            Float.self, forKey: .repetitionPenalty)
        n = try container.decodeIfPresent(Int.self, forKey: .n)
        logprobs = try container.decodeIfPresent(Bool.self, forKey: .logprobs)
        presencePenalty = try container.decodeIfPresent(
            Float.self, forKey: .presencePenalty)
        frequencyPenalty = try container.decodeIfPresent(
            Float.self, forKey: .frequencyPenalty)
        chatTemplateKwargs = try container.decodeIfPresent(
            [String: JSONValue].self, forKey: .chatTemplateKwargs)
        let kwargs = chatTemplateKwargs ?? [:]
        let kwargKeys = kwargs.filter { $0.value != .null }.keys
        if let unknown = kwargKeys.filter({ !Self.chatTemplateKwargs.contains($0) }).sorted().first {
            let shown = boundedForDisplay(unknown, maxLength: Self.maximumNamedKeyLength)
            throw ServerRequestError.invalid(
                message: "unsupported chat_template_kwargs field \(String(reflecting: shown)); "
                    + "TUFF accepts enable_thinking and preserve_thinking there",
                param: "chat_template_kwargs",
                code: "unknown_parameter")
        }
        func flag(_ key: CodingKeys) throws -> Bool? {
            let top = try container.decodeIfPresent(Bool.self, forKey: key)
            let nested: Bool?
            switch kwargs[key.stringValue] {
            case nil, .null?: nested = nil
            case .bool(let value)?: nested = value
            default:
                throw ServerRequestError.invalid(
                    message: "chat_template_kwargs.\(key.stringValue) must be a boolean",
                    param: "chat_template_kwargs", code: "invalid_value")
            }
            if let top, let nested, top != nested {
                throw ServerRequestError.invalid(
                    message: "\(key.stringValue) and chat_template_kwargs.\(key.stringValue) disagree",
                    param: "chat_template_kwargs", code: "invalid_value")
            }
            return top ?? nested
        }
        enableThinking = try flag(.enableThinking)
        preserveThinking = try flag(.preserveThinking)
        reasoningEffort = try container.decodeIfPresent(
            GPTOSSReasoningEffort.self, forKey: .reasoningEffort)
        responseFormat = try container.decodeIfPresent(
            JSONValue.self, forKey: .responseFormat)
        switch try container.decodeIfPresent(JSONValue.self, forKey: .promptCacheKey) {
        case .string(let key)? where !key.isEmpty:
            // Bounded so a key cannot become a large per-entry allocation.
            promptCacheKey = String(key.utf8.prefix(256)) ?? String(key.prefix(64))
        default:
            promptCacheKey = nil
        }
    }

    /// Names at most eight keys, each bounded, and for a single key suggests
    /// the declared field it most likely meant.
    private static func unknownKeyMessage(_ unknown: [String]) -> String {
        let shown = unknown.prefix(maximumNamedKeys).map {
            String(reflecting: boundedForDisplay($0, maxLength: maximumNamedKeyLength))
        }
        var message = "unrecognized request field\(unknown.count == 1 ? "" : "s") "
            + shown.joined(separator: ", ")
        if unknown.count > shown.count {
            message += ", and \(unknown.count - shown.count) more"
        }
        if unknown.count == 1 {
            if let equivalent = equivalentKeys[unknown[0]] {
                message += "; use \(equivalent)"
            } else if let suggestion = closestDeclaredKey(to: unknown[0]) {
                message += "; did you mean \(suggestion)?"
            }
        }
        return message
    }

    /// The declared key within two edits of `key`, if exactly one is closest.
    /// Only short keys are compared, so a hostile key costs nothing.
    static func closestDeclaredKey(to key: String) -> String? {
        let candidate = Array(key.utf8)
        guard (2...maximumNamedKeyLength).contains(candidate.count) else { return nil }
        var best: (key: String, distance: Int)?
        var tied = false
        for declared in CodingKeys.allCases.map(\.stringValue) {
            let distance = editDistance(candidate, Array(declared.utf8))
            guard distance <= 2 else { continue }
            if best == nil || distance < best!.distance {
                best = (declared, distance)
                tied = false
            } else if distance == best!.distance {
                tied = true
            }
        }
        return tied ? nil : best?.key
    }

    private static func editDistance(_ a: [UInt8], _ b: [UInt8]) -> Int {
        var previous = Array(0...b.count)
        for (i, x) in a.enumerated() {
            var current = [i + 1] + Array(repeating: 0, count: b.count)
            for (j, y) in b.enumerated() {
                current[j + 1] = Swift.min(previous[j + 1] + 1,
                                           current[j] + 1,
                                           previous[j] + (x == y ? 0 : 1))
            }
            previous = current
        }
        return previous[b.count]
    }
}

/// Bounds text a rejection echoes back. The request body cap is 5 MiB, so a
/// caller must not be able to have an arbitrary slice of its request quoted.
/// Counted in UTF-8 bytes, not Characters: one Character can carry megabytes
/// of combining marks.
func boundedForDisplay(_ text: String, maxLength: Int) -> String {
    var bytes = 0
    var head = String.UnicodeScalarView()
    for scalar in text.unicodeScalars {
        bytes += scalar.utf8.count
        if bytes > maxLength {
            return String(head) + "..."
        }
        head.append(scalar)
    }
    return text
}

public struct OpenAIUsage: Codable, Equatable, Sendable {
    public struct PromptTokensDetails: Codable, Equatable, Sendable {
        public let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }

        public init(cachedTokens: Int) {
            self.cachedTokens = cachedTokens
        }
    }

    public struct CompletionTokensDetails: Codable, Equatable, Sendable {
        public let reasoningTokens: Int

        enum CodingKeys: String, CodingKey {
            case reasoningTokens = "reasoning_tokens"
        }

        public init(reasoningTokens: Int) {
            self.reasoningTokens = reasoningTokens
        }
    }

    public let tokenCountsAvailable: Bool?
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let promptTokensDetails: PromptTokensDetails
    /// Present only when the response carries reasoning.
    public let completionTokensDetails: CompletionTokensDetails?

    enum CodingKeys: String, CodingKey {
        case tokenCountsAvailable = "token_counts_available"
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
        case completionTokensDetails = "completion_tokens_details"
    }

    public init(promptTokens: Int,
                completionTokens: Int,
                totalTokens: Int,
                cachedTokens: Int = 0,
                reasoningTokens: Int? = nil,
                tokenCountsAvailable: Bool? = nil) {
        self.tokenCountsAvailable = tokenCountsAvailable
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: cachedTokens)
        self.completionTokensDetails = reasoningTokens.map(CompletionTokensDetails.init)
    }
    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encodeIfPresent(tokenCountsAvailable, forKey: .tokenCountsAvailable)
        if tokenCountsAvailable == false {
            try values.encodeNil(forKey: .promptTokens)
            try values.encodeNil(forKey: .completionTokens)
            try values.encodeNil(forKey: .totalTokens)
            try values.encodeNil(forKey: .promptTokensDetails)
        } else {
            try values.encode(promptTokens, forKey: .promptTokens)
            try values.encode(completionTokens, forKey: .completionTokens)
            try values.encode(totalTokens, forKey: .totalTokens)
            try values.encode(promptTokensDetails, forKey: .promptTokensDetails)
            try values.encodeIfPresent(completionTokensDetails, forKey: .completionTokensDetails)
        }
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        tokenCountsAvailable = try values.decodeIfPresent(Bool.self, forKey: .tokenCountsAvailable)
        promptTokens = try values.decodeIfPresent(Int.self, forKey: .promptTokens) ?? 0
        completionTokens = try values.decodeIfPresent(Int.self, forKey: .completionTokens) ?? 0
        totalTokens = try values.decodeIfPresent(Int.self, forKey: .totalTokens) ?? 0
        promptTokensDetails = try values.decodeIfPresent(PromptTokensDetails.self, forKey: .promptTokensDetails) ?? .init(cachedTokens: 0)
        completionTokensDetails = try values.decodeIfPresent(CompletionTokensDetails.self, forKey: .completionTokensDetails)
    }

}

public struct OpenAIModelList: Codable, Equatable, Sendable {
    public struct Model: Codable, Equatable, Sendable {
        public let id: String
        public let object: String
        public let created: Int
        public let ownedBy: String
        public let capabilities: [String]?
        public let contextLength: Int?
        public let maxOutputTokens: Int?

        enum CodingKeys: String, CodingKey {
            case id, object, created, capabilities
            case ownedBy = "owned_by"
            case contextLength = "context_length"
            case maxOutputTokens = "max_output_tokens"
        }

        public init(id: String, object: String, created: Int,
                    ownedBy: String, capabilities: [String]? = nil,
                    contextLength: Int? = nil, maxOutputTokens: Int? = nil) {
            self.id = id
            self.object = object
            self.created = created
            self.ownedBy = ownedBy
            self.capabilities = capabilities
            self.contextLength = contextLength
            self.maxOutputTokens = maxOutputTokens
        }
    }

    public let object: String
    public let data: [Model]
}

public enum ServerRequestError: Error, Codable, Equatable, Sendable {
    case invalid(message: String, param: String?, code: String)
    case unknownModel
    case queueFull
    /// A known model that cannot be served here: not installed, or too large
    /// for this Mac.
    case modelUnavailable(String)
    /// The model would fit on its own, but another TUFF process holds enough
    /// memory that loading it now would overcommit the Mac.
    case modelMemoryBusy(String)
    /// The shared serving host cannot be reached, or went away mid-request.
    case hostUnavailable(String)
    /// The model is held by another client in a configuration this request
    /// cannot use, and the holder is busy.
    case modelBusy(String)

    public var envelope: OpenAIErrorEnvelope {
        switch self {
        case .modelUnavailable(let message):
            OpenAIErrorEnvelope(message: message, param: "model", code: "model_not_found")
        case .modelMemoryBusy(let message):
            OpenAIErrorEnvelope(message: message, code: "model_memory_busy")
        case .hostUnavailable(let message):
            OpenAIErrorEnvelope(message: message, code: "inference_host_unavailable",
                                type: "server_error")
        case .modelBusy(let message):
            OpenAIErrorEnvelope(message: message, code: "model_busy")
        case .invalid(let message, let param, let code):
            OpenAIErrorEnvelope(message: message, param: param, code: code)
        case .unknownModel:
            OpenAIErrorEnvelope(message: "requested model is not available",
                                param: "model", code: "model_not_found")
        case .queueFull:
            OpenAIErrorEnvelope(message: "generation queue is full",
                                code: "queue_full")
        }
    }
}

public struct ValidatedChatRequest: Codable, Sendable {
    public let messages: [InferenceChatMessage]
    public let multimodalMessages: [MultimodalMessage]?
    public let imageFiles: [UUID: URL]
    /// Content SHA-256 of each message's images, in order, aligned with
    /// `messages`. Staged image UUIDs are fresh per request, so they cannot
    /// identify an image across turns; the content hash can. Empty for
    /// text-only requests.
    public let imageIdentities: [[String]]
    public let tools: [InferenceFunctionDefinition]
    public let stream: Bool
    public let includeUsage: Bool
    public let generationConfig: GenerationConfig
    public let maximumCompletionTokens: Int
    public let reasoning: ChatReasoning
    public let preserveThinking: Bool
    public let reasoningEffort: GPTOSSReasoningEffort?
    /// The local calendar day rendered into Harmony's system message. It is
    /// captured during validation so queued requests and cache comparisons do
    /// not change identity across midnight.
    public let harmonyCurrentDate: String?
    /// Every staging directory this request's image files live in. The parser
    /// and the validator's store each stage under their own lease, and a
    /// request may carry files from both, so dropping either would delete
    /// files the other path staged before `generate` reads them.
    fileprivate var attachmentLeases: [ServerAttachmentLease] = []
    /// See `OpenAIChatRequest.promptCacheKey`.
    public var promptCacheKey: String?

    private enum CodingKeys: String, CodingKey {
        case messages, multimodalMessages, imageFiles, imageIdentities, tools
        case stream, includeUsage, generationConfig, maximumCompletionTokens
        case reasoning, preserveThinking, reasoningEffort, harmonyCurrentDate, promptCacheKey
    }

    public init(
        messages: [InferenceChatMessage],
        multimodalMessages: [MultimodalMessage]? = nil,
        imageFiles: [UUID: URL] = [:],
        imageIdentities: [[String]] = [],
        tools: [InferenceFunctionDefinition],
        stream: Bool,
        includeUsage: Bool,
        generationConfig: GenerationConfig,
        maximumCompletionTokens: Int,
        reasoning: ChatReasoning = .off,
        preserveThinking: Bool = false,
        reasoningEffort: GPTOSSReasoningEffort? = nil,
        harmonyCurrentDate: String? = nil
    ) {
        self.messages = messages
        self.multimodalMessages = multimodalMessages
        self.imageFiles = imageFiles
        self.imageIdentities = imageIdentities
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.reasoning = reasoning
        self.preserveThinking = preserveThinking
        self.reasoningEffort = reasoningEffort
        self.harmonyCurrentDate = harmonyCurrentDate
        self.attachmentLeases = []
    }

    fileprivate init(
        messages: [InferenceChatMessage],
        multimodalMessages: [MultimodalMessage]?,
        imageFiles: [UUID: URL],
        imageIdentities: [[String]],
        tools: [InferenceFunctionDefinition],
        stream: Bool,
        includeUsage: Bool,
        generationConfig: GenerationConfig,
        maximumCompletionTokens: Int,
        reasoning: ChatReasoning,
        preserveThinking: Bool,
        reasoningEffort: GPTOSSReasoningEffort?,
        harmonyCurrentDate: String?,
        attachmentLeases: [ServerAttachmentLease]
    ) {
        self.messages = messages
        self.multimodalMessages = multimodalMessages
        self.imageFiles = imageFiles
        self.imageIdentities = imageIdentities
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.reasoning = reasoning
        self.preserveThinking = preserveThinking
        self.reasoningEffort = reasoningEffort
        self.harmonyCurrentDate = harmonyCurrentDate
        self.attachmentLeases = attachmentLeases
    }
}

private enum OpenAIToolName {
    static let maximumLength = 64

    static func isValid(_ name: String) -> Bool {
        let bytes = name.utf8
        guard !bytes.isEmpty, bytes.count <= maximumLength else { return false }
        return bytes.allSatisfy { byte in
            switch byte {
            case 45, 48...57, 65...90, 95, 97...122:
                true
            default:
                false
            }
        }
    }

    static func validationMessage(for name: String) -> String {
        let displayed = String(reflecting: boundedForDisplay(name, maxLength: maximumLength))
        return "tool name \(displayed) must contain 1 to 64 ASCII letters, numbers, underscores, or hyphens"
    }
}

public enum OpenAIRequestValidator {
    public static func validate(_ request: OpenAIChatRequest,
                                modelID: String,
                                dialect: ChatDialect = .gemma) throws -> ValidatedChatRequest {
        try validate(
            request,
            modelID: modelID,
            dialect: dialect,
            preStagedImages: [:],
            attachmentLease: nil)
    }

    static func validate(_ request: OpenAIChatRequest,
                         modelID: String,
                         dialect: ChatDialect = .gemma,
                         preStagedImages: [String: ServerStagedImage],
                         attachmentLease: ServerAttachmentLease?,
                         appleModel: Bool = false) throws -> ValidatedChatRequest {
        guard request.model == modelID else { throw ServerRequestError.unknownModel }
        guard request.n == nil || request.n == 1 else {
            throw request.n.map { $0 < 1 } == true
                ? invalid("n must be at least 1", "n", "invalid_value")
                : invalid("only n=1 is supported", "n", "unsupported_value")
        }
        guard request.logprobs != true else {
            throw invalid("logprobs are not supported", "logprobs", "unsupported_value")
        }
        guard request.presencePenalty == nil || request.presencePenalty == 0 else {
            throw invalid("presence_penalty must be zero", "presence_penalty", "unsupported_value")
        }
        guard request.frequencyPenalty == nil || request.frequencyPenalty == 0 else {
            throw invalid("frequency_penalty must be zero", "frequency_penalty", "unsupported_value")
        }
        guard request.parallelToolCalls != false else {
            throw invalid("parallel_tool_calls=false is not supported",
                          "parallel_tool_calls", "unsupported_value")
        }
        try validateResponseFormat(request.responseFormat)

        let isApple = appleModel || ["apple", "apple.on-device"].contains(modelID)
        if isApple, request.topK != nil || request.topP != nil || request.seed != nil
            || !(request.stop?.values.isEmpty ?? true) || request.enableThinking == true
            || request.reasoningEffort != nil || !(request.tools?.isEmpty ?? true)
            || (request.repetitionPenalty ?? 1) != 1 {
            throw invalid("Apple supports text, temperature and output limits; sampling filters, seeds, stops, reasoning, tools and penalties are unsupported", "model", "unsupported_parameter")
        }
        let temperature = request.temperature ?? 0.2
        guard temperature >= 0, temperature <= 2 else {
            throw invalid("temperature must be between 0 and 2",
                          "temperature", "invalid_value")
        }
        let topP = request.topP ?? 0.95
        guard topP > 0, topP <= 1 else {
            throw invalid("top_p must be greater than 0 and at most 1",
                          "top_p", "invalid_value")
        }
        let topK = request.topK ?? 64
        guard (1...256).contains(topK) else {
            throw invalid("top_k must be between 1 and 256", "top_k", "invalid_value")
        }
        let repetitionPenalty = request.repetitionPenalty ?? 1
        guard repetitionPenalty > 0 else {
            throw invalid("repetition_penalty must be positive",
                          "repetition_penalty", "invalid_value")
        }
        let maximum = request.maxCompletionTokens ?? request.maxTokens ?? 4096
        guard maximum > 0 else {
            throw invalid("maximum completion tokens must be positive",
                          request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                          "invalid_value")
        }
        let reasoning: ChatReasoning
        let reasoningEffort: GPTOSSReasoningEffort?
        let harmonyCurrentDate: String?
        if dialect == .harmony {
            guard request.enableThinking == nil else {
                throw invalid(
                    "enable_thinking is not supported by GPT-OSS; use reasoning_effort",
                    "enable_thinking", "unsupported_parameter")
            }
            reasoning = .off
            reasoningEffort = request.reasoningEffort ?? .medium
            harmonyCurrentDate = InferenceCalendarDate.calendarDate()
        } else if dialect == .minimax {
            guard request.reasoningEffort == nil else {
                throw invalid(
                    "reasoning_effort is only supported by GPT-OSS",
                    "reasoning_effort", "unsupported_parameter")
            }
            reasoning = .on
            reasoningEffort = nil
            harmonyCurrentDate = nil
        } else {
            guard request.reasoningEffort == nil else {
                throw invalid(
                    "reasoning_effort is only supported by GPT-OSS",
                    "reasoning_effort", "unsupported_parameter")
            }
            reasoning = request.enableThinking == true ? .on : .off
            reasoningEffort = nil
            harmonyCurrentDate = nil
        }

        let includeTools: Bool
        switch request.toolChoice {
        case nil, .some(.string("auto")):
            includeTools = true
        case .some(.string("none")):
            includeTools = false
        case .some(.string("required")):
            throw invalid("tool_choice=required is not supported",
                          "tool_choice", "unsupported_value")
        default:
            throw invalid("named tool choices are not supported",
                          "tool_choice", "unsupported_value")
        }

        let tools = try (includeTools ? request.tools ?? [] : []).map {
            try validateTool($0, dialect: dialect)
        }
        let validatedMessages = try validateMessages(
            request.messages,
            dialect: dialect,
            preStagedImages: preStagedImages,
            attachmentLease: attachmentLease)
        let config = GenerationConfig(maxNewTokens: maximum,
                                      temperature: temperature,
                                      topK: isApple ? nil : topK,
                                      topP: isApple ? nil : topP,
                                      repetitionPenalty: repetitionPenalty,
                                      seed: request.seed,
                                      stopStrings: request.stop?.values ?? [])
        var validated = ValidatedChatRequest(messages: validatedMessages.messages,
                                    multimodalMessages: validatedMessages.multimodal,
                                    imageFiles: validatedMessages.imageFiles,
                                    imageIdentities: validatedMessages.imageIdentities,
                                    tools: tools,
                                    stream: request.stream ?? false,
                                    includeUsage: request.streamOptions?.includeUsage ?? false,
                                    generationConfig: config,
                                    maximumCompletionTokens: maximum,
                                    reasoning: reasoning,
                                    preserveThinking: request.preserveThinking ?? false,
                                    reasoningEffort: reasoningEffort,
                                    harmonyCurrentDate: harmonyCurrentDate,
                                    attachmentLeases: validatedMessages.leases)
        validated.promptCacheKey = request.promptCacheKey
        return validated
    }

    /// `{"type": "text"}` is what TUFF already produces. Structured output is
    /// refused rather than ignored, so a caller relying on JSON mode is not
    /// handed free text it will fail to parse.
    private static func validateResponseFormat(_ format: JSONValue?) throws {
        guard let format else { return }
        guard case .object(let fields) = format else {
            throw invalid(#"response_format must be an object such as {"type": "text"}"#,
                          "response_format", "invalid_value")
        }
        switch fields["type"] {
        case .string("text")?:
            return
        case .string("json_object")?, .string("json_schema")?:
            throw invalid("structured output is not supported",
                          "response_format", "unsupported_value")
        case .string(let type)?:
            throw invalid(
                "response_format type \(String(reflecting: boundedForDisplay(type, maxLength: 64))) "
                    + "is not recognized",
                "response_format", "invalid_value")
        case nil, .null?:
            throw invalid("response_format.type is required",
                          "response_format", "invalid_value")
        default:
            throw invalid("response_format.type must be a string",
                          "response_format", "invalid_value")
        }
    }

    private static func validateTool(_ tool: OpenAITool,
                                     dialect: ChatDialect) throws -> InferenceFunctionDefinition {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard OpenAIToolName.isValid(name) else {
            throw invalid(OpenAIToolName.validationMessage(for: name),
                          "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid("tool parameters must be an object schema",
                          "tools", "invalid_tool_schema")
        }
        try validateSchemaKeys(tool.function.parameters, dialect: dialect)
        // The Gemma template can only render a restricted schema subset, so it
        // needs the adaptation pass. ChatML and Harmony preserve the schema.
        let parameters = dialect == .gemma
            ? try GemmaToolSchema.adapted(tool.function.parameters, toolName: name)
            : tool.function.parameters
        guard (try? parameters.jinjaSendableValue()) != nil else {
            throw invalid("tool schema contains a number that cannot be represented exactly",
                          "tools", "invalid_tool_schema")
        }
        return InferenceFunctionDefinition(name: name,
                                              description: tool.function.description ?? "",
                                              parameters: parameters)
    }

    private static func validateSchemaKeys(_ schema: JSONValue,
                                           dialect: ChatDialect) throws {
        switch schema {
        case .object(let object):
            for (schemaKey, value) in object {
                if schemaKey == "properties" {
                    guard case .object(let definitions) = value else {
                        throw invalid("tool schema properties must be an object",
                                      "tools", "invalid_tool_schema")
                    }
                    for (key, definition) in definitions {
                        // Gemma's tool-call DSL cannot round-trip arbitrary
                        // parameter names; ChatML and Harmony calls are JSON.
                        guard dialect != .gemma
                                || GemmaToolCallParser.isRepresentableObjectKey(key) else {
                            throw invalid(
                                "tool parameter names may contain only letters, numbers, _, -, ., and $",
                                "tools",
                                "invalid_tool_schema")
                        }
                        try validateSchemaKeys(definition, dialect: dialect)
                    }
                } else {
                    try validateSchemaKeys(value, dialect: dialect)
                }
            }
        case .array(let values):
            for value in values {
                try validateSchemaKeys(value, dialect: dialect)
            }
        default:
            break
        }
    }

    private struct ValidatedMessages {
        let messages: [InferenceChatMessage]
        let multimodal: [MultimodalMessage]?
        let imageFiles: [UUID: URL]
        let imageIdentities: [[String]]
        let leases: [ServerAttachmentLease]
    }

    private static func validateMessages(
        _ input: [OpenAIChatMessage],
        dialect: ChatDialect,
        preStagedImages: [String: ServerStagedImage],
        attachmentLease: ServerAttachmentLease?
    ) throws -> ValidatedMessages {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        var knownCalls: [String: (name: String, resolved: Bool)] = [:]
        var result: [InferenceChatMessage] = []
        var multimodal: [MultimodalMessage] = []
        var imageFiles: [UUID: URL] = [:]
        var imageIdentities: [[String]] = []
        var messageIdentities: [String] = []
        var store: ServerAttachmentStore?
        var sawConversationMessage = false
        for message in input {
            guard let role = InferenceChatRole(rawValue: message.role) else {
                throw invalid("unsupported message role \(message.role)",
                              "messages", "invalid_message")
            }
            if role == .system || role == .developer {
                guard !sawConversationMessage else {
                    throw invalid("system or developer guidance must precede the conversation",
                                  "messages", "invalid_message")
                }
            } else {
                sawConversationMessage = true
            }
            var orderedContent: [MultimodalContentPart] = []
            let content: String?
            switch message.content {
            case nil:
                content = nil
            case .text(let text):
                content = text
                orderedContent = [.text(text)]
            case .parts(let parts):
                var joined = ""
                for part in parts {
                    switch part.type {
                    case "text":
                        guard let text = part.text else {
                            throw invalid("text content part requires text",
                                          "messages", "invalid_message")
                        }
                        joined += text
                        orderedContent.append(.text(text))
                    case "image_url":
                        guard role == .user else {
                            throw invalid("image_url is supported only in user messages",
                                          "messages", "unsupported_content")
                        }
                        guard let image = part.imageURL else {
                            throw invalid("image_url content part requires an image_url object",
                                          "messages", "invalid_message")
                        }
                        guard image.detail == nil || image.detail == "auto" else {
                            throw invalid("image detail must be absent or auto",
                                          "messages", "unsupported_value")
                        }
                        let staged: ServerStagedImage
                        let prefix = "tuff-attachment:"
                        if image.url.hasPrefix(prefix) {
                            let token = String(image.url.dropFirst(prefix.count))
                            guard let existing = preStagedImages[token] else {
                                throw invalid("image attachment lease is missing",
                                              "messages", "invalid_image")
                            }
                            staged = existing
                        } else {
                            if store == nil { store = try ServerAttachmentStore() }
                            staged = try store!.stage(dataURL: image.url)
                        }
                        imageFiles[staged.id] = staged.fileURL
                        messageIdentities.append(staged.sha256)
                        orderedContent.append(.image(id: staged.id))
                    default:
                        throw invalid("unsupported content part \(part.type)",
                                      "messages", "unsupported_content")
                    }
                }
                content = joined
            }
            // The semantic bound on images is the context budget, checked
            // against the model's actual context in `ServerModelSession.prepare`
            // where the per-image token cost is known. Staging enforces its own
            // per-file, per-request, and count resource caps; no further count
            // check belongs here.

            let calls: [InferenceToolCall] = try (message.toolCalls ?? []).map { call in
                guard role == .assistant, call.type == "function",
                      !call.id.isEmpty, knownCalls[call.id] == nil else {
                    throw invalid("invalid or duplicate historical tool call",
                                  "messages", "invalid_tool_call")
                }
                guard OpenAIToolName.isValid(call.function.name) else {
                    throw invalid(OpenAIToolName.validationMessage(for: call.function.name),
                                  "messages", "invalid_tool_call")
                }
                let data = Data(call.function.arguments.utf8)
                let arguments = try JSONDecoder().decode(JSONValue.self, from: data)
                guard arguments.objectValue != nil else {
                    throw invalid("historical tool arguments must be a JSON object",
                                  "messages", "invalid_tool_arguments")
                }
                guard dialect == .chatml
                        || (try? arguments.gemmaToolArgumentBody()) != nil,
                      (try? arguments.jinjaSendableValue()) != nil else {
                    throw invalid(
                        "historical tool arguments cannot be represented exactly: "
                            + "unsupported value",
                        "messages",
                        "invalid_tool_arguments")
                }
                knownCalls[call.id] = (call.function.name, false)
                return InferenceToolCall(
                    id: call.id, name: call.function.name, arguments: arguments)
            }
            if role == .tool {
                guard let id = message.toolCallID,
                      let call = knownCalls[id], !call.resolved else {
                    throw invalid("tool result must reference one unresolved call",
                                  "messages", "invalid_tool_result")
                }
                knownCalls[id] = (call.name, true)
                guard content != nil else {
                    throw invalid("tool result content is required",
                                  "messages", "invalid_tool_result")
                }
            } else if content == nil && calls.isEmpty {
                throw invalid("message content is required",
                              "messages", "invalid_message")
            }
            guard message.reasoningContent == nil || role == .assistant else {
                throw invalid("reasoning_content is supported only in assistant messages",
                              "messages", "invalid_message")
            }
            result.append(InferenceChatMessage(role: role,
                                              content: content,
                                              thinking: message.reasoningContent,
                                              toolCalls: calls,
                                              toolCallID: message.toolCallID,
                                              name: message.name))
            if orderedContent.isEmpty, let content {
                orderedContent = [.text(content)]
            }
            multimodal.append(MultimodalMessage(
                role: role,
                content: orderedContent,
                thinking: message.reasoningContent,
                toolCalls: calls,
                toolCallID: message.toolCallID,
                name: message.name))
            imageIdentities.append(messageIdentities)
            messageIdentities = []
        }
        return ValidatedMessages(
            messages: result,
            multimodal: imageFiles.isEmpty ? nil : multimodal,
            imageFiles: imageFiles,
            imageIdentities: imageFiles.isEmpty ? [] : imageIdentities,
            leases: [attachmentLease, store?.lease].compactMap { $0 })
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}
