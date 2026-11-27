import TUFFModelMetadata
import Foundation
import TUFFInferenceTypes
import TUFFRuntimeDescriptions

public struct AppGenerationRequest: Equatable, Sendable {
    public var modelDirectory: URL
    public var prompt: String
    /// The system prompt, rendered as a system message ahead of the whole
    /// conversation. Empty means none.
    public var systemPrompt: String
    /// Text the model should treat as the beginning of its own reply, appended
    /// after the generation prompt so decoding carries it on. Empty means a
    /// fresh answer. Used by Continue.
    public var assistantPrefix: String
    /// Completed turns preceding `prompt`, oldest first. Empty is the
    /// single-turn behavior this type had before.
    public var history: [AppChatTurn]
    public var imageAttachments: [AppImageAttachment]
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    public var reasoning: ChatReasoning
    public var reasoningEffort: ModelReasoningEffort?
    public var preserveThinking: Bool
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var runtimeOptions: AppRuntimeOptions
    /// Tools declared to the model, in the model's native tool format.
    public var tools: [InferenceFunctionDefinition]
    /// Rounds already run for the message being answered: the model's calls
    /// and the results that answered them, rendered after `prompt`.
    public var currentRounds: [AppToolRound]
    /// The chat this request belongs to. Orders the search among retained
    /// conversation states; never establishes a match by itself.
    public var conversationKey: String?
    /// Auto refuses to silently remove history when its token estimate was low.
    public var preservesFullHistory: Bool
    public var ignoreEOS: Bool

    public init(modelDirectory: URL,
                prompt: String,
                systemPrompt: String = "",
                assistantPrefix: String = "",
                history: [AppChatTurn] = [],
                imageAttachments: [AppImageAttachment] = [],
                maxNewTokens: Int = 4_096,
                maxContextTokens: Int = 4096,
                reasoning: ChatReasoning = .off,
                reasoningEffort: ModelReasoningEffort? = nil,
                preserveThinking: Bool = false,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                runtimeOptions: AppRuntimeOptions = AppRuntimeOptions(),
                tools: [InferenceFunctionDefinition] = [],
                currentRounds: [AppToolRound] = [],
                conversationKey: String? = nil, preservesFullHistory: Bool = false,
                ignoreEOS: Bool = false) {
        self.modelDirectory = modelDirectory
        self.prompt = prompt
        self.systemPrompt = systemPrompt
        self.assistantPrefix = assistantPrefix
        self.history = history
        self.imageAttachments = imageAttachments
        self.maxNewTokens = maxNewTokens
        self.maxContextTokens = maxContextTokens
        self.reasoning = reasoning
        self.reasoningEffort = reasoningEffort
        self.preserveThinking = preserveThinking
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.runtimeOptions = runtimeOptions
        self.tools = tools
        self.currentRounds = currentRounds
        self.conversationKey = conversationKey
        self.preservesFullHistory = preservesFullHistory
        self.ignoreEOS = ignoreEOS
    }

    /// Whether the prompt is rendered through the model's tool template:
    /// tools are declared, or the conversation already contains tool turns.
    public var usesToolTemplate: Bool {
        !tools.isEmpty || !currentRounds.isEmpty
            || history.contains { !$0.toolRounds.isEmpty }
    }

    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

    public func validate(fileManager: FileManager = .default,
                         requireModelDirectory: Bool = true) throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !imageAttachments.isEmpty else {
            throw AppInferenceError.invalidRequest("Prompt or image cannot be empty.")
        }
        // The same context-derived rule the server uses. A fixed four here
        // meant a request the API accepted was refused in the app.
        guard Set(imageAttachments.map(\.id)).count == imageAttachments.count else {
            throw AppInferenceError.invalidRequest("Images must be distinct.")
        }
        let family = (try? ManifestReader.resolveArchitecture(
            directoryURL: modelDirectory).family)
            ?? .gemma4
        try Self.validateReasoning(
            family: family,
            reasoning: reasoning,
            reasoningEffort: reasoningEffort)
        try Self.validateImageCapacity(
            imageCount: imageAttachments.count,
            maxContextTokens: maxContextTokens,
            family: family)
        for attachment in imageAttachments {
            guard attachment.fileURL.isFileURL,
                  attachment.encodedBytes >= 0,
                  attachment.encodedBytes <= VisionImageLimits().maximumEncodedBytes,
                  attachment.sha256.count == 64,
                  attachment.sha256.unicodeScalars.allSatisfy({
                      (48...57).contains($0.value) || (97...102).contains($0.value)
                  }) else {
                throw AppInferenceError.invalidRequest(
                    "Invalid image attachment \(attachment.displayName).")
            }
        }
        if ignoreEOS, usesToolTemplate {
            throw AppInferenceError.invalidRequest("ignore_eos requires a request without tools or tool history.")
        }
        guard maxNewTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max response length must be greater than zero.")
        }
        guard maxContextTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max context must be greater than zero.")
        }
        guard temperature >= 0 else {
            throw AppInferenceError.invalidRequest("Temperature cannot be negative.")
        }
        if let topK {
            guard (1...256).contains(topK) else {
                throw AppInferenceError.invalidRequest("Top-K must be between 1 and 256.")
            }
        }
        if let topP {
            guard topP > 0, topP <= 1 else {
                throw AppInferenceError.invalidRequest("Top-P must be greater than 0 and at most 1.")
            }
            if temperature > 0, topP < 1, topK == nil {
                throw AppInferenceError.invalidRequest(
                    "Top-P below 1 requires Top-K to be enabled.")
            }
        }
        guard currentRounds.allSatisfy(\.isComplete) else {
            throw AppInferenceError.invalidRequest(
                "Every tool call must have a result before the model continues.")
        }
        guard Set(tools.map(\.name)).count == tools.count else {
            throw AppInferenceError.invalidRequest("Tool names must be distinct.")
        }
        guard repetitionPenalty >= 1 else {
            throw AppInferenceError.invalidRequest(
                "Repetition penalty must be at least 1.")
        }
        try runtimeOptions.validate()

        if requireModelDirectory {
            var isDirectory = ObjCBool(false)
            guard fileManager.fileExists(atPath: modelDirectory.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw AppInferenceError.modelNotFound(modelDirectory.path)
            }
        }
    }

    static func validateReasoning(
        family: ModelFamily,
        reasoning: ChatReasoning,
        reasoningEffort: ModelReasoningEffort?
    ) throws {
        if family == .gptOss {
            guard reasoning == .off else {
                throw AppInferenceError.invalidRequest(
                    "GPT-OSS uses graded reasoning effort, not an on/off switch.")
            }
        } else if family != .inkling2 && reasoningEffort != nil {
            throw AppInferenceError.invalidRequest(
                "Graded reasoning effort is supported by GPT-OSS and Inkling 2.")
        }
    }

    static func validateImageCapacity(
        imageCount: Int,
        maxContextTokens: Int,
        family: ModelFamily
    ) throws {
        guard imageCount > 0 else { return }
        guard family != .gptOss else {
            throw AppInferenceError.invalidRequest(
                "GPT-OSS does not support image input.")
        }
        let maximumTokensPerImage = VisionImageTokenBudget.maximumTokensPerImage(
            family: family)
        let capacity = VisionImageTokenBudget.capacity(
            maxContext: maxContextTokens,
            reservedTextTokens: 0,
            family: family)
        guard imageCount <= capacity else {
            throw AppInferenceError.invalidRequest(
                "\(imageCount) images need up to "
                    + "\(imageCount * maximumTokensPerImage) "
                    + "tokens, beyond the \(maxContextTokens)-token context.")
        }
    }
}
