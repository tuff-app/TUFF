import Foundation

public typealias MultimodalContentPart = TUFFInferenceTypes.MultimodalContentPart
public typealias MultimodalContinuationPart = TUFFInferenceTypes.MultimodalContinuationPart
public typealias MultimodalContinuationTokens = TUFFInferenceTypes.MultimodalContinuationTokens
public typealias MultimodalMessage = TUFFInferenceTypes.MultimodalMessage

public enum MultimodalPromptRendererError: Error, Equatable {
    case emptyMessages
    case emptyContent
    case reservedImageMarker
    case missingImage(UUID)
    case unexpectedImage(UUID)
    case placeholderMismatch
}

public enum MultimodalPromptRenderer {
    public static let placeholder = "<|image|>"
    public static let imageTokenID = Int32(258_880)
    public static let beginImageTokenID = Int32(255_999)
    public static let endImageTokenID = Int32(258_882)

    public struct FamilyPolicy: Sendable, Equatable {
        public let family: ModelFamily
        public let placeholder: String
        public let imageTokenID: Int32
        public let beginImageTokenID: Int32
        public let endImageTokenID: Int32
        public let rendererAddsBoundaryTokens: Bool

        public static let gemma4 = FamilyPolicy(
            family: .gemma4,
            placeholder: MultimodalPromptRenderer.placeholder,
            imageTokenID: MultimodalPromptRenderer.imageTokenID,
            beginImageTokenID: MultimodalPromptRenderer.beginImageTokenID,
            endImageTokenID: MultimodalPromptRenderer.endImageTokenID,
            rendererAddsBoundaryTokens: true)
        public static let qwen36 = FamilyPolicy(
            family: .qwen36,
            placeholder: "<|vision_start|><|image_pad|><|vision_end|>",
            imageTokenID: 248_056,
            beginImageTokenID: 248_053,
            endImageTokenID: 248_054,
            rendererAddsBoundaryTokens: false)

        /// Flash Next tokenizes the image markers to the same ids as Qwen
        /// 3.6 — 248,053 / 248,056 / 248,054 — because it inherits the same
        /// vocabulary. Only the family differs, and that is what the feature
        /// buffers are checked against.
        public static let qwen4Exp = FamilyPolicy(
            family: .qwen4Exp,
            placeholder: qwen36.placeholder,
            imageTokenID: qwen36.imageTokenID,
            beginImageTokenID: qwen36.beginImageTokenID,
            endImageTokenID: qwen36.endImageTokenID,
            rendererAddsBoundaryTokens: qwen36.rendererAddsBoundaryTokens)

        public static func forFamily(_ family: ModelFamily) -> FamilyPolicy {
            switch family {
            case .gemma4: return .gemma4
            case .qwen36: return .qwen36
            case .qwen4Exp: return .qwen4Exp
            case .gptOss, .minimaxM2, .flint:
                preconditionFailure("This model family does not accept image input")
            }
        }
    }

    public static func render(
        messages: [MultimodalMessage],
        featuresByID: [UUID: VisionFeatures],
        tokenizer: GFTokenizer,
        tools: [GFTokenizer.FunctionDefinition] = [],
        family: ModelFamily = .gemma4,
        modelVariant: ModelVariant? = nil,
        reasoning: ChatReasoning = .off
    ) throws -> MultimodalPrefillInput {
        try render(
            messages: messages,
            featuresByID: featuresByID,
            tokenizer: tokenizer,
            tools: tools,
            family: family,
            modelVariant: modelVariant,
            reasoning: reasoning,
            preserveThinking: false)
    }

    public static func render(
        messages: [MultimodalMessage],
        featuresByID: [UUID: VisionFeatures],
        tokenizer: GFTokenizer,
        tools: [GFTokenizer.FunctionDefinition] = [],
        family: ModelFamily = .gemma4,
        modelVariant: ModelVariant? = nil,
        reasoning: ChatReasoning = .off,
        preserveThinking: Bool
    ) throws -> MultimodalPrefillInput {
        let policy = FamilyPolicy.forFamily(family)
        guard !messages.isEmpty else { throw MultimodalPromptRendererError.emptyMessages }
        var orderedImages: [(UUID, VisionFeatures)] = []
        let tokenizerMessages = try messages.map { message in
            guard !message.content.isEmpty || !message.toolCalls.isEmpty else {
                throw MultimodalPromptRendererError.emptyContent
            }
            var text = ""
            for part in message.content {
                switch part {
                case .text(let value):
                    guard !value.contains(policy.placeholder),
                          !value.contains("<|image_pad|>"),
                          !value.contains("<|vision_start|>"),
                          !value.contains("<|vision_end|>") else {
                        throw MultimodalPromptRendererError.reservedImageMarker
                    }
                    text += value
                case .image(let id):
                    guard let features = featuresByID[id] else {
                        throw MultimodalPromptRendererError.missingImage(id)
                    }
                    guard features.family == family else {
                        throw MultimodalPromptRendererError.placeholderMismatch
                    }
                    text += policy.placeholder
                    orderedImages.append((id, features))
                }
            }
            return GFTokenizer.Message(
                role: message.role,
                content: text,
                thinking: message.thinking,
                toolCalls: message.toolCalls,
                toolCallID: message.toolCallID,
                name: message.name)
        }
        guard Set(orderedImages.map(\.0)) == Set(featuresByID.keys) else {
            let used = Set(orderedImages.map(\.0))
            throw MultimodalPromptRendererError.unexpectedImage(
                featuresByID.keys.first { !used.contains($0) }!)
        }

        let usesToolTemplate = !tools.isEmpty || tokenizerMessages.contains {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }
        let templateTokens: [Int32]
        if usesToolTemplate {
            templateTokens = try tokenizer.encodeToolChat(
                messages: tokenizerMessages,
                tools: tools,
                reasoning: reasoning,
                preserveThinking: preserveThinking)
        } else {
            let rendered = try tokenizer.applyChatTemplate(
                tokenizerMessages,
                modelVariant: modelVariant,
                reasoning: reasoning,
                preserveThinking: preserveThinking)
            templateTokens = tokenizer.encode(rendered, addBOS: false)
        }
        let placeholders = templateTokens.indices.filter {
            templateTokens[$0] == policy.imageTokenID
        }
        guard placeholders.count == orderedImages.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }

        var effective: [Int32] = []
        var embedding: [Int32] = []
        var spans: [MultimodalImageSpan] = []
        effective.reserveCapacity(
            templateTokens.count + orderedImages.reduce(0) { $0 + $1.1.tokenCount + 1 })
        embedding.reserveCapacity(effective.capacity)
        var imageIndex = 0
        for token in templateTokens {
            guard token == policy.imageTokenID else {
                effective.append(token)
                embedding.append(token)
                continue
            }
            let features = orderedImages[imageIndex].1
            if policy.rendererAddsBoundaryTokens {
                effective.append(policy.beginImageTokenID)
                embedding.append(policy.beginImageTokenID)
            }
            let lower = effective.count
            effective.append(contentsOf: repeatElement(
                policy.imageTokenID, count: features.tokenCount))
            embedding.append(contentsOf: repeatElement(Int32(0), count: features.tokenCount))
            spans.append(MultimodalImageSpan(
                tokenRange: lower..<effective.count,
                features: features))
            if policy.rendererAddsBoundaryTokens {
                effective.append(policy.endImageTokenID)
                embedding.append(policy.endImageTokenID)
            }
            imageIndex += 1
        }
        return try MultimodalPrefillInput(
            effectiveTokenIDs: effective,
            embeddingTokenIDs: embedding,
            imageSpans: spans,
            family: family)
    }
}
