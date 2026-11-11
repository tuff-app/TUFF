import Foundation
import TUFFAppCore

/// Apple owns model loading; this backend never starts a TUFF engine helper.
struct AppleServerInferenceBackend: ServerInferenceBackend {
    let client: any AppInferenceClient
    init(client: any AppInferenceClient = AppleInferenceClient()) { self.client = client }
    var visionCapability: String { "unsupported" }

    static func request(_ request: ValidatedChatRequest) throws -> AppGenerationRequest {
        guard request.tools.isEmpty, request.imageFiles.isEmpty,
              request.multimodalMessages?.contains(where: { message in
                  message.content.contains { if case .image = $0 { return true }; return false }
              }) != true, request.reasoning == .off, request.reasoningEffort == nil,
              !request.preserveThinking, request.generationConfig.stopStrings.isEmpty,
              request.generationConfig.topK == nil, request.generationConfig.topP == nil,
              request.generationConfig.seed == nil, request.generationConfig.repetitionPenalty == 1 else {
            throw ServerRequestError.invalid(message: "Apple supports text chat, system instructions, temperature and output limits; images, tools, reasoning, stops, seeds and repetition penalties are unsupported.", param: "messages", code: "unsupported_value")
        }
        do {
            return try AppleTextRequest.make(messages: request.messages,
                maxNewTokens: request.maximumCompletionTokens,
                temperature: request.generationConfig.temperature)
        } catch {
            throw ServerRequestError.invalid(message: String(describing: error), param: "messages", code: "unsupported_value")
        }
    }

    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        _ = try Self.request(request)
        return ServerPreparedRequest(request: request)
    }

    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        let input = try Self.request(request)
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            var content = ""
            for try await event in client.generate(input) {
                try Task.checkCancellation()
                switch event {
                case .token(let token):
                    content += token.textDelta
                    onEvent(.content(token.textDelta))
                case .failed(let error, _): throw error
                case .cancelled: throw CancellationError()
                default: break
                }
            }
            try Task.checkCancellation()
            return ServerCompletion(content: content, toolCalls: [], finishReason: "stop",
                usage: OpenAIUsage(promptTokens: 0, completionTokens: 0, totalTokens: 0,
                                   tokenCountsAvailable: false))
        } onCancel: { client.cancel() }
    }
}
