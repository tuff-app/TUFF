import Foundation
import TUFFInferenceTypes

/// Shared text conversation mapping for the app's Apple adapter, CLI and API.
public enum AppleTextRequest {
    public static func make(messages: [InferenceChatMessage], maxNewTokens: Int,
                            temperature: Float) throws -> AppGenerationRequest {
        func invalid(_ message: String) -> AppInferenceError { .invalidRequest(message) }
        guard maxNewTokens > 0, temperature.isFinite, (0...2).contains(temperature) else {
            throw invalid("Apple requires a positive output limit and temperature between 0 and 2.")
        }
        var instructions: [String] = [], history: [AppChatTurn] = []
        var pending: String?
        for message in messages {
            guard message.toolCalls.isEmpty, message.toolCallID == nil,
                  message.thinking?.isEmpty ?? true else {
                throw invalid("Apple supports text conversations without tools or reasoning history.")
            }
            let text = message.content ?? ""
            switch message.role {
            case .system, .developer:
                guard history.isEmpty, pending == nil else {
                    throw invalid("System instructions must precede the conversation.")
                }
                instructions.append(text)
            case .user:
                guard pending == nil else { throw invalid("User and assistant messages must alternate.") }
                pending = text
            case .assistant:
                guard let prompt = pending else { throw invalid("An assistant reply needs a preceding user message.") }
                history.append(AppChatTurn(prompt: prompt, response: text))
                pending = nil
            case .tool:
                throw invalid("Apple tool messages are unsupported.")
            }
        }
        guard let prompt = pending else { throw invalid("Apple conversations must end with a user message.") }
        return AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/"), prompt: prompt,
            systemPrompt: instructions.joined(separator: "\n\n"), history: history,
            maxNewTokens: maxNewTokens, temperature: temperature, topK: nil, topP: nil)
    }
}
