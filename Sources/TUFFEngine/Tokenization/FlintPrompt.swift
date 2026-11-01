import Foundation
import TUFFInferenceTypes

/// The publisher's Flint template uses role markers without ChatML turn ends.
/// User and tool text cannot inject control markers into the transcript.
enum FlintPrompt {
    static func safe(_ text: String) -> String {
        var value = text.replacingOccurrences(of: "\0", with: "")
            .replacingOccurrences(of: "<think>", with: "")
            .replacingOccurrences(of: "</think>", with: "")
        for marker in ["begin_of_text", "end_of_text", "system", "developer", "user", "assistant", "observation"] {
            value = value.replacingOccurrences(of: "<|\(marker)|>", with: "&lt;|\(marker)|&gt;")
        }
        return value
    }

    static func render(messages: [InferenceChatMessage], tools: [InferenceFunctionDefinition],
                       preserveThinking: Bool, addGenerationPrompt: Bool) throws -> String {
        var text = "", toolsPending = !tools.isEmpty
        func preamble() throws -> String {
            guard !tools.isEmpty else { return "" }
            var text = "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in tools {
                let value: JSONValue = .object(["name": .string(tool.name),
                    "description": .string(tool.description), "parameters": tool.parameters])
                text += "\n" + (try value.encoded())
            }
            return text + "\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n"
        }
        for message in messages {
            if toolsPending && message.role != .system { text += try preamble(); toolsPending = false }
            switch message.role {
            case .system: text += "<|system|>" + safe(message.content ?? "")
            case .developer: text += "<|developer|>" + safe(message.content ?? "")
            case .user: text += "<|user|>" + safe(message.content ?? "")
            case .tool: text += "<|observation|><tool_response>" + safe(message.content ?? "") + "</tool_response>"
            case .assistant:
                text += "<|assistant|>"
                if preserveThinking, let thinking = message.thinking, !thinking.isEmpty {
                    text += "<think>" + thinking + "</think>"
                }
                text += message.content ?? ""
                for call in message.toolCalls {
                    guard case .object(let arguments) = call.arguments else {
                        throw GFTokenizerError.invalidChatTemplate("Flint tool arguments must be an object")
                    }
                    text += "<tool_call>" + call.name
                    for key in arguments.keys.sorted() {
                        let value = arguments[key]!
                        let body: String
                        if case .string(let literal) = value { body = literal }
                        else { body = try value.encoded() }
                        text += "<arg_key>" + key + "</arg_key><arg_value>" + body + "</arg_value>"
                    }
                    text += "</tool_call>"
                }
            }
        }
        if toolsPending { text += try preamble() }
        if addGenerationPrompt { text += "<|assistant|>" }
        return text
    }
}
