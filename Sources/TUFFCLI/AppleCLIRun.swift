import Foundation
import TUFFAppCore
import TUFFCLIArguments
import TUFFInferenceTypes

func runApple(args: Args, stdout: FileHandle, stderr: FileHandle) async -> RunResult {
    do {
        guard args.images.isEmpty, args.visionPack == nil, args.thinking == .off,
              args.reasoningEffort == nil, args.stops.isEmpty, args.seed == nil,
              args.repetitionPenalty == 1, args.speculativeMode == .off,
              (args.topK == nil || args.topK == 64), (args.topP == nil || args.topP == 0.95),
              args.automaticMemory != true else {
            throw AppInferenceError.invalidRequest("Apple supports text, system instructions, temperature and output limits; images, reasoning controls, stop strings, seeds, penalties and engine controls are unavailable.")
        }
        var messages: [InferenceChatMessage]
        if let file = args.messagesFile {
            messages = try JSONDecoder().decode([AppleCLIMessage].self,
                from: Data(contentsOf: URL(fileURLWithPath: file))).map(\.message)
        } else {
            messages = args.systemPrompt.map { [.init(role: .system, content: $0)] } ?? []
            guard let prompt = args.chatPrompt ?? args.prompt else { throw ArgsError.modeMissing }
            messages.append(.init(role: .user, content: prompt))
        }
        let request = try AppleTextRequest.make(messages: messages, maxNewTokens: args.maxNew,
                                                temperature: args.temperature)
        let client = AppleInferenceClient()
        try await withTaskCancellationHandler {
            try Task.checkCancellation()
            for try await event in client.generate(request) {
                try Task.checkCancellation()
                switch event {
                case .token(let token): try stdout.write(contentsOf: Data(token.textDelta.utf8))
                case .failed(let error, _): throw error
                case .cancelled: throw CancellationError()
                default: break
                }
            }
            try Task.checkCancellation()
            try stdout.write(contentsOf: Data("\n".utf8))
        } onCancel: { client.cancel() }
        return RunResult(exitCode: 0)
    } catch {
        try? stderr.write(contentsOf: Data("error: \(error)\n".utf8))
        return RunResult(exitCode: Task.isCancelled ? 130 : 1)
    }
}

private struct AppleCLIMessage: Decodable {
    let message: InferenceChatMessage
    enum CodingKeys: String, CodingKey { case role, content, toolCalls = "tool_calls", toolCallID = "tool_call_id", reasoning = "reasoning_content" }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        guard !values.contains(.toolCalls), !values.contains(.toolCallID), !values.contains(.reasoning) else {
            throw AppInferenceError.invalidRequest("Apple messages files support text conversations without tool or reasoning fields.")
        }
        message = .init(role: try values.decode(InferenceChatRole.self, forKey: .role),
                        content: try values.decode(String.self, forKey: .content))
    }
}
