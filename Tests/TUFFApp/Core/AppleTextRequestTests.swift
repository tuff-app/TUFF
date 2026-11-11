import Foundation
import Testing
import TUFFInferenceTypes
@testable import TUFFAppCore

@Suite struct AppleTextRequestTests {
    @Test func preservesInstructionsAndHistory() throws {
        let request = try AppleTextRequest.make(messages: [
            .init(role: .system, content: "Be brief."), .init(role: .developer, content: "Use English."),
            .init(role: .user, content: "Remember blue"), .init(role: .assistant, content: "OK"),
            .init(role: .user, content: "Which color?")], maxNewTokens: 32, temperature: 0)
        #expect(request.systemPrompt == "Be brief.\n\nUse English.")
        #expect(request.history.map(\.prompt) == ["Remember blue"])
        #expect(request.history.map(\.response) == ["OK"])
        #expect(request.prompt == "Which color?")
    }
    @Test func refusesUnrepresentableConversations() {
        for messages: [InferenceChatMessage] in [[], [.init(role: .assistant, content: "prefix")],
            [.init(role: .user, content: "one"), .init(role: .user, content: "two")],
            [.init(role: .user, content: "one"), .init(role: .system, content: "late")]] {
            #expect(throws: AppInferenceError.self) {
                try AppleTextRequest.make(messages: messages, maxNewTokens: 16, temperature: 0)
            }
        }
    }
}
