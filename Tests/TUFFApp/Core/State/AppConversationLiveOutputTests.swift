import Testing
@testable import TUFFAppCore

@Suite struct AppConversationLiveOutputTests {
    @MainActor
    @Test func switchedChatKeepsItsStreamAndCompletesItsOwnPendingTurn() throws {
        let store = AppConversationStore()
        store.startNewConversation(modelID: "gemma4-e4b")
        let firstID = try #require(store.selectedConversationID)
        store.promptText = "draft for first chat"
        store.outputPromptText = "first question"
        store.beginTurn(AppChatTurn(prompt: "first question", response: ""),
                        attachments: [], modelID: "gemma4-e4b")
        store.beginOutputUpdate(for: firstID)
        store.outputText = "partial"
        store.endOutputUpdate()

        store.startNewConversation(modelID: "gemma4-e4b")
        let secondID = try #require(store.selectedConversationID)
        store.promptText = "draft for second chat"
        store.outputPromptText = "second question"
        store.outputText = "separate output"

        store.beginOutputUpdate(for: firstID)
        store.outputText += " answer"
        store.completeTurn(response: store.outputText, thinking: nil)
        store.endOutputUpdate()

        #expect(store.selectedConversationID == secondID)
        #expect(store.outputText == "separate output")
        #expect(store.promptText == "draft for second chat")
        store.selectConversation(id: firstID)
        #expect(store.promptText == "draft for first chat")
        #expect(store.outputText == "partial answer")
        #expect(store.conversation.last?.response == "partial answer")
    }
    @MainActor
    @Test func returningToRunningChatDoesNotDuplicateItsPendingTurn() throws {
        let store = AppConversationStore()
        store.startNewConversation(modelID: "gemma4-e4b")
        let firstID = try #require(store.selectedConversationID)
        store.outputPromptText = "first question"
        store.beginTurn(AppChatTurn(prompt: "first question", response: ""),
                        attachments: [], modelID: "gemma4-e4b")
        store.outputText = "partial"
        store.startNewConversation(modelID: "gemma4-e4b")
        let secondID = try #require(store.selectedConversationID)
        store.selectConversation(id: firstID)
        #expect(store.conversation.isEmpty)
        #expect(store.outputPromptText == "first question")
        #expect(store.outputText == "partial")
        store.beginOutputUpdate(for: firstID)
        store.outputText += " answer"
        store.completeTurn(response: store.outputText, thinking: nil)
        store.endOutputUpdate()
        #expect(store.conversation.count == 1)
        #expect(store.conversation.first?.response == "partial answer")
        #expect(store.outputTurnIsRecorded)
        store.selectConversation(id: secondID)
        store.selectConversation(id: firstID)
        #expect(store.conversation.count == 1)
        #expect(store.conversation.first?.response == "partial answer")
    }

}
