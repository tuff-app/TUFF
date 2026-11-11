import Foundation
import Testing
@testable import TUFFAppCore

@Suite(.serialized) @MainActor struct AppleProviderRoutingTests {
    @Test func snapshotsPreserveIncrementallyCompletedGraphemes() throws {
        let chunks = ["Cafe", "Cafe\u{301}", "Cafe\u{301} 👩", "Cafe\u{301} 👩‍💻"]
        var previous = ""
        var result = ""
        for current in chunks {
            result += try AppleInferenceClient.snapshotDelta(previous: previous, current: current)
            previous = current
        }
        #expect(result.utf8.elementsEqual(chunks.last!.utf8))
        #expect(throws: AppInferenceError.self) {
            try AppleInferenceClient.snapshotDelta(previous: "old", current: "new")
        }
    }

    @Test func selectionAndConversationRestore() throws {
        let model = makeAppModel()
        model.selectAppleModel()
        #expect(model.selectedProviderKey == AppModel.appleModelID)
        #expect(!model.requiresModelInstallation)
        #expect(!model.canLoadModel)
        #expect(!model.canUnloadModel)
        #expect(!model.isImageInputAvailable)
        #expect(model.effectiveChatCapabilities == .none)
        #expect(!model.canContinueLastAnswer)
        model.clearOutput()
        let record = try #require(model.conversationStore.selectedConversation)
        #expect(record.modelID == AppModel.appleModelID)
        let data = try JSONEncoder().encode(record)
        let restored = try JSONDecoder().decode(AppConversationRecord.self, from: data)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = AppConversationRepository(rootURL: root)
        try repository.save(.init(selectedConversationID: restored.id, conversations: [restored]))
        let other = makeAppModel(conversationStore: AppConversationStore(repository: repository))
        #expect(other.isAppleSelected)
        let local = other.selectedInstall
        other.selectModel(local)
        #expect(!other.isAppleSelected)
        other.selectAppleModel()
        #expect(other.isAppleSelected)
        #expect(other.modelShortName(forProfileKey: restored.modelID) == AppleModelKind.onDevice.name)
    }

    @Test func unavailableCannotFallBackToEngine() throws {
        for state in AppleModelAvailability.allCases where !state.canGenerate {
            let model = makeAppModel()
            model.appleAvailabilityProvider = { state }
            model.selectAppleModel()
            model.promptText = "Hello"
            #expect(!model.canRun)
            #expect(!model.canSubmit)
            #expect(!model.requiresModelInstallation)
            #expect(!model.toolSupport.allowsTools)
            #expect(throws: AppInferenceError.self) { try model.makeRequest() }
            model.run()
            #expect(!model.isRunning)
            #expect(model.isAppleSelected)
        }
    }

    @Test func appleInstructionsDoNotChangeDownloadedModel() throws {
        let model = makeAppModel()
        let localPrompt = model.modelSystemPrompt
        model.selectAppleModel()
        model.modelSystemPrompt = "Always answer in French."
        model.promptText = "Hello"
        if model.appleAvailability.canGenerate {
            #expect(try model.makeRequest().systemPrompt == "Always answer in French.")
        }
        model.loadModel()
        #expect(!model.loadState.isLoading)
        model.selectModel(model.selectedInstall)
        #expect(model.modelSystemPrompt == localPrompt)
        model.selectAppleModel()
        #expect(model.modelSystemPrompt == "Always answer in French.")
        #expect(model.selectedModelName == AppleModelKind.onDevice.name)
    }

    @Test func deviceInventoryExcludesUnavailableEndpoints() {
        let model = makeAppModel()
        model.appleAvailabilityProvider = { .intelligenceDisabled }
        model.appleCloudAvailabilityProvider = { .updateRequired }
        model.refreshAppleAvailability()
        #expect(model.availableAppleModels.isEmpty)
        model.appleAvailabilityProvider = { .available }
        model.refreshAppleAvailability()
        #expect(model.availableAppleModels == [.onDevice])
        model.appleCloudAvailabilityProvider = { .available }
        model.refreshAppleAvailability()
        #expect(model.availableAppleModels == [.onDevice, .privateCloudCompute])
    }

    @Test func helperFailureKeepsDetailsOutOfUserMessage() {
        let raw = "ServingHelperTransportLost(message: \"unexpectedEOF\")"
        let error = AppInferenceError.unknown(raw)
        #expect(error.isEngineTransportFailure)
        #expect(!error.userMessage.contains("unexpectedEOF"))
        #expect(error.technicalDetail == raw)
        #expect(!AppInferenceError.invalidRequest("Try a shorter prompt.").isEngineTransportFailure)
    }

    @Test func liveAppModelRoute() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_TEST_APPLE_LIVE"] == "1" else { return }
        #expect(AppleModelAvailability.current() == .available)
        let model = makeAppModel()
        model.selectAppleModel()
        model.promptText = "Reply with just hello."
        #expect(model.canRun)
        model.run()
        for _ in 0..<600 where model.isRunning { try await Task.sleep(for: .milliseconds(100)) }
        #expect(!model.isRunning)
        #expect(model.error == nil)
        #expect(!model.outputText.isEmpty)
        #expect(model.conversation.last?.modelID == AppModel.appleModelID)
        #expect(model.conversationPlainText.contains(AppleModelKind.onDevice.name + ":"))
        let appleChat = try #require(model.conversationStore.selectedConversation)
        model.clearOutput()
        model.selectModel(model.selectedInstall)
        model.selectConversation(appleChat)
        #expect(model.isAppleSelected)
        model.promptText = "Write a long story about the ocean."
        model.run()
        #expect(model.isRunning)
        for _ in 0..<600 where model.outputText.isEmpty && model.isRunning { try await Task.sleep(for: .milliseconds(100)) }
        #expect(!model.outputText.isEmpty)
        #expect(model.isRunning)
        model.cancel()
        for _ in 0..<100 where model.isRunning { try await Task.sleep(for: .milliseconds(100)) }
        #expect(!model.isRunning)
        #expect(model.error == .cancelled)
        model.promptText = "Reply with just hello."
        model.run()
        for _ in 0..<600 where model.isRunning { try await Task.sleep(for: .milliseconds(100)) }
        #expect(!model.isRunning)
        #expect(model.error == nil)
        #expect(!model.outputText.isEmpty)
    }
}
