import Foundation
import Testing
@testable import TUFFAppCore

@Suite struct AppStoresTests {
    @MainActor
    @Test func appModelProxiesConversationAndSettingsState() {
        let model = makeAppModel(installer: MockModelInstallerClient(descriptor: .default))

        model.promptText = "draft"
        model.temperature = 0.45

        #expect(model.conversationStore.promptText == "draft")
        #expect(model.settingsStore.temperature == 0.45)

        model.conversationStore.outputText = "answer"
        model.settingsStore.topK = 32

        #expect(model.outputText == "answer")
        #expect(model.topK == 32)
    }

    @MainActor
    @Test func appModelProxiesLibraryAndInferenceState() {
        let model = makeAppModel(installer: MockModelInstallerClient(descriptor: .default))

        model.modelLibraryStore.modelPathText = "/tmp/example.gturbo"
        model.inferenceStore.liveTokenCount = 12
        model.inferenceStore.runState = .running

        #expect(model.modelPathText == "/tmp/example.gturbo")
        #expect(model.liveTokenCount == 12)
        #expect(model.inferenceStore.runState == .running)
        // Running is derived from live answer runs, not from the stored state.
        #expect(!model.isRunning)
    }

    @MainActor
    @Test func storeInstancesDoNotShareMutableState() {
        let first = AppConversationStore()
        let second = AppConversationStore()

        first.promptText = "only the first store"

        #expect(second.promptText.isEmpty)
    }
}
