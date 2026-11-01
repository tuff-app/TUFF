import Foundation
import Testing
@testable import TUFFAppCore

/// Chats keep generating on their own models while the selection moves.
@Suite struct AppModelPooledClientTests {
    @MainActor
    private func twoModelApp(client: any AppInferenceClient) throws
        -> (model: AppModel, first: URL, second: URL, secondInstall: ModelInstallCoordinator) {
        let first = try makeCompleteModelInstall("pool-a")
        let second = try makeCompleteModelInstall("pool-b", stampedAs: .qwen36)
        let install = ModelInstallCoordinator(
            descriptor: .qwen36, directoryURL: second,
            client: MockModelInstallerClient(descriptor: .qwen36))
        let model = makeAppModel(modelDirectory: first, client: client, otherInstalls: [install])
        return (model, first, second, install)
    }

    @MainActor
    private func waitUntil(_ condition: @MainActor () -> Bool) async {
        for _ in 0..<400 where !condition() { try? await Task.sleep(for: .milliseconds(5)) }
    }

    @MainActor
    @Test func switchingModelsNeverCancelsOrUnloadsABackgroundRun() async throws {
        let client = FakePooledInferenceClient()
        let (model, first, second, secondInstall) = try twoModelApp(client: client)
        defer { try? FileManager.default.removeItem(at: first); try? FileManager.default.removeItem(at: second) }
        model.loadModel()
        await waitUntil { model.loadState.isReady }
        #expect(model.loadState.isReady)

        model.promptText = "first chat"
        model.run()
        await waitUntil { client.activeCount == 1 }
        let firstChat = try #require(model.conversationStore.selectedConversationID)
        #expect(model.isSelectedConversationRunning)
        // The running chat's own model cannot be swapped under it.
        #expect(!model.canSelectModel(secondInstall))

        // A new chat is idle, so the selection can move to another model
        // while the first chat keeps generating.
        model.clearOutput()
        #expect(!model.isSelectedConversationRunning)
        #expect(model.isRunning)
        #expect(model.canSelectModel(secondInstall))
        model.selectModel(secondInstall)
        #expect(model.selectedModelID == secondInstall.id)
        #expect(model.loadState == .notLoaded)
        #expect(client.targetedUnloads.isEmpty && client.globalUnloads == 0,
                "the previous model stays resident for the background chat")
        #expect(client.activeCount(for: first) == 1)

        // Loading the second model is allowed while the first model's run is live.
        #expect(model.canLoadModel)
        model.loadModel()
        await waitUntil { model.loadState.isReady }
        #expect(model.loadState.isReady)
        #expect(client.isResident(first) && client.isResident(second))
        #expect(client.activeCount(for: first) == 1, "the background run was not interrupted")

        // Both chats generate at once, each on its own model.
        model.promptText = "second chat"
        #expect(model.canRun)
        model.run()
        await waitUntil { client.activeCount == 2 }
        #expect(client.activeCount(for: first) == 1 && client.activeCount(for: second) == 1)
        #expect(Set(client.started.map(\.sessionID)).count == 2)

        // The background chat finishes into its own conversation.
        client.release(directory: first, text: "answer one")
        await waitUntil { client.activeCount == 1 }
        let record = try #require(model.conversationStore.conversations.first { $0.id == firstChat })
        for _ in 0..<200 where record.turns.last?.response.isEmpty != false {
            try? await Task.sleep(for: .milliseconds(5))
        }
        let finished = try #require(model.conversationStore.conversations.first { $0.id == firstChat })
        #expect(finished.turns.last?.response == "answer one")
        #expect(model.isSelectedConversationRunning, "the selected chat is still answering")

        client.release(directory: second, text: "answer two")
        await waitUntil { !model.isRunning }
        #expect(!model.isRunning)

        // Selecting the first model again finds it resident in the pool.
        let firstInstall = try #require(model.installs.first { $0.directoryURL.standardizedFileURL == first.standardizedFileURL })
        model.clearOutput()
        model.selectModel(firstInstall)
        #expect(model.loadState.isReady)
    }

    @MainActor
    @Test func aSingleModelClientStillRefusesToSwitchModelsUnderARunningChat() async throws {
        let client = MockLifecycleInferenceClient()
        let (model, first, second, secondInstall) = try twoModelApp(client: client)
        defer { try? FileManager.default.removeItem(at: first); try? FileManager.default.removeItem(at: second) }
        model.loadModel()
        await client.waitForLoadStart()
        for _ in 0..<200 where !model.loadState.isReady { try? await Task.sleep(for: .milliseconds(5)) }
        model.promptText = "go"
        model.run()
        #expect(!model.canSelectModel(secondInstall))
    }
}
