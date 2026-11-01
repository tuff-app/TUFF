import Foundation
import Testing
import TUFFEngine
@testable import TUFFAppCore

/// The app's real run path with a scripted model and a fixture web
/// transport: a web answer runs its search, feeds the result back as a tool
/// round, and records the round and its sources with the turn.
@Suite(.serialized) struct AppModelToolTests {
    @MainActor
    private func readyModel(client: ScriptedInferenceClient,
                            transport: FixtureHTTPTransport) -> AppModel {
        let model = makeAppModel(client: client,
                                 toolStore: .inMemory(transport: transport))
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory,
                                 loadSeconds: 1)
        return model
    }

    @MainActor
    private func waitUntilIdle(_ model: AppModel) async throws {
        for _ in 0..<500 where model.isRunning {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(!model.isRunning)
    }

    @MainActor
    @Test func aWebAnswerRecordsItsRoundAndSources() async throws {
        let transport = FixtureHTTPTransport { _ in .html(DuckDuckGoFixtures.ordinary) }
        let client = ScriptedInferenceClient([
            ToolLoopFixtures.callRound([ToolLoopFixtures.search("swift actors")]),
            ToolLoopFixtures.answer("Actors isolate mutable state [1]."),
        ])
        let model = readyModel(client: client, transport: transport)
        model.webSearchEnabled = true
        #expect(model.effectiveChatCapabilities == .init(web: true))
        model.promptText = "What are Swift actors?"
        model.run()
        try await waitUntilIdle(model)

        #expect(model.error == nil)
        let turn = try #require(model.conversation.first)
        #expect(turn.response == "Actors isolate mutable state [1].")
        #expect(turn.capabilities == .init(web: true))
        #expect(turn.toolRounds.count == 1)
        #expect(turn.toolRounds.first?.results.first?.status == .succeeded)
        #expect(turn.sources.map(\.url) == [
            "https://docs.example.org/actors", "https://blog.example.net/post?id=4",
            "https://docs.example.org/reentrancy"])
        #expect(AppCitations.check(turn.response, sources: turn.sources).valid == [1])
        #expect(model.outputToolActivities.map(\.state) == [.succeeded])
        #expect(model.diagnostics?.toolRounds == 1)

        let requests = client.requests
        #expect(requests.count == 2)
        #expect(requests[0].tools.map(\.name) == ["web_search", "read_webpage"])
        #expect(requests[0].systemPrompt.contains("Cite retrieved web or file-search sources"))
        #expect(requests[0].conversationKey != nil)
        #expect(requests[1].conversationKey == requests[0].conversationKey)
        #expect(requests[1].currentRounds.count == 1)
        #expect(transport.requests.count == 1)
        #expect(!model.toolStore.inferenceActivity.isActive)
    }

    @MainActor
    @Test func withToolsOffNothingIsDeclaredOrRun() async throws {
        let transport = FixtureHTTPTransport { _ in .html(DuckDuckGoFixtures.ordinary) }
        let client = ScriptedInferenceClient([ToolLoopFixtures.answer("Plain.")])
        let model = readyModel(client: client, transport: transport)
        model.webSearchEnabled = false
        model.fileSearchEnabled = true
        // Files is on but there is no folder to search.
        #expect(model.effectiveChatCapabilities == .none)
        model.promptText = "Hello"
        model.run()
        try await waitUntilIdle(model)
        #expect(client.requests.first?.tools.isEmpty == true)
        #expect(client.requests.first?.systemPrompt.contains("Cite") == false)
        #expect(transport.requests.isEmpty)
    }

    @MainActor
    @Test func localFilesWithoutAWorkspaceGrantsNoRawFileTools() async throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-legacy-files-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try "note".write(to: folder.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: folder) }
        let transport = FixtureHTTPTransport { _ in .html(DuckDuckGoFixtures.ordinary) }
        let model = readyModel(client: ScriptedInferenceClient([]), transport: transport)
        model.toolStore.addFolders([folder])
        model.fileSearchEnabled = true
        // The upgrade path: Local Files on, a folder added, no workspace ever chosen.
        #expect(model.selectedCodingFolder == nil)
        #expect(model.effectiveChatCapabilities.files)
        #expect(!model.effectiveChatCapabilities.workspaceRead)
        #expect(AppToolCatalog.definitions(for: model.effectiveChatCapabilities).map(\.name) == ["search_files"])
        model.codingFolderID = try #require(model.toolStore.folders.first).id.uuidString
        #expect(model.effectiveChatCapabilities.workspaceRead)
        model.codingFolderID = ""
        #expect(!model.effectiveChatCapabilities.workspaceRead)
        model.toolStore.cancelIndexing()
    }

    @MainActor
    @Test func stoppingDuringASearchStopsTheAnswer() async throws {
        let transport = FixtureHTTPTransport { _ in
            try await Task.sleep(for: .seconds(30))
            return .html(DuckDuckGoFixtures.ordinary)
        }
        let client = ScriptedInferenceClient([
            ToolLoopFixtures.callRound([ToolLoopFixtures.search("slow")]),
            ToolLoopFixtures.answer("never"),
        ])
        let model = readyModel(client: client, transport: transport)
        model.webSearchEnabled = true
        model.promptText = "Search slowly"
        model.run()
        for _ in 0..<200 where model.outputToolActivities.isEmpty {
            try await Task.sleep(for: .milliseconds(10))
        }
        model.cancel()
        try await waitUntilIdle(model)
        #expect(model.error == .cancelled)
        #expect(client.requests.count == 1)
        #expect(model.conversation.first?.toolRounds.first?.results.first?.status == .cancelled)
    }

    @MainActor
    @Test func offlinePausePreservesIntent() throws {
        let model = readyModel(client: ScriptedInferenceClient([]),
            transport: FixtureHTTPTransport { _ in .html(DuckDuckGoFixtures.ordinary) })
        model.webSearchEnabled = true
        model.networkStatus.isOffline = true
        // Opt-in: existing settings continue to permit web tools.
        #expect(model.effectiveChatCapabilities.web)
        model.pauseWebSearchWhenOffline = true
        #expect(model.isWebSearchPaused)
        #expect(!model.effectiveChatCapabilities.web)
        #expect(model.webSearchEnabled)
        model.networkStatus.isOffline = false
        #expect(model.effectiveChatCapabilities.web)
        model.networkStatus.isOffline = true
        model.webSearchEnabled = false
        model.networkStatus.isOffline = false
        #expect(!model.effectiveChatCapabilities.web)
        #expect(!model.webSearchEnabled)
        model.webSearchEnabled = true
        model.networkStatus.isOffline = true
        model.pauseWebSearchWhenOffline = false
        #expect(model.effectiveChatCapabilities.web)
    }

    @MainActor
    @Test func offlineMessageDoesNotDeclareOrExecuteWebTools() async throws {
        let transport = FixtureHTTPTransport { _ in .html(DuckDuckGoFixtures.ordinary) }
        let client = ScriptedInferenceClient([ToolLoopFixtures.answer("Offline answer.")])
        let model = readyModel(client: client, transport: transport)
        model.webSearchEnabled = true
        model.pauseWebSearchWhenOffline = true
        model.networkStatus.isOffline = true
        model.promptText = "Hello"
        model.run()
        try await waitUntilIdle(model)
        #expect(client.requests.first?.tools.isEmpty == true)
        #expect(transport.requests.isEmpty)
        #expect(model.conversation.first?.capabilities.web == false)
        #expect(model.webSearchEnabled)
    }

    @MainActor
    @Test func uncheckedModelsSaySo() {
        #expect(AppToolSupport.forModel(.gemma4_26B_A4B) == .validated)
        #expect(AppToolSupport.forModel(.qwen38FlashNext) == .validated)
        #expect(AppToolSupport.forModel(.minimaxM27) == .untested)
        #expect(AppToolSupport.forModel(.minimaxM27).note != nil)
        #expect(AppToolSupport.forModel(.gptOss_120B).allowsTools)
        #expect(!AppToolSupport.unavailable("x").allowsTools)
    }
}
