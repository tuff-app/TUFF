import Foundation
import Testing
@testable import TUFFAppCore
import Synchronization

private final class SessionCleanupClient: AppSessionInferenceClient, Sendable {
    struct State: Sendable { var ended: [UUID] = []; var fail = false; var hold = false }
    let state = Mutex(State())
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
    func generate(_ request: AppGenerationRequest, sessionID: UUID, requestID: UUID)
        -> AsyncThrowingStream<AppInferenceEvent, Error> { generate(request) }
    func cancel() {}
    func cancel(requestID: UUID) {}
    func endSession(_ sessionID: UUID, requestID: UUID) async throws {
        let failure = state.withLock { value in value.ended.append(sessionID); return value.fail }
        while state.withLock({ $0.hold }) { try await Task.sleep(for: .milliseconds(5)) }
        if failure { throw CocoaError(.fileWriteUnknown) }
    }
}


@Suite(.serialized) struct AppConversationPersistenceTests {
    @MainActor
    @Test func deletingChatEndsSessionAndRetainsIdentityOnFailure() async throws {
        let store = AppConversationStore(), client = SessionCleanupClient()
        store.recordCompletedTurn(AppChatTurn(prompt: "hello", response: "hi"), attachments: [], modelID: "test")
        let record = try #require(store.selectedConversation)
        let session = store.sessionID(for: record.id)
        client.state.withLock { $0.fail = true }
        let model = makeAppModel(client: client, installer: MockModelInstallerClient(descriptor: .default),
                                 conversationStore: store)
        model.deleteConversation(record)
        for _ in 0..<200 where client.state.withLock({ $0.ended.isEmpty }) {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(client.state.withLock { $0.ended } == [session])
        #expect(store.existingSessionID(for: record.id) == session)
        client.state.withLock { $0.fail = false }
        model.reclaimIdleConversationSessions()
        for _ in 0..<200 where store.existingSessionID(for: record.id) != nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(store.existingSessionID(for: record.id) == nil)
        #expect(client.state.withLock { $0.ended } == [session, session])
    }

    @MainActor
    @Test func rebindingChatEndsOldSession() async throws {
        let store = AppConversationStore(), client = SessionCleanupClient()
        store.recordCompletedTurn(AppChatTurn(prompt: "hello", response: "hi"), attachments: [],
                                  modelID: AppModelInstallDescriptor.default.settingsProfileKey)
        let id = try #require(store.selectedConversationID)
        let session = store.sessionID(for: id)
        let qwen = ModelInstallCoordinator(descriptor: .qwen36,
            directoryURL: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString),
            client: MockModelInstallerClient(descriptor: .qwen36))
        let model = makeAppModel(client: client, installer: MockModelInstallerClient(descriptor: .default),
                                 otherInstalls: [qwen], conversationStore: store)
        model.selectModel(qwen)
        for _ in 0..<200 where store.existingSessionID(for: id) != nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(client.state.withLock { $0.ended } == [session])
        #expect(store.existingSessionID(for: id) == nil)
        #expect(store.sessionID(for: id) != session)
    }

    @MainActor
    @Test func idleCleanupBlocksNewTurnsUntilAcknowledged() async throws {
        let directory = try makeCompleteModelInstall("session-cleanup-gate")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppConversationStore(), client = SessionCleanupClient()
        store.recordCompletedTurn(AppChatTurn(prompt: "hello", response: "hi"), attachments: [],
                                  modelID: AppModelInstallDescriptor.default.settingsProfileKey)
        let id = try #require(store.selectedConversationID)
        let session = store.sessionID(for: id)
        let model = makeAppModel(client: client, installer: MockModelInstallerClient(descriptor: .default),
                                 conversationStore: store)
        model.setModelURL(directory)
        model.promptText = "next"
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        #expect(model.canRun)
        client.state.withLock { $0.hold = true }
        model.reclaimIdleConversationSessions()
        for _ in 0..<200 where client.state.withLock({ $0.ended.isEmpty }) {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(client.state.withLock { $0.ended } == [session])
        #expect(!model.canRun)
        #expect(!model.canSubmit)
        #expect(store.existingSessionID(for: id) == session)
        client.state.withLock { $0.hold = false }
        for _ in 0..<200 where store.existingSessionID(for: id) != nil {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(store.existingSessionID(for: id) == nil)
        #expect(model.canRun)
        #expect(store.sessionID(for: id) != session)
    }

    @Test func localChatRootUsesScopedApplicationSupport() {
        let root = AppConversationRepository.defaultRootURL(environment:
            ["TUFF_APPLICATION_SUPPORT_ROOT": "/tmp/tuff-isolation-fixture"])
        #expect(root.path == "/tmp/tuff-isolation-fixture/TUFF/Chats/v1")
        #expect(root != AppConversationRepository.defaultRootURL(environment: [:]))
    }
    private func makeRepository() -> (URL, AppConversationRepository) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-chats-\(UUID().uuidString)", isDirectory: true)
        return (root, AppConversationRepository(rootURL: root))
    }

    @Test func archiveRoundTripsThroughAtomicFile() throws {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let archive = AppConversationArchive(
            selectedConversationID: id,
            conversations: [AppConversationRecord(
                id: id,
                title: "A saved chat",
                modelID: "gemma4-e4b",
                createdAt: Date(timeIntervalSince1970: 1_700_000_000),
                updatedAt: Date(timeIntervalSince1970: 1_700_000_001),
                turns: [AppPersistedChatTurn(prompt: "hello", response: "hi")])])

        try repository.save(archive)

        // Without tool data the file is written at schema 2, which TUFF 7 reads.
        var written = archive
        written.schemaVersion = 2
        #expect(try repository.load() == written)
        let siblings = try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil)
        #expect(siblings.map(\.lastPathComponent) == ["conversations.json"])
    }

    @MainActor
    @Test func newerArchiveIsNeverOverwritten() throws {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let original = Data("{\"schemaVersion\":99,\"sentinel\":true}".utf8)
        try original.write(to: repository.archiveURL)

        let store = AppConversationStore(repository: repository)
        store.startNewConversation(modelID: "gemma4-e4b")

        #expect(store.persistenceError != nil)
        #expect(try Data(contentsOf: repository.archiveURL) == original)
    }

    @MainActor
    @Test func completedTurnsRestoreWithNameAndModelBinding() {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = AppConversationStore(repository: repository)
        first.recordCompletedTurn(
            AppChatTurn(
                prompt: "Explain why local models are useful",
                response: "Privacy.",
                thinking: "The user asked for one benefit."),
            attachments: [],
            modelID: "gpt-oss-20b",
            now: Date(timeIntervalSince1970: 1_700_000_000))

        let restored = AppConversationStore(repository: repository)

        #expect(restored.conversations.count == 1)
        #expect(restored.selectedConversation?.title
            == "Explain why local models are useful")
        #expect(restored.selectedConversation?.modelID == "gpt-oss-20b")
        #expect(restored.conversation == [
            AppChatTurn(
                id: restored.conversation[0].id,
                prompt: "Explain why local models are useful",
                response: "Privacy.",
                thinking: "The user asked for one benefit.",
                modelID: "gpt-oss-20b")
        ])

        let chat = restored.selectedConversation!
        restored.renameConversation(id: chat.id, title: "  local   privacy  ")
        #expect(restored.selectedConversation?.title == "local privacy")
    }

    @MainActor
    @Test func managedAttachmentsSurviveRestartAndLeaveWithChat() throws {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        let staging = AppImageAttachmentStore(directoryURL: root
            .appendingPathComponent("staging", isDirectory: true))
        let attachment = try staging.stage(
            data: Data("image fixture".utf8),
            displayName: "photo.png")
        let store = AppConversationStore(repository: repository)
        let turn = AppChatTurn(prompt: "What is this?", response: "A fixture.")

        store.recordCompletedTurn(
            turn,
            attachments: [attachment],
            modelID: "gemma4-26b-a4b")
        staging.removeAll()

        let restored = AppConversationStore(repository: repository)
        let durable = try #require(
            restored.conversation.first { $0.id == turn.id }?.images.first)
        #expect(try Data(contentsOf: durable.fileURL) == Data("image fixture".utf8))
        #expect(durable.fileURL.path.hasPrefix(repository.attachmentsRootURL.path))

        let chat = try #require(restored.selectedConversation)
        restored.deleteConversation(id: chat.id)
        #expect(!FileManager.default.fileExists(atPath: durable.fileURL.path))
        #expect(try repository.load().conversations.isEmpty)
    }

    @MainActor
    @Test func changingModelKeepsTheChatAndRebindsIt() {
        let store = AppConversationStore()
        store.recordCompletedTurn(
            AppChatTurn(prompt: "hello", response: "hi"),
            attachments: [],
            modelID: AppModelInstallDescriptor.default.settingsProfileKey)
        let qwenInstaller = MockModelInstallerClient(descriptor: .qwen36)
        let qwen = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen-switch-\(UUID().uuidString).gturbo"),
            client: qwenInstaller)
        let model = makeAppModel(
            modelDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent("gemma-switch-\(UUID().uuidString).gturbo"),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwen],
            conversationStore: store)

        model.selectModel(qwen)

        // The chat carries over rather than being replaced: same conversation,
        // same history, now bound to the model that will answer the next turn.
        #expect(model.selectedModelID == qwen.id)
        #expect(model.conversation.count == 1)
        #expect(store.conversations.count == 1)
        #expect(store.selectedConversation?.modelID
            == AppModelInstallDescriptor.qwen36.settingsProfileKey)
    }

    @MainActor
    @Test func switchingToATextOnlyModelWithImagesInHistoryIsRefused() throws {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("photo.png")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("png".utf8).write(to: image)

        let store = AppConversationStore(repository: repository)
        store.recordCompletedTurn(
            AppChatTurn(prompt: "what is this", response: "a photo"),
            attachments: [
                AppImageAttachment(
                    fileURL: image,
                    displayName: "photo.png",
                    encodedBytes: 3,
                    sha256: String(repeating: "0", count: 64)),
            ],
            modelID: AppModelInstallDescriptor.default.settingsProfileKey)
        #expect(store.selectedConversation?.turns.first?.attachments.isEmpty == false)

        // GPT-OSS has no vision tower, so it is the genuine text-only case.
        let textOnlyDescriptor = try #require(
            AppModelInstallDescriptor.descriptor(for: .gptOss_20B))
        let textOnlyInstaller = MockModelInstallerClient(descriptor: textOnlyDescriptor)
        let textOnly = ModelInstallCoordinator(
            descriptor: textOnlyDescriptor,
            directoryURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("gptoss-switch-\(UUID().uuidString).gturbo"),
            client: textOnlyInstaller)
        let model = makeAppModel(
            modelDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent("gemma-switch-\(UUID().uuidString).gturbo"),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [textOnly],
            conversationStore: store)

        let before = model.selectedModelID
        model.selectModel(textOnly)

        // Carrying images into a model that cannot read them would mean
        // answering as though they had never been sent, so this is refused.
        #expect(model.selectedModelID == before)
        #expect(model.error != nil)
        #expect(store.selectedConversation?.modelID
            == AppModelInstallDescriptor.default.settingsProfileKey)
    }

    @Test func attachmentPathsCannotEscapeManagedStorage() {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        let escaped = AppConversationAttachment(
            id: UUID(),
            relativePath: "attachments/../outside",
            displayName: "outside.png",
            encodedBytes: 1,
            sha256: String(repeating: "0", count: 64))

        #expect(throws: AppConversationPersistenceError.self) {
            try repository.resolve(escaped)
        }
    }

    /// A saved chat is continued, not just re-read: the file that the first
    /// answer was about has to be in the prompt of the next turn too, and the
    /// model that produced each answer has to stay attached to it.
    @MainActor
    @Test func filesAndTheAnsweringModelSurviveARestart() {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = AppConversationStore(repository: repository)
        first.recordCompletedTurn(
            AppChatTurn(
                prompt: "What is the total?",
                response: "Twelve dollars.",
                documents: [AppDocumentAttachment(
                    displayName: "receipt.txt", text: "Total: $12")]),
            attachments: [],
            modelID: "gemma4-e4b")

        let restored = AppConversationStore(repository: repository)

        let turn = restored.conversation.first
        #expect(turn?.documents.map(\.displayName) == ["receipt.txt"])
        #expect(turn?.documents.first?.text == "Total: $12")
        #expect(turn?.modelID == "gemma4-e4b")
        #expect(turn?.modelPrompt.contains("Total: $12") == true,
                "a continued chat lost the file the first answer was about")
    }

    /// Schema 1 archives were written before turns carried documents or a model
    /// name. Failing to decode one puts the store into read-only mode and the
    /// user's whole chat history stops saving, so absent fields have to read as
    /// empty rather than as an error.
    @MainActor
    @Test func aSchemaOneArchiveStillLoads() throws {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        let id = UUID()
        let legacy = """
        {
          "schemaVersion": 1,
          "selectedConversationID": "\(id.uuidString)",
          "conversations": [
            {
              "id": "\(id.uuidString)",
              "title": "Older chat",
              "modelID": "gemma4-e4b",
              "createdAt": "2026-01-01T00:00:00Z",
              "updatedAt": "2026-01-01T00:00:01Z",
              "turns": [
                {
                  "id": "\(UUID().uuidString)",
                  "prompt": "hello",
                  "response": "hi",
                  "attachments": []
                }
              ]
            }
          ]
        }
        """
        try Data(legacy.utf8).write(to: repository.archiveURL)

        let store = AppConversationStore(repository: repository)

        #expect(store.persistenceError == nil)
        #expect(store.conversations.count == 1)
        #expect(store.conversation.first?.prompt == "hello")
        #expect(store.conversation.first?.documents.isEmpty == true)
        #expect(store.conversation.first?.modelID == nil)
    }

    @Test func decodeServiceAcceptsManagedApplicationSupportAttachments() {
        let managed = AppConversationRepository.defaultRootURL
            .appendingPathComponent("attachments/chat/image")

        #expect(AppImageAttachmentStore.contains(managed))
        #expect(!AppImageAttachmentStore.contains(
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("unmanaged-image.png")))
    }
}
