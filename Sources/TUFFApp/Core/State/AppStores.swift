import Foundation
import Observation
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFModelMetadata
import TUFFModelCatalog

public enum AppRunState: Equatable, Sendable {
    case idle
    case running
}

/// Chat drafts, transcript state, and attachment ownership. Persistence is
/// layered onto this store so the UI does not have to know where chats live.
@MainActor
@Observable
public final class AppConversationStore {
    private struct DraftState {
        var promptText = ""
        var images: [AppImageAttachment] = []
        var documents: [AppDocumentAttachment] = []
        var imageError: String?
        var documentNotice: String?
        var addingImagesCount = 0
        var isEmpty: Bool {
            promptText.isEmpty && images.isEmpty && documents.isEmpty
                && imageError == nil && documentNotice == nil
        }
    }
    private var draftByConversation: [UUID: DraftState] = [:]
    private var unscopedDraft = DraftState()
    private var draftState: DraftState {
        get {
            if let selectedConversationID {
                return draftByConversation[selectedConversationID] ?? DraftState()
            }
            return unscopedDraft
        }
        set {
            if let selectedConversationID { draftByConversation[selectedConversationID] = newValue }
            else { unscopedDraft = newValue }
        }
    }
    public var promptText: String {
        get { draftState.promptText }
        set { var state = draftState; state.promptText = newValue; draftState = state }
    }
    public var imageAttachments: [AppImageAttachment] {
        get { draftState.images }
        set { var state = draftState; state.images = newValue; draftState = state }
    }
    public var documentAttachments: [AppDocumentAttachment] {
        get { draftState.documents }
        set { var state = draftState; state.documents = newValue; draftState = state }
    }
    public var imageAttachmentError: String? {
        get { draftState.imageError }
        set { var state = draftState; state.imageError = newValue; draftState = state }
    }
    public var documentAttachmentNotice: String? {
        get { draftState.documentNotice }
        set { var state = draftState; state.documentNotice = newValue; draftState = state }
    }
    var addingImagesCount: Int {
        get { draftState.addingImagesCount }
        set { var state = draftState; state.addingImagesCount = newValue; draftState = state }
    }
    private struct OutputState {
        var promptText = ""
        var assistantPrefix = ""
        var images: [AppImageAttachment] = []
        var documents: [AppDocumentAttachment] = []
        var text = ""
        var thinking = ""
        var toolActivities: [AppToolActivity] = []
        var toolRounds: [AppToolRound] = []
        var sources: [AppSource] = []
        var capabilities: AppChatCapabilities = .none
        var turnIsRecorded = false
    }

    private var outputByConversation: [UUID: OutputState] = [:]
    private var unscopedOutput = OutputState()
    private var outputWriteTargetID: UUID?
    private var sessionIDs: [UUID: UUID] = [:]

    private var outputState: OutputState {
        get {
            if let id = outputWriteTargetID ?? selectedConversationID {
                return outputByConversation[id] ?? OutputState()
            }
            return unscopedOutput
        }
        set {
            if let id = outputWriteTargetID ?? selectedConversationID {
                outputByConversation[id] = newValue
            } else {
                unscopedOutput = newValue
            }
        }
    }

    public var outputPromptText: String {
        get { outputState.promptText }
        set { var state = outputState; state.promptText = newValue; outputState = state }
    }
    public var outputAssistantPrefix: String {
        get { outputState.assistantPrefix }
        set { var state = outputState; state.assistantPrefix = newValue; outputState = state }
    }
    public var outputImageAttachments: [AppImageAttachment] {
        get { outputState.images }
        set { var state = outputState; state.images = newValue; outputState = state }
    }
    public var outputDocumentAttachments: [AppDocumentAttachment] {
        get { outputState.documents }
        set { var state = outputState; state.documents = newValue; outputState = state }
    }
    /// Hard links this session's runs staged so the transcript keeps showing
    /// what was sent. Released when the conversation itself moves on, not when
    /// a message is folded into history: a chat whose images could not be saved
    /// to the archive is still drawing — and still sending — these.
    var retainedRunImages: [AppImageAttachment] = []
    /// Whether the exchange in the output slot has already been folded into
    /// `conversation`. The slot is kept afterwards — the copy commands and the
    /// diagnostics still describe the last run — so the transcript needs to be
    /// told, or it draws the same exchange twice.
    public internal(set) var outputTurnIsRecorded: Bool {
        get { outputState.turnIsRecorded }
        set { var state = outputState; state.turnIsRecorded = newValue; outputState = state }
    }
    public var outputText: String {
        get { outputState.text }
        set { var state = outputState; state.text = newValue; outputState = state }
    }
    public var outputThinkingText: String {
        get { outputState.thinking }
        set { var state = outputState; state.thinking = newValue; outputState = state }
    }
    /// Tool work for the answer in the output slot: what is running or ran,
    /// the finished rounds, and the sources found so far.
    public var outputToolActivities: [AppToolActivity] {
        get { outputState.toolActivities }
        set { var state = outputState; state.toolActivities = newValue; outputState = state }
    }
    public var outputToolRounds: [AppToolRound] {
        get { outputState.toolRounds }
        set { var state = outputState; state.toolRounds = newValue; outputState = state }
    }
    public var outputSources: [AppSource] {
        get { outputState.sources }
        set { var state = outputState; state.sources = newValue; outputState = state }
    }
    public var outputCapabilities: AppChatCapabilities {
        get { outputState.capabilities }
        set { var state = outputState; state.capabilities = newValue; outputState = state }
    }
    public var conversation: [AppChatTurn] = []
    public var runIdentity = 0
    public private(set) var conversations: [AppConversationRecord] = []
    public private(set) var selectedConversationID: UUID?
    public private(set) var persistenceError: String?

    private let repository: AppConversationRepository?
    private var archive = AppConversationArchive()
    private var persistenceIsReadOnly = false

    public init(repository: AppConversationRepository? = nil) {
        self.repository = repository
        guard let repository else { return }
        do {
            archive = try repository.load()
            conversations = archive.conversations.sorted { $0.updatedAt > $1.updatedAt }
            selectedConversationID = archive.selectedConversationID.flatMap { selected in
                conversations.contains(where: { $0.id == selected }) ? selected : nil
            } ?? conversations.first?.id
            restoreSelectedTurns()
        } catch {
            // Do not overwrite an archive this build cannot understand or
            // decode. The in-memory chat remains usable and the error is shown
            // in Settings/Chat once that screen lands.
            persistenceError = String(describing: error)
            persistenceIsReadOnly = true
        }
    }

    public static func persistentDefault() -> AppConversationStore {
        AppConversationStore(repository: AppConversationRepository())
    }

    public var selectedConversation: AppConversationRecord? {
        guard let selectedConversationID else { return nil }
        return conversations.first { $0.id == selectedConversationID }
    }

    public var hasCompletedMessages: Bool {
        !conversation.isEmpty
    }

    public func startNewConversation(modelID: String, now: Date = Date()) {
        let record = AppConversationRecord(modelID: modelID, createdAt: now, updatedAt: now)
        if selectedConversationID == nil, !unscopedDraft.isEmpty {
            draftByConversation[record.id] = unscopedDraft
            unscopedDraft = DraftState()
        }
        discardEmptyConversations(keeping: record.id)
        conversations.insert(record, at: 0)
        selectedConversationID = record.id
        conversation = []
        clearPendingTurn()
        outputPromptText = ""
        outputImageAttachments = []
        outputDocumentAttachments = []
        outputText = ""
        outputThinkingText = ""
        clearOutputTools()
        outputTurnIsRecorded = false
        persistArchive()
    }

    /// Point the selected conversation at another model, keeping its history.
    /// Switching models mid-chat re-renders that history through the new
    /// model's prompt format on the next turn, so the chat continues rather
    /// than starting over.
    public func rebindConversation(to modelID: String) {
        guard let selectedConversationID,
              let index = conversations.firstIndex(where: { $0.id == selectedConversationID })
        else { return }
        conversations[index].modelID = modelID
        conversations[index].updatedAt = Date()
        moveConversationToFront(at: index)
        persistArchive()
    }

    public func bindEmptyConversation(to modelID: String) {
        guard let selectedConversationID,
              let index = conversations.firstIndex(where: { $0.id == selectedConversationID }),
              conversations[index].turns.isEmpty else { return }
        conversations[index].modelID = modelID
        conversations[index].updatedAt = Date()
        moveConversationToFront(at: index)
        persistArchive()
    }

    public func selectConversation(id: UUID) {
        guard conversations.contains(where: { $0.id == id }) else { return }
        discardEmptyConversations(keeping: id)
        selectedConversationID = id
        restoreSelectedTurns()
        persistArchive()
    }

    /// Drops every turn from `turnID` onward, and hands back the one that was
    /// removed so it can be sent again.
    ///
    /// This is what regenerate and edit-and-resend are built on: an answer is
    /// replaced by rewinding to the message that produced it, rather than by
    /// mutating a turn in place.
    @discardableResult
    public func truncate(from turnID: UUID) -> AppPersistedChatTurn? {
        guard let selectedConversationID,
              let index = conversations.firstIndex(where: { $0.id == selectedConversationID }),
              let turnIndex = conversations[index].turns.firstIndex(
                where: { $0.id == turnID }) else { return nil }
        let removed = conversations[index].turns[turnIndex]
        conversations[index].turns.removeSubrange(turnIndex...)
        conversations[index].updatedAt = Date()
        // The saved state still holds the removed turns.
        removeSavedState(conversationID: selectedConversationID)
        // The attachment files stay: an image is shared by every turn that
        // carries it forward, and this chat may still hold earlier ones. The
        // directory goes when the chat does.
        restoreSelectedTurns()
        outputPromptText = ""
        outputImageAttachments = []
        outputDocumentAttachments = []
        outputText = ""
        outputThinkingText = ""
        clearOutputTools()
        outputTurnIsRecorded = false
        clearPendingTurn()
        persistArchive()
        return removed
    }

    public func renameConversation(id: UUID, title: String) {
        let cleaned = Self.cleanedTitle(title)
        guard !cleaned.isEmpty,
              let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[index].title = cleaned
        conversations[index].updatedAt = Date()
        moveConversationToFront(at: index)
        persistArchive()
    }

    public func deleteConversation(id: UUID) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations.remove(at: index)
        outputByConversation.removeValue(forKey: id)
        draftByConversation.removeValue(forKey: id)
        pendingTurns.removeValue(forKey: id)
        if selectedConversationID == id {
            selectedConversationID = conversations.first?.id
            outputPromptText = ""
            outputImageAttachments = []
            outputDocumentAttachments = []
            outputText = ""
            outputThinkingText = ""
            clearOutputTools()
            outputTurnIsRecorded = false
            restoreSelectedTurns()
        }
        do {
            try repository?.deleteAttachments(conversationID: id)
        } catch {
            persistenceError = String(describing: error)
        }
        removeSavedState(conversationID: id)
        persistArchive()
    }

    /// The engine saves a chat's computed state so it can resume after a
    /// restart. That state is the conversation in another form, so it goes
    /// when the conversation does.
    private func removeSavedState(conversationID: UUID) {
        let key = conversationID.uuidString
        Task.detached(priority: .utility) {
            TUFFPrefixSnapshotLocation.removeChatSnapshots(conversationKey: key)
        }
    }

    /// The turn that has been sent but not yet answered, and the chat it went
    /// into. Nil whenever no run is in flight.
    private struct PendingTurn {
        let turnID: UUID
        let attachments: [AppConversationAttachment]
        let stagedAttachments: [AppImageAttachment]
    }
    private var pendingTurns: [UUID: PendingTurn] = [:]

    /// Saves a message the moment it is sent, before a single token comes back.
    ///
    /// A chat used to come into existence only when an answer finished, so a
    /// run stopped halfway — or one the app was quit during — left nothing at
    /// all: no chat in the sidebar, no title, nothing on disk. Stopping the
    /// model was indistinguishable from never having asked.
    ///
    /// The turn is written to the archive here and stays out of `conversation`,
    /// which is the transcript's history: the message being generated is drawn
    /// from the live output slot instead, so it keeps the streaming fast path.
    /// `completeTurn` moves it across when the run ends.
    public func beginTurn(
        _ turn: AppChatTurn,
        attachments: [AppImageAttachment],
        modelID: String,
        now: Date = Date()
    ) {
        var index: Int
        if let selectedConversationID,
           let selected = conversations.firstIndex(where: { $0.id == selectedConversationID }),
           conversations[selected].modelID == modelID {
            index = selected
        } else {
            let record = AppConversationRecord(
                modelID: modelID, createdAt: now, updatedAt: now)
            // The caller filled the output slot before the chat existed.
            // Carry that slot to the new chat instead of dropping it with the
            // slot of whichever chat was selected.
            // The composer draft (text, files, images) travels the same way:
            // what was composed is what is being sent.
            let pending = outputState
            let draft: DraftState
            if let previous = selectedConversationID {
                outputByConversation.removeValue(forKey: previous)
                draft = draftByConversation.removeValue(forKey: previous) ?? DraftState()
            } else {
                unscopedOutput = OutputState()
                draft = unscopedDraft
                unscopedDraft = DraftState()
            }
            conversations.insert(record, at: 0)
            selectedConversationID = record.id
            outputByConversation[record.id] = pending
            draftByConversation[record.id] = draft
            index = 0
        }

        let durableAttachments: [AppConversationAttachment]
        do {
            durableAttachments = try repository?.persistAttachments(
                attachments,
                conversationID: conversations[index].id) ?? []
        } catch {
            // The text is still useful and remains model context. Image
            // persistence fails visibly instead of pretending the images were
            // saved when their managed copies do not exist.
            persistenceError = String(describing: error)
            durableAttachments = []
        }

        conversations[index].turns.append(AppPersistedChatTurn(
            id: turn.id,
            prompt: turn.prompt,
            response: "",
            thinking: nil,
            attachments: durableAttachments,
            documents: turn.documents,
            modelID: modelID,
            capabilities: turn.capabilities))
        if conversations[index].title == "New Chat" {
            conversations[index].title = Self.automaticTitle(for: Self.titleSource(turn))
        }
        conversations[index].updatedAt = now
        moveConversationToFront(at: index)
        let conversationID = conversations.first {
            $0.turns.contains(where: { $0.id == turn.id })
        }?.id ?? conversations[0].id
        pendingTurns[conversationID] = PendingTurn(
            turnID: turn.id, attachments: durableAttachments,
            stagedAttachments: attachments)
        persistArchive()
    }

    /// Files the answer, whatever the run produced, and folds the turn into
    /// history.
    ///
    /// Called for every terminal outcome, not only a clean finish: an answer
    /// cut off halfway is what the reader has, and throwing it away was the
    /// behaviour this replaced.
    public func clearOutputTools() {
        outputToolActivities = []
        outputToolRounds = []
        outputSources = []
        outputCapabilities = .none
    }

    /// The next citation number in the selected chat: sources are numbered
    /// across the whole chat so a number never means two things.
    public var nextSourceID: Int {
        (conversation.flatMap(\.sources).map(\.id).max() ?? 0) + 1
    }

    public func completeTurn(
        response: String,
        thinking: String?,
        toolRounds: [AppToolRound] = [],
        sources: [AppSource] = [],
        now: Date = Date()
    ) {
        let targetID = outputWriteTargetID ?? selectedConversationID
        guard let targetID, let pending = pendingTurns[targetID],
              let index = conversations.firstIndex(
                where: { $0.id == targetID }),
              let turnIndex = conversations[index].turns.firstIndex(
                where: { $0.id == pending.turnID }) else { return }
        conversations[index].turns[turnIndex].response = response
        conversations[index].turns[turnIndex].thinking = thinking
        conversations[index].turns[turnIndex].toolRounds = toolRounds
        conversations[index].turns[turnIndex].sources = sources
        conversations[index].updatedAt = now
        moveConversationToFront(at: index)

        // The saved copies, not the run's staged links: the next run releases
        // those, and the turn has to keep showing — and re-sending — its images
        // long after that. When nothing durable was written, the staged copies
        // are all there is, which is the same reach the display had before.
        var recorded = conversations.first { $0.id == targetID }?
            .turns.first { $0.id == pending.turnID }?.chatTurn
            ?? AppChatTurn(prompt: "", response: response)
        let resolved = pending.attachments.compactMap { try? repository?.resolve($0) }
        recorded.images = resolved.count == pending.stagedAttachments.count
            ? resolved : pending.stagedAttachments
        if selectedConversationID == targetID { conversation.append(recorded) }
        outputTurnIsRecorded = true
        clearPendingTurn(conversationID: targetID)
        persistArchive()
    }

    private func clearPendingTurn(conversationID: UUID? = nil) {
        let id = conversationID ?? outputWriteTargetID ?? selectedConversationID
        if let id { pendingTurns.removeValue(forKey: id) }
    }

    func beginOutputUpdate(for conversationID: UUID?) { outputWriteTargetID = conversationID }
    func endOutputUpdate() { outputWriteTargetID = nil }
    public func sessionID(for conversationID: UUID) -> UUID {
        if let existing = sessionIDs[conversationID] { return existing }
        let id = UUID()
        sessionIDs[conversationID] = id
        return id
    }

    func existingSessionID(for conversationID: UUID) -> UUID? { sessionIDs[conversationID] }
    var existingSessionIDs: [UUID: UUID] { sessionIDs }
    func forgetSessionID(_ sessionID: UUID, for conversationID: UUID) {
        if sessionIDs[conversationID] == sessionID { sessionIDs.removeValue(forKey: conversationID) }
    }

    /// A whole exchange at once, for a caller that already has the answer.
    ///
    /// The app always goes through `beginTurn` and `completeTurn`, because a run
    /// has a gap between the two. This is the same thing with no gap.
    public func recordCompletedTurn(
        _ turn: AppChatTurn,
        attachments: [AppImageAttachment],
        modelID: String,
        now: Date = Date()
    ) {
        beginTurn(turn, attachments: attachments, modelID: modelID, now: now)
        completeTurn(response: turn.response, thinking: turn.thinking, now: now)
    }

    /// What a chat is named after: the typed message, or the first thing
    /// attached when nothing was typed.
    private static func titleSource(_ turn: AppChatTurn) -> String {
        if !turn.prompt.isEmpty { return turn.prompt }
        if let document = turn.documents.first { return document.displayName }
        return turn.images.first?.displayName ?? ""
    }

    /// Drops saved chats that never received a turn, except the one being
    /// opened.
    ///
    /// A chat is written to disk the moment New Chat is pressed, so one
    /// abandoned before its first message stayed in the list forever as another
    /// identical row called "New Chat". That was survivable while the list only
    /// appeared on the Chat screen; now that it is always on screen, a week of
    /// use buries the real chats under them. A chat with a turn in it is never
    /// touched.
    private func discardEmptyConversations(keeping keptID: UUID?) {
        let discarded = conversations.filter {
            $0.turns.isEmpty && $0.id != keptID
                && (draftByConversation[$0.id]?.isEmpty ?? true)
        }
        guard !discarded.isEmpty else { return }
        conversations.removeAll { $0.turns.isEmpty && $0.id != keptID }
        for record in discarded {
            outputByConversation.removeValue(forKey: record.id)
            draftByConversation.removeValue(forKey: record.id)
            pendingTurns.removeValue(forKey: record.id)
            // An empty chat has recorded no turn, so it can own no attachment.
            // Asking anyway costs nothing and cannot strand one.
            try? repository?.deleteAttachments(conversationID: record.id)
        }
        if let current = selectedConversationID,
           !conversations.contains(where: { $0.id == current }) {
            selectedConversationID = keptID ?? conversations.first?.id
        }
    }

    private func restoreSelectedTurns() {
        guard let record = selectedConversation else {
            conversation = []
            return
        }
        // A pending turn is already displayed by this chat's live output
        // slot. Restoring it as history duplicates it on a switch back and
        // appends another copy when the answer finishes.
        let pendingID = pendingTurns[record.id]?.turnID
        conversation = record.turns.filter { $0.id != pendingID }.map { persisted in
            var turn = persisted.chatTurn
            turn.images = persisted.attachments.compactMap {
                try? repository?.resolve($0)
            }
            return turn
        }
    }

    private func persistArchive() {
        guard let repository, !persistenceIsReadOnly else { return }
        archive = AppConversationArchive(
            selectedConversationID: selectedConversationID,
            conversations: conversations)
        do {
            try repository.save(archive)
        } catch {
            persistenceError = String(describing: error)
            persistenceIsReadOnly = true
        }
    }

    private func moveConversationToFront(at index: Int) {
        guard index > 0 else { return }
        let record = conversations.remove(at: index)
        conversations.insert(record, at: 0)
    }

    private static func automaticTitle(for prompt: String) -> String {
        let cleaned = cleanedTitle(prompt)
        guard cleaned.count > 52 else { return cleaned.isEmpty ? "New Chat" : cleaned }
        let end = cleaned.index(cleaned.startIndex, offsetBy: 49)
        return String(cleaned[..<end]).trimmingCharacters(in: .whitespaces) + "…"
    }

    private static func cleanedTitle(_ value: String) -> String {
        value.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}

/// Download, selection, and optional add-on state for the model library.
@MainActor
@Observable
public final class AppModelLibraryStore {
    public var modelPathText = ""
    public var installs: [ModelInstallCoordinator] = []
    public var selectedModelID = ""
    public var isConfirmingVisionPackRemoval = false
    public var visionInstallETAPresentation: DownloadETAPresentation = .hidden
    public var visionInstallETAText: String?
    public var visionInstallState: AppModelInstallState = .idle
    public var visionActivationProgress: Double?
    public var visionInstallReadiness: AppModelInstallReadiness = .checking
    public var visionInstallationStatus: AppVisionPackInstallationStatus = .missing
    public var visionInstallTargetModelID: String?

    public init() {}
}

/// User-configurable defaults and observable profiles, persisted by AppModel.
@MainActor
@Observable
public final class AppSettingsStore {
    public var runtimeOptions = AppRuntimeOptions()
    public var automaticMemory = true
    public var automaticMemoryProfile: AppAutomaticMemoryProfile = .balanced
    public var maxNewTokensOverride: Int?
    public var reasoning: ChatReasoning = .off
    public var reasoningEffort: GPTOSSReasoningEffort = .medium
    public var preserveThinking = false
    public var maxContextTokens = 4_096
    public var temperature = 0.2
    public var topKEnabled = true
    public var topK = 64
    public var topPEnabled = true
    public var topP = 0.95
    public var newlineShortcut: AppNewlineShortcut = .shiftReturn
    public var showPromptExamples = true
    public var sentPromptBehavior: AppSentPromptBehavior = .keep
    public var loadModelOnLaunch = false
    public var smoothStreaming = true
    public var appearanceMode: AppAppearanceMode = .system
    public var translucentWindow = true
    public var accentColorMode: AppAccentColorMode = .appDefault
    public var customAccentColorHex: String = AppHexColor.defaultPurple.hexString
    public var zoomLevel: AppZoomLevel = .default
    public var webSearchEnabled = false
    public var pauseWebSearchWhenOffline = false
    public var runWebInSandbox = false
    public var runFilesInSandbox = false
    public var fileSearchEnabled = false
    public var codingWriteEnabled = false
    public var codingCommandEnabled = false
    public var codingNetworkEnabled = false
    public var codingFolderID: String?
    public var searchProvider: AppSearchProviderKind = .duckDuckGo
    /// App-wide escape hatch from the hardware and memory gates.
    public var bypassModelRestrictions = false
    /// The selected model's system prompt, loaded from its profile.
    public var systemPrompt = ""
    /// Observable stored profiles, including manual values hidden by Auto.
    public internal(set) var modelProfiles: [String: AppModelSettingsProfile] = [:]

    public init() {}
}

/// Chat's inference state. The inference client and task ownership stay in
/// AppModel.
@MainActor
@Observable
public final class AppSharedInferenceStore {
    public var runState: AppRunState = .idle
    public var diagnostics: AppDiagnostics?
    public var error: AppInferenceError?
    public var loadState: AppModelLoadState = .notLoaded
    public var loadedRuntimeKey: AppLoadedRuntimeKey?
    public var phase: AppGenerationPhase = .idle
    public var liveTokenCount = 0
    public var liveElapsedDecodeSeconds = 0.0
    public var livePrefillDone = 0
    public var livePrefillTotal = 0
    public var liveMemoryBytes: UInt64?
    public var liveResidentBytes: UInt64?
    public var visionTowerMappedBytes: UInt64?
    public var isCancellationPending = false

    public init() {}
}
