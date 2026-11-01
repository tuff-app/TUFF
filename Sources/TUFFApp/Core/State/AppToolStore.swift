import Foundation
import Observation
import Synchronization
import TUFFModelCatalog

/// Whether a model can be offered tools. Decided per model from real-model
/// checks; a model that has not been checked says so rather than pretending.
public enum AppToolSupport: Equatable, Sendable {
    /// A real tool round (call, result, cited answer) passed on the
    /// reference Mac. That is a functional check, not a quality rating.
    case validated
    /// Its template supports tools but no real tool round was run.
    case untested
    case unavailable(String)

    public var allowsTools: Bool {
        if case .unavailable = self { return false }
        return true
    }

    public var note: String? {
        switch self {
        case .validated: nil
        case .untested: "Web and file search have not been checked with this model yet."
        case .unavailable(let reason): reason
        }
    }

    public static func forModel(_ id: TUFFModelID) -> AppToolSupport {
        switch id {
        case .gemma4_26B_A4B, .gemma4_26B_Heretic, .qwen38FlashNext, .gemma4_12B_QAT, .gemma4_E4B, .gemma4_E2B,
             .qwen36_35B_A3B, .gptOss_20B, .ornith15_9B, .ornith15_35B_A3B, .ornith15_122B, .qwen38_27B, .ornith15_397B:
            .validated
        case .gptOss_120B, .minimaxM27, .flint40B, .bonsai2_2B, .bonsai2_27B:
            .untested
        }
    }
}

/// A flag the indexer reads from its own task to stay out of inference's way.
public final class AppInferenceActivityFlag: Sendable {
    private let state = Mutex(false)
    public init() {}
    public var isActive: Bool { state.withLock { $0 } }
    public func set(_ value: Bool) { state.withLock { $0 = value } }
}

/// Folders, the local index, search-provider keys, and the tools built from
/// them.
@MainActor
@Observable
public final class AppToolStore {
    public private(set) var folders: [AppSearchFolder]
    public private(set) var folderStatus: [UUID: AppIndexFolderStatus] = [:]
    public private(set) var indexingFolderID: UUID?
    public private(set) var lastError: String?
    /// Which providers have a saved key. Filled on request so the Keychain is
    /// not read until the person looks at search settings or searches.
    public private(set) var keyPresence: [AppSearchProviderKind: Bool] = [:]

    @ObservationIgnored let folderStore: AppSearchFolderStore?
    @ObservationIgnored let keyStore: any AppSearchKeyStore
    @ObservationIgnored let indexURL: URL
    @ObservationIgnored let transport: any AppHTTPTransport
    #if os(macOS)
    /// The sandbox web requests go through when an answer runs with the
    /// sandbox setting on. Set by the model; nothing else owns one.
    @ObservationIgnored var sandboxWeb: SandboxWebAccess?
    /// The sandbox the folders are read through when `indexInSandbox` is on.
    @ObservationIgnored var sandboxFiles: SandboxFilesService?
    #endif
    /// Whether files are read and turned into passages in the files sandbox
    /// instead of by the app. Mirrors the setting; with no sandbox to use,
    /// indexing fails and nothing is read directly.
    @ObservationIgnored public var indexInSandbox = false
    @ObservationIgnored let indexLimits: AppLocalIndexLimits
    @ObservationIgnored private var index: AppLocalSearchIndex?
    @ObservationIgnored private var indexTask: Task<Void, Never>?
    @ObservationIgnored private var indexRunID = UUID()
    @ObservationIgnored private var indexDrainTask: Task<Void, Never>?
    @ObservationIgnored private var indexMaintenanceTask: Task<Void, Never>?
    @ObservationIgnored public nonisolated let inferenceActivity = AppInferenceActivityFlag()

    public init(folderStore: AppSearchFolderStore?, keyStore: any AppSearchKeyStore,
                indexURL: URL, transport: any AppHTTPTransport,
                indexLimits: AppLocalIndexLimits = .standard) {
        self.folderStore = folderStore
        self.keyStore = keyStore
        self.indexURL = indexURL
        self.transport = transport
        self.indexLimits = indexLimits
        folders = folderStore?.load() ?? []
    }

    /// The app's stores: folders in Application Support, the index in
    /// Caches, keys in the Keychain.
    public static func standard() -> AppToolStore {
        AppToolStore(folderStore: .standard(), keyStore: KeychainSearchKeyStore(),
                     indexURL: AppLocalSearchIndex.standardURL(),
                     transport: URLSessionHTTPTransport())
    }

    /// Nothing persisted, nothing read from the person's Keychain or folders.
    public static func inMemory(
        indexURL: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-index-\(UUID().uuidString).sqlite"),
        transport: any AppHTTPTransport = URLSessionHTTPTransport(),
        keyStore: any AppSearchKeyStore = InMemorySearchKeyStore()
    ) -> AppToolStore {
        AppToolStore(folderStore: nil, keyStore: keyStore, indexURL: indexURL,
                     transport: transport)
    }

    private func openIndex() throws -> AppLocalSearchIndex {
        if let index { return index }
        let opened = try AppLocalSearchIndex(databaseURL: indexURL)
        index = opened
        return opened
    }

    // MARK: Folders

    public func addFolders(_ urls: [URL]) {
        var changed = false
        for url in urls {
            do {
                let folder = try AppSearchFolderStore.folder(for: url)
                guard !folders.contains(where: { $0.path == folder.path }) else { continue }
                folders.append(folder)
                changed = true
            } catch {
                lastError = "\(error)"
            }
        }
        guard changed else { return }
        saveFolders()
        reindex()
    }

    /// Stops search reading the folder and deletes everything indexed from it.
    public func removeFolder(id: UUID) {
        folders.removeAll { $0.id == id }
        folderStatus[id] = nil
        saveFolders()
        // Restart even when this folder was queued behind another one.
        // The old task captured the previous folder list.
        stopIndexing()
        removeIndexedFolder(id)
        reindex()
    }

    /// Removes every folder and deletes the index.
    public func removeAllFolders() {
        stopIndexing()
        folders = []
        folderStatus = [:]
        saveFolders()
        removeIndexedFolder(nil)
        stopFilesSandboxWithoutFolders()
    }

    /// With no folder left there is nothing for the files sandbox to read,
    /// and the folders must not stay mounted into it.
    private func stopFilesSandboxWithoutFolders() {
        #if os(macOS)
        guard indexInSandbox, let service = sandboxFiles else { return }
        Task { await service.stop() }
        #endif
    }

    /// Forgets the files that were skipped, so the next update reads them
    /// again with whichever reader is in use. For when the setting that
    /// chooses the reader changes.
    public func forgetSkippedFiles() {
        let previous = indexMaintenanceTask
        let running = indexDrainTask
        let index = try? openIndex()
        indexMaintenanceTask = Task {
            await previous?.value
            await running?.value
            try? await index?.forgetSkipped()
        }
    }

    private func removeIndexedFolder(_ id: UUID?) {
        let previous = indexMaintenanceTask
        let running = indexDrainTask
        let index = try? openIndex()
        indexMaintenanceTask = Task {
            await previous?.value
            // An already queued SQLite write must finish before deletion.
            await running?.value
            if let id { try? await index?.remove(folder: id) }
            else { try? await index?.removeAll() }
        }
    }

    private func saveFolders() {
        do {
            try folderStore?.save(folders)
        } catch {
            lastError = "The folder list could not be saved: \(error)"
        }
    }

    /// Brings the index up to date for every folder, one at a time, in the
    /// background. Unchanged files are not read again.
    public func reindex() {
        stopIndexing()
        let runID = indexRunID
        let folders = folders
        guard !folders.isEmpty else {
            stopFilesSandboxWithoutFolders()
            return
        }
        let index: AppLocalSearchIndex
        do {
            index = try openIndex()
        } catch {
            lastError = "\(error)"
            return
        }
        let indexer: AppLocalIndexer
        #if os(macOS)
        let sandboxService: SandboxFilesService?
        if indexInSandbox {
            guard let service = sandboxFiles else {
                for folder in folders {
                    var status = folderStatus[folder.id] ?? AppIndexFolderStatus()
                    status.lastError = "The file sandbox is not available."
                    folderStatus[folder.id] = status
                }
                return
            }
            sandboxService = service
            indexer = AppLocalIndexer(index: index, limits: indexLimits,
                                      reader: SandboxFileReader(access: service.access))
        } else {
            sandboxService = nil
            indexer = AppLocalIndexer(index: index, limits: indexLimits)
        }
        #else
        indexer = AppLocalIndexer(index: index, limits: indexLimits)
        #endif
        let activity = inferenceActivity
        let previous = indexDrainTask
        let maintenance = indexMaintenanceTask
        indexTask = Task(priority: .background) { [weak self] in
            await previous?.value
            await maintenance?.value
            guard !Task.isCancelled else { return }
            #if os(macOS)
            // Folders macOS did not let the sandbox read: left out, with the
            // reason, while the others are indexed.
            var blockedFolders: [String: String] = [:]
            if let sandboxService {
                // The sandbox runs with exactly these folders mounted before
                // any file is read. Starting it can take a while: indexing
                // shows as under way meanwhile.
                await MainActor.run {
                    guard self?.indexRunID == runID else { return }
                    self?.indexingFolderID = folders.first?.id
                }
                do {
                    blockedFolders = try await sandboxService.ensureRunning(for: folders)
                } catch {
                    let message = error is CancellationError ? nil : "\(error)"
                    await MainActor.run {
                        guard self?.indexRunID == runID else { return }
                        if let message {
                            for folder in folders {
                                var status = self?.folderStatus[folder.id] ?? AppIndexFolderStatus()
                                status.lastError = message
                                self?.folderStatus[folder.id] = status
                            }
                        }
                        self?.indexingFolderID = nil
                        self?.indexTask = nil
                    }
                    return
                }
            }
            #endif
            for folder in folders {
                guard !Task.isCancelled else { break }
                #if os(macOS)
                if let message = blockedFolders[folder.id.uuidString.lowercased()] {
                    await MainActor.run {
                        guard self?.indexRunID == runID else { return }
                        var status = self?.folderStatus[folder.id] ?? AppIndexFolderStatus()
                        status.lastError = message
                        self?.folderStatus[folder.id] = status
                    }
                    continue
                }
                #endif
                await MainActor.run {
                    guard self?.indexRunID == runID else { return }
                    self?.indexingFolderID = folder.id
                }
                do {
                    let status = try await indexer.update(
                        folder, shouldPause: { activity.isActive },
                        progress: { status in
                            Task { @MainActor in
                                guard self?.indexRunID == runID else { return }
                                self?.folderStatus[folder.id] = status
                            }
                        })
                    await MainActor.run {
                        guard self?.indexRunID == runID else { return }
                        self?.folderStatus[folder.id] = status
                    }
                } catch is CancellationError {
                    break
                } catch {
                    await MainActor.run {
                        guard self?.indexRunID == runID else { return }
                        var status = self?.folderStatus[folder.id] ?? AppIndexFolderStatus()
                        status.lastError = "\(error)"
                        self?.folderStatus[folder.id] = status
                    }
                }
            }
            await MainActor.run {
                guard self?.indexRunID == runID else { return }
                self?.indexingFolderID = nil
                self?.indexTask = nil
            }
        }
        indexDrainTask = indexTask
    }

    public func stopIndexing() {
        indexRunID = UUID()
        indexTask?.cancel()
        indexTask = nil
        indexingFolderID = nil
    }

    /// "Stop Indexing" from the settings: the run ends, and a start of the
    /// files sandbox under way for it does not finish and leave the folders
    /// mounted. (Not part of `stopIndexing`, which a new run calls first.)
    public func cancelIndexing() {
        stopIndexing()
        #if os(macOS)
        if indexInSandbox, let service = sandboxFiles, service.controller.state.isBusy {
            Task { await service.stop() }
        }
        #endif
    }

    public var isIndexing: Bool { indexingFolderID != nil }

    // MARK: Keys

    public func refreshKeyPresence() {
        for provider in AppSearchProviderKind.allCases where provider.requiresKey {
            keyPresence[provider] = keyStore.key(for: provider) != nil
        }
    }

    public func setKey(_ key: String, for provider: AppSearchProviderKind) {
        do {
            try keyStore.setKey(key, for: provider)
            keyPresence[provider] = !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            lastError = nil
        } catch {
            lastError = "\(error)"
        }
    }

    public func removeKey(for provider: AppSearchProviderKind) {
        do {
            try keyStore.removeKey(for: provider)
            keyPresence[provider] = false
        } catch {
            lastError = "\(error)"
        }
    }

    // MARK: Tools

    /// The tools one answer runs with. The provider key is read here, in
    /// this process, and kept in the closure only for the answer's duration.
    /// With `webInSandbox` every web request goes through the sandbox
    /// (searches to its `/v1/http`, pages to its `/v1/page`); if there is no
    /// sandbox to use, web requests fail and nothing is sent directly.
    public func makeToolbox(provider kind: AppSearchProviderKind,
                            limits: AppToolLimits = .standard,
                            webInSandbox: Bool = false, workspace: String? = nil,
                            workspaceWrite: Bool = false, commandNetwork: Bool = false,
                            codingAllowed: @escaping @Sendable (AppValidatedToolCall) async -> Bool = { _ in true }) -> AppToolbox {
        let timeout = limits.requestTimeoutSeconds
        let transport: any AppHTTPTransport
        let readPage: @Sendable (URL) async throws -> AppWebPage
        if webInSandbox {
            // Never the direct path: with no sandbox to use, requests fail.
            var sandboxTransport: any AppHTTPTransport = Self.unavailableTransport
            var sandboxPages: @Sendable (URL) async throws -> AppWebPage = { _ in
                throw AppWebPageError.transport(Self.sandboxUnavailable)
            }
            #if os(macOS)
            if let access = sandboxWeb {
                let pageReader = SandboxPageReader(access: access, timeoutSeconds: timeout)
                sandboxTransport = SandboxHTTPTransport(access: access)
                sandboxPages = { url in try await pageReader.read(url) }
            }
            #endif
            transport = sandboxTransport
            readPage = sandboxPages
        } else {
            transport = self.transport
            let reader = AppWebPageReader(transport: self.transport, timeoutSeconds: timeout)
            readPage = { url in try await reader.read(url) }
        }
        let provider: any AppWebSearchProvider = switch kind {
        case .duckDuckGo:
            DuckDuckGoSearchProvider(transport: transport, timeoutSeconds: timeout)
        case .brave:
            BraveSearchProvider(transport: transport, key: keyStore.key(for: .brave) ?? "",
                                timeoutSeconds: timeout)
        case .tavily:
            TavilySearchProvider(transport: transport, key: keyStore.key(for: .tavily) ?? "",
                                 timeoutSeconds: timeout)
        }
        let folders = folders
        let index = try? openIndex()
        return AppToolbox(
            webSearch: { query, count in try await provider.search(query, count: count) },
            readWebpage: readPage,
            searchFiles: { query, count in
                guard let index else {
                    throw AppLocalSearchError.index("The local search index could not be opened.")
                }
                return try await index.search(query, folders: folders, limit: count)
            }, coding: { call in
                #if os(macOS)
                guard let workspace else { throw AppToolValidationError("No workspace is selected.") }
                guard await codingAllowed(call) else { throw AppToolValidationError("Workspace access changed while this answer was running.") }
                let files = AppWorkspaceFiles(root: workspace)
                switch call {
                case .listFiles(let path): return try files.list(path)
                case .readFile(let path, let startLine, let lineCount):
                    let lines = try files.read(path).components(separatedBy: "\n")
                    let start = min(lines.count, startLine - 1)
                    let end = min(lines.count, start + lineCount)
                    return lines[start..<end].enumerated().map { "\(start + $0.offset + 1): \($0.element)" }.joined(separator: "\n")
                case .writeFile(let path, let content): try files.write(path, content: content); return "Wrote \(path)."
                case .editFile(let path, let old, let new): try files.edit(path, old: old, new: new); return "Edited \(path)."
                case .runCommand(let command):
                    return try await AppWorkspaceCommand(root: workspace, allowWrite: workspaceWrite,
                                                         allowNetwork: commandNetwork).run(command)
                default: throw AppToolValidationError("Command execution is unavailable.")
                }
                #else
                throw AppToolValidationError("Workspace tools require macOS.")
                #endif
            })
    }

    nonisolated private static let sandboxUnavailable = AppHTTPError.sandbox("The web sandbox is not available.")

    /// What a sandbox-mode answer uses when no sandbox was given to the
    /// store: every request fails, none is made.
    nonisolated private static let unavailableTransport: any AppHTTPTransport = FailingHTTPTransport(
        error: sandboxUnavailable)
}

private struct FailingHTTPTransport: AppHTTPTransport {
    let error: AppHTTPError
    func perform(_ request: AppHTTPRequest) async throws -> AppHTTPResponse { throw error }
}
