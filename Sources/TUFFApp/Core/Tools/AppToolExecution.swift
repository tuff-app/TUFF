import Foundation

/// The operations tools run. Production wires the search provider, the page
/// reader and the local index; tests pass fixtures.
public struct AppToolbox: Sendable {
    public var webSearch: @Sendable (_ query: String, _ count: Int) async throws -> AppWebSearchResponse
    public var readWebpage: @Sendable (_ url: URL) async throws -> AppWebPage
    public var searchFiles: @Sendable (_ query: String, _ count: Int) async throws -> [AppFilePassage]
    public var coding: @Sendable (AppValidatedToolCall) async throws -> String

    public init(webSearch: @escaping @Sendable (String, Int) async throws -> AppWebSearchResponse,
                readWebpage: @escaping @Sendable (URL) async throws -> AppWebPage,
                searchFiles: @escaping @Sendable (String, Int) async throws -> [AppFilePassage],
                coding: @escaping @Sendable (AppValidatedToolCall) async throws -> String = { _ in throw AppToolValidationError("Workspace tools are unavailable.") }) {
        self.webSearch = webSearch
        self.readWebpage = readWebpage
        self.searchFiles = searchFiles
        self.coding = coding
    }
}

/// What the transcript shows while a tool runs and after it finishes.
public struct AppToolActivity: Equatable, Sendable, Identifiable {
    public enum State: String, Equatable, Sendable { case running, succeeded, failed, refused, cancelled }
    public let id: String
    public let name: String
    public let title: String
    public let detail: String
    public var state: State
    public var summary: String
    /// Bounded text returned to the model, available on demand in the UI.
    public var resultText: String = ""

    public static func title(for name: String) -> String {
        switch AppToolName(rawValue: name) {
        case .webSearch?: "Searching the web"
        case .readWebpage?: "Reading a web page"
        case .searchFiles?: "Searching your files"
        case .listFiles?: "Listing files"
        case .readFile?: "Reading a file"
        case .writeFile?: "Writing a file"
        case .editFile?: "Editing a file"
        case .runCommand?: "Running a command"
        case nil: "Using a tool"
        }
    }
}

/// One answer's tool use: what has run, the sources it found, and the limits
/// still left. Owned by the task generating that answer; not shared.
public final class AppToolAnswerSession: @unchecked Sendable {
    public let capabilities: AppChatCapabilities
    public let limits: AppToolLimits
    private let toolbox: AppToolbox
    private var completedToolSeconds = 0.0
    private var activeToolStart: ContinuousClock.Instant?
    private let clock = ContinuousClock()

    public private(set) var rounds: [AppToolRound] = []
    public private(set) var sources: [AppSource] = []
    private var nextSourceID: Int
    private var webRequests = 0
    private var fileSearches = 0
    private var consecutiveRefusedRounds = 0
    private var failedCodingCalls: [AppToolCall] = []
    /// URLs the model may read: those search returned and those the user
    /// wrote. A page's own links are not added, so retrieved text cannot send
    /// the reader somewhere new.
    private var readableURLs: Set<String>
    /// Local text the model has seen in this answer. A web query that copies
    /// a long run of it is refused, so file contents are not sent to a
    /// search provider.
    private var localTexts: [String] = []
    private var hasLocalContext: Bool
    private let userText: String
    /// Search texts the user wrote as the whole message, a whole line or a
    /// quoted phrase. Each may be sent once after local context exists.
    private var userQueryCandidates: Set<String>
    private static let minimumUserQueryCharacters = 4

    public init(capabilities: AppChatCapabilities, limits: AppToolLimits = .standard,
                toolbox: AppToolbox, firstSourceID: Int = 1, userText: String,
                hasLocalContext: Bool = false) {
        self.capabilities = capabilities
        self.limits = limits
        self.toolbox = toolbox
        self.hasLocalContext = hasLocalContext
        self.userText = userText
        self.userQueryCandidates = Self.userQueryCandidates(in: userText)
        self.nextSourceID = max(1, firstSourceID)
        readableURLs = Set(Self.urls(in: userText).map(Self.normalized))
    }

    /// The model has used its rounds, or keeps making calls that are refused:
    /// the next generation is told to answer.
    public var mustAnswer: Bool {
        rounds.count >= limits.maximumToolRounds || consecutiveRefusedRounds >= 2
            || elapsedSeconds >= limits.maximumToolSeconds
    }

    public var elapsedSeconds: Double {
        guard let activeToolStart else { return completedToolSeconds }
        let elapsed = clock.now - activeToolStart
        return completedToolSeconds + Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
    }

    /// Runs one round's calls in order. Calls beyond the per-round limit, or
    /// after the model has used its rounds, are refused without running.
    /// `characterBudget` is the total result text the context can still take.
    public func execute(calls: [AppToolCall], thinking: String?, content: String,
                        characterBudget: Int,
                        onActivity: @Sendable (AppToolActivity) async -> Void) async -> AppToolRound {
        activeToolStart = clock.now
        defer {
            completedToolSeconds = elapsedSeconds
            activeToolStart = nil
        }
        var round = AppToolRound(thinking: thinking?.isEmpty == false ? thinking : nil,
                                 content: content, calls: calls)
        let answering = mustAnswer
        let contextFull = characterBudget < limits.minimumResultCharacters * max(1, calls.count)
        let perCall = max(limits.minimumResultCharacters,
                          min(limits.maximumResultCharacters, characterBudget / max(1, calls.count)))
        for (index, call) in calls.enumerated() {
            var activity = AppToolActivity(
                id: call.id, name: call.name, title: AppToolActivity.title(for: call.name),
                detail: Self.detail(for: call), state: .running, summary: "")
            let result: AppToolResult
            if Task.isCancelled {
                result = .init(callID: call.id, name: call.name, status: .cancelled,
                               modelText: "Stopped by the user.", summary: "Stopped")
            } else if answering {
                result = refused(call, "Tool limit reached. Answer now with the information you already have.")
            } else if contextFull {
                result = refused(call, "The context is nearly full. Answer now with the information you already have.")
            } else if index >= limits.maximumCallsPerRound {
                result = refused(call, "Only \(limits.maximumCallsPerRound) tool calls run per step. Make this call again later if you still need it.")
            } else {
                await onActivity(activity)
                result = await run(call, characterLimit: perCall)
            }
            activity.state = switch result.status {
            case .succeeded: .succeeded
            case .failed: .failed
            case .refused: .refused
            case .cancelled: .cancelled
            }
            activity.summary = result.summary
            activity.resultText = result.modelText
            await onActivity(activity)
            round.results.append(result)
        }
        let allRefused = !round.results.isEmpty
            && round.results.allSatisfy { $0.status == .refused }
        consecutiveRefusedRounds = allRefused ? consecutiveRefusedRounds + 1 : 0
        rounds.append(round)
        return round
    }

    private func refused(_ call: AppToolCall, _ reason: String) -> AppToolResult {
        AppToolResult(callID: call.id, name: call.name, status: .refused,
                      modelText: "Not run: \(reason)", summary: reason)
    }

    private func run(_ call: AppToolCall, characterLimit: Int) async -> AppToolResult {
        let validated: AppValidatedToolCall
        do {
            validated = try AppToolCatalog.validate(call, capabilities: capabilities)
        } catch {
            return refused(call, "\(error)")
        }
        if elapsedSeconds >= limits.maximumToolSeconds {
            return refused(call, "The time allowed for tools in one answer is used up.")
        }
        let isCoding: Bool
        switch validated {
        case .listFiles, .readFile, .writeFile, .editFile, .runCommand: isCoding = true
        default: isCoding = false
        }
        if isCoding, failedCodingCalls.contains(where: {
            $0.name == call.name && $0.arguments == call.arguments
        }) {
            return refused(call, "This identical workspace call already failed. Inspect the error and change the approach before retrying.")
        }
        do {
            switch validated {
            case .webSearch(let query, let count):
                guard webRequests < limits.maximumWebRequests else {
                    return refused(call, "The web request limit for this answer is reached.")
                }
                // A substring detector cannot stop short, paraphrased or
                // encoded secrets, and arbitrary substrings of the user's
                // message would leak bits by choice. After local material
                // enters the context, permit only the whole message, a whole
                // line or a quoted phrase of this turn's user message, at
                // least a few characters long and once each.
                if hasLocalContext {
                    guard userQueryCandidates.remove(query) != nil else {
                        return refused(call, "This chat contains local files or images. To search the web, write the exact search terms as your whole message, on a line of their own, or in double quotes. Each phrase can be used once. TUFF does not send model-derived queries from local material.")
                    }
                }
                if let copied = copiedLocalText(in: query) {
                    return refused(call, "The query repeats text from the user's files (\"\(copied)...\"). File contents are not sent to search providers.")
                }
                webRequests += 1
                let response = try await toolbox.webSearch(query, count)
                return searchResult(call, query: query, response: response,
                                    characterLimit: characterLimit)
            case .readWebpage(let url):
                guard webRequests < limits.maximumWebRequests else {
                    return refused(call, "The web request limit for this answer is reached.")
                }
                guard readableURLs.contains(Self.normalized(url)) else {
                    return refused(call, "Only pages returned by web_search or written by the user can be read.")
                }
                webRequests += 1
                let page = try await toolbox.readWebpage(url)
                return pageResult(call, requested: url, page: page, characterLimit: characterLimit)
            case .searchFiles(let query, let count):
                guard fileSearches < limits.maximumFileSearches else {
                    return refused(call, "The file search limit for this answer is reached.")
                }
                fileSearches += 1
                let passages = try await toolbox.searchFiles(query, count)
                if !passages.isEmpty { hasLocalContext = true }
                return fileResult(call, query: query, passages: passages,
                                  characterLimit: characterLimit)
            case .listFiles, .readFile, .writeFile, .editFile, .runCommand:
                let text = try await toolbox.coding(validated)
                // A successful write/edit/command may have changed the
                // workspace, so previously failing calls can be useful again.
                switch validated {
                case .writeFile, .editFile, .runCommand: failedCodingCalls.removeAll()
                default: break
                }
                hasLocalContext = true
                localTexts.append(text)
                let limited = String(text.prefix(characterLimit))
                return .init(callID: call.id, name: call.name, status: .succeeded,
                             modelText: limited + (text.count > characterLimit ? "\n[Output truncated.]" : ""),
                             summary: "Completed")
            }
        } catch is CancellationError {
            return .init(callID: call.id, name: call.name, status: .cancelled,
                         modelText: "Stopped by the user.", summary: "Stopped")
        } catch AppHTTPError.cancelled, AppWebSearchError.transport(_, .cancelled),
                AppWebPageError.transport(.cancelled) {
            return .init(callID: call.id, name: call.name, status: .cancelled,
                         modelText: "Stopped by the user.", summary: "Stopped")
        } catch let failure as AppCommandFailure {
            failedCodingCalls.append(call)
            // A non-zero command can still print private workspace contents.
            // Treat captured output just like successful file/command output
            // before deciding whether a later call may send a query online.
            hasLocalContext = true
            let text = failure.description
            localTexts.append(text)
            let limited = String(text.prefix(characterLimit))
            return .init(callID: call.id, name: call.name, status: .failed,
                         modelText: limited + (text.count > characterLimit ? "\n[Output truncated.]" : ""),
                         summary: failure.summary)
        } catch {
            // The error may quote a file name or path from the workspace.
            switch validated {
            case .listFiles, .readFile, .writeFile, .editFile, .runCommand:
                failedCodingCalls.append(call)
                hasLocalContext = true
            case .webSearch, .readWebpage, .searchFiles: break
            }
            let message = "\(error)"
            return .init(callID: call.id, name: call.name, status: .failed,
                         modelText: "Failed: \(message) No information was retrieved; do not invent any.",
                         summary: message)
        }
    }

    // MARK: Results

    private func addSource(_ make: (Int) -> AppSource) -> AppSource {
        let source = make(nextSourceID)
        nextSourceID += 1
        sources.append(source)
        return source
    }

    private func searchResult(_ call: AppToolCall, query: String, response: AppWebSearchResponse,
                              characterLimit: Int) -> AppToolResult {
        var lines: [String] = []
        var ids: [Int] = []
        var remaining = characterLimit
        for result in response.results {
            let key = Self.normalized(result.url)
            readableURLs.insert(key)
            let existing = sources.first { $0.kind == .web && $0.url.flatMap(URL.init(string:)).map(Self.normalized) == key }
            let source = existing ?? addSource { id in
                AppSource(id: id, kind: .web, title: result.title, url: result.url.absoluteString,
                          excerpt: result.excerpt, origin: response.provider.displayName)
            }
            let block = "[\(source.id)] \(result.title)\n\(result.url.absoluteString)\n\(result.excerpt)"
            guard block.count <= remaining || lines.isEmpty else { break }
            lines.append(String(block.prefix(remaining)))
            remaining -= block.count + 2
            ids.append(source.id)
        }
        let text = "Web results from \(response.provider.displayName) for \"\(query)\":\n\n"
            + lines.joined(separator: "\n\n")
        return AppToolResult(callID: call.id, name: call.name, status: .succeeded,
                             modelText: text,
                             summary: "\(ids.count) result\(ids.count == 1 ? "" : "s") from \(response.provider.displayName)",
                             sourceIDs: ids)
    }

    private func pageResult(_ call: AppToolCall, requested: URL, page: AppWebPage,
                            characterLimit: Int) -> AppToolResult {
        let title = page.title ?? page.finalURL.host ?? requested.absoluteString
        let header = "\(title)\n\(page.finalURL.absoluteString)\n"
        let room = max(0, characterLimit - header.count - 16)
        let excerpt = page.text.count > room
            ? String(page.text.prefix(room)) + "\n[Page text cut at \(room) of \(page.text.count) characters.]"
            : page.text
        let key = Self.normalized(requested)
        let source: AppSource
        if let index = sources.firstIndex(where: { $0.kind == .web && $0.url.flatMap(URL.init(string:)).map(Self.normalized) == key }) {
            // The page replaces the search snippet as what the answer used.
            let old = sources[index]
            source = AppSource(id: old.id, kind: .web, title: old.title, url: old.url,
                               excerpt: excerpt, origin: old.origin)
            sources[index] = source
        } else {
            source = addSource { id in
                AppSource(id: id, kind: .web, title: title, url: page.finalURL.absoluteString,
                          excerpt: excerpt, origin: "Web page")
            }
        }
        return AppToolResult(callID: call.id, name: call.name, status: .succeeded,
                             modelText: "[\(source.id)] \(header)\(excerpt)",
                             summary: "Read \(page.finalURL.host ?? title)",
                             sourceIDs: [source.id])
    }

    private func fileResult(_ call: AppToolCall, query: String, passages: [AppFilePassage],
                            characterLimit: Int) -> AppToolResult {
        guard !passages.isEmpty else {
            return AppToolResult(callID: call.id, name: call.name, status: .succeeded,
                                 modelText: "No passages in the selected folders matched \"\(query)\".",
                                 summary: "No matching passages")
        }
        var blocks: [String] = []
        var ids: [Int] = []
        let perPassage = max(200, characterLimit / passages.count - 80)
        for passage in passages {
            let text = passage.text.count > perPassage
                ? String(passage.text.prefix(perPassage)) + "..." : passage.text
            localTexts.append(passage.text)
            let label = [passage.displayName, passage.location].compactMap { $0 }
                .joined(separator: ", ")
            let source = addSource { id in
                AppSource(id: id, kind: .file, title: passage.displayName,
                          filePath: passage.filePath, page: passage.page,
                          location: passage.location, excerpt: text, origin: "Local files")
            }
            blocks.append("[\(source.id)] \(label)\n\(text)")
            ids.append(source.id)
        }
        let files = Set(passages.map(\.filePath)).count
        return AppToolResult(callID: call.id, name: call.name, status: .succeeded,
                             modelText: "Passages from the selected folders for \"\(query)\":\n\n"
                                + blocks.joined(separator: "\n\n"),
                             summary: "\(passages.count) passage\(passages.count == 1 ? "" : "s") from \(files) file\(files == 1 ? "" : "s")",
                             sourceIDs: ids)
    }

    // MARK: Guards

    /// The first 40 characters of a run of at least 48 that `query` copies
    /// from local text this answer has seen, ignoring case and spacing.
    func copiedLocalText(in query: String) -> String? {
        let normalizedQuery = AppHTML.collapse(query.lowercased())
        let window = 48
        guard normalizedQuery.count >= window, !localTexts.isEmpty else { return nil }
        let haystacks = localTexts.map { AppHTML.collapse($0.lowercased()) }
        let characters = Array(normalizedQuery)
        for start in 0...(characters.count - window) {
            let piece = String(characters[start..<(start + window)])
            if haystacks.contains(where: { $0.contains(piece) }) {
                return String(piece.prefix(40))
            }
        }
        return nil
    }

    static func normalized(_ url: URL) -> String {
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            return url.absoluteString
        }
        components.fragment = nil
        components.scheme = components.scheme?.lowercased()
        components.host = components.host?.lowercased()
        if components.path.isEmpty { components.path = "/" }
        return components.string ?? url.absoluteString
    }

    /// The whole message, each whole line and each double-quoted phrase that
    /// is long enough to be a deliberate search.
    static func userQueryCandidates(in text: String) -> Set<String> {
        var found: Set<String> = []
        func add(_ raw: String) {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if value.count >= minimumUserQueryCharacters, value.count <= AppToolCatalog.maximumQueryCharacters {
                found.insert(value)
            }
        }
        add(text)
        for line in text.split(whereSeparator: \.isNewline) { add(String(line)) }
        let quoted = [("\"", "\""), ("\u{201C}", "\u{201D}")]
        for (open, close) in quoted {
            var rest = Substring(text)
            while let start = rest.range(of: open) {
                let after = rest[start.upperBound...]
                guard let end = after.range(of: close) else { break }
                add(String(after[..<end.lowerBound]))
                rest = after[end.upperBound...]
            }
        }
        return found
    }

    static func urls(in text: String) -> [URL] {
        guard let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue)
        else { return [] }
        return detector.matches(in: text, range: NSRange(text.startIndex..., in: text))
            .compactMap(\.url)
            .filter { ["http", "https"].contains($0.scheme?.lowercased() ?? "") }
    }

    public static func detail(for call: AppToolCall) -> String {
        guard case .object(let arguments) = call.arguments else { return "" }
        let value: String?
        switch arguments["query"] ?? arguments["url"] ?? arguments["path"] ?? arguments["command"] {
        case .string(let text)?: value = text
        default: value = nil
        }
        guard let value else { return "" }
        return value.count > 120 ? String(value.prefix(120)) + "..." : value
    }
}
