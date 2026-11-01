import Foundation
import Synchronization
import Testing
import TUFFEngine
@testable import TUFFAppCore

/// Replays one scripted generation per `generate` call and records requests.
final class ScriptedInferenceClient: AppInferenceClient, Sendable {
    private let scripts: Mutex<[[AppInferenceEvent]]>
    private let recorded = Mutex<[AppGenerationRequest]>([])
    private let cancels = Mutex(0)

    init(_ scripts: [[AppInferenceEvent]]) {
        self.scripts = Mutex(scripts)
    }

    var requests: [AppGenerationRequest] { recorded.withLock { $0 } }
    var cancelCount: Int { cancels.withLock { $0 } }

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        recorded.withLock { $0.append(request) }
        let events = scripts.withLock { $0.isEmpty ? [] : $0.removeFirst() }
        return AsyncThrowingStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.finish()
        }
    }

    func cancel() { cancels.withLock { $0 += 1 } }
}

/// Queries a fixture search was asked.
final class QueryLog: Sendable {
    private let state = Mutex<[String]>([])
    func append(_ query: String) { state.withLock { $0.append(query) } }
    var queries: [String] { state.withLock { $0 } }
}

enum ToolLoopFixtures {
    static let request = AppGenerationRequest(
        modelDirectory: URL(fileURLWithPath: "/tmp/model"),
        prompt: "What are Swift actors?",
        maxNewTokens: 256,
        maxContextTokens: 8_192,
        tools: AppToolCatalog.definitions(for: .init(web: true, files: true)))

    static func diagnostics(prompt: Int = 400, generated: Int = 20,
                            stop: AppStopReason = .endOfTurn) -> AppDiagnostics {
        AppDiagnostics(generatedTokens: generated, stopReason: stop, promptTokenCount: prompt,
                       timeToFirstTokenSeconds: nil, decodeSeconds: 1, tokensPerSecond: 20,
                       peakMemoryBytes: nil, runtimeOptions: AppRuntimeOptions())
    }

    static func token(_ text: String) -> AppInferenceEvent {
        .token(AppTokenEvent(index: 0, textDelta: text, elapsedDecodeSeconds: 0))
    }

    static func search(_ query: String, id: String = "call_1") -> AppToolCall {
        AppToolCall(id: id, name: "web_search", arguments: .object(["query": .string(query)]))
    }

    static func callRound(_ calls: [AppToolCall], text: String = "") -> [AppInferenceEvent] {
        (text.isEmpty ? [] : [token(text)])
            + [.toolCalls(calls), .finished(diagnostics(stop: .toolCalls))]
    }

    static func answer(_ text: String) -> [AppInferenceEvent] {
        [token(text), .finished(diagnostics())]
    }

    static func toolbox(
        results: [AppWebResult] = [AppWebResult(
            title: "Actors", url: URL(string: "https://docs.example.org/actors")!,
            excerpt: "Actors isolate state.")],
        page: AppWebPage? = nil,
        passages: [AppFilePassage] = [],
        searchDelay: Duration? = nil,
        searches: QueryLog? = nil
    ) -> AppToolbox {
        AppToolbox(
            webSearch: { query, _ in
                searches?.append(query)
                if let searchDelay { try await Task.sleep(for: searchDelay) }
                return AppWebSearchResponse(provider: .duckDuckGo, results: results)
            },
            readWebpage: { url in
                page ?? AppWebPage(finalURL: url, title: "Page", text: "Full page text.")
            },
            searchFiles: { _, _ in passages })
    }

    /// Runs one answer and returns every event the app would see.
    static func run(_ client: ScriptedInferenceClient, toolbox: AppToolbox,
                    capabilities: AppChatCapabilities = .init(web: true, files: true),
                    limits: AppToolLimits = .standard,
                    request: AppGenerationRequest = request,
                    cancellation: AppAnswerCancellation = AppAnswerCancellation(),
                    userText: String = "What are Swift actors?") async -> [AppAnswerEvent] {
        let events = Mutex<[AppAnswerEvent]>([])
        await AppAnswerRunner(client: client, toolbox: toolbox, limits: limits)
            .run(request, capabilities: capabilities, firstSourceID: 1,
                 userText: userText, cancellation: cancellation) { event in
                events.withLock { $0.append(event) }
            }
        return events.withLock { $0 }
    }
}

@Suite struct AppToolCatalogTests {
    private let both = AppChatCapabilities(web: true, files: true)

    @Test func definitionsFollowTheCapabilities() {
        #expect(AppToolCatalog.definitions(for: .none).isEmpty)
        #expect(AppToolCatalog.definitions(for: .init(web: true)).map(\.name)
            == ["web_search", "read_webpage"])
        #expect(AppToolCatalog.definitions(for: .init(files: true)).map(\.name) == ["search_files"])
        #expect(AppToolCatalog.definitions(for: .init(workspaceRead: true)).map(\.name) == ["list_files", "read_file"])
        #expect(AppToolCatalog.systemInstructions(for: .none, currentDate: "x").isEmpty)
        let instructions = AppToolCatalog.systemInstructions(for: both, currentDate: "October 5, 2026")
        #expect(instructions.contains("Today is October 5, 2026."))
        #expect(instructions.contains("[1]"))
        #expect(!instructions.contains("\u{2014}"))
    }

    @Test func workspaceInstructionsMatchEnabledCapabilities() {
        let readOnly = AppToolCatalog.systemInstructions(for: .init(workspaceRead: true), currentDate: "today")
        #expect(readOnly.contains("may list and read"))
        #expect(readOnly.contains("writing and editing are not enabled"))
        #expect(readOnly.contains("Command execution is not enabled"))
        #expect(!readOnly.contains("[1]"))
        let full = AppToolCatalog.systemInstructions(for: .init(files: true, workspaceRead: true, fileWrite: true, command: true), currentDate: "today")
        #expect(full.contains("smallest exact edit"))
        #expect(full.contains("command network access is disabled"))
        #expect(full.contains("do not invent citations"))
        #expect(full.contains("when the user asks for it"))
        let networked = AppToolCatalog.systemInstructions(for: .init(command: true, commandNetwork: true), currentDate: "today")
        #expect(networked.contains("command network access enabled"))
    }

    @Test func repeatedFailedWorkspaceCallsWaitForAChange() async {
        let attempts = Mutex(0)
        let toolbox = AppToolbox(webSearch: { _, _ in throw AppToolValidationError("unused") },
            readWebpage: { _ in throw AppToolValidationError("unused") }, searchFiles: { _, _ in [] },
            coding: { call in
                if case .readFile = call {
                    let count = attempts.withLock { $0 += 1; return $0 }
                    if count == 1 { throw AppToolValidationError("missing file") }
                }
                return "ok"
            })
        let session = AppToolAnswerSession(capabilities: .init(workspaceRead: true, fileWrite: true),
            toolbox: toolbox, userText: "fix the workspace")
        func read(_ id: String) -> AppToolCall {
            .init(id: id, name: "read_file", arguments: .object(["path": .string("a.txt")]))
        }
        let write = AppToolCall(id: "write", name: "write_file", arguments: .object([
            "path": .string("a.txt"), "content": .string("created")]))
        let round = await session.execute(calls: [read("first"), read("second"), write], thinking: nil,
                                          content: "", characterBudget: 20_000, onActivity: { _ in })
        #expect(round.results.map(\.status) == [.failed, .refused, .succeeded])
        let retry = await session.execute(calls: [read("third")], thinking: nil,
                                          content: "", characterBudget: 20_000, onActivity: { _ in })
        #expect(retry.results.first?.status == .succeeded)
        #expect(attempts.withLock { $0 } == 2)
    }

    @Test func validCallsAreAccepted() throws {
        #expect(try AppToolCatalog.validate(ToolLoopFixtures.search("  swift  "), capabilities: both)
            == .webSearch(query: "swift", maxResults: 5))
        let counted = AppToolCall(id: "c", name: "search_files",
                                  arguments: .object(["query": .string("notes"),
                                                      "max_results": .string("3")]))
        #expect(try AppToolCatalog.validate(counted, capabilities: both)
            == .searchFiles(query: "notes", maxResults: 3))
        let page = AppToolCall(id: "c", name: "read_webpage",
                               arguments: .object(["url": .string("https://example.org/a")]))
        #expect(try AppToolCatalog.validate(page, capabilities: both)
            == .readWebpage(URL(string: "https://example.org/a")!))
    }

    @Test(arguments: [
        AppToolCall(id: "c", name: "run_shell", arguments: .object(["cmd": .string("ls")])),
        AppToolCall(id: "c", name: "web_search", arguments: .string("swift")),
        AppToolCall(id: "c", name: "web_search", arguments: .object([:])),
        AppToolCall(id: "c", name: "web_search", arguments: .object(["query": .string("   ")])),
        AppToolCall(id: "c", name: "web_search", arguments: .object(["query": .integer(4)])),
        AppToolCall(id: "c", name: "web_search",
                    arguments: .object(["query": .string(String(repeating: "a", count: 301))])),
        AppToolCall(id: "c", name: "web_search",
                    arguments: .object(["query": .string("x"), "region": .string("us")])),
        AppToolCall(id: "c", name: "web_search",
                    arguments: .object(["query": .string("x"), "max_results": .integer(9)])),
        AppToolCall(id: "c", name: "web_search",
                    arguments: .object(["query": .string("x"), "max_results": .number(2.5)])),
        AppToolCall(id: "c", name: "read_webpage", arguments: .object(["url": .string("file:///etc/hosts")])),
        AppToolCall(id: "c", name: "read_webpage", arguments: .object(["url": .string("not a url")])),
    ])
    func invalidCallsAreRefusedBeforeRunning(call: AppToolCall) {
        #expect(throws: AppToolValidationError.self) {
            _ = try AppToolCatalog.validate(call, capabilities: both)
        }
    }

    @Test func aCapabilityThatIsOffRefusesItsTools() {
        #expect(throws: AppToolValidationError.self) {
            _ = try AppToolCatalog.validate(ToolLoopFixtures.search("x"),
                                            capabilities: .init(files: true))
        }
    }
}

@Suite struct AppToolAnswerSessionTests {
    @Test func searchResultsAreNumberedAndPagesReuseTheirNumber() async {
        let session = AppToolAnswerSession(
            capabilities: .init(web: true), toolbox: ToolLoopFixtures.toolbox(),
            firstSourceID: 4, userText: "")
        let first = await session.execute(calls: [ToolLoopFixtures.search("actors")],
                                          thinking: nil, content: "", characterBudget: 20_000,
                                          onActivity: { _ in })
        #expect(first.results[0].status == .succeeded)
        #expect(first.results[0].modelText.contains("[4] Actors"))
        #expect(first.results[0].sourceIDs == [4])
        let read = AppToolCall(id: "call_2", name: "read_webpage",
                               arguments: .object(["url": .string("https://docs.example.org/actors#intro")]))
        let second = await session.execute(calls: [read], thinking: nil, content: "",
                                           characterBudget: 20_000, onActivity: { _ in })
        #expect(second.results[0].sourceIDs == [4])
        #expect(session.sources.count == 1)
        #expect(session.sources[0].excerpt == "Full page text.")
    }

    @Test func pagesNotFoundOrWrittenByTheUserAreNotRead() async {
        let session = AppToolAnswerSession(
            capabilities: .init(web: true), toolbox: ToolLoopFixtures.toolbox(),
            userText: "Summarize https://user.example.com/post please")
        let unseen = AppToolCall(id: "a", name: "read_webpage",
                                 arguments: .object(["url": .string("https://evil.example/collect?x=1")]))
        let written = AppToolCall(id: "b", name: "read_webpage",
                                  arguments: .object(["url": .string("https://user.example.com/post")]))
        let round = await session.execute(calls: [unseen, written], thinking: nil, content: "",
                                          characterBudget: 20_000, onActivity: { _ in })
        #expect(round.results[0].status == .refused)
        #expect(round.results[1].status == .succeeded)
    }

    @Test func aQueryCopyingLocalTextIsNotSentToTheProvider() async {
        let secret = "The quarterly revenue target for the northern region is forty two million dollars"
        let searches = QueryLog()
        let session = AppToolAnswerSession(
            capabilities: .init(web: true, files: true),
            toolbox: ToolLoopFixtures.toolbox(
                passages: [AppFilePassage(filePath: "/tmp/plan.md", folderID: UUID(), page: nil,
                                          lineStart: 1, lineEnd: 3, text: secret, score: 1)],
                searches: searches),
            userText: "Please search \"northern region revenue\"")
        let files = AppToolCall(id: "f", name: "search_files",
                                arguments: .object(["query": .string("revenue target")]))
        _ = await session.execute(calls: [files], thinking: nil, content: "",
                                  characterBudget: 20_000, onActivity: { _ in })
        let leak = ToolLoopFixtures.search("revenue target for the northern region is forty two million", id: "w")
        let round = await session.execute(calls: [leak], thinking: nil, content: "",
                                          characterBudget: 20_000, onActivity: { _ in })
        #expect(round.results[0].status == .refused)
        #expect(searches.queries.isEmpty)
        let fine = ToolLoopFixtures.search("northern region revenue", id: "w2")
        let allowed = await session.execute(calls: [fine], thinking: nil, content: "",
                                            characterBudget: 20_000, onActivity: { _ in })
        #expect(allowed.results[0].status == .succeeded)
    }

    @Test func directionControlsInAFileNameAreWrittenOutNotApplied() async {
        let name = "report\u{202E}fdp.exe"
        let session = AppToolAnswerSession(
            capabilities: .init(web: false, files: true),
            toolbox: ToolLoopFixtures.toolbox(
                passages: [AppFilePassage(filePath: "/tmp/\(name)", folderID: UUID(), page: nil,
                                          lineStart: 1, lineEnd: 1, text: "plain text", score: 1)],
                searches: QueryLog()),
            userText: "Please search the report")
        let files = AppToolCall(id: "f", name: "search_files",
                                arguments: .object(["query": .string("report")]))
        let round = await session.execute(calls: [files], thinking: nil, content: "",
                                          characterBudget: 20_000, onActivity: { _ in })
        let text = round.results[0].modelText
        #expect(text.contains("report\\u{202E}fdp.exe"))
        #expect(!text.unicodeScalars.contains("\u{202E}"))
    }

    @Test func fileNamesKeepOrdinaryCharacters() {
        #expect(AppFileNameText.safe("Übersicht 日本語 – notes (1).txt") == "Übersicht 日本語 – notes (1).txt")
        #expect(AppFileNameText.safe("a\u{200B}b\nc") == "a\\u{200B}b\\u{A}c")
    }

    @Test func callsBeyondTheRoundLimitAndRequestLimitAreRefused() async {
        var limits = AppToolLimits.standard
        limits.maximumCallsPerRound = 2
        limits.maximumWebRequests = 3
        let session = AppToolAnswerSession(capabilities: .init(web: true), limits: limits,
                                           toolbox: ToolLoopFixtures.toolbox(), userText: "")
        let calls = (1...3).map { ToolLoopFixtures.search("q\($0)", id: "c\($0)") }
        let round = await session.execute(calls: calls, thinking: nil, content: "",
                                          characterBudget: 20_000, onActivity: { _ in })
        #expect(round.results.map(\.status) == [.succeeded, .succeeded, .refused])
        let next = await session.execute(calls: Array(calls.prefix(2)), thinking: nil, content: "",
                                         characterBudget: 20_000, onActivity: { _ in })
        #expect(next.results.map(\.status) == [.succeeded, .refused])
    }

    @Test func afterItsRoundsTheModelIsToldToAnswer() async {
        var limits = AppToolLimits.standard
        limits.maximumToolRounds = 1
        let session = AppToolAnswerSession(capabilities: .init(web: true), limits: limits,
                                           toolbox: ToolLoopFixtures.toolbox(), userText: "")
        _ = await session.execute(calls: [ToolLoopFixtures.search("a")], thinking: nil,
                                  content: "", characterBudget: 20_000, onActivity: { _ in })
        #expect(session.mustAnswer)
        let late = await session.execute(calls: [ToolLoopFixtures.search("b", id: "c2")],
                                         thinking: nil, content: "", characterBudget: 20_000,
                                         onActivity: { _ in })
        #expect(late.results[0].status == .refused)
        #expect(late.results[0].modelText.contains("Answer now"))
    }

    @Test func aFullContextRefusesInsteadOfOverflowing() async {
        let session = AppToolAnswerSession(capabilities: .init(web: true),
                                           toolbox: ToolLoopFixtures.toolbox(), userText: "")
        let round = await session.execute(calls: [ToolLoopFixtures.search("a")], thinking: nil,
                                          content: "", characterBudget: 100, onActivity: { _ in })
        #expect(round.results[0].status == .refused)
        #expect(round.results[0].modelText.contains("context is nearly full"))
    }

    @Test func resultsAreCutToTheContextBudget() async {
        let long = String(repeating: "word ", count: 5_000)
        let session = AppToolAnswerSession(
            capabilities: .init(web: true),
            toolbox: ToolLoopFixtures.toolbox(page: AppWebPage(
                finalURL: URL(string: "https://docs.example.org/actors")!, title: "Long", text: long)),
            userText: "https://docs.example.org/actors")
        let read = AppToolCall(id: "r", name: "read_webpage",
                               arguments: .object(["url": .string("https://docs.example.org/actors")]))
        let round = await session.execute(calls: [read], thinking: nil, content: "",
                                          characterBudget: 1_500, onActivity: { _ in })
        #expect(round.results[0].modelText.count < 1_700)
        #expect(round.results[0].modelText.contains("Page text cut at"))
    }

    @Test func aFailedSearchSaysNothingWasRetrieved() async {
        let toolbox = AppToolbox(
            webSearch: { _, _ in throw AppWebSearchError.challenge(.duckDuckGo) },
            readWebpage: { _ in throw AppWebPageError.empty },
            searchFiles: { _, _ in [] })
        let session = AppToolAnswerSession(capabilities: .init(web: true), toolbox: toolbox,
                                           userText: "")
        let round = await session.execute(calls: [ToolLoopFixtures.search("a")], thinking: nil,
                                          content: "", characterBudget: 20_000, onActivity: { _ in })
        #expect(round.results[0].status == .failed)
        #expect(round.results[0].modelText.contains("verification step"))
        #expect(round.results[0].modelText.contains("do not invent"))
        #expect(session.sources.isEmpty)
    }

    @Test func activityIsReportedInOrder() async {
        let seen = Mutex<[AppToolActivity.State]>([])
        let session = AppToolAnswerSession(capabilities: .init(web: true),
                                           toolbox: ToolLoopFixtures.toolbox(), userText: "")
        _ = await session.execute(calls: [ToolLoopFixtures.search("a")], thinking: nil,
                                  content: "", characterBudget: 20_000,
                                  onActivity: { activity in seen.withLock { $0.append(activity.state) } })
        #expect(seen.withLock { $0 } == [.running, .succeeded])
    }
}

@Suite struct AppAnswerRunnerTests {
    @Test func aToolRoundFeedsResultsBackAsNativeRounds() async throws {
        let client = ScriptedInferenceClient([
            ToolLoopFixtures.callRound([ToolLoopFixtures.search("swift actors")],
                                       text: "Let me check."),
            ToolLoopFixtures.answer("Actors isolate state [1]."),
        ])
        let events = await ToolLoopFixtures.run(client, toolbox: ToolLoopFixtures.toolbox())
        let requests = client.requests
        #expect(requests.count == 2)
        #expect(requests[0].currentRounds.isEmpty)
        let round = try #require(requests[1].currentRounds.first)
        #expect(round.content == "Let me check.")
        #expect(round.calls == [ToolLoopFixtures.search("swift actors")])
        #expect(round.results.map(\.callID) == ["call_1"])
        #expect(round.results[0].modelText.contains("[1] Actors"))
        // The user's message is unchanged: results are never folded into it.
        #expect(requests[1].prompt == requests[0].prompt)
        guard case .finished(let diagnostics)? = events.last else {
            Issue.record("expected a finished answer, got \(String(describing: events.last))")
            return
        }
        #expect(diagnostics.toolRounds == 1)
        #expect(events.contains { if case .toolRoundFinished = $0 { true } else { false } })
    }

    @Test func anUnreadableCallIsRetriedOnceAndThenReported() async {
        let malformed: [AppInferenceEvent] = [.failed(.malformedToolCall("malformed call"), partial: nil)]
        let recovered = ScriptedInferenceClient([malformed, ToolLoopFixtures.answer("Fine.")])
        let first = await ToolLoopFixtures.run(recovered, toolbox: ToolLoopFixtures.toolbox())
        #expect(recovered.requests.count == 2)
        #expect(recovered.requests[0].systemPrompt != recovered.requests[1].systemPrompt)
        #expect(recovered.requests[1].systemPrompt.contains("invalid tool call"))
        #expect(first.contains(.retryingMalformedToolCall("malformed call")))
        guard case .finished? = first.last else { Issue.record("expected finish"); return }

        let failing = ScriptedInferenceClient([malformed, malformed, ToolLoopFixtures.answer("never")])
        let second = await ToolLoopFixtures.run(failing, toolbox: ToolLoopFixtures.toolbox())
        #expect(failing.requests.count == 2)
        #expect(second.last == .failed(.malformedToolCall("malformed call"), partial: nil))
    }

    @Test func aModelThatNeverStopsCallingIsStopped() async {
        var limits = AppToolLimits.standard
        limits.maximumToolRounds = 2
        let scripts = (0..<10).map { index in
            ToolLoopFixtures.callRound([ToolLoopFixtures.search("q\(index)", id: "c\(index)")])
        }
        let client = ScriptedInferenceClient(scripts)
        let events = await ToolLoopFixtures.run(client, toolbox: ToolLoopFixtures.toolbox(),
                                                limits: limits)
        // Two rounds, then refused calls, then the cap on generations.
        #expect(client.requests.count <= limits.maximumToolRounds + 2 + limits.maximumMalformedRetries)
        guard case .failed(let error, let partial)? = events.last else {
            Issue.record("expected a failure, got \(String(describing: events.last))")
            return
        }
        #expect(error.userMessage.contains("kept calling tools"))
        #expect((partial?.toolRounds ?? 0) >= 2)
    }

    @Test func stoppingDuringToolsEndsTheAnswerWithoutAnotherGeneration() async throws {
        let client = ScriptedInferenceClient([
            ToolLoopFixtures.callRound([ToolLoopFixtures.search("slow")]),
            ToolLoopFixtures.answer("should not run"),
        ])
        let cancellation = AppAnswerCancellation()
        let toolbox = ToolLoopFixtures.toolbox(searchDelay: .seconds(30))
        let started = Date()
        async let events = ToolLoopFixtures.run(client, toolbox: toolbox, cancellation: cancellation)
        try await Task.sleep(for: .milliseconds(200))
        cancellation.cancel()
        let finished = await events
        #expect(Date().timeIntervalSince(started) < 5)
        #expect(client.requests.count == 1)
        guard case .cancelled? = finished.last else {
            Issue.record("expected cancellation, got \(String(describing: finished.last))")
            return
        }
        let round = finished.compactMap { event -> AppToolRound? in
            if case .toolRoundFinished(let round, _) = event { return round }
            return nil
        }.first
        #expect(round?.results.first?.status == .cancelled)
    }

    @Test func aFailedGenerationEndsTheAnswer() async {
        let client = ScriptedInferenceClient([[.failed(.contextOverflow(prompt: 9_000, maxNew: 1,
                                                                        maxContext: 8_192), partial: nil)]])
        let events = await ToolLoopFixtures.run(client, toolbox: ToolLoopFixtures.toolbox())
        #expect(events.last == .failed(.contextOverflow(prompt: 9_000, maxNew: 1, maxContext: 8_192),
                                       partial: nil))
    }

    @Test func withoutCallsTheAnswerFinishesInOneGeneration() async {
        let client = ScriptedInferenceClient([ToolLoopFixtures.answer("Plain answer.")])
        let events = await ToolLoopFixtures.run(client, toolbox: ToolLoopFixtures.toolbox())
        #expect(client.requests.count == 1)
        guard case .finished(let diagnostics)? = events.last else { Issue.record("no finish"); return }
        #expect(diagnostics.toolRounds == 0)
        #expect(diagnostics.toolSeconds == nil)
    }
}

@Suite struct AppCitationTests {
    private let sources = [
        AppSource(id: 1, kind: .web, title: "A", url: "https://a.example", excerpt: "", origin: "x"),
        AppSource(id: 3, kind: .file, title: "notes.md", filePath: "/tmp/notes.md", excerpt: "",
                  origin: "Local files"),
    ]

    @Test func validAndInventedCitationsAreToldApart() {
        let check = AppCitations.check("Actors [1] and files [3, 7]. Also [9].", sources: sources)
        #expect(check.valid == [1, 3])
        #expect(check.invalid == [7, 9])
    }

    @Test func onlyValidNumbersAreLinked() {
        let text = "See [1] and [3, 7]. [12]"
        let ranges = AppCitations.numberRanges(in: text, sources: sources)
        #expect(ranges.map(\.number) == [1, 3])
        let ns = text as NSString
        #expect(ranges.map { ns.substring(with: $0.range) } == ["1", "3"])
        #expect(AppCitations.url(for: 3) == URL(string: "tuff-source:3"))
    }

    @Test func codeIsNotCheckedAsCitations() {
        let text = "Use `array[1]` or\n```\nlet x = y[3]\n```\nand [docs](https://x.example) [1]"
        #expect(AppCitations.check(text, sources: sources).valid == [1])
        #expect(AppCitations.check(text, sources: sources).invalid.isEmpty)
    }

    @Test func aLinkResolvesToItsSource() {
        #expect(AppCitations.source(for: URL(string: "tuff-source:3")!, in: sources)?.title == "notes.md")
        #expect(AppCitations.source(for: URL(string: "tuff-source:2")!, in: sources) == nil)
        #expect(AppCitations.source(for: URL(string: "https://x")!, in: sources) == nil)
        #expect(sources[1].openURL == URL(fileURLWithPath: "/tmp/notes.md"))
        #expect(AppSource(id: 5, kind: .web, title: "x", url: "javascript:alert(1)", excerpt: "",
                          origin: "x").openURL == nil)
    }
}

@Suite struct AppConversationMessageTests {
    @Test func toolRoundsRenderAsAssistantCallsAndToolResults() {
        let round = AppToolRound(thinking: "Search first.", content: "",
                                 calls: [ToolLoopFixtures.search("actors")],
                                 results: [AppToolResult(callID: "call_1", name: "web_search",
                                                         status: .succeeded, modelText: "[1] Actors",
                                                         summary: "1 result")])
        var request = ToolLoopFixtures.request
        request.systemPrompt = "Be brief."
        request.history = [AppChatTurn(prompt: "Earlier", response: "Before", toolRounds: [round])]
        request.currentRounds = [round]
        let messages = AppConversationMessages.messages(for: request, turns: request.history[...],
                                                        finalThinking: false)
        #expect(messages.map(\.role) == [.system, .user, .assistant, .tool, .assistant,
                                         .user, .assistant, .tool])
        #expect(messages[2].toolCalls.first?.name == "web_search")
        #expect(messages[2].thinking == "Search first.")
        #expect(messages[2].content == nil)
        #expect(messages[3].toolCallID == "call_1")
        #expect(messages[3].content == "[1] Actors")
        #expect(messages[5].content == request.prompt)
        let identities = AppConversationMessages.imageIdentities(for: request,
                                                                 turns: request.history[...])
        #expect(identities.count == messages.count)
    }

    @Test func anIncompleteRoundIsNeverSentBack() {
        var round = AppToolRound(calls: [ToolLoopFixtures.search("a")])
        #expect(!round.isComplete)
        let turn = AppChatTurn(prompt: "p", response: "r", toolRounds: [round])
        #expect(turn.requestTurn(carrying: []).toolRounds.isEmpty)
        var request = ToolLoopFixtures.request
        request.currentRounds = [round]
        #expect(throws: AppInferenceError.self) { try request.validate(requireModelDirectory: false) }
        round.results = [AppToolResult(callID: "call_1", name: "web_search", status: .refused,
                                       modelText: "Not run", summary: "")]
        #expect(round.isComplete)
    }
}

@Suite struct AppAnswerExportTests {
    @Test func theSavedAnswerKeepsItsSources() {
        let text = AppAnswerExport.markdown(
            prompt: "When are the tides?", response: "Twice a day [1]. See notes [2].",
            modelName: "Gemma 4 26B",
            sources: [AppSource(id: 1, kind: .web, title: "Tides", url: "https://sea.example/t",
                                excerpt: "", origin: "DuckDuckGo"),
                      AppSource(id: 2, kind: .file, title: "log.md", filePath: "/tmp/log.md",
                                location: "lines 1-4", excerpt: "", origin: "Local files")],
            date: Date(timeIntervalSince1970: 1_790_000_000))
        #expect(text.contains("Twice a day [1]."))
        #expect(text.contains("1. [Tides](https://sea.example/t) (DuckDuckGo)"))
        #expect(text.contains("2. log.md: /tmp/log.md, lines 1-4"))
        #expect(text.contains("Gemma 4 26B, 2026-"))
        #expect(!text.contains("\u{2014}"))
    }

    @Test func fileNamesComeFromTheQuestion() {
        #expect(AppAnswerExport.suggestedFileName(prompt: "When are the tides? /etc/x") == "When are the tides etcx.md")
        #expect(AppAnswerExport.suggestedFileName(prompt: "???") == "TUFF answer.md")
    }
}

@Suite struct LocalContextWebBoundaryTests {
    private func workspaceToolTurn(name: AppToolName, status: AppToolResultStatus,
                                   output: String) throws -> AppChatTurn {
        let argument: JSONValue = switch name {
        case .runCommand: .object(["command": .string("printf private")])
        case .readFile: .object(["path": .string("private.txt")])
        default: .object([:])
        }
        let call = AppToolCall(id: "local", name: name.rawValue, arguments: argument)
        let result = AppToolResult(callID: call.id, name: call.name, status: status,
                                   modelText: output, summary: "fixture")
        let round = AppToolRound(calls: [call], results: [result])
        let turn = AppChatTurn(prompt: "Inspect the workspace", response: "Done",
                               toolRounds: [round])
        return try JSONDecoder().decode(AppChatTurn.self, from: JSONEncoder().encode(turn))
    }

    private func assertHistoricalLocalTurnBlocksSearch(_ turn: AppChatTurn) async {
        var request = ToolLoopFixtures.request
        request.history = [turn]
        let client = ScriptedInferenceClient([
            ToolLoopFixtures.callRound([ToolLoopFixtures.search("private history marker")]),
            ToolLoopFixtures.answer("I will keep local output local.")])
        let queries = QueryLog()
        _ = await ToolLoopFixtures.run(client, toolbox: ToolLoopFixtures.toolbox(searches: queries),
                                      capabilities: .init(web: true), request: request,
                                      userText: "Summarize the project.")
        #expect(queries.queries.isEmpty)
        #expect(client.requests.last?.currentRounds.first?.results.first?.status == .refused)
    }

    @Test func onlyTheWholeMessageALineOrAQuotedPhraseMayBeSearchedOnceAfterLocalRetrieval() async {
        let queries = QueryLog()
        let message = "Find Swift actors\nplease use the \"structured concurrency\" docs\nand “task groups” too"
        let session = AppToolAnswerSession(capabilities: .init(web: true),
            toolbox: ToolLoopFixtures.toolbox(searches: queries), userText: message,
            hasLocalContext: true)
        // Secrets, encodings, a mere substring, a too-short quoted fragment.
        let refusedFirst = await session.execute(calls: [
            ToolLoopFixtures.search("secret-123"), ToolLoopFixtures.search("c2VjcmV0", id: "b"),
            ToolLoopFixtures.search("Swift", id: "c")],
            thinking: nil, content: "", characterBudget: 20_000, onActivity: { _ in })
        #expect(refusedFirst.results.map(\.status) == [.refused, .refused, .refused])
        let allowed = await session.execute(calls: [
            ToolLoopFixtures.search("Find Swift actors", id: "d"),
            ToolLoopFixtures.search("structured concurrency", id: "e"),
            ToolLoopFixtures.search("task groups", id: "f")],
            thinking: nil, content: "", characterBudget: 20_000, onActivity: { _ in })
        #expect(allowed.results.map(\.status) == [.succeeded, .succeeded, .succeeded])
        let repeated = await session.execute(calls: [ToolLoopFixtures.search("Find Swift actors", id: "g")],
            thinking: nil, content: "", characterBudget: 20_000, onActivity: { _ in })
        #expect(repeated.results.map(\.status) == [.refused])
        #expect(queries.queries == ["Find Swift actors", "structured concurrency", "task groups"])
    }

    @Test func theWholeSingleLineMessageStillWorksAndShortOnesDoNot() async {
        let queries = QueryLog()
        let session = AppToolAnswerSession(capabilities: .init(web: true),
            toolbox: ToolLoopFixtures.toolbox(searches: queries), userText: "Find Swift actors",
            hasLocalContext: true)
        let round = await session.execute(calls: [ToolLoopFixtures.search("Swift actors"),
                                                  ToolLoopFixtures.search("Find Swift actors", id: "b")],
            thinking: nil, content: "", characterBudget: 20_000, onActivity: { _ in })
        #expect(round.results.map(\.status) == [.refused, .succeeded])
        let tiny = AppToolAnswerSession(capabilities: .init(web: true),
            toolbox: ToolLoopFixtures.toolbox(searches: queries), userText: "a", hasLocalContext: true)
        let tinyRound = await tiny.execute(calls: [ToolLoopFixtures.search("a")],
            thinking: nil, content: "", characterBudget: 20_000, onActivity: { _ in })
        #expect(tinyRound.results.map(\.status) == [.refused])
        #expect(queries.queries == ["Find Swift actors"])
    }

    @Test func aFailedCodingToolStillMarksTheAnswerLocal() async {
        let queries = QueryLog()
        let toolbox = AppToolbox(
            webSearch: { query, _ in queries.append(query); throw AppToolValidationError("unused") },
            readWebpage: { _ in throw AppToolValidationError("unused") },
            searchFiles: { _, _ in [] },
            coding: { _ in throw AppToolValidationError("secret-name.txt could not be opened") })
        let session = AppToolAnswerSession(capabilities: .init(web: true, workspaceRead: true),
                                           toolbox: toolbox, userText: "Find Swift actors")
        let read = AppToolCall(id: "r", name: "read_file", arguments: .object(["path": .string("secret-name.txt")]))
        let round = await session.execute(calls: [read, ToolLoopFixtures.search("secret-name.txt", id: "s")],
            thinking: nil, content: "", characterBudget: 20_000, onActivity: { _ in })
        #expect(round.results.map(\.status) == [.failed, .refused])
        #expect(queries.queries.isEmpty)
    }

    @Test func persistedPriorFileSourcesStillProtectTheNextTurn() async throws {
        let source = AppSource(id: 1, kind: .file, title: "Private", filePath: "/tmp/private.txt",
                               excerpt: "pin 1234", origin: "Local files")
        let turn = AppChatTurn(prompt: "Read my note", response: "Done", sources: [source])
        // Encoding/decoding simulates reopening the conversation.
        let restored = try JSONDecoder().decode(AppChatTurn.self, from: JSONEncoder().encode(turn))
        var request = ToolLoopFixtures.request
        request.history = [restored]
        let client = ScriptedInferenceClient([
            ToolLoopFixtures.callRound([ToolLoopFixtures.search("pin 1234")]),
            ToolLoopFixtures.answer("I cannot send that query.")])
        let queries = QueryLog()
        _ = await ToolLoopFixtures.run(client, toolbox: ToolLoopFixtures.toolbox(searches: queries),
                                      request: request)
        #expect(queries.queries.isEmpty)
        #expect(client.requests.last?.currentRounds.first?.results.first?.status == .refused)
    }

    @Test func persistedFailedCommandOutputProtectsTheNextTurn() async throws {
        let turn = try workspaceToolTurn(name: .runCommand, status: .failed,
                                         output: "private history marker\n[Exit status 7.]")
        await assertHistoricalLocalTurnBlocksSearch(turn)
    }

    @Test func persistedReadFileOutputProtectsTheNextTurn() async throws {
        let turn = try workspaceToolTurn(name: .readFile, status: .succeeded,
                                         output: "1: private history marker")
        await assertHistoricalLocalTurnBlocksSearch(turn)
    }

    @Test func anAttachedDocumentCannotAuthorizeItsOwnURL() async {
        let read = AppToolCall(id: "a", name: "read_webpage",
                              arguments: .object(["url": .string("https://evil.example/collect")] ))
        let session = AppToolAnswerSession(capabilities: .init(web: true),
            toolbox: ToolLoopFixtures.toolbox(), userText: "Summarize my attachment",
            hasLocalContext: true)
        let round = await session.execute(calls: [read], thinking: nil, content: "",
            characterBudget: 20_000, onActivity: { _ in })
        #expect(round.results.first?.status == .refused)
    }
}

@Suite struct ToolExecutionTimeTests {
    @Test func modelGenerationTimeDoesNotSpendTheToolBudget() async throws {
        var limits = AppToolLimits.standard
        limits.maximumToolSeconds = 0.01
        let session = AppToolAnswerSession(capabilities: .init(web: true), limits: limits,
            toolbox: ToolLoopFixtures.toolbox(), userText: "")
        try await Task.sleep(for: .milliseconds(30))
        let round = await session.execute(calls: [ToolLoopFixtures.search("Swift")],
            thinking: nil, content: "", characterBudget: 10_000, onActivity: { _ in })
        #expect(round.results.first?.status == .succeeded)
    }
}
