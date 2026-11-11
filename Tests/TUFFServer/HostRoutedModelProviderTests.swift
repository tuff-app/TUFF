import Darwin
import Foundation
import Synchronization
import Testing
import TUFFAppCore
import TUFFDecodeProtocol
import TUFFEngine
import TUFFInferenceTypes
import TUFFModelCatalog
import TUFFRuntimeDescriptions
@testable import TUFFServerCore

// MARK: - A scripted serving host

/// What the fake host does for one generation.
private struct Script: Sendable {
    var thinking: [String] = []
    var text: [String] = ["Paris"]
    var toolCalls: [InferenceToolCall] = []
    var promptTokens = 7
    var tokens = 3
    var cached = 0
    var stopReason = "eos"
    var failure: String?
    /// Stay open after the text until released or cancelled.
    var hold = false
}

private struct HostRecord: Sendable {
    var resident: [String: ServingModelPreparation] = [:]
    var prepares: [ServingModelPreparation] = []
    var loads: [String] = []
    var unloads: [String] = []
    var started: [ServingGenerationRequest] = []
    var cancelled: Set<UUID> = []
    var ended: [UUID] = []
    var active = 0
    var endFailures = 0
    var scripts: [Script] = []
    var defaultScript = Script()
    var released = false
    var imageChecks: [(exists: Bool, contained: Bool)] = []
}

/// Speaks the real private serving protocol over a real socket, with the
/// coordinator's compatibility rules and a scripted model behind it.
private final class RecordStore: Sendable {
    private let box = Mutex(HostRecord())
    func withLock<R: Sendable>(_ body: (inout HostRecord) -> R) -> R { box.withLock { body(&$0) } }
}

private final class ScriptedAPIHost: @unchecked Sendable {
    let root: URL
    let profile: ServingHostProfile
    let record = RecordStore()
    private let server = Mutex<PrivateServingIPCServer?>(nil)

    init() throws {
        root = try servingTestRoot("api")
        profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.tuff.api")
        try start()
    }

    func start() throws {
        let record = self.record
        let created = PrivateServingIPCServer(
            profile: profile, token: try profile.token(), buildIdentity: "test",
            statusSnapshot: {
                record.withLock { value in
                    .init(loadedModelIDs: value.resident.keys.sorted(),
                          loadedModels: value.resident.keys.sorted().compactMap { id in
                              value.resident[id].map {
                                  ServingLoadedModel(modelID: id, maxContextTokens: $0.maxContextTokens,
                                                     runtimeOptions: $0.runtimeOptions,
                                                     forceLogitsHead: $0.forceLogitsHead)
                              }
                          },
                          activeRequests: value.active, openSessions: 0)
                }
            }
        ) { context, command, emit in
            let clientID = context.clientID
            _ = clientID
            switch command {
            case .prepare(let preparation):
                let outcome: ServingLoadState = record.withLock { value in
                    value.prepares.append(preparation)
                    if let existing = value.resident[preparation.modelID] {
                        let same = existing.maxContextTokens == preparation.maxContextTokens
                            && existing.runtimeOptions == preparation.runtimeOptions
                            && existing.forceLogitsHead == preparation.forceLogitsHead
                        return same ? .ready(loadSeconds: 0)
                            : .failed("Unload the model before changing its runtime configuration")
                    }
                    value.resident[preparation.modelID] = preparation
                    value.loads.append(preparation.modelID)
                    return .ready(loadSeconds: 0)
                }
                emit(.lifecycle(operationID: preparation.operationID, modelID: preparation.modelID,
                                state: outcome))
            case .unloadModel(let operationID, let modelID):
                let removed = record.withLock { value -> Bool in
                    guard value.resident[modelID] != nil, value.active == 0 else { return false }
                    value.resident.removeValue(forKey: modelID)
                    value.unloads.append(modelID)
                    return true
                }
                emit(.lifecycle(operationID: operationID, modelID: modelID,
                                state: removed ? .unloaded
                                    : .failed("Model is not loaded or has active requests")))
            case .start(let request):
                context.markAdmitted()
                await Self.run(request, clientID: clientID, record: record, emit: emit)
            case .cancel(_, let requestID):
                _ = record.withLock { $0.cancelled.insert(requestID) }
            case .endSession(let sessionID):
                let failed = record.withLock { value -> Bool in
                    value.ended.append(sessionID)
                    if value.endFailures > 0 { value.endFailures -= 1; return true }
                    return false
                }
                emit(failed ? .sessionEndFailed(sessionID: sessionID, message: "cleanup failed")
                    : .sessionEnded(sessionID: sessionID))
            default: break
            }
        }
        try created.start()
        server.withLock { $0 = created }
    }

    func stop() {
        server.withLock { $0 }?.shutdown()
        server.withLock { $0 = nil }
        record.withLock { value in
            value.resident = [:]
            value.active = 0
        }
    }

    func finish() { stop(); try? FileManager.default.removeItem(at: root) }

    private static func run(_ request: ServingGenerationRequest, clientID: UUID,
                            record: RecordStore,
                            emit: @escaping @Sendable (ServingEvent) -> Void) async {
        let identity = ServingRequestIdentity(clientID: clientID, request: request)
        let id = request.requestID
        let script = record.withLock { value -> Script in
            value.started.append(request)
            value.active += 1
            for attachment in request.payload.imageAttachments ?? [] {
                let url = URL(fileURLWithPath: attachment.path)
                value.imageChecks.append((
                    FileManager.default.isReadableFile(atPath: url.path),
                    AppImageAttachmentStore.contains(url)))
            }
            return value.scripts.isEmpty ? value.defaultScript : value.scripts.removeFirst()
        }
        defer { record.withLock { $0.active -= 1 } }
        func send(_ event: DecodeServiceEvent) {
            emit(.generation(identity: identity, event: event))
        }
        emit(.queue(identity: identity, state: .admitted, position: nil))
        if let failure = script.failure {
            send(DecodeServiceEvent(kind: .failed, generationID: id, error: failure))
            return
        }
        let prefill = DecodeServiceEvent(kind: .prefill, generationID: id, sequence: 1,
                                         prefillDone: 1, prefillTotal: 1)
        send(prefill)
        var sequence: UInt64 = 2
        for piece in script.thinking {
            send(DecodeServiceEvent(kind: .snapshot, generationID: id, sequence: sequence,
                                    thinkingDelta: piece))
            sequence += 1
        }
        for piece in script.text {
            send(DecodeServiceEvent(kind: .snapshot, generationID: id, sequence: sequence,
                                    textDelta: piece))
            sequence += 1
        }
        if script.hold {
            while !record.withLock({ $0.cancelled.contains(id) || $0.released }) {
                try? await Task.sleep(for: .milliseconds(5))
            }
            if record.withLock({ $0.cancelled.contains(id) }) {
                var cancelled = DecodeServiceEvent(kind: .cancelled, generationID: id,
                                                   tokenCount: script.tokens,
                                                   promptTokenCount: script.promptTokens)
                cancelled.stopReason = "cancelled"
                send(cancelled)
                return
            }
        }
        var finished = DecodeServiceEvent(
            kind: .finished, generationID: id, tokenCount: script.tokens,
            promptTokenCount: script.promptTokens, stopReason: script.stopReason)
        finished.cachedPromptTokens = script.cached
        finished.toolCalls = script.toolCalls.isEmpty ? nil : script.toolCalls
        send(finished)
    }
}

// MARK: - Harness

private let plan = AppAutomaticMemoryPlan(
    contextTokens: 8_192, expertCacheSlots: 16, estimatedWorkingSetBytes: 1 << 30,
    safeBudgetBytes: 8 << 30, denseResidencyPolicy: .resident, prefillChunkTokens: 128,
    retainedConversationBytes: 0)

private let model = TUFFModelCatalog.gemma4_E2B

private final class Harness: @unchecked Sendable {
    let root: URL
    let host: ScriptedAPIHost
    let provider: HostRoutedModelProvider
    let link: HostServingLink

    init(queueLimit: Int = 4, unloadDelay: TUFFModelUnloadDelay = .seconds(3_600),
         vision: Bool = false, startHost: Bool = true) throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("TUFFHostAPITests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try installFakeModel(model, root: root)
        if vision {
            let directory = root.appendingPathComponent(model.installDirectoryName)
            try FileManager.default.createDirectory(
                at: try VisionPackLocation.companionURL(forTextModel: directory),
                withIntermediateDirectories: true)
        }
        host = try ScriptedAPIHost()
        if !startHost { host.stop() }
        let profile = host.profile
        let connect: HostServingLink.Connector = {
            try PrivateServingIPCConnection.connect(
                profile: profile, clientID: try profile.clientID(for: "api"), role: .api)
        }
        link = HostServingLink(launching: connect, existing: connect)
        provider = HostRoutedModelProvider(
            installed: ServerInstalledModels(modelsRoot: root, device: .init(
                unifiedMemoryBytes: 16 << 30, macOSMajorVersion: 26, appleSiliconGeneration: 2)),
            settings: ServerSettingsSource(url: nil, fallback: .init(
                defaultModel: "gemma4-e2b", unloadDelay: unloadDelay, queueLimit: queueLimit)),
            link: link, automaticPlan: { _ in plan })
    }

    func finish() async {
        await provider.shutdown()
        host.finish()
        try? FileManager.default.removeItem(at: root)
    }

    func validated(_ messages: [[String: Any]], extra: [String: Any] = [:]) throws -> ValidatedChatRequest {
        var body: [String: Any] = ["model": "default", "messages": messages]
        for (key, value) in extra { body[key] = value }
        let decoded = try JSONDecoder().decode(
            OpenAIChatRequest.self, from: try JSONSerialization.data(withJSONObject: body))
        let target = try provider.route("default")
        return try OpenAIRequestValidator.validate(
            decoded, modelID: target.requestedID, dialect: target.dialect)
    }

    /// One request through the provider the way the HTTP layer drives it.
    func complete(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void = { _ in },
                  onQueued: @escaping @Sendable () -> Void = {}) async throws -> ServerCompletion {
        try await provider.run(try provider.route("default"), onQueued: onQueued,
            prepare: { backend in try await backend.prepare(request) },
            operation: { backend, prepared in try await backend.generate(prepared, onEvent: onEvent) })
    }
}

private final class Box<Value: Sendable>: Sendable {
    let value: Mutex<Value>
    init(_ initial: Value) { value = Mutex(initial) }
}

private func eventually(_ condition: () -> Bool) async -> Bool {
    for _ in 0..<600 {
        if condition() { return true }
        try? await Task.sleep(for: .milliseconds(10))
    }
    return condition()
}

private func user(_ text: String) -> [String: Any] { ["role": "user", "content": text] }

private struct Reply: Sendable {
    let status: Int
    let data: Data
    var text: String { String(decoding: data, as: UTF8.self) }
    var json: [String: Any]? { (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] }
}

private func post(_ port: Int, _ body: [String: Any], path: String = "/v1/chat/completions") async throws -> Reply {
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    let (data, response) = try await URLSession.shared.data(for: request)
    return Reply(status: (response as? HTTPURLResponse)?.statusCode ?? 0, data: data)
}

// MARK: - Tests

@Suite("Host-backed API", .serialized) struct HostRoutedModelProviderTests {
    @Test func stalledGenerationIsBoundedAndCancelled() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        h.host.record.withLock {
            $0.defaultScript = Script(text: Array(repeating: "x", count: PrivateServingIPCClient.streamCapacity * 3), hold: true)
        }
        let client = try await h.link.attach()
        let id = UUID()
        let payload = DecodeGenerationRequest(prompt: "hello", maxNewTokens: 8,
                                              maxContextTokens: 128, temperature: 0,
                                              generationID: id)
        let request = try ServingGenerationRequest(modelID: model.apiModelID, sessionID: UUID(),
                                                   requestID: id, payload: payload)
        let generation = HostServingGeneration(link: h.link, client: client, request: request)
        #expect(await eventually { h.host.record.withLock { $0.cancelled.contains(id) } })
        var failed = false
        do {
            while try await generation.next() != nil {}
        } catch { failed = true }
        #expect(failed)
        await generation.drain(cancel: true)
    }

    @Test func aTextRequestIsRoutedTranslatedAndReturnsUsage() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        h.host.record.withLock { $0.defaultScript = Script(text: ["Pa", "ris"], promptTokens: 9, tokens: 2, cached: 4) }
        let server = TUFFHTTPServer(provider: h.provider)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let reply = try await post(port, [
            "model": "default", "max_tokens": 50, "temperature": 0.5, "top_p": 0.8, "top_k": 20, "seed": 7,
            "messages": [["role": "system", "content": "Be brief."], ["role": "developer", "content": "Answer in one word."],
                         user("Capital of France?"), ["role": "assistant", "content": "Let me think."],
                         user("Say it.")]])
        #expect(reply.status == 200)
        let choice = try #require((reply.json?["choices"] as? [[String: Any]])?.first)
        #expect((choice["message"] as? [String: Any])?["content"] as? String == "Paris")
        #expect(choice["finish_reason"] as? String == "stop")
        let usage = try #require(reply.json?["usage"] as? [String: Any])
        #expect(usage["prompt_tokens"] as? Int == 9)
        #expect(usage["completion_tokens"] as? Int == 2)
        #expect(usage["total_tokens"] as? Int == 11)
        #expect((usage["prompt_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 4)

        let started = try #require(h.host.record.withLock { $0.started.first })
        let payload = started.payload
        #expect(started.modelID == model.apiModelID)
        #expect(payload.prompt == "Say it.")
        #expect(payload.systemPrompt == "Be brief.\n\nAnswer in one word.")
        #expect(payload.history.count == 1)
        #expect(payload.history[0].prompt == "Capital of France?" && payload.history[0].response == "Let me think.")
        #expect(payload.maxNewTokens == 50 && payload.temperature == 0.5)
        #expect(payload.topP == 0.8 && payload.topK == 20 && payload.seed == 7)
        #expect(payload.maxContextTokens == 8_192)
        #expect(payload.tools == nil && payload.currentRounds == nil && payload.imageAttachments == nil)
        #expect(payload.preservesFullHistory)
        #expect(payload.generationID == started.requestID)
        // The model was prepared once, with the plan's configuration and a
        // sampling logits head because the request samples.
        let prepares = h.host.record.withLock { $0.prepares }
        #expect(prepares.count == 1)
        #expect(prepares[0].maxContextTokens == 8_192 && prepares[0].forceLogitsHead)
        #expect(prepares[0].runtimeOptions == payload.runtimeOptions)
        try await server.shutdown()
        await h.finish()
    }

    @Test func streamingCarriesReasoningContentAndUsage() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        h.host.record.withLock { $0.defaultScript = Script(thinking: ["hmm"], text: ["Hel", "lo"], tokens: 4) }
        let server = TUFFHTTPServer(provider: h.provider)
        let port = try #require(try await server.start(port: 0).localAddress?.port)
        let reply = try await post(port, ["model": "default", "stream": true,
            "stream_options": ["include_usage": true], "messages": [user("hi")]])
        #expect(reply.status == 200)
        #expect(reply.text.contains(#""reasoning_content":"hmm""#))
        #expect(reply.text.contains(#""content":"Hel""#) && reply.text.contains(#""content":"lo""#))
        #expect(reply.text.contains(#""finish_reason":"stop""#))
        #expect(reply.text.contains(#""completion_tokens":4"#))
        #expect(reply.text.contains("data: [DONE]"))
        try await server.shutdown()
        await h.finish()
    }

    @Test func aToolCallRoundTrips() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        let call = InferenceToolCall(id: "call_1", name: "get_weather",
                                     arguments: .object(["city": .string("Paris")]))
        h.host.record.withLock { value in
            value.scripts = [Script(text: [], toolCalls: [call], stopReason: "toolCalls"),
                             Script(text: ["Sunny"])]
        }
        let tools: [[String: Any]] = [["type": "function", "function": [
            "name": "get_weather", "description": "Weather",
            "parameters": ["type": "object", "properties": ["city": ["type": "string"]],
                           "required": ["city"]]]]]
        let server = TUFFHTTPServer(provider: h.provider)
        let port = try #require(try await server.start(port: 0).localAddress?.port)
        let first = try await post(port, ["model": "default", "tools": tools,
                                          "messages": [user("Weather in Paris?")]])
        let choice = try #require((first.json?["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "tool_calls")
        let calls = try #require((choice["message"] as? [String: Any])?["tool_calls"] as? [[String: Any]])
        #expect(calls.count == 1)
        let function = try #require(calls[0]["function"] as? [String: Any])
        #expect(function["name"] as? String == "get_weather")
        #expect((function["arguments"] as? String)?.contains("Paris") == true)

        let second = try await post(port, ["model": "default", "tools": tools, "messages": [
            user("Weather in Paris?"),
            ["role": "assistant", "content": NSNull(), "tool_calls": [[
                "id": "call_1", "type": "function",
                "function": ["name": "get_weather", "arguments": #"{"city":"Paris"}"#]]]],
            ["role": "tool", "tool_call_id": "call_1", "content": "22C sunny"]]])
        #expect(second.status == 200)
        let started = h.host.record.withLock { $0.started }
        #expect(started.count == 2)
        #expect(started[0].payload.tools?.map(\.name) == ["get_weather"])
        let rounds = try #require(started[1].payload.currentRounds)
        #expect(rounds.count == 1)
        #expect(rounds[0].calls.map(\.name) == ["get_weather"])
        #expect(rounds[0].results == [DecodeToolResult(callID: "call_1", name: "get_weather", content: "22C sunny")])
        #expect(started[1].payload.prompt == "Weather in Paris?")
        try await server.shutdown()
        await h.finish()
    }

    @Test func aToolRoundInHistoryBecomesATurnWithRounds() throws {
        let call = InferenceToolCall(id: "c", name: "f", arguments: .object([:]))
        let messages: [[String: Any]] = [
            user("one"),
            ["role": "assistant", "content": "checking", "tool_calls": [[
                "id": "c", "type": "function", "function": ["name": "f", "arguments": "{}"]]]],
            ["role": "tool", "tool_call_id": "c", "content": "ok"],
            ["role": "assistant", "content": "done"],
            user("two")]
        let decoded = try JSONDecoder().decode(OpenAIChatRequest.self, from: try JSONSerialization.data(
            withJSONObject: ["model": "m", "messages": messages]))
        let request = try OpenAIRequestValidator.validate(decoded, modelID: "m", dialect: .gemma)
        let payload = try HostServingTranslation.decodeRequest(
            request, generationID: UUID(), maxContextTokens: 8_192,
            runtimeOptions: DecodeRuntimeOptions(), images: [:])
        #expect(payload.prompt == "two" && payload.history.count == 1)
        let turn = payload.history[0]
        #expect(turn.prompt == "one" && turn.response == "done")
        #expect(turn.toolRounds?.first?.content == "checking")
        #expect(turn.toolRounds?.first?.calls == [call])
        #expect(turn.toolRounds?.first?.results.first?.content == "ok")
    }

    @Test(arguments: [
        #"[{"role":"user","content":"a"},{"role":"user","content":"b"}]"#,
        #"[{"role":"assistant","content":"a"},{"role":"user","content":"b"}]"#,
        #"[{"role":"user","content":"a"},{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}],"content":null}]"#,
    ])
    func messageShapesTheDecodePathCannotExpressAreRefused(json: String) async throws {
        let messages = try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]])
        let h = try Harness(); defer { Task { await h.finish() } }
        let server = TUFFHTTPServer(provider: h.provider)
        let port = try #require(try await server.start(port: 0).localAddress?.port)
        let reply = try await post(port, ["model": "default", "messages": messages])
        #expect(reply.status == 400)
        #expect(reply.text.contains("unsupported_value"))
        #expect(h.host.record.withLock { $0.started.isEmpty })
        try await server.shutdown()
        await h.finish()
    }

    @Test func aTrailingAssistantMessageContinuesAndStopStringsEndTheReply() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        h.host.record.withLock { $0.defaultScript = Script(text: ["Hello ST", "OP and more"], hold: true) }
        let request = try h.validated(
            [user("say"), ["role": "assistant", "content": "Well,"]], extra: ["stop": ["STOP"]])
        let collected = Box("")
        let completion = try await h.complete(request) { event in
            if case .content(let text) = event { collected.value.withLock { $0 += text } }
        }
        #expect(completion.content == "Hello ")
        #expect(collected.value.withLock { $0 } == "Hello ")
        #expect(completion.finishReason == "stop")
        let started = try #require(h.host.record.withLock { $0.started.first })
        #expect(started.payload.assistantPrefix == "Well,")
        #expect(h.host.record.withLock { $0.cancelled.contains(started.requestID) })
        await h.finish()
    }

    @Test func aHostRejectionForContextOverflowIsABadRequest() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        h.host.record.withLock { $0.defaultScript = Script(
            failure: "Prompt (9000 tokens) plus max response (0) exceeds the 8192-token context.") }
        let server = TUFFHTTPServer(provider: h.provider)
        let port = try #require(try await server.start(port: 0).localAddress?.port)
        for stream in [false, true] {
            let reply = try await post(port, ["model": "default", "stream": stream, "messages": [user("big")]])
            #expect(reply.status == 400)
            #expect(reply.text.contains("context_length_exceeded"))
        }
        try await server.shutdown()
        await h.finish()
    }

    @Test func clientDisconnectCancelsTheHostGenerationAndReleasesTheSlot() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        h.host.record.withLock { $0.scripts = [Script(hold: true)] }
        let request = try h.validated([user("long")])
        let task = Task { try await h.complete(request) }
        #expect(await eventually { h.host.record.withLock { $0.started.count == 1 } })
        let started = try #require(h.host.record.withLock { $0.started.first })
        task.cancel()
        #expect(await eventually { h.host.record.withLock { $0.cancelled.contains(started.requestID) } })
        switch await task.result {
        case .success: Issue.record("a cancelled request must not complete")
        case .failure(let error): #expect(error is CancellationError)
        }
        // Drained: the private session was ended and the slot is free.
        #expect(h.host.record.withLock { $0.ended } == [started.sessionID])
        #expect(!(await h.provider.isActive))
        let next = try await h.complete(try h.validated([user("again")]))
        #expect(next.content == "Paris")
        await h.finish()
    }

    @Test func aRestartedHostIsReconnectedAndReloaded() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        let first = try await h.complete(try h.validated([user("a")]))
        #expect(first.content == "Paris")
        #expect(h.host.record.withLock { $0.loads } == [model.apiModelID])
        h.host.stop()
        try h.host.start()
        // The model is gone with the old host; the next request loads it
        // again over a fresh connection.
        let second = try await h.complete(try h.validated([user("b")]))
        #expect(second.content == "Paris")
        #expect(h.host.record.withLock { $0.loads } == [model.apiModelID, model.apiModelID])
        #expect(h.host.record.withLock { $0.started.count } == 2)
        #expect(h.host.record.withLock { $0.prepares.count } == 2)
        await h.finish()
    }

    @Test func aRequestWhileTheHostIsDownFailsClearlyThenRecovers() async throws {
        let h = try Harness(startHost: false); defer { Task { await h.finish() } }
        await #expect(throws: ServerRequestError.self) {
            _ = try await h.complete(try h.validated([user("a")]))
        }
        try h.host.start()
        #expect(try await h.complete(try h.validated([user("a")])).content == "Paris")
        await h.finish()
    }

    @Test func theQueueIsBoundedAsBefore() async throws {
        let h = try Harness(queueLimit: 1); defer { Task { await h.finish() } }
        h.host.record.withLock { $0.scripts = [Script(hold: true)] }
        let queued = Box(0)
        let first = Task { try await h.complete(try h.validated([user("1")])) }
        #expect(await eventually { h.host.record.withLock { $0.started.count == 1 } })
        let second = Task { try await h.complete(try h.validated([user("2")]), onQueued: { queued.value.withLock { $0 += 1 } }) }
        #expect(await eventually { queued.value.withLock { $0 == 1 } })
        #expect(await h.provider.queuedCount == 1)
        await #expect(throws: ServerRequestError.queueFull) {
            _ = try await h.complete(try h.validated([user("3")]))
        }
        h.host.record.withLock { $0.released = true }
        #expect(try await first.value.content == "Paris")
        #expect(try await second.value.content == "Paris")
        await h.finish()
    }

    @Test func requestsShareOnePreparedModelAndAdoptTheResidentConfiguration() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        // The app loaded the model first, with a smaller context and other cache settings.
        let resident = try ServingModelPreparation(
            modelID: model.apiModelID, maxContextTokens: 4_096,
            runtimeOptions: DecodeRuntimeOptions(expertCacheSlots: 8, prefillEnabled: false),
            forceLogitsHead: true)
        h.host.record.withLock { $0.resident[model.apiModelID] = resident }
        for text in ["one", "two"] {
            #expect(try await h.complete(try h.validated([user(text)])).content == "Paris")
        }
        let record = h.host.record.withLock { $0 }
        #expect(record.loads.isEmpty && record.unloads.isEmpty)
        #expect(record.prepares.count == 2)
        for request in record.started {
            #expect(request.payload.maxContextTokens == 4_096)
            #expect(request.payload.runtimeOptions == resident.runtimeOptions)
        }
        // The app's model is not the API's to unload.
        #expect(await h.provider.unloadIfIdle())
        #expect(h.host.record.withLock { $0.resident[model.apiModelID] != nil })
        await h.finish()
    }

    @Test func twoAPIRequestsLoadTheModelOnce() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        _ = try await h.complete(try h.validated([user("one")]))
        _ = try await h.complete(try h.validated([user("two")]))
        let record = h.host.record.withLock { $0 }
        #expect(record.loads == [model.apiModelID])
        #expect(record.prepares.count == 2)
        #expect(Set(record.started.map(\.sessionID)).count == 2)
        // It is the API's own load, so releasing an idle API unloads it.
        #expect(await h.provider.unloadIfIdle())
        #expect(h.host.record.withLock { $0.unloads } == [model.apiModelID])
        await h.finish()
    }

    @Test func replacingOwnSamplingModelPreservesIdleUnloadOwnership() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        _ = try await h.complete(try h.validated([user("sample")], extra: ["temperature": 0.8]))
        _ = try await h.complete(try h.validated([user("greedy")], extra: ["temperature": 0]))
        #expect(h.host.record.withLock { $0.loads.count } == 2)
        #expect(await h.provider.unloadIfIdle())
        #expect(h.host.record.withLock { $0.resident.isEmpty })
        #expect(h.host.record.withLock { $0.unloads.count } == 2)
        await h.finish()
    }

    @Test func aGreedyRequestAgainstASamplingModelReplacesItOnlyWhenIdle() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        let resident = try ServingModelPreparation(
            modelID: model.apiModelID, maxContextTokens: 4_096,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: true)
        h.host.record.withLock { $0.resident[model.apiModelID] = resident }
        let greedy = try h.validated([user("x")], extra: ["temperature": 0])
        h.host.record.withLock { $0.active = 1 }
        await #expect(throws: ServerRequestError.self) { _ = try await h.complete(greedy) }
        #expect(h.host.record.withLock { $0.unloads.isEmpty })
        h.host.record.withLock { $0.active = 0 }
        #expect(try await h.complete(greedy).content == "Paris")
        let record = h.host.record.withLock { $0 }
        #expect(record.unloads == [model.apiModelID] && record.loads == [model.apiModelID])
        // The resident context and cache settings were kept.
        #expect(record.resident[model.apiModelID]?.maxContextTokens == 4_096)
        #expect(record.resident[model.apiModelID]?.forceLogitsHead == false)
        await h.finish()
    }

    @Test func sessionsAreEndedAndAFailedCleanupIsRetried() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        h.host.record.withLock { $0.endFailures = 1 }
        _ = try await h.complete(try h.validated([user("a")]))
        let record = h.host.record.withLock { $0 }
        let session = try #require(record.started.first?.sessionID)
        // The first cleanup failed and was retried until the host confirmed.
        #expect(record.ended == [session, session])
        #expect(await h.provider.state.withLock { $0.failedSessionEnds.isEmpty })
        await h.finish()
    }

    @Test func aCleanupThatKeepsFailingIsRemembered() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        h.host.record.withLock { $0.endFailures = 3 }
        _ = try await h.complete(try h.validated([user("a")]))
        let session = try #require(h.host.record.withLock { $0.started.first?.sessionID })
        #expect(h.provider.state.withLock { $0.failedSessionEnds } == [session])
        // The next request retries it first.
        _ = try await h.complete(try h.validated([user("b")]))
        #expect(h.provider.state.withLock { $0.failedSessionEnds.isEmpty })
        #expect(h.host.record.withLock { $0.ended.filter { $0 == session }.count } == 4)
        await h.finish()
    }

    @Test func aPromptCacheKeyKeepsItsSessionUntilTheAPIIsIdle() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        for key in ["chat-a", "chat-a", "chat-b"] {
            _ = try await h.complete(try h.validated([user("hi")], extra: ["prompt_cache_key": key]))
        }
        let record = h.host.record.withLock { $0 }
        #expect(record.started[0].sessionID == record.started[1].sessionID)
        #expect(record.started[2].sessionID != record.started[0].sessionID)
        #expect(record.started.allSatisfy { $0.payload.conversationKey != nil })
        #expect(record.ended.isEmpty)
        #expect(await h.provider.unloadIfIdle())
        #expect(Set(h.host.record.withLock { $0.ended }) == [record.started[0].sessionID, record.started[2].sessionID])
        await h.finish()
    }

    @Test func anIdleDelayOfZeroUnloadsTheModelAfterTheResponse() async throws {
        let h = try Harness(unloadDelay: .immediately); defer { Task { await h.finish() } }
        _ = try await h.complete(try h.validated([user("a")]))
        #expect(await eventually { h.host.record.withLock { $0.unloads == [model.apiModelID] } })
        await h.finish()
    }

    @Test func healthReportsWhatTheHostHoldsWithoutStartingIt() async throws {
        let down = try Harness(startHost: false); defer { Task { await down.finish() } }
        let none = await down.provider.health()
        #expect(none.loadedModel == nil)
        #expect(down.host.record.withLock { $0.prepares.isEmpty })

        let h = try Harness(); defer { Task { await h.finish() } }
        #expect(await h.provider.health().loadedModel == nil)
        h.host.record.withLock { $0.resident[model.apiModelID] = try! ServingModelPreparation(
            modelID: model.apiModelID, maxContextTokens: 4_096,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false) }
        let health = await h.provider.health()
        #expect(health.loadedModel == model.apiModelID)
        #expect(health.activeRequests == 0 && health.queuedRequests == 0)
        let status = await h.provider.controlStatus(version: "t")
        #expect(status.residentModel == model.apiModelID)
        await h.finish()
        await down.finish()
    }

    @Test func imagesAreReadableByTheHostUntilTheEndAndThenRemoved() async throws {
        let h = try Harness(vision: true); defer { Task { await h.finish() } }
        let server = TUFFHTTPServer(provider: h.provider)
        let port = try #require(try await server.start(port: 0).localAddress?.port)
        let pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII="
        let reply = try await post(port, ["model": "default", "messages": [["role": "user", "content": [
            ["type": "text", "text": "Describe"],
            ["type": "image_url", "image_url": ["url": "data:image/png;base64,\(pixel)"]]]]]])
        #expect(reply.status == 200)
        let record = h.host.record.withLock { $0 }
        let request = try #require(record.started.first)
        let attachment = try #require(request.payload.imageAttachments?.first)
        #expect(record.imageChecks.count == 1 && record.imageChecks[0].exists && record.imageChecks[0].contained)
        #expect(attachment.encodedBytes > 0 && attachment.sha256.count == 64)
        #expect(!FileManager.default.fileExists(atPath: attachment.path))
        try await server.shutdown()
        await h.finish()
    }

    @Test func theHostHeaderCheckAndLoopbackBindStillApply() async throws {
        let h = try Harness(); defer { Task { await h.finish() } }
        let server = TUFFHTTPServer(provider: h.provider)
        let channel = try await server.start(port: 0)
        #expect(channel.localAddress?.ipAddress == "127.0.0.1")
        let port = try #require(channel.localAddress?.port)
        let body = #"{"model":"default","messages":[{"role":"user","content":"hi"}]}"#
        func send(_ host: String) throws -> String {
            let socket = try connectedSocket(port: port)
            defer { _ = Darwin.close(socket) }
            try writeAll(socket: socket, text: "POST /v1/chat/completions HTTP/1.1\r\nHost: \(host)\r\n"
                + "Content-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\n"
                + "Connection: close\r\n\r\n\(body)")
            return try readUntil(socket: socket, timeoutMilliseconds: 4_000) {
                $0.contains("\r\n\r\n") && ($0.hasSuffix("}") || $0.hasSuffix("[DONE]\n\n"))
            }
        }
        for refused in ["attacker.example:\(port)", "127.0.0.1.attacker.example", "0.0.0.0:\(port)"] {
            let response = try send(refused)
            #expect(response.hasPrefix("HTTP/1.1 403"), "\(refused)")
            #expect(response.contains("forbidden_host"))
        }
        // Nothing reached the host for a refused request.
        #expect(h.host.record.withLock { $0.started.isEmpty && $0.prepares.isEmpty })
        let served = try send("127.0.0.1:\(port)")
        #expect(served.hasPrefix("HTTP/1.1 200"))
        #expect(h.host.record.withLock { $0.started.count } == 1)
        try await server.shutdown()
        await h.finish()
    }
}
