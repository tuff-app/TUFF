import Foundation
import Synchronization
import Testing
import TUFFAppCore
import TUFFInferenceTypes
@testable import TUFFServerCore

private final class AppleServerStub: AppInferenceClient {
    let captured = Mutex<AppGenerationRequest?>(nil)
    func cancel() {}
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        captured.withLock { $0 = request }
        return AsyncThrowingStream { continuation in
            continuation.yield(.token(.init(index: 0, textDelta: "blue", elapsedDecodeSeconds: 0)))
            continuation.finish()
        }
    }
}

@Suite struct AppleServerTests {
    @Test func discoveryAndDefaultRouteNeedNoInstalledModelOrHelper() async throws {
        let unavailable = HostServingLink(launching: { throw CocoaError(.fileNoSuchFile) },
                                          existing: { throw CocoaError(.fileNoSuchFile) })
        let provider = HostRoutedModelProvider(
            installed: .init(modelsRoot: URL(fileURLWithPath: "/nonexistent-tuff-test-models")),
            settings: .init(url: nil, fallback: .init(defaultModel: "apple.on-device")),
            link: unavailable, appleAvailability: { .available })
        #expect(provider.modelList().data.map(\.id) == ["apple.on-device"])
        let route = try provider.route("default")
        #expect(route.id == "apple.on-device")
        let value = try await provider.run(route, onQueued: {}, prepare: { _ in "prepared" },
                                           operation: { _, value in value })
        #expect(value == "prepared")
        #expect(!unavailable.isAttached)
        await provider.shutdown()
    }

    @Test func unavailableModelIsNotAdvertised() throws {
        let link = HostServingLink(launching: { throw CocoaError(.fileNoSuchFile) },
                                  existing: { throw CocoaError(.fileNoSuchFile) })
        let provider = HostRoutedModelProvider(
            installed: .init(modelsRoot: URL(fileURLWithPath: "/nonexistent-tuff-test-models")),
            settings: .init(url: nil, fallback: .init()), link: link,
            appleAvailability: { .intelligenceDisabled })
        #expect(provider.modelList().data.isEmpty)
        #expect(throws: ServerRequestError.self) { try provider.route("apple.on-device") }
    }

    @Test func backendStreamsAndPreservesSystemAndHistoryWithoutInventingUsage() async throws {
        let client = AppleServerStub()
        let actual = AppleServerInferenceBackend(client: client)
        let request = ValidatedChatRequest(messages: [
            .init(role: .system, content: "Be brief"),
            .init(role: .user, content: "Remember blue"), .init(role: .assistant, content: "OK"),
            .init(role: .user, content: "Which color?")], tools: [], stream: true,
            includeUsage: true, generationConfig: .init(temperature: 0), maximumCompletionTokens: 32)
        let chunks = Mutex<[String]>([])
        let prepared = try await actual.prepare(request)
        let completion = try await actual.generate(prepared) { event in
            if case .content(let delta) = event { chunks.withLock { $0.append(delta) } }
        }
        #expect(chunks.withLock { $0 } == ["blue"])
        #expect(completion.content == "blue")
        #expect(client.captured.withLock { $0?.systemPrompt } == "Be brief")
        #expect(client.captured.withLock { $0?.history.count } == 1)
        let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(completion.usage)) as! [String: Any]
        #expect(json["completion_tokens"] is NSNull)
        #expect(json["token_counts_available"] as? Bool == false)
        #expect(try JSONDecoder().decode(OpenAIUsage.self, from: JSONEncoder().encode(completion.usage)) == completion.usage)
    }

    @Test func unsupportedParametersFailBeforeGenerationIncludingDefaultAlias() throws {
        for parameter in ["top_p": "0.9", "seed": "1", "enable_thinking": "true", "stop": "\"end\""] {
            let json = "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"\(parameter.key)\":\(parameter.value)}"
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
            #expect(throws: ServerRequestError.self) {
                try OpenAIRequestValidator.validate(request, modelID: "default", dialect: .chatml,
                    preStagedImages: [:], attachmentLease: nil, appleModel: true)
            }
        }
    }
}

private final class HoldingAppleServerStub: AppInferenceClient {
    let started = AsyncStream<Void>.makeStream()
    let continuation = Mutex<AsyncThrowingStream<AppInferenceEvent, Error>.Continuation?>(nil)
    let cancelled = Mutex(false)
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { output in
            continuation.withLock { $0 = output }
            started.continuation.yield(())
        }
    }
    func cancel() {
        cancelled.withLock { $0 = true }
        continuation.withLock { $0?.finish(throwing: CancellationError()); $0 = nil }
    }
}

extension AppleServerTests {
    @Test func cancellationReachesTheAppleClient() async throws {
        let client = HoldingAppleServerStub()
        let backend = AppleServerInferenceBackend(client: client)
        let request = ValidatedChatRequest(messages: [.init(role: .user, content: "Hello")],
            tools: [], stream: true, includeUsage: false, generationConfig: .init(),
            maximumCompletionTokens: 32)
        let generation = Task { try await backend.generate(request) { _ in } }
        for await _ in client.started.stream { break }
        generation.cancel()
        await #expect(throws: CancellationError.self) { try await generation.value }
        #expect(client.cancelled.withLock { $0 })
    }
}

extension AppleServerTests {
    @Test func serverArgumentsAcceptAppleDefaults() throws {
        for name in ["apple", "apple.on-device"] {
            let arguments = try RouterServerArguments.parse(["--default-model", name])
            #expect(arguments.defaultModel == name)
        }
    }
}

extension AppleServerTests {
    @Test func httpPreservesUnknownUsageForBothResponseModes() async throws {
        let server = TUFFHTTPServer(modelID: "apple.on-device", queueLimit: 1,
            backend: AppleServerInferenceBackend(client: AppleServerStub()), chatDialect: .chatml)
        let channel = try await server.start(port: 0)
        do {
            let port = try #require(channel.localAddress?.port)
            for stream in [false, true] {
                var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: [
                    "model": "apple.on-device", "messages": [["role": "user", "content": "Hi"]],
                    "stream": stream, "stream_options": ["include_usage": true]])
                let (data, response) = try await URLSession.shared.data(for: request)
                #expect((response as? HTTPURLResponse)?.statusCode == 200)
                let usage: [String: Any]
                if stream {
                    let events: [[String: Any]] = try String(decoding: data, as: UTF8.self)
                        .components(separatedBy: "\n").filter { $0.hasPrefix("data: {") }.map {
                            try #require(JSONSerialization.jsonObject(with: Data($0.dropFirst(6).utf8)) as? [String: Any])
                        }
                    usage = try #require(events.compactMap { $0["usage"] as? [String: Any] }.last)
                } else {
                    let body = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
                    usage = try #require(body["usage"] as? [String: Any])
                }
                #expect(usage["prompt_tokens"] is NSNull)
                #expect(usage["completion_tokens"] is NSNull)
                #expect(usage["total_tokens"] is NSNull)
                #expect(usage["token_counts_available"] as? Bool == false)
            }
            try await server.shutdown()
        } catch {
            try? await server.shutdown()
            throw error
        }
    }
}
