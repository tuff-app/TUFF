import Foundation
import Synchronization
import Testing
import TUFFEngine
import TUFFAppCore
import TUFFDecodeProtocol
import TUFFEngineAdapter
import TUFFModelCatalog
@testable import TUFFDecodeServiceCore

@Suite(.serialized, .timeLimit(.minutes(3))) struct DecodeServiceConcurrentTests {
    private final class Fake: DecodeServiceInference, @unchecked Sendable {
        typealias Continuation = AsyncThrowingStream<AppInferenceEvent, Error>.Continuation
        private let active = Mutex<[UUID: Continuation]>([:])
        var currentVisionTowerBytes: UInt64? { nil }
        func ensureLoaded(modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions,
                          forceLogitsHead: Bool, onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {}
        func unload() async {}
        func waitUntilIdle() async {}
        func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
            AsyncThrowingStream { $0.finish() }
        }
        func generate(_ request: AppGenerationRequest, sessionID: UUID,
                      requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error> {
            AsyncThrowingStream { continuation in
                active.withLock { $0[requestID] = continuation }
                continuation.yield(.token(AppTokenEvent(index: 0, textDelta: request.prompt,
                                                         elapsedDecodeSeconds: 0.1)))
                continuation.yield(.prefillProgress(done: 1, total: 1))
            }
        }
        func cancel() { cancelAllRequests() }
        func cancel(requestID: UUID) { finish(requestID, reason: .cancelled) }
        func cancelAllRequests() {
            let ids = active.withLock { Array($0.keys) }
            for id in ids { cancel(requestID: id) }
        }
        func endSession(_ sessionID: UUID) async {}
        func finish(_ id: UUID, reason: AppStopReason) {
            guard let continuation = active.withLock({ $0.removeValue(forKey: id) }) else { return }
            let diagnostics = AppDiagnostics(generatedTokens: 1, stopReason: reason,
                timeToFirstTokenSeconds: nil, decodeSeconds: 0, tokensPerSecond: 0,
                peakMemoryBytes: nil, runtimeOptions: AppRuntimeOptions())
            continuation.yield(reason == .cancelled ? .cancelled(diagnostics) : .finished(diagnostics))
            continuation.finish()
        }
    }

    private final class Events: @unchecked Sendable {
        private let condition = NSCondition()
        private var events: [DecodeServiceEvent] = []
        init(_ handle: FileHandle) {
            Thread { [self] in
                while let event = try? DecodeFrameCodec.read(DecodeServiceEvent.self, from: handle) {
                    condition.lock(); events.append(event); condition.broadcast(); condition.unlock()
                }
            }.start()
        }
        func wait(_ predicate: @escaping @Sendable ([DecodeServiceEvent]) -> Bool) async -> Bool {
            await withCheckedContinuation { continuation in
                DispatchQueue.global().async { [self] in
                    condition.lock()
                    let deadline = Date().addingTimeInterval(90)
                    while !predicate(events), condition.wait(until: deadline) {}
                    let result = predicate(events)
                    condition.unlock()
                    continuation.resume(returning: result)
                }
            }
        }
        func text(for id: UUID) -> String {
            condition.lock(); defer { condition.unlock() }
            return events.filter { $0.generationID == id }.compactMap(\.textDelta).joined()
        }
        func has(_ kind: DecodeServiceEventKind, id: UUID) -> Bool {
            condition.lock(); defer { condition.unlock() }
            return events.contains { $0.generationID == id && $0.kind == kind }
        }
    }

    private final class Real: DecodeServiceInference, @unchecked Sendable {
        let client: RealInferenceClient
        private let loadCount = Mutex(0)
        init(directory: URL) {
            let coordinator = AppResidencyCoordinator(
                registry: TUFFResidencyRegistry(directory: directory),
                budgetBytes: 12 * 1_073_741_824, reclaim: { _ in false })
            client = RealInferenceClient(residencyCoordinator: coordinator)
        }
        var loads: Int { loadCount.withLock { $0 } }
        var currentVisionTowerBytes: UInt64? { client.currentVisionTowerBytes }
        func ensureLoaded(modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions,
                          forceLogitsHead: Bool, onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
            loadCount.withLock { $0 += 1 }
            try await client.ensureLoaded(modelDirectory: modelDirectory, maxContextTokens: maxContextTokens,
                                          options: options, forceLogitsHead: forceLogitsHead, onState: onState)
        }
        func unload() async { await client.unload() }
        func waitUntilIdle() async { await client.waitUntilIdle() }
        func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
            client.generate(request)
        }
        func generate(_ request: AppGenerationRequest, sessionID: UUID,
                      requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error> {
            client.generate(request, sessionID: sessionID, requestID: requestID)
        }
        func cancel() { client.cancel() }
        func cancel(requestID: UUID) { client.cancel(requestID: requestID) }
        func cancelAllRequests() { client.cancelAllRequests() }
        func endSession(_ sessionID: UUID) async { await client.endSession(sessionID) }
    }

    @Test func twoStreamsProgressAndCancelIndependently() async throws {
        let input = Pipe(), output = Pipe()
        DecodeUnixSocket.disableSIGPIPE(on: output.fileHandleForWriting.fileDescriptor)
        let fake = Fake()
        let service = DecodeService(client: fake, input: input.fileHandleForReading,
                                    output: output.fileHandleForWriting)
        let events = Events(output.fileHandleForReading)
        let run = Task { await service.run() }
        func send(_ command: DecodeServiceCommand) throws {
            try input.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(command))
        }
        try send(.load(DecodeLoadRequest(modelPath: "/tmp/fake", maxContextTokens: 1024)))
        #expect(await events.wait { $0.contains { $0.kind == .ready } })
        let first = UUID(), second = UUID()
        let firstSession = UUID(), secondSession = UUID()
        let a = DecodeGenerationRequest(prompt: "first", maxNewTokens: 4, maxContextTokens: 1024,
                                        temperature: 0, generationID: first)
        let b = DecodeGenerationRequest(prompt: "second", maxNewTokens: 4, maxContextTokens: 1024,
                                        temperature: 0, generationID: second)
        try send(.generateSession(sessionID: firstSession, request: a))
        try send(.generateSession(sessionID: secondSession, request: b))
        #expect(await events.wait { events in
            events.contains { $0.generationID == first && $0.textDelta == "first" }
                && events.contains { $0.generationID == second && $0.textDelta == "second" }
        })
        try send(.cancelRequest(first))
        #expect(await events.wait { $0.contains { $0.generationID == first && $0.kind == .cancelled } })
        let nextID = UUID()
        let next = DecodeGenerationRequest(prompt: "next", maxNewTokens: 4, maxContextTokens: 1024,
                                           temperature: 0, generationID: nextID)
        try send(.generateSession(sessionID: firstSession, request: next))
        #expect(await events.wait { events in
            events.contains { $0.generationID == nextID && $0.textDelta == "next" }
                || events.contains { $0.generationID == nextID && $0.kind == .failed }
        })
        #expect(!events.has(.failed, id: nextID))
        fake.finish(nextID, reason: .eos)
        #expect(await events.wait { $0.contains { $0.generationID == nextID && $0.kind == .finished } })
        fake.finish(second, reason: .eos)
        #expect(await events.wait { $0.contains { $0.generationID == second && $0.kind == .finished } })
        try send(.shutdown)
        _ = await run.value
        try? input.fileHandleForWriting.close()
        try? output.fileHandleForWriting.close()
    }

    @Test func realSharedModelSessionsWhenExplicitlyRequested() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["TUFF_SESSION_TEXT_MODEL"]
            ?? environment["TUFF_PRISM_2B_OUTPUT"] else { return }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let input = Pipe(), output = Pipe()
        DecodeUnixSocket.disableSIGPIPE(on: output.fileHandleForWriting.fileDescriptor)
        let real = Real(directory: root)
        let service = DecodeService(client: real, input: input.fileHandleForReading,
                                    output: output.fileHandleForWriting)
        let events = Events(output.fileHandleForReading)
        let run = Task { await service.run() }
        func send(_ command: DecodeServiceCommand) throws {
            try input.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(command))
        }
        try send(.load(DecodeLoadRequest(modelPath: path, maxContextTokens: 1024)))
        #expect(await events.wait { $0.contains { $0.kind == .ready || $0.kind == .failed } })
        let solo = UUID(), second = UUID(), first = UUID(), soloSession = UUID()
        let history = [DecodeChatTurn(prompt: "What is 2 plus 2?", response: "4")]
        let baseline = DecodeGenerationRequest(prompt: "What is 10 minus 7?", history: history,
            maxNewTokens: 8, maxContextTokens: 1024, temperature: 0, generationID: solo)
        try send(.generateSession(sessionID: soloSession, request: baseline))
        #expect(await events.wait { $0.contains { $0.generationID == solo && [.finished, .failed].contains($0.kind) } })
        #expect(events.has(.finished, id: solo))
        let expected = events.text(for: solo)
        #expect(!expected.isEmpty)
        let closeID = UUID()
        try send(.endSession(sessionID: soloSession, requestID: closeID))
        #expect(await events.wait { $0.contains { $0.generationID == closeID && [.unloaded, .failed].contains($0.kind) } })
        #expect(events.has(.unloaded, id: closeID))
        var concurrent = baseline
        concurrent.generationID = second
        let cancelled = DecodeGenerationRequest(prompt: "List the numbers from 1 to 100.",
            history: [DecodeChatTurn(prompt: "Say hello", response: "Hello")],
            maxNewTokens: 128, maxContextTokens: 1024, temperature: 0, generationID: first)
        let firstSession = UUID()
        try send(.generateSession(sessionID: firstSession, request: cancelled))
        try send(.generateSession(sessionID: UUID(), request: concurrent))
        #expect(await events.wait { events in
            events.contains { $0.generationID == first && $0.kind == .prefill }
                && events.contains { $0.generationID == second && $0.kind == .prefill }
                && !events.contains { ($0.generationID == first || $0.generationID == second)
                    && [.finished, .cancelled, .failed].contains($0.kind) }
        })
        try send(.cancelRequest(first))
        #expect(await events.wait { $0.contains { $0.generationID == first && [.cancelled, .failed].contains($0.kind) } })
        #expect(events.has(.cancelled, id: first))
        let nextID = UUID()
        let next = DecodeGenerationRequest(prompt: "What is 3 plus 3?", maxNewTokens: 4,
            maxContextTokens: 1024, temperature: 0, generationID: nextID)
        try send(.generateSession(sessionID: firstSession, request: next))
        #expect(await events.wait { $0.contains { $0.generationID == nextID && [.finished, .failed].contains($0.kind) } })
        #expect(events.has(.finished, id: nextID))
        #expect(await events.wait { $0.contains { $0.generationID == second && [.finished, .failed].contains($0.kind) } })
        #expect(events.has(.finished, id: second))
        #expect(events.text(for: second) == expected)
        #expect(real.loads == 1)
        try send(.shutdown)
        _ = await run.value
        try? input.fileHandleForWriting.close()
        try? output.fileHandleForWriting.close()
    }

    @Test func sharedOwnerPreservesImagePrefillWhenExplicitlyRequested() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["TUFF_SESSION_VISION_MODEL"],
              let imagePath = environment["TUFF_SESSION_VISION_IMAGE"] else { return }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AppImageAttachmentStore()
        defer { store.removeAll() }
        let attachment = try store.stage(URL(fileURLWithPath: imagePath))
        let image = DecodeImageAttachment(id: attachment.id, path: attachment.fileURL.path,
            displayName: attachment.displayName, encodedBytes: attachment.encodedBytes, sha256: attachment.sha256)
        let input = Pipe(), output = Pipe()
        DecodeUnixSocket.disableSIGPIPE(on: output.fileHandleForWriting.fileDescriptor)
        let real = Real(directory: root)
        let service = DecodeService(client: real, input: input.fileHandleForReading,
                                    output: output.fileHandleForWriting)
        let events = Events(output.fileHandleForReading)
        let run = Task { await service.run() }
        func send(_ command: DecodeServiceCommand) throws {
            try input.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(command))
        }
        try send(.load(DecodeLoadRequest(modelPath: path, maxContextTokens: 1024)))
        #expect(await events.wait { $0.contains { [.ready, .failed].contains($0.kind) } })
        let solo = UUID(), shared = UUID()
        let request = DecodeGenerationRequest(prompt: "Describe the image briefly.",
            imageAttachments: [image], maxNewTokens: 8, maxContextTokens: 1024,
            temperature: 0, generationID: solo)
        try send(.generate(request))
        #expect(await events.wait { $0.contains { $0.generationID == solo && [.finished, .failed].contains($0.kind) } })
        #expect(events.has(.finished, id: solo))
        let expected = events.text(for: solo)
        #expect(!expected.isEmpty)
        var peerRequest = request
        peerRequest.generationID = shared
        try send(.generateSession(sessionID: UUID(), request: peerRequest))
        #expect(await events.wait { $0.contains { $0.generationID == shared && [.finished, .failed].contains($0.kind) } })
        #expect(events.has(.finished, id: shared))
        #expect(events.text(for: shared) == expected)
        #expect(real.loads == 1)
        try send(.shutdown)
        _ = await run.value
        try? input.fileHandleForWriting.close()
        try? output.fileHandleForWriting.close()
    }
}
