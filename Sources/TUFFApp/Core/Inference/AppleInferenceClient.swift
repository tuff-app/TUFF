import Foundation
import Synchronization
#if canImport(FoundationModels)
import FoundationModels
#endif

/// Text-only adapter. Each request reconstructs the typed transcript from TUFF's
/// durable turns; cancellation can therefore never poison the next conversation.
public final class AppleInferenceClient: AppInferenceClient, Sendable {
    private let task = Mutex<Task<Void, Never>?>(nil)
    private let kind: AppleModelKind
    private let availability: @Sendable () -> AppleModelAvailability
    public init(kind: AppleModelKind = .onDevice) {
        self.kind = kind
        availability = { AppleModelAvailability.current(for: kind) }
    }
    init(availability: @escaping @Sendable () -> AppleModelAvailability) {
        self.kind = .onDevice
        self.availability = availability
    }

    static func snapshotDelta(previous: String, current: String) throws -> String {
        guard current.utf8.starts(with: previous.utf8) else {
            throw AppInferenceError.invalidRequest("Apple revised an already streamed response.")
        }
        return String(decoding: current.utf8.dropFirst(previous.utf8.count), as: UTF8.self)
    }

    public func cancel() { task.withLock { $0?.cancel() } }

    public func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            task.withLock { active in
                guard active == nil else {
                    continuation.finish(throwing: AppInferenceError.generationInFlight)
                    return
                }
                active = Task {
                    do {
                        let state = self.availability()
                        guard state.canGenerate else {
                            throw AppInferenceError.invalidRequest(state.label)
                        }
                        guard request.imageAttachments.isEmpty, request.tools.isEmpty,
                              request.currentRounds.isEmpty, request.assistantPrefix.isEmpty,
                              request.history.allSatisfy({ $0.images.isEmpty && $0.toolRounds.isEmpty }) else {
                            throw AppInferenceError.invalidRequest("Apple supports text chat here; images, tools and Continue are unavailable.")
                        }
                        #if canImport(FoundationModels) && os(macOS)
                        if #available(macOS 26, *) {
                            try await Self.respond(request, kind: self.kind, continuation: continuation)
                        }
                        #endif
                        self.task.withLock { $0 = nil }
                        continuation.finish()
                    } catch {
                        self.task.withLock { $0 = nil }
                        continuation.finish(throwing: Task.isCancelled || error is CancellationError ? AppInferenceError.cancelled : error)
                    }
                }
            }
            continuation.onTermination = { [weak self] termination in
                if case .cancelled = termination { self?.cancel() }
            }
        }
    }

    #if canImport(FoundationModels) && os(macOS)
    @available(macOS 26, *)
    private static func respond(_ request: AppGenerationRequest, kind: AppleModelKind,
                                continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async throws {
        var entries: [Transcript.Entry] = []
        if !request.systemPrompt.isEmpty {
            entries.append(.instructions(.init(segments: [.text(.init(content: request.systemPrompt))], toolDefinitions: [])))
        }
        for turn in request.history {
            entries.append(.prompt(.init(segments: [.text(.init(content: turn.modelPrompt))])))
            entries.append(.response(.init(assetIDs: [], segments: [.text(.init(content: turn.response))])))
        }
        let session: LanguageModelSession
        if kind == .privateCloudCompute {
            #if compiler(>=6.4)
            if #available(macOS 27, *) {
                session = LanguageModelSession(model: PrivateCloudComputeLanguageModel(), transcript: Transcript(entries: entries))
            } else { throw AppInferenceError.invalidRequest("Private Cloud Compute requires macOS 27.") }
            #else
            throw AppInferenceError.invalidRequest("Private Cloud Compute is unavailable in this build.")
            #endif
        } else {
            session = LanguageModelSession(transcript: Transcript(entries: entries))
        }
        let start = ContinuousClock.now
        var previous = ""
        var snapshotIndex = 0
        let options = GenerationOptions(sampling: request.temperature == 0 ? .greedy : nil,
                                        temperature: Double(request.temperature),
                                        maximumResponseTokens: request.maxNewTokens)
        for try await snapshot in session.streamResponse(to: request.prompt, options: options) {
            try Task.checkCancellation()
            // Foundation Models emits cumulative snapshots. Refuse a revised
            // prefix rather than duplicating or silently corrupting streamed text.
            let delta = try snapshotDelta(previous: previous, current: snapshot.content)
            previous = snapshot.content
            if !delta.isEmpty {
                let duration = start.duration(to: .now)
                let seconds = Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
                // Index identifies streamed deltas, not Apple's unexposed token IDs.
                continuation.yield(.token(.init(index: snapshotIndex, textDelta: delta, elapsedDecodeSeconds: seconds)))
                snapshotIndex += 1
            }
        }
        try Task.checkCancellation()
        let elapsed = start.duration(to: .now)
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        // Apple's public stream does not expose token IDs/counts or engine memory.
        continuation.yield(.finished(.init(generatedTokens: 0, stopReason: .endOfTurn,
            timeToFirstTokenSeconds: nil, decodeSeconds: seconds, tokensPerSecond: 0,
            peakMemoryBytes: nil, runtimeOptions: request.runtimeOptions)))
    }
    #endif
}
