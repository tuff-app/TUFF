import Foundation
@testable import TUFFAppCore

/// A serving-pool stand-in: several models resident at once, generations that
/// stay open until released, and a record of every lifecycle call.
final class FakePooledInferenceClient: AppPooledInferenceClient, AppSessionInferenceClient,
    @unchecked Sendable {
    typealias Continuation = AsyncThrowingStream<AppInferenceEvent, Error>.Continuation
    private struct Held {
        let directory: URL
        let continuation: Continuation
    }
    private let lock = NSLock()
    private var resident: [String: (Int, AppRuntimeOptions, Bool)] = [:]
    private var held: [UUID: Held] = [:]
    private(set) var targetedUnloads: [URL] = []
    private(set) var globalUnloads = 0
    private(set) var cancelledRequests: [UUID] = []
    private(set) var started: [(directory: URL, sessionID: UUID)] = []

    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions,
                      forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        onState(.loading(.validatingDirectory))
        try await Task.sleep(for: .milliseconds(10))
        lock.withLock { resident[modelDirectory.standardizedFileURL.path] = (maxContextTokens, options, forceLogitsHead) }
        onState(.ready(modelDirectory: modelDirectory.standardizedFileURL, loadSeconds: 0))
    }

    func unload() async { lock.withLock { globalUnloads += 1; resident.removeAll() } }

    func unload(modelDirectory: URL) async {
        lock.withLock {
            targetedUnloads.append(modelDirectory.standardizedFileURL)
            resident.removeValue(forKey: modelDirectory.standardizedFileURL.path)
        }
    }

    func residentRuntime(forModelDirectory directory: URL)
        -> (maxContextTokens: Int, options: AppRuntimeOptions, forceLogitsHead: Bool)? {
        lock.withLock {
            resident[directory.standardizedFileURL.path].map { ($0.0, $0.1, $0.2) }
        }
    }

    func isResident(_ directory: URL) -> Bool {
        lock.withLock { resident[directory.standardizedFileURL.path] != nil }
    }

    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        generate(request, sessionID: UUID(), requestID: UUID())
    }

    func generate(_ request: AppGenerationRequest, sessionID: UUID,
                  requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            lock.withLock {
                held[requestID] = Held(directory: request.modelDirectory.standardizedFileURL,
                                       continuation: continuation)
                started.append((request.modelDirectory.standardizedFileURL, sessionID))
            }
            continuation.yield(.prefillProgress(done: 1, total: 2))
        }
    }

    var activeCount: Int { lock.withLock { held.count } }

    func activeCount(for directory: URL) -> Int {
        lock.withLock { held.values.filter { $0.directory == directory.standardizedFileURL }.count }
    }

    /// Finishes every open generation on `directory` with an answer.
    func release(directory: URL, text: String) {
        let finishing = lock.withLock { () -> [Continuation] in
            let ids = held.filter { $0.value.directory == directory.standardizedFileURL }.map(\.key)
            return ids.compactMap { held.removeValue(forKey: $0)?.continuation }
        }
        for continuation in finishing {
            continuation.yield(.token(AppTokenEvent(index: 0, textDelta: text, elapsedDecodeSeconds: 0.1)))
            continuation.yield(.finished(AppDiagnostics(
                generatedTokens: 1, stopReason: .eos, promptTokenCount: 1, prefillSeconds: nil,
                timeToFirstTokenSeconds: nil, decodeSeconds: 0.1, tokensPerSecond: 10,
                peakMemoryBytes: nil, runtimeOptions: AppRuntimeOptions())))
            continuation.finish()
        }
    }

    func cancel() {
        let all = lock.withLock { () -> [Continuation] in
            defer { held.removeAll() }
            return held.values.map(\.continuation)
        }
        for continuation in all { continuation.finish() }
    }

    func cancel(requestID: UUID) {
        let target = lock.withLock { () -> Continuation? in
            cancelledRequests.append(requestID)
            return held.removeValue(forKey: requestID)?.continuation
        }
        target?.yield(.cancelled(AppDiagnostics(
            generatedTokens: 0, stopReason: .cancelled, promptTokenCount: 1, prefillSeconds: nil,
            timeToFirstTokenSeconds: nil, decodeSeconds: 0, tokensPerSecond: 0,
            peakMemoryBytes: nil, runtimeOptions: AppRuntimeOptions())))
        target?.finish()
    }

    func endSession(_ sessionID: UUID, requestID: UUID) async throws {}
}
