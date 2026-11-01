import Foundation
import Synchronization

public protocol AppInferenceClient: Sendable {
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error>
    func cancel()
}

/// A client that can run several independent chat sessions against the same
/// loaded model. Each scoped client permits one active writer for its session
/// and cancels only its own generation request.
public protocol AppSessionInferenceClient: AppInferenceClient {
    func generate(_ request: AppGenerationRequest, sessionID: UUID,
                  requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error>
    func cancel(requestID: UUID)
    func endSession(_ sessionID: UUID, requestID: UUID) async throws
}

public extension AppSessionInferenceClient {
    func scopedSession(_ sessionID: UUID) -> any AppInferenceClient {
        AppScopedInferenceClient(client: self, sessionID: sessionID)
    }
}

private final class AppScopedInferenceClient: AppInferenceClient, @unchecked Sendable {
    private struct State { var activeRequestID: UUID? }
    private let state = Mutex(State())
    private let client: any AppSessionInferenceClient
    private let sessionID: UUID

    init(client: any AppSessionInferenceClient, sessionID: UUID) {
        self.client = client
        self.sessionID = sessionID
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        let requestID = UUID()
        let reserved = state.withLock { current -> Bool in
            guard current.activeRequestID == nil else { return false }
            current.activeRequestID = requestID
            return true
        }
        return AsyncThrowingStream { continuation in
            guard reserved else {
                let error = AppInferenceError.generationInFlight
                continuation.yield(.failed(error, partial: nil))
                continuation.finish(throwing: error)
                return
            }
            let stream = client.generate(request, sessionID: sessionID, requestID: requestID)
            let task = Task {
                var failure: (any Error)?
                do {
                    for try await event in stream {
                        continuation.yield(event)
                    }
                } catch {
                    failure = error
                }
                // Release the per-chat writer before waking the consumer from
                // stream completion, so an immediately submitted next turn
                // observes the terminal request as drained.
                state.withLock {
                    if $0.activeRequestID == requestID { $0.activeRequestID = nil }
                }
                if let failure { continuation.finish(throwing: failure) }
                else { continuation.finish() }
            }
            continuation.onTermination = { [client] termination in
                guard case .cancelled = termination else { return }
                // Keep draining until the helper reports its terminal event;
                // the session remains occupied until the producer has stopped.
                client.cancel(requestID: requestID)
            }
        }
    }

    func cancel() {
        guard let requestID = state.withLock({ $0.activeRequestID }) else { return }
        client.cancel(requestID: requestID)
    }
}

public protocol AppInferenceEngineReporting: Sendable {
    var currentEngineIdentity: String? { get }
}

/// A client that owns a loadable model session. Loading is split from
/// generation so the UI can pre-load the ~1.6 GB resident weights once and
/// keep them warm across runs. Generation never loads or replaces a session.
public protocol AppModelLifecycleClient: AnyObject, AppInferenceClient {
    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws
    func unload() async
    /// Releases one model's resident state. A client that serves several
    /// models at once must not touch the others; a single-model client has
    /// only one thing to release.
    func unload(modelDirectory: URL) async
}

public extension AppModelLifecycleClient {
    func unload(modelDirectory: URL) async { await unload() }
}

/// A lifecycle client backed by a shared serving pool that keeps several
/// models resident at once and outlives this app's windows.
public protocol AppPooledInferenceClient: AppModelLifecycleClient {
    /// The configuration this client last loaded the model with, when the
    /// pool still holds it for this directory.
    func residentRuntime(forModelDirectory directory: URL)
        -> (maxContextTokens: Int, options: AppRuntimeOptions, forceLogitsHead: Bool)?
}

public protocol AppInferenceMemoryReporting: AnyObject {
    var currentInferenceMemoryBytes: UInt64? { get }
    /// Resident bytes, which include the mapped weights the footprint omits.
    /// Defaulted so a reporter that cannot answer simply does not.
    var currentInferenceResidentBytes: UInt64? { get }
    /// Bytes of image tower the inference process holds mapped, or nil when it
    /// has no vision runtime. The only figure that separates the two image
    /// residency policies: both charge the process the same few MB.
    var currentInferenceTowerBytes: UInt64? { get }
}

extension AppInferenceMemoryReporting {
    public var currentInferenceResidentBytes: UInt64? { nil }
    public var currentInferenceTowerBytes: UInt64? { nil }
}

public protocol AppInferenceTranscriptReporting: AnyObject {
    var generationTranscriptMailbox: GenerationTranscriptMailbox { get }
}
