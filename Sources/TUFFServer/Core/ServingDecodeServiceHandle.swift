import Foundation
import TUFFAppCore
import TUFFDecodeProtocol
import TUFFModelCatalog
import TUFFRuntimeDescriptions

/// Resolves only installed catalog or registered custom models. IPC callers
/// send a model identifier, never a filesystem path.
public struct ServingModelHandleResolver: Sendable {
    private let installed: ServerInstalledModels
    private let helperExecutable: URL?

    /// - Parameter helperExecutable: A decode-service executable to launch
    ///   directly instead of acquiring the installed engine package. For
    ///   local builds and tests only.
    public init(installed: ServerInstalledModels, helperExecutable: URL? = nil) {
        self.installed = installed
        self.helperExecutable = helperExecutable
    }

    public func resolve(modelID: String) async throws -> any ServingModelHandle {
        guard let descriptor = installed.descriptor(named: modelID),
              installed.isInstalled(descriptor), installed.isEligible(descriptor) else {
            throw ServerRequestError.modelUnavailable("The requested model is not installed or cannot run on this Mac.")
        }
        return ServingDecodeServiceHandle(descriptor: descriptor,
            directory: installed.directory(for: descriptor), helperExecutable: helperExecutable)
    }
}

/// One family-scoped helper and its one shared model handle. Every chat/API
/// client for this model is multiplexed through the helper's session protocol.
private final class ServingDecodeServiceHandle: ServingModelHandle, @unchecked Sendable {
    private let descriptor: TUFFModelDescriptor
    private let directory: URL
    private let client: DecodeServiceInferenceClient
    private let lock = NSLock()
    private var preparation: ServingModelPreparation?

    init(descriptor: TUFFModelDescriptor, directory: URL, helperExecutable: URL?) {
        self.client = DecodeServiceInferenceClient(serviceURL: helperExecutable)
        self.descriptor = descriptor
        self.directory = directory.standardizedFileURL
    }

    func prepare(_ request: ServingModelPreparation,
                 progress: @escaping @Sendable (ServingLoadState) -> Void) async throws {
        guard request.modelID == descriptor.apiModelID else {
            throw ServerRequestError.unknownModel
        }
        let options = try Self.appOptions(request.runtimeOptions)
        try await client.ensureLoaded(modelDirectory: directory,
            maxContextTokens: request.maxContextTokens, options: options,
            forceLogitsHead: request.forceLogitsHead) { state in
                switch state {
                case .loading(let phase):
                    let mapped: ServingLoadPhase
                    switch phase {
                    case .preparingEngine: mapped = .preparingEngine
                    case .validatingDirectory: mapped = .validatingDirectory
                    case .tokenizer: mapped = .tokenizer
                    case .verifyingWeights: mapped = .verifyingWeights
                    case .preparingRunner: mapped = .preparingRunner
                    case .mappingImageTower: mapped = .mappingImageTower
                    case .calibratingAuto: return
                    }
                    progress(.loading(mapped))
                case .cancelling: progress(.cancelling)
                case .unloading: progress(.unloading)
                case .ready(_, let seconds): progress(.ready(loadSeconds: seconds))
                case .failed(let error): progress(.failed(error.localizedDescription))
                case .notLoaded: break
                }
            }
        lock.withLock { preparation = request }
    }

    func start(_ request: DecodeGenerationRequest, sessionID: UUID)
        -> AsyncThrowingStream<DecodeServiceEvent, Error> {
        let prepared = lock.withLock { preparation }
        guard prepared != nil else {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: AppInferenceError.modelNotLoaded)
            }
        }
        let stream = client.startRaw(request, sessionID: sessionID)
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    for try await event in stream { continuation.yield(event) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: ServingHelperTransportLost("\(error)"))
                }
            }
        }
    }

    func cancel(requestID: UUID) { client.cancel(requestID: requestID) }
    func cancelWarmup() { client.cancelWarmup() }

    func endSession(_ sessionID: UUID, requestID: UUID) async throws {
        try await client.endSession(sessionID, requestID: requestID)
    }

    func unload() async {
        await client.unload()
        lock.withLock { preparation = nil }
    }

    private static func appOptions(_ options: DecodeRuntimeOptions) throws -> AppRuntimeOptions {
        guard let cachePolicy = AppExpertCachePolicy(rawValue: options.expertCachePolicy),
              let rdadvisePolicy = AppRDAdvicePolicy(rawValue: options.rdadvisePolicy),
              let verification = AppModelVerification(rawValue: options.modelVerification),
              let visionPolicy = VisionResidencyPolicy(
                rawValue: options.visionResidencyPolicy ?? VisionResidencyPolicy.onDemand.rawValue) else {
            throw AppInferenceError.invalidRequest("Unsupported model runtime options")
        }
        let densePolicy: DenseResidencyPolicy = options.denseWeightBudgetBytes.map {
            .automatic(weightBudgetBytes: $0)
        } ?? options.denseResidentLayers.map { .bounded(residentLayers: Set($0)) } ?? .resident
        let result = AppRuntimeOptions(expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: cachePolicy, prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens, rdadvisePolicy: rdadvisePolicy,
            modelVerification: verification, visionResidencyPolicy: visionPolicy,
            denseResidencyPolicy: densePolicy,
            retainedConversationBytes: options.retainedConversationBytes)
        try result.validate()
        return result
    }
}
