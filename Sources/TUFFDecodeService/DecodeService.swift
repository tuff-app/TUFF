import TUFFEngineAdapter
import Foundation
import TUFFEngine
import TUFFAppCore
import TUFFDecodeProtocol
import Synchronization

/// The inference operations the decode service drives. `RealInferenceClient`
/// is the production client; tests substitute a fake.
protocol DecodeServiceInference: AppModelLifecycleClient {
    var currentVisionTowerBytes: UInt64? { get }
    func waitUntilIdle() async
    func waitUntilRequestIdle(_ requestID: UUID) async
    func generate(_ request: AppGenerationRequest, sessionID: UUID,
                  requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error>
    func cancel(requestID: UUID)
    func cancelAllRequests()
    func cancelWarmup()
    func endSession(_ sessionID: UUID) async
}

extension RealInferenceClient: DecodeServiceInference {}
extension DecodeServiceInference {
    func generate(_ request: AppGenerationRequest, sessionID: UUID,
                  requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: AppInferenceError.modelNotLoaded) }
    }
    func cancel(requestID: UUID) { cancel() }
    func waitUntilRequestIdle(_ requestID: UUID) async {}
    func cancelAllRequests() { cancel() }
    func cancelWarmup() {}
    func endSession(_ sessionID: UUID) async {}
}

/// The decode service's command loop over one app transport. Commands are
/// read on their own thread so a cancel reaches the client while a generation
/// runs. If either direction of the transport fails, the service cancels its
/// work, releases the session and returns.
final class DecodeService: @unchecked Sendable {
    private let requiresHandshake: Bool
    private let capabilities = DecodeEngineCapabilities(
        architectureIDs: ArchConfig.registeredArchitectures.keys.map(\.rawValue).sorted())
    private let client: any DecodeServiceInference
    private let input: FileHandle
    private let output: FileHandle
    private let commands = DecodeCommandQueue()
    private let frameWriter: DecodeServiceFrameWriter
    private struct ActiveRequest: Sendable {
        let sessionID: UUID
        var task: Task<Void, Never>?
        var drained = false
    }
    private let activeRequests = Mutex<[UUID: ActiveRequest]>([:])
    let lifecycle: DecodeTransportLifecycle
    /// Test hook that runs after a generation command is accepted and before
    /// its inference starts.
    private let beforeGenerationStarts: (@Sendable (DecodeTransportLifecycle) -> Void)?

    init(client: any DecodeServiceInference,
         input: FileHandle,
         output: FileHandle,
         requiresHandshake: Bool = false,
         beforeGenerationStarts: (@Sendable (DecodeTransportLifecycle) -> Void)? = nil) {
        self.client = client
        self.requiresHandshake = requiresHandshake
        self.input = input
        self.output = output
        self.frameWriter = DecodeServiceFrameWriter(handle: output)
        self.beforeGenerationStarts = beforeGenerationStarts
        lifecycle = DecodeTransportLifecycle(
            commands: commands,
            cancelInference: { [client] in client.cancelAllRequests() })
    }

    /// Runs until an orderly shutdown (nil) or a transport loss (the loss).
    /// Either way the loaded session is released before this returns.
    func run() async -> DecodeTransportLoss? {
        startInputThread()
        var modelDirectory: URL?
        var loadedOptions: DecodeRuntimeOptions?
        var negotiatedArchitecture: String?
        commandLoop: while let command = await nextCommand() {
            guard lifecycle.lostTransport == nil else { break }
            switch command {
            case .hello(let request):
                if capabilities.accepts(request) {
                    negotiatedArchitecture = request.architectureID
                    var reply = DecodeServiceEvent(kind: .hello, generationID: request.requestID)
                    reply.capabilities = capabilities
                    send(reply)
                } else {
                    negotiatedArchitecture = nil
                    send(DecodeServiceEvent(kind: .failed, generationID: request.requestID,
                        error: "engine protocol, architecture or model format is incompatible"))
                }
            case .load(let request):
                let directory = URL(fileURLWithPath: request.modelPath)
                do {
                    if requiresHandshake {
                        guard let negotiatedArchitecture else {
                            throw AppInferenceError.modelLoadFailed("engine handshake is required before loading a model")
                        }
                        let architecture = try ManifestReader.resolveArchitecture(directoryURL: directory)
                        guard negotiatedArchitecture == architecture.variant.rawValue else {
                            throw AppInferenceError.modelLoadFailed("engine handshake is required before loading this architecture")
                        }
                    }
                    let options = try Self.appRuntimeOptions(request.runtimeOptions)
                    try await client.ensureLoaded(
                        modelDirectory: directory,
                        maxContextTokens: request.maxContextTokens,
                        options: options,
                        forceLogitsHead: request.forceLogitsHead) { _ in }
                    modelDirectory = directory
                    loadedOptions = request.runtimeOptions
                    let memory = AppMemorySampler().sample()
                    send(DecodeServiceEvent(
                        kind: .ready, generationID: request.requestID,
                        currentMemoryBytes: memory, peakMemoryBytes: memory))
                } catch {
                    send(DecodeServiceEvent(
                        kind: .failed, generationID: request.requestID,
                        error: "\(error)"))
                }
            case .generate(let request):
                guard activeRequests.withLock({ $0.isEmpty }) else {
                    send(DecodeServiceEvent(kind: .failed, generationID: request.generationID,
                                            error: "concurrent sessions are active"))
                    continue
                }
                guard let modelDirectory else {
                    send(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "model is not loaded"))
                    continue
                }
                // Prefill is chosen per request, not at load, so it must not be
                // part of this comparison: toggling it and pressing Generate was
                // refused as a mismatched session.
                var comparable = request.runtimeOptions
                comparable.prefillEnabled = loadedOptions?.prefillEnabled
                    ?? comparable.prefillEnabled
                comparable.prefillChunkTokens = loadedOptions?.prefillChunkTokens
                    ?? comparable.prefillChunkTokens
                guard comparable == loadedOptions else {
                    send(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "generation runtime options do not match the loaded session"))
                    continue
                }
                await generate(request, modelDirectory: modelDirectory)
            case .generateSession(let sessionID, let request):
                guard let modelDirectory, let loadedOptions else {
                    send(DecodeServiceEvent(kind: .failed, generationID: request.generationID,
                                            error: "model is not loaded"))
                    continue
                }
                var comparable = request.runtimeOptions
                comparable.prefillEnabled = loadedOptions.prefillEnabled
                comparable.prefillChunkTokens = loadedOptions.prefillChunkTokens
                guard comparable == loadedOptions else {
                    send(DecodeServiceEvent(kind: .failed, generationID: request.generationID,
                                            error: "generation runtime options do not match the loaded session"))
                    continue
                }
                startConcurrent(request, sessionID: sessionID, modelDirectory: modelDirectory)
            case .cancelRequest(let requestID):
                client.cancel(requestID: requestID)
            case .cancelWarmup:
                break // Applied by the input thread without waiting for the command loop.
            case .endSession(let sessionID, let requestID):
                let active = activeRequests.withLock { requests in
                    requests.filter { $0.value.sessionID == sessionID }
                        .map { ($0.key, $0.value.task) }
                }
                for (id, _) in active { client.cancel(requestID: id) }
                for (_, task) in active { await task?.value }
                await client.endSession(sessionID)
                send(DecodeServiceEvent(kind: .unloaded, generationID: requestID))
            case .cancel:
                break
            case .unload(let requestID):
                client.cancelAllRequests()
                await client.waitUntilIdle()
                await client.unload()
                modelDirectory = nil
                loadedOptions = nil
                send(DecodeServiceEvent(kind: .unloaded, generationID: requestID))
            case .shutdown:
                break commandLoop
            }
        }
        client.cancelAllRequests()
        await client.waitUntilIdle()
        let remaining = activeRequests.withLock { $0.values.compactMap(\.task) }
        for task in remaining { await task.value }
        await client.unload()
        return lifecycle.lostTransport
    }

    private func startInputThread() {
        let thread = Thread { [input, client, commands, lifecycle] in
            do {
                while true {
                    let command = try DecodeFrameCodec.read(
                        DecodeServiceCommand.self, from: input)
                    if case .cancel = command { client.cancel() }
                    if case .cancelRequest(let id) = command { client.cancel(requestID: id) }
                    if case .cancelWarmup = command { client.cancelWarmup() }
                    commands.append(command)
                    if case .shutdown = command { break }
                }
            } catch {
                // End of file or a broken frame: the app is gone or no longer
                // speaking the protocol, so nothing it queued may still run.
                lifecycle.abort(.inputClosed)
            }
        }
        thread.name = "TUFF.DecodeService.Input"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    private func generate(_ request: DecodeGenerationRequest,
                          modelDirectory: URL) async {
        let outbox = DecodeServiceOutbox(
            generationID: request.generationID,
            towerBytes: { [client] in client.currentVisionTowerBytes },
            onOverflow: { [client] in client.cancel() })
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread { [output, lifecycle, frameWriter] in
            defer { writerFinished.signal() }
            do { try outbox.runWriter(to: output, frameWriter: frameWriter) }
            catch {
                FileHandle.standardError.write(Data("IPC writer failed: \(error)\n".utf8))
                lifecycle.abort(.outputFailed)
            }
        }
        writer.name = "TUFF.DecodeService.Writer"
        writer.qualityOfService = .userInitiated
        writer.start()

        beforeGenerationStarts?(lifecycle)
        let inference = Task { [client, lifecycle] in
            // A loss recorded after this command was dequeued must not start
            // inference. One recorded after this check cancels this task
            // through the registration below, which ends the stream and the
            // client's generation with it.
            guard lifecycle.lostTransport == nil else { throw CancellationError() }
            try Task.checkCancellation()
            let generation = try Self.appGenerationRequest(
                request, modelDirectory: modelDirectory)
            for try await event in client.generate(generation) {
                outbox.publish(event)
            }
        }
        lifecycle.attachGeneration { inference.cancel() }
        do {
            try await inference.value
            outbox.finish()
        } catch {
            outbox.finish(error: error)
        }
        // Ending the stream consumer does not acknowledge the independently
        // running producer. Drain it before another command can reuse or
        // unload the session and release its residency lease.
        await client.waitUntilIdle()
        lifecycle.detachGeneration()
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                writerFinished.wait()
                continuation.resume()
            }
        }
    }

    private func startConcurrent(_ request: DecodeGenerationRequest, sessionID: UUID,
                                 modelDirectory: URL) {
        let reserved = activeRequests.withLock { requests -> Bool in
            guard requests[request.generationID] == nil,
                  !requests.values.contains(where: { $0.sessionID == sessionID && !$0.drained }) else { return false }
            requests[request.generationID] = ActiveRequest(sessionID: sessionID)
            return true
        }
        guard reserved else {
            send(DecodeServiceEvent(kind: .failed, generationID: request.generationID,
                                    error: "session or request is already active"))
            return
        }
        do {
            let generation = try Self.appGenerationRequest(request, modelDirectory: modelDirectory)
            let stream = client.generate(generation, sessionID: sessionID, requestID: request.generationID)
            if lifecycle.lostTransport != nil { client.cancel(requestID: request.generationID) }
            let task = Task { [self] in
                let outbox = DecodeServiceOutbox(generationID: request.generationID,
                    towerBytes: { [client] in client.currentVisionTowerBytes },
                    onOverflow: { [client] in client.cancel(requestID: request.generationID) })
                let writerDone = DispatchSemaphore(value: 0)
                let writer = Thread { [self] in
                    defer { writerDone.signal() }
                    do { try outbox.runWriter(to: output, frameWriter: frameWriter) }
                    catch { lifecycle.abort(.outputFailed) }
                }
                writer.start()
                var terminal: AppInferenceEvent?
                var failure: (any Error)?
                do {
                    for try await event in stream {
                        switch event {
                        case .finished, .cancelled, .failed: terminal = event
                        default: outbox.publish(event)
                        }
                    }
                } catch { failure = error }
                // A stream consumer can observe its terminal before the
                // producer has released GPU, cache and session state. Do not
                // acknowledge it until this specific producer is drained.
                await client.waitUntilRequestIdle(request.generationID)
                activeRequests.withLock { $0[request.generationID]?.drained = true }
                if let terminal { outbox.publish(terminal) }
                outbox.finish(error: failure)
                await withCheckedContinuation { continuation in
                    DispatchQueue.global().async { writerDone.wait(); continuation.resume() }
                }
                _ = activeRequests.withLock { $0.removeValue(forKey: request.generationID) }
            }
            activeRequests.withLock { $0[request.generationID]?.task = task }
        } catch {
            _ = activeRequests.withLock { $0.removeValue(forKey: request.generationID) }
            send(DecodeServiceEvent(kind: .failed, generationID: request.generationID, error: "\(error)"))
        }
    }

    private static func appGenerationRequest(
        _ request: DecodeGenerationRequest,
        modelDirectory: URL
    ) throws -> AppGenerationRequest {
        // The trust boundary: these paths arrive over a socket, and
        // this process opens and hashes whatever they name. Without
        // this, a peer could use the service to report on any file
        // the user can read.
        // History carries its own images now, so the same rule has
        // to cover them: a path that reaches this process is opened
        // and hashed whether it arrived as the current message or as
        // an earlier one.
        let everyAttachment = (request.imageAttachments ?? [])
            + request.history.flatMap { $0.images ?? [] }
        if let outside = everyAttachment.first(where: {
            !AppImageAttachmentStore.contains(URL(fileURLWithPath: $0.path))
        }) {
            throw DecodeServiceError.attachmentOutsideStore(
                path: outside.path)
        }
        let options = try appRuntimeOptions(request.runtimeOptions)
        return AppGenerationRequest(
            modelDirectory: modelDirectory, prompt: request.prompt,
            systemPrompt: request.systemPrompt ?? "",
            assistantPrefix: request.assistantPrefix ?? "",
            history: request.history.map { turn in
                AppChatTurn(
                    prompt: turn.prompt,
                    response: turn.response,
                    thinking: turn.thinking,
                    images: (turn.images ?? []).map {
                        AppImageAttachment(
                            id: $0.id,
                            fileURL: URL(fileURLWithPath: $0.path),
                            displayName: $0.displayName,
                            encodedBytes: $0.encodedBytes,
                            sha256: $0.sha256)
                    },
                    toolRounds: (turn.toolRounds ?? []).map(appToolRound))
            },
            imageAttachments: (request.imageAttachments ?? []).map {
                AppImageAttachment(
                    id: $0.id,
                    fileURL: URL(fileURLWithPath: $0.path),
                    displayName: $0.displayName,
                    encodedBytes: $0.encodedBytes,
                    sha256: $0.sha256)
            },
            maxNewTokens: request.maxNewTokens,
            maxContextTokens: request.maxContextTokens,
            reasoning: ChatReasoning(
                rawValue: request.reasoning.rawValue) ?? .off,
            reasoningEffort: request.reasoningEffort.flatMap {
                GPTOSSReasoningEffort(rawValue: $0.rawValue)
            },
            preserveThinking: request.preserveThinking,
            temperature: request.temperature,
            topK: request.topK,
            topP: request.topP,
            repetitionPenalty: request.repetitionPenalty,
            seed: request.seed,
            runtimeOptions: options,
            tools: request.tools ?? [],
            currentRounds: (request.currentRounds ?? []).map(appToolRound),
            conversationKey: request.conversationKey,
            preservesFullHistory: request.preservesFullHistory)
    }

    /// A round as the model reads it. Status and summaries stay in the app;
    /// the service only renders the calls and the result text.
    static func appToolRound(_ round: DecodeToolRound) -> AppToolRound {
        AppToolRound(
            thinking: round.thinking, content: round.content,
            calls: round.calls.map { AppToolCall(id: $0.id, name: $0.name, arguments: $0.arguments) },
            results: round.results.map {
                AppToolResult(callID: $0.callID, name: $0.name, status: .succeeded,
                              modelText: $0.content, summary: "")
            })
    }

    private func nextCommand() async -> DecodeServiceCommand? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [commands] in
                continuation.resume(returning: commands.next())
            }
        }
    }

    /// Writes one event outside a generation. A failed write means the app
    /// can no longer hear the service, which ends it like a closed input.
    private func send(_ event: DecodeServiceEvent) {
        do {
            try frameWriter.write(DecodeFrameCodec.encode(event))
        } catch {
            FileHandle.standardError.write(Data("IPC write failed: \(error)\n".utf8))
            lifecycle.abort(.outputFailed)
        }
    }

    static func appRuntimeOptions(_ options: DecodeRuntimeOptions) throws
        -> AppRuntimeOptions {
        guard let cachePolicy = AppExpertCachePolicy(
            rawValue: options.expertCachePolicy) else {
            throw AppInferenceError.invalidRequest(
                "unknown expert cache policy \(options.expertCachePolicy)")
        }
        guard let rdadvisePolicy = AppRDAdvicePolicy(
            rawValue: options.rdadvisePolicy) else {
            throw AppInferenceError.invalidRequest(
                "unknown RDADVISE policy \(options.rdadvisePolicy)")
        }
        guard let modelVerification = AppModelVerification(
            rawValue: options.modelVerification) else {
            throw AppInferenceError.invalidRequest(
                "unknown model verification \(options.modelVerification)")
        }
        // An unknown policy is a request for behaviour that does not exist;
        // absent means the shipped default.
        guard let visionResidencyPolicy = VisionResidencyPolicy(
            rawValue: options.visionResidencyPolicy ?? VisionResidencyPolicy.onDemand.rawValue)
        else {
            throw AppInferenceError.invalidRequest(
                "unknown vision residency policy \(options.visionResidencyPolicy ?? "")")
        }
        let resolved = AppRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: cachePolicy,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: rdadvisePolicy,
            modelVerification: modelVerification,
            visionResidencyPolicy: visionResidencyPolicy,
            denseResidencyPolicy: options.denseWeightBudgetBytes.map { .automatic(weightBudgetBytes: $0) }
                ?? options.denseResidentLayers.map { .bounded(residentLayers: Set($0)) } ?? .resident,
            retainedConversationBytes: options.retainedConversationBytes)
        try resolved.validate()
        return resolved
    }

}
