import Darwin
import Foundation
import TUFFAppCore
import TUFFDecodeProtocol
import TUFFInferenceTypes
import TUFFModelCatalog

/// Images of one request, readable by the host's helper.
///
/// The helper refuses any attachment outside the app's attachment roots, and
/// the server stages inline images under its own root. Each image is hard
/// linked (copied when the volume differs) into a request directory under the
/// shared root. The links are independent of the server's own staging lease,
/// so the helper can read them until the request's terminal event, whatever
/// happens to the lease; `remove()` deletes them once the host is done.
struct HostServingStagedImages {
    let directory: URL
    let attachments: [UUID: DecodeImageAttachment]

    static func stage(_ sources: [HostServingTranslation.ImageSource]) throws -> HostServingStagedImages {
        let directory = AppImageAttachmentStore.root
            .appendingPathComponent("pid-\(getpid())", isDirectory: true)
            .appendingPathComponent("api-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        var attachments: [UUID: DecodeImageAttachment] = [:]
        do {
            for (index, source) in sources.enumerated() {
                let destination = directory.appendingPathComponent(source.id.uuidString)
                if link(source.fileURL.path, destination.path) != 0 {
                    try FileManager.default.copyItem(at: source.fileURL, to: destination)
                    chmod(destination.path, S_IRUSR)
                }
                let size = (try FileManager.default.attributesOfItem(
                    atPath: destination.path)[.size] as? Int) ?? 0
                attachments[source.id] = DecodeImageAttachment(
                    id: source.id, path: destination.path,
                    displayName: "image-\(index + 1)", encodedBytes: size, sha256: source.sha256)
            }
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw ServerRequestError.invalid(
                message: "an attached image could not be prepared for the inference host",
                param: "messages", code: "invalid_image")
        }
        return HostServingStagedImages(directory: directory, attachments: attachments)
    }

    func remove() { try? FileManager.default.removeItem(at: directory) }
}

/// One API request's view of the host: it makes the model resident, renders
/// the request into the host's decode protocol, starts the generation in a host
/// session, and turns the host's events back into API events.
///
/// Preparation starts the generation, so a request the host rejects (context
/// overflow, unsupported shape) fails before the HTTP layer commits to a
/// stream. The provider calls `finish` once the request is over, however it
/// ended.
actor HostServingBackend: ServerInferenceBackend {
    nonisolated let visionCapability: String
    private let provider: HostRoutedModelProvider
    private let descriptor: TUFFModelDescriptor
    private var generation: HostServingGeneration?
    private var session: (id: UUID, keyed: Bool)?
    private var staged: HostServingStagedImages?
    private var completed = false
    private var retriedMalformedToolCall = false

    init(provider: HostRoutedModelProvider, descriptor: TUFFModelDescriptor) {
        self.provider = provider
        self.descriptor = descriptor
        visionCapability = provider.installed.visionCapability(for: descriptor)
    }

    /// Appended to the system prompt for the one retry after a tool call the
    /// decoder could not read. The same words the app's answer runner uses;
    /// nothing from the failed output is echoed back to the model.
    static let malformedToolCallCorrection = "Your previous response contained an invalid tool call. Use the declared tool format, tool names and required argument types, and finish every argument. If you cannot make a valid call, answer directly without calling tools."

    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        guard generation == nil else {
            throw HostServingError("a request can be prepared only once")
        }
        if !request.imageFiles.isEmpty, visionCapability != "ready" {
            throw ServerRequestError.invalid(
                message: "The loaded model cannot accept images on this Mac",
                param: "messages", code: "image_input_unavailable")
        }
        do {
            try await start(request, requestID: UUID(), correction: nil)
        } catch ServerRequestError.malformedToolCall where !retriedMalformedToolCall {
            // The whole generation failed before the host reported progress.
            try await retryWithCorrection(request)
        }
        guard let started = generation else {
            throw HostServingError("the host generation did not start")
        }
        return ServerPreparedRequest(request: request, remotePreparationID: started.requestID)
    }

    /// Ends the failed attempt and starts the one corrective retry, in a
    /// fresh session so none of the failed attempt's state is reused.
    private func retryWithCorrection(_ request: ValidatedChatRequest) async throws {
        retriedMalformedToolCall = true
        ServerLog.malformedToolCallRetry(model: descriptor.apiModelID)
        if let running = generation {
            let terminal = await running.sawTerminal
            await running.drain(cancel: !terminal)
        }
        generation = nil
        if let session {
            await provider.closeSession(session.id, keyed: session.keyed, reuse: false)
        }
        session = nil
        try Task.checkCancellation()
        try await start(request, requestID: UUID(), correction: Self.malformedToolCallCorrection)
    }

    /// Starts one host generation for the request and waits until the host
    /// accepted it.
    private func start(_ request: ValidatedChatRequest, requestID: UUID,
                       correction: String?) async throws {
        let config = request.generationConfig
        let force = HostServingTranslation.forceLogitsHead(
            temperature: config.temperature, repetitionPenalty: config.repetitionPenalty)
        var attempt = 0
        while true {
            let preparation = try await provider.acquire(descriptor, forceLogitsHead: force)
            try Task.checkCancellation()
            if staged == nil, !request.imageFiles.isEmpty {
                staged = try HostServingStagedImages.stage(HostServingTranslation.imageSources(request))
            }
            let payload = try HostServingTranslation.decodeRequest(
                request, generationID: requestID,
                maxContextTokens: preparation.maxContextTokens,
                runtimeOptions: preparation.runtimeOptions,
                images: staged?.attachments ?? [:],
                correction: correction)
            if session == nil {
                session = await provider.openSession(
                    model: descriptor.apiModelID, cacheKey: request.promptCacheKey)
            }
            let routed: ServingGenerationRequest
            do {
                routed = try ServingGenerationRequest(
                    modelID: descriptor.apiModelID, sessionID: session!.id,
                    requestID: requestID, payload: payload)
            } catch {
                throw ServerRequestError.invalid(
                    message: "\(error)", param: nil, code: "invalid_request")
            }
            let client = try await provider.link.attach()
            let started = HostServingGeneration(
                link: provider.link, client: client, request: routed)
            generation = started
            do {
                try await started.awaitStart()
                return
            } catch ServerRequestError.hostUnavailable where attempt == 0 {
                // The host went away between the model check and the start
                // (it may have idled out or restarted). Reconnect, re-prepare
                // the model, and try once more.
                attempt += 1
                await started.drain(cancel: false)
                generation = nil
                provider.link.close()
                try Task.checkCancellation()
            }
        }
    }

    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        let prepared = try await prepare(request)
        return try await generate(prepared, onEvent: onEvent)
    }

    func generate(_ prepared: ServerPreparedRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        guard let first = generation, prepared.remotePreparationID == first.requestID else {
            throw HostServingError("the request was not prepared by this backend")
        }
        while true {
            guard let running = generation else {
                throw HostServingError("the request was not prepared by this backend")
            }
            do {
                return try await run(running, request: prepared.request, onEvent: onEvent)
            } catch let failure as HostServingAttemptFailure {
                // A tool call the decoder could not read, before anything
                // reached the client: regenerate once with a correction, as
                // the app does, instead of failing a request that was valid.
                // Once text has reached the client, a retry would repeat it,
                // so the error goes to the client instead.
                guard !retriedMalformedToolCall, !Task.isCancelled,
                      !failure.emitted || !prepared.eventsReachClient else {
                    throw failure.error
                }
                prepared.onDiscardedAttempt?()
                try await retryWithCorrection(prepared.request)
            }
        }
    }

    /// Reads one host generation to its end. A failure is wrapped with
    /// whether any event reached the client.
    private func run(_ running: HostServingGeneration, request: ValidatedChatRequest,
                     onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        var translator = HostServingEventTranslator(
            stops: request.generationConfig.stopStrings)
        var emitted = false
        return try await withTaskCancellationHandler {
            while true {
                guard let event = try await running.next() else {
                    try Task.checkCancellation()
                    throw ServerRequestError.hostUnavailable(
                        "The TUFF inference host connection ended before the generation finished.")
                }
                let output = translator.consume(event)
                if !Task.isCancelled, !output.events.isEmpty {
                    emitted = true
                    for item in output.events { onEvent(item) }
                }
                if output.requestStop { running.cancel() }
                guard let terminal = output.terminal else { continue }
                await running.markTerminal()
                switch terminal {
                case .completed(let completion):
                    try Task.checkCancellation()
                    completed = true
                    return completion
                case .cancelled:
                    try Task.checkCancellation()
                    throw HostServingError("The TUFF inference host cancelled the generation.")
                case .failed(let error):
                    if let requestError = error as? ServerRequestError,
                       case .malformedToolCall = requestError {
                        throw HostServingAttemptFailure(error: requestError, emitted: emitted)
                    }
                    throw error
                }
            }
        } onCancel: {
            running.cancel()
        }
    }

    /// Releases everything the request holds once the host is done with it:
    /// waits for the host's terminal event (cancelling first when none has
    /// arrived), removes the staged images, and ends or keeps the session.
    func finish(succeeded: Bool) async {
        if let running = generation {
            let terminal = await running.sawTerminal
            await running.drain(cancel: !terminal)
        }
        generation = nil
        staged?.remove()
        staged = nil
        if let session {
            await provider.closeSession(session.id, keyed: session.keyed,
                                        reuse: succeeded && completed)
        }
        session = nil
    }
}

/// A failed attempt the backend may retry.
private struct HostServingAttemptFailure: Error {
    let error: ServerRequestError
    let emitted: Bool
}
