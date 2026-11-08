import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFModelMetadata

public actor TUFFHTTPServer {
    public static let maximumBodyBytes = StreamingChatRequestBody.maximumWireBytes
    /// Accepted sockets are unbounded otherwise, and each one costs a file
    /// descriptor plus whatever its half-sent request has staged.
    public static let maximumConnections = 128
    /// How long a connection may sit without being read from before it is
    /// closed. Generous, because it must never interrupt a slow client that is
    /// still uploading images.
    public static let idleTimeout = TimeAmount.seconds(120)

    private let group: MultiThreadedEventLoopGroup
    private let provider: any ServerModelProvider
    private let control: ServerControl?
    private let apiKey: String?
    private let heartbeatInterval: TimeAmount
    private let onRequestError: @Sendable (String) -> Void
    private let attachmentRoot: URL
    private let idleTimeout: TimeAmount
    private let childChannels = ChildChannelRegistry()
    private var channel: Channel?
    private var shutdownTask: Task<Void, any Error>?

    /// A server over any model provider. `control` adds the loopback
    /// `/tuff/v1/status` and `/tuff/v1/unload` routes. `apiKey` admits
    /// callers that are not on this Mac; without it only loopback is served.
    public init(provider: any ServerModelProvider,
                control: ServerControl? = nil,
                apiKey: String? = nil,
                heartbeatInterval: TimeAmount = .seconds(5),
                onRequestError: @escaping @Sendable (String) -> Void = { _ in },
                attachmentRoot: URL = ServerAttachmentDirectory.root,
                idleTimeout: TimeAmount = TUFFHTTPServer.idleTimeout,
                group: MultiThreadedEventLoopGroup = .init(numberOfThreads: 1)) {
        self.group = group
        self.provider = provider
        self.control = control
        self.apiKey = apiKey
        self.heartbeatInterval = heartbeatInterval
        self.onRequestError = onRequestError
        self.attachmentRoot = attachmentRoot
        self.idleTimeout = idleTimeout
        ServerAttachmentDirectory.sweepAbandoned(in: attachmentRoot)
    }

    public func start(host: String = "127.0.0.1", port: Int) async throws -> Channel {
        let provider = self.provider
        let control = self.control
        let apiKey = self.apiKey
        let heartbeatInterval = self.heartbeatInterval
        let childChannels = self.childChannels
        let onRequestError = self.onRequestError
        let attachmentRoot = self.attachmentRoot
        let idleTimeout = self.idleTimeout
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 16)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                childChannels.insert(channel)
                do {
                    try channel.pipeline.syncOperations.addHandler(ConnectionCloseWatcher())
                    try channel.pipeline.syncOperations.addHandler(
                        IdleStateHandler(readTimeout: idleTimeout))
                } catch {
                    return channel.eventLoop.makeFailedFuture(error)
                }
                return channel.pipeline.configureHTTPServerPipeline(
                    withPipeliningAssistance: true,
                    withErrorHandling: true).flatMap {
                    channel.pipeline.addHandler(ServerHTTPHandler(
                        provider: provider,
                        control: control,
                        apiKey: apiKey,
                        heartbeatInterval: heartbeatInterval,
                        onRequestError: onRequestError,
                        attachmentRoot: attachmentRoot,
                        childChannels: childChannels))
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
        let channel = try await bootstrap.bind(host: host, port: port).get()
        self.channel = channel
        return channel
    }

    public func shutdown() async throws {
        if let shutdownTask {
            try await shutdownTask.value
            return
        }

        let listeningChannel = channel
        channel = nil
        let childChannels = self.childChannels
        let provider = self.provider
        let group = self.group
        let task = Task { @Sendable in
            var firstError: (any Error)?
            await provider.shutdown()
            if let listeningChannel {
                do {
                    try await listeningChannel.close().get()
                } catch ChannelError.alreadyClosed {
                } catch {
                    firstError = error
                }
            }
            await childChannels.closeAll()
            do {
                try await group.shutdownGracefully()
            } catch {
                if firstError == nil {
                    firstError = error
                }
            }
            if let firstError {
                throw firstError
            }
        }
        shutdownTask = task
        try await task.value
    }

    var queuedRequestCount: Int {
        get async { await provider.queuedCount }
    }

    var hasActiveRequest: Bool {
        get async { await provider.isActive }
    }

    var acceptedConnectionCount: Int {
        childChannels.count
    }
}

final class ServerHTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let provider: any ServerModelProvider
    private let control: ServerControl?
    private let apiKey: String?
    private let heartbeatInterval: TimeAmount
    private let childChannels: ChildChannelRegistry
    private let onRequestError: @Sendable (String) -> Void
    private let attachmentRoot: URL
    private var bodyParser: StreamingChatRequestBody?
    private var bodyError: (any Error)?
    private var requestArrival = DispatchTime.now().uptimeNanoseconds
    private var receivedBodyBytes = 0
    /// Past the wire cap the drain toward `.end` is unbounded - a chunked
    /// stream may never send one - so the request is answered immediately
    /// and everything after it dropped until the connection closes.
    private var discardingUntilClose = false
    /// Requests still producing an answer. A generation reads nothing for as
    /// long as it runs, so the idle timeout must not close under it.
    private var inFlightRequests = 0
    private var errorAPI: InferenceAPI = .chat
    private var head: HTTPRequestHead?
    private var activeTask: Task<Void, Never>?

    fileprivate init(provider: any ServerModelProvider,
            control: ServerControl?,
            apiKey: String?,
            heartbeatInterval: TimeAmount,
            onRequestError: @escaping @Sendable (String) -> Void,
            attachmentRoot: URL,
            childChannels: ChildChannelRegistry) {
        self.provider = provider
        self.control = control
        self.apiKey = apiKey
        self.heartbeatInterval = heartbeatInterval
        self.onRequestError = onRequestError
        self.attachmentRoot = attachmentRoot
        self.childChannels = childChannels
    }

    static let chatCompletionsPath = "/v1/chat/completions"

    static func requestPath(_ head: HTTPRequestHead) -> String {
        head.uri.split(separator: "?", maxSplits: 1,
                       omittingEmptySubsequences: false)
            .first.map(String.init) ?? head.uri
    }

    // A web page can point its own domain at 127.0.0.1 (DNS rebinding), and
    // the browser then sends that domain as Host and Origin. Only this Mac's
    // loopback names are served, and only `http` or `https` pages on them.
    // A request without Host (HTTP/1.0, or a non-browser client) cannot come
    // from a browser and passes.
    static let loopbackHosts: Set<String> = ["127.0.0.1", "localhost", "[::1]"]

    static func acceptsCaller(_ head: HTTPRequestHead) -> Bool {
        let hosts = head.headers["host"]
        guard hosts.count <= 1 else { return false }
        if let host = hosts.first, !isLoopback(authority: host) { return false }
        return head.headers["origin"].allSatisfy(isLoopback(origin:))
    }

    // Compares the name without case, a trailing dot or a numeric port.
    // Anything else after the name, such as `@` or a path, is refused.
    static func isLoopback(authority: String) -> Bool {
        var name = Substring(authority.trimmingCharacters(in: .whitespaces).lowercased())
        if name.hasPrefix("[") {
            guard let close = name.firstIndex(of: "]") else { return false }
            let port = name[name.index(after: close)...]
            guard port.isEmpty || isPort(port) else { return false }
            name = name[...close]
        } else if let colon = name.firstIndex(of: ":") {
            guard isPort(name[colon...]) else { return false }
            name = name[..<colon]
        }
        if name.count > 1, name.hasSuffix(".") { name = name.dropLast() }
        return loopbackHosts.contains(String(name))
    }

    // A caller with the API key is served whatever its Host and Origin: a
    // browser cannot attach the key to a rebound request. Without it, only a
    // caller on this Mac is served, under the loopback name check.
    static func refusal(_ head: HTTPRequestHead, fromLoopback: Bool,
                        apiKey: String?) -> ServerRequestError? {
        if let apiKey, ServerAPIKey.presented(in: head.headers, matches: apiKey) { return nil }
        guard fromLoopback else {
            return .invalid(
                message: "requests from other devices need this server's API key, "
                    + "sent as Authorization: Bearer <key> or x-api-key: <key>",
                param: nil, code: "invalid_api_key")
        }
        guard acceptsCaller(head) else {
            return .invalid(
                message: "Host or Origin is not a loopback name "
                    + "(127.0.0.1, localhost or [::1])",
                param: nil, code: "forbidden_host")
        }
        return nil
    }

    static func isLoopback(peer: SocketAddress?) -> Bool {
        guard let peer else { return false }
        switch peer {
        case .v4(let address):
            return UInt32(bigEndian: address.address.sin_addr.s_addr) >> 24 == 127
        case .v6(let address):
            let bytes = withUnsafeBytes(of: address.address.sin6_addr) { Array($0) }
            if bytes.dropLast().allSatisfy({ $0 == 0 }) && bytes.last == 1 { return true }
            // An IPv4 caller on a dual-stack socket: ::ffff:127.x.x.x.
            return bytes.prefix(10).allSatisfy { $0 == 0 }
                && bytes[10] == 0xFF && bytes[11] == 0xFF && bytes[12] == 127
        case .unixDomainSocket:
            return true
        }
    }

    // `null`, every other scheme and every other site are refused.
    static func isLoopback(origin: String) -> Bool {
        let lowered = origin.trimmingCharacters(in: .whitespaces).lowercased()
        for scheme in ["http://", "https://"] where lowered.hasPrefix(scheme) {
            return isLoopback(authority: String(lowered.dropFirst(scheme.count)))
        }
        return false
    }

    // A colon and one to five digits.
    private static func isPort(_ text: Substring) -> Bool {
        let digits = text.dropFirst()
        return text.first == ":" && (1...5).contains(digits.count)
            && digits.allSatisfy { $0.isASCII && $0.isNumber }
    }

    static func carriesChatBody(_ head: HTTPRequestHead) -> Bool {
        head.method == .POST
            && InferenceAPI.forPath(requestPath(head)) != nil
            && head.headers.first(name: "content-type")?
                .lowercased().hasPrefix("application/json") == true
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            guard !discardingUntilClose else { return }
            self.head = head
            errorAPI = InferenceAPI.forPath(Self.requestPath(head)) ?? .chat
            requestArrival = DispatchTime.now().uptimeNanoseconds
            // The parser stages inline images to disk, so it is created only
            // once the request is known to carry a chat body. A body sent
            // elsewhere is counted and dropped.
            let refusal = Self.refusal(
                head, fromLoopback: Self.isLoopback(peer: context.channel.remoteAddress),
                apiKey: apiKey)
            let accepted = refusal == nil
            bodyParser = accepted && Self.carriesChatBody(head)
                ? StreamingChatRequestBody(attachmentRoot: attachmentRoot,
                                           visionCapability: provider.parserVisionCapability)
                : nil
            bodyError = nil
            receivedBodyBytes = 0
            if let refusal {
                // Answered at .end, so a refused body is never parsed or staged.
                bodyError = refusal
            }
        case .body(var part):
            guard !discardingUntilClose else { return }
            receivedBodyBytes += part.readableBytes
            if receivedBodyBytes > StreamingChatRequestBody.maximumWireBytes {
                // A body already rejected for its own reason keeps it: telling a
                // client its image was too large is more useful than telling it
                // the request was, and the drain past the cap is what forces the
                // answer now.
                let capError = (bodyError as? ServerRequestError)
                    ?? ServerRequestError.invalid(
                        message: "request body is too large",
                        param: nil, code: "request_too_large")
                bodyError = nil
                bodyParser = nil
                head = nil
                discardingUntilClose = true
                writeError(context, status: capError.httpStatus, capError.envelope,
                           closeAfter: true)
                return
            }
            guard bodyError == nil else { return }
            guard let parser = bodyParser else { return }
            do { try parser.feed(&part) }
            catch {
                bodyError = error
                // Rejection is already certain, so release the staged bytes now.
                bodyParser = nil
            }
        case .end:
            guard !discardingUntilClose else { return }
            guard let head else { return }
            self.head = nil
            let parser = bodyParser
            // Cleared on every path below: a retained parser pins its staged
            // images until the connection closes.
            bodyParser = nil
            if let bodyError {
                self.bodyError = nil
                if let requestError = bodyError as? ServerRequestError {
                    writeError(context, status: requestError.httpStatus,
                               requestError.envelope)
                } else {
                    writeError(context, status: .badRequest,
                               OpenAIErrorEnvelope(message: "malformed JSON request",
                                                   code: "invalid_json"))
                }
                return
            }
            do {
                let parsed = try parser?.finish()
                    ?? ParsedChatRequestBody(json: Data(), stagedImages: [:], lease: nil)
                route(head: head, body: parsed, context: context)
            } catch let error as ServerRequestError {
                writeError(context, status: error.httpStatus, error.envelope)
            } catch {
                writeError(context, status: .badRequest,
                           OpenAIErrorEnvelope(message: "malformed JSON request",
                                               code: "invalid_json"))
            }
        }
    }


    func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
        if event is IdleStateHandler.IdleStateEvent, inFlightRequests == 0 {
            context.close(promise: nil)
            return
        }
        context.fireUserInboundEventTriggered(event)
    }

    func channelInactive(context: ChannelHandlerContext) {
        activeTask?.cancel()
        activeTask = nil
        // Frees the staged images of a request abandoned mid-body.
        bodyParser = nil
        bodyError = nil
        head = nil
        childChannels.remove(context.channel)
        context.fireChannelInactive()
    }

    private func route(head: HTTPRequestHead,
                       body: ParsedChatRequestBody,
                       context: ChannelHandlerContext) {
        let path = head.uri.split(separator: "?", maxSplits: 1,
                                  omittingEmptySubsequences: false).first.map(String.init) ?? head.uri
        switch (head.method, path) {
        case (.GET, "/health"):
            respondToHealth(context)
        case (.GET, "/v1/models"):
            writeCodable(context, status: .ok, provider.modelList())
        case (.GET, ServerControl.statusPath) where control != nil:
            respondToControl(context, head: head, unload: false)
        case (.POST, ServerControl.unloadPath) where control != nil:
            respondToControl(context, head: head, unload: true)
        case (.POST, "/v1/chat/completions"), (.POST, "/v1/messages"), (.POST, "/v1/responses"):
            guard head.headers.first(name: "content-type")?
                .lowercased().hasPrefix("application/json") == true else {
                writeError(context, status: .unsupportedMediaType,
                           OpenAIErrorEnvelope(message: "content-type must be application/json",
                                               code: "unsupported_media_type"))
                return
            }
            handleCompletion(body: body, api: InferenceAPI.forPath(path)!, context: context)
        case (_, "/health"), (_, "/v1/models"), (_, "/v1/chat/completions"), (_, "/v1/messages"), (_, "/v1/responses"),
             (_, ServerControl.statusPath) where control != nil,
             (_, ServerControl.unloadPath) where control != nil:
            writeError(context, status: .methodNotAllowed,
                       OpenAIErrorEnvelope(message: "method not allowed",
                                           code: "method_not_allowed"))
        default:
            writeError(context, status: .notFound,
                       OpenAIErrorEnvelope(message: "route not found",
                                           code: "not_found"))
        }
    }

    private func respondToControl(_ context: ChannelHandlerContext,
                                  head: HTTPRequestHead,
                                  unload: Bool) {
        guard let control else { return }
        if unload, !control.authorizes(head.headers.first(name: "authorization")) {
            writeError(context, status: .unauthorized,
                       OpenAIErrorEnvelope(message: "a valid control token is required",
                                           code: "unauthorized"))
            return
        }
        let contextBox = SendableContext(context)
        inFlightRequests += 1
        activeTask = childChannels.startTask {
            defer {
                contextBox.value.eventLoop.execute { self.inFlightRequests -= 1 }
            }
            // An unload never interrupts work: a busy server answers 409 and
            // keeps its model.
            let unloaded = unload ? await control.unloadIfIdle() : true
            let status = await control.status()
            self.writeCodable(contextBox.value, status: unloaded ? .ok : .conflict, status)
        }
    }

    private func respondToHealth(_ context: ChannelHandlerContext) {
        let contextBox = SendableContext(context)
        inFlightRequests += 1
        activeTask = childChannels.startTask {
            defer {
                contextBox.value.eventLoop.execute { self.inFlightRequests -= 1 }
            }
            let health = await self.provider.health()
            self.writeCodable(contextBox.value, status: .ok, health)
        }
    }

    private func handleCompletion(body: ParsedChatRequestBody,
                                  api: InferenceAPI,
                                  context: ChannelHandlerContext) {
        var requestedModel: String?
        do {
            let decoded = try api.request(body.json)
            requestedModel = decoded.model
            let target = try provider.route(decoded.model)
            let modelID = target.id
            try api.validateNativeHistory(decoded, dialect: target.dialect)
            let request = try OpenAIRequestValidator.validate(
                decoded,
                modelID: target.requestedID,
                dialect: target.dialect,
                preStagedImages: body.stagedImages,
                attachmentLease: body.lease)
            // A router stages images before it knows the model, so the model
            // it chose must take them. Images never pass silently unused.
            if !request.imageFiles.isEmpty, target.visionCapability != "ready" {
                throw ServerRequestError.invalid(
                    message: "\(modelID) does not accept images on this Mac",
                    param: "messages", code: "image_input_unavailable")
            }
            let responseID = (api == .messages ? "msg_" : api == .responses ? "resp_" : "chatcmpl-") + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
            let created = Int(Date().timeIntervalSince1970)
            let adapter = api == .chat ? nil : InferenceAdapterWire(api: api, id: responseID, model: modelID, created: created, dialect: target.dialect)
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let phaseState = RequestPhaseState()
            let timings = ServerRequestTimings(arrival: requestArrival,
                validated: DispatchTime.now().uptimeNanoseconds)
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                      streamState.start(eventLoop: contextBox.value.eventLoop,
                                        interval: self.heartbeatInterval,
                                        ping: {
                          self.writeHeartbeat(contextBox.value)
                      }) else { return }
                let future: EventLoopFuture<Void>
                if let adapter {
                    // Messages needs the actual prompt count in message_start.
                    // A queued stream can send comments until preparation finishes.
                    future = self.beginAdapterStream(contextBox.value,
                        frames: api == .messages ? [] : adapter.start())
                } else {
                    future = self.beginStream(
                        contextBox.value,
                        self.chunk(model: modelID, id: responseID, created: created,
                                   delta: ["role": "assistant"], finishReason: nil))
                }
                streamState.setStartFuture(future)
            }
            let onQueued: @Sendable () -> Void = {
                phaseState.set("queued")
                ServerLog.queued(id: responseID)
                startStream()
            }
            inFlightRequests += 1
            ConnectionCloseWatcher.watch(context)
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
                // Back on the event loop, where `inFlightRequests` lives. The hop
                // also orders this after the assignment above, which is still
                // executing in this same event-loop tick.
                defer {
                    contextBox.value.eventLoop.execute {
                        self.inFlightRequests -= 1
                    }
                }
                let started = ContinuousClock.now
                let recorder = Mutex<(@Sendable (ServerCompletion) -> Void)?>(nil)
                ServerLog.accepted(id: responseID, streaming: request.stream)
                do {
                    var completion = try await self.provider.run(
                        target,
                        onQueued: onQueued,
                        prepare: { backend in
                            // The model is loaded by now, so its load is not
                            // counted as part of the run.
                            recorder.withLock { $0 = self.provider.completionRecorder(for: target) }
                            timings.preparing()
                            let prepared = try await backend.prepare(request)
                            timings.prepared()
                            phaseState.set("prepared")
                            ServerLog.prepared(id: responseID,
                                               promptTokens: prepared.promptTokenCount)
                            return prepared
                        },
                        operation: { (backend: any ServerInferenceBackend, prepared: ServerPreparedRequest) in
                            try Task.checkCancellation()
                            startStream()
                            try await streamState.waitUntilStarted()
                            if request.stream, let adapter, api == .messages {
                                self.writeAdapterFrames(contextBox.value,
                                    frames: adapter.start(promptTokens: prepared.promptTokenCount ?? 0))
                            }
                            try Task.checkCancellation()
                            timings.generating()
                            phaseState.set("generating")
                            ServerLog.generating(id: responseID)
                            return try await backend.generate(prepared) { event in
                                if let adapter {
                                    let frames = adapter.event(event)
                                    if !frames.isEmpty { timings.visibleEvent() }
                                    if request.stream {
                                        self.writeAdapterFrames(contextBox.value, frames: frames)
                                    }
                                    return
                                }
                                switch event {
                                case .content(let text), .reasoning(let text):
                                    if !text.isEmpty { timings.visibleEvent() }
                                case .toolCall: timings.visibleEvent()
                                }
                                guard request.stream else { return }
                                switch event {
                                case .content(let text):
                                    self.writeStreamChunk(
                                        contextBox.value,
                                        self.chunk(model: modelID, id: responseID, created: created,
                                                   delta: ["content": text],
                                                   finishReason: nil))
                                case .reasoning(let text):
                                    self.writeStreamChunk(
                                        contextBox.value,
                                        self.chunk(model: modelID, id: responseID, created: created,
                                                   delta: ["reasoning_content": text],
                                                   finishReason: nil))
                                case .toolCall(let call):
                                    self.writeToolCall(contextBox.value,
                                                       model: modelID,
                                                       id: responseID,
                                                       created: created,
                                                       toolIndex: streamState.nextToolIndex(),
                                                       call: call)
                                }
                            }
                    })
                    try adapter?.validateCompletion()
                    completion.timingSeconds.merge(timings.snapshot(), uniquingKeysWith: { _, new in new })
                    ServerLog.completed(id: responseID,
                                        duration: started.duration(to: .now),
                                        completion: completion)
                    recorder.withLock { $0 }?(completion)
                    if request.stream {
                        streamState.stop()
                        if let adapter {
                            self.writeAdapterFrames(contextBox.value, frames: adapter.finish(completion), end: true)
                        } else {
                            self.finishStream(contextBox.value,
                                              model: modelID,
                                              id: responseID,
                                              created: created,
                                              completion: completion,
                                              includeUsage: request.includeUsage)
                        }
                    } else if let adapter {
                        self.writeJSON(contextBox.value, status: .ok, object: try adapter.completed(completion))
                    } else {
                        self.writeCompletion(contextBox.value,
                                             model: modelID,
                                             id: responseID,
                                             created: created,
                                             completion: completion)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncError(error,
                                          context: contextBox.value,
                                          id: responseID,
                                          phase: phaseState.value,
                                          stream: streamState.isStarted,
                                          started: started,
                                          adapter: adapter)
                }
            }
        } catch let error as ServerRequestError {
            if case .unknownModel = error, let requestedModel {
                writeError(context, status: error.httpStatus, OpenAIErrorEnvelope(
                    message: "model \"\(requestedModel)\" is not available; "
                        + "GET /v1/models lists the installed models",
                    param: "model", code: "model_not_found"))
                return
            }
            writeError(context,
                       status: error.httpStatus,
                       error.envelope)
        } catch {
            writeError(context, status: .badRequest, OpenAIErrorEnvelope.decoding(error))
        }
    }

    private func writeCompletion(_ context: ChannelHandlerContext,
                                 model modelID: String,
                                 id: String,
                                 created: Int,
                                 completion: ServerCompletion) {
        let encodedContent: Any =
            completion.content.isEmpty && !completion.toolCalls.isEmpty
                ? NSNull()
                : completion.content
        var message: [String: Any] = [
            "role": "assistant",
            "content": encodedContent,
        ]
        if !completion.toolCalls.isEmpty {
            message["tool_calls"] = completion.toolCalls.map(toolCallObject)
        }
        if !completion.reasoning.isEmpty {
            message["reasoning_content"] = completion.reasoning
        }
        let object: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": modelID,
            "choices": [[
                "index": 0,
                "message": message,
                "finish_reason": completion.finishReason,
            ]],
            "usage": usageObject(completion.usage),
            "tuff_timings_seconds": completion.timingSeconds,
        ]
        writeJSON(context, status: .ok, object: object)
    }

    private func adapterBytes(_ frames: [AdapterSSEFrame]) -> Data {
        var data = Data()
        for frame in frames {
            guard let json = try? JSONSerialization.data(withJSONObject: frame.object) else { continue }
            data.append(contentsOf: "event: \(frame.event)\ndata: ".utf8)
            data.append(json)
            data.append(contentsOf: "\n\n".utf8)
        }
        return data
    }

    private func beginAdapterStream(_ context: ChannelHandlerContext,
                                    frames: [AdapterSSEFrame]) -> EventLoopFuture<Void> {
        let bytes = adapterBytes(frames)
        let box = SendableContext(context)
        let promise = context.eventLoop.makePromise(of: Void.self)
        context.eventLoop.execute {
            let headers = HTTPHeaders([("content-type", "text/event-stream"), ("cache-control", "no-cache"), ("connection", "keep-alive")])
            box.value.write(self.wrapOutboundOut(.head(HTTPResponseHead(version: .http1_1, status: .ok, headers: headers))), promise: nil)
            var buffer = box.value.channel.allocator.buffer(capacity: bytes.count)
            buffer.writeBytes(bytes)
            box.value.writeAndFlush(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: promise)
        }
        return promise.futureResult
    }

    private func writeAdapterFrames(_ context: ChannelHandlerContext,
                                    frames: [AdapterSSEFrame], end: Bool = false) {
        let bytes = adapterBytes(frames)
        let box = SendableContext(context)
        context.eventLoop.execute {
            guard box.value.channel.isActive else { return }
            if !bytes.isEmpty {
                var buffer = box.value.channel.allocator.buffer(capacity: bytes.count)
                buffer.writeBytes(bytes)
                box.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            }
            if end { box.value.write(self.wrapOutboundOut(.end(nil)), promise: nil) }
            box.value.flush()
        }
    }

    private func beginStream(
        _ context: ChannelHandlerContext,
        _ initialChunk: [String: Any]
    ) -> EventLoopFuture<Void> {
        guard let data = try? JSONSerialization.data(withJSONObject: initialChunk) else {
            return context.eventLoop.makeFailedFuture(ServerRequestError.invalid(
                message: "stream response could not be encoded",
                param: nil,
                code: "internal_error"))
        }
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let contextBox = SendableContext(context)
        let promise = context.eventLoop.makePromise(of: Void.self)
        context.eventLoop.execute {
            contextBox.value.write(self.wrapOutboundOut(.head(head)),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count + 8)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\n")
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))),
                promise: promise)
        }
        return promise.futureResult
    }

    private func writeToolCall(_ context: ChannelHandlerContext,
                               model modelID: String,
                               id: String,
                               created: Int,
                               toolIndex: Int,
                               call: ParsedToolCall) {
        let fragments = utf8Fragments(call.argumentsJSON, maximumBytes: 1024)
        for (index, fragment) in fragments.enumerated() {
            var function: [String: Any] = ["arguments": fragment]
            var tool: [String: Any] = ["index": toolIndex, "function": function]
            if index == 0 {
                function["name"] = call.name
                tool["id"] = call.id
                tool["type"] = "function"
                tool["function"] = function
            }
            writeStreamChunk(
                context,
                chunk(model: modelID, id: id, created: created,
                      delta: ["tool_calls": [tool]],
                      finishReason: nil))
        }
    }

    private func finishStream(_ context: ChannelHandlerContext,
                              model modelID: String,
                              id: String,
                              created: Int,
                              completion: ServerCompletion,
                              includeUsage: Bool) {
        var terminal = chunk(model: modelID, id: id, created: created,
                             delta: [:], finishReason: completion.finishReason)
        terminal["tuff_timings_seconds"] = completion.timingSeconds
        writeStreamChunk(context, terminal)
        if includeUsage {
            writeStreamChunk(context, [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": modelID,
                "choices": [],
                "usage": usageObject(completion.usage),
            ])
        }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            let buffer = contextBox.value.channel.allocator.buffer(string: "data: [DONE]\n\n")
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func chunk(model modelID: String,
                       id: String,
                       created: Int,
                       delta: [String: Any],
                       finishReason: String?) -> [String: Any] {
        let encodedReason: Any = finishReason.map { $0 as Any } ?? NSNull()
        return [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": modelID,
            "choices": [[
                "index": 0,
                "delta": delta,
                "finish_reason": encodedReason,
            ]],
        ]
    }

    private func writeStreamChunk(_ context: ChannelHandlerContext,
                                  _ object: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count + 8)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\n")
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
        }
    }

    private func writeHeartbeat(_ context: ChannelHandlerContext) {
        let buffer = context.channel.allocator.buffer(string: ": ping\n\n")
        context.writeAndFlush(
            wrapOutboundOut(.body(.byteBuffer(buffer))),
            promise: nil)
    }

    private func handleAsyncError(_ error: Error,
                                  context: ChannelHandlerContext,
                                  id: String,
                                  phase: String,
                                  stream: Bool,
                                  started: ContinuousClock.Instant,
                                  adapter: InferenceAdapterWire? = nil) {
        let envelope: OpenAIErrorEnvelope
        let status: HTTPResponseStatus
        if let requestError = error as? ServerRequestError {
            status = requestError.httpStatus
            envelope = requestError.envelope
        } else {
            status = .internalServerError
            envelope = OpenAIErrorEnvelope(
                message: "generation failed; see TUFFServer stderr",
                code: "internal_error",
                type: "server_error")
        }
        if error is CancellationError {
            // Not a failure, and not silence either: without a terminal line the
            // request's last record stayed `generating` forever, so a reader could
            // not tell running from abandoned from crashed.
            ServerLog.cancelled(id: id, phase: phase,
                                duration: started.duration(to: .now))
        } else {
            onRequestError(String(describing: error))
            ServerLog.failed(id: id, phase: phase, status: status.code, error: error)
        }
        if stream, let adapter {
            writeAdapterFrames(context, frames: adapter.error(envelope), end: true)
            return
        }
        if stream {
            finishStreamWithError(context, envelope: envelope)
            return
        }
        if adapter?.api == .messages {
            writeJSON(context, status: status, object: ["type": "error", "error": ["type": status.code >= 500 ? "api_error" : envelope.error.type, "message": envelope.error.message]])
        } else {
            writeCodable(context, status: status, envelope)
        }
    }

    private func finishStreamWithError(_ context: ChannelHandlerContext,
                                       envelope: OpenAIErrorEnvelope) {
        guard let data = try? JSONEncoder().encode(envelope) else { return }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            guard contextBox.value.channel.isActive else { return }
            var buffer = contextBox.value.channel.allocator.buffer(
                capacity: data.count + 32)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\ndata: [DONE]\n\n")
            contextBox.value.write(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func writeCodable<T: Encodable>(_ context: ChannelHandlerContext,
                                            status: HTTPResponseStatus,
                                            _ value: T,
                                            closeAfter: Bool = false) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        writeData(context, status: status, data: data, closeAfter: closeAfter)
    }

    private func writeError(_ context: ChannelHandlerContext,
                            status: HTTPResponseStatus,
                            _ error: OpenAIErrorEnvelope,
                            closeAfter: Bool = false) {
        if errorAPI == .messages {
            let type = status.code >= 500 ? "api_error"
                : error.error.code == "forbidden_host" ? "permission_error"
                : error.error.code == "invalid_api_key" ? "authentication_error" : error.error.type
            let object: [String: Any] = ["type": "error", "error": ["type": type, "message": error.error.message]]
            guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
            writeData(context, status: status, data: data, closeAfter: closeAfter)
        } else {
            writeCodable(context, status: status, error, closeAfter: closeAfter)
        }
    }

    private func writeJSON(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           object: Any) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        writeData(context, status: status, data: data)
    }

    private func writeData(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           data: Data,
                           closeAfter: Bool = false) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "\(data.count)")
            if closeAfter {
                headers.add(name: "connection", value: "close")
            }
            // A full queue clears as requests finish; clients that honour
            // Retry-After back off instead of retrying at once.
            if status == .tooManyRequests {
                headers.add(name: "retry-after", value: "5")
            }
            contextBox.value.write(self.wrapOutboundOut(.head(
                HTTPResponseHead(version: .http1_1, status: status, headers: headers))),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            if closeAfter {
                let promise = contextBox.value.eventLoop.makePromise(of: Void.self)
                let channel = contextBox.value.channel
                promise.futureResult.whenComplete { _ in
                    // Lingering close: an immediate close RSTs away the very
                    // response that explains the rejection while the client's
                    // remaining bytes are still in flight. The grace is a hard
                    // bound - a client that streams past it is cut off with its
                    // response undelivered, which is on the client.
                    _ = channel.eventLoop.scheduleTask(in: .seconds(2)) {
                        channel.close(promise: nil)
                    }
                }
                contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: promise)
            } else {
                contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
            }
        }
    }

    private func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        var object: [String: Any] = [
            "prompt_tokens": usage.promptTokens,
            "completion_tokens": usage.completionTokens,
            "total_tokens": usage.totalTokens,
            "prompt_tokens_details": [
                "cached_tokens": usage.promptTokensDetails.cachedTokens,
            ],
        ]
        if let details = usage.completionTokensDetails {
            object["completion_tokens_details"] = [
                "reasoning_tokens": details.reasoningTokens,
            ]
        }
        return object
    }

    private func toolCallObject(_ call: ParsedToolCall) -> [String: Any] {
        [
            "id": call.id,
            "type": "function",
            "function": [
                "name": call.name,
                "arguments": call.argumentsJSON,
            ],
        ]
    }

    private func utf8Fragments(_ text: String, maximumBytes: Int) -> [String] {
        guard !text.isEmpty else { return [""] }
        var result: [String] = []
        var current = ""
        var bytes = 0
        for character in text {
            let size = String(character).utf8.count
            if bytes + size > maximumBytes, !current.isEmpty {
                result.append(current)
                current = ""
                bytes = 0
            }
            current.append(character)
            bytes += size
        }
        if !current.isEmpty { result.append(current) }
        return result
    }
}

/// The one mapping from a request rejection to its wire status. Every write
/// site derives the status from here, so a new resource-limit code cannot
/// return 400 at one site and 413 at another.
private extension ServerRequestError {
    var httpStatus: HTTPResponseStatus {
        switch self {
        case .unknownModel, .modelUnavailable:
            .notFound
        case .queueFull:
            .tooManyRequests
        case .modelMemoryBusy, .hostUnavailable, .modelBusy:
            .serviceUnavailable
        case .invalid(_, _, let code):
            switch code {
            case "request_too_large", "image_too_large", "too_many_images":
                .payloadTooLarge
            case "forbidden_host":
                .forbidden
            case "invalid_api_key":
                .unauthorized
            default:
                .badRequest
            }
        }
    }
}

private final class ChildChannelRegistry: Sendable {
    private struct State {
        var channels: [ObjectIdentifier: Channel] = [:]
        var tasks: [UUID: Task<Void, Never>] = [:]
        var shuttingDown = false
    }

    private let state = Mutex(State())

    /// Registers a connection, or closes it because the server is shutting
    /// down or already at its connection cap.
    func insert(_ channel: Channel) {
        let shouldClose = state.withLock {
            guard !$0.shuttingDown,
                  $0.channels.count < TUFFHTTPServer.maximumConnections
            else { return true }
            $0.channels[ObjectIdentifier(channel)] = channel
            return false
        }
        if shouldClose {
            channel.close(promise: nil)
        }
    }

    func remove(_ channel: Channel) {
        _ = state.withLock {
            $0.channels.removeValue(forKey: ObjectIdentifier(channel))
        }
    }

    func startTask(
        _ operation: @escaping @Sendable () async -> Void
    ) -> Task<Void, Never> {
        state.withLock { state in
            let id = UUID()
            let task = Task { [self] in
                defer {
                    _ = self.state.withLock {
                        $0.tasks.removeValue(forKey: id)
                    }
                }
                await operation()
            }
            state.tasks[id] = task
            if state.shuttingDown {
                task.cancel()
            }
            return task
        }
    }

    func closeAll() async {
        let channels = state.withLock {
            $0.shuttingDown = true
            return Array($0.channels.values)
        }
        for channel in channels {
            try? await channel.close().get()
        }
        let tasks = state.withLock { Array($0.tasks.values) }
        for task in tasks {
            task.cancel()
        }
        for task in tasks {
            await task.value
        }
    }

    var count: Int {
        state.withLock { $0.channels.count }
    }
}

private final class SendableContext: @unchecked Sendable {
    let value: ChannelHandlerContext

    init(_ value: ChannelHandlerContext) {
        self.value = value
    }
}

private final class RequestPhaseState: Sendable {
    private let state = Mutex("accepted")

    var value: String { state.withLock { $0 } }

    func set(_ value: String) {
        state.withLock { $0 = value }
    }
}

private final class StreamState: @unchecked Sendable {
    private let lock = NSLock()
    private var started = false
    private var stopped = false
    private var heartbeat: RepeatedTask?
    private var startFuture: EventLoopFuture<Void>?
    private var toolIndex = 0

    var isStarted: Bool {
        lock.withLock { started }
    }

    func start(eventLoop: EventLoop,
               interval: TimeAmount,
               ping: @escaping @Sendable () -> Void) -> Bool {
        lock.withLock {
            guard !started else { return false }
            started = true
            stopped = false
            startFuture = nil
            heartbeat = eventLoop.scheduleRepeatedTask(
                initialDelay: interval,
                delay: interval) { [weak self] _ in
                    guard self?.shouldPing == true else { return }
                    ping()
                }
            return true
        }
    }

    func setStartFuture(_ future: EventLoopFuture<Void>) {
        lock.withLock { startFuture = future }
    }

    func waitUntilStarted() async throws {
        let future = lock.withLock { startFuture }
        if let future {
            try await future.get()
        }
    }

    private var shouldPing: Bool {
        lock.withLock { started && !stopped }
    }

    func stop() {
        lock.withLock {
            stopped = true
            heartbeat?.cancel()
            heartbeat = nil
        }
    }

    func nextToolIndex() -> Int {
        lock.withLock {
            defer { toolIndex += 1 }
            return toolIndex
        }
    }
}

/// Sits nearest the socket so a read can be issued from below NIO's
/// pipelining handler. That handler holds reads while a response is pending,
/// so a client that left before a non-streamed answer went unnoticed and the
/// model kept generating for nobody. Streamed answers write as they go and
/// see the close that way.
final class ConnectionCloseWatcher: ChannelInboundHandler {
    typealias InboundIn = ByteBuffer
    typealias InboundOut = ByteBuffer

    /// Reads once from the socket. A close ends the channel, which cancels the
    /// request; anything else that arrives is buffered as a pipelined request.
    static func watch(_ context: ChannelHandlerContext) {
        (try? context.pipeline.syncOperations.context(
            handlerType: ConnectionCloseWatcher.self))?.read()
    }
}

extension OpenAIErrorEnvelope {
    /// Valid JSON in the wrong shape names the field rather than calling the
    /// JSON malformed, which sent clients looking for a syntax error.
    static func decoding(_ error: any Error) -> OpenAIErrorEnvelope {
        func path(_ codingPath: [any CodingKey], _ last: (any CodingKey)? = nil) -> String {
            let keys = codingPath + (last.map { [$0] } ?? [])
            let joined = keys.map { $0.intValue.map { "[\($0)]" } ?? ".\($0.stringValue)" }
                .joined()
            return joined.hasPrefix(".") ? String(joined.dropFirst()) : joined
        }
        switch error as? DecodingError {
        case .keyNotFound(let key, let context)?:
            let field = path(context.codingPath, key)
            return OpenAIErrorEnvelope(message: "missing required field: \(field)",
                                       param: field, code: "missing_required_parameter")
        case .typeMismatch(let type, let context)?, .valueNotFound(let type, let context)?:
            let field = path(context.codingPath)
            let expected = String(describing: type).lowercased()
            return OpenAIErrorEnvelope(
                message: field.isEmpty
                    ? "the request body must be a JSON object"
                    : "\(field) has the wrong type; expected \(expected)",
                param: field.isEmpty ? nil : field, code: "invalid_type")
        case .dataCorrupted(let context)? where !context.codingPath.isEmpty:
            let field = path(context.codingPath)
            return OpenAIErrorEnvelope(message: "invalid value for \(field)",
                                       param: field, code: "invalid_value")
        default:
            return OpenAIErrorEnvelope(message: "malformed JSON request", code: "invalid_json")
        }
    }
}
