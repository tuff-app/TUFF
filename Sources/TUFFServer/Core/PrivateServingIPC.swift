import Darwin
import Foundation
import Synchronization
import TUFFDecodeProtocol

public enum PrivateServingIPCError: Error, Sendable, Equatable, CustomStringConvertible {
    case hostAlreadyRunning
    case unsafeSocketPath
    case socket(String, Int32)
    case authentication
    case incompatibleProfile
    case incompatibleProtocol
    case incompatibleBuild(host: String)
    /// The running host is from an older build. `connectOrLaunch` asks it to
    /// shut down and starts this build's host.
    case olderBuild(host: String, pid: pid_t)
    case rejected(String)
    /// The host accepted the connection after committing to an idle exit. The
    /// client is not at fault: it relaunches the host and connects again.
    case hostRetiring

    public var description: String {
        switch self {
        case .hostAlreadyRunning: "A TUFF serving host already owns this profile"
        case .unsafeSocketPath: "The TUFF private socket path is unsafe"
        case .socket(let operation, let code): "Private serving socket \(operation) failed (errno \(code))"
        case .authentication: "Private serving socket authentication failed"
        case .incompatibleProfile: "Private serving host belongs to a different profile"
        case .incompatibleProtocol: "Private serving host uses an incompatible protocol"
        case .incompatibleBuild(let host):
            "A TUFF serving host from another build (\(host)) is running. Quit it or wait for it to go idle, then try again."
        case .olderBuild(let host, _):
            "A TUFF serving host from an older build (\(host)) is still shutting down. Try again in a moment."
        case .rejected(let reason): "The TUFF serving host refused this client: \(reason)"
        case .hostRetiring: "The TUFF serving host was exiting; reconnecting to a fresh one"
        }
    }
}

/// Identifies the command's origin: the client, and the one connection it
/// arrived on. A client that reconnects keeps its `clientID` and usually its
/// session IDs, so state scoped by `clientID` alone cannot tell an old
/// connection's cleanup from the new connection's work; `connectionID` can.
public struct ServingCommandContext: Sendable {
    public let clientID: UUID
    public let connectionID: UUID
    private let onAdmitted: @Sendable () -> Void

    public init(clientID: UUID, connectionID: UUID? = nil,
                onAdmitted: @escaping @Sendable () -> Void = {}) {
        self.clientID = clientID
        self.connectionID = connectionID ?? clientID
        self.onAdmitted = onAdmitted
    }

    /// The handler calls this once its ordering-relevant state for the
    /// command is registered (the request is active, the session is closing,
    /// the preparation is cancellable). The connection's dispatcher hands over
    /// the next command only after this, or after the command answered or
    /// returned. Idempotent.
    public func markAdmitted() { onAdmitted() }
}

/// One-shot admission signal for the connection dispatcher.
private final class ServingAdmissionGate: Sendable {
    private struct State {
        var isOpen = false
        var waiter: CheckedContinuation<Void, Never>?
    }
    private let state = Mutex(State())

    func open() {
        let waiter = state.withLock { current -> CheckedContinuation<Void, Never>? in
            current.isOpen = true
            defer { current.waiter = nil }
            return current.waiter
        }
        waiter?.resume()
    }

    func wait() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let ready = state.withLock { current -> Bool in
                if current.isOpen { return true }
                current.waiter = continuation
                return false
            }
            if ready { continuation.resume() }
        }
    }
}

/// One process owns this listener and the model coordinator behind it. HTTP
/// and GUI clients are adapters that submit to the same command handler.
public final class PrivateServingIPCServer: @unchecked Sendable {
    private static let maximumConnections = 32
    private static let handshakeTimeoutSeconds: Int32 = 10
    /// The handler must remain suspended until the command reaches a terminal
    /// event. This keeps request ownership registered for the full stream and
    /// prevents late events from being mistaken for an unowned request.
    public typealias CommandHandler = @Sendable (
        ServingCommandContext, ServingCommand, @escaping @Sendable (ServingEvent) -> Void
    ) async -> Void
    /// Coordinator-owned counters the server cannot know itself.
    public struct HostLoad: Sendable, Equatable {
        public var loadedModelIDs: [String]
        public var loadedModels: [ServingLoadedModel]
        public var activeRequests: Int
        public var openSessions: Int
        public init(loadedModelIDs: [String] = [], loadedModels: [ServingLoadedModel] = [],
                    activeRequests: Int = 0, openSessions: Int = 0) {
            self.loadedModelIDs = loadedModelIDs
            self.loadedModels = loadedModels
            self.activeRequests = activeRequests
            self.openSessions = openSessions
        }
    }
    public typealias StatusSnapshot = @Sendable () async -> HostLoad

    public let hostID = UUID()
    private let profile: ServingHostProfile
    private let token: String
    private let buildIdentity: String
    private let statusSnapshot: StatusSnapshot
    private let writeTimeout: TimeInterval
    private let commandHandler: CommandHandler
    // Internal synchronization seam for exercising publication ordering.
    var beforeEventSend: (@Sendable (ServingEvent) -> Void)?
    private let state = Mutex(State())

    /// Includes authenticated and handshaking peers so the private-host idle
    /// timer cannot shut down while a local client is completing its hello.
    public var connectedClientCount: Int {
        state.withLock { $0.clients.count }
    }

    private struct State {
        var listener: Int32?
        var lease: ServingHostLease?
        var clients: Set<Int32> = []
        var authenticatedClientIDs: Set<UUID> = []
        var apiClientIDs: Set<UUID> = []
        var stopping = false
        /// Set once an idle exit is committed. From then on every new
        /// connection is refused at the handshake.
        var retiring = false
    }

    private struct OwnedRequest {
        let sessionID: UUID
        let modelID: String
    }

    private struct OwnedLifecycle {
        let modelID: String
        let cancellablePreparation: Bool
    }

    private struct ConnectionOwnership {
        var requests: [UUID: OwnedRequest] = [:]
        var sessions: Set<UUID> = []
        var activeSessionRequests: [UUID: UUID] = [:]
        var lifecycleOperations: [UUID: OwnedLifecycle] = [:]
    }

    private final class ConnectionState: Sendable {
        private let mutex = Mutex(ConnectionOwnership())
        func withLock<R: Sendable>(_ body: (inout ConnectionOwnership) -> R) -> R {
            mutex.withLock { body(&$0) }
        }
    }

    public init(profile: ServingHostProfile, token: String,
                buildIdentity: String,
                writeTimeout: TimeInterval = 5,
                statusSnapshot: @escaping StatusSnapshot = { HostLoad() },
                commandHandler: @escaping CommandHandler) {
        self.profile = profile
        self.token = token
        self.buildIdentity = buildIdentity
        self.statusSnapshot = statusSnapshot
        self.writeTimeout = max(0.01, writeTimeout)
        self.commandHandler = commandHandler
    }

    /// Returns only after this process has acquired the profile lease and
    /// bound the socket. A competing foreground/launchd/CLI process must
    /// connect to the existing owner instead of creating another model pool.
    public func start() throws {
        guard state.withLock({ $0.listener == nil && !$0.stopping }) else {
            throw PrivateServingIPCError.socket("start", EALREADY)
        }
        try profile.prepare()
        guard let lease = try profile.acquireHostLease() else {
            throw PrivateServingIPCError.hostAlreadyRunning
        }
        let listener = try Self.makeListener(at: profile.socketURL)
        let thread = Thread { [weak self] in self?.acceptLoop(listener) }
        thread.name = "TUFF.PrivateServing.Accept"
        thread.qualityOfService = .userInitiated
        let started = state.withLock { current -> Bool in
            guard current.listener == nil, !current.stopping else { return false }
            current.listener = listener
            current.lease = lease
            return true
        }
        guard started else {
            close(listener)
            _ = unlink(profile.socketURL.path)
            lease.release()
            throw PrivateServingIPCError.socket("start", EALREADY)
        }
        thread.start()
    }

    /// Commits an idle host to exiting, atomically with connection admission.
    /// Accepting a connection registers it under the same lock, so either a
    /// connection (even one still handshaking) is already counted and this
    /// returns false, or it is accepted after this and refused with
    /// `ServingProtocol.hostRetiringReason`. Callers shut the host down after
    /// a true result.
    public func retireIfIdle() -> Bool {
        state.withLock { current in
            guard !current.stopping, !current.retiring, current.clients.isEmpty else { return false }
            current.retiring = true
            return true
        }
    }

    public func shutdown() {
        let old = state.withLock { current -> (Int32?, [Int32], ServingHostLease?) in
            guard !current.stopping,
                  current.listener != nil || current.lease != nil else {
                return (nil, [], nil)
            }
            current.stopping = true
            let result = (current.listener, Array(current.clients), current.lease)
            current.listener = nil
            current.clients.removeAll()
            current.lease = nil
            return result
        }
        if let listener = old.0 {
            _ = Darwin.shutdown(listener, SHUT_RDWR)
            close(listener)
        }
        for client in old.1 {
            _ = Darwin.shutdown(client, SHUT_RDWR)
        }
        _ = unlink(profile.socketURL.path)
        old.2?.release()
    }

    private func acceptLoop(_ listener: Int32) {
        while !state.withLock({ $0.stopping }) {
            let client = accept(listener, nil, nil)
            if client < 0 {
                if errno == EINTR { continue }
                if state.withLock({ $0.stopping }) { break }
                usleep(20_000)
                continue
            }
            let accepted = state.withLock { current -> Bool? in
                guard !current.stopping, current.clients.count < Self.maximumConnections else {
                    return nil
                }
                current.clients.insert(client)
                return current.retiring
            }
            guard let retiring = accepted else {
                close(client)
                if state.withLock({ $0.stopping }) { break }
                continue
            }
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.serve(client, retiring: retiring)
            }
        }
    }

    private func serve(_ descriptor: Int32, retiring: Bool) {
        let writer = ServingSocketWriter(descriptor: descriptor, timeout: writeTimeout)
        var authenticatedClientID: UUID?
        defer {
            state.withLock { current in
                current.clients.remove(descriptor)
                if let authenticatedClientID {
                    current.authenticatedClientIDs.remove(authenticatedClientID)
                    current.apiClientIDs.remove(authenticatedClientID)
                }
            }
            writer.close()
        }
        var peerUID: uid_t = 0
        var peerGID: gid_t = 0
        guard getpeereid(descriptor, &peerUID, &peerGID) == 0,
              peerUID == getuid() else {
            try? writer.send(.rejected("peer UID is not authorized"))
            return
        }
        let input = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        do {
            try Self.setReceiveTimeout(descriptor, seconds: Self.handshakeTimeoutSeconds)
            let first = try DecodeFrameCodec.read(ServingIPCMessage.self, from: input)
            guard case .clientHello(let hello) = first else {
                try writer.send(.rejected("client hello is required"))
                return
            }
            guard hello.protocolVersion == ServingProtocol.version else {
                try writer.send(.rejected("unsupported protocol version"))
                return
            }
            guard hello.profileIdentity == profile.profileIdentity else {
                try writer.send(.rejected("profile identity mismatch"))
                return
            }
            guard Self.constantTimeEqual(hello.token, token) else {
                try writer.send(.rejected("authentication failed"))
                return
            }
            // The idle exit was committed before this connection was accepted
            // (see `retireIfIdle`). Refuse it distinctly so the client
            // relaunches the host instead of treating this as a failure.
            guard !retiring else {
                try writer.send(.rejected(ServingProtocol.hostRetiringReason))
                return
            }
            let clientClaimed = state.withLock { current -> Bool in
                guard !current.authenticatedClientIDs.contains(hello.clientID) else { return false }
                current.authenticatedClientIDs.insert(hello.clientID)
                if hello.role == .api { current.apiClientIDs.insert(hello.clientID) }
                return true
            }
            guard clientClaimed else {
                try writer.send(.rejected("client identity already has an active connection"))
                return
            }
            authenticatedClientID = hello.clientID
            // The connection cap bounds idle peers. Keep an authenticated
            // client alive for arbitrarily long generations and chat pauses.
            try Self.setReceiveTimeout(descriptor, seconds: 0)
            try writer.send(.hostHello(ServingHostHello(
                profileIdentity: profile.profileIdentity, hostID: hostID,
                buildIdentity: buildIdentity)))

            let clientID = hello.clientID
            // Every command a connection runs, and the cleanup after it
            // closes, is scoped to this identity. A reconnecting client
            // reuses its client and session IDs, so the old connection's
            // cleanup must never be able to name the new connection's state.
            let connectionID = UUID()
            let ownership = ConnectionState()
            let emit = makeEmit(clientID: clientID, ownership: ownership, writer: writer)
            let (commands, commandInput) = AsyncStream.makeStream(of: ServingCommand.self)
            defer { commandInput.finish() }
            dispatchCommands(commands, clientID: clientID, connectionID: connectionID,
                             ownership: ownership, emit: emit)

            while !state.withLock({ $0.stopping }) {
                let message: ServingIPCMessage
                do { message = try DecodeFrameCodec.read(ServingIPCMessage.self, from: input) }
                catch { break }
                guard case .command(let command) = message else {
                    try writer.send(.rejected("command frame required"))
                    break
                }
                do { try command.validate() }
                catch {
                    // Correlated to the one operation that caused it, so it
                    // can never fail other streams sharing this connection.
                    if let refusal = command.refusal(
                        clientID: clientID, message: String(describing: error)) {
                        try writer.send(.event(refusal))
                    }
                    continue
                }
                if case .hostStatus(let operationID) = command {
                    let snapshot = statusSnapshot
                    let hostID = hostID, buildIdentity = buildIdentity
                    Task.detached { [weak self] in
                        let load = await snapshot()
                        guard let self else { return }
                        let status = self.state.withLock { current in
                            ServingHostStatus(
                                hostID: hostID, buildIdentity: buildIdentity,
                                apiAdapterAttached: !current.apiClientIDs.isEmpty,
                                connectedClients: current.clients.count,
                                loadedModelIDs: load.loadedModelIDs,
                                loadedModels: load.loadedModels,
                                activeRequests: load.activeRequests,
                                openSessions: load.openSessions)
                        }
                        do { try writer.send(.event(.hostStatus(operationID: operationID, status: status))) }
                        catch { writer.abort() }
                    }
                    continue
                }
                let admitted = ownership.withLock { owned -> Bool in
                    switch command {
                    case .prepare(let preparation):
                        guard owned.lifecycleOperations[preparation.operationID] == nil else { return false }
                        owned.lifecycleOperations[preparation.operationID] = OwnedLifecycle(
                            modelID: preparation.modelID, cancellablePreparation: true)
                    case .cancelPreparation(let operationID):
                        guard owned.lifecycleOperations[operationID]?.cancellablePreparation == true
                        else { return false }
                    case .unloadModel(let operationID, let modelID):
                        guard owned.lifecycleOperations[operationID] == nil else { return false }
                        owned.lifecycleOperations[operationID] = OwnedLifecycle(
                            modelID: modelID, cancellablePreparation: false)
                    case .start(let request):
                        guard owned.requests[request.requestID] == nil,
                              owned.activeSessionRequests[request.sessionID] == nil else { return false }
                        owned.requests[request.requestID] = OwnedRequest(
                            sessionID: request.sessionID, modelID: request.modelID)
                        owned.activeSessionRequests[request.sessionID] = request.requestID
                        owned.sessions.insert(request.sessionID)
                    case .cancel(let sessionID, let requestID):
                        guard owned.requests[requestID]?.sessionID == sessionID,
                              owned.activeSessionRequests[sessionID] == requestID else { return false }
                    case .endSession(let sessionID):
                        guard owned.sessions.contains(sessionID) else { return false }
                    case .hostStatus: return false
                    }
                    return true
                }
                guard admitted else {
                    // Rejections are correlated to the one operation that
                    // caused them. An uncorrelated failure would fail every
                    // stream sharing this connection, so stale cancels are
                    // dropped and a repeated live ID never answers over the
                    // stream that legitimately owns it.
                    switch command {
                    case .start(let request):
                        let known = ownership.withLock { $0.requests[request.requestID] != nil }
                        if !known {
                            try writer.send(.event(.failed(
                                identity: ServingRequestIdentity(clientID: clientID, request: request),
                                message: "This chat already has a running request")))
                        }
                    case .endSession(let sessionID):
                        // Nothing was opened on this connection, so the
                        // session is already closed.
                        try writer.send(.event(.sessionEnded(sessionID: sessionID)))
                    default:
                        break
                    }
                    continue
                }
                if case .start(let request) = command {
                    do { try request.validate() }
                    catch {
                        let identity = ServingRequestIdentity(clientID: clientID, request: request)
                        ownership.withLock {
                            $0.requests.removeValue(forKey: request.requestID)
                            if $0.activeSessionRequests[request.sessionID] == request.requestID {
                                $0.activeSessionRequests.removeValue(forKey: request.sessionID)
                            }
                        }
                        try writer.send(.event(.failed(identity: identity,
                            message: String(describing: error))))
                        continue
                    }
                }
                commandInput.yield(command)
            }
        } catch {
            // A malformed, oversized, truncated or incompatible peer is
            // dropped without invoking any model operation.
        }
    }

    /// Builds the connection's event sink. An event is published only while
    /// the connection still owns the operation it names; terminal events
    /// release that ownership before they are written, so a peer that has
    /// seen the terminal can immediately reuse the request or session.
    private func makeEmit(clientID: UUID, ownership: ConnectionState,
                          writer: ServingSocketWriter) -> @Sendable (ServingEvent) -> Void {
        let beforeEventSend = self.beforeEventSend
        return { event in
            guard (try? event.validate()) != nil else { return }
            let ownsIdentity = ownership.withLock { owned -> Bool in
                switch event {
                case .lifecycle(let operationID, let modelID, let phase):
                    guard owned.lifecycleOperations[operationID]?.modelID == modelID
                    else { return false }
                    switch phase {
                    case .ready(_), .cancelled, .unloaded, .failed(_):
                        owned.lifecycleOperations.removeValue(forKey: operationID)
                    case .loading(_), .cancelling, .unloading: break
                    }
                    return true
                case .queue(let identity, _, _), .generation(let identity, _),
                     .failed(let identity?, _):
                    guard identity.clientID == clientID,
                          let request = owned.requests[identity.requestID],
                          request.sessionID == identity.sessionID,
                          request.modelID == identity.modelID else { return false }
                    let terminal: Bool
                    switch event {
                    case .generation(_, let generation):
                        terminal = generation.kind == .finished
                            || generation.kind == .failed || generation.kind == .cancelled
                    case .failed: terminal = true
                    default: terminal = false
                    }
                    if terminal {
                        owned.requests.removeValue(forKey: identity.requestID)
                        if owned.activeSessionRequests[identity.sessionID] == identity.requestID {
                            owned.activeSessionRequests.removeValue(forKey: identity.sessionID)
                        }
                    }
                    return true
                case .failed(nil, _): return true
                case .hostStatus: return false
                case .sessionEndFailed(let sessionID, _):
                    return owned.sessions.contains(sessionID)
                case .sessionEnded(let sessionID):
                    guard owned.sessions.remove(sessionID) != nil else { return false }
                    owned.requests = owned.requests.filter { $0.value.sessionID != sessionID }
                    owned.activeSessionRequests.removeValue(forKey: sessionID)
                    return true
                }
            }
            guard ownsIdentity else { return }
            beforeEventSend?(event)
            do { try writer.send(.event(event)) }
            catch { writer.abort(); return }
        }
    }

    /// The connection's single serial dispatcher. Commands reach the handler
    /// in the order the peer sent them: the next command is handed over only
    /// once the previous one has been admitted (its ownership registered, or
    /// it has answered or returned). A `start` therefore always precedes its
    /// `cancel`, and an `endSession` precedes a later `start` of that session,
    /// without the peer re-sending anything. When the stream ends (the peer
    /// closed), everything the connection still owns is cancelled and ended.
    private func dispatchCommands(
        _ commands: AsyncStream<ServingCommand>, clientID: UUID, connectionID: UUID,
        ownership: ConnectionState, emit: @escaping @Sendable (ServingEvent) -> Void
    ) {
        let handler = commandHandler
        Task.detached(priority: .userInitiated) {
            for await command in commands {
                let gate = ServingAdmissionGate()
                let context = ServingCommandContext(
                    clientID: clientID, connectionID: connectionID, onAdmitted: { gate.open() })
                Task.detached(priority: .userInitiated) {
                    await handler(context, command) { event in
                        gate.open()
                        emit(event)
                    }
                    if case .start(let request) = command {
                        ownership.withLock {
                            $0.requests.removeValue(forKey: request.requestID)
                            if $0.activeSessionRequests[request.sessionID] == request.requestID {
                                $0.activeSessionRequests.removeValue(forKey: request.sessionID)
                            }
                        }
                    }
                    gate.open()
                }
                await gate.wait()
            }
            let closing = ServingCommandContext(clientID: clientID, connectionID: connectionID)
            let (activeIDs, openSessions, activeOperations) = ownership.withLock {
                ($0.requests, $0.sessions, $0.lifecycleOperations)
            }
            for (requestID, ownedRequest) in activeIDs {
                await handler(closing,
                    .cancel(sessionID: ownedRequest.sessionID, requestID: requestID)) { _ in }
            }
            for sessionID in openSessions {
                await handler(closing, .endSession(sessionID: sessionID)) { _ in }
            }
            for (operationID, operation) in activeOperations
                where operation.cancellablePreparation {
                await handler(closing, .cancelPreparation(operationID: operationID)) { _ in }
            }
        }
    }

    private static func constantTimeEqual(_ first: String, _ second: String) -> Bool {
        let lhs = Array(first.utf8), rhs = Array(second.utf8)
        guard lhs.count == rhs.count else { return false }
        var difference: UInt8 = 0
        for (left, right) in zip(lhs, rhs) { difference |= left ^ right }
        return difference == 0
    }

    private static func makeListener(at url: URL) throws -> Int32 {
        let path = url.path
        guard path.utf8.count < MemoryLayout<sockaddr_un>.size else {
            throw ServingHostProfileError.socketPathTooLong
        }
        var existing = stat()
        if path.withCString({ lstat($0, &existing) }) == 0 {
            guard (existing.st_mode & S_IFMT) == S_IFSOCK,
                  existing.st_uid == getuid() else {
                throw PrivateServingIPCError.unsafeSocketPath
            }
            guard path.withCString({ unlink($0) }) == 0 else {
                throw PrivateServingIPCError.socket("remove stale path", errno)
            }
        } else if errno != ENOENT {
            throw PrivateServingIPCError.socket("inspect path", errno)
        }
        let descriptor = socket(AF_UNIX, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw PrivateServingIPCError.socket("create", errno) }
        var noSigPipe: Int32 = 1
        _ = setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe,
                       socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_un()
        address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        address.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8) + [0]
        withUnsafeMutableBytes(of: &address.sun_path) { storage in
            storage.copyBytes(from: bytes)
        }
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bound == 0 else {
            let code = errno; close(descriptor)
            throw PrivateServingIPCError.socket("bind", code)
        }
        guard chmod(path, mode_t(0o600)) == 0, listen(descriptor, 16) == 0 else {
            let code = errno; close(descriptor); _ = unlink(path)
            throw PrivateServingIPCError.socket("listen", code)
        }
        return descriptor
    }

    private static func setReceiveTimeout(_ descriptor: Int32, seconds: Int32) throws {
        var timeout = timeval(tv_sec: Int(seconds), tv_usec: 0)
        guard setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                         socklen_t(MemoryLayout<timeval>.size)) == 0 else {
            throw PrivateServingIPCError.socket("set read timeout", errno)
        }
    }
}

private final class ServingSocketWriter: @unchecked Sendable {
    private let lock = NSLock()
    private let descriptor: Int32
    private let timeout: TimeInterval
    private var closed = false
    private var descriptorClosed = false

    init(descriptor: Int32, timeout: TimeInterval = 5) {
        self.descriptor = descriptor
        self.timeout = max(0.01, timeout)
    }

    func send(_ message: ServingIPCMessage) throws {
        let frame = try DecodeFrameCodec.encode(message)
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { throw PrivateServingIPCError.socket("write", EBADF) }
        let deadline = ProcessInfo.processInfo.systemUptime + timeout
        var offset = 0
        while offset < frame.count {
            let sent = frame.withUnsafeBytes { bytes in
                Darwin.send(descriptor, bytes.baseAddress!.advanced(by: offset),
                            min(64 * 1024, frame.count - offset), MSG_DONTWAIT)
            }
            if sent > 0 { offset += sent; continue }
            if sent < 0, errno == EINTR { continue }
            if sent < 0, (errno == EAGAIN || errno == EWOULDBLOCK) {
                let remaining = deadline - ProcessInfo.processInfo.systemUptime
                guard remaining > 0 else {
                    throw PrivateServingIPCError.socket("write timeout", ETIMEDOUT)
                }
                var pollDescriptor = pollfd(fd: descriptor, events: Int16(POLLOUT), revents: 0)
                let milliseconds = Int32(min(5_000, max(1, Int(remaining * 1_000))))
                let ready = Darwin.poll(&pollDescriptor, 1, milliseconds)
                if ready < 0, errno == EINTR { continue }
                if ready > 0 { continue }
                if ready == 0 { throw PrivateServingIPCError.socket("write timeout", ETIMEDOUT) }
            }
            throw PrivateServingIPCError.socket("write", errno)
        }
    }

    func close() {
        lock.lock()
        defer { lock.unlock() }
        guard !descriptorClosed else { return }
        descriptorClosed = true
        if !closed {
            closed = true
            _ = Darwin.shutdown(descriptor, SHUT_RDWR)
        }
        Darwin.close(descriptor)
    }

    func abort() {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        closed = true
        _ = Darwin.shutdown(descriptor, SHUT_RDWR)
    }
}

/// Connects only to a socket owned by the current user. The caller still has
/// to complete the authenticated profile/version handshake before sending a
/// command.
public enum PrivateServingIPCClientSocket {
    public static func connect(to url: URL) throws -> Int32 {
        let path = url.path
        guard path.utf8.count < MemoryLayout<sockaddr_un>.size else {
            throw ServingHostProfileError.socketPathTooLong
        }
        var info = stat()
        guard path.withCString({ lstat($0, &info) }) == 0,
              (info.st_mode & S_IFMT) == S_IFSOCK,
              info.st_uid == getuid(), (info.st_mode & 0o077) == 0 else {
            throw PrivateServingIPCError.unsafeSocketPath
        }
        let descriptor = socket(AF_UNIX, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw PrivateServingIPCError.socket("create", errno) }
        var noSigPipe: Int32 = 1
        _ = setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe,
                       socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_un()
        address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        address.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8) + [0]
        withUnsafeMutableBytes(of: &address.sun_path) { $0.copyBytes(from: bytes) }
        let connected = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connected == 0 else {
            let code = errno; close(descriptor)
            throw PrivateServingIPCError.socket("connect", code)
        }
        var timeout = timeval(tv_sec: 10, tv_usec: 0)
        guard setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                         socklen_t(MemoryLayout<timeval>.size)) == 0 else {
            let code = errno; close(descriptor)
            throw PrivateServingIPCError.socket("set handshake timeout", code)
        }
        var uid: uid_t = 0
        var gid: gid_t = 0
        guard getpeereid(descriptor, &uid, &gid) == 0, uid == getuid() else {
            close(descriptor)
            throw PrivateServingIPCError.authentication
        }
        return descriptor
    }
}

/// Authenticated connection to the one serving host for a profile. Callers
/// should keep one reader task for the socket and dispatch returned events by
/// their request/session IDs; multiple requests share this connection.
public final class PrivateServingIPCConnection: @unchecked Sendable {
    public let host: ServingHostHello
    private let descriptor: Int32
    private let input: FileHandle
    private let writer: ServingSocketWriter

    private init(descriptor: Int32, host: ServingHostHello) {
        self.descriptor = descriptor
        self.input = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        self.writer = ServingSocketWriter(descriptor: descriptor)
        self.host = host
    }

    public static func connect(profile: ServingHostProfile, clientID: UUID,
                               role: ServingClientRole = .tool,
                               expectedBuildIdentity: String? = nil) throws
        -> PrivateServingIPCConnection {
        try profile.prepare()
        let descriptor = try PrivateServingIPCClientSocket.connect(to: profile.socketURL)
        let writer = ServingSocketWriter(descriptor: descriptor)
        do {
            try writer.send(.clientHello(ServingClientHello(
                profileIdentity: profile.profileIdentity, clientID: clientID,
                token: try profile.token(), role: role)))
            let input = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
            let response = try DecodeFrameCodec.read(ServingIPCMessage.self, from: input)
            if case .rejected(let reason) = response {
                writer.close()
                throw reason == ServingProtocol.hostRetiringReason
                    ? PrivateServingIPCError.hostRetiring
                    : PrivateServingIPCError.rejected(reason)
            }
            guard case .hostHello(let host) = response,
                  host.protocolVersion == ServingProtocol.version,
                  host.profileIdentity == profile.profileIdentity else {
                writer.close()
                throw PrivateServingIPCError.authentication
            }
            if let expected = expectedBuildIdentity, expected != host.buildIdentity {
                let pid = ServingHostTakeover.peerPID(descriptor)
                writer.close()
                if let pid, ServingHostTakeover.isOlder(host.buildIdentity, than: expected) {
                    throw PrivateServingIPCError.olderBuild(host: host.buildIdentity, pid: pid)
                }
                throw PrivateServingIPCError.incompatibleBuild(host: host.buildIdentity)
            }
            // Keep the authenticated socket open through long generations and
            // while the app is idle; server admission bounds peer count.
            var noTimeout = timeval(tv_sec: 0, tv_usec: 0)
            _ = setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &noTimeout,
                           socklen_t(MemoryLayout<timeval>.size))
            return PrivateServingIPCConnection(descriptor: descriptor, host: host)
        } catch {
            writer.close()
            throw error
        }
    }

    public func send(_ command: ServingCommand) throws {
        try command.validate()
        try writer.send(.command(command))
    }

    public func receive() throws -> ServingEvent {
        guard case .event(let event) = try DecodeFrameCodec.read(
            ServingIPCMessage.self, from: input) else {
            throw PrivateServingIPCError.authentication
        }
        try event.validate()
        return event
    }

    public func close() { writer.close() }
    deinit { close() }
}
