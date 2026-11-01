import Foundation
import TUFFDecodeProtocol

/// A handle owns one family-scoped helper and its one loaded model. Its
/// implementation resolves only host-registered model directories.
public protocol ServingModelHandle: Sendable {
    func prepare(_ request: ServingModelPreparation,
                 progress: @escaping @Sendable (ServingLoadState) -> Void) async throws
    func start(_ request: DecodeGenerationRequest, sessionID: UUID)
        -> AsyncThrowingStream<DecodeServiceEvent, Error>
    func cancel(requestID: UUID)
    func cancelWarmup()
    func endSession(_ sessionID: UUID, requestID: UUID) async throws
    func unload() async
}
/// A helper's transport has failed, rather than a model request failing normally.
public struct ServingHelperTransportLost: Error, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
}


/// Multiplexes one handle per model and keeps request/session identities
/// scoped to the authenticated client. Model and peer allocations are
/// admitted by the helpers against the profile's common residency registry.
public actor ServingModelCoordinator {
    public typealias Resolver = @Sendable (String) async throws -> any ServingModelHandle
    public typealias Emit = @Sendable (ServingEvent) -> Void

    /// Sessions and requests are scoped to the connection that opened them,
    /// not just the client. A reconnecting client reuses its client and session
    /// IDs, so the previous connection's cleanup can only name its own state.
    private struct SessionKey: Hashable, Sendable {
        let connection: UUID
        let model: String
        let session: UUID
    }
    private struct RequestKey: Hashable, Sendable {
        let connection: UUID
        let request: UUID
    }
    private struct Active: Sendable {
        let session: SessionKey
        let helperRequestID: UUID
        let identity: ServingRequestIdentity
        let task: Task<Void, Never>
    }
    private let resolve: Resolver
    private let maxModels: Int
    private let maxSessions: Int
    private let maxActiveRequests: Int
    private var handles: [String: any ServingModelHandle] = [:]
    private var configurations: [String: ServingModelPreparation] = [:]
    private var loading: Set<String> = []
    private var modelPreparations: [String: Task<Void, Never>] = [:]
    private var handleIDs: [String: UUID] = [:]
    private var unloading: Set<String> = []
    private var unloadWaiters: [CheckedContinuation<Void, Never>] = []
    private var sessions: [SessionKey: UUID] = [:]
    private var ending: [SessionKey: Task<Bool, Never>] = [:]
    private var active: [RequestKey: Active] = [:]
    private var preparations: [RequestKey: Task<Void, Never>] = [:]
    private var stopping = false
    private var shutdownTask: Task<Void, Never>?

    public init(maxModels: Int = 4, maxSessions: Int = 32, maxActiveRequests: Int = 16,
                resolve: @escaping Resolver) {
        self.maxModels = max(1, maxModels)
        self.maxSessions = max(1, maxSessions)
        self.maxActiveRequests = max(1, maxActiveRequests)
        self.resolve = resolve
    }

    /// Counters for the live host status.
    public func snapshot() -> PrivateServingIPCServer.HostLoad {
        .init(loadedModelIDs: handles.keys.sorted(),
              loadedModels: handles.keys.sorted().compactMap { id in
                  configurations[id].map {
                      ServingLoadedModel(modelID: id, maxContextTokens: $0.maxContextTokens,
                                         runtimeOptions: $0.runtimeOptions,
                                         forceLogitsHead: $0.forceLogitsHead)
                  }
              },
              activeRequests: active.count, openSessions: sessions.count)
    }

    /// True while any request, load, unload or session cleanup is in flight.
    public var isBusy: Bool {
        !active.isEmpty || !loading.isEmpty || !unloading.isEmpty || !ending.isEmpty
    }

    public func handle(clientID: UUID, command: ServingCommand,
                       emit: @escaping Emit) async {
        await handle(context: ServingCommandContext(clientID: clientID),
                     command: command, emit: emit)
    }

    public func handle(context: ServingCommandContext, command: ServingCommand,
                       emit: @escaping Emit) async {
        let clientID = context.clientID
        let connection = context.connectionID
        // Refusals answer the one operation that caused them; an uncorrelated
        // failure would fail every stream sharing the connection.
        func refuse(_ message: String) {
            if let event = command.refusal(clientID: clientID, message: message) { emit(event) }
        }
        guard !stopping else { refuse("Serving host is shutting down"); return }
        do { try command.validate() }
        catch { refuse("\(error)"); return }
        switch command {
        case .prepare(let request):
            let key = RequestKey(connection: connection, request: request.operationID)
            guard preparations[key] == nil,
                  !unloading.contains(request.modelID) else {
                emit(.lifecycle(operationID: request.operationID, modelID: request.modelID,
                                state: .failed("Model lifecycle operation is already in progress")))
                return
            }
            if let pending = modelPreparations[request.modelID] {
                emit(.lifecycle(operationID: request.operationID, modelID: request.modelID,
                                state: .loading(.preparingRunner)))
                context.markAdmitted()
                await pending.value
                guard !stopping else {
                    emit(.lifecycle(operationID: request.operationID, modelID: request.modelID,
                                    state: .cancelled))
                    return
                }
                guard handles[request.modelID] != nil else {
                    emit(.lifecycle(operationID: request.operationID, modelID: request.modelID,
                                    state: .failed("Model preparation failed")))
                    return
                }
            }
            if handles[request.modelID] != nil {
                let existing = configurations[request.modelID]
                let compatible = existing?.maxContextTokens == request.maxContextTokens
                    && existing?.runtimeOptions == request.runtimeOptions
                    && existing?.forceLogitsHead == request.forceLogitsHead
                emit(.lifecycle(operationID: request.operationID, modelID: request.modelID,
                    state: compatible ? .ready(loadSeconds: 0)
                        : .failed("Unload the model before changing its runtime configuration")))
                return
            }
            guard handles.count + loading.count < maxModels else {
                emit(.lifecycle(operationID: request.operationID, modelID: request.modelID,
                                state: .failed("Loaded model limit reached")))
                return
            }
            // A new model may begin its own best-effort warmup after loading.
            // Stop existing warmups first so idle models do not concurrently
            // compete for aggregate page-cache and memory bandwidth.
            for handle in handles.values { handle.cancelWarmup() }
            loading.insert(request.modelID)
            let task = Task { [self] in await prepare(request, key: key, emit: emit) }
            preparations[key] = task
            modelPreparations[request.modelID] = task
            context.markAdmitted()
            await task.value
        case .cancelPreparation(let id):
            preparations[RequestKey(connection: connection, request: id)]?.cancel()
        case .unloadModel(let operationID, let modelID):
            guard let handle = handles[modelID], !loading.contains(modelID),
                  !unloading.contains(modelID),
                  !ending.keys.contains(where: { $0.model == modelID }),
                  !active.values.contains(where: { $0.session.model == modelID }) else {
                emit(.lifecycle(operationID: operationID, modelID: modelID,
                                state: .failed("Model is not loaded or has active requests")))
                return
            }
            unloading.insert(modelID)
            context.markAdmitted()
            emit(.lifecycle(operationID: operationID, modelID: modelID, state: .unloading))
            await handle.unload()
            handles.removeValue(forKey: modelID)
            configurations.removeValue(forKey: modelID)
            handleIDs.removeValue(forKey: modelID)
            sessions = sessions.filter { $0.key.model != modelID }
            finishUnload(modelID)
            emit(.lifecycle(operationID: operationID, modelID: modelID, state: .unloaded))
        case .start(let request):
            let session = SessionKey(connection: connection, model: request.modelID,
                                     session: request.sessionID)
            // A session that is being closed (an explicit end, or an idle
            // reclaim) finishes first; the turn then starts on a fresh one
            // instead of failing.
            while let closing = ending[session] { _ = await closing.value }
            if sessions[session] == nil, sessions.count >= maxSessions {
                _ = await reclaimIdleSessions(keeping: session)
            }
            if let task = start(context: context, request: request, emit: emit) {
                await task.value
            }
        case .hostStatus:
            // Answered by the IPC server, which owns connection state.
            break
        case .cancel(let sessionID, let requestID):
            let key = RequestKey(connection: connection, request: requestID)
            if let entry = active[key], entry.session.session == sessionID {
                handles[entry.session.model]?.cancel(requestID: entry.helperRequestID)
            }
        case .endSession(let sessionID):
            // Every close is registered before any is awaited so a command
            // behind this one on the connection already sees the session as
            // closing.
            let matches = sessions.keys.filter {
                $0.connection == connection && $0.session == sessionID
            }
            var waits: [Task<Bool, Never>] = []
            for key in matches {
                if let existing = ending[key] { waits.append(existing); continue }
                let task = Task { [self] in await closeSession(key) }
                ending[key] = task
                waits.append(task)
            }
            context.markAdmitted()
            var closed = true
            for task in waits { if !(await task.value) { closed = false } }
            if closed { emit(.sessionEnded(sessionID: sessionID)) }
            else {
                emit(.sessionEndFailed(sessionID: sessionID,
                    message: "Session cleanup failed; its reservation remains active"))
            }
        }
    }

    private func prepare(_ request: ServingModelPreparation, key: RequestKey,
                         emit: @escaping Emit) async {
        var handle: (any ServingModelHandle)?
        do {
            let resolved = try await resolve(request.modelID)
            handle = resolved
            try Task.checkCancellation()
            let report: @Sendable (ServingLoadState) -> Void = { state in
                if case .loading = state {
                    emit(.lifecycle(operationID: request.operationID, modelID: request.modelID, state: state))
                }
            }
            do {
                try await resolved.prepare(request, progress: report)
            } catch where ServingMemoryPressure.isMemoryBusy(error) {
                // Another resident model holds the memory this one needs.
                // Idle models yield; models with active or closing work never do.
                try Task.checkCancellation()
                guard !(await evictIdleModels()).isEmpty else { throw error }
                try Task.checkCancellation()
                try await resolved.prepare(request, progress: report)
            }
            try Task.checkCancellation()
            handles[request.modelID] = resolved
            handleIDs[request.modelID] = UUID()
            configurations[request.modelID] = request
            if !active.isEmpty || loading.count > 1 { resolved.cancelWarmup() }
            emit(.lifecycle(operationID: request.operationID, modelID: request.modelID,
                            state: .ready(loadSeconds: 0)))
        } catch {
            await handle?.unload()
            emit(.lifecycle(operationID: request.operationID, modelID: request.modelID,
                            state: error is CancellationError ? .cancelled : .failed("\(error)")))
        }
        loading.remove(request.modelID)
        modelPreparations.removeValue(forKey: request.modelID)
        preparations.removeValue(forKey: key)
    }

    private func start(context: ServingCommandContext, request: ServingGenerationRequest,
                       emit: @escaping Emit) -> Task<Void, Never>? {
        let clientID = context.clientID
        let identity = ServingRequestIdentity(clientID: clientID, request: request)
        let key = RequestKey(connection: context.connectionID, request: request.requestID)
        let session = SessionKey(connection: context.connectionID, model: request.modelID,
                                 session: request.sessionID)
        guard let handle = handles[request.modelID], !unloading.contains(request.modelID),
              active.count < maxActiveRequests,
              (sessions[session] != nil || sessions.count < maxSessions),
              active[key] == nil, ending[session] == nil,
              !active.values.contains(where: { $0.session == session }) else {
            emit(.failed(identity: identity, message: "Model is unavailable or session is already active"))
            return nil
        }
        // Profile-wide warmup must yield whenever any foreground work starts.
        for value in handles.values { value.cancelWarmup() }
        let helperSession = sessions[session] ?? UUID()
        sessions[session] = helperSession
        let helperID = UUID()
        var payload = request.payload
        payload.generationID = helperID
        let handleID = handleIDs[request.modelID]
        let routedPayload = payload
        emit(.queue(identity: identity, state: .admitted, position: nil))
        let hasImages = request.payload.imageAttachments?.isEmpty == false
            || request.payload.history.contains { $0.images?.isEmpty == false }
        let task = Task { [self] in
            await Task.detached { [self] in
                var reclaimed = false
                attempts: while true {
                    do {
                        for try await var event in handle.start(routedPayload, sessionID: helperSession) {
                            guard event.generationID == helperID else { continue }
                            // Image work needs the whole memory budget. Before
                            // failing the request, release what only keeps other
                            // chats warm (idle sessions, idle other models); their
                            // history is replayed from the transcript when they next run.
                            if event.kind == .failed, !reclaimed,
                               ((hasImages && ServingMemoryPressure.isExclusiveBusy(event.error ?? ""))
                                || (event.error ?? "").contains(ServingMemoryPressure.phrase)) {
                                let sessionsReleased = await reclaimIdleSessions(keeping: session)
                                let modelsReleased = await evictIdleModels(excluding: request.modelID)
                                if sessionsReleased || !modelsReleased.isEmpty {
                                    reclaimed = true
                                    continue attempts
                                }
                            }
                            event.generationID = identity.requestID
                            if [.finished, .cancelled, .failed].contains(event.kind) {
                                await finish(key)
                                emit(.generation(identity: identity, event: event))
                                return
                            }
                            emit(.generation(identity: identity, event: event))
                        }
                        await finish(key)
                        emit(.failed(identity: identity, message: "Model stream ended without a terminal event"))
                    } catch {
                        if error is ServingHelperTransportLost {
                            await helperLost(modelID: request.modelID, handleID: handleID)
                        }
                        await finish(key)
                        emit(.failed(identity: identity, message: "\(error)"))
                    }
                    return
                }
            }.value
        }
        active[key] = Active(session: session, helperRequestID: helperID, identity: identity, task: task)
        context.markAdmitted()
        return task
    }

    private func finish(_ key: RequestKey) { active.removeValue(forKey: key) }

    private func helperLost(modelID: String, handleID: UUID?) {
        guard let handleID, handleIDs[modelID] == handleID else { return }
        let handle = handles.removeValue(forKey: modelID)
        handleIDs.removeValue(forKey: modelID)
        configurations.removeValue(forKey: modelID)
        sessions = sessions.filter { $0.key.model != modelID }
        for entry in active.values where entry.session.model == modelID {
            handle?.cancel(requestID: entry.helperRequestID)
        }
        if let handle { Task { await handle.unload() } }
    }

    /// Closes the sessions that are idle right now (other than `keeping`) and
    /// returns whether any memory was released. Idleness is decided in the
    /// same synchronous step that marks the session as closing, so a session
    /// that is running is never touched and no request can start on one that
    /// is closing (it waits for the close). This never cancels a request and
    /// never unloads a model: when nothing idle remains, the caller's
    /// memory-busy failure goes to that request alone.
    private func reclaimIdleSessions(keeping session: SessionKey) async -> Bool {
        var closing: [Task<Bool, Never>] = []
        for candidate in sessions.keys where candidate != session {
            guard ending[candidate] == nil,
                  !active.values.contains(where: { $0.session == candidate }) else { continue }
            let task = Task { [self] in await closeSession(candidate) }
            ending[candidate] = task
            closing.append(task)
        }
        var released = false
        for task in closing { if await task.value { released = true } }
        return released
    }

    private func closeSession(_ key: SessionKey) async -> Bool {
        let entries = active.values.filter { $0.session == key }
        for entry in entries { handles[key.model]?.cancel(requestID: entry.helperRequestID) }
        for entry in entries { await entry.task.value }
        if let handle = handles[key.model], let helperSession = sessions[key] {
            do { try await handle.endSession(helperSession, requestID: UUID()) }
            catch { ending.removeValue(forKey: key); return false }
        }
        sessions.removeValue(forKey: key)
        ending.removeValue(forKey: key)
        return true
    }

    /// Releases models with no active requests or session cleanup in progress.
    /// The host may call this when its idle policy or memory pressure requires it.
    @discardableResult
    public func evictIdleModels(excluding keep: String? = nil) async -> [String] {
        guard !stopping else { return [] }
        let idle = handles.keys.filter { id in
            id != keep && !loading.contains(id) && !unloading.contains(id)
                && !active.values.contains(where: { $0.session.model == id })
                && !ending.keys.contains(where: { $0.model == id })
        }.sorted()
        var evicted: [String] = []
        for id in idle {
            guard !stopping else { break }
            guard let handle = handles[id], !loading.contains(id), !unloading.contains(id),
                  !active.values.contains(where: { $0.session.model == id }),
                  !ending.keys.contains(where: { $0.model == id }) else { continue }
            unloading.insert(id)
            await handle.unload()
            handles.removeValue(forKey: id)
            handleIDs.removeValue(forKey: id)
            configurations.removeValue(forKey: id)
            sessions = sessions.filter { $0.key.model != id }
            finishUnload(id)
            evicted.append(id)
        }
        return evicted
    }

    /// Stops admission, cancels and drains all requests and pending loads,
    /// then releases all helpers. Concurrent shutdown callers share one drain.
    public func shutdown() async {
        if let shutdownTask { await shutdownTask.value; return }
        stopping = true
        let task = Task { [self] in await drainForShutdown() }
        shutdownTask = task
        await task.value
    }

    private func drainForShutdown() async {
        let pending = Array(preparations.values)
        for task in pending { task.cancel() }
        for task in pending { await task.value }
        let running = Array(active.values)
        for entry in running { handles[entry.session.model]?.cancel(requestID: entry.helperRequestID) }
        for entry in running { await entry.task.value }
        let closing = Array(ending.values)
        for task in closing { _ = await task.value }
        if !unloading.isEmpty {
            await withCheckedContinuation { unloadWaiters.append($0) }
        }
        let all = Array(handles.values)
        handles.removeAll()
        handleIDs.removeAll()
        configurations.removeAll()
        sessions.removeAll()
        for handle in all { await handle.unload() }
    }

    private func finishUnload(_ modelID: String) {
        unloading.remove(modelID)
        guard unloading.isEmpty else { return }
        let waiters = unloadWaiters
        unloadWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}
