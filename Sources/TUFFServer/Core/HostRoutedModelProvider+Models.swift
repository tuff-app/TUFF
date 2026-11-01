import Foundation
import TUFFAppCore
import TUFFDecodeProtocol
import TUFFModelCatalog
import TUFFRuntimeDescriptions

enum HostServingLog {
    static func write(_ message: String) {
        FileHandle.standardError.write(Data("[\(Date().formatted(.iso8601))] \(message)\n".utf8))
    }
}

extension HostRoutedModelProvider {
    // MARK: - Model residency

    /// The runtime configuration this adapter would load `descriptor` with:
    /// the same Auto plan `tuff serve` chooses.
    private func ownPreparation(for descriptor: TUFFModelDescriptor,
                                forceLogitsHead: Bool) throws
        -> (preparation: ServingModelPreparation, plan: AppAutomaticMemoryPlan) {
        guard let automatic = automaticPlan(descriptor) else {
            throw ServerRequestError.modelUnavailable(
                "No safe automatic allocation is available for this model on this Mac.")
        }
        let runtime = installed.runtimeConfiguration(for: descriptor, resolvedPlan: automatic)
        let options = AppRuntimeOptions(
            expertCacheSlots: runtime.expertCacheSlots, expertCachePolicy: .lfu,
            prefillEnabled: runtime.prefillPolicy != .off,
            prefillChunkTokens: runtime.prefillChunkTokens, rdadvisePolicy: .off,
            // The install was verified when it finished; a receipt check is
            // what the API has always loaded with.
            modelVerification: .trustedInstall, visionResidencyPolicy: .onDemand,
            denseResidencyPolicy: automatic.denseResidencyPolicy,
            retainedConversationBytes: automatic.retainedConversationBytes ?? 0)
        do {
            try options.validate()
            return (try ServingModelPreparation(
                modelID: descriptor.apiModelID, maxContextTokens: automatic.contextTokens,
                runtimeOptions: DecodeServiceInferenceClient.decodeRuntimeOptions(options),
                forceLogitsHead: forceLogitsHead), automatic)
        } catch let error as ServerRequestError {
            throw error
        } catch {
            throw ServerRequestError.modelUnavailable(
                "\(descriptor.displayName) has no valid serving configuration on this Mac: \(error)")
        }
    }

    private static func sameConfiguration(_ lhs: ServingModelPreparation,
                                          _ rhs: ServingLoadedModel) -> Bool {
        lhs.maxContextTokens == rhs.maxContextTokens && lhs.runtimeOptions == rhs.runtimeOptions
            && lhs.forceLogitsHead == rhs.forceLogitsHead
    }

    /// Makes `descriptor` resident in the host and returns the configuration
    /// the request must run under.
    ///
    /// A model the host already holds is used as it is. The API does not
    /// reload a model the app loaded just because its preferred context or
    /// cache settings differ; the request runs in the resident configuration
    /// and the host reports a context overflow if it does not fit. The one
    /// property the helper insists on matching is whether it has a sampling
    /// logits head (a sampled request needs it, a greedy one must not have
    /// it). When that differs the model is replaced, keeping the resident
    /// context and cache settings, but only while the host has no active work.
    func acquire(_ descriptor: TUFFModelDescriptor,
                 forceLogitsHead: Bool) async throws -> ServingModelPreparation {
        let modelID = descriptor.apiModelID
        await retryFailedSessionEnds()
        for attempt in 0..<2 {
            guard let status = try await link.status(launch: true) else {
                throw ServerRequestError.hostUnavailable("The TUFF inference host did not answer.")
            }
            let loaded = status.loadedModels.first { $0.modelID == modelID }
            var target: ServingModelPreparation
            var plan: AppAutomaticMemoryPlan?
            var adopted = false
            var weLoad = false
            if let loaded {
                if loaded.forceLogitsHead == forceLogitsHead {
                    target = try ServingModelPreparation(
                        modelID: modelID, maxContextTokens: loaded.maxContextTokens,
                        runtimeOptions: loaded.runtimeOptions, forceLogitsHead: forceLogitsHead)
                    adopted = true
                } else {
                    guard status.activeRequests == 0 else {
                        throw ServerRequestError.modelBusy(
                            "\(descriptor.displayName) is in use by another TUFF client in a mode this request cannot share. Retry when it is idle.")
                    }
                    guard try await link.unload(modelID: modelID) else {
                        throw ServerRequestError.modelBusy(
                            "\(descriptor.displayName) is in use by another TUFF client. Retry when it is idle.")
                    }
                    ServerLog.modelUnloaded(modelID)
                    target = try ServingModelPreparation(
                        modelID: modelID, maxContextTokens: loaded.maxContextTokens,
                        runtimeOptions: loaded.runtimeOptions, forceLogitsHead: forceLogitsHead)
                }
            } else {
                let own = try ownPreparation(for: descriptor, forceLogitsHead: forceLogitsHead)
                target = own.preparation
                plan = own.plan
                weLoad = true
            }
            state.withLock { $0.transition = weLoad ? "loading \(modelID)" : nil }
            defer { state.withLock { $0.transition = nil } }
            do {
                try await link.prepare(target)
            } catch ServerRequestError.modelBusy where attempt == 0 {
                // Another client reconfigured the model between the status
                // query and the request; look again.
                continue
            }
            state.withLock { current in
                if weLoad {
                    current.apiLoaded[modelID] = target
                    current.ranOwnPlan[modelID] = true
                } else if let loaded, adopted,
                          let mine = current.apiLoaded[modelID], Self.sameConfiguration(mine, loaded) {
                    current.ranOwnPlan[modelID] = true
                } else {
                    current.apiLoaded.removeValue(forKey: modelID)
                    current.ranOwnPlan[modelID] = false
                }
            }
            if weLoad, let plan {
                ServerLoadedPlans.set(plan, for: installed.directory(for: descriptor))
                ServerLog.modelLoaded(modelID)
            }
            return target
        }
        throw ServerRequestError.modelBusy(
            "\(descriptor.displayName) was reconfigured by another TUFF client while this request started. Retry the request.")
    }

    // MARK: - Sessions

    /// The host session a request runs in. A request with a prompt-cache key
    /// reuses that key's session so the host keeps its conversation state;
    /// any other request gets a private one.
    func openSession(model: String, cacheKey: String?) async -> (id: UUID, keyed: Bool) {
        guard let cacheKey, !cacheKey.isEmpty else { return (UUID(), false) }
        let composite = model + "\u{0}" + cacheKey
        var evicted: [UUID] = []
        let id = state.withLock { current -> UUID in
            if var existing = current.keyed[composite] {
                existing.lastUsed = Date()
                current.keyed[composite] = existing
                return existing.id
            }
            let id = UUID()
            current.keyed[composite] = KeyedSession(id: id, model: model, lastUsed: Date())
            while current.keyed.count > Self.maximumKeyedSessions,
                  let oldest = current.keyed.min(by: { $0.value.lastUsed < $1.value.lastUsed }),
                  oldest.key != composite {
                evicted.append(oldest.value.id)
                current.keyed.removeValue(forKey: oldest.key)
            }
            return id
        }
        for session in evicted { await endSession(session) }
        return (id, true)
    }

    /// A request's session after its generation is over. A keyed session
    /// whose generation completed stays open for the next request with the
    /// same key; every other session is ended.
    func closeSession(_ id: UUID, keyed: Bool, reuse: Bool) async {
        if keyed {
            if reuse {
                state.withLock { current in
                    for (key, value) in current.keyed where value.id == id {
                        var touched = value
                        touched.lastUsed = Date()
                        current.keyed[key] = touched
                    }
                }
                return
            }
            state.withLock { current in
                current.keyed = current.keyed.filter { $0.value.id != id }
            }
        }
        await endSession(id)
    }

    /// Ends a session in the host, retrying a failed cleanup. The host keeps
    /// the session's reservation on failure, so one that still fails is
    /// remembered and tried again before the next request.
    func endSession(_ id: UUID) async {
        // Cleanup runs after the request's task was cancelled; a cancelled
        // wait for the host's answer would report success it never saw.
        await Task.detached { [self] in
            for attempt in 0..<3 {
                do {
                    try await link.endSession(id)
                    state.withLock { _ = $0.failedSessionEnds.remove(id) }
                    return
                } catch {
                    HostServingLog.write("session cleanup failed (attempt \(attempt + 1)): \(error)")
                    if attempt < 2 { try? await Task.sleep(for: .milliseconds(100 << attempt)) }
                }
            }
            state.withLock { _ = $0.failedSessionEnds.insert(id) }
        }.value
    }

    func retryFailedSessionEnds() async {
        let failed = state.withLock { Array($0.failedSessionEnds) }
        for id in failed { await endSession(id) }
    }

    // MARK: - Release

    /// Ends this adapter's open sessions and unloads the models it loaded.
    /// The host refuses an unload while anything is active, so another
    /// client's work is never interrupted. True when nothing of this
    /// adapter's remains resident. The caller holds the request slot.
    func releaseIdleModels() async -> Bool {
        let sessions = state.withLock { current -> [UUID] in
            defer { current.keyed.removeAll() }
            return current.keyed.values.map(\.id)
        }
        for id in sessions { await endSession(id) }
        await retryFailedSessionEnds()
        let owned = state.withLock { Array($0.apiLoaded.keys) }
        guard !owned.isEmpty else { return true }
        let resident: Set<String>
        do {
            resident = Set(try await link.status(launch: false)?.loadedModelIDs ?? [])
        } catch {
            return false
        }
        var free = true
        for id in owned {
            guard resident.contains(id) else {
                state.withLock { _ = $0.apiLoaded.removeValue(forKey: id) }
                continue
            }
            do {
                if try await link.unload(modelID: id) {
                    state.withLock { _ = $0.apiLoaded.removeValue(forKey: id) }
                    ServerLog.modelUnloaded(id)
                } else {
                    free = false
                }
            } catch {
                free = false
            }
        }
        return free
    }
}
