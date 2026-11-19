import Foundation
import Darwin

/// Shared installation and authenticated acquisition for app, CLI and API hosts.
/// Concurrent models requesting one engine share a single download task.
public actor InferenceEngineDistributionController {
    private struct Flight {
        let task: Task<Void, Error>
        var waiters: Set<UUID>
    }
    public nonisolated let catalog: InferenceEngineCatalog
    private let store: InferenceEngineStore
    private let installer: InferenceEngineInstaller
    private var flights: [String: Flight] = [:]

    public init(root: URL, catalog: InferenceEngineCatalog,
                transport: any InferenceEngineDownloadTransport = InferenceEngineHTTPTransport()) throws {
        try catalog.validate()
        self.catalog = catalog
        let store = InferenceEngineStore(root: root, trustedPublisherKeys: catalog.publisherKeys)
        self.store = store
        installer = InferenceEngineInstaller(store: store, trustedPublisherKeys: catalog.publisherKeys,
                                            transport: transport)
    }

    public func acquire(architectureID: String, modelFormatVersion: Int = 1,
                        quantizationLayout: String, protocolVersion: Int,
                        macOSMajorVersion: Int) async throws -> InferenceEngineLease {
        try Task.checkCancellation()
        let release = try catalog.release(architectureID: architectureID, macOSMajorVersion: macOSMajorVersion)
        do {
            return try await store.acquire(packageID: release.packageID, version: release.version,
                architectureID: architectureID, modelFormatVersion: modelFormatVersion,
                quantizationLayout: quantizationLayout, protocolVersion: protocolVersion,
                macOSMajorVersion: macOSMajorVersion)
        } catch {
            guard Self.isMissingOrCorrupt(error) else { throw error }
        }
        let request = InferenceEngineInstallRequest(packageID: release.packageID, version: release.version,
            manifestURL: release.manifestURL, archiveURL: release.archiveURL,
            architectureID: architectureID, modelFormatVersion: modelFormatVersion,
            quantizationLayout: quantizationLayout, protocolVersion: protocolVersion,
            macOSMajorVersion: macOSMajorVersion, repairCorrupt: true)
        try await installShared(request)
        try Task.checkCancellation()
        return try await store.acquire(packageID: release.packageID, version: release.version,
            architectureID: architectureID, modelFormatVersion: modelFormatVersion,
            quantizationLayout: quantizationLayout, protocolVersion: protocolVersion,
            macOSMajorVersion: macOSMajorVersion)
    }

    public func release(_ lease: InferenceEngineLease) async throws { try await store.release(lease) }

    /// See `InferenceEngineStore.bindingsLockWaits`.
    nonisolated var bindingsLockWaits: Int { store.bindingsLockWaits.load(ordering: .relaxed) }

    public func bindModel(_ modelID: String, engine: InferenceEngineLease,
                          rollback: [InferenceEnginePackageReference] = [], directory: URL? = nil) async throws {
        let current = InferenceEnginePackageReference(packageID: engine.manifest.packageID, version: engine.manifest.version)
        let previous = try await store.knownGood(for: modelID)
        let retained = previous.map { $0.packageID == current.packageID && $0 != current ? [$0] : [] } ?? []
        // A rollback engine whose receipt no longer verifies (for example one
        // signed by an earlier publisher key) can't be used, so it isn't kept.
        var fallback: [InferenceEnginePackageReference] = []
        for reference in rollback + retained where fallback.isEmpty {
            if (try? await store.status(packageID: reference.packageID, version: reference.version))?.verified == true {
                fallback.append(reference)
            }
        }
        try await store.bindModel(modelID, to: [current] + fallback, directory: directory)
    }

    public func markKnownGood(_ modelID: String, engine: InferenceEngineLease) async throws {
        try await store.markKnownGood(modelID, reference: .init(packageID: engine.manifest.packageID, version: engine.manifest.version))
    }

    public func acquireKnownGood(_ modelID: String, architectureID: String, quantizationLayout: String,
                                 protocolVersion: Int, macOSMajorVersion: Int) async throws -> InferenceEngineLease? {
        guard let reference = try await store.knownGood(for: modelID) else { return nil }
        return try await store.acquire(packageID: reference.packageID, version: reference.version,
            architectureID: architectureID, modelFormatVersion: 1, quantizationLayout: quantizationLayout,
            protocolVersion: protocolVersion, macOSMajorVersion: macOSMajorVersion)
    }

    public func unbindModel(_ modelID: String) async throws { try await store.unbindModel(modelID) }

    public func removeUnused(_ reference: InferenceEnginePackageReference) async throws -> Bool {
        try await store.pruneMissingModelBindings()
        return try await store.remove(packageID: reference.packageID, version: reference.version,
                               requiredByInstalledModel: false)
    }

    public func statuses() async throws -> [InferenceEnginePackageStatus] {
        try await store.pruneMissingModelBindings()
        var result: [InferenceEnginePackageStatus] = []
        for package in Set(catalog.releases.map(\.packageID)).sorted() {
            for version in try await store.versions(packageID: package) {
                result.append(try await store.status(packageID: package, version: version))
            }
        }
        return result
    }

    public func removeUnusedPackages() async throws -> Int {
        var removed = 0
        for status in try await statuses() where status.requiredModelCount == 0 && !status.inUse {
            if try await removeUnused(status.reference) { removed += 1 }
        }
        return removed
    }

    /// Removes engine versions an update replaced: older than every version
    /// this catalog installs for the same family, with no model bound to them
    /// (a version kept for rollback is bound to its model), no process
    /// running them, and nothing in the family's folder changed for at least
    /// `minimumAge`. Each engine release otherwise left its predecessor on
    /// disk until someone pressed Remove Unused Engines in Settings.
    ///
    /// Narrower than `removeUnusedPackages` on purpose, because this runs
    /// without anyone asking and other TUFF builds share the folder:
    /// - A current engine no model has used yet stays, so it is not
    ///   downloaded again.
    /// - A version newer than this catalog belongs to a newer build and is
    ///   never touched.
    /// - A family in which something was installed, repaired or recovered
    ///   recently is left for a later launch; see `removeSuperseded`.
    /// - The bindings of a model whose folder is missing are left alone,
    ///   since it may be on a drive that is not plugged in.
    @discardableResult
    public func removeSupersededPackages(minimumAge: TimeInterval = 86_400,
                                         now: Date = Date()) async -> Int {
        var oldestCurrent: [String: InferenceEngineVersion] = [:]
        for release in catalog.releases {
            oldestCurrent[release.packageID] = min(release.version,
                                                   oldestCurrent[release.packageID] ?? release.version)
        }
        var removed = 0
        for (package, current) in oldestCurrent.sorted(by: { $0.key < $1.key }) {
            removed += (try? await store.removeSuperseded(
                packageID: package, olderThan: current,
                installedBefore: now.addingTimeInterval(-minimumAge))) ?? 0
        }
        return removed
    }

    private func installShared(_ request: InferenceEngineInstallRequest) async throws {
        let key = "\(request.packageID):\(request.version.directoryName)"
        let id = UUID()
        let task: Task<Void, Error>
        if var current = flights[key] {
            current.waiters.insert(id)
            flights[key] = current
            task = current.task
        } else {
            let installer = self.installer
            task = Task { _ = try await installer.install(request) }
            flights[key] = Flight(task: task, waiters: [id])
        }
        do {
            try await withTaskCancellationHandler {
                try await task.value
                try Task.checkCancellation()
            } onCancel: { [weak self] in
                Task { await self?.finishWaiter(key: key, id: id) }
            }
            finishWaiter(key: key, id: id)
        } catch {
            finishWaiter(key: key, id: id)
            // A different process may have completed the same installation and
            // acquired it before our exclusive activation lock was available.
            if error as? InferenceEngineStoreError == .packageBusy { return }
            throw error
        }
    }

    private func finishWaiter(key: String, id: UUID) {
        guard var flight = flights[key], flight.waiters.remove(id) != nil else { return }
        if flight.waiters.isEmpty {
            flight.task.cancel()
            flights.removeValue(forKey: key)
        } else { flights[key] = flight }
    }

    private static func isMissingOrCorrupt(_ error: Error) -> Bool {
        if let error = error as? POSIXError { return error.code == .ENOENT }
        if let error = error as? InferenceEngineStoreError {
            switch error {
            case .missingFile, .digestMismatch, .unsafeFile, .unexpectedFile: return true
            default: return false
            }
        }
        if let manifest = error as? InferenceEngineManifestError { return manifest != .incompatibleEngine }
        return error is DecodingError
    }
}
