import TUFFRuntimeManagement
import TUFFModelCatalog
import Foundation
import TUFFRepackCore
import Synchronization

public final class RepackModelInstallerClient: AppModelInstallerClient, Sendable {
    typealias InstallRunner = @Sendable (
        URL,
        @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> URL
    typealias DiscardRunner = @Sendable (URL) async throws -> Void

    private struct ActiveInstall: Sendable {
        let id: UUID
        let task: Task<Void, Never>
    }

    private final class InstallTaskState: Sendable {
        let value = Mutex<ActiveInstall?>(nil)
    }

    public let descriptor: AppModelInstallDescriptor
    private let runInstall: InstallRunner
    private let runDiscard: DiscardRunner
    private let taskState = InstallTaskState()

    public init(descriptor: AppModelInstallDescriptor = .default) {
        self.descriptor = descriptor
        self.runInstall = { outputDirectory, progress in
            guard let id = descriptor.catalogID, let model = TUFFModelCatalog.model(id: id) else {
                throw InferenceEngineCatalogError.noCompatibleEngine(descriptor.displayName)
            }
            let engine = try await InferenceEngineEnvironment.acquire(
                architectureID: model.architecture.id.rawValue,
                quantizationLayout: model.architecture.weightLayout.rawValue)
            do {
                if let size = PrismRemoteInstaller.Size(modelID: id) {
                    let directory = try await PrismRemoteInstaller.install(size: size, outputDirectory: outputDirectory,
                        token: ProcessInfo.processInfo.environment["HF_TOKEN"], reserveBytes: descriptor.reserveBytes,
                        progress: progress)
                    try await InferenceEngineEnvironment.bind(engine, to: directory)
                    await engine.release()
                    return directory
                }
                let paths = try RemoteInstallPaths(outputDirectory: outputDirectory.path)
                let resume = FileManager.default.fileExists(atPath: paths.checkpointFile)
                let options = RemoteStreamingRepackOptions(
                    repoID: descriptor.repoID,
                    revision: descriptor.revision,
                    outputDir: outputDirectory.path,
                    token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                    requireKnownSource: true,
                    minFreeReserveBytes: descriptor.reserveBytes,
                    overwrite: true,
                    resume: resume)
                let result = try await RemoteStreamingRepacker(options: options).run(progress: progress)
                let directory = URL(fileURLWithPath: result.outputDir).standardizedFileURL
                try await InferenceEngineEnvironment.bind(engine, to: directory)
                await engine.release()
                return directory
            } catch {
                await engine.release()
                throw error
            }
        }
        self.runDiscard = { outputDirectory in
            if let id = descriptor.catalogID, PrismRemoteInstaller.Size(modelID: id) != nil {
                try PrismRemoteInstaller.discardPartial(outputDirectory: outputDirectory)
                return
            }
            try RemoteStreamingRepacker.discardPartial(
                outputDirectory: outputDirectory.path)
        }
    }

    init(descriptor: AppModelInstallDescriptor = .default,
         runInstall: @escaping InstallRunner,
         runDiscard: @escaping DiscardRunner = { _ in }) {
        self.descriptor = descriptor
        self.runInstall = runInstall
        self.runDiscard = runDiscard
    }

    public func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement {
        if let id = descriptor.catalogID, let size = PrismRemoteInstaller.Size(modelID: id) {
            let needed = size.snapshotBytes + size.outputReservationBytes + 8 * 1024 * 1024
                + descriptor.engineInstallationTemporaryBytes
            let requirement = try DiskSpaceChecker.assess(path: outputDirectory.path, bytes: needed,
                                                           reserveBytes: descriptor.reserveBytes)
            return AppModelInstallRequirement(probePath: requirement.path,
                requiredBytes: requirement.requiredBytes, availableBytes: requirement.availableBytes)
        }
        let saved = try RemoteStreamingRepacker.inspectPersistentInstall(
            outputDirectory: outputDirectory.path,
            repoID: descriptor.repoID,
            requestedRevision: descriptor.revision)
        let remainingBytes: UInt64
        if let saved {
            let checkpointPath = try RemoteInstallPaths(
                outputDirectory: outputDirectory.path).checkpointFile
            let reused = try saved.validatedDestinationBytes(
                maximum: descriptor.installedBytes,
                path: checkpointPath)
            remainingBytes = descriptor.installedBytes > reused
                ? descriptor.installedBytes - reused
                : 0
        } else {
            remainingBytes = descriptor.installedBytes
        }
        let requested = remainingBytes.addingReportingOverflow(
            descriptor.rangeStagingBytes + descriptor.engineInstallationTemporaryBytes)
        guard !requested.overflow else {
            throw RepackError.configurationInvalid(
                detail: "model install requirement overflows UInt64")
        }
        let requirement = try DiskSpaceChecker.assess(
            path: outputDirectory.path,
            bytes: requested.partialValue,
            reserveBytes: descriptor.reserveBytes)
        return AppModelInstallRequirement(probePath: requirement.path,
                                          requiredBytes: requirement.requiredBytes,
                                          availableBytes: requirement.availableBytes)
    }

    public func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let id = UUID()
            let task = Task { [runInstall] in
                do {
                    continuation.yield(.checking)
                    let completedDirectory = try await runInstall(outputDirectory) { progress in
                        continuation.yield(Self.event(for: progress))
                    }
                    try Task.checkCancellation()
                    continuation.yield(.installed(completedDirectory))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            let previous = taskState.value.withLock { active in
                let previous = active?.task
                active = ActiveInstall(id: id, task: task)
                return previous
            }
            previous?.cancel()

            continuation.onTermination = { [taskState] _ in
                let task = taskState.value.withLock { active -> Task<Void, Never>? in
                    guard active?.id == id else { return nil }
                    defer { active = nil }
                    return active?.task
                }
                task?.cancel()
            }
        }
    }

    public func cancel() {
        let task = taskState.value.withLock { active -> Task<Void, Never>? in
            defer { active = nil }
            return active?.task
        }
        task?.cancel()
    }

    public func blockedInstallEntries(outputDirectory: URL) -> [String] {
        guard let paths = try? RemoteInstallPaths(
            outputDirectory: outputDirectory.standardizedFileURL.path),
              let blocked = try? paths.blockingEntries() else { return [] }
        return blocked.map(\.path)
    }

    public func clearBlockedInstallPath(outputDirectory: URL) async throws {
        let directory = outputDirectory.standardizedFileURL.path
        try await Task.detached(priority: .utility) {
            let paths = try RemoteInstallPaths(outputDirectory: directory)
            try paths.removeBlockingEntries()
        }.value
    }

    public func discardPartialInstall(outputDirectory: URL) async throws {
        let directory = outputDirectory.standardizedFileURL
        try await Task.detached(priority: .utility) { [runDiscard] in
            try await runDiscard(directory)
        }.value
    }

    public func removeInstalledModel(outputDirectory: URL) async throws {
        let directory = outputDirectory.standardizedFileURL
        try await Task.detached(priority: .utility) {
            try Self.removeInstalled(at: directory)
        }.value
    }

    /// Moves the folder aside under the install lock, then deletes it. The
    /// model stops looking installed the moment it moves, and no download can
    /// start into it meanwhile. An interrupted delete leaves a hidden
    /// `.<name>.deleting-…` folder that the next delete clears, never a
    /// half-removed model that still looks installed.
    static func removeInstalled(at directory: URL) throws {
        let lock = try InstallLock.acquire(outputDirectory: directory.path)
        let parent = lock.paths.parentDirectory
        let prefix = "." + URL(fileURLWithPath: lock.paths.finalDirectory).lastPathComponent + ".deleting-"
        let trash = (parent as NSString).appendingPathComponent(prefix + UUID().uuidString)
        try withExtendedLifetime(lock) {
            try Posix.rename(from: lock.paths.finalDirectory, to: trash)
            try? Posix.fsyncDirectory(parent)
        }
        let leftovers = (try? FileManager.default.contentsOfDirectory(atPath: parent)) ?? []
        for name in leftovers where name.hasPrefix(prefix) {
            try FileManager.default.removeItem(atPath: (parent as NSString).appendingPathComponent(name))
        }
    }

    static func event(for progress: ModelInstallProgress) -> AppModelInstallEvent {
        switch progress {
        case .downloadingMetadata:
            return .downloadingMetadata
        case .planning:
            return .planning
        case .checkingDisk:
            return .checking
        case .reservingOutput:
            return .reservingOutput
        case .copyingPayload(let reused, let downloadedThisRun, let total):
            return .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
        case .hashingOutput(let file):
            return .hashingOutput(file)
        case .finalizing:
            return .finalizing
        }
    }
}
