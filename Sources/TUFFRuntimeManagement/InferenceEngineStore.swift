import Foundation
import TUFFInferenceTypes
import CryptoKit
import Darwin
import Synchronization

public enum InferenceEngineStoreError: Error, Equatable {
    case unsafeFile(String)
    case unexpectedFile(String)
    case missingFile(String)
    case digestMismatch(String)
    case conflictingVersion
    case invalidLease
    case packageBusy
    case insufficientDiskSpace(requiredBytes: UInt64, availableBytes: UInt64)
}

public struct InferenceEngineLease: Sendable {
    public let id: UUID
    public let executableURL: URL
    public let lockFileURL: URL
    public let manifest: InferenceEngineManifest
}

/// Operates on authenticated, already-extracted packages. The downloader must
/// verify the signed archive digest before extraction; this store independently
/// verifies the exact extracted file set before activation and before launch.
public actor InferenceEngineStore {
    private let root: URL
    private let publisherKeys: [String: Data]
    private var leases: [UUID: String] = [:]
    private var leaseDescriptors: [UUID: Int32] = [:]
    /// How many times this store found the bindings lock busy and waited.
    /// Readable while the store is waiting, which is when a test needs it.
    nonisolated let bindingsLockWaits = Atomic<Int>(0)

    public init(root: URL, trustedPublisherKeys: [String: Data]) {
        self.root = root.standardizedFileURL
        self.publisherKeys = trustedPublisherKeys
    }

    deinit {
        for fd in leaseDescriptors.values { close(fd) }
    }

    /// Charge the downloaded archive, extraction and verified activation copy.
    /// Check both volumes before transfer; an existing repair backup is already
    /// included in the filesystem's used capacity.
    public func requireInstallationSpace(for manifest: InferenceEngineManifest) throws {
        let payload = try EngineByteMath.sum(manifest.files.map(\.bytes))
        let required = try EngineByteMath.sum([manifest.archiveBytes, payload, payload, 1 << 20])
        for location in [root, FileManager.default.temporaryDirectory] {
            var ancestor = location
            while !FileManager.default.fileExists(atPath: ancestor.path), ancestor.path != "/" {
                ancestor.deleteLastPathComponent()
            }
            let attributes = try FileManager.default.attributesOfFileSystem(forPath: ancestor.path)
            guard let available = (attributes[.systemFreeSize] as? NSNumber)?.uint64Value else {
                throw POSIXError(.EIO)
            }
            guard available >= required else {
                throw InferenceEngineStoreError.insufficientDiskSpace(requiredBytes: required, availableBytes: available)
            }
        }
    }

    public func install(extractedDirectory: URL, signedManifest: SignedInferenceEngineManifest,
                        repairCorrupt: Bool = false) throws {
        try Task.checkCancellation()
        let manifest = try signedManifest.verifiedManifest(trustedPublisherKeys: publisherKeys)
        try ensureStore()
        let lock = try packageLock(packageID: manifest.packageID, version: manifest.version, exclusive: true)
        defer { close(lock) }
        let packageRoot = root.appendingPathComponent(manifest.packageID, isDirectory: true)
        try createOwnedDirectory(packageRoot)
        let target = packageRoot.appendingPathComponent(manifest.version.directoryName, isDirectory: true)
        try recoverRepair(packageRoot: packageRoot, target: target, version: manifest.version)
        var replacingCorrupt = false
        if FileManager.default.fileExists(atPath: target.path) {
            do {
                let existing = try readManifest(at: target)
                guard existing == manifest else { throw InferenceEngineStoreError.conflictingVersion }
                try verifyPayload(at: target, manifest: manifest, hasReceipt: true)
                return
            } catch InferenceEngineStoreError.conflictingVersion {
                throw InferenceEngineStoreError.conflictingVersion
            } catch {
                guard repairCorrupt else { throw error }
                // Replacement requires the exclusive version lock acquired above.
                // A helper or another host's live lease therefore prevents repair.
                replacingCorrupt = true
            }
        }
        try verifyPayload(at: extractedDirectory, manifest: manifest, hasReceipt: false)
        let staging = packageRoot.appendingPathComponent(".staging-\(UUID().uuidString)", isDirectory: true)
        try createOwnedDirectory(staging)
        defer { try? FileManager.default.removeItem(at: staging) }
        for file in manifest.files {
            try Task.checkCancellation()
            let destination = staging.appendingPathComponent(file.path)
            try createOwnedDirectory(destination.deletingLastPathComponent())
            // Read through a no-follow descriptor, then verify the copied bytes.
            // A source changed between initial validation and copy never activates.
            try copyRegularFile(from: extractedDirectory.appendingPathComponent(file.path),
                to: destination, expectedBytes: file.bytes, executable: file.executable)
        }
        try verifyPayload(at: staging, manifest: manifest, hasReceipt: false)
        let receipt = staging.appendingPathComponent(".engine-receipt.json")
        try JSONEncoder().encode(signedManifest).write(to: receipt, options: .atomic)
        guard chmod(receipt.path, 0o600) == 0 else { throw posixError() }
        try Task.checkCancellation()
        if replacingCorrupt {
            let backup = packageRoot.appendingPathComponent(".repair-backup-\(manifest.version.directoryName)")
            guard !FileManager.default.fileExists(atPath: backup.path) else {
                throw InferenceEngineStoreError.packageBusy
            }
            try FileManager.default.moveItem(at: target, to: backup)
            do {
                try FileManager.default.moveItem(at: staging, to: target)
            } catch {
                try? FileManager.default.moveItem(at: backup, to: target)
                throw error
            }
            try FileManager.default.removeItem(at: backup)
        } else {
            try FileManager.default.moveItem(at: staging, to: target)
        }
    }

    public func acquire(packageID: String, version: InferenceEngineVersion,
                        architectureID: String, modelFormatVersion: Int,
                        quantizationLayout: String, protocolVersion: Int,
                        macOSMajorVersion: Int) throws -> InferenceEngineLease {
        try ensureStore()
        let directory = try versionDirectory(packageID: packageID, version: version, allowMissingVersion: true)
        let packageRoot = directory.deletingLastPathComponent()
        let backup = packageRoot.appendingPathComponent(".repair-backup-\(version.directoryName)")
        if FileManager.default.fileExists(atPath: backup.path) {
            let recoveryLock = try packageLock(packageID: packageID, version: version, exclusive: true)
            defer { close(recoveryLock) }
            try recoverRepair(packageRoot: packageRoot, target: directory, version: version)
        }
        let lock = try packageLock(packageID: packageID, version: version, exclusive: false)
        var retained = false
        defer { if !retained { close(lock) } }
        let manifest = try readManifest(at: directory)
        guard manifest.packageID == packageID, manifest.version == version else {
            throw InferenceEngineStoreError.conflictingVersion
        }
        try manifest.requireCompatibility(architectureID: architectureID,
            modelFormatVersion: modelFormatVersion, quantizationLayout: quantizationLayout,
            protocolVersion: protocolVersion, macOSMajorVersion: macOSMajorVersion)
        try verifyPayload(at: directory, manifest: manifest, hasReceipt: true)
        let id = UUID()
        leases[id] = directory.path
        leaseDescriptors[id] = lock
        retained = true
        return InferenceEngineLease(id: id,
            executableURL: directory.appendingPathComponent(manifest.executablePath),
            lockFileURL: root.appendingPathComponent(".locks/\(packageID)-\(version.directoryName).lock"),
            manifest: manifest)
    }

    public func release(_ lease: InferenceEngineLease) throws {
        guard leases.removeValue(forKey: lease.id) != nil else {
            throw InferenceEngineStoreError.invalidLease
        }
        if let fd = leaseDescriptors.removeValue(forKey: lease.id) { close(fd) }
    }

    /// The runtime manager supplies the current registered-model requirements.
    /// Returns false while a model or active process still requires the package.
    public func remove(packageID: String, version: InferenceEngineVersion,
                       requiredByInstalledModel: Bool) throws -> Bool {
        try ensureStore()
        let directory = try versionDirectory(packageID: packageID, version: version)
        // A bound version is left without touching its lock: holding it, even
        // for an instant, would refuse another process leasing it just then.
        guard !requiredByInstalledModel, !leases.values.contains(directory.path),
              try !boundVersions(packageID: packageID).contains(version) else { return false }
        let lock: Int32
        do { lock = try packageLock(packageID: packageID, version: version, exclusive: true) }
        catch InferenceEngineStoreError.packageBusy { return false }
        defer { close(lock) }
        // Read again under the exclusive lock. Binding a model holds the
        // version's shared lock, so the answer cannot change from here on,
        // and the bindings are not held while the files are deleted.
        guard try !boundVersions(packageID: packageID).contains(version) else { return false }
        try FileManager.default.removeItem(at: directory)
        return true
    }

    /// Removes the versions of one package that an update replaced, for a
    /// cleanup nobody asked for: older than `current`, not bound to a model,
    /// not held by a process, and untouched since before `installedBefore`.
    ///
    /// Other TUFF builds share this folder, older ones included. One of them
    /// may have installed, repaired or recovered a version a moment ago and
    /// not leased it yet. So every candidate is locked exclusively before
    /// anything is examined: installing, repairing and recovering a version
    /// need that lock, and leasing or binding needs its shared form. Then the
    /// package folder itself must be unchanged since `installedBefore`. Any
    /// installation or recovery renames a directory into it, which shows
    /// there even when the version restored keeps its old dates, as an older
    /// build's recovery does. A package that changed recently waits for a
    /// later launch.
    public func removeSuperseded(packageID: String, olderThan current: InferenceEngineVersion,
                                 installedBefore: Date) throws -> Int {
        try ensureStore()
        let candidates = try versions(packageID: packageID).filter { $0 < current }
        guard !candidates.isEmpty else { return 0 }
        let package = root.appendingPathComponent(packageID, isDirectory: true)
        let bound = try boundVersions(packageID: packageID)
        var held: [(version: InferenceEngineVersion, directory: URL, lock: Int32)] = []
        defer { for entry in held { close(entry.lock) } }
        for version in candidates where !bound.contains(version) {
            guard let directory = try? versionDirectory(packageID: packageID, version: version),
                  !leases.values.contains(directory.path),
                  let lock = try? packageLock(packageID: packageID, version: version, exclusive: true)
            else { continue }
            held.append((version, directory, lock))
        }
        guard !held.isEmpty, let changed = Self.modificationDate(of: package),
              changed < installedBefore else { return 0 }
        // Read again now that the candidates are locked; see `remove`.
        let stillBound = try boundVersions(packageID: packageID)
        var removed = 0
        for entry in held where !stillBound.contains(entry.version) {
            let backup = package.appendingPathComponent(".repair-backup-\(entry.version.directoryName)")
            guard let placed = Self.modificationDate(of: entry.directory), placed < installedBefore,
                  !FileManager.default.fileExists(atPath: backup.path) else { continue }
            try FileManager.default.removeItem(at: entry.directory)
            removed += 1
        }
        return removed
    }

    /// The versions of `packageID` that any model is bound to, read under a
    /// briefly held bindings lock.
    private func boundVersions(packageID: String) throws -> [InferenceEngineVersion] {
        let lock = try modelBindingsLock(exclusive: false)
        defer { close(lock) }
        return try readModelBindings().models.values.joined()
            .filter { $0.packageID == packageID }.map(\.version)
    }

    private static func modificationDate(of directory: URL) -> Date? {
        var info = stat()
        guard lstat(directory.path, &info) == 0, info.st_mode & S_IFMT == S_IFDIR else { return nil }
        return Date(timeIntervalSince1970: TimeInterval(info.st_mtimespec.tv_sec)
            + TimeInterval(info.st_mtimespec.tv_nsec) / 1e9)
    }

    /// Pins every engine version an installed model needs, including an old
    /// version deliberately retained for rollback. All managers read this ledger
    /// under the same lock before removal.
    public func bindModel(_ modelID: String, to references: [InferenceEnginePackageReference],
                          directory: URL? = nil) throws {
        guard !modelID.isEmpty, modelID.utf8.count <= 256, references.count <= 8,
              !references.isEmpty else { throw InferenceEngineStoreError.unsafeFile("model binding") }
        try ensureStore()
        // The versions are locked and verified before the bindings are
        // locked. Hashing a payload takes tens of milliseconds, and every
        // process that loads a model waits on the bindings lock. The shared
        // version locks keep these files from being removed until the record
        // is written.
        var versionLocks: [Int32] = []
        defer { for fd in versionLocks { close(fd) } }
        for reference in references {
            versionLocks.append(try packageLock(packageID: reference.packageID,
                version: reference.version, exclusive: false))
            let directory = try versionDirectory(packageID: reference.packageID, version: reference.version)
            let manifest = try readManifest(at: directory)
            guard manifest.packageID == reference.packageID, manifest.version == reference.version else {
                throw InferenceEngineStoreError.conflictingVersion
            }
            try verifyPayload(at: directory, manifest: manifest, hasReceipt: true)
        }
        let lock = try modelBindingsLock(exclusive: true)
        defer { close(lock) }
        var bindings = try readModelBindings()
        bindings.models[modelID] = references
        if let good = bindings.knownGood?[modelID], !references.contains(good) {
            bindings.knownGood?.removeValue(forKey: modelID)
        }
        if let directory {
            let path = directory.standardizedFileURL.resolvingSymlinksInPath().path
            guard path.hasPrefix("/"), !path.contains("\0"), path.utf8.count <= 4096 else {
                throw InferenceEngineStoreError.unsafeFile("model directory")
            }
            if bindings.modelPaths == nil { bindings.modelPaths = [:] }
            bindings.modelPaths?[modelID] = path
        }
        try writeModelBindings(bindings)
    }

    public func knownGood(for modelID: String) throws -> InferenceEnginePackageReference? {
        try ensureStore()
        let lock = try modelBindingsLock(exclusive: false)
        defer { close(lock) }
        return try readModelBindings().knownGood?[modelID]
    }

    public func status(packageID: String, version: InferenceEngineVersion) throws -> InferenceEnginePackageStatus {
        try ensureStore()
        let directory = try versionDirectory(packageID: packageID, version: version)
        let reference = InferenceEnginePackageReference(packageID: packageID, version: version)
        // Only the count needs the bindings. The payload is hashed below, and
        // holding the lock through that made other processes' model loads wait.
        let required: Int
        do {
            let bindingLock = try modelBindingsLock(exclusive: false)
            defer { close(bindingLock) }
            required = try readModelBindings().models.values.filter { $0.contains(reference) }.count
        }
        var inUse = leases.values.contains(directory.path)
        do { let fd = try packageLock(packageID: packageID, version: version, exclusive: true); close(fd) }
        catch InferenceEngineStoreError.packageBusy { inUse = true }
        let manifest = try? readManifest(at: directory)
        let valid = manifest.map { (try? verifyPayload(at: directory, manifest: $0, hasReceipt: true)) != nil } ?? false
        return .init(reference: reference, verified: valid, requiredModelCount: required,
                     inUse: inUse, installedBytes: manifest?.installedBytes)
    }

    public func versions(packageID: String) throws -> [InferenceEngineVersion] {
        try ensureStore()
        guard InferenceEngineManifest.safeRelativePath(packageID), !packageID.contains("/"), !packageID.hasPrefix(".") else {
            throw InferenceEngineStoreError.unsafeFile(packageID)
        }
        let directory = root.appendingPathComponent(packageID)
        if !FileManager.default.fileExists(atPath: directory.path) { return [] }
        try requireOwnedDirectory(directory)
        let entries = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        guard entries.count <= 128 else { throw InferenceEngineStoreError.unsafeFile("too many engine versions") }
        return entries.compactMap { name in
            let parts = name.split(separator: ".", omittingEmptySubsequences: false).compactMap { Int($0) }
            guard parts.count == 3, parts.allSatisfy({ $0 >= 0 }) else { return nil }
            let version = InferenceEngineVersion(major: parts[0], minor: parts[1], patch: parts[2])
            guard version.directoryName == name else { return nil }
            return version
        }.sorted(by: >)
    }

    public func markKnownGood(_ modelID: String, reference: InferenceEnginePackageReference) throws {
        try ensureStore()
        let lock = try modelBindingsLock(exclusive: true)
        defer { close(lock) }
        var bindings = try readModelBindings()
        guard bindings.models[modelID]?.contains(reference) == true else {
            throw InferenceEngineStoreError.unsafeFile("unbound engine")
        }
        if bindings.knownGood == nil { bindings.knownGood = [:] }
        bindings.knownGood?[modelID] = reference
        try writeModelBindings(bindings)
    }

    /// Only dependency records are pruned. Model directories are never removed.
    public func pruneMissingModelBindings() throws {
        try ensureStore()
        let lock = try modelBindingsLock(exclusive: true)
        defer { close(lock) }
        var bindings = try readModelBindings()
        for (id, path) in bindings.modelPaths ?? [:] {
            var info = stat()
            if lstat(path, &info) < 0, errno == ENOENT {
                bindings.models.removeValue(forKey: id)
                bindings.modelPaths?.removeValue(forKey: id)
                bindings.knownGood?.removeValue(forKey: id)
            }
        }
        try writeModelBindings(bindings)
    }

    public func unbindModel(_ modelID: String) throws {
        try ensureStore()
        let lock = try modelBindingsLock(exclusive: true)
        defer { close(lock) }
        var bindings = try readModelBindings()
        bindings.models.removeValue(forKey: modelID)
        bindings.modelPaths?.removeValue(forKey: modelID)
        bindings.knownGood?.removeValue(forKey: modelID)
        try writeModelBindings(bindings)
    }

    public func requirements(for modelID: String) throws -> [InferenceEnginePackageReference] {
        try ensureStore()
        let lock = try modelBindingsLock(exclusive: false)
        defer { close(lock) }
        return try readModelBindings().models[modelID] ?? []
    }

    private func modelBindingsLock(exclusive: Bool) throws -> Int32 {
        // Reserved package name; independent of all published engine locks.
        // Every holder reads or rewrites one small file and lets go, so a
        // busy lock is about to be free. The app, the CLI and the server
        // share it: failing at once made one process's model load fail with
        // `packageBusy` because another happened to be reading the bindings
        // at that instant. Wait for it, up to about two seconds, instead.
        var attempts = 0
        while true {
            do {
                return try packageLock(packageID: "model-bindings",
                                       version: .init(major: 0, minor: 0, patch: 0),
                                       exclusive: exclusive)
            } catch InferenceEngineStoreError.packageBusy where attempts < Self.bindingsLockAttempts {
                attempts += 1
                bindingsLockWaits.add(1, ordering: .relaxed)
                usleep(Self.bindingsLockRetryMicroseconds)
            }
        }
    }

    /// About two seconds in all, far longer than any holder keeps the lock.
    private static let bindingsLockAttempts = 400
    private static let bindingsLockRetryMicroseconds: UInt32 = 5_000

    private func readModelBindings() throws -> InferenceEngineModelBindings {
        let url = root.appendingPathComponent(".model-bindings.json")
        let fd = open(url.path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        if fd < 0 && errno == ENOENT { return InferenceEngineModelBindings() }
        guard fd >= 0 else { throw posixError() }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == getuid(), info.st_mode & 0o077 == 0,
              info.st_size > 0, info.st_size <= 1_048_576 else {
            throw InferenceEngineStoreError.unsafeFile(url.path)
        }
        var data = Data(count: Int(info.st_size))
        var position = 0
        while position < data.count {
            let remaining = data.count - position
            let got = data.withUnsafeMutableBytes {
                pread(fd, $0.baseAddress!.advanced(by: position), remaining, off_t(position))
            }
            if got < 0 && errno == EINTR { continue }
            guard got > 0 else { throw InferenceEngineStoreError.unsafeFile(url.path) }
            position += got
        }
        let result = try JSONDecoder().decode(InferenceEngineModelBindings.self, from: data)
        guard result.schemaVersion == 1, result.models.count <= 4096,
              (result.modelPaths?.count ?? 0) <= 4096,
              result.modelPaths?.values.allSatisfy({ $0.hasPrefix("/") && !$0.contains("\0") && $0.utf8.count <= 4096 }) ?? true,
              (result.knownGood?.count ?? 0) <= 4096,
              result.knownGood?.allSatisfy({ result.models[$0.key]?.contains($0.value) == true }) ?? true,
              result.models.allSatisfy({ !$0.key.isEmpty && $0.key.utf8.count <= 256 && $0.value.count <= 8 }),
              result.models.values.joined().allSatisfy({
                  InferenceEngineManifest.safeRelativePath($0.packageID) && !$0.packageID.contains("/")
                      && !$0.packageID.hasPrefix(".") && $0.version.major >= 0
                      && $0.version.minor >= 0 && $0.version.patch >= 0
              }) else { throw InferenceEngineStoreError.unsafeFile(url.path) }
        return result
    }

    private func writeModelBindings(_ bindings: InferenceEngineModelBindings) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(bindings)
        guard bindings.models.count <= 4096, data.count <= 1_048_576 else {
            throw InferenceEngineStoreError.unsafeFile("model binding limit")
        }
        let url = root.appendingPathComponent(".model-bindings.json")
        try data.write(to: url, options: .atomic)
        guard chmod(url.path, 0o600) == 0 else { throw posixError() }
    }

    private func recoverRepair(packageRoot: URL, target: URL,
                               version: InferenceEngineVersion) throws {
        let backup = packageRoot.appendingPathComponent(".repair-backup-\(version.directoryName)")
        guard FileManager.default.fileExists(atPath: backup.path) else { return }
        try requireOwnedDirectory(backup)
        if FileManager.default.fileExists(atPath: target.path) {
            try requireOwnedDirectory(target)
            if let manifest = try? readManifest(at: target),
               (try? verifyPayload(at: target, manifest: manifest, hasReceipt: true)) != nil {
                try FileManager.default.removeItem(at: backup)
                return
            }
            if let previous = try? readManifest(at: backup),
               (try? verifyPayload(at: backup, manifest: previous, hasReceipt: true)) != nil {
                try FileManager.default.removeItem(at: target)
                try FileManager.default.moveItem(at: backup, to: target)
                // The restored directory keeps its old date. Whoever is
                // recovering it is about to lease it, so mark it as newly
                // placed for the automatic cleanup, which also sees the
                // rename in the package folder.
                try? FileManager.default.setAttributes(
                    [.modificationDate: Date()], ofItemAtPath: target.path)
                return
            }
            // Both managed copies are corrupt. Keep the active bytes for the
            // authenticated repair path and discard the stale repair backup.
            try FileManager.default.removeItem(at: backup)
        } else {
            // Preserve the previous bytes. A subsequent authenticated repair can
            // replace them; acquisition never treats the backup as verified.
            try FileManager.default.moveItem(at: backup, to: target)
            try? FileManager.default.setAttributes(
                [.modificationDate: Date()], ofItemAtPath: target.path)
        }
    }

    private func ensureStore() throws { try createOwnedDirectory(root) }

    private func packageLock(packageID: String, version: InferenceEngineVersion,
                             exclusive: Bool) throws -> Int32 {
        guard InferenceEngineManifest.safeRelativePath(packageID), !packageID.contains("/"),
              !packageID.hasPrefix("."), version.major >= 0, version.minor >= 0, version.patch >= 0 else {
            throw InferenceEngineStoreError.unsafeFile(packageID)
        }
        let lockDirectory = root.appendingPathComponent(".locks", isDirectory: true)
        try createOwnedDirectory(lockDirectory)
        let file = lockDirectory.appendingPathComponent("\(packageID)-\(version.directoryName).lock")
        let fd = open(file.path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw posixError() }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == getuid(), info.st_mode & 0o077 == 0 else {
            close(fd)
            throw InferenceEngineStoreError.unsafeFile(file.path)
        }
        guard flock(fd, (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB) == 0 else {
            let code = errno
            close(fd)
            if code == EWOULDBLOCK { throw InferenceEngineStoreError.packageBusy }
            throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
        }
        return fd
    }

    private func createOwnedDirectory(_ url: URL) throws {
        if !FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
        }
        var info = stat()
        guard lstat(url.path, &info) == 0 else { throw posixError() }
        guard info.st_mode & S_IFMT == S_IFDIR, info.st_uid == getuid(),
              info.st_mode & 0o022 == 0 else { throw InferenceEngineStoreError.unsafeFile(url.path) }
    }

    private func versionDirectory(packageID: String, version: InferenceEngineVersion,
                                  allowMissingVersion: Bool = false) throws -> URL {
        guard InferenceEngineManifest.safeRelativePath(packageID), !packageID.contains("/"),
              !packageID.hasPrefix("."), version.major >= 0, version.minor >= 0, version.patch >= 0 else {
            throw InferenceEngineStoreError.unsafeFile(packageID)
        }
        let package = root.appendingPathComponent(packageID, isDirectory: true)
        try requireOwnedDirectory(package)
        let directory = package.appendingPathComponent(version.directoryName, isDirectory: true)
        if !allowMissingVersion { try requireOwnedDirectory(directory) }
        return directory
    }

    private func requireOwnedDirectory(_ url: URL) throws {
        var info = stat()
        guard lstat(url.path, &info) == 0 else { throw posixError() }
        guard info.st_mode & S_IFMT == S_IFDIR, info.st_uid == getuid(),
              info.st_mode & 0o022 == 0 else { throw InferenceEngineStoreError.unsafeFile(url.path) }
    }

    private func readManifest(at directory: URL) throws -> InferenceEngineManifest {
        try requireOwnedDirectory(directory)
        let receipt = directory.appendingPathComponent(".engine-receipt.json")
        let fd = try regularDescriptor(receipt)
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size > 0,
              info.st_size <= 2 * SignedInferenceEngineManifest.maximumPayloadBytes else {
            throw InferenceEngineStoreError.unsafeFile(receipt.path)
        }
        let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: false)
        let limit = 2 * SignedInferenceEngineManifest.maximumPayloadBytes
        let data = try handle.read(upToCount: limit + 1) ?? Data()
        guard data.count <= limit else { throw InferenceEngineStoreError.unsafeFile(receipt.path) }
        return try JSONDecoder().decode(SignedInferenceEngineManifest.self, from: data)
            .verifiedManifest(trustedPublisherKeys: publisherKeys)
    }

    private func verifyPayload(at directory: URL, manifest: InferenceEngineManifest,
                               hasReceipt: Bool) throws {
        try requireOwnedDirectory(directory)
        let expected = Set(manifest.files.map(\.path))
        let paths = try FileManager.default.subpathsOfDirectory(atPath: directory.path)
        var found: Set<String> = []
        for path in paths {
            let url = directory.appendingPathComponent(path)
            var info = stat()
            guard lstat(url.path, &info) == 0 else { throw posixError() }
            if info.st_mode & S_IFMT == S_IFDIR {
                try requireOwnedDirectory(url)
                continue
            }
            guard info.st_mode & S_IFMT == S_IFREG else {
                throw InferenceEngineStoreError.unsafeFile(path)
            }
            if hasReceipt && path == ".engine-receipt.json" { continue }
            guard expected.contains(path) else { throw InferenceEngineStoreError.unexpectedFile(path) }
            found.insert(path)
        }
        for file in manifest.files {
            try Task.checkCancellation()
            guard found.contains(file.path) else { throw InferenceEngineStoreError.missingFile(file.path) }
            let url = directory.appendingPathComponent(file.path)
            let fd = try regularDescriptor(url)
            defer { close(fd) }
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_size >= 0, UInt64(info.st_size) == file.bytes,
                  !file.executable || info.st_mode & 0o111 != 0 else {
                throw InferenceEngineStoreError.unsafeFile(file.path)
            }
            let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: false)
            var hash = SHA256()
            var readBytes: UInt64 = 0
            while let bytes = try handle.read(upToCount: 1 << 20), !bytes.isEmpty {
                try Task.checkCancellation()
                let sum = readBytes.addingReportingOverflow(UInt64(bytes.count))
                guard !sum.overflow, sum.partialValue <= file.bytes else {
                    throw InferenceEngineStoreError.unsafeFile(file.path)
                }
                readBytes = sum.partialValue
                hash.update(data: bytes)
            }
            guard readBytes == file.bytes else { throw InferenceEngineStoreError.unsafeFile(file.path) }
            let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
            guard digest == file.sha256 else { throw InferenceEngineStoreError.digestMismatch(file.path) }
        }
    }

    private func regularDescriptor(_ url: URL) throws -> Int32 {
        let fd = open(url.path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw InferenceEngineStoreError.unsafeFile(url.path) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == getuid(), info.st_mode & 0o022 == 0 else {
            close(fd)
            throw InferenceEngineStoreError.unsafeFile(url.path)
        }
        return fd
    }

    private func copyRegularFile(from source: URL, to destination: URL,
                                 expectedBytes: UInt64, executable: Bool) throws {
        let inputFD = try regularDescriptor(source)
        defer { close(inputFD) }
        let outputFD = open(destination.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
            executable ? 0o755 : 0o644)
        guard outputFD >= 0 else { throw posixError() }
        defer { close(outputFD) }
        let input = FileHandle(fileDescriptor: inputFD, closeOnDealloc: false)
        let output = FileHandle(fileDescriptor: outputFD, closeOnDealloc: false)
        var copied: UInt64 = 0
        while let bytes = try input.read(upToCount: 1 << 20), !bytes.isEmpty {
            try Task.checkCancellation()
            let sum = copied.addingReportingOverflow(UInt64(bytes.count))
            guard !sum.overflow, sum.partialValue <= expectedBytes else {
                throw InferenceEngineStoreError.unsafeFile(source.path)
            }
            copied = sum.partialValue
            try output.write(contentsOf: bytes)
        }
        guard copied == expectedBytes else { throw InferenceEngineStoreError.unsafeFile(source.path) }
        guard fsync(outputFD) == 0 else { throw posixError() }
    }

    private func posixError() -> POSIXError { POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
}
