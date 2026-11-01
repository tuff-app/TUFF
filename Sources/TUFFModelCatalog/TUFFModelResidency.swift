import Darwin
import Foundation

/// Which TUFF processes currently hold a model in memory, so the app and the
/// background server do not load two large models side by side.
///
/// Each holder keeps a small JSON file open with an exclusive `flock`. The
/// kernel drops the lock when the process exits for any reason, so a file
/// whose lock can be taken belongs to a process that is gone and is removed.
/// Nothing here is trusted for security: it only coordinates TUFF's own
/// processes running as the same user.
public struct TUFFResidencyRecord: Codable, Equatable, Sendable {
    public enum Owner: String, Codable, Sendable {
        case app
        case backgroundServer = "background-server"
    }

    public var owner: Owner
    public var processID: Int32
    public var modelID: String
    public var estimatedBytes: UInt64
    /// Loopback port of a background server, so the app can ask it to unload
    /// an idle model. Nil for the app.
    public var controlPort: Int?

    public init(owner: Owner, processID: Int32 = getpid(), modelID: String,
                estimatedBytes: UInt64, controlPort: Int? = nil) {
        self.owner = owner
        self.processID = processID
        self.modelID = modelID
        self.estimatedBytes = estimatedBytes
        self.controlPort = controlPort
    }
}

public final class TUFFResidencyLease: @unchecked Sendable {
    public let record: TUFFResidencyRecord
    public let url: URL
    private let lock = NSLock()
    private var descriptor: Int32

    fileprivate init(record: TUFFResidencyRecord, url: URL, descriptor: Int32) {
        self.record = record
        self.url = url
        self.descriptor = descriptor
    }

    /// Removes the record and releases the lock. Safe to call twice.
    public func release() {
        lock.lock()
        defer { lock.unlock() }
        guard descriptor >= 0 else { return }
        unlink(url.path)
        flock(descriptor, LOCK_UN)
        close(descriptor)
        descriptor = -1
    }

    deinit { release() }
}

public struct TUFFResidencyRegistry: Sendable {
    public struct MemoryBusy: Error, Sendable {
        public let holders: [TUFFResidencyRecord]
    }
    public let directory: URL

    public init(directory: URL) {
        self.directory = directory
    }

    public static func directory(applicationSupport: URL) -> URL {
        applicationSupport.appendingPathComponent("TUFF/Runtime/Residency", isDirectory: true)
    }

    /// Reserve memory before loading. A shared lock makes check and reservation
    /// indivisible across the app and server, including simultaneous first loads.
    public func reserve(_ record: TUFFResidencyRecord, budgetBytes: UInt64) throws -> TUFFResidencyLease {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        let fd = open(directory.appendingPathComponent("admission.lock").path,
                      O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        defer { close(fd) }
        guard flock(fd, LOCK_EX) == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        defer { flock(fd, LOCK_UN) }
        let others = activeRecords()
        if case .blocked = TUFFResidencyAdmission.evaluate(
            neededBytes: record.estimatedBytes, others: others, budgetBytes: budgetBytes) {
            throw MemoryBusy(holders: others)
        }
        return try acquire(record)
    }

    /// Grants a temporary exclusive workspace only when every other live
    /// reservation belongs to the caller. The full-budget record prevents
    /// new model/session reservations until the workspace lease is released.
    public func reserveExclusive(_ record: TUFFResidencyRecord,
                                 allowing own: [TUFFResidencyLease]) throws -> TUFFResidencyLease {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        let fd = open(directory.appendingPathComponent("admission.lock").path,
                      O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        defer { close(fd) }
        guard flock(fd, LOCK_EX) == 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        defer { flock(fd, LOCK_UN) }
        let others = activeRecords(excluding: own)
        guard others.isEmpty else { throw MemoryBusy(holders: others) }
        return try acquire(record)
    }

    /// Writes and locks a record for this process.
    public func acquire(_ record: TUFFResidencyRecord) throws -> TUFFResidencyLease {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        let url = directory.appendingPathComponent(
            "\(record.owner.rawValue)-\(record.processID)-\(UUID().uuidString).json")
        let pending = url.appendingPathExtension("pending")
        let fd = open(pending.path, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            close(fd)
            unlink(pending.path)
            throw POSIXError(.EWOULDBLOCK)
        }
        let data = try JSONEncoder().encode(record)
        let written = data.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
        guard written == data.count else {
            flock(fd, LOCK_UN)
            close(fd)
            unlink(pending.path)
            throw POSIXError(.EIO)
        }
        // Publish only a fully written, locked record. Status readers must not
        // mistake the create-to-lock interval for a stale lease.
        guard rename(pending.path, url.path) == 0 else {
            close(fd)
            unlink(pending.path)
            throw POSIXError(.init(rawValue: errno) ?? .EIO)
        }
        return TUFFResidencyLease(record: record, url: url, descriptor: fd)
    }

    /// Records held by live processes. Stale files are deleted on the way.
    /// `excluding` leaves out leases this process holds itself.
    public func activeRecords(excluding own: [TUFFResidencyLease] = []) -> [TUFFResidencyRecord] {
        let ownPaths = Set(own.map(\.url.standardizedFileURL.path))
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: directory.path)
        else { return [] }
        var records: [TUFFResidencyRecord] = []
        for name in names.sorted() where name.hasSuffix(".json") {
            let url = directory.appendingPathComponent(name).standardizedFileURL
            if ownPaths.contains(url.path) { continue }
            let fd = open(url.path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { continue }
            defer { close(fd) }
            if flock(fd, LOCK_SH | LOCK_NB) == 0 {
                // Nobody holds it: the owner exited without releasing.
                flock(fd, LOCK_UN)
                unlink(url.path)
                continue
            }
            guard let data = try? Data(contentsOf: url),
                  let record = try? JSONDecoder().decode(TUFFResidencyRecord.self, from: data)
            else { continue }
            records.append(record)
        }
        return records
    }
}

/// Whether one more model fits beside the models other TUFF processes hold.
public enum TUFFResidencyAdmission: Equatable, Sendable {
    case admitted
    case blocked(by: [TUFFResidencyRecord], neededBytes: UInt64, budgetBytes: UInt64)

    public static func evaluate(neededBytes: UInt64,
                                others: [TUFFResidencyRecord],
                                budgetBytes: UInt64) -> TUFFResidencyAdmission {
        var total = neededBytes
        for record in others {
            let sum = total.addingReportingOverflow(record.estimatedBytes)
            total = sum.overflow ? .max : sum.partialValue
        }
        return total <= budgetBytes || others.isEmpty
            ? .admitted
            : .blocked(by: others, neededBytes: neededBytes, budgetBytes: budgetBytes)
    }
}
