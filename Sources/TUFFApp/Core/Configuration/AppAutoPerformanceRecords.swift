import Foundation
import Darwin
import Synchronization
import TUFFInferenceTypes
import TUFFModelCatalog
import TUFFModelMetadata
import TUFFRuntimeManagement
import TUFFRuntimeDescriptions

/// Stores timings and allocation choices only; prompts and responses never
/// enter tuning records. A source, engine, device, OS or schema change gives a
/// new identity. Workload classes keep cold/warm and image/text timings apart.
public enum AppAutoPerformanceRecords {
    private struct Entry: Codable { let identity: String; let date: Date; let observation: InferencePlanObservation }
    private struct Records: Codable { var schema = 1; var entries: [Entry] = [] }
    private static let lock = Mutex<Void>(())
    private static let maximumBytes = 1 << 20
    private static var url: URL {
        TUFFApplicationDirectories.applicationSupportURL().appendingPathComponent("TUFF/auto-performance.json")
    }
    static var revisionToken: String {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) else { return "empty" }
        return "\(url.path):\(attributes[.modificationDate] ?? ""):\(attributes[.size] ?? "")"
    }

    public static func identity(summary: ModelWeightLayoutSummary, device: TUFFDeviceCapabilities,
                                engineIdentity: String? = nil, modelDirectory: URL? = nil) -> String {
        let engine = engineIdentity ?? (try? InferenceEngineEnvironment.identity(architectureID: summary.architectureID)) ?? "development"
        func systemString(_ name: String) -> String {
            var count = 0
            guard sysctlbyname(name, nil, &count, nil, 0) == 0, count > 0, count <= 4096 else { return "unknown" }
            var bytes = [CChar](repeating: 0, count: count)
            guard sysctlbyname(name, &bytes, &count, nil, 0) == 0 else { return "unknown" }
            return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }
        var volume = "uninstalled"
        if let modelDirectory {
            var info = statfs()
            if statfs(modelDirectory.path, &info) == 0 {
                volume = "\(info.f_fsid.val.0):\(info.f_fsid.val.1):\(info.f_bsize)"
            }
        }
        let switches = ["TUFF_CONVERSATION_CACHE_MB", "TUFF_EXPERT_LOOKAHEAD", "TUFF_SMALL_BLOCK_PREFILL", "TUFF_SHARED_EXPERT_OVERLAP"]
            .map { ProcessInfo.processInfo.environment[$0] ?? "default" }.joined(separator: ":")
        let source = "auto-v3:\(TUFFVersion.current):\(summary.manifestSHA256):\(summary.weightsSHA256):\(engine):\(volume):\(switches):\(systemString("hw.model")):\(systemString("machdep.cpu.brand_string")):\(device.unifiedMemoryBytes):\(ProcessInfo.processInfo.operatingSystemVersionString)"
        return Sha256Verifier.hashData(Data(source.utf8))
    }

    public static func workloadKey(promptTokens: Int, reusedPrefix: Bool, imageInput: Bool,
                                   sampled: Bool = false) -> String {
        let length = promptTokens <= 512 ? "short" : (promptTokens <= 4096 ? "medium" : "long")
        let thermal = ProcessInfo.processInfo.thermalState.rawValue
        return "\(imageInput ? "image" : "text")-\(reusedPrefix ? "warm" : "cold")-\(length)-t\(thermal)-s\(sampled ? 1 : 0)"
    }

    public static func observations(identity: String) -> [InferencePlanObservation] {
        lock.withLock { _ in
            guard let records = try? read() else { return [] }
            let cutoff = Date().addingTimeInterval(-30 * 24 * 3600)
            return records.entries.filter { entry in
                entry.identity == identity && entry.date > cutoff
                    // Pressure quarantines a choice while the environment is
                    // recovering; it does not permanently ban it. Paging stays
                    // on record, so the selector can tell a one-off from a
                    // plan that keeps paging.
                    && (entry.observation.pressure == .normal
                        || entry.date > Date().addingTimeInterval(-900))
            }.map(\.observation)
        }
    }

    public static func record(identity: String, observation: InferencePlanObservation) {
        guard accepts(observation), identity.count == 64 else { return }
        lock.withLock { _ in
            // The app, `tuff prompt` and `tuff serve` all add records, so the
            // read-modify-write is also held against other processes.
            try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                     withIntermediateDirectories: true)
            let lockFD = open(url.path + ".lock", O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
            if lockFD >= 0 { flock(lockFD, LOCK_EX) }
            defer { if lockFD >= 0 { flock(lockFD, LOCK_UN); close(lockFD) } }
            // A corrupt/unknown record remains untouched; Auto safely falls
            // back to calculated allocation until the person resets it.
            guard var records = try? read() else { return }
            records.entries.append(.init(identity: identity, date: Date(), observation: observation))
            records.entries = Array(records.entries.suffix(512))
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            guard let data = try? encoder.encode(records), data.count <= maximumBytes else { return }
            do {
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try data.write(to: url, options: .atomic)
                _ = chmod(url.path, 0o600)
            } catch {}
        }
    }

    /// The settings a run used, in the form the planner names its candidates.
    public static func candidateID(contextTokens: Int, expertCacheSlots: Int,
                                   residentDenseLayers: Int, prefillChunkTokens: Int) -> String {
        "c\(contextTokens)-e\(expertCacheSlots)-d\(residentDenseLayers)-p\(prefillChunkTokens)"
    }

    /// Resident dense layers for a policy, or nil when Auto has not resolved
    /// one, in which case the run is not attributable to a candidate.
    public static func residentDenseLayers(_ policy: DenseResidencyPolicy,
                                           summary: ModelWeightLayoutSummary) -> Int? {
        switch policy {
        case .resident: summary.layout.denseGroupBytes.count
        case .bounded(let layers): layers.count
        case .automatic: nil
        }
    }

    /// A cancelled prefill can expose paging before producing any decode
    /// samples. Preserve that safety signal without inventing timing evidence.
    static func accepts(_ observation: InferencePlanObservation) -> Bool {
        guard observation.decodeTokens >= 0, observation.decodeSeconds.isFinite,
              observation.decodeSeconds >= 0, observation.timeToFirstTokenSeconds.isFinite,
              observation.timeToFirstTokenSeconds >= 0 else { return false }
        return observation.pressure != .normal || observation.swapGrowthBytes > 0
            || (observation.decodeTokens >= 2 && observation.decodeSeconds > 0)
    }

    public static func reset() throws {
        try lock.withLock { _ in
            var info = stat()
            if lstat(url.path, &info) != 0 {
                if errno == ENOENT { return }
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            guard info.st_mode & S_IFMT == S_IFREG, info.st_uid == getuid() else {
                throw CocoaError(.fileWriteNoPermission)
            }
            try JSONEncoder().encode(Records()).write(to: url, options: .atomic)
            guard chmod(url.path, 0o600) == 0 else { throw POSIXError(.EIO) }
        }
    }

    private static func read() throws -> Records {
        let fd = open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        if fd < 0 && errno == ENOENT { return Records() }
        guard fd >= 0 else { throw CocoaError(.fileReadNoPermission) }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == getuid(), info.st_size > 0, info.st_size <= maximumBytes else { throw CocoaError(.fileReadCorruptFile) }
        let input = FileHandle(fileDescriptor: fd, closeOnDealloc: false)
        let data = try input.read(upToCount: maximumBytes + 1) ?? Data()
        guard data.count <= maximumBytes else { throw CocoaError(.fileReadCorruptFile) }
        let records = try JSONDecoder().decode(Records.self, from: data)
        guard records.schema == 1, records.entries.count <= 512 else { throw CocoaError(.fileReadCorruptFile) }
        return records
    }
}

public final class AppAutoMemoryPressureMonitor: @unchecked Sendable {
    public static let shared = AppAutoMemoryPressureMonitor()
    private let state = Mutex((pressure: InferenceMemoryPressure.normal, warnings: UInt64(0)))
    private let source: DispatchSourceMemoryPressure
    private init() {
        source = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: .global(qos: .utility))
        source.setEventHandler { [weak self] in
            guard let self else { return }
            let event = source.data
            state.withLock {
                $0.pressure = event.contains(.critical) ? .critical : (event.contains(.warning) ? .warning : .normal)
                if $0.pressure != .normal { $0.warnings &+= 1 }
            }
        }
        source.resume()
    }
    public var snapshot: (pressure: InferenceMemoryPressure, warnings: UInt64) { state.withLock { $0 } }
    public static func swapBytes() -> UInt64 {
        var usage = xsw_usage(), count = MemoryLayout<xsw_usage>.size
        return sysctlbyname("vm.swapusage", &usage, &count, nil, 0) == 0 ? usage.xsu_used : 0
    }

    /// Net swap occupancy can stay flat while macOS repeatedly writes new
    /// pages. Count cumulative VM swap-outs as well; these are system-wide,
    /// so concurrent applications may conservatively disqualify a probe.
    public static func swapWrittenBytes() -> UInt64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        let bytes = stats.swapouts.multipliedReportingOverflow(by: UInt64(getpagesize()))
        return bytes.overflow ? UInt64.max : bytes.partialValue
    }

    static func pagingBytes(startUsed: UInt64, endUsed: UInt64,
                            startWritten: UInt64, endWritten: UInt64) -> UInt64 {
        max(endUsed > startUsed ? endUsed - startUsed : 0,
            endWritten > startWritten ? endWritten - startWritten : 0)
    }
}

/// Memory pressure and paging across one run, measured the same way by the
/// app, `tuff prompt` and `tuff serve` so their records are comparable.
public struct AppAutoObservationWindow: Sendable {
    let swapUsed: UInt64
    let swapWritten: UInt64
    let warnings: UInt64

    public static func begin() -> Self {
        Self(swapUsed: AppAutoMemoryPressureMonitor.swapBytes(),
             swapWritten: AppAutoMemoryPressureMonitor.swapWrittenBytes(),
             warnings: AppAutoMemoryPressureMonitor.shared.snapshot.warnings)
    }

    /// Pressure seen during the run: a warning raised meanwhile counts even
    /// if it has since cleared.
    public func pressure() -> InferenceMemoryPressure {
        let now = AppAutoMemoryPressureMonitor.shared.snapshot
        return now.warnings > warnings ? .warning : now.pressure
    }

    public func pagingBytes() -> UInt64 {
        AppAutoMemoryPressureMonitor.pagingBytes(
            startUsed: swapUsed, endUsed: AppAutoMemoryPressureMonitor.swapBytes(),
            startWritten: swapWritten, endWritten: AppAutoMemoryPressureMonitor.swapWrittenBytes())
    }

    public func observation(candidateID: String, workloadKey: String, decodeTokens: Int,
                            decodeSeconds: Double, timeToFirstTokenSeconds: Double) -> InferencePlanObservation {
        InferencePlanObservation(candidateID: candidateID, workloadKey: workloadKey,
            decodeTokens: decodeTokens, decodeSeconds: decodeSeconds,
            timeToFirstTokenSeconds: timeToFirstTokenSeconds,
            pressure: pressure(), swapGrowthBytes: pagingBytes())
    }
}
