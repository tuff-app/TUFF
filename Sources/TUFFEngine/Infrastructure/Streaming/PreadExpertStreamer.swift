import Darwin
import Foundation
import Metal

public struct ExpertIOAdviceResult: Sendable, Equatable {
    public let requested: Int
    public let failed: Int
    public let calls: Int
    public let bytes: UInt64
    public let skipped: Int
    public let maxCallNanos: UInt64

    public init(requested: Int,
                failed: Int,
                calls: Int? = nil,
                bytes: UInt64 = 0,
                skipped: Int = 0,
                maxCallNanos: UInt64 = 0) {
        self.requested = requested
        self.failed = failed
        self.calls = calls ?? requested
        self.bytes = bytes
        self.skipped = skipped
        self.maxCallNanos = maxCallNanos
    }

    public static func skipped(requested: Int, bytes: UInt64 = 0) -> ExpertIOAdviceResult {
        ExpertIOAdviceResult(requested: requested,
                             failed: 0,
                             calls: 0,
                             bytes: bytes,
                             skipped: requested)
    }

}

public struct ExpertCachePlan: Sendable, Equatable {
    public let experts: [Int]
    public let assignedSlots: [Int]
    public let misses: [Int]
    public let hits: Int
    public let purpose: ExpertReadPurpose

    public init(experts: [Int], assignedSlots: [Int], misses: [Int], hits: Int) {
        self.init(experts: experts, assignedSlots: assignedSlots, misses: misses,
                  hits: hits, purpose: .demand)
    }

    public init(experts: [Int], assignedSlots: [Int], misses: [Int], hits: Int,
                purpose: ExpertReadPurpose) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.misses = misses
        self.hits = hits
        self.purpose = purpose
    }
}

public typealias ExpertCachePolicy = TUFFRuntimeDescriptions.ExpertCachePolicy

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }

    public let layout: StreamLayout
    public let slotCount: Int
    private let slotAllocationSize: Int
    public let cachePolicy: ExpertCachePolicy

    private let fd: Int32
    private let bypassesFileCache: Bool
    private let readAheadLock = NSLock()
    private var readAheadGeneration = 0
    private var readAheadBytes: UInt64 = 0
    private let slotPointers: [UnsafeMutableRawPointer]
    private let slotBuffers: [MTLBuffer]

    private var nextSlot = 0
    private let cursorLock = NSLock()

    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var slotUnusedPrefetch: [Bool]
    private var expertDemandUseCount: [Int]
    private var expertPredictionUseCount: [Int]
    /// Whether each expert last left the cache to make room for a prediction.
    private var evictedForPrediction: [Bool]
    private var useClock = 0
    private let cacheLock = NSLock()
    private let metricsLock = NSLock()
    private var metrics = ExpertReadMetrics()

    public var readMetrics: ExpertReadMetrics {
        metricsLock.lock()
        defer { metricsLock.unlock() }
        return metrics
    }

    public convenience init(layout: StreamLayout,
                            device: MTLDevice,
                            slotCount: Int,
                            cachePolicy: ExpertCachePolicy = .lfu) throws {
        try self.init(layout: layout,
                      device: device,
                      slotCount: slotCount,
                      cachePolicy: cachePolicy,
                      fileDescriptor: nil)
    }

    package init(layout: StreamLayout,
                 device: MTLDevice,
                 slotCount: Int,
                 cachePolicy: ExpertCachePolicy = .lfu,
                 fileDescriptor: Int32?,
                 bypassFilesystemCache: Bool = false) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        self.layout = layout
        self.slotCount = slotCount
        self.cachePolicy = cachePolicy
        let pageSize = Int(getpagesize())

        let openedFD = fileDescriptor.map { fcntl($0, F_DUPFD_CLOEXEC, 0) }
            ?? open(layout.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.fd = openedFD
        var closeFDOnFailure = true
        defer { if closeFDOnFailure { close(openedFD) } }
        if bypassFilesystemCache { _ = fcntl(openedFD, F_NOCACHE, 1) }
        self.bypassesFileCache = bypassFilesystemCache

        var fileStats = stat()
        guard fstat(openedFD, &fileStats) == 0,
              (fileStats.st_mode & S_IFMT) == S_IFREG,
              fileStats.st_size >= 0 else {
            throw StreamerError.openFailed(
                path: layout.path, errno: errno == 0 ? EINVAL : errno)
        }
        let (required, requiredOverflow) = layout.streamOffset
            .addingReportingOverflow(layout.streamSize)
        guard !requiredOverflow, UInt64(fileStats.st_size) >= required else {
            throw StreamerError.sizeMismatch(
                expected: requiredOverflow ? UInt64.max : required,
                actual: UInt64(fileStats.st_size))
        }
        guard layout.expertStride > 0,
              layout.expertStride <= UInt64(Int.max - (pageSize - 1)) else {
            throw StreamerError.invalidIOSplitConfiguration(
                "expertStride \(layout.expertStride) is not addressable")
        }

        let allocationSize = ((Int(layout.expertStride) + pageSize - 1) / pageSize) * pageSize
        self.slotAllocationSize = allocationSize
        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)

        func unwind() {
            for index in buffers.count..<pointers.count {
                free(pointers[index])
            }
        }

        for _ in 0..<slotCount {
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, Self.scratchAlignment, allocationSize)
            guard result == 0, let pointer = raw else {
                unwind()
                throw StreamerError.allocFailed(errno: result)
            }
            pointers.append(pointer)
            nonisolated(unsafe) let capturedPointer = pointer
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: allocationSize,
                options: .storageModeShared,
                deallocator: { _, _ in free(capturedPointer) })
            else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            buffers.append(buffer)
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.slotUnusedPrefetch = [Bool](repeating: false, count: slotCount)
        self.expertDemandUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        self.expertPredictionUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        self.evictedForPrediction = [Bool](repeating: false, count: max(1, layout.expertsPerLayer))
        closeFDOnFailure = false
    }

    deinit {
        close(fd)
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        cursorLock.lock()
        let slot = nextSlot
        nextSlot = (nextSlot + 1) % slotCount
        cursorLock.unlock()
        return try loadExpert(layer: layer, expert: expert, slot: slot)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int,
                           purpose: ExpertReadPurpose = .demand) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        var bytes: UInt64 = 0
        var succeeded = false
        defer {
            metricsLock.lock()
            metrics.record(purpose: purpose, bytes: bytes, succeeded: succeeded)
            metricsLock.unlock()
        }
        try readFull(
            into: slotPointers[slot],
            fileOffset: layout.streamOffset + regionOffset,
            count: Int(layout.expertStride), bytesRead: &bytes)
        succeeded = true
        return (slotBuffers[slot], 0, layout.expertStride)
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    public func planExpertsCached(experts: [Int],
                                  avoidingSlots: Set<Int> = []) -> ExpertCachePlan {
        planExpertsCached(experts: experts, avoidingSlots: avoidingSlots, purpose: .demand)
    }

    public func planExpertsCached(experts: [Int],
                                  avoidingSlots: Set<Int> = [],
                                  purpose: ExpertReadPurpose) -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots, purpose: purpose) else {
            preconditionFailure("expert cache cannot place requested misses")
        }
        return plan
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            avoidingSlots: Set<Int> = []) -> ExpertCachePlan? {
        planExpertsCachedIfPossible(experts: experts, avoidingSlots: avoidingSlots, purpose: .demand)
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            avoidingSlots: Set<Int> = [],
                                            purpose: ExpertReadPurpose) -> ExpertCachePlan? {
        makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots, purpose: purpose)
    }

    private func makeExpertCachePlan(experts: [Int],
                                     avoidingSlots rawAvoidingSlots: Set<Int>,
                                     purpose: ExpertReadPurpose) -> ExpertCachePlan? {
        precondition(experts.count <= slotCount,
                     "expert cache needs at least \(experts.count) slots")
        let avoidingSlots = Set(rawAvoidingSlots.filter { $0 >= 0 && $0 < slotCount })

        cacheLock.lock()
        defer { cacheLock.unlock() }

        let clock = useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)

        for index in experts.indices {
            for slot in 0..<slotCount
                where !reserved[slot] && slotExpert[slot] == experts[index] {
                assignedSlots[index] = slot
                reserved[slot] = true
                break
            }
        }
        for slot in avoidingSlots where !reserved[slot] {
            reserved[slot] = true
        }

        let misses = experts.indices.filter { assignedSlots[$0] == -1 }
        guard misses.count <= reserved.lazy.filter({ !$0 }).count else { return nil }
        if purpose == .demand {
            let polluted = misses.reduce(UInt64(0)) { count, index in
                let expert = experts[index]
                return expert >= 0 && expert < evictedForPrediction.count
                    && evictedForPrediction[expert] ? count + 1 : count
            }
            if polluted > 0 {
                metricsLock.lock()
                metrics.pollutedDemandMisses += polluted
                metricsLock.unlock()
            }
        }

        useClock = clock
        for expert in experts where expert >= 0 && expert < expertDemandUseCount.count {
            if purpose == .demand {
                if expertDemandUseCount[expert] < Int.max { expertDemandUseCount[expert] += 1 }
            } else {
                if expertPredictionUseCount[expert] < Int.max { expertPredictionUseCount[expert] += 1 }
            }
        }
        metricsLock.lock()
        if purpose == .demand { metrics.demandRequests += UInt64(experts.count) }
        else { metrics.predictionRequests += UInt64(experts.count) }
        metricsLock.unlock()
        for slot in assignedSlots where slot >= 0 {
            slotLastUse[slot] = clock
            if purpose == .demand && slotUnusedPrefetch[slot] {
                metricsLock.lock()
                metrics.usefulPrefetchReads += 1
                metricsLock.unlock()
                slotUnusedPrefetch[slot] = false
            }
        }
        // Decode selects only a small top-k (four or eight experts). Scanning
        // once per miss avoids allocating and sorting every unreserved cache
        // slot, which matters once Auto grows caches beyond 32 slots. Ties keep
        // the lowest slot, matching the old iteration order deterministically.
        for index in misses {
            var selectedSlot: Int?
            for slot in 0..<slotCount where !reserved[slot] {
                guard let current = selectedSlot else {
                    selectedSlot = slot
                    continue
                }
                if shouldEvictSlot(slot, before: current) {
                    selectedSlot = slot
                }
            }
            guard let slot = selectedSlot else { return nil }
            assignedSlots[index] = slot
            reserved[slot] = true
            if slotUnusedPrefetch[slot] {
                metricsLock.lock()
                metrics.unusedPrefetchEvictions += 1
                metricsLock.unlock()
            }
            slotUnusedPrefetch[slot] = false
            let evicted = slotExpert[slot]
            if evicted >= 0 && evicted < evictedForPrediction.count {
                evictedForPrediction[evicted] = purpose == .prefetch
            }
            let incoming = experts[index]
            if incoming >= 0 && incoming < evictedForPrediction.count {
                evictedForPrediction[incoming] = false
            }
            slotExpert[slot] = -1
            slotLastUse[slot] = clock
        }

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            misses: misses,
            hits: experts.count - misses.count, purpose: purpose)
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(plan, purpose: plan.purpose)
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan,
                                       purpose: ExpertReadPurpose) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        let readPurpose = purpose
        guard !plan.misses.isEmpty else { return expertCachePlanBuffers(plan) }

        // A single read already runs on the caller's I/O worker. Only fan
        // out when there is more than one independent SSD read to perform.
        if plan.misses.count == 1 {
            let index = plan.misses[0]
            _ = try loadExpert(layer: 0, expert: plan.experts[index],
                               slot: plan.assignedSlots[index], purpose: readPurpose)
        } else {
            try loadCacheMissesConcurrently(plan, purpose: readPurpose)
        }

        cacheLock.lock()
        for index in plan.misses {
            slotExpert[plan.assignedSlots[index]] = plan.experts[index]
            slotUnusedPrefetch[plan.assignedSlots[index]] = readPurpose == .prefetch
        }
        cacheLock.unlock()

        return expertCachePlanBuffers(plan)
    }

    private func loadCacheMissesConcurrently(_ plan: ExpertCachePlan,
                                              purpose: ExpertReadPurpose) throws {
        let errorLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: plan.misses.count) { missOffset in
            let index = plan.misses[missOffset]
            do {
                _ = try self.loadExpert(
                    layer: 0,
                    expert: plan.experts[index],
                    slot: plan.assignedSlots[index], purpose: purpose)
            } catch {
                errorLock.lock()
                if firstError == nil { firstError = error }
                errorLock.unlock()
            }
        }
        if let firstError { throw firstError }
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], UInt64(0), layout.expertStride)
        }
    }

    public func adviseExpertCachePlanMisses(_ plan: ExpertCachePlan) -> ExpertIOAdviceResult {
        let experts = plan.misses.map { plan.experts[$0] }
        return adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    /// Reads the given experts that are not in a cache slot into the file
    /// cache in the background, in order, a few at a time, and returns at
    /// once. The cache-slot reads that follow then copy from memory, and the
    /// SSD keeps several reads in flight instead of one tile's worth. A later
    /// call stops what is left of an earlier one. Nothing to do when this
    /// streamer bypasses the file cache.
    public func readAheadExpertMisses(experts: [Int], concurrency: Int = 8,
                                      lowPriority: Bool = false) {
        guard !bypassesFileCache, concurrency > 0 else { return }
        cacheLock.lock()
        let misses = experts.filter { !slotExpert.contains($0) }
        cacheLock.unlock()
        let ranges = expertAdviceRanges(experts: misses)
        guard !ranges.isEmpty else { return }
        readAheadLock.lock()
        readAheadGeneration &+= 1
        let generation = readAheadGeneration
        readAheadLock.unlock()
        let work = ReadAheadWork(ranges: ranges)
        let chunk = 1 << 20
        for _ in 0..<min(concurrency, ranges.count) {
            // Strong capture: the file descriptor stays open until every
            // worker has finished.
            (lowPriority ? Self.lookaheadQueue : Self.readAheadQueue).async { [self] in
                let buffer = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: Int(getpagesize()))
                defer { buffer.deallocate() }
                while let range = nextReadAheadRange(work, generation: generation) {
                    var done: UInt64 = 0
                    while done < range.count {
                        let length = Int(min(UInt64(chunk), range.count - done))
                        let count = pread(fd, buffer, length, off_t(range.offset + done))
                        if count < 0, errno == EINTR { continue }
                        guard count > 0 else { break }
                        done += UInt64(count)
                    }
                    readAheadLock.lock()
                    readAheadBytes &+= done
                    readAheadLock.unlock()
                }
            }
        }
    }

    /// Bytes read ahead into the file cache so far. Diagnostic.
    public var readAheadBytesRead: UInt64 {
        readAheadLock.lock()
        defer { readAheadLock.unlock() }
        return readAheadBytes
    }

    private static let readAheadQueue = DispatchQueue(
        label: "TUFF expert read-ahead", qos: .userInitiated, attributes: .concurrent)
    /// Predicted experts for a later layer read at utility priority, which
    /// macOS throttles behind the reads the current layer is waiting on.
    private static let lookaheadQueue = DispatchQueue(
        label: "TUFF expert lookahead", qos: .utility, attributes: .concurrent)

    private final class ReadAheadWork: @unchecked Sendable {
        let ranges: [(offset: UInt64, count: UInt64)]
        var next = 0
        init(ranges: [(offset: UInt64, count: UInt64)]) { self.ranges = ranges }
    }

    private func nextReadAheadRange(_ work: ReadAheadWork, generation: Int)
        -> (offset: UInt64, count: UInt64)? {
        readAheadLock.lock()
        defer { readAheadLock.unlock() }
        guard generation == readAheadGeneration, work.next < work.ranges.count else { return nil }
        defer { work.next += 1 }
        return work.ranges[work.next]
    }

    public func adviseExperts(experts: [Int]) -> ExpertIOAdviceResult {
        adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExpertMisses(experts: [Int]) -> ExpertIOAdviceResult {
        cacheLock.lock()
        let misses = experts.filter { !slotExpert.contains($0) }
        cacheLock.unlock()
        return adviseRanges(expertAdviceRanges(experts: misses), requested: misses.count)
    }

    static func coalescedAdjacentAdviceRanges(_ ranges: [(offset: UInt64, count: UInt64)])
        -> [(offset: UInt64, count: UInt64)] {
        let sorted = ranges.filter { $0.count > 0 }.sorted {
            $0.offset == $1.offset ? $0.count < $1.count : $0.offset < $1.offset
        }
        var result: [(offset: UInt64, count: UInt64)] = []
        for range in sorted {
            guard var last = result.popLast() else {
                result.append(range)
                continue
            }
            let lastEnd = last.offset &+ last.count
            let rangeEnd = range.offset &+ range.count
            if range.offset <= lastEnd {
                last.count = max(lastEnd, rangeEnd) - last.offset
                result.append(last)
            } else {
                result.append(last)
                result.append(range)
            }
        }
        return result
    }

    private func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        if cachePolicy == .lru {
            return slotLastUse[lhs] < slotLastUse[rhs]
        }
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        // Keep the qualified 6.0.2 ranking. Demand-only frequency and a
        // separate unused-prefetch priority did not qualify end to end.
        // Demand and prediction histories remain separate for accounting.
        let lhsCount = combinedPlanFrequency(expert: lhsExpert)
        let rhsCount = combinedPlanFrequency(expert: rhsExpert)
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    private func combinedPlanFrequency(expert: Int) -> Int {
        guard expert >= 0 && expert < expertDemandUseCount.count else { return 0 }
        let sum = expertDemandUseCount[expert].addingReportingOverflow(expertPredictionUseCount[expert])
        return sum.overflow ? Int.max : sum.partialValue
    }

    func diagnosticExpertUseCounts(expert: Int) -> (demand: Int, prediction: Int) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard expert >= 0 && expert < expertDemandUseCount.count else { return (0, 0) }
        return (expertDemandUseCount[expert], expertPredictionUseCount[expert])
    }

    private func expertAdviceRanges(experts: [Int]) -> [(offset: UInt64, count: UInt64)] {
        experts.compactMap { expert in
            let regionOffset = layout.expertOffset(layer: 0, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else { return nil }
            return (layout.streamOffset + regionOffset, layout.expertStride)
        }
    }

    private func adviseRanges(_ ranges: [(offset: UInt64, count: UInt64)],
                              requested: Int) -> ExpertIOAdviceResult {
        let coalesced = Self.coalescedAdjacentAdviceRanges(ranges)
        var failed = 0
        var bytes: UInt64 = 0
        var maxCallNanos: UInt64 = 0
        for range in coalesced {
            let result = RDAdvice.call(fd: fd, offset: range.offset, byteCount: range.count)
            if !result.succeeded { failed += 1 }
            bytes &+= result.requestedBytes
            maxCallNanos = max(maxCallNanos, result.elapsedNanos)
        }
        return ExpertIOAdviceResult(
            requested: requested,
            failed: failed,
            calls: coalesced.count,
            bytes: bytes,
            maxCallNanos: maxCallNanos)
    }

    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int, bytesRead: inout UInt64) throws {
        var filled = 0
        while filled < count {
            let readCount = pread(
                fd,
                destination.advanced(by: filled),
                count - filled,
                off_t(fileOffset) + off_t(filled))
            if readCount < 0 {
                if errno == EINTR { continue }
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
            bytesRead += UInt64(readCount)
        }
    }

    /// Model-derived CPU-side scratch owned by this streamer: its aligned
    /// slot allocation. Diagnostic metadata for residency reporting, not an
    /// ownership or lifetime API.
    public var diagnosticSlotScratchBytes: UInt64 {
        UInt64(slotCount) * UInt64(slotAllocationSize)
    }
}
