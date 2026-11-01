import Foundation
import Darwin
import Metal

/// One `mmap`'d window of `model_weights.bin`, wrapped in an `MTLBuffer`.
final class ResidentBuffer {
    let buffer: MTLBuffer
    let fileOffset: UInt64
    let residentSize: UInt64
    let mappedLength: Int

    /// `mmap` the page-aligned window covering `[fileOffset, fileOffset + residentSize)`
    /// inside the file at `fileURL`. The wrapped `MTLBuffer` starts at the
    /// (sub-page) offset within the mapping so the resident bytes start at
    /// byte 0 of the buffer.
    init(fileURL: URL,
                fileOffset: UInt64,
                residentSize: UInt64,
                device: MTLDevice,
                fileDescriptor: Int32? = nil) throws {
        let pageSize = Int(getpagesize())

        let fd = fileDescriptor.map { fcntl($0, F_DUPFD_CLOEXEC, 0) }
            ?? open(fileURL.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))", errno: errno)
        }
        defer { close(fd) }

        var info = stat()
        guard fstat(fd, &info) == 0 else {
            throw ModelError.posixFailed(call: "fstat(\(fileURL.path))", errno: errno)
        }
        guard (info.st_mode & S_IFMT) == S_IFREG, info.st_size >= 0 else {
            throw ModelError.indexCorrupt(detail: "resident weights are not a regular file")
        }
        let (fileEnd, endOverflow) = fileOffset.addingReportingOverflow(residentSize)
        guard residentSize > 0, !endOverflow, fileEnd <= UInt64(info.st_size),
              fileOffset <= UInt64(Int64.max) else {
            throw ModelError.indexCorrupt(detail: "resident mapping exceeds the weights file")
        }

        let alignedOffset = (fileOffset / UInt64(pageSize)) * UInt64(pageSize)
        let sliceShift = Int(fileOffset - alignedOffset)
        guard residentSize <= UInt64(Int.max - sliceShift) else {
            throw ModelError.indexCorrupt(detail: "resident mapping exceeds addressable memory")
        }
        let mappedLen = sliceShift + Int(residentSize)
        let mapped = mmap(nil, mappedLen, PROT_READ, MAP_PRIVATE,
                          fd, off_t(alignedOffset))
        if mapped == MAP_FAILED {
            throw ModelError.posixFailed(call: "mmap", errno: errno)
        }
        let base = mapped!

        // Random, not sequential. Sequential advice was measured on the model
        // that suffers most from re-faulting — Gemma 4 12B QAT, 10 GB of dense
        // weights on a 16 GB Mac — and was no better there (0.049 and 0.063
        // against 0.072 tokens per second), because it also invites the kernel
        // to drop pages behind the read position, which hurts every model that
        // does fit in memory.
        _ = posix_madvise(base, mappedLen, POSIX_MADV_RANDOM)

        let sliceStart = base.advanced(by: sliceShift)

        // Capture pointer + length for the deallocator. Do NOT capture self
        // here — that would create a retain cycle through the MTLBuffer.
        nonisolated(unsafe) let captureBase = base
        let captureLen = mappedLen
        guard let buf = device.makeBuffer(
            bytesNoCopy: sliceStart,
            length: Int(residentSize),
            options: .storageModeShared,
            deallocator: { _, _ in
                munmap(captureBase, captureLen)
            }
        ) else {
            munmap(base, mappedLen)
            throw ModelError.residentBufferWrapFailed
        }

        self.buffer = buf
        self.fileOffset = fileOffset
        self.residentSize = residentSize
        self.mappedLength = mappedLen
    }
}

/// Tensor-safe collection of resident mappings. Metal limits the length of a
/// single buffer even when it wraps existing virtual memory, so a dense model
/// can legitimately be larger than `MTLDevice.maxBufferLength`. Regions are
/// cut only between complete index entries: a tensor's weights, scales, and
/// biases always remain in the same buffer.
final class ResidentBufferSet {
    struct RegionPlan: Equatable, Sendable {
        let fileOffset: UInt64
        let size: UInt64
        let tensorNames: [String]

        var endOffset: UInt64 { fileOffset + size }
    }

    struct ResolvedEntry {
        let buffer: MTLBuffer
        let regionFileOffset: UInt64
        var bufferOffset: UInt64 = 0
    }

    /// Target size of one mmap'd resident region when splitting is safe.
    static let preferredRegionBytes: UInt64 = 64 << 20

    /// Splitting is only safe while the whole resident payload can stay
    /// resident. Above this share of physical memory the model is mapped as
    /// one region instead. See `shouldSplitRegions`.
    static let splitPayloadMemoryShareDenominator: UInt64 = 4

    /// Whether to cut the resident weights into `preferredRegionBytes` regions.
    ///
    /// Splitting makes every dispatch cheaper, because Metal's per-encoder cost
    /// for a bound buffer grows with that buffer's size. That is worth a lot to
    /// a model whose weights fit in memory: Gemma 4 E2B went from 28.4 to 42.3
    /// tokens per second.
    ///
    /// It is ruinous to a model whose weights do not fit. Gemma 4 12B QAT is
    /// 10 GB of dense weights on a 16 GB Mac, and splitting it collapsed decode
    /// from 6.97 to 0.034 tokens per second — every token re-reading the model
    /// from SSD. One large mapping is paged lazily by the kernel; many separate
    /// Metal buffers are not, and the whole set is forced resident.
    ///
    /// So the split is applied only when the payload is a small enough share of
    /// physical memory to stay cached alongside everything else.
    static func shouldSplitRegions(payloadSize: UInt64,
                                   physicalMemory: UInt64) -> Bool {
        guard physicalMemory > 0 else { return false }
        return payloadSize <= physicalMemory / splitPayloadMemoryShareDenominator
    }

    private let denseLayers: DenseLayerStorage?
    var stagesDenseLayers: Bool { denseLayers != nil }
    var denseStagingBytes: UInt64 { denseLayers?.plan.stagingBytes ?? 0 }
    var streamedProjectionLayers: Set<Int> { Set(denseLayers?.plan.layers.keys.map { $0 } ?? []) }
    func prepareDenseLayer(_ layer: Int) throws { try denseLayers?.prepare(layer: layer) }

    private let buffers: [ResidentBuffer]
    var metalBuffers: [MTLBuffer] { buffers.map(\.buffer) }
    private let regionByTensorName: [String: Int]

    init(fileURL: URL,
         index: ResidentIndex,
         device: MTLDevice,
         fileDescriptor: Int32? = nil,
         denseResidency: DenseResidencyPolicy = .resident,
         numLayers: Int = 0) throws {
        let streamedNames: Set<String>
        let selectedResidency: DenseResidencyPolicy
        if case .automatic(let budget) = denseResidency {
            if index.header.residentSize <= budget {
                selectedResidency = .resident
            } else {
                let all = try DenseLayerStorage.makePlan(index: index, residentLayers: [],
                    numLayers: numLayers, maximumBytes: UInt64(device.maxBufferLength))
                let fixed = index.entries.values.filter { DenseLayerStorage.layerNumber($0.name) == nil }
                let fixedPlans = try Self.planRegions(entries: fixed,
                    payloadOffset: index.header.indexSize, payloadSize: index.header.residentSize,
                    maximumLength: 1, hardMaximumLength: UInt64(device.maxBufferLength))
                func mappedBytes(_ plans: [RegionPlan]) throws -> UInt64 {
                    let page = UInt64(getpagesize())
                    var total: UInt64 = 0
                    for plan in plans {
                        let bytes = plan.size.addingReportingOverflow(plan.fileOffset % page)
                        let rounded = bytes.partialValue.addingReportingOverflow(page - 1)
                        guard !bytes.overflow, !rounded.overflow else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        let sum = total.addingReportingOverflow(rounded.partialValue / page * page)
                        guard !sum.overflow else { throw ModelError.residentBufferWrapFailed }
                        total = sum.partialValue
                    }
                    return total
                }
                var used = try mappedBytes(fixedPlans)
                let baseline = used.addingReportingOverflow(all.stagingBytes)
                guard !baseline.overflow, baseline.partialValue <= budget else {
                    throw ModelError.indexCorrupt(detail: "weight budget cannot hold global tables and one dense layer")
                }
                used = baseline.partialValue
                var retained: Set<Int> = []
                // Keep complete layers; the staging allocation remains reserved.
                // Local tuning can request other subsets through .bounded.
                for layer in 0..<numLayers {
                    guard let slices = all.layers[layer] else { continue }
                    let retainedPlans = try Self.planRegions(
                        entries: slices.compactMap { index.entries[$0.name] },
                        payloadOffset: index.header.indexSize, payloadSize: index.header.residentSize,
                        maximumLength: 1, hardMaximumLength: UInt64(device.maxBufferLength))
                    let bytes = try mappedBytes(retainedPlans)
                    let total = used.addingReportingOverflow(bytes)
                    if !total.overflow && total.partialValue <= budget {
                        retained.insert(layer)
                        used = total.partialValue
                    }
                }
                selectedResidency = .bounded(residentLayers: retained)
            }
        } else { selectedResidency = denseResidency }
        switch selectedResidency {
        case .automatic:
            preconditionFailure("unresolved dense residency")
        case .resident:
            denseLayers = nil
            streamedNames = []
        case .bounded(let layers):
            guard let fileDescriptor else { throw ModelError.residentBufferWrapFailed }
            if layers.count == numLayers && layers.allSatisfy({ $0 >= 0 && $0 < numLayers }) {
                denseLayers = nil
                streamedNames = []
            } else {
                let storage = try DenseLayerStorage(index: index, residentLayers: layers,
                    numLayers: numLayers, device: device, fileDescriptor: fileDescriptor)
                denseLayers = storage
                streamedNames = Set(storage.plan.layers.values.flatMap { $0 }.map(\.name))
            }
        }
        // Metal's per-encoder cost for a bound buffer grows with the buffer's
        // size (measured: ~3 us for a 64 KB or 256 MB buffer, ~10-25 us for a
        // 1-2.6 GB buffer, per dispatch). Decode issues hundreds of dispatches
        // per token, so weights that comfortably fit in memory are mapped as
        // many moderately sized regions. A model too large for that is mapped
        // as one region and left to the kernel's paging; see
        // `shouldSplitRegions`.
        let split = Self.shouldSplitRegions(
            payloadSize: index.header.residentSize,
            physicalMemory: ProcessInfo.processInfo.physicalMemory)
        let plans = try Self.planRegions(
            entries: index.entries.values.filter { !streamedNames.contains($0.name) },
            payloadOffset: index.header.indexSize,
            payloadSize: index.header.residentSize,
            maximumLength: denseLayers != nil ? 1 : (split
                ? min(UInt64(device.maxBufferLength), Self.preferredRegionBytes)
                : UInt64(device.maxBufferLength)),
            hardMaximumLength: UInt64(device.maxBufferLength))

        var opened: [ResidentBuffer] = []
        opened.reserveCapacity(plans.count)
        var lookup: [String: Int] = [:]
        lookup.reserveCapacity(index.entries.count)
        for (regionIndex, plan) in plans.enumerated() {
            opened.append(try ResidentBuffer(
                fileURL: fileURL,
                fileOffset: plan.fileOffset,
                residentSize: plan.size,
                device: device,
                fileDescriptor: fileDescriptor))
            for name in plan.tensorNames {
                lookup[name] = regionIndex
            }
        }
        self.buffers = opened
        self.regionByTensorName = lookup
    }

    func resolve(tensorName: String) throws -> ResolvedEntry {
        if let resolved = denseLayers?.resolve(tensorName) { return resolved }
        guard let regionIndex = regionByTensorName[tensorName] else {
            throw ModelError.tensorNotFound(name: tensorName)
        }
        let region = buffers[regionIndex]
        return ResolvedEntry(buffer: region.buffer,
                             regionFileOffset: region.fileOffset)
    }

    /// Pure planning step kept separate so boundary and oversized-tensor
    /// behavior can be tested without allocating large Metal buffers.
    /// `maximumLength` is the preferred cap for a region; a single tensor
    /// wider than it still gets its own region as long as it fits Metal's
    /// `hardMaximumLength` (defaults to the preferred cap, so existing
    /// callers keep their strict-limit semantics).
    static func planRegions(entries: [ResidentIndexEntry],
                            payloadOffset: UInt64,
                            payloadSize: UInt64,
                            maximumLength: UInt64,
                            hardMaximumLength: UInt64? = nil) throws -> [RegionPlan] {
        let hardMaximumLength = hardMaximumLength ?? maximumLength
        guard maximumLength > 0, hardMaximumLength >= maximumLength else {
            throw ModelError.residentBufferWrapFailed
        }
        let (payloadEnd, payloadOverflow) = payloadOffset.addingReportingOverflow(payloadSize)
        guard !payloadOverflow else {
            throw ModelError.indexCorrupt(detail: "resident payload range overflows UInt64")
        }

        struct SpannedEntry {
            let entry: ResidentIndexEntry
            let start: UInt64
            let end: UInt64
        }
        func fieldEnd(_ offset: UInt64, _ size: UInt64,
                      name: String, field: String) throws -> UInt64? {
            if size == 0 {
                guard offset == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "\(name).\(field) has an absent nonzero offset")
                }
                return nil
            }
            let (value, overflow) = offset.addingReportingOverflow(size)
            guard !overflow, offset >= payloadOffset, value <= payloadEnd else {
                throw ModelError.indexCorrupt(
                    detail: "\(name).\(field) exceeds the resident payload")
            }
            return value
        }

        let spanned = try entries.map { entry -> SpannedEntry in
            var starts: [UInt64] = []
            var ends: [UInt64] = []
            if let value = try fieldEnd(entry.fileOffset, entry.sizeBytes,
                                        name: entry.name, field: "weights") {
                starts.append(entry.fileOffset); ends.append(value)
            }
            if let value = try fieldEnd(entry.scaleOffset, entry.scaleSize,
                                        name: entry.name, field: "scales") {
                starts.append(entry.scaleOffset); ends.append(value)
            }
            if let value = try fieldEnd(entry.biasOffset, entry.biasSize,
                                        name: entry.name, field: "biases") {
                starts.append(entry.biasOffset); ends.append(value)
            }
            guard let start = starts.min(), let finish = ends.max() else {
                throw ModelError.indexCorrupt(detail: "\(entry.name) has no resident bytes")
            }
            guard finish - start <= hardMaximumLength else {
                throw ModelError.indexCorrupt(
                    detail: "\(entry.name) spans \(finish - start) bytes, exceeding Metal's maximum buffer length \(hardMaximumLength)")
            }
            return SpannedEntry(entry: entry, start: start, end: finish)
        }.sorted {
            if $0.start != $1.start { return $0.start < $1.start }
            if $0.end != $1.end { return $0.end < $1.end }
            return $0.entry.name < $1.entry.name
        }

        var plans: [RegionPlan] = []
        var currentStart: UInt64?
        var currentEnd: UInt64 = 0
        var currentNames: [String] = []
        func appendCurrent() {
            guard let start = currentStart else { return }
            plans.append(RegionPlan(fileOffset: start,
                                    size: currentEnd - start,
                                    tensorNames: currentNames))
        }
        for item in spanned {
            if let start = currentStart {
                let proposedEnd = max(currentEnd, item.end)
                if proposedEnd - start <= maximumLength {
                    currentEnd = proposedEnd
                    currentNames.append(item.entry.name)
                    continue
                }
                appendCurrent()
            }
            currentStart = item.start
            currentEnd = item.end
            currentNames = [item.entry.name]
        }
        appendCurrent()
        return plans
    }
}
