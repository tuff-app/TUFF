import Foundation
import Darwin
import Metal

/// Best-effort page-cache warm-up. This never requests Metal residency or wires
/// memory. A request cancels it without waiting for a page fault already in flight.
public final class ResidentWarmup: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var completedChecksum: UInt8 = 0

    /// Result published when the pass finishes or is cancelled. Exposing the
    /// value keeps the page reads observable in optimized builds.
    public var checksum: UInt8 {
        lock.lock()
        defer { lock.unlock() }
        return completedChecksum
    }

    static func eligible(payloadSize: UInt64, physicalMemory: UInt64,
                         stagesDenseLayers: Bool, isDense: Bool) -> Bool {
        physicalMemory > 0 && payloadSize > 0 && payloadSize <= physicalMemory / 2
            && !stagesDenseLayers && isDense
    }

    public func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    // The callback seam lets tests stop a pass during an individual page read.
    func run(length: Int, startPageOffset: Int, pageSize: Int, readByte: (Int) -> Void) {
        guard length > 0, pageSize > 0, startPageOffset >= 0,
              startPageOffset < pageSize else { return }
        var offset = 0
        while offset < length {
            if isCancelled { return }
            readByte(offset)
            let step = offset == 0 ? pageSize - startPageOffset : pageSize
            guard step < length - offset else { return }
            offset += step
        }
    }

    static func start(buffers: [MTLBuffer]) -> ResidentWarmup {
        let warmup = ResidentWarmup()
        let retained = RetainedBuffers(buffers)
        DispatchQueue.global(qos: .background).async {
            let pageSize = Int(getpagesize())
            var checksum: UInt8 = 0
            for buffer in retained.buffers {
                let contents = buffer.contents()
                let startPageOffset = Int(UInt(bitPattern: contents) % UInt(pageSize))
                warmup.run(length: buffer.length, startPageOffset: startPageOffset,
                           pageSize: pageSize) { offset in
                    // One byte per VM page faults the file-backed page in without
                    // copying it or forcing Metal to keep it resident.
                    checksum ^= contents.load(fromByteOffset: offset, as: UInt8.self)
                }
                if warmup.isCancelled { break }
            }
            warmup.lock.lock()
            warmup.completedChecksum = checksum
            warmup.lock.unlock()
        }
        return warmup
    }

    private final class RetainedBuffers: @unchecked Sendable {
        let buffers: [MTLBuffer]
        init(_ buffers: [MTLBuffer]) { self.buffers = buffers }
    }
}
