import Foundation
import Darwin
import Metal

/// Best-effort warm-up of the resident weights after a model reports ready.
///
/// The first prompt after a load used to pay for the GPU mapping every
/// resident page: 2.5 s of a 26-token prompt on Ornith 1.5 9B, with the file
/// already in the page cache, so reading the pages from the CPU (what 10.0
/// did) left most of it. Touching one byte per page from the GPU, on a queue
/// of its own, maps them while the person is still typing. It never wires
/// memory or requests Metal residency. A request cancels it between chunks;
/// one that arrives straight after the load measured no slower.
public final class ResidentWarmup: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var finished = false

    /// Bytes touched per command buffer, so a cancel takes effect quickly.
    static let chunkBytes = 256 << 20

    /// Resident weights at most half of memory, all of them mapped up front.
    /// MoE models qualify too: only their resident part is touched.
    static func eligible(payloadSize: UInt64, physicalMemory: UInt64,
                         stagesDenseLayers: Bool) -> Bool {
        physicalMemory > 0 && payloadSize > 0 && payloadSize <= physicalMemory / 2
            && !stagesDenseLayers
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

    /// True once every chunk ran or the pass stopped.
    public var isFinished: Bool {
        lock.lock()
        defer { lock.unlock() }
        return finished
    }

    private func markFinished() {
        lock.lock()
        finished = true
        lock.unlock()
    }

    /// Walks each buffer in chunks, checking for a cancel before each one.
    /// The callback seam lets tests stop a pass part way.
    func run(lengths: [Int], chunkBytes: Int, touch: (_ buffer: Int, _ start: Int, _ length: Int) -> Bool) {
        defer { markFinished() }
        guard chunkBytes > 0 else { return }
        for (index, total) in lengths.enumerated() {
            var start = 0
            while start < total {
                if isCancelled { return }
                let length = min(chunkBytes, total - start)
                guard touch(index, start, length) else { return }
                start += length
            }
        }
    }

    static func start(buffers: [MTLBuffer], device: MTLDevice) -> ResidentWarmup? {
        guard let toucher = PageToucher(device: device) else { return nil }
        let warmup = ResidentWarmup()
        let retained = RetainedBuffers(buffers)
        DispatchQueue.global(qos: .utility).async {
            warmup.run(lengths: retained.buffers.map(\.length), chunkBytes: chunkBytes) { index, start, length in
                toucher.touch(retained.buffers[index], start: start, length: length)
            }
        }
        return warmup
    }

    /// One compute pass that reads a byte from every page of a buffer range.
    final class PageToucher: @unchecked Sendable {
        let queue: MTLCommandQueue
        let pipeline: MTLComputePipelineState
        /// Written only if the first page's byte is 0xA5, which keeps the
        /// reads from being optimised away and lets a test see one happened.
        let sink: MTLBuffer
        let pageSize = Int(getpagesize())

        init?(device: MTLDevice) {
            guard let queue = device.makeCommandQueue(),
                  let library = try? device.makeLibrary(source: Self.source, options: nil),
                  let function = library.makeFunction(name: "tuff_touch_pages"),
                  let pipeline = try? device.makeComputePipelineState(function: function),
                  let sink = device.makeBuffer(length: 4, options: .storageModeShared) else { return nil }
            queue.label = "TUFF resident warm-up"
            self.queue = queue
            self.pipeline = pipeline
            self.sink = sink
            sink.contents().storeBytes(of: 0, as: UInt32.self)
        }

        var sinkValue: UInt32 { sink.contents().load(as: UInt32.self) }

        func touch(_ buffer: MTLBuffer, start: Int, length: Int) -> Bool {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else { return false }
            var stride = UInt32(pageSize), count = UInt32(length)
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: start, index: 0)
            encoder.setBuffer(sink, offset: 0, index: 1)
            encoder.setBytes(&stride, length: 4, index: 2)
            encoder.setBytes(&count, length: 4, index: 3)
            let pages = (length + pageSize - 1) / pageSize
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(MTLSize(width: pages, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return commandBuffer.status == .completed
        }

        static let source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void tuff_touch_pages(device const uchar *base [[buffer(0)]],
                                     device atomic_uint *sink [[buffer(1)]],
                                     constant uint &stride [[buffer(2)]],
                                     constant uint &length [[buffer(3)]],
                                     uint page [[thread_position_in_grid]]) {
            ulong offset = ulong(page) * ulong(stride);
            if (offset >= length) return;
            if (base[offset] == 0xA5 && page == 0) {
                atomic_fetch_add_explicit(sink, 1, memory_order_relaxed);
            }
        }
        """
    }

    private final class RetainedBuffers: @unchecked Sendable {
        let buffers: [MTLBuffer]
        init(_ buffers: [MTLBuffer]) { self.buffers = buffers }
    }
}
