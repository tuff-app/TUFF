import Foundation
import Metal
import ObjectiveC

/// Diagnostic per-dispatch GPU timing, enabled only by `TUFF_GPU_OP_PROFILE=1`.
///
/// When enabled, compute-encoder creation is routed through
/// `MTLComputePassDescriptor` sample-buffer attachments so every encoder gets a
/// start/end GPU timestamp, and each encoder is labelled by its first compute
/// pipeline and dispatch grid. When disabled nothing is installed: no swizzle,
/// no sample buffer, and the only residual cost is one lazy static check when a
/// pipeline is created.
public final class GPUOpProfiler: @unchecked Sendable {
    public static let shared: GPUOpProfiler? = {
        ProcessInfo.processInfo.environment["TUFF_GPU_OP_PROFILE"] == "1" ? GPUOpProfiler() : nil
    }()

    public static func registerPipeline(_ pipeline: MTLComputePipelineState, name: String) {
        shared?.register(pipeline, name: name)
    }

    private struct Entry {
        var kernel = "?"
        var grid = ""
        var start: Int
    }

    private struct Aggregate {
        var nanos = 0.0
        var count = 0
    }

    private let lock = NSLock()
    private var device: MTLDevice?
    private var sampleBuffer: MTLCounterSampleBuffer?
    private var capacity = 0
    private var used = 0
    private var entries: [Entry] = []
    private var current = -1
    private var building = false
    private var pipelineNames: [ObjectIdentifier: String] = [:]
    private var swizzledEncoderClasses = Set<String>()
    private var installed = false
    private var cpu0: UInt64 = 0
    private var gpu0: UInt64 = 0
    private var totals: [String: [String: Aggregate]] = [:]
    private var steps: [String: Int] = [:]
    private var spanNanos: [String: Double] = [:]

    private func register(_ pipeline: MTLComputePipelineState, name: String) {
        lock.lock()
        pipelineNames[ObjectIdentifier(pipeline)] = name
        lock.unlock()
    }

    /// Install once per process. `queue` is only used to find the concrete
    /// command-buffer class.
    public func install(device: MTLDevice, queue: MTLCommandQueue) {
        lock.lock()
        defer { lock.unlock() }
        guard !installed else { return }
        guard device.supportsCounterSampling(.atStageBoundary),
              let set = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else {
            FileHandle.standardError.write(Data("[gpu-op-profile] stage-boundary timestamps unsupported\n".utf8))
            return
        }
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = set
        descriptor.storageMode = .shared
        descriptor.label = "TUFF op profile"
        var made: MTLCounterSampleBuffer?
        var failure: Error?
        for count in [65536, 16384, 8192, 4096, 2048] {
            descriptor.sampleCount = count
            do { made = try device.makeCounterSampleBuffer(descriptor: descriptor); break } catch { failure = error }
        }
        guard let buffer = made, let probe = queue.makeCommandBuffer() else {
            FileHandle.standardError.write(Data("[gpu-op-profile] sample buffer unavailable: \(String(describing: failure))\n".utf8))
            return
        }
        self.device = device
        sampleBuffer = buffer
        capacity = descriptor.sampleCount
        let (cpu, gpu) = device.sampleTimestamps()
        cpu0 = cpu
        gpu0 = gpu

        let selector = NSSelectorFromString("computeCommandEncoder")
        guard let method = class_getInstanceMethod(object_getClass(probe), selector) else { return }
        typealias Make = @convention(c) (AnyObject, Selector) -> AnyObject?
        let original = unsafeBitCast(method_getImplementation(method), to: Make.self)
        let block: @convention(block) (AnyObject) -> AnyObject? = { [unowned self] commandBuffer in
            self.makeEncoder(commandBuffer) { original(commandBuffer, selector) }
        }
        method_setImplementation(method, imp_implementationWithBlock(block))
        installed = true
    }

    private func makeEncoder(_ commandBuffer: AnyObject, fallback: () -> AnyObject?) -> AnyObject? {
        lock.lock()
        guard let sampleBuffer, !building, used + 2 <= capacity,
              let typed = commandBuffer as? MTLCommandBuffer else {
            current = -1
            lock.unlock()
            return fallback()
        }
        let index = used
        used += 2
        entries.append(Entry(start: index))
        current = entries.count - 1
        building = true
        lock.unlock()

        let descriptor = MTLComputePassDescriptor()
        let attachment = descriptor.sampleBufferAttachments[0]!
        attachment.sampleBuffer = sampleBuffer
        attachment.startOfEncoderSampleIndex = index
        attachment.endOfEncoderSampleIndex = index + 1
        let encoder = typed.makeComputeCommandEncoder(descriptor: descriptor)
        lock.lock()
        building = false
        lock.unlock()
        if let encoder { swizzleEncoderClass(of: encoder) }
        return encoder
    }

    private func swizzleEncoderClass(of encoder: AnyObject) {
        guard let cls = object_getClass(encoder) else { return }
        let className = String(cString: class_getName(cls))
        lock.lock()
        let first = swizzledEncoderClasses.insert(className).inserted
        lock.unlock()
        guard first else { return }

        let pipelineSelector = NSSelectorFromString("setComputePipelineState:")
        if let method = class_getInstanceMethod(cls, pipelineSelector) {
            typealias Fn = @convention(c) (AnyObject, Selector, AnyObject) -> Void
            let original = unsafeBitCast(method_getImplementation(method), to: Fn.self)
            let block: @convention(block) (AnyObject, AnyObject) -> Void = { [unowned self] encoder, state in
                original(encoder, pipelineSelector, state)
                self.noteKernel(state)
            }
            method_setImplementation(method, imp_implementationWithBlock(block))
        }
        for name in ["dispatchThreadgroups:threadsPerThreadgroup:", "dispatchThreads:threadsPerThreadgroup:"] {
            let selector = NSSelectorFromString(name)
            guard let method = class_getInstanceMethod(cls, selector) else { continue }
            typealias Fn = @convention(c) (AnyObject, Selector, MTLSize, MTLSize) -> Void
            let original = unsafeBitCast(method_getImplementation(method), to: Fn.self)
            let block: @convention(block) (AnyObject, MTLSize, MTLSize) -> Void = { [unowned self] encoder, grid, threads in
                original(encoder, selector, grid, threads)
                self.noteDispatch(grid)
            }
            method_setImplementation(method, imp_implementationWithBlock(block))
        }
    }

    private func noteKernel(_ state: AnyObject) {
        lock.lock()
        if current >= 0, entries[current].kernel == "?" {
            entries[current].kernel = pipelineNames[ObjectIdentifier(state)] ?? "unknown"
        }
        lock.unlock()
    }

    private func noteDispatch(_ grid: MTLSize) {
        lock.lock()
        if current >= 0, entries[current].grid.isEmpty {
            entries[current].grid = "\(grid.width)x\(grid.height)x\(grid.depth)"
        }
        lock.unlock()
    }

    /// Discard samples from encoders already completed; call with no command
    /// buffer in flight.
    public func beginStep() {
        lock.lock()
        used = 0
        entries.removeAll(keepingCapacity: true)
        current = -1
        lock.unlock()
    }

    /// Resolve the timestamps of the step's completed command buffers and
    /// add them to `kind`'s table.
    public func endStep(kind: String) {
        lock.lock()
        defer { lock.unlock() }
        guard let sampleBuffer, let device, used > 0, !entries.isEmpty,
              let data = try? sampleBuffer.resolveCounterRange(0..<used) else { return }
        let (cpu, gpu) = device.sampleTimestamps()
        guard gpu > gpu0 else { return }
        // sampleTimestamps reports CPU time in nanoseconds on Apple silicon.
        let nanosPerTick = Double(cpu - cpu0) / Double(gpu - gpu0)
        let stamps = data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
        var first = UInt64.max, last: UInt64 = 0
        var table = totals[kind] ?? [:]
        for entry in entries where entry.start + 1 < stamps.count {
            let start = stamps[entry.start], end = stamps[entry.start + 1]
            guard start != UInt64.max, end != UInt64.max, end >= start else { continue }
            first = min(first, start)
            last = max(last, end)
            let key = entry.grid.isEmpty ? entry.kernel : "\(entry.kernel) [\(entry.grid)]"
            var aggregate = table[key] ?? Aggregate()
            aggregate.nanos += Double(end - start) * nanosPerTick
            aggregate.count += 1
            table[key] = aggregate
        }
        totals[kind] = table
        steps[kind, default: 0] += 1
        if last > first { spanNanos[kind, default: 0] += Double(last - first) * nanosPerTick }
    }

    /// Per-step averages, longest first.
    public func report() -> String {
        lock.lock()
        defer { lock.unlock() }
        var text = ""
        for kind in totals.keys.sorted() {
            guard let count = steps[kind], count > 0, let table = totals[kind] else { continue }
            let sum = table.values.reduce(0) { $0 + $1.nanos }
            text += String(format: "\n[gpu op profile: %@, %d steps; per-step encoder time %.3f ms, first-to-last span %.3f ms]\n",
                           kind, count, sum / Double(count) / 1e6, (spanNanos[kind] ?? 0) / Double(count) / 1e6)
            for (key, aggregate) in table.sorted(by: { $0.value.nanos > $1.value.nanos }) {
                text += String(format: "  %8.3f ms %5.1f%% %6.1f disp  %@\n",
                               aggregate.nanos / Double(count) / 1e6, aggregate.nanos / sum * 100,
                               Double(aggregate.count) / Double(count), key)
            }
        }
        return text
    }
}
