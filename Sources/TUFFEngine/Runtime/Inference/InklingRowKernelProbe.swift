import Foundation
import Metal

extension InklingForwardRunner {
    /// Whether the several-rows expert kernels give the one-row kernels'
    /// bytes on this GPU: both phases of the fused experts in each expert
    /// format, on random weights with shapes that leave lanes uneven. Runs on its own
    /// queue, once per runner. Any difference, failure or non-finite or
    /// all-zero output keeps the one-row kernels.
    static func rowKernelsMatch(_ kernels: InklingKernels, device: MTLDevice) -> Bool {
        autoreleasepool {
            guard let queue = device.makeCommandQueue() else { return false }
            var random = SystemRandomNumberGenerator()
            let small = RowKernelProbe(kernels: kernels, device: device, queue: queue)
            // Wide enough that lanes take two or three MXFP8 blocks of a row
            // with uneven tails, as Inkling 2 Mini's rows of 2304 and 1152
            // do; two blobs shared by the eight experts keep it quick.
            let wide = RowKernelProbe(kernels: kernels, device: device, queue: queue,
                                      columns: 2112, intermediate: 1056, distinctBlobs: 2)
            return ((try? small.run(using: &random)) ?? false)
                && ((try? wide.run(using: &random)) ?? false)
        }
    }
}

struct RowKernelProbe {
    let kernels: InklingKernels
    let device: MTLDevice
    let queue: MTLCommandQueue
    // The defaults: 72 words or 18 MXFP8 blocks per row of 576, so lanes do
    // uneven work; 20 words or 5 blocks per row of 160, fewer than the 32
    // lanes. Both multiples of 32 and of `InklingKernels.rowsPerGroup`.
    var columns = 576, intermediate = 160
    /// Distinct expert blobs, bound round-robin to the eight experts.
    var distinctBlobs = 8

    struct ProbeError: Error {}

    func buffer(_ bytes: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: max(16, bytes), options: .storageModeShared) else {
            throw ProbeError()
        }
        memset(buffer.contents(), 0, buffer.length)
        return buffer
    }

    func floats<G: RandomNumberGenerator>(_ count: Int, _ range: ClosedRange<Float>,
                                         using random: inout G) throws -> MTLBuffer {
        let result = try buffer(count * 4)
        let values = result.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<count { values[index] = Float.random(in: range, using: &random) }
        return result
    }

    /// Random bytes; with `fp8`, never an E4M3 NaN (0x7f, 0xff).
    func fill(_ target: UnsafeMutableRawPointer, _ count: Int, fp8: Bool) {
        arc4random_buf(target, count)
        guard fp8 else { return }
        let bytes = target.assumingMemoryBound(to: UInt8.self)
        for index in 0..<count where bytes[index] & 0x7f == 0x7f { bytes[index] ^= 1 }
    }

    func execute(_ encode: (InklingPass) -> Void) throws {
        let pass = try InklingPass(queue)
        encode(pass)
        pass.endEncoding()
        pass.commandBuffer.commit()
        pass.commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(pass.commandBuffer)
    }

    func same(_ a: MTLBuffer, _ b: MTLBuffer, count: Int, half: Bool = false) -> Bool {
        let bytes = count * (half ? 2 : 4)
        guard memcmp(a.contents(), b.contents(), bytes) == 0 else { return false }
        var nonZero = false
        if half {
            let values = a.contents().assumingMemoryBound(to: Float16.self)
            for index in 0..<count {
                guard values[index].isFinite else { return false }
                nonZero = nonZero || values[index] != 0
            }
        } else {
            let values = a.contents().assumingMemoryBound(to: Float.self)
            for index in 0..<count {
                guard values[index].isFinite else { return false }
                nonZero = nonZero || values[index] != 0
            }
        }
        return nonZero
    }

    func run<G: RandomNumberGenerator>(using random: inout G) throws -> Bool {
        try expertsMatch(nvfp4: true, using: &random)
            && expertsMatch(nvfp4: false, using: &random)
    }

    func expertsMatch<G: RandomNumberGenerator>(nvfp4: Bool, using random: inout G) throws -> Bool {
        let width = columns
        // codes, then scales, for gate/up (2I x W) and down (W x I).
        let gateUpCodes = 2 * intermediate * width / (nvfp4 ? 2 : 1)
        let gateUpScales = 2 * intermediate * width / (nvfp4 ? 16 : 32)
        let downCodes = width * intermediate / (nvfp4 ? 2 : 1)
        let downScales = width * intermediate / (nvfp4 ? 16 : 32)
        let offsets = [0, gateUpCodes, gateUpCodes + gateUpScales, gateUpCodes + gateUpScales + downCodes]
        let blobBytes = offsets[3] + downScales
        var blobs: [MTLBuffer] = []
        for _ in 0..<max(1, min(8, distinctBlobs)) {
            let blob = try buffer(blobBytes)
            let base = blob.contents()
            fill(base, gateUpCodes, fp8: !nvfp4)
            fill(base + offsets[2], downCodes, fp8: !nvfp4)
            for (offset, count) in [(offsets[1], gateUpScales), (offsets[3], downScales)] {
                let bytes = (base + offset).assumingMemoryBound(to: UInt8.self)
                for index in 0..<count {
                    // E4M3 scales below 1 for NVFP4; powers of two near 1 for MXFP8.
                    bytes[index] = nvfp4 ? UInt8.random(in: 0x08...0x37, using: &random)
                        : UInt8.random(in: 120...131, using: &random)
                }
            }
            blobs.append(blob)
        }
        let bound = (0..<8).map { InklingArgument.buffer(blobs[$0 % blobs.count]) }
        let format = nvfp4 ? "nvfp4" : "mxfp8"
        let factors = (0..<8).map { _ in Float.random(in: 0.05...0.5, using: &random) }
        let x = try floats(width, -2...2, using: &random)
        let hiddenOne = try buffer(8 * intermediate * 4), hiddenMany = try buffer(8 * intermediate * 4)
        let outOne = try buffer(width * 4), outMany = try buffer(width * 4)
        let span = InklingKernels.rowsPerGroup
        let threads = InklingKernels.rowGroupThreads
        func phase1(_ hidden: MTLBuffer) -> [InklingArgument] {
            bound + [.buffer(x), .buffer(hidden), .floats(factors), .u32(offsets[0]), .u32(offsets[1]),
                     .u32(width), .u32(intermediate), .f32(4), .f32(25)]
        }
        func phase2(_ hidden: MTLBuffer, _ out: MTLBuffer) -> [InklingArgument] {
            bound + [.buffer(hidden), .buffer(out), .floats(factors), .u32(offsets[2]), .u32(offsets[3]),
                     .u32(intermediate)]
        }
        try execute { pass in
            kernels.groups(pass, "ink_expert_phase1_" + format, phase1(hiddenOne),
                           grid: (intermediate, 8), threads: 32)
            kernels.groups(pass, "ink_expert_phase1_rows_" + format, phase1(hiddenMany),
                           grid: (intermediate / span, 8), threads: threads)
        }
        guard same(hiddenOne, hiddenMany, count: 8 * intermediate) else { return false }
        // Both phase 2 kernels read the same hidden rows.
        try execute { pass in
            kernels.groups(pass, "ink_expert_phase2_" + format, phase2(hiddenOne, outOne),
                           grid: (width, 1), threads: 32)
            kernels.groups(pass, "ink_expert_phase2_rows_" + format, phase2(hiddenOne, outMany),
                           grid: (width / span, 1), threads: threads)
        }
        return same(outOne, outMany, count: width)
    }
}

