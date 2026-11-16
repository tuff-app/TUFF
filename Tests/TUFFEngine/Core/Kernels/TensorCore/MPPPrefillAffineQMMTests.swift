import Foundation
import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

@Suite struct MPPPrefillAffineQMMTests {
    /// Every layout TUFF ships, with shapes that leave partial M and N tiles
    /// and K spanning several 64-wide tiles.
    @Test(arguments: [
        (4, 32, 70, 45, 192),
        (4, 64, 70, 45, 192),
        (8, 64, 70, 45, 192),
        (8, 32, 1, 33, 128),
        (4, 32, 129, 96, 640),
        // Rows that select the 16- and 32-row tiles.
        (4, 64, 16, 45, 192),
        (4, 32, 9, 70, 256),
        (8, 64, 30, 33, 128),
        (4, 64, 33, 64, 192),
    ])
    func matchesTheCPUReference(_ c: (Int, Int, Int, Int, Int)) throws {
        let (bits, group, m, n, k) = c
        let context = try MetalContext()
        guard let qmm = MPPPrefillAffineQMM(context: context, bits: bits, groupSize: group)
        else {
            // Absent only where MSL 4 tensors are unavailable; the callers
            // keep their per-token path there.
            #expect(!context.device.supportsFamily(.apple8)
                    || ProcessInfo.processInfo.operatingSystemVersion.majorVersion < 26)
            return
        }

        let groups = k / group
        let rowBytes = k * bits / 8
        var packed = [UInt8](repeating: 0, count: n * rowBytes)
        for i in packed.indices { packed[i] = UInt8(truncatingIfNeeded: i &* 37 &+ 0x29) }
        var scales = [UInt16](repeating: 0, count: n * groups)
        var biases = [UInt16](repeating: 0, count: n * groups)
        let step: Float = bits == 4 ? 0.00025 : 0.00002
        for row in 0..<n {
            for g in 0..<groups {
                scales[row * groups + g] = Quantization.bf16Bits(
                    (bits == 4 ? 0.001 : 0.0001) + Float((row + g) % 5) * step)
                biases[row * groups + g] = Quantization.bf16Bits(
                    -0.01 + Float((row * 3 + g) % 7) * 0.002)
            }
        }
        var x = [Float16](repeating: 0, count: m * k)
        for i in x.indices { x[i] = Float16(Float((i * 11) % 29 - 14) / 64.0) }

        var reference = [Float](repeating: 0, count: m * n)
        for t in 0..<m {
            for row in 0..<n {
                var acc: Float = 0
                for kk in 0..<k {
                    let q: Float
                    if bits == 4 {
                        let byte = packed[row * rowBytes + kk / 2]
                        q = Float(kk % 2 == 0 ? byte & 0x0F : byte >> 4)
                    } else {
                        q = Float(packed[row * rowBytes + kk])
                    }
                    let g = row * groups + kk / group
                    // The kernel stages each dequantized weight as FP16.
                    let w = Float(Float16(q * Quantization.bf16ToFloat(scales[g])
                                          + Quantization.bf16ToFloat(biases[g])))
                    acc += w * Float(x[t * k + kk])
                }
                reference[t * n + row] = acc
            }
        }

        let device = context.device
        // .gturbo tensors are not 4-byte aligned; start some cases mid-word.
        let weightsOffset = m % 3
        let wBuf = try #require(device.makeBuffer(
            bytes: [UInt8](repeating: 0xA5, count: weightsOffset) + packed,
            length: weightsOffset + packed.count))
        let sBuf = try #require(device.makeBuffer(bytes: scales, length: scales.count * 2))
        let bBuf = try #require(device.makeBuffer(bytes: biases, length: biases.count * 2))
        let xBuf = try #require(device.makeBuffer(bytes: x, length: x.count * 2))
        let yBuf = try #require(device.makeBuffer(length: m * n * 2, options: .storageModeShared))
        let cb = try #require(context.queue.makeCommandBuffer())
        #expect(qmm.encode(commandBuffer: cb,
                           weights: wBuf, weightsOffset: weightsOffset,
                           scales: sBuf, scalesOffset: 0,
                           biases: bBuf, biasesOffset: 0,
                           x: xBuf, y: yBuf, m: m, n: n, k: k))
        cb.commit()
        cb.waitUntilCompleted()
        try checkCommandBufferError(cb)

        let pointer = yBuf.contents().bindMemory(to: Float16.self, capacity: m * n)
        let actual = (0..<(m * n)).map { Float(pointer[$0]) }
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < 5e-3, "w\(bits)g\(group) m=\(m) n=\(n) k=\(k) rel=\(rel)")
    }

    @Test func refusesAShapeItCannotTile() throws {
        let context = try MetalContext()
        guard let qmm = MPPPrefillAffineQMM(context: context, bits: 4, groupSize: 32) else { return }
        let buffer = try #require(context.device.makeBuffer(length: 4_096))
        let cb = try #require(context.queue.makeCommandBuffer())
        #expect(!qmm.encode(commandBuffer: cb,
                            weights: buffer, weightsOffset: 0,
                            scales: buffer, scalesOffset: 0,
                            biases: buffer, biasesOffset: 0,
                            x: buffer, y: buffer, m: 4, n: 4, k: 96))
        #expect(MPPPrefillAffineQMM(context: context, bits: 3, groupSize: 64) == nil)
    }

    /// Row counts around the 128-row blocks the wide tile serves, including
    /// the short and ragged blocks prefill and routed experts produce.
    private static let parityRows = [1, 2, 31, 32, 127, 128, 130, 255, 256, 383]

    /// The wide tile must reproduce the 64 x 32 tile bit for bit at the
    /// shapes the models run: Ornith 9B's dense FFN, GDN and attention
    /// projections, Gemma 4 12B's INT8 FFN and INT4 attention, and group-32
    /// layouts at Qwen 3.8 Flash Next's width.
    @Test(arguments: [
        (4, 64, 12288, 4096), (4, 64, 4096, 12288), (4, 64, 8192, 4096),
        (4, 64, 4096, 4096), (4, 64, 1024, 4096),
        (8, 64, 15360, 3840), (8, 64, 3840, 15360), (4, 64, 4096, 3840),
        (4, 32, 4096, 2048), (8, 32, 2048, 512),
        // N that does not fill the last 16-column tile.
        (4, 64, 1034, 1024), (8, 64, 72, 512),
    ])
    func wideTileMatchesTheReferenceTilesBitForBit(_ c: (Int, Int, Int, Int)) throws {
        let (bits, group, n, k) = c
        let context = try MetalContext()
        guard let qmm = MPPPrefillAffineQMM(context: context, bits: bits, groupSize: group)
        else { return }
        #expect(qmm.wideTileRows(m: 256, k: k) == 256,
                "wide tile unavailable for w\(bits)g\(group)")
        let device = context.device
        var rng = SplitMix64(seed: UInt64(bits * 1_000_003 + group * 7_919 + n + k))
        let maxRows = Self.parityRows.max()!
        let groups = k / group
        let rowBytes = k * bits / 8
        // Odd offsets, as .gturbo tensors and chunk scratch slices sit.
        let weightsOffset = 3
        let scalesOffset = 2
        let xOffset = 6 * k
        let yOffset = 10 * n
        var packed = [UInt8](repeating: 0, count: weightsOffset + n * rowBytes)
        packed.withUnsafeMutableBytes { raw in
            for i in stride(from: weightsOffset, through: raw.count - 8, by: 8) {
                raw.storeBytes(of: rng.next(), toByteOffset: i, as: UInt64.self)
            }
        }
        let weights = try #require(device.makeBuffer(bytes: packed, length: packed.count,
                                                     options: .storageModeShared))
        let scaleBase: Float = bits == 4 ? 0.004 : 0.0003
        let scales = (0..<(n * groups)).map { _ in
            Quantization.bf16Bits(scaleBase * rng.uniform(0.5, 1.5))
        }
        let biases = (0..<(n * groups)).map { _ in Quantization.bf16Bits(rng.uniform(-0.04, -0.02)) }
        let sBuf = try #require(device.makeBuffer(bytes: [UInt16](repeating: 0, count: 1) + scales,
                                                  length: scalesOffset + scales.count * 2,
                                                  options: .storageModeShared))
        let bBuf = try #require(device.makeBuffer(bytes: biases, length: biases.count * 2,
                                                  options: .storageModeShared))
        let x = (0..<(maxRows * k + xOffset / 2)).map { _ in Float16(rng.uniform(-2, 2)) }
        let xBuf = try #require(device.makeBuffer(bytes: x, length: x.count * 2,
                                                  options: .storageModeShared))
        let yBytes = yOffset + maxRows * n * 2

        for m in Self.parityRows {
            func run(referenceTiles: Bool) throws -> MTLBuffer {
                let y = try #require(device.makeBuffer(length: yBytes, options: .storageModeShared))
                // NaN, so a row the kernel never writes fails the check below.
                memset(y.contents(), 0xFF, yBytes)
                let cb = try #require(context.queue.makeCommandBuffer())
                #expect(qmm.encode(commandBuffer: cb,
                                   weights: weights, weightsOffset: weightsOffset,
                                   scales: sBuf, scalesOffset: scalesOffset,
                                   biases: bBuf, biasesOffset: 0,
                                   x: xBuf, xOffset: xOffset, y: y, yOffset: yOffset,
                                   m: m, n: n, k: k, referenceTiles: referenceTiles))
                cb.commit()
                cb.waitUntilCompleted()
                try checkCommandBufferError(cb)
                return y
            }
            let reference = try run(referenceTiles: true)
            let candidate = try run(referenceTiles: false)
            #expect(memcmp(reference.contents(), candidate.contents(), yBytes) == 0,
                    "w\(bits)g\(group) m=\(m) n=\(n) k=\(k) differs")
            // Guard against a vacuous pass: every output was written.
            let out = reference.contents().advanced(by: yOffset)
                .bindMemory(to: Float16.self, capacity: m * n)
            #expect((0..<(m * n)).allSatisfy { out[$0].isFinite },
                    "w\(bits)g\(group) m=\(m) has unwritten or non-finite outputs")
            #expect((0..<n).contains { out[(m - 1) * n + $0] != 0 })
        }
    }

    @Test func wideTileServesOnlyWhole128RowBlocks() throws {
        let context = try MetalContext()
        guard let qmm = MPPPrefillAffineQMM(context: context, bits: 4, groupSize: 64) else { return }
        #expect(qmm.wideTileRows(m: 127, k: 4096) == 0)
        #expect(qmm.wideTileRows(m: 255, k: 4096) == 128)
        #expect(qmm.wideTileRows(m: 1024, k: 4096) == 1024)
        // K must fill whole 256-deep steps: Gemma 4 26B's 2112-wide down
        // projection keeps the 64 x 32 tile.
        #expect(qmm.wideTileRows(m: 256, k: 2112) == 0)
    }
}
