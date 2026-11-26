import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// Two things a Qwen4-Exp forward pass depends on that nothing else in the
/// lineup exercises, both found by bisecting the real checkpoint against the
/// reference implementation:
///
/// 1. **The INT4 group size has to reach every pipeline that dequantizes.**
///    Metal function constants are per-pipeline, and `quant_group_size()`
///    falls back to 64 wherever the constant is not set. `GDN` and
///    `FusedQKVGEMV` both call `dequant_int4_gemv_simd_body`, and both used to
///    build their pipelines with no constants at all — so on a group-32
///    checkpoint they decoded plausible nonsense and nothing failed.
///
/// 2. **The gated-DeltaNet output gate is not always silu.** Qwen 3.6 leaves
///    `output_gate_type` unset and gates with the model's `hidden_act`;
///    Qwen4-Exp sets it to `sigmoid`. `silu(z)` is `z * sigmoid(z)`, so the
///    two differ by a factor of z — not a rescaling that later norms absorb.
@Suite struct GDNQuantGroupAndGateTests {

    /// One INT4 matrix laid out the way the repacker writes a sub-tensor:
    /// packed weights, then BF16 scales, then BF16 biases, in one buffer.
    private struct PackedProjection {
        let view: TensorView
        let rows: [[Float]]

        init(device: MTLDevice, rows rowCount: Int, n: Int, groupSize: Int,
             rng: inout SplitMix64) {
            let packedPerRow = n / 2
            let groups = n / groupSize
            var weights = [UInt8](repeating: 0, count: rowCount * packedPerRow)
            var scales = [UInt16](repeating: 0, count: rowCount * groups)
            var biases = [UInt16](repeating: 0, count: rowCount * groups)
            var dequantized: [[Float]] = []
            for row in 0..<rowCount {
                let values = (0..<n).map { _ in rng.uniform(-0.5, 0.5) }
                let q = Quantization.quantizeInt4Affine(values, groupSize: groupSize)
                for i in 0..<packedPerRow { weights[row * packedPerRow + i] = q.packed[i] }
                for i in 0..<groups {
                    scales[row * groups + i] = q.scales[i]
                    biases[row * groups + i] = q.biases[i]
                }
                dequantized.append(Quantization.dequantizeInt4Affine(
                    q, n: n, groupSize: groupSize))
            }
            let scaleOffset = weights.count
            let biasOffset = scaleOffset + scales.count * 2
            let total = biasOffset + biases.count * 2
            var bytes = [UInt8](repeating: 0, count: total)
            bytes.replaceSubrange(0..<weights.count, with: weights)
            scales.withUnsafeBufferPointer {
                bytes.replaceSubrange(scaleOffset..<(scaleOffset + UnsafeRawBufferPointer($0).count),
                                      with: UnsafeRawBufferPointer($0))
            }
            biases.withUnsafeBufferPointer {
                bytes.replaceSubrange(biasOffset..<(biasOffset + UnsafeRawBufferPointer($0).count),
                                      with: UnsafeRawBufferPointer($0))
            }
            let buffer = device.makeBuffer(bytes: bytes, length: total,
                                           options: .storageModeShared)!
            self.rows = dequantized
            self.view = TensorView(buffer: buffer,
                                   offset: 0, length: UInt64(weights.count),
                                   scaleOffset: UInt64(scaleOffset),
                                   scaleLength: UInt64(scales.count * 2),
                                   biasOffset: UInt64(biasOffset),
                                   biasLength: UInt64(biases.count * 2),
                                   shape: (UInt32(rowCount), UInt32(n), 1, 1),
                                   dtype: 0)
        }

        func product(with x: [Float]) -> [Float] {
            rows.map { row in
                var acc: Float = 0
                for i in 0..<x.count { acc += row[i] * x[i] }
                return acc
            }
        }
    }

    /// Qwen3.8 Flash Next's own linear-attention geometry.
    private static let qwen4Exp = LinearAttentionConfig(
        numKHeads: 16, numVHeads: 48, keyHeadDim: 128, valueHeadDim: 128,
        convKernelSize: 4, outputGate: .sigmoid)

    // MARK: - The group size has to reach the GDN input projection

    private static func runInProj(groupSize: Int,
                                  hiddenSize: Int = 2_560,
                                  seed: UInt64) throws
        -> (actual: [Float], reference: [Float])? {
        var rng = SplitMix64(seed: seed)
        let ctx = try MetalContext()
        let cfg = qwen4Exp
        let gdn = try GDN(context: ctx, config: cfg,
                          specializedHiddenSize: hiddenSize, groupSize: groupSize)
        #expect(gdn.groupSize == groupSize)
        let qkv = PackedProjection(device: ctx.device, rows: cfg.qkvDim,
                                   n: hiddenSize, groupSize: groupSize, rng: &rng)
        let z = PackedProjection(device: ctx.device, rows: cfg.valueDim,
                                 n: hiddenSize, groupSize: groupSize, rng: &rng)
        let a = PackedProjection(device: ctx.device, rows: cfg.numVHeads,
                                 n: hiddenSize, groupSize: groupSize, rng: &rng)
        let b = PackedProjection(device: ctx.device, rows: cfg.numVHeads,
                                 n: hiddenSize, groupSize: groupSize, rng: &rng)
        let xHalves = (0..<hiddenSize).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xHalves),
              let qkvOut = Fp16Buffer.make(ctx.device, count: cfg.qkvDim),
              let zOut = Fp16Buffer.make(ctx.device, count: cfg.valueDim),
              let aOut = Fp16Buffer.make(ctx.device, count: cfg.numVHeads),
              let bOut = Fp16Buffer.make(ctx.device, count: cfg.numVHeads) else {
            Issue.record("alloc failed"); return nil
        }
        let cb = ctx.queue.makeCommandBuffer()!
        try gdn.encodeInputProjections(commandBuffer: cb, x: xBuf,
                                   qkv: qkv.view, qkvOut: qkvOut,
                                   z: z.view, zOut: zOut,
                                   a: a.view, aOut: aOut,
                                   b: b.view, bOut: bOut,
                                   hiddenSize: hiddenSize)
        cb.commit(); cb.waitUntilCompleted()

        let x = xHalves.map { Float($0) }
        let actual = Fp16Buffer.read(qkvOut, count: cfg.qkvDim)
            + Fp16Buffer.read(zOut, count: cfg.valueDim)
            + Fp16Buffer.read(aOut, count: cfg.numVHeads)
            + Fp16Buffer.read(bOut, count: cfg.numVHeads)
        let reference = qkv.product(with: x) + z.product(with: x)
            + a.product(with: x) + b.product(with: x)
        return (actual, reference)
    }

    @Test func gdnInputProjectionDecodesGroupsOf32() throws {
        guard let r = try Self.runInProj(groupSize: 32, seed: 0x6764_6E33_3200)
        else { return }
        let relErr = RelError.compute(actual: r.actual, reference: r.reference)
        #expect(relErr < Tolerance.fp16Reduction, "relErr=\(relErr)")
    }

    /// The path every shipping model takes.
    @Test func gdnInputProjectionStillDecodesGroupsOf64() throws {
        guard let r = try Self.runInProj(groupSize: 64, seed: 0x6764_6E36_3400)
        else { return }
        let relErr = RelError.compute(actual: r.actual, reference: r.reference)
        #expect(relErr < Tolerance.fp16Reduction, "relErr=\(relErr)")
    }

    /// Without this, both tests above would pass on a GDN that ignored the
    /// constant and always decoded at 64 — which is exactly what it did.
    @Test func gdnInputProjectionAtTheWrongGroupSizeIsWrong() throws {
        var rng = SplitMix64(seed: 0x6764_6E4E_4547)
        let ctx = try MetalContext()
        let cfg = Self.qwen4Exp
        let hiddenSize = 2_560
        let projection = PackedProjection(device: ctx.device, rows: cfg.qkvDim,
                                          n: hiddenSize, groupSize: 32, rng: &rng)
        let xHalves = (0..<hiddenSize).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let x = xHalves.map { Float($0) }
        let reference = projection.product(with: x)

        var results: [Int: [Float]] = [:]
        for groupSize in [32, 64] {
            let gdn = try GDN(context: ctx, config: cfg, groupSize: groupSize)
            let z = PackedProjection(device: ctx.device, rows: cfg.valueDim,
                                     n: hiddenSize, groupSize: 32, rng: &rng)
            let a = PackedProjection(device: ctx.device, rows: cfg.numVHeads,
                                     n: hiddenSize, groupSize: 32, rng: &rng)
            guard let xBuf = Fp16Buffer.make(ctx.device, halves: xHalves),
                  let qkvOut = Fp16Buffer.make(ctx.device, count: cfg.qkvDim),
                  let zOut = Fp16Buffer.make(ctx.device, count: cfg.valueDim),
                  let aOut = Fp16Buffer.make(ctx.device, count: cfg.numVHeads),
                  let bOut = Fp16Buffer.make(ctx.device, count: cfg.numVHeads) else {
                Issue.record("alloc failed"); return
            }
            let cb = ctx.queue.makeCommandBuffer()!
            try gdn.encodeInputProjections(commandBuffer: cb, x: xBuf,
                                       qkv: projection.view, qkvOut: qkvOut,
                                       z: z.view, zOut: zOut,
                                       a: a.view, aOut: aOut,
                                       b: a.view, bOut: bOut,
                                       hiddenSize: hiddenSize)
            cb.commit(); cb.waitUntilCompleted()
            results[groupSize] = Fp16Buffer.read(qkvOut, count: cfg.qkvDim)
        }
        let right = RelError.compute(actual: results[32]!, reference: reference)
        let wrong = RelError.compute(actual: results[64]!, reference: reference)
        #expect(right < Tolerance.fp16Reduction, "g32 relErr=\(right)")
        #expect(wrong > 0.1, "g64 on group-32 data should not match: \(wrong)")
    }

    // MARK: - The group size has to reach the packed q/k/v GEMV

    @Test(arguments: [32, 64])
    func fusedQKVGEMVDecodesAtItsGroupSize(groupSize: Int) throws {
        var rng = SplitMix64(seed: UInt64(0x716B_7600) + UInt64(groupSize))
        let ctx = try MetalContext()
        let kernel = try FusedQKVGEMV(context: ctx, groupSize: groupSize)
        #expect(kernel.groupSize == groupSize)
        let n = 2_560, qRows = 512, kvRows = 128
        let q = PackedProjection(device: ctx.device, rows: qRows, n: n,
                                 groupSize: groupSize, rng: &rng)
        let k = PackedProjection(device: ctx.device, rows: kvRows, n: n,
                                 groupSize: groupSize, rng: &rng)
        let v = PackedProjection(device: ctx.device, rows: kvRows, n: n,
                                 groupSize: groupSize, rng: &rng)
        let xHalves = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xHalves),
              let qOut = Fp16Buffer.make(ctx.device, count: qRows),
              let kOut = Fp16Buffer.make(ctx.device, count: kvRows),
              let vOut = Fp16Buffer.make(ctx.device, count: kvRows) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb,
                      qWeights: q.view.buffer, qWeightsOffset: Int(q.view.offset),
                      qScales: q.view.buffer, qScalesOffset: Int(q.view.scaleOffset),
                      qBiases: q.view.buffer, qBiasesOffset: Int(q.view.biasOffset),
                      kWeights: k.view.buffer, kWeightsOffset: Int(k.view.offset),
                      kScales: k.view.buffer, kScalesOffset: Int(k.view.scaleOffset),
                      kBiases: k.view.buffer, kBiasesOffset: Int(k.view.biasOffset),
                      vWeights: v.view.buffer, vWeightsOffset: Int(v.view.offset),
                      vScales: v.view.buffer, vScalesOffset: Int(v.view.scaleOffset),
                      vBiases: v.view.buffer, vBiasesOffset: Int(v.view.biasOffset),
                      x: xBuf, qOut: qOut, kOut: kOut, vOut: vOut,
                      qRows: UInt32(qRows), kvRows: UInt32(kvRows), n: UInt32(n))
        cb.commit(); cb.waitUntilCompleted()

        let x = xHalves.map { Float($0) }
        let actual = Fp16Buffer.read(qOut, count: qRows)
            + Fp16Buffer.read(kOut, count: kvRows)
            + Fp16Buffer.read(vOut, count: kvRows)
        let reference = q.product(with: x) + k.product(with: x) + v.product(with: x)
        let relErr = RelError.compute(actual: actual, reference: reference)
        #expect(relErr < Tolerance.fp16Reduction, "g\(groupSize) relErr=\(relErr)")
    }

    // MARK: - The output gate

    private static func runGatedNorm(gate: LinearAttentionConfig.OutputGate,
                                     seed: UInt64) throws -> (got: [Float],
                                                              want: [Float])? {
        var rng = SplitMix64(seed: seed)
        let ctx = try MetalContext()
        let cfg = LinearAttentionConfig(
            numKHeads: 2, numVHeads: 4, keyHeadDim: 64, valueHeadDim: 128,
            convKernelSize: 4, outputGate: gate)
        let gdn = try GDN(context: ctx, config: cfg)
        let count = cfg.valueDim
        let yValues = (0..<count).map { _ in Float(rng.uniform(-2.0, 2.0)) }
        let zValues = (0..<count).map { _ in Float(rng.uniform(-3.0, 3.0)) }
        let wValues = (0..<cfg.valueHeadDim).map { _ in Float(rng.uniform(0.5, 1.5)) }
        guard let yBuf = Fp16Buffer.make(ctx.device, halves: yValues.map { Float16($0) }),
              let zBuf = Fp16Buffer.make(ctx.device, halves: zValues.map { Float16($0) }),
              let outBuf = Fp16Buffer.make(ctx.device, count: count) else {
            Issue.record("alloc failed"); return nil
        }
        let bits = wValues.map { Quantization.bf16Bits($0) }
        let wBuf = ctx.device.makeBuffer(length: bits.count * 2,
                                         options: .storageModeShared)!
        bits.withUnsafeBytes {
            wBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        let cb = ctx.queue.makeCommandBuffer()!
        gdn.encodeGatedNorm(commandBuffer: cb, y: yBuf, z: zBuf,
                            weight: wBuf, weightOffset: 0, out: outBuf)
        cb.commit(); cb.waitUntilCompleted()

        // The kernel's arithmetic, on the same FP16 inputs it reads.
        let dv = cfg.valueHeadDim
        var want = [Float](repeating: 0, count: count)
        for head in 0..<cfg.numVHeads {
            let base = head * dv
            var sumsq: Float = 0
            for i in 0..<dv {
                let value = Float(Float16(yValues[base + i]))
                sumsq += value * value
            }
            let invRms = 1.0 / (sumsq / Float(dv) + 1e-6).squareRoot()
            for i in 0..<dv {
                let z = Float(Float16(zValues[base + i]))
                let activation = gate == .sigmoid
                    ? 1.0 / (1.0 + exp(-z))
                    : z / (1.0 + exp(-z))
                want[base + i] = Float(Float16(yValues[base + i]))
                    * invRms * Quantization.bf16ToFloat(Quantization.bf16Bits(wValues[i]))
                    * activation
            }
        }
        return (Fp16Buffer.read(outBuf, count: count), want)
    }

    @Test(arguments: [LinearAttentionConfig.OutputGate.silu, .sigmoid])
    func gatedNormAppliesTheConfiguredActivation(
        gate: LinearAttentionConfig.OutputGate
    ) throws {
        guard let r = try Self.runGatedNorm(gate: gate, seed: 0x6761_7465_0001)
        else { return }
        let relErr = RelError.compute(actual: r.got, reference: r.want)
        #expect(relErr < Tolerance.fp16Reduction, "\(gate) relErr=\(relErr)")
    }

    /// silu and sigmoid must actually produce different output, or the test
    /// above would pass on a kernel that ignored the setting.
    @Test func theTwoGatesDisagree() throws {
        guard let silu = try Self.runGatedNorm(gate: .silu, seed: 0x6761_7465_0002),
              let sigmoid = try Self.runGatedNorm(gate: .sigmoid, seed: 0x6761_7465_0002)
        else { return }
        let diff = RelError.compute(actual: silu.got, reference: sigmoid.got)
        #expect(diff > 0.2, "silu and sigmoid gates agreed too closely: \(diff)")
    }

    /// Qwen3.8 Flash Next gates with sigmoid; Qwen 3.6 with silu. Both are
    /// read straight off the architecture profile, so a regression there is
    /// silent everywhere else.
    @Test func theArchitectureProfilesCarryTheRightGate() {
        #expect(ArchConfig.qwen38FlashNext.linearAttention.outputGate == .sigmoid)
        #expect(ArchConfig.qwen36_35B_A3B.linearAttention.outputGate == .silu)
    }
}
