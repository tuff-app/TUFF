import Foundation
import Metal

/// Batched affine-quantized projection for chunked prefill: `Y[M, N] =
/// X[M, K] * W[N, K]^T` on the MPP tensor path, for INT4 or INT8 weights at
/// group 32 or 64.
///
/// Prefill used to run the dense and shared MLPs, and every projection of a
/// group-32 checkpoint, as one GEMV per token. On a 16 GB M2 that was 96% of
/// Gemma 4 E4B's prefill GPU time and about 30% of Qwen 3.8 Flash Next's.
/// This reads each weight tile once per tile of tokens instead, choosing an
/// 8-, 16-, 32- or 64-row tile by the row count so a routed expert with a
/// handful of tokens does not pay for a 64-row matmul.
///
/// Nil where the MSL 4 tensor pipeline does not build (macOS 15, or a GPU
/// whose compiler rejects MPP cooperative tensors); callers keep their
/// per-token path there.
final class MPPPrefillAffineQMM {
    static let tileM = 64
    static let tileN = 32
    static let tileK = 64

    let bits: Int
    let groupSize: Int
    /// Pipelines by tile shape, smallest M first. A routed expert sees a
    /// handful of rows per chunk, where a 64-row tile is mostly padding.
    private let tiles: [(rows: Int, columns: Int, pso: MTLComputePipelineState)]
    /// The 128 x 16 tile with a 256-deep K step, for whole 128-row blocks.
    /// Optional: a device that cannot build it, or whose results differ from
    /// the tiles above in a check at load, keeps those tiles.
    private let wideTile: MTLComputePipelineState?
    /// Projections this instance has encoded. Tests read it to confirm which
    /// path a prefill took.
    private(set) var encodedProjections = 0

    init?(context: MetalContext, bits: Int, groupSize: Int,
          shapes: [(suffix: String, rows: Int, columns: Int)] = MPPPrefillAffineQMM.shapes) {
        guard [4, 8].contains(bits), [32, 64].contains(groupSize),
              let library = try? MetalContext.privateLibrary(device: context.device,
                                                             module: "tensorops")
        else { return nil }
        var tiles: [(rows: Int, columns: Int, pso: MTLComputePipelineState)] = []
        for shape in shapes {
            let name = "mpp_prefill_affine_qmm_f16_w\(bits)g\(groupSize)\(shape.suffix)"
            guard let function = library.makeFunction(name: name),
                  let pso = try? context.device.makeComputePipelineState(function: function)
            else { return nil }
            tiles.append((shape.rows, shape.columns, pso))
        }
        self.bits = bits
        self.groupSize = groupSize
        self.tiles = tiles
        let wide = library.makeFunction(
            name: "mpp_prefill_affine_qmm_f16_w\(bits)g\(groupSize)_m128").flatMap {
                try? context.device.makeComputePipelineState(function: $0)
            }.flatMap { $0.maxTotalThreadsPerThreadgroup >= $0.threadExecutionWidth * 4 ? $0 : nil }
        if let wide, let reference = tiles.last,
           autoreleasepool(invoking: {
               Self.wideTileMatches(reference, wide: wide, context: context,
                                    bits: bits, groupSize: groupSize)
           }) {
            self.wideTile = wide
        } else {
            self.wideTile = nil
        }
    }

    /// Runs one small projection through the wide tile and through
    /// `reference` and compares the bytes. The wide tile sums 256 products
    /// per step where the others sum 64, and the order inside a step belongs
    /// to the GPU, so a device that rounds this probe differently keeps the
    /// reference tiles. One input cannot prove every case; the parity tests
    /// cover the model shapes on the GPUs they run on.
    private static func wideTileMatches(
        _ reference: (rows: Int, columns: Int, pso: MTLComputePipelineState),
        wide: MTLComputePipelineState, context: MetalContext,
        bits: Int, groupSize: Int
    ) -> Bool {
        // Two K steps and an N that leaves a partial tile in both shapes.
        let m = wideTileRows, n = 40, k = 2 * wideTileK
        let device = context.device
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(bits * 131 + groupSize)
        func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 16
        }
        func unit() -> Float { Float(next() & 0xFFFF) / 65_536 }
        func bf16(_ value: Float) -> UInt16 { UInt16(truncatingIfNeeded: value.bitPattern >> 16) }
        let packed = (0..<(n * k * bits / 8)).map { _ in UInt8(truncatingIfNeeded: next()) }
        let groups = n * k / groupSize
        let scaleBase: Float = bits == 4 ? 0.004 : 0.0003
        let scales = (0..<groups).map { _ in bf16(scaleBase * (0.5 + unit())) }
        let biases = (0..<groups).map { _ in bf16(-0.04 + 0.02 * unit()) }
        let x = (0..<(m * k)).map { _ in Float16(4 * unit() - 2) }
        let yBytes = m * n * MemoryLayout<Float16>.stride
        guard let weights = device.makeBuffer(bytes: packed, length: packed.count,
                                              options: .storageModeShared),
              let sBuf = device.makeBuffer(bytes: scales, length: scales.count * 2,
                                           options: .storageModeShared),
              let bBuf = device.makeBuffer(bytes: biases, length: biases.count * 2,
                                           options: .storageModeShared),
              let xBuf = device.makeBuffer(bytes: x, length: x.count * 2,
                                           options: .storageModeShared),
              let yReference = device.makeBuffer(length: yBytes, options: .storageModeShared),
              let yWide = device.makeBuffer(length: yBytes, options: .storageModeShared),
              // Its own queue, so the probe leaves the model's queue untouched.
              let queue = device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return false }
        // Different fills, so a kernel that writes nothing cannot pass.
        memset(yReference.contents(), 0x00, yBytes)
        memset(yWide.contents(), 0xFF, yBytes)
        for (pso, rows, columns, y) in [
            (reference.pso, reference.rows, reference.columns, yReference),
            (wide, wideTileRows, wideTileColumns, yWide),
        ] {
            dispatch(encoder, pso: pso, rows: rows, columns: columns,
                     weights: weights, weightsOffset: 0, scales: sBuf, scalesOffset: 0,
                     biases: bBuf, biasesOffset: 0, x: xBuf, xOffset: 0, y: y, yOffset: 0,
                     m: m, n: n, k: k)
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
            && memcmp(yReference.contents(), yWide.contents(), yBytes) == 0
    }

    static let shapes: [(suffix: String, rows: Int, columns: Int)] = [
        ("_m8", 8, 64), ("_m16", 16, 64), ("_m32", 32, 32), ("", tileM, tileN),
    ]

    static let wideTileRows = 128
    static let wideTileColumns = 16
    static let wideTileK = 256

    /// Rows of an `m`-row projection with inner size `k` that the wide tile
    /// serves: every whole 128-row block. On a 16 GB M2 it runs 10-15%
    /// faster than the 64 x 32 tile, with identical results, but a partial
    /// block costs as much as a full one, so the remainder keeps the tiles
    /// above.
    func wideTileRows(m: Int, k: Int) -> Int {
        guard wideTile != nil, k.isMultiple(of: Self.wideTileK) else { return 0 }
        return m / Self.wideTileRows * Self.wideTileRows
    }

    /// Whether a projection of this shape can run here. K must fill whole
    /// 64-wide tiles; X and Y rows must be contiguous.
    static func supports(k: Int, n: Int, m: Int) -> Bool {
        m > 0 && n > 0 && k > 0 && k.isMultiple(of: tileK)
    }

    /// Encodes the projection, or returns false without encoding when the
    /// shape or an offset cannot be served.
    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int,
                scales: MTLBuffer, scalesOffset: Int,
                biases: MTLBuffer, biasesOffset: Int,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                m: Int, n: Int, k: Int,
                referenceTiles: Bool = false) -> Bool {
        guard accepts(scalesOffset: scalesOffset, biasesOffset: biasesOffset,
                      xOffset: xOffset, yOffset: yOffset, m: m, n: n, k: k),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return false
        }
        encode(into: encoder,
               weights: weights, weightsOffset: weightsOffset,
               scales: scales, scalesOffset: scalesOffset,
               biases: biases, biasesOffset: biasesOffset,
               x: x, xOffset: xOffset, y: y, yOffset: yOffset,
               m: m, n: n, k: k, referenceTiles: referenceTiles)
        encoder.endEncoding()
        return true
    }

    /// Whether `encode(into:)` will serve these arguments.
    func accepts(scalesOffset: Int, biasesOffset: Int,
                 xOffset: Int, yOffset: Int, m: Int, n: Int, k: Int) -> Bool {
        let halfBytes = MemoryLayout<Float16>.stride
        return Self.supports(k: k, n: n, m: m)
            && k.isMultiple(of: groupSize)
            && scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride)
            && biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride)
            && xOffset.isMultiple(of: halfBytes)
            && yOffset.isMultiple(of: halfBytes)
    }

    /// Encodes into an encoder the caller owns, so many projections can share
    /// one concurrent pass. The caller checks `accepts` first.
    /// `referenceTiles` keeps the wide tile out, for parity tests.
    func encode(into encoder: MTLComputeCommandEncoder,
                weights: MTLBuffer, weightsOffset: Int,
                scales: MTLBuffer, scalesOffset: Int,
                biases: MTLBuffer, biasesOffset: Int,
                x: MTLBuffer, xOffset: Int,
                y: MTLBuffer, yOffset: Int,
                m: Int, n: Int, k: Int,
                referenceTiles: Bool = false) {
        encodedProjections += 1
        // Whole 128-row blocks take the wide tile and the remaining rows the
        // smallest tile that covers them. Each output row is computed on its
        // own, so splitting the rows does not change any result.
        let wideRows = referenceTiles ? 0 : wideTileRows(m: m, k: k)
        if wideRows > 0, let wideTile {
            Self.dispatch(encoder, pso: wideTile, rows: Self.wideTileRows,
                     columns: Self.wideTileColumns,
                     weights: weights, weightsOffset: weightsOffset,
                     scales: scales, scalesOffset: scalesOffset,
                     biases: biases, biasesOffset: biasesOffset,
                     x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                     m: wideRows, n: n, k: k)
        }
        let rest = m - wideRows
        guard rest > 0 else { return }
        let halfBytes = MemoryLayout<Float16>.stride
        let tile = tiles.first { rest <= $0.rows } ?? tiles[tiles.count - 1]
        Self.dispatch(encoder, pso: tile.pso, rows: tile.rows, columns: tile.columns,
                 weights: weights, weightsOffset: weightsOffset,
                 scales: scales, scalesOffset: scalesOffset,
                 biases: biases, biasesOffset: biasesOffset,
                 x: x, xOffset: xOffset + wideRows * k * halfBytes,
                 y: y, yOffset: yOffset + wideRows * n * halfBytes,
                 m: rest, n: n, k: k)
    }

    private static func dispatch(_ encoder: MTLComputeCommandEncoder,
                          pso: MTLComputePipelineState, rows: Int, columns: Int,
                          weights: MTLBuffer, weightsOffset: Int,
                          scales: MTLBuffer, scalesOffset: Int,
                          biases: MTLBuffer, biasesOffset: Int,
                          x: MTLBuffer, xOffset: Int,
                          y: MTLBuffer, yOffset: Int,
                          m: Int, n: Int, k: Int) {
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m), nValue = UInt32(n), kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + columns - 1) / columns,
                    height: (m + rows - 1) / rows,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pso.threadExecutionWidth * 4,
                                           height: 1, depth: 1))
    }
}
