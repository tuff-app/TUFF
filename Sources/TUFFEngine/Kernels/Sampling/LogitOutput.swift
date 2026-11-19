import Foundation
import Metal
import Synchronization

/// Swift wrapper for `logit_softcap_softmax`.
///
///   z'[i] = softcap * tanh(z[i] / softcap)
///   p[i]  = exp(z'[i] - max_j z'[j]) / sum_j exp(z'[j] - max_j z'[j])
///
/// One threadgroup, single online-softmax pass (Milakov & Gimelshein). FP16
/// storage in and out, FP32 accumulation. The softcap value lives in the
/// kernel signature instead of being hardcoded so that downstream callers
/// (and tests) can disable it by passing a very large number.
final class LogitSoftcapSoftmax {
    private let pso: MTLComputePipelineState
    /// The same result from three dispatches, two of them one thread per
    /// logit; nil where it is unavailable or did not match on this GPU.
    private let split: Split?

    /// Whether `encode` uses the three-dispatch path for a vocabulary that fits.
    var usesSplitPath: Bool { split != nil }

    private struct Split {
        let values: MTLComputePipelineState
        let reduce: MTLComputePipelineState
        let normalize: MTLComputePipelineState
        /// The capped logits as FP32, written by `values` and read by the
        /// other two.
        let capped: MTLBuffer
        /// The maximum and the reciprocal of the sum.
        let stats: MTLBuffer
        let capacity: Int
        /// The cap the comparison ran with; another cap uses the single kernel.
        let softcap: Float
    }

    enum Path {
        /// The split path where it is available and matches on this GPU.
        /// `TUFF_SOFTMAX_SPLIT=off` keeps the single kernel, for comparisons.
        case automatic
        case single
        /// The split path without the comparison; the parity tests use it.
        case split
    }

    /// `vocab` sizes the split path's scratch (4 bytes per logit). Without
    /// it only the single kernel is used.
    init(context: MetalContext, vocab: Int? = nil, softcap: Float = 30.0,
         path: Path = .automatic) throws {
        let single = try context.pipeline("logit_softcap_softmax")
        self.pso = single
        let wanted: Bool
        switch path {
        case .single: wanted = false
        case .split: wanted = true
        case .automatic:
            wanted = ProcessInfo.processInfo.environment["TUFF_SOFTMAX_SPLIT"] != "off"
        }
        guard wanted, let vocab, vocab > 0,
              let values = try? context.pipeline("logit_softcap_values"),
              let reduce = try? context.pipeline("logit_softmax_reduce"),
              let normalize = try? context.pipeline("logit_softmax_normalize"),
              let capped = context.device.makeBuffer(
                  length: vocab * MemoryLayout<Float>.stride, options: .storageModePrivate),
              let stats = context.device.makeBuffer(
                  length: 2 * MemoryLayout<Float>.stride, options: .storageModePrivate)
        else {
            self.split = nil
            return
        }
        capped.label = "softmax capped logits"
        stats.label = "softmax max and 1/sum"
        let split = Split(values: values, reduce: reduce, normalize: normalize,
                          capped: capped, stats: stats, capacity: vocab, softcap: softcap)
        if path == .split || Self.splitMatchesOnce(split, single: single,
                                                   device: context.device, softcap: softcap) {
            self.split = split
        } else {
            self.split = nil
        }
    }

    /// Logits in each comparison: the largest vocabulary the sampler takes,
    /// less a few so the 256 chains differ in length.
    private static let comparisonCount = 262_144 - 37

    /// One comparison per set of compiled kernels and cap: every session on
    /// a loaded model builds its own sampler, and a context hands each of
    /// them the same pipeline objects. Another context's library is compared
    /// again. The entries keep their pipelines alive, so an identity cannot
    /// be reused.
    private static let comparisons = Mutex<[Comparison]>([])

    private static let comparisonCacheLimit = 16

    private struct Comparison: @unchecked Sendable {
        let pipelines: [MTLComputePipelineState]
        let softcap: UInt32
        let matches: Bool

        func covers(_ other: [MTLComputePipelineState], softcap: UInt32) -> Bool {
            self.softcap == softcap && pipelines.count == other.count
                && zip(pipelines, other).allSatisfy { $0 === $1 }
        }
    }

    private static func splitMatchesOnce(_ split: Split, single: MTLComputePipelineState,
                                         device: MTLDevice, softcap: Float) -> Bool {
        let pipelines = [single, split.values, split.reduce, split.normalize]
        if let known = comparisons.withLock({ entries in
            entries.first { $0.covers(pipelines, softcap: softcap.bitPattern) }?.matches
        }) { return known }
        let matches = autoreleasepool { () -> Bool in
            // Its own scratch at a fixed size, so the answer does not depend
            // on which sampler happened to ask first.
            guard let capped = device.makeBuffer(
                      length: comparisonCount * MemoryLayout<Float>.stride,
                      options: .storageModePrivate),
                  let stats = device.makeBuffer(
                      length: 2 * MemoryLayout<Float>.stride, options: .storageModePrivate)
            else { return false }
            let probe = Split(values: split.values, reduce: split.reduce,
                              normalize: split.normalize, capped: capped, stats: stats,
                              capacity: comparisonCount, softcap: softcap)
            // Each shape exercises a different part: `mixed` has logits far
            // out in both tails; `ordinary` keeps thousands of probabilities
            // above FP16's smallest values; `flat` has every term matter to
            // the sum; `peaked` is one confident token.
            return ComparisonShape.allCases.allSatisfy {
                splitMatches(probe, single: single, device: device, softcap: softcap, shape: $0)
            }
        }
        comparisons.withLock { entries in
            // Another caller may have finished the same comparison meanwhile.
            guard !entries.contains(where: {
                $0.covers(pipelines, softcap: softcap.bitPattern)
            }) else { return }
            // A process normally has one context. Contexts made and dropped
            // over and over would otherwise pin their pipelines here for
            // good; the oldest result is simply measured again if wanted.
            if entries.count >= comparisonCacheLimit { entries.removeFirst() }
            entries.append(Comparison(pipelines: pipelines, softcap: softcap.bitPattern,
                                      matches: matches))
        }
        return matches
    }

    /// Encodes the kernel onto `commandBuffer`. `logits` and `probs` are FP16
    /// buffers of length `v`. Gemma 4 uses `softcap=30.0`.
    func encode(commandBuffer: MTLCommandBuffer,
                       logits: MTLBuffer,
                       probs: MTLBuffer,
                       v: UInt32,
                       softcap: Float = 30.0) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        if let split, Int(v) <= split.capacity, softcap.bitPattern == split.softcap.bitPattern {
            Self.encodeSplit(enc, split, logits: logits, probs: probs, v: v, softcap: softcap)
        } else {
            Self.encodeSingle(enc, pso, logits: logits, probs: probs, v: v, softcap: softcap)
        }
        enc.endEncoding()
    }

    private static func encodeSingle(_ enc: MTLComputeCommandEncoder,
                                     _ pso: MTLComputePipelineState,
                                     logits: MTLBuffer, probs: MTLBuffer,
                                     v: UInt32, softcap: Float) {
        enc.setComputePipelineState(pso)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(probs,  offset: 0, index: 1)
        var vVar       = v
        var softcapVar = softcap
        enc.setBytes(&vVar,       length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&softcapVar, length: MemoryLayout<Float>.size,  index: 3)

        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    private static func encodeSplit(_ enc: MTLComputeCommandEncoder, _ split: Split,
                                    logits: MTLBuffer, probs: MTLBuffer,
                                    v: UInt32, softcap: Float) {
        var vVar       = v
        var softcapVar = softcap
        func perElement(_ pso: MTLComputePipelineState) {
            let width = min(pso.maxTotalThreadsPerThreadgroup, 256)
            enc.dispatchThreads(MTLSize(width: Int(v), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        }

        enc.setComputePipelineState(split.values)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(split.capped, offset: 0, index: 1)
        enc.setBytes(&vVar,       length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&softcapVar, length: MemoryLayout<Float>.size,  index: 3)
        perElement(split.values)

        // The one group of 256 threads the single kernel sums on, so every
        // chain and merge adds the same logits in the same order.
        enc.setComputePipelineState(split.reduce)
        enc.setBuffer(split.capped, offset: 0, index: 0)
        enc.setBuffer(split.stats, offset: 0, index: 1)
        enc.setBytes(&vVar, length: MemoryLayout<UInt32>.size, index: 2)
        let threadsPerGroup = min(Int(split.reduce.maxTotalThreadsPerThreadgroup), 256)
        let group = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(group, threadsPerThreadgroup: group)

        enc.setComputePipelineState(split.normalize)
        enc.setBuffer(split.capped, offset: 0, index: 0)
        enc.setBuffer(split.stats, offset: 0, index: 1)
        enc.setBuffer(probs, offset: 0, index: 2)
        enc.setBytes(&vVar, length: MemoryLayout<UInt32>.size, index: 3)
        perElement(split.normalize)
    }

    /// Runs one set of logits through both paths and compares the bytes.
    /// Each value comes from the same expression in either path, but the two
    /// are compiled separately, so a GPU or driver that rounds one of them
    /// differently keeps the single kernel. One input cannot prove every
    /// case; the parity tests cover more on the GPUs they run on.
    private enum ComparisonShape: CaseIterable {
        case mixed, ordinary, flat, peaked
    }

    private static func splitMatches(_ split: Split, single: MTLComputePipelineState,
                                     device: MTLDevice, softcap: Float,
                                     shape: ComparisonShape) -> Bool {
        let count = split.capacity
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func unit() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((state >> 16) & 0xFFFF) / 65_536
        }
        // Every logit finite: an infinite one can turn both paths' sums into
        // NaN, and two buffers of NaN would compare equal while proving
        // nothing.
        let logits = (0..<count).map { index -> Float16 in
            switch shape {
            case .mixed:
                // Deep in the cap where there is one.
                return Float16((unit() - 0.5) * (index % 97 == 0 ? 400 : 24))
            case .ordinary:
                return Float16((unit() - 0.5) * 12)
            case .flat:
                return Float16((unit() - 0.5) * 0.02)
            case .peaked:
                return index == count / 3 ? 24 : Float16((unit() - 1) * 6)
            }
        }
        let bytes = count * MemoryLayout<Float16>.stride
        guard let input = device.makeBuffer(bytes: logits, length: bytes, options: .storageModeShared),
              let fromSingle = device.makeBuffer(length: bytes, options: .storageModeShared),
              let fromSplit = device.makeBuffer(length: bytes, options: .storageModeShared),
              // Its own queue, so the probe leaves the model's queue untouched.
              let queue = device.makeCommandQueue(),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return false }
        // Different fills, so a path that writes nothing cannot pass.
        memset(fromSingle.contents(), 0x00, bytes)
        memset(fromSplit.contents(), 0xFF, bytes)
        encodeSingle(encoder, single, logits: input, probs: fromSingle,
                     v: UInt32(count), softcap: softcap)
        encodeSplit(encoder, split, logits: input, probs: fromSplit,
                    v: UInt32(count), softcap: softcap)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed,
              memcmp(fromSingle.contents(), fromSplit.contents(), bytes) == 0 else { return false }
        // Equal bytes only count if they are probabilities: all finite, none
        // negative, adding up to one within FP16 rounding of this many terms.
        let probabilities = UnsafeBufferPointer(
            start: fromSplit.contents().bindMemory(to: Float16.self, capacity: count), count: count)
        var sum: Double = 0
        for probability in probabilities {
            guard probability.isFinite, probability >= 0 else { return false }
            sum += Double(probability)
        }
        return abs(sum - 1) < 0.01
    }
}

/// Swift wrapper for `sample`.
///
/// Reads softmaxed probabilities (output of `LogitSoftcapSoftmax`) and writes
/// one UInt32 token id. With `temperature == 0` performs greedy argmax and
/// ignores top-k / top-p / seed. With `temperature > 0` applies top-p against
/// the full distribution, caps the surviving set with top-k, then applies
/// temperature sharpening (`p^(1/T)`) and samples via a seeded PRNG. The
/// plain-temperature fast path uses the same
/// `(seed, position, row)` Gumbel stream as the fused lm_head sampler.
final class Sample {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("sample")
    }

    /// Encodes the sampler. `probs` is an FP16 buffer of length `v`. `outToken`
    /// must point at storage for one UInt32. The seed is the full 64-bit PRNG
    /// state — tests pass identical seeds to assert determinism.
    func encode(commandBuffer: MTLCommandBuffer,
                       probs: MTLBuffer,
                       outToken: MTLBuffer,
                       v: UInt32,
                       temperature: Float = 1.0,
                       topK: UInt32 = 0,
                       topP: Float = 1.0,
                       seed: UInt64,
                       position: UInt32 = 0) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(probs,    offset: 0, index: 0)
        enc.setBuffer(outToken, offset: 0, index: 1)
        var vVar    = v
        var tVar    = temperature
        var kVar    = topK
        var pVar    = topP
        var sVar    = seed
        var posVar  = position
        enc.setBytes(&vVar, length: MemoryLayout<UInt32>.size,  index: 2)
        enc.setBytes(&tVar, length: MemoryLayout<Float>.size,   index: 3)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size,  index: 4)
        enc.setBytes(&pVar, length: MemoryLayout<Float>.size,   index: 5)
        enc.setBytes(&sVar, length: MemoryLayout<UInt64>.size,  index: 6)
        enc.setBytes(&posVar, length: MemoryLayout<UInt32>.size, index: 7)

        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}
enum SampleTopK64Error: Error {
    case unsupportedVocabulary(Int)
    case scratchAllocationFailed
}

/// Three-stage selection retaining 64 candidates per tile for top-k up to 64.
/// Intermediate pairs remain in private GPU memory.
final class SampleTopK64 {
    private static let tileSize = 1024
    private static let keptPerTile = 64

    private let ctx: MetalContext
    private let stage1PSO: MTLComputePipelineState
    private let reducePSO: MTLComputePipelineState
    private let finalPSO: MTLComputePipelineState
    private let stage1Values: MTLBuffer
    private let stage1Indices: MTLBuffer
    private let stage2Values: MTLBuffer
    private let stage2Indices: MTLBuffer
    private let stage1Groups: Int
    private let stage2Groups: Int
    private let stage1Count: Int
    private let stage2Count: Int

    public let vocab: Int
    public let scratchBytes: Int

    public init(context: MetalContext, vocab: Int) throws {
        guard vocab > 0, vocab <= 262_144 else {
            throw SampleTopK64Error.unsupportedVocabulary(vocab)
        }
        self.ctx = context
        self.vocab = vocab
        self.stage1PSO = try context.pipeline("sample_topk64_stage1")
        self.reducePSO = try context.pipeline("sample_topk64_reduce")
        self.finalPSO = try context.pipeline("sample_topk64_final")

        self.stage1Groups = (vocab + Self.tileSize - 1) / Self.tileSize
        self.stage1Count = stage1Groups * Self.keptPerTile
        self.stage2Groups = (stage1Count + Self.tileSize - 1) / Self.tileSize
        self.stage2Count = stage2Groups * Self.keptPerTile

        let stage1ValueBytes = stage1Count * MemoryLayout<Float>.stride
        let stage1IndexBytes = stage1Count * MemoryLayout<UInt32>.stride
        let stage2ValueBytes = stage2Count * MemoryLayout<Float>.stride
        let stage2IndexBytes = stage2Count * MemoryLayout<UInt32>.stride
        self.scratchBytes = stage1ValueBytes + stage1IndexBytes
            + stage2ValueBytes + stage2IndexBytes

        guard let stage1Values = context.device.makeBuffer(
                  length: stage1ValueBytes, options: .storageModePrivate),
              let stage1Indices = context.device.makeBuffer(
                  length: stage1IndexBytes, options: .storageModePrivate),
              let stage2Values = context.device.makeBuffer(
                  length: stage2ValueBytes, options: .storageModePrivate),
              let stage2Indices = context.device.makeBuffer(
                  length: stage2IndexBytes, options: .storageModePrivate)
        else {
            throw SampleTopK64Error.scratchAllocationFailed
        }
        self.stage1Values = stage1Values
        self.stage1Indices = stage1Indices
        self.stage2Values = stage2Values
        self.stage2Indices = stage2Indices
    }

    public func encode(commandBuffer: MTLCommandBuffer,
                       probs: MTLBuffer,
                       outToken: MTLBuffer,
                       temperature: Float,
                       topP: Float,
                       seed: UInt64,
                       topK: UInt32 = 64) {
        precondition((1...64).contains(topK))
        let threads = MTLSize(width: 256, height: 1, depth: 1)

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(stage1PSO)
            enc.setBuffer(probs, offset: 0, index: 0)
            enc.setBuffer(stage1Values, offset: 0, index: 1)
            enc.setBuffer(stage1Indices, offset: 0, index: 2)
            var v = UInt32(vocab)
            enc.setBytes(&v, length: MemoryLayout<UInt32>.size, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: stage1Groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: threads)
            enc.endEncoding()
        }

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(reducePSO)
            enc.setBuffer(stage1Values, offset: 0, index: 0)
            enc.setBuffer(stage1Indices, offset: 0, index: 1)
            enc.setBuffer(stage2Values, offset: 0, index: 2)
            enc.setBuffer(stage2Indices, offset: 0, index: 3)
            var count = UInt32(stage1Count)
            enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: stage2Groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: threads)
            enc.endEncoding()
        }

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(finalPSO)
            enc.setBuffer(stage2Values, offset: 0, index: 0)
            enc.setBuffer(stage2Indices, offset: 0, index: 1)
            enc.setBuffer(outToken, offset: 0, index: 2)
            var count = UInt32(stage2Count)
            var temp = temperature
            var p = topP
            var rngSeed = seed
            enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&temp, length: MemoryLayout<Float>.size, index: 4)
            enc.setBytes(&p, length: MemoryLayout<Float>.size, index: 5)
            enc.setBytes(&rngSeed, length: MemoryLayout<UInt64>.size, index: 6)
            var k = topK
            enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: threads)
            enc.endEncoding()
        }
    }
}
