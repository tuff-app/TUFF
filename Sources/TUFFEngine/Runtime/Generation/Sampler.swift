import Foundation
import Metal

/// Generation knobs threaded from the caller through the `Generator` into the
/// sampler. Pure value type; one per `generate(...)` call.
///
/// Canonical home is here (the sampler is the primary consumer); `Generator`
/// reuses the same type rather than redeclaring it.
public typealias GenerationConfig = TUFFInferenceTypes.GenerationConfig

/// Which path a `sample(...)` call took.
enum SamplePath: Sendable, Equatable {
    case greedyGPU
    case gpuSampled
    case hostPenalty
}

/// Turns `GenerationConfig` + a logits buffer into one token id, staying
/// GPU-resident wherever the kernels allow.
///
/// The built `sample` kernel already does temperature / top-k / top-p / seeded
/// draw / greedy argmax on GPU reading softmaxed probs, so this type's job is:
/// (1) run the softcap+softmax front-end (`logit_softcap_softmax`), (2) apply
/// repetition penalty — the one policy that needs `history` random access — as
/// a single in-place CPU pass over the (shared) logits before the front-end,
/// and (3) derive a per-position seed so a fixed `seed` is reproducible across
/// token positions.
///
/// The chosen id lands in a 1-element UInt32 buffer. The generation loop reads
/// that value after the command buffer completes.
///
/// Truncation follows mlx-lm's sampler order: Top-P is computed from the
/// model's full probability distribution, Top-K caps that surviving set, and
/// temperature is applied only to the final categorical draw.
final class Sampler {
    private let softcap: LogitSoftcapSoftmax
    private let sampleKernel: Sample
    private let topK64Kernel: SampleTopK64
    let vocab: Int
    private let logitSoftcap: Float
    // One bit per vocabulary entry: 32 KiB at the largest supported vocab.
    // Reused on every penalty pass instead of allocating and hashing a Set.
    private var penaltySeenWords: [UInt64]

    init(context: MetalContext, vocab: Int = 262_144,
                logitSoftcap: Float = 30.0) throws {
        self.softcap = try LogitSoftcapSoftmax(context: context, vocab: vocab,
                                               softcap: logitSoftcap)
        self.sampleKernel = try Sample(context: context)
        self.topK64Kernel = try SampleTopK64(context: context, vocab: vocab, maxTopK: 256)
        self.vocab = vocab
        self.logitSoftcap = logitSoftcap
        self.penaltySeenWords = [UInt64](repeating: 0, count: (vocab + 63) / 64)
    }

    /// Encode the sampler onto `commandBuffer`. `logits` is FP16 [vocab],
    /// post-lm_head and pre-softcap, in a `.storageModeShared` buffer (the
    /// repetition-penalty path edits it in place). `probs` is a preallocated
    /// FP16 [vocab] scratch. `outToken` holds one UInt32. `position` indexes the
    /// per-position seed advance. Returns the path taken.
    @discardableResult
    func sample(commandBuffer: MTLCommandBuffer,
                       logits: MTLBuffer,
                       probs: MTLBuffer,
                       history: [Int32],
                       config: GenerationConfig,
                       position: Int,
                       outToken: MTLBuffer) -> SamplePath {
        let v = UInt32(vocab)

        let appliedPenalty = config.repetitionPenalty != 1.0 && !history.isEmpty
        if appliedPenalty {
            applyRepetitionPenaltyInPlace(logits: logits,
                                          history: history,
                                          penalty: config.repetitionPenalty)
        }

        softcap.encode(commandBuffer: commandBuffer,
                       logits: logits, probs: probs, v: v, softcap: logitSoftcap)

        let isGreedy = config.temperature == 0
        let seed = Self.seedFor(config: config, position: position)
        if config.temperature > 0,
           let k = config.topK, (1...256).contains(k) {
            topK64Kernel.encode(commandBuffer: commandBuffer,
                                probs: probs,
                                outToken: outToken,
                                temperature: config.temperature,
                                topP: config.topP ?? 1.0,
                                seed: seed,
                                topK: UInt32(k))
        } else {
            sampleKernel.encode(commandBuffer: commandBuffer,
                                probs: probs, outToken: outToken, v: v,
                                temperature: isGreedy ? 0.0 : config.temperature,
                                topK: UInt32(config.topK ?? 0),
                                topP: config.topP ?? 1.0,
                                seed: seed,
                                position: UInt32(position))
        }

        if appliedPenalty { return .hostPenalty }
        return isGreedy ? .greedyGPU : .gpuSampled
    }

    // MARK: - Repetition penalty (host, in place)

    /// HF convention: for each token id seen in `history`, a positive logit is
    /// divided by `penalty`, a negative logit multiplied. Edits the shared
    /// `logits` buffer in place — no full-buffer copy, only the unique history
    /// entries are touched (counted for the audit).
    ///
    /// The penalty must act on the POST-softcap logit (HF applies it to the
    /// model's output logits, and Gemma's output includes the 30*tanh(z/30)
    /// cap). Real Gemma 4 raw logits reach the hundreds, deep in tanh
    /// saturation, where dividing the raw value by 1.1 moves the capped logit
    /// by ~nothing — the penalty silently no-ops on exactly the
    /// high-confidence tokens that form repetition loops. So: softcap the raw
    /// value, penalize, and invert through atanh so the downstream
    /// softcap+softmax kernel reproduces the penalized capped logit.
    private func applyRepetitionPenaltyInPlace(logits: MTLBuffer,
                                               history: [Int32],
                                               penalty: Float) {
        let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
        penaltySeenWords.withUnsafeMutableBufferPointer { seen in
            // Histories can shrink or change between calls; no prefix is assumed.
            seen.initialize(repeating: 0)
            for id in history {
                guard id >= 0 && Int(id) < vocab else { continue }
                let i = Int(id)
                let word = i >> 6
                let mask = UInt64(1) << (i & 63)
                guard seen[word] & mask == 0 else { continue }
                seen[word] |= mask
                let z = Float(ptr[i])
                let penalized: Float
                if logitSoftcap > 0 {
                    let capped = logitSoftcap * tanhf(z / logitSoftcap)
                    let cappedPenalized = capped > 0 ? capped / penalty : capped * penalty
                    // A saturated negative logit times the penalty can leave the
                    // softcap's open interval; clamp inside it so atanh stays
                    // finite.
                    let limit = logitSoftcap * 0.9999
                    let clamped = max(min(cappedPenalized, limit), -limit)
                    penalized = logitSoftcap * atanhf(clamped / logitSoftcap)
                } else {
                    penalized = z > 0 ? z / penalty : z * penalty
                }
                ptr[i] = Float16(penalized)
            }
        }
    }

    // MARK: - Seed

    /// Deterministic per-position seed when `config.seed != nil` so a fixed seed
    /// reproduces across token positions; clock-derived (non-zero) otherwise.
    /// xorshift64 in the kernel has a fixed point at 0, so we never emit 0.
    static func seedFor(config: GenerationConfig, position: Int) -> UInt64 {
        if let s = config.seed {
            let mixed = Self.splitmix64(s &+ UInt64(bitPattern: Int64(position)))
            return mixed == 0 ? 0x9E3779B97F4A7C15 : mixed
        }
        var t = timespec()
        clock_gettime(CLOCK_MONOTONIC, &t)
        let raw = UInt64(bitPattern: Int64(t.tv_nsec)) &* 0x9E3779B97F4A7C15
            &+ UInt64(bitPattern: Int64(t.tv_sec))
        return raw == 0 ? 0x9E3779B97F4A7C15 : raw
    }

    private static func splitmix64(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
