import Metal
import Foundation
import Darwin
import Testing
@testable import TUFFEngine

@Suite struct SampleTopK64Tests {
    @Test func truncationDefaultsDoNotDisableGreedyEligibility() {
        let config = GenerationConfig(temperature: 0, topK: 64, topP: 0.95)
        #expect(config.isPureGreedy)
    }

    @Test func generationConfigRejectsSamplerStatesTheKernelCannotHonor() throws {
        #expect(throws: GeneratorError.self) {
            try GenerationConfig(temperature: 1, topK: 257, topP: 0.95).validate()
        }
        #expect(throws: GeneratorError.self) {
            try GenerationConfig(temperature: 1, topK: nil, topP: 0.95).validate()
        }
        try GenerationConfig(temperature: 0, topK: nil, topP: 0.95).validate()
    }

    /// `validate()` checked temperature for finiteness from the start and never
    /// checked the penalty beside it, so `--repetition-penalty inf` parsed, passed
    /// validation, reached the kernel and quietly degraded the output rather than
    /// being refused: a prompt that counts to eight broke into another language
    /// mid-count and then apologised for itself. Every CLI float guard without an
    /// upper bound admits infinity, which is how it got that far.
    @Test func generationConfigRejectsAPenaltyThatCannotBeApplied() throws {
        #expect(throws: GeneratorError.self) {
            try GenerationConfig(temperature: 0, repetitionPenalty: .infinity).validate()
        }
        #expect(throws: GeneratorError.self) {
            try GenerationConfig(temperature: 0, repetitionPenalty: 0).validate()
        }
        #expect(throws: GeneratorError.self) {
            try GenerationConfig(temperature: 0, repetitionPenalty: .nan).validate()
        }
        // The default, and an ordinary setting, both stay valid.
        try GenerationConfig(temperature: 0).validate()
        try GenerationConfig(temperature: 0, repetitionPenalty: 1.1).validate()
    }

    private final class Rig {
        let context: MetalContext
        let current: Sample
        let candidate: SampleTopK64
        let probs: MTLBuffer
        let currentOutput: MTLBuffer
        let candidateOutput: MTLBuffer
        let vocab: Int

        init(vocab: Int, maxTopK: Int = 64) throws {
            self.context = try MetalContext()
            self.current = try Sample(context: context)
            self.candidate = try SampleTopK64(context: context, vocab: vocab, maxTopK: maxTopK)
            self.vocab = vocab
            guard let probs = context.device.makeBuffer(
                      length: vocab * MemoryLayout<Float16>.stride,
                      options: .storageModeShared),
                  let currentOutput = context.device.makeBuffer(
                      length: MemoryLayout<UInt32>.stride,
                      options: .storageModeShared),
                  let candidateOutput = context.device.makeBuffer(
                      length: MemoryLayout<UInt32>.stride,
                      options: .storageModeShared)
            else {
                throw MetalError.noDevice
            }
            self.probs = probs
            self.currentOutput = currentOutput
            self.candidateOutput = candidateOutput
        }

        func write(_ values: (Int) -> Float) {
            let ptr = probs.contents().bindMemory(to: Float16.self, capacity: vocab)
            for i in 0..<vocab {
                ptr[i] = Float16(values(i))
            }
        }

        func draw(seed: UInt64,
                  temperature: Float = 1.0,
                  topP: Float,
                  topK: UInt32 = 64) -> (current: UInt32, candidate: UInt32) {
            let cb = context.queue.makeCommandBuffer()!
            current.encode(commandBuffer: cb,
                           probs: probs,
                           outToken: currentOutput,
                           v: UInt32(vocab),
                           temperature: temperature,
                           topK: topK,
                           topP: topP,
                           seed: seed)
            candidate.encode(commandBuffer: cb,
                             probs: probs,
                             outToken: candidateOutput,
                             temperature: temperature,
                             topP: topP,
                             seed: seed, topK: topK)
            cb.commit()
            cb.waitUntilCompleted()
            return (currentOutput.contents().load(as: UInt32.self),
                    candidateOutput.contents().load(as: UInt32.self))
        }
    }

    @Test(arguments: [UInt32(20), 40, 64])
    func productionVocabularyMatchesCurrentSampler(topK: UInt32) throws {
        let rig = try Rig(vocab: 262_144)
        #expect(rig.candidate.scratchBytes == 139_264)
        rig.write { i in
            let mixed = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407
            return Float(UInt32(mixed >> 40) + 1)
                * (1.0 / 16_777_217.0) * (2.0 / Float(rig.vocab))
        }

        for temperature: Float in [0.2, 0.7, 0.85, 1.0, 1.5] {
            for seed: UInt64 in [1, 2, 0x1234_5678_9ABC_DEF0, UInt64.max] {
                let result = rig.draw(seed: seed, temperature: temperature, topP: 0.95, topK: topK)
                #expect(result.candidate == result.current,
                        "temperature \(temperature), seed \(seed): candidate \(result.candidate), current \(result.current)")
            }
        }
    }

    @Test(arguments: [UInt32(20), 40, 64])
    func tiesAndPartialTailMatchCurrentSampler(topK: UInt32) throws {
        let rig = try Rig(vocab: 1_003)
        rig.write { _ in 1.0 }

        for seed in UInt64(1)...UInt64(8) {
            let result = rig.draw(seed: seed, topP: 0.95, topK: topK)
            #expect(result.candidate == result.current,
                    "seed \(seed): candidate \(result.candidate), current \(result.current)")
            #expect(result.candidate < topK)
        }
    }

    @Test func lowTemperatureDoesNotUnderflowTheWholeDrawToZero() throws {
        let rig = try Rig(vocab: 1024)
        rig.write { _ in 1.0 / 1024 }
        var selected = Set<UInt32>()
        for seed in UInt64(1)...UInt64(16) {
            let result = rig.draw(seed: seed, temperature: 0.01, topP: 1)
            #expect(result.current == result.candidate)
            #expect(result.candidate < 64)
            selected.insert(result.candidate)
        }
        // Equal logits must stay a categorical draw at every positive
        // temperature. pow(p, 100) previously erased every probability and
        // picked token zero on every seed.
        #expect(selected.count > 1)
    }

    @Test func topPUsesFullVocabularyMassBeforeTopK() throws {
        let rig = try Rig(vocab: 1_003)
        rig.write { _ in 1.0 / 1_003.0 }

        // The full-distribution 0.95 nucleus is much wider than 64 tokens, so
        // mlx-lm's Top-P-then-Top-K chain leaves all Top-64 entries eligible.
        // The previous Top-K-renormalize-then-Top-P order kept only 61.
        var sawLastThree = false
        for seed in UInt64(1)...UInt64(256) {
            let result = rig.draw(seed: seed, topP: 0.95)
            #expect(result.candidate == result.current)
            #expect(result.candidate < 64)
            if result.candidate >= 61 { sawLastThree = true }
        }
        #expect(sawLastThree, "Top-P incorrectly truncated the renormalized Top-64 set")
    }
    @Test(arguments: [1, 19, 20, 39, 40, 63, 64, 1025, 200003])
    func smallVocabularyAndTailBoundariesMatch(vocab: Int) throws {
        let rig = try Rig(vocab: vocab)
        rig.write { i in Float((i * 37) % 101 + 1) / 1000 }
        for k: UInt32 in [20, 40, 64] {
            for p: Float in [0.01, 0.5, 0.95, 1] {
                for seed: UInt64 in [0, 1, 9, UInt64.max] {
                    let result = rig.draw(seed: seed, temperature: 0.85, topP: p, topK: k)
                    #expect(result.current == result.candidate)
                }
            }
        }
    }

    @Test(arguments: [UInt32(20), 40, 64])
    func nonfiniteAndUnderflowInputsKeepReferenceBehavior(topK: UInt32) throws {
        let rig = try Rig(vocab: 1025)
        let cases: [(Int) -> Float] = [
            { i in i == 1024 ? 0.5 : (i % 3 == 0 ? .nan : -.infinity) },
            { i in i == 7 || i == 1024 ? .infinity : 0.01 },
            { _ in .nan }, { _ in -.infinity }, { _ in 0 },
            { i in Float(i % 13 + 1) * 0.00001 }
        ]
        for values in cases {
            rig.write(values)
            for temperature: Float in [0.2, 1, 1.5] {
                for seed: UInt64 in [0, 1, 42, UInt64.max] {
                    let result = rig.draw(seed: seed, temperature: temperature, topP: 0.95, topK: topK)
                    #expect(result.candidate == result.current)
                }
            }
        }
    }

    @Test(arguments: [UInt32(20), 40, 64])
    func independentSortedCDFMatchesSeededDraw(topK: UInt32) throws {
        let rig = try Rig(vocab: 2049)
        let probabilities = (0..<rig.vocab).map { i in
            Float(Float16(Float((i * 29) % 41 + 1) / 43000))
        }
        rig.write { probabilities[$0] }
        let ranked = probabilities.indices.sorted {
            probabilities[$0] == probabilities[$1] ? $0 < $1 : probabilities[$0] > probabilities[$1]
        }
        for p: Float in [0.001, 0.01, 0.95, 1] {
            var selected = Array(ranked.prefix(Int(topK)))
            if p < 1 {
                var cumulative: Float = 0
                for (rank, index) in selected.enumerated() {
                    cumulative += probabilities[index]
                    if cumulative >= p { selected = Array(selected.prefix(rank + 1)); break }
                }
            }
            let sum = selected.reduce(Float(0)) { $0 + probabilities[$1] }
            for seed: UInt64 in [0, 1, 7, 42, UInt64.max] {
                var state = seed
                func advance() -> UInt64 {
                    state ^= state << 13; state ^= state >> 7; state ^= state << 17
                    return state &* 2685821657736338717
                }
                _ = advance()
                let threshold = Float(UInt32(advance() >> 40)) / 16777216 * sum
                var cumulative: Float = 0
                var expected = selected[0]
                for index in selected {
                    cumulative += probabilities[index]
                    if threshold <= cumulative { expected = index; break }
                }
                let result = rig.draw(seed: seed, topP: p, topK: topK)
                #expect(result.candidate == UInt32(expected))
                #expect(result.current == UInt32(expected))
            }
        }
    }

    @Test func wideTopKMatchesSortedCDFThroughMultipleMerges() throws {
        let rig = try Rig(vocab: 65_539, maxTopK: 256)
        // All inputs tie, so the exact surviving IDs are 0 through K-1.
        // A seed near the end of the CDF exposes a selector capped at 64.
        rig.write { _ in 1 }
        for topK: UInt32 in [32, 65, 128, 256] {
            for seed: UInt64 in [42, 0x1234_5678_9ABC_DEF0, UInt64.max] {
                var state = seed
                for _ in 0..<2 {
                    state ^= state << 13
                    state ^= state >> 7
                    state ^= state << 17
                }
                let bits = UInt32((state &* 2685821657736338717) >> 40)
                let threshold = Float(bits) / 16_777_216 * Float(topK)
                let expected = UInt32(max(0, Int(threshold.rounded(.up)) - 1))
                let cb = try #require(rig.context.queue.makeCommandBuffer())
                rig.candidate.encode(commandBuffer: cb, probs: rig.probs, outToken: rig.candidateOutput,
                                     temperature: 1, topP: 1, seed: seed, topK: topK)
                cb.commit()
                cb.waitUntilCompleted()
                try checkCommandBufferError(cb)
                #expect(rig.candidateOutput.contents().load(as: UInt32.self) == expected)
            }
        }
    }

    @Test func untruncatedGumbelDrawCannotRoundUpToOne() throws {
        let rig = try Rig(vocab: 2)
        rig.write { $0 == 0 ? 1 : 0.000_001 }
        let cb = try #require(rig.context.queue.makeCommandBuffer())
        // SplitMix64(seed XOR 1) is UInt64.max. Token 1 used to
        // receive +infinity and win even at this very low temperature.
        rig.current.encode(commandBuffer: cb, probs: rig.probs, outToken: rig.currentOutput,
                           v: 2, temperature: 0.1, topK: 0, topP: 1,
                           seed: 0x3162_8AF6_7B21_31AA, position: 0)
        cb.commit()
        cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        #expect(rig.currentOutput.contents().load(as: UInt32.self) == 0)
    }

    @Test func optionalSequentialSamplerTiming() throws {
        guard ProcessInfo.processInfo.environment["TUFF_SAMPLER_BENCHMARK"] == "1" else { return }
        let cases: [(UInt32, Int)] = [(20, 248320), (40, 200064), (64, 262144)]
        for (k, vocab) in cases {
            let rig = try Rig(vocab: vocab)
            rig.write { i in Float((i * 37) % 1009 + 1) / 132000000 }
            var oldTimes = [Double](), newTimes = [Double]()
            var oldGPU = [Double](), newGPU = [Double]()
            for repetition in 0..<40 {
                for candidate in repetition.isMultiple(of: 2) ? [false, true] : [true, false] {
                    let cb = rig.context.queue.makeCommandBuffer()!
                    let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    if candidate {
                        rig.candidate.encode(commandBuffer: cb, probs: rig.probs,
                          outToken: rig.candidateOutput, temperature: 1, topP: 0.95, seed: 42, topK: k)
                    } else {
                        rig.current.encode(commandBuffer: cb, probs: rig.probs,
                          outToken: rig.currentOutput, v: UInt32(rig.vocab), temperature: 1,
                          topK: k, topP: 0.95, seed: 42)
                    }
                    cb.commit(); cb.waitUntilCompleted()
                    #expect(cb.status == .completed)
                    let elapsed = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start) / 1e6
                    if repetition >= 4 {
                        let gpu = max(0, cb.gpuEndTime - cb.gpuStartTime) * 1000
                        if candidate { newTimes.append(elapsed); newGPU.append(gpu) }
                        else { oldTimes.append(elapsed); oldGPU.append(gpu) }
                    }
                }
                #expect(rig.currentOutput.contents().load(as: UInt32.self)
                  == rig.candidateOutput.contents().load(as: UInt32.self))
            }
            let data = try JSONSerialization.data(withJSONObject:
              ["top_k": k, "vocab": rig.vocab, "reference_ms": oldTimes,
               "hierarchical_ms": newTimes, "reference_gpu_ms": oldGPU,
               "hierarchical_gpu_ms": newGPU], options: [.sortedKeys])
            print("[sampler timing] " + String(decoding: data, as: UTF8.self))
        }
    }

}
