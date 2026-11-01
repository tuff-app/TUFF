import Foundation
import Metal
import Synchronization
import Testing
@testable import TUFFEngine

private actor TwoSessionBarrier {
    private var waiting: CheckedContinuation<Void, Never>?
    func arrive() async {
        if let waiting {
            self.waiting = nil
            waiting.resume()
        } else {
            await withCheckedContinuation { waiting = $0 }
        }
    }
}

@Suite(.serialized) struct LeasedModelRunnerTests {
    @Test func twoSessionsShareStagedWeightsWithoutSharingKV() async throws {
        let directory = try DenseGemmaToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: .gemma4E4BToy(), denseResidency: .bounded(residentLayers: []))
        #expect(model.stagesDenseLayers)
        let options = RuntimeConfiguration(forceLogitsHead: true)
        let lease = FairForwardLease()
        let first = LeasedModelRunner(runner: try ModelForwardRunner(
            model: model, context: context, maxContext: 64, runtimeConfiguration: options), lease: lease)
        let second = LeasedModelRunner(runner: try ModelForwardRunner(
            model: model, context: context, maxContext: 64, runtimeConfiguration: options), lease: lease)
        let firstOutput = context.device.makeBuffer(length: 256 * MemoryLayout<Float16>.stride,
                                                     options: .storageModeShared)!
        let secondOutput = context.device.makeBuffer(length: 256 * MemoryLayout<Float16>.stride,
                                                      options: .storageModeShared)!
        let firstTokens: [Int32] = [2, 7, 19, 5]
        let secondTokens: [Int32] = [3, 11, 4, 23]
        let barrier = TwoSessionBarrier()

        func run(_ runner: LeasedModelRunner, _ output: MTLBuffer,
                 _ tokens: [Int32], interleave: Bool = false) async throws -> [[Float16]] {
            var result: [[Float16]] = []
            for (position, token) in tokens.enumerated() {
                try await runner.produce(token: token, position: position, into: output)
                result.append(Array(UnsafeBufferPointer(
                    start: output.contents().assumingMemoryBound(to: Float16.self), count: 256)))
                if interleave && position == 0 { await barrier.arrive() }
                await Task.yield()
            }
            return result
        }

        let expectedFirst = try await run(first, firstOutput, firstTokens)
        let expectedSecond = try await run(second, secondOutput, secondTokens)
        first.reset()
        second.reset()
        async let concurrentFirst = run(first, firstOutput, firstTokens, interleave: true)
        async let concurrentSecond = run(second, secondOutput, secondTokens, interleave: true)
        let (actualFirst, actualSecond) = try await (concurrentFirst, concurrentSecond)
        #expect(actualFirst == expectedFirst)
        #expect(actualSecond == expectedSecond)
        #expect(first.continuationPosition == firstTokens.count)
        #expect(second.continuationPosition == secondTokens.count)
    }

    /// A session ahead of the image prompt in the lease queue must run between
    /// the prompt's planned chunks, not after the whole prefill.
    @Test func imagePrefillLetsAnotherSessionForwardBetweenChunks() async throws {
        let config = ArchConfig.qwen4ExpToy()
        let directory = try Qwen4ExpToySynthetic.write(config: config)
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: config, streamingMode: .pread(slotCount: 16))
        let lease = FairForwardLease()
        let leased = LeasedModelRunner(runner: try ModelForwardRunner(
            model: model, context: context, maxContext: 128,
            runtimeConfiguration: RuntimeConfiguration()), lease: lease)
        let features = try #require(context.device.makeBuffer(
            length: 4 * config.hiddenSize * 2, options: .storageModeShared))
        let data = features.contents().assumingMemoryBound(to: Float16.self)
        for index in 0..<(4 * config.hiddenSize) { data[index] = Float16(sin(Float(index) * 0.3)) }
        let visionFeatures = VisionFeatures(buffer: features, tokenCount: 4,
            hiddenSize: config.hiddenSize, family: .qwen4Exp, patchGridWidth: 4, patchGridHeight: 4,
            gpuNanoseconds: 0, scratchBytes: 0, attentionVariant: .native72,
            projectorPath: .fallback, expertResidencyTransition: nil, preprocessing: nil)
        var effective = (0..<100).map { Int32(($0 * 7 + 11) % 31) }
        var embedding = effective
        for index in 40..<44 { effective[index] = 8; embedding[index] = 0 }
        let input = try MultimodalPrefillInput(
            effectiveTokenIDs: effective, embeddingTokenIDs: embedding,
            imageSpans: [.init(tokenRange: 40..<44, features: visionFeatures)], family: .qwen4Exp)
        let logits = try #require(context.device.makeBuffer(
            length: config.vocabSize * MemoryLayout<Float16>.stride, options: .storageModeShared))

        let holderOwnsLease = Mutex(false)
        let releaseHolder = Mutex(false)
        let completed = Mutex(0)
        let seenByCompetitor = Mutex(-1)
        func waitUntil(_ condition: () async -> Bool) async throws {
            for _ in 0..<2_000 {
                if await condition() { return }
                try await Task.sleep(for: .milliseconds(5))
            }
            Issue.record("condition was not reached")
        }

        let holder = Task {
            try await lease.withLease {
                holderOwnsLease.withLock { $0 = true }
                while !releaseHolder.withLock({ $0 }) { try await Task.sleep(for: .milliseconds(1)) }
            }
        }
        try await waitUntil { holderOwnsLease.withLock { $0 } }
        let prefill = Task {
            _ = try await leased.prefillMultimodal(
                input: input, startPosition: 0, outputMode: .logits,
                config: .production(chunkTokens: 32), into: logits,
                onProgress: { value in completed.withLock { $0 = value } })
        }
        try await waitUntil { await lease.waitingCount == 1 }
        let competitor = Task {
            try await lease.withLease { seenByCompetitor.withLock { $0 = completed.withLock { $0 } } }
        }
        try await waitUntil { await lease.waitingCount == 2 }
        releaseHolder.withLock { $0 = true }
        try await holder.value
        _ = try await prefill.value
        try await competitor.value
        #expect(completed.withLock { $0 } == 100)
        // Whole-call leasing would have let the prompt finish first.
        #expect(seenByCompetitor.withLock { $0 } < 100)
    }
}
