import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// Qwen4-Exp runtime integration against the synthetic toy: does the four
/// stream layer graph load, and does a token actually come out the other end?
///
/// This is the first thing in the port that runs the whole path rather than a
/// piece of it — embedding tiled across the streams, a hyper-connection read
/// before each block, gated-DeltaNet and full attention on alternating layers,
/// ten routed experts, gated injection back into every stream, and the mixer
/// collapsing four streams into the LM head with no final norm anywhere.
///
/// It says nothing about numerical agreement: the toy's weights are stand-ins.
/// `HyperConnectionGoldenTests` covers the arithmetic against the reference.
@Suite struct Qwen4ExpRunnerTests {

    private func makeRunner(maxContext: Int = 64)
        throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try Qwen4ExpToySynthetic.write()
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen4ExpToy()).withSlotCache()
        let runner = try RealForwardRunner(model: model,
                                           context: ctx,
                                           maxContext: maxContext)
        return (dir, ctx, runner)
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    /// The architecture has no `input_layernorm`, no
    /// `post_attention_layernorm` and no `model.norm`, and the fixture does
    /// not contain them — so a runner that still reached for one would throw
    /// `tensorNotFound` here rather than at generation time.
    @Test func runnerInitDoesNotReachForNormsTheArchitectureLacks() throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.maxContext == 64)
        // The fused greedy head folds a final RMSNorm into the LM head; with a
        // mixer in that position there is nothing to fold.
        #expect(!runner.usesFusedGreedyHead)
    }

    @Test func mixedCacheHitsMatchColdAndFullyCachedDecode() async throws {
        let config = ArchConfig.qwen4ExpToy()
        let dir = try Qwen4ExpToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let coldModel = try Model.load(directoryURL: dir, device: ctx.device,
                                       expecting: config).withSlotCache()
        let mixedModel = try Model.load(directoryURL: dir, device: ctx.device,
                                        expecting: config).withSlotCache()
        // Ten of sixteen experts are routed. Preloading eight guarantees
        // both hits and misses, independently of the router's choices.
        for layer in 0..<config.numLayers {
            _ = try await mixedModel.fetchRoutedExperts(layer: layer,
                                                        experts: Array(0..<8))
        }
        let cold = try RealForwardRunner(model: coldModel, context: ctx, maxContext: 64)
        let mixed = try RealForwardRunner(model: mixedModel, context: ctx, maxContext: 64)
        let logits = try makeLogits(ctx, vocab: config.vocabSize)
        try await cold.produce(token: 7, position: 0, into: logits)
        let expected = Fp16Buffer.read(logits, count: config.vocabSize)
        try await mixed.produce(token: 7, position: 0, into: logits)
        #expect(mixed.totalRoutedExpertCacheHits > 0)
        #expect(mixed.totalRoutedExpertCacheMisses > 0)
        #expect(Fp16Buffer.read(logits, count: config.vocabSize) == expected)
        let missesBeforeRepeat = mixed.totalRoutedExpertCacheMisses
        mixed.reset()
        try await mixed.produce(token: 7, position: 0, into: logits)
        #expect(mixed.totalRoutedExpertCacheMisses == missesBeforeRepeat)
        #expect(Fp16Buffer.read(logits, count: config.vocabSize) == expected)
    }

    @Test func mappedExpertsMatchTheSlotCacheExactly() async throws {
        let config = ArchConfig.qwen4ExpToy()
        let dir = try Qwen4ExpToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let slotModel = try Model.load(directoryURL: dir, device: ctx.device,
                                       expecting: config).withSlotCache()
        let mappedModel = try Model.load(directoryURL: dir, device: ctx.device, expecting: config)
        let slots = try RealForwardRunner(model: slotModel, context: ctx, maxContext: 64)
        let mapped = try RealForwardRunner(model: mappedModel, context: ctx, maxContext: 64)
        let logits = try makeLogits(ctx, vocab: config.vocabSize)
        for (position, token) in [Int32(7), 3, 11, 5].enumerated() {
            try await slots.produce(token: token, position: position, into: logits)
            let expected = Fp16Buffer.read(logits, count: config.vocabSize)
            try await mapped.produce(token: token, position: position, into: logits)
            #expect(Fp16Buffer.read(logits, count: config.vocabSize) == expected)
        }
        #expect(mappedModel.expertCacheAllocatedBytes == 0)
        #expect(slotModel.expertCacheAllocatedBytes > 0)
    }

    @Test(arguments: [ExpertCachePolicy.lfu, .lru])
    func prefetchedExpertsMatchColdLogitsAndCountTheirFirstDemandHit(policy: ExpertCachePolicy) async throws {
        let config = ArchConfig.qwen4ExpToy()
        let dir = try Qwen4ExpToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let coldModel = try Model.load(directoryURL: dir, device: ctx.device,
                                      expecting: config, expertCachePolicy: policy).withSlotCache()
        let prefetchedModel = try Model.load(directoryURL: dir, device: ctx.device,
                                            expecting: config, expertCachePolicy: policy).withSlotCache()
        for layer in 0..<config.numLayers {
            let plan = try #require(try prefetchedModel.planRoutedExperts(
                layer: layer, experts: Array(0..<8), purpose: .prefetch))
            let operation = try prefetchedModel.expertPrefetchOperation(plan: plan)
            try await Task.detached { try operation() }.value
        }
        #expect(prefetchedModel.expertReadMetrics.usefulPrefetchReads == 0)
        let cold = try RealForwardRunner(model: coldModel, context: ctx, maxContext: 64)
        let prefetched = try RealForwardRunner(model: prefetchedModel, context: ctx, maxContext: 64)
        let logits = try makeLogits(ctx, vocab: config.vocabSize)
        try await cold.produce(token: 7, position: 0, into: logits)
        let expected = Fp16Buffer.read(logits, count: config.vocabSize)
        try await prefetched.produce(token: 7, position: 0, into: logits)
        #expect(Fp16Buffer.read(logits, count: config.vocabSize) == expected)
        #expect(prefetchedModel.expertReadMetrics.usefulPrefetchReads >= 2)
        let beforeRepeat = prefetchedModel.expertReadMetrics
        let misses = prefetched.totalRoutedExpertCacheMisses
        prefetched.reset()
        try await prefetched.produce(token: 7, position: 0, into: logits)
        #expect(Fp16Buffer.read(logits, count: config.vocabSize) == expected)
        #expect(prefetched.totalRoutedExpertCacheMisses == misses)
        #expect(prefetchedModel.expertReadMetrics.usefulPrefetchReads == beforeRepeat.usefulPrefetchReads)
    }

    /// One decode step over the full layer graph.
    @Test func decodeProducesLogitsOverTheFourStreamGraph() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = ArchConfig.qwen4ExpToy()
        let logits = try makeLogits(ctx, vocab: config.vocabSize)

        try await runner.produce(token: 7, position: 0, into: logits)

        let values = Fp16Buffer.read(logits, count: config.vocabSize)
        #expect(values.count == config.vocabSize)
        // Uniform stand-in weights still have to produce a finite, non-empty
        // distribution: a NaN here means a stream was left uninitialized, and
        // an all-zero one means a branch never reached the residual.
        #expect(values.allSatisfy { $0.isFinite })
        #expect(values.contains { $0 != 0 })
    }

    /// The per-layer sink is how a forward pass gets bisected against a
    /// reference trace — it found a GDN decoding INT4 at the wrong group size
    /// and a full-attention layer reading stream 0 instead of the stream
    /// mixture. It has to fire once before each layer and once after the last,
    /// with everything already encoded completed, or the residual it reports
    /// belongs to no particular layer.
    @Test func theLayerSinkSeesEveryLayerBoundary() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = ArchConfig.qwen4ExpToy()
        let logits = try makeLogits(ctx, vocab: config.vocabSize)
        let width = config.hyperConnection.streamCount * config.hiddenSize

        var seen: [Int] = []
        var residuals: [Int: [Float]] = [:]
        runner.debugLayerSink = { layer in
            seen.append(layer)
            residuals[layer] = Fp16Buffer.read(runner.debugResidual, count: width)
        }
        try await runner.produce(token: 7, position: 0, into: logits)
        runner.debugLayerSink = nil

        #expect(seen == Array(0...config.numLayers))
        // Layer 0's input is the tiled embedding, so every stream matches.
        let first = residuals[0]!
        #expect(first.allSatisfy { $0.isFinite })
        let stream0 = Array(first[0..<config.hiddenSize])
        let stream1 = Array(first[config.hiddenSize..<(2 * config.hiddenSize)])
        #expect(stream0 == stream1)
        // By the end the blocks have injected, so it no longer does.
        let last = residuals[config.numLayers]!
        #expect(last.allSatisfy { $0.isFinite })
        #expect(last != first)

        // With no sink installed the pass must still produce the same logits.
        let sunk = Fp16Buffer.read(logits, count: config.vocabSize)
        runner.reset()
        try await runner.produce(token: 7, position: 0, into: logits)
        #expect(Fp16Buffer.read(logits, count: config.vocabSize) == sunk)
    }

    /// Several tokens in sequence, which exercises the KV cache on the
    /// full-attention layers and the recurrent state on the linear ones while
    /// the four streams carry across positions.
    @Test func severalDecodeStepsAdvanceWithoutDiverging() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = ArchConfig.qwen4ExpToy()
        let logits = try makeLogits(ctx, vocab: config.vocabSize)

        for position in 0..<6 {
            try await runner.produce(token: Int32(position + 3),
                                     position: position,
                                     into: logits)
            let values = Fp16Buffer.read(logits, count: config.vocabSize)
            #expect(values.allSatisfy { $0.isFinite },
                    "position \(position) produced a non-finite logit")
        }
    }

    /// Reset has to clear the streams as well as the caches, or a second
    /// sequence starts from the first one's residual.
    @Test func resetReturnsTheRunnerToAFreshSequence() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = ArchConfig.qwen4ExpToy()
        let logits = try makeLogits(ctx, vocab: config.vocabSize)

        try await runner.produce(token: 11, position: 0, into: logits)
        let first = Fp16Buffer.read(logits, count: config.vocabSize)

        runner.reset()
        try await runner.produce(token: 11, position: 0, into: logits)
        let second = Fp16Buffer.read(logits, count: config.vocabSize)

        #expect(first == second,
                "the same token at position 0 gave different logits after reset")
    }

    /// The n-gram embedding on its own layer: rows resolved from the token
    /// history, read out of a table file, gated against the layer's queries
    /// and folded into the residual through a dilated convolution. This is the
    /// only test that runs that path end to end.
    @Test func theNgramEmbeddingRunsOnItsLayer() async throws {
        let cfg = ArchConfig.qwen4ExpToy(ngramLayer: 1)
        let dir = try Qwen4ExpToySynthetic.write(config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: cfg).withSlotCache()
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = try makeLogits(ctx, vocab: cfg.vocabSize)

        for position in 0..<4 {
            try await runner.produce(token: Int32(position + 5),
                                     position: position, into: logits)
            let values = Fp16Buffer.read(logits, count: cfg.vocabSize)
            #expect(values.allSatisfy { $0.isFinite },
                    "position \(position) produced a non-finite logit")
        }
        #expect(Fp16Buffer.read(logits, count: cfg.vocabSize).contains { $0 != 0 })
    }

    /// The n-gram hash carries token history, and the convolution carries its
    /// own taps. Reset has to clear both, or a second sequence starts inside
    /// the first one's n-grams.
    @Test func resetClearsTheNgramHistory() async throws {
        let cfg = ArchConfig.qwen4ExpToy(ngramLayer: 1)
        let dir = try Qwen4ExpToySynthetic.write(config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: cfg).withSlotCache()
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 64)
        let logits = try makeLogits(ctx, vocab: cfg.vocabSize)

        // Two tokens, so the history is genuinely non-empty, then the same
        // opening token again after a reset.
        try await runner.produce(token: 9, position: 0, into: logits)
        let first = Fp16Buffer.read(logits, count: cfg.vocabSize)
        try await runner.produce(token: 12, position: 1, into: logits)

        runner.reset()
        try await runner.produce(token: 9, position: 0, into: logits)
        let afterReset = Fp16Buffer.read(logits, count: cfg.vocabSize)
        #expect(first == afterReset,
                "the n-gram history or convolution taps survived a reset")
    }

    /// The embedding is tiled across the four streams, so they start
    /// identical. Each block then injects its output through a *per-stream*
    /// gate, so by the end of a token they must have diverged — if they have
    /// not, the injection is applying one gate to all four and the extra
    /// streams are costing memory while carrying nothing.
    @Test func theFourStreamsDivergeOnceBlocksHaveInjected() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = ArchConfig.qwen4ExpToy()
        let logits = try makeLogits(ctx, vocab: config.vocabSize)
        try await runner.produce(token: 7, position: 0, into: logits)

        let hidden = config.hyperConnection.streamCount * config.hiddenSize
        let residual = Fp16Buffer.read(runner.debugResidual, count: hidden)
        let streams = (0..<config.hyperConnection.streamCount).map { s in
            Array(residual[(s * config.hiddenSize)..<((s + 1) * config.hiddenSize)])
        }
        #expect(streams.allSatisfy { $0.allSatisfy { $0.isFinite } })
        for other in 1..<streams.count {
            #expect(streams[0] != streams[other],
                    "stream \(other) is identical to stream 0")
        }
    }

    /// A checkpoint missing part of the n-gram module has to be refused when
    /// it is opened. Discovering it partway through the first token would mean
    /// a 110 GB install that looked fine until someone typed something.
    @Test func aCheckpointMissingItsNgramTensorsIsRefused() throws {
        let cfg = ArchConfig.qwen4ExpToy(ngramLayer: 1)
        let dir = try Qwen4ExpToySynthetic.write(config: cfg)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Declare the architecture without telling the loader the table
        // exists: the resident tensors are there, the manifest layout is not.
        let manifestURL = dir.appendingPathComponent("manifest.json")
        var root = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        root.removeValue(forKey: "ngramTable")
        try JSONSerialization.data(withJSONObject: root,
                                   options: [.sortedKeys])
            .write(to: manifestURL)

        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: cfg).withSlotCache()
        #expect(throws: (any Error).self) {
            _ = try RealForwardRunner(model: model, context: ctx, maxContext: 64)
        }
    }

    /// The toy routes ten of sixteen experts. Eight slots — the count that
    /// satisfied every earlier model — can no longer hold one token's
    /// selection, and has to be refused rather than silently truncated.
    @Test func aCacheSmallerThanTheRoutedSelectionIsRefused() throws {
        let dir = try Qwen4ExpToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ctx = try MetalContext()
        // The slot count is fixed when the model is opened, not by the
        // runtime configuration, so it has to be constrained here.
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .qwen4ExpToy(),
                                   streamingMode: .pread(slotCount: 8)).withSlotCache()
        #expect(throws: RuntimeConfigurationError.expertCacheTooSmall(
            configured: 8, required: 10)) {
            _ = try RealForwardRunner(model: model, context: ctx, maxContext: 64)
        }
    }
}
