import Foundation
import Metal
import Testing
import TUFFValidationSupport
@testable import TUFFEngine

/// The runner against the NumPy reference (`Scripts/inkling_reference.py`),
/// which itself matches the publisher's implementation on a tiny checkpoint.
/// Needs the installed model: set `TUFF_INKLING_OUTPUT` to its folder.
@Suite(.serialized) struct InklingReferenceTests {
    private struct Reference: Decodable {
        struct Step: Decodable {
            let top_ids: [Int]
            let top_logits: [Double]
        }
        let input_ids: [Int]
        let steps: [Step]
    }

    private static func reference() throws -> Reference {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Tokenization/Fixtures/InklingTokenizer/greedy_reference.json")
        return try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }

    /// Twelve greedy steps: the same token each time, and the ten largest
    /// logits within half a unit of the reference after the softcap. Once the
    /// way a current Mac runs (batched prompt block, fused decode) and once
    /// with the per-row kernels throughout.
    @Test(arguments: [true, false])
    func pinnedChatGreedyReference(fast: Bool) async throws {
        guard let path = ProcessInfo.processInfo.environment["TUFF_INKLING_OUTPUT"] else { return }
        let reference = try Self.reference()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device,
                                   expecting: .inkling2Mini)
        let runner = try InklingForwardRunner(model: model, context: context, maxContext: 256,
                                              runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true),
                                              batchedProducts: fast, fusedDecode: fast)
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                            options: .storageModeShared))
        let prefill = try await runner.prefillChunked(
            tokens: reference.input_ids.map(Int32.init)[...], startPosition: 0, outputMode: .logits,
            config: RuntimeConfiguration().prefillConfig, into: logits, onProgress: { _ in })
        #expect(prefill.newPosition == reference.input_ids.count)
        let cap = Float(model.config.finalLogitSoftcap)
        for (step, expected) in reference.steps.enumerated() {
            let actual = Fp16Buffer.read(logits, count: model.config.vocabSize)
            let top = try #require(actual.indices.max(by: { actual[$0] < actual[$1] }))
            #expect(actual.allSatisfy { $0.isFinite })
            #expect(top == expected.top_ids[0], "step \(step)")
            for (index, value) in zip(expected.top_ids, expected.top_logits) {
                let capped = Double(tanh(Float(actual[index]) / cap) * cap)
                #expect(abs(capped - value) < 0.5, "step \(step) token \(index): \(capped) vs \(value)")
            }
            guard step + 1 < reference.steps.count else { break }
            try await runner.produce(token: Int32(expected.top_ids[0]),
                                     position: reference.input_ids.count + step, into: logits)
        }
    }

    /// A sequence that is captured, replaced and restored continues exactly
    /// as it would have, and a rewind to a checkpoint replays the same tokens.
    @Test func stateRestoresAndRewinds() async throws {
        guard let path = ProcessInfo.processInfo.environment["TUFF_INKLING_OUTPUT"] else { return }
        let reference = try Self.reference()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device,
                                   expecting: .inkling2Mini)
        let runner = try InklingForwardRunner(model: model, context: context, maxContext: 256,
                                              runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                            options: .storageModeShared))
        let prompt = reference.input_ids.map(Int32.init)
        func argmax() -> Int {
            let values = Fp16Buffer.read(logits, count: model.config.vocabSize)
            return values.indices.max(by: { values[$0] < values[$1] })!
        }
        func feed(_ tokens: [Int32], from position: Int) async throws {
            for (offset, token) in tokens.enumerated() {
                try await runner.produce(token: token, position: position + offset, into: logits)
            }
        }
        let split = 8
        try await feed(Array(prompt[..<split]), from: 0)
        let checkpoint = try runner.capturePrefixCheckpoint()
        try await feed(Array(prompt[split...]), from: split)
        #expect(argmax() == reference.steps[0].top_ids[0])
        let whole = try runner.captureState()

        // Another sequence takes the runner, then the first one comes back.
        runner.reset()
        try await feed([prompt[0], prompt[1], prompt[2]], from: 0)
        try runner.restoreState(whole)
        #expect(runner.continuationPosition == prompt.count)
        try await feed([Int32(reference.steps[0].top_ids[0])], from: prompt.count)
        #expect(argmax() == reference.steps[1].top_ids[0])

        // Back to the checkpoint: the rest of the prompt gives the same answer.
        try runner.rewind(to: checkpoint)
        #expect(runner.continuationPosition == split)
        try await feed(Array(prompt[split...]), from: split)
        #expect(argmax() == reference.steps[0].top_ids[0])
    }

    /// A block stopped part way leaves the sequence at the block's first
    /// position, and the same tokens read again give the reference's answer.
    @Test func cancelledBlockLeavesTheSequenceUsable() async throws {
        guard let path = ProcessInfo.processInfo.environment["TUFF_INKLING_OUTPUT"] else { return }
        let reference = try Self.reference()
        let (context, model, logits) = try Self.loaded(path)
        let runner = try InklingForwardRunner(model: model, context: context, maxContext: 256,
                                              runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let prompt = reference.input_ids.map(Int32.init)
        let split = 8
        func read(_ range: Range<Int>) async throws {
            _ = try await runner.prefillChunked(
                tokens: prompt[range], startPosition: range.lowerBound, outputMode: .logits,
                config: RuntimeConfiguration().prefillConfig, into: logits, onProgress: { _ in })
        }
        try await read(0..<split)

        // Stop inside the second block, after 21 of its 48 layer visits.
        nonisolated(unsafe) let output = logits
        let rest = Array(prompt[split...])
        let task = Task {
            _ = try await runner.prefillChunked(
                tokens: rest[...], startPosition: split, outputMode: .logits,
                config: RuntimeConfiguration().prefillConfig, into: output, onProgress: { _ in })
        }
        runner.afterAttention = { if $0 == 20 { task.cancel() } }
        await #expect(throws: CancellationError.self) { try await task.value }
        runner.afterAttention = nil
        #expect(runner.continuationPosition == split)

        try await read(split..<prompt.count)
        let top = Self.top(logits, model)
        #expect(top[0].id == reference.steps[0].top_ids[0])
        #expect(abs(Double(top[0].value) - reference.steps[0].top_logits[0]) < 0.5)
    }

    private static func loaded(_ path: String) throws -> (MetalContext, Model, MTLBuffer) {
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device,
                                   expecting: .inkling2Mini)
        guard let logits = context.device.makeBuffer(length: model.config.vocabSize * 2,
                                                     options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return (context, model, logits)
    }

    /// The largest logits, softcapped as the reference reports them.
    private static func top(_ logits: MTLBuffer, _ model: Model, count: Int = 6) -> [(id: Int, value: Float)] {
        let values = Fp16Buffer.read(logits, count: model.config.vocabSize)
        let cap = Float(model.config.finalLogitSoftcap)
        return values.indices.sorted { values[$0] > values[$1] }.prefix(count)
            .map { ($0, tanh(Float(values[$0]) / cap) * cap) }
    }

    /// A prompt read in blocks against the same prompt read a token at a time.
    /// With the per-row kernels both ways the arithmetic is the same, so the
    /// logits must agree: this is the check that a block carries keys, values,
    /// taps and loop messages across rows and blocks correctly. The batched
    /// products round differently, which can change a near-tied expert choice
    /// and with it everything downstream, so they are held to the decision at
    /// a position with a clear winner. The prompt spans three blocks, and
    /// every sequence then decodes.
    @Test func blocksMatchTokenByToken() async throws {
        guard let path = ProcessInfo.processInfo.environment["TUFF_INKLING_OUTPUT"] else { return }
        let (context, model, logits) = try Self.loaded(path)
        let passage = try Self.windowReference().input_ids.map(Int32.init)
        let prompt = Array((0..<5).flatMap { _ in passage }.prefix(300))
        let steps = 4

        func run(prefill: Bool, batched: Bool) async throws -> (tops: [[(id: Int, value: Float)]], batchedBlocks: Int) {
            let runner = try InklingForwardRunner(
                model: model, context: context, maxContext: 512,
                runtimeConfiguration: RuntimeConfiguration(prefillEnabled: prefill, forceLogitsHead: true),
                batchedProducts: batched, fusedDecode: batched)
            if prefill {
                _ = try await runner.prefillChunked(
                    tokens: prompt[...], startPosition: 0, outputMode: .logits,
                    config: RuntimeConfiguration().prefillConfig, into: logits, onProgress: { _ in })
            } else {
                for (position, token) in prompt.enumerated() {
                    try await runner.produce(token: token, position: position, into: logits)
                }
            }
            var tops = [Self.top(logits, model)]
            for step in 0..<steps {
                try await runner.produce(token: Int32(tops[step][0].id), position: prompt.count + step,
                                         into: logits)
                tops.append(Self.top(logits, model))
            }
            return (tops, runner.batchedBlocks)
        }

        let single = try await run(prefill: false, batched: false)
        let perRow = try await run(prefill: true, batched: false)
        let batched = try await run(prefill: true, batched: true)
        #expect(single.batchedBlocks == 0 && perRow.batchedBlocks == 0)
        if #available(macOS 26, *) { #expect(batched.batchedBlocks == 3) }
        for (step, (expected, actual)) in zip(single.tops, perRow.tops).enumerated() {
            #expect(actual.map(\.id) == expected.map(\.id), "per-row step \(step)")
            for (mine, theirs) in zip(actual, expected) {
                #expect(abs(mine.value - theirs.value) < 0.05, "per-row step \(step) token \(theirs.id)")
            }
        }
        // The decoded tokens follow from the prompt's last position, so only
        // a clear winner there is a fair comparison.
        let expected = single.tops[0], actual = batched.tops[0]
        #expect(expected[0].value - expected[1].value > 3)
        #expect(actual[0].id == expected[0].id)
        #expect(batched.tops.allSatisfy { $0.allSatisfy { $0.value.isFinite } })
    }

    private struct WindowReference: Decodable {
        struct Position: Decodable {
            let norms: [Float]
            let projections: [Float]
        }
        let window: Int
        let input_ids: [Int]
        let attention: [Position]
    }

    private static func windowReference() throws -> WindowReference {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Tokenization/Fixtures/InklingTokenizer/window_reference.json")
        return try JSONDecoder().decode(WindowReference.self, from: Data(contentsOf: url))
    }

    /// Keys leaving a sliding layer's window. The real window is 1,024
    /// positions, which the reference cannot reach in reasonable time, so both
    /// run with a 16-position window: a token at a time the ring wraps at the
    /// 18th token, and in 32-token blocks inside the second block.
    ///
    /// What is compared is layer 0's attention output. Nothing before it is
    /// routed, so it differs from the reference only by rounding, and a key
    /// that is seen too long, dropped too soon or read from the wrong slot
    /// shows at once. Later layers inherit expert choices that rounding can
    /// flip, and their states are not comparable digit for digit.
    @Test func keysLeaveTheSlidingWindowLikeTheReference() async throws {
        guard let path = ProcessInfo.processInfo.environment["TUFF_INKLING_OUTPUT"] else { return }
        let reference = try Self.windowReference()
        let (context, model, logits) = try Self.loaded(path)
        let prompt = reference.input_ids.map(Int32.init)
        let heads = model.config.numHeads, headDim = model.config.headDim

        func check(_ rows: [Float], from position: Int, _ name: String, tolerance: Float) {
            var worst: Float = 0, worstAt = position
            for row in 0..<(rows.count / (heads * headDim)) {
                let expected = reference.attention[position + row]
                for head in 0..<heads {
                    let values = rows[((row * heads + head) * headDim)..<((row * heads + head + 1) * headDim)]
                    let norm = values.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
                    let projection = values.enumerated().reduce(Float(0)) {
                        $0 + ($1.offset % 3 == 0 ? $1.element : -$1.element)
                    }
                    let error = max(abs(norm - expected.norms[head]),
                                    abs(projection - expected.projections[head])) / expected.norms[head]
                    if error > worst { (worst, worstAt) = (error, position + row) }
                }
            }
            #expect(worst < tolerance, "\(name): \(worst) at position \(worstAt)")
        }

        let single = try InklingForwardRunner(
            model: model, context: context, maxContext: 128,
            runtimeConfiguration: RuntimeConfiguration(prefillEnabled: false, forceLogitsHead: true),
            slidingWindow: reference.window)
        var seen: [Float] = []
        single.afterAttention = { if $0 == 0 { seen = single.attentionRows(1) } }
        for (position, token) in prompt.enumerated() {
            try await single.produce(token: token, position: position, into: logits)
            check(seen, from: position, "single", tolerance: 0.004)
        }

        for batched in [false, true] {
            let configuration = RuntimeConfiguration(prefillChunkTokens: 32, forceLogitsHead: true)
            let blocks = try InklingForwardRunner(
                model: model, context: context, maxContext: 128, runtimeConfiguration: configuration,
                slidingWindow: reference.window, batchedProducts: batched)
            var position = 0
            while position < prompt.count {
                let end = min(prompt.count, position + 32)
                let count = end - position
                blocks.afterAttention = { if $0 == 0 { seen = blocks.attentionRows(count) } }
                _ = try await blocks.prefillChunked(
                    tokens: prompt[position..<end], startPosition: position, outputMode: .logits,
                    config: configuration.prefillConfig, into: logits, onProgress: { _ in })
                check(seen, from: position, batched ? "batched blocks" : "per-row blocks",
                      tolerance: batched ? 0.03 : 0.004)
                position = end
            }
        }
    }
}
