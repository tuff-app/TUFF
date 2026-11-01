import Foundation
import Metal
import Testing
@testable import TUFFEngine

/// Opt-in real-model check that Flint chunked prefill matches token-by-token
/// decode: every routed expert set identical, final logits equal.
/// `TUFF_FLINT_OUTPUT` is the pack, `TUFF_FLINT_TOOL_REFERENCE` a JSON file with
/// `input_ids`; `TUFF_FLINT_PREFILL_TOKENS` optionally truncates the prompt and
/// `TUFF_FLINT_PREFILL_CHUNK` sets the chunk size (default 32).
@Suite(.serialized) struct FlintBatchedPrefillTests {
    @Test func chunkedPrefillMatchesTokenByTokenDecode() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["TUFF_FLINT_OUTPUT"],
              let promptPath = env["TUFF_FLINT_TOOL_REFERENCE"] else { return }
        let object = try #require(JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: promptPath))) as? [String: Any])
        var tokens = try #require(object["input_ids"] as? [Int]).map { Int32($0) }
        if let limit = env["TUFF_FLINT_PREFILL_TOKENS"].flatMap(Int.init) { tokens = Array(tokens.prefix(limit)) }
        let chunk = env["TUFF_FLINT_PREFILL_CHUNK"].flatMap(Int.init) ?? 32
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: modelPath),
                                   device: context.device, expecting: .flint40B)
        let configuration = RuntimeConfiguration(prefillChunkTokens: chunk, forceLogitsHead: true)
        let vocabulary = model.config.vocabSize
        let routedLayers = model.config.numLayers - 2
        let topK = model.config.topKExperts
        func logitBuffer() throws -> MTLBuffer {
            try #require(context.device.makeBuffer(length: vocabulary * 2, options: .storageModeShared))
        }

        // Reference: one decode token at a time.
        let decodeRunner = try RealForwardRunner(model: model, context: context,
            maxContext: tokens.count + 8, runtimeConfiguration: configuration)
        var decodeRoutes = Array(repeating: UInt32.max, count: tokens.count * routedLayers * topK)
        var position = 0
        decodeRunner.debugRouterSink = { layer, experts in
            guard layer >= 2 else { return }
            let base = (position * routedLayers + layer - 2) * topK
            for (i, expert) in experts.enumerated() { decodeRoutes[base + i] = UInt32(expert) }
        }
        let decodeLogits = try logitBuffer()
        for (i, token) in tokens.enumerated() {
            position = i
            try await decodeRunner.produce(token: token, position: i, into: decodeLogits)
        }
        let expected = Array(UnsafeBufferPointer(
            start: decodeLogits.contents().assumingMemoryBound(to: Float16.self), count: vocabulary))

        // Candidate: chunked prefill.
        let prefillRunner = try RealForwardRunner(model: model, context: context,
            maxContext: tokens.count + 8, runtimeConfiguration: configuration)
        var prefillRoutes = Array(repeating: UInt32.max, count: tokens.count * routedLayers * topK)
        var routeSinkCalls = 0
        prefillRunner.debugFlintPrefillRouteSink = { layer, first, ids in
            routeSinkCalls += 1
            let rows = ids.count / topK
            for row in 0..<rows {
                let base = ((first + row) * routedLayers + layer - 2) * topK
                for k in 0..<topK { prefillRoutes[base + k] = ids[row * topK + k] }
            }
        }
        // The decode path reports the final token's routes through the same sink.
        var finalPosition = tokens.count - 1
        prefillRunner.debugRouterSink = { layer, experts in
            guard layer >= 2 else { return }
            let base = (finalPosition * routedLayers + layer - 2) * topK
            for (i, expert) in experts.enumerated() { prefillRoutes[base + i] = UInt32(expert) }
        }
        let prefillLogits = try logitBuffer()
        let started = Date()
        let result = try await prefillRunner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: chunk), into: prefillLogits) { _ in }
        let seconds = Date().timeIntervalSince(started)
        finalPosition = tokens.count - 1
        let actual = Array(UnsafeBufferPointer(
            start: prefillLogits.contents().assumingMemoryBound(to: Float16.self), count: vocabulary))

        var maximumDifference: Float = 0
        var differing = 0
        for i in 0..<vocabulary {
            let difference = abs(Float(expected[i]) - Float(actual[i]))
            if difference > 0 { differing += 1 }
            maximumDifference = max(maximumDifference, difference)
        }
        func argmax(_ values: [Float16]) -> Int { values.indices.max { Float(values[$0]) < Float(values[$1]) }! }
        // The last layer of a prefill chunk only writes K/V rows: nothing reads
        // its routing, so only the final token (decode path) reports it.
        var mismatchedRoutes = 0
        for position in 0..<tokens.count {
            for layer in 0..<routedLayers {
                if layer == routedLayers - 1, position < tokens.count - 1 { continue }
                let base = (position * routedLayers + layer) * topK
                for k in 0..<topK where decodeRoutes[base + k] != prefillRoutes[base + k] { mismatchedRoutes += 1 }
            }
        }
        print("Flint prefill equivalence: tokens=\(tokens.count) chunk=\(chunk) seconds=\(seconds) routeSinkCalls=\(routeSinkCalls) "
              + "maxLogitDifference=\(maximumDifference) differingLogits=\(differing) routeMismatches=\(mismatchedRoutes) "
              + "argmax decode=\(argmax(expected)) prefill=\(argmax(actual))")
        #expect(result.newPosition == tokens.count)
        #expect(result.seed == .logitsWritten)
        #expect(mismatchedRoutes == 0)
        #expect(argmax(expected) == argmax(actual))
        #expect(maximumDifference <= 0.05)
    }
}
