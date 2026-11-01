import Foundation
import Metal
import Testing
@testable import TUFFEngine

/// Opt-in capture of the untruncated native logits and every routed expert set.
@Suite(.serialized) struct FlintFidelityCaptureTests {
    @Test func captureLogitsAndRoutes() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["TUFF_FLINT_OUTPUT"],
              let promptPath = env["TUFF_FLINT_TOOL_REFERENCE"],
              let output = env["TUFF_FLINT_FIDELITY_OUT"] else { return }
        let object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: promptPath))) as? [String: Any])
        let tokens = try #require(object["input_ids"] as? [Int])
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: modelPath), device: context.device, expecting: .flint40B)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: max(1024, tokens.count + 1),
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        let logitsURL = URL(fileURLWithPath: output + ".logits.f16")
        #expect(FileManager.default.createFile(atPath: logitsURL.path, contents: nil))
        let file = try FileHandle(forWritingTo: logitsURL)
        defer { try? file.close() }
        let routedLayers = model.config.numLayers - 2
        var routes = Array(repeating: UInt32.max, count: tokens.count * routedLayers * model.config.topKExperts)
        var position = 0
        runner.debugRouterSink = { layer, indices in
            guard layer >= 2 && layer < model.config.numLayers else { return }
            let base = (position * routedLayers + layer - 2) * model.config.topKExperts
            for (i, expert) in indices.enumerated() { routes[base + i] = UInt32(expert) }
        }
        for (i, token) in tokens.enumerated() {
            position = i
            try await runner.produce(token: Int32(token), position: i, into: logits)
            try file.write(contentsOf: Data(bytes: logits.contents(), count: model.config.vocabSize * 2))
        }
        try routes.withUnsafeBufferPointer { try Data(buffer: $0).write(to: URL(fileURLWithPath: output + ".routes.u32")) }
        print("Flint fidelity captured \(tokens.count) positions, vocabulary \(model.config.vocabSize), \(routedLayers) routed layers")
    }

    @Test func captureOperationBoundaries() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["TUFF_FLINT_OUTPUT"],
              let promptPath = env["TUFF_FLINT_TOOL_REFERENCE"],
              let output = env["TUFF_FLINT_OP_OUT"] else { return }
        let object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: promptPath))) as? [String: Any])
        let tokens = try #require(object["input_ids"] as? [Int])
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: modelPath), device: context.device, expecting: .flint40B)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 1024,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        var captures: [String: [Float]] = [:]
        var position = 0
        func halfValues(_ buffer: MTLBuffer, count: Int, offset: Int = 0) -> [Float] {
            let p = buffer.contents().advanced(by: offset).assumingMemoryBound(to: Float16.self)
            return (0..<count).map { Float(p[$0]) }
        }
        func target(_ layer: Int) -> Bool {
            (layer == 6 && position == 95) || (layer == 14 && position == 104)
        }
        runner.debugRouterSink = { layer, indices in
            guard target(layer) else { return }
            let prefix = "layer\(layer)_pos\(position)_"
            let width = model.config.hiddenSize
            captures[prefix + "q_rope"] = halfValues(runner.debugFlintQuery, count: model.config.numHeads * model.config.headDim)
            captures[prefix + "attention_context"] = halfValues(runner.debugFlintAttentionContext, count: model.config.numHeads * model.config.headDim)
            captures[prefix + "attention_out"] = halfValues(runner.debugAttentionOut, count: width)
            captures[prefix + "ffn_input"] = halfValues(runner.debugFlintFFNInput, count: width)
            captures[prefix + "router_indices"] = indices.map(Float.init)
            if let residual = runner.debugFlintResidual {
                let p = residual.contents().assumingMemoryBound(to: Float.self)
                captures[prefix + "attention_residual"] = Array(UnsafeBufferPointer(start: p, count: width))
            }
            var keys: [Float] = [], values: [Float] = []
            for t in 0...position {
                guard let slot = runner.debugKVSlot(layer: layer, position: t) else { return }
                let count = model.config.numKVHeads * model.config.headDim
                keys.append(contentsOf: halfValues(slot.key.buffer, count: count, offset: slot.key.offset))
                values.append(contentsOf: halfValues(slot.value.buffer, count: count, offset: slot.value.offset))
            }
            captures[prefix + "kv_keys"] = keys
            captures[prefix + "kv_values"] = values
        }
        runner.debugLayerSink = { layer in
            let previous = layer - 1
            guard target(previous) else { return }
            let prefix = "layer\(previous)_pos\(position)_"
            captures[prefix + "shared_output"] = halfValues(runner.debugSharedExpertOut, count: model.config.hiddenSize)
            captures[prefix + "combined_output"] = halfValues(runner.debugRoutedOut, count: model.config.hiddenSize)
        }
        for (i, token) in tokens.prefix(105).enumerated() {
            position = i
            try await runner.produce(token: Int32(token), position: i, into: logits)
        }
        try JSONSerialization.data(withJSONObject: captures).write(to: URL(fileURLWithPath: output))
        print("Flint operation capture wrote \(captures.count) buffers to \(output)")
    }
}
