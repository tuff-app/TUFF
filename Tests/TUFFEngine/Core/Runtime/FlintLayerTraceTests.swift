import Foundation
import Metal
import Testing
import TUFFValidationSupport
@testable import TUFFEngine

/// Diagnostic: dumps the native residual stream entering every layer (plus the
/// final one) for every prompt position as Float32 [position][layer 0...numLayers][hidden],
/// for comparison against a layer-by-layer publisher reference.
@Suite(.serialized) struct FlintLayerTraceTests {
    @Test func dumpsPerLayerResidualsForToolPrompt() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["TUFF_FLINT_OUTPUT"], let reference = env["TUFF_FLINT_TOOL_REFERENCE"],
              let out = env["TUFF_FLINT_TRACE_OUT"] else { return }
        let object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: reference))) as? [String: Any])
        let ids = try #require(object["input_ids"] as? [Int])
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device, expecting: .flint40B)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 1024,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        let hidden = model.config.hiddenSize
        var data = Data()
        data.reserveCapacity(ids.count * (model.config.numLayers + 1) * hidden * 4)
        runner.debugLayerSink = { _ in
            let values = Fp16Buffer.read(runner.debugResidual, count: hidden).map { Float($0) }
            values.withUnsafeBufferPointer { data.append(UnsafeBufferPointer(start: $0.baseAddress, count: $0.count)) }
        }
        var topIDs: [Int] = [], topValues: [Double] = [], secondValues: [Double] = []
        for (position, token) in ids.enumerated() {
            try await runner.produce(token: Int32(token), position: position, into: logits)
            let values = Fp16Buffer.read(logits, count: model.config.vocabSize)
            let top = values.indices.max(by: { values[$0] < values[$1] })!
            var second = -Float.infinity
            for i in values.indices where i != top { second = max(second, values[i]) }
            topIDs.append(top); topValues.append(Double(values[top])); secondValues.append(Double(second))
        }
        let summary: [String: Any] = ["position_top_ids": topIDs, "position_top_logits": topValues, "position_second_logits": secondValues]
        try JSONSerialization.data(withJSONObject: summary).write(to: URL(fileURLWithPath: out + ".json"))
        try data.write(to: URL(fileURLWithPath: out))
        print("Flint trace wrote \(ids.count) positions x \(model.config.numLayers + 1) layers x \(hidden) to \(out)")
    }
}
