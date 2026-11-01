import Foundation
import Metal
import Testing
import TUFFInferenceTypes
import TUFFValidationSupport
@testable import TUFFEngine

/// Real-model opt-in smoke: generated calls must pass the app's structured decoder.
@Suite(.serialized) struct NewModelToolSmokeTests {
    @Test func flint() async throws { try await smoke("TUFF_FLINT_OUTPUT", architecture: .flint40B) }
    @Test func bonsai2B() async throws { try await smoke("TUFF_PRISM_2B_OUTPUT", architecture: .bonsai2_2B) }
    @Test func bonsai27B() async throws { try await smoke("TUFF_PRISM_27B_OUTPUT", architecture: .bonsai2_27B) }

    private func smoke(_ variable: String, architecture: ArchConfig) async throws {
        guard let path = ProcessInfo.processInfo.environment[variable] else { return }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device, expecting: architecture)
        let tokenizer = try await GFTokenizer.load(from: URL(fileURLWithPath: path).appendingPathComponent("tokenizer"))
        let tools: [GFTokenizer.FunctionDefinition] = [.init(name: "add", description: "Add two integers.",
            parameters: .object(["type": .string("object"), "properties": .object([
                "a": .object(["type": .string("integer")]), "b": .object(["type": .string("integer")])]),
                "required": .array([.string("a"), .string("b")])]))]
        let tokens = try tokenizer.encodeToolChat(messages: [
            .init(role: .system, content: "Call the provided tool directly without explaining."),
            .init(role: .user, content: "Use add to calculate 2 plus 2.")], tools: tools, reasoning: .off)
        print("\(variable) prompt ids=\(tokens); rendered=\(tokenizer.decode(tokens, skipSpecialTokens: false))")
        let runner = try RealForwardRunner(model: model, context: context, maxContext: max(1024, tokens.count + 128),
                                           runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        let prefillStart = Date()
        for (position, token) in tokens.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
            let values = Fp16Buffer.read(logits, count: model.config.vocabSize)
            guard values.allSatisfy({ $0.isFinite }) else {
                Issue.record("Non-finite logits after prefill position \(position), token \(token)")
                return
            }
        }
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        let decoder = StructuredAssistantDecoder(tokenizer: tokenizer, allowedTools: ["add"])
        var detokenizer = GFDetokenizer(tokenizer: tokenizer, barrierTokenIDs: tokenizer.structuralMarkerIDs)
        var generated: [Int32] = []
        var calls: [ParsedToolCall] = []
        let decodeStart = Date()
        for step in 0..<96 {
            let values = Fp16Buffer.read(logits, count: model.config.vocabSize)
            guard values.allSatisfy({ $0.isFinite }) else {
                Issue.record("Non-finite logits at generated step \(step)")
                return
            }
            let top = try #require(values.indices.max(by: { values[$0] < values[$1] }))
            let token = Int32(top)
            generated.append(token)
            let events = try decoder.consume(tokenID: token, delta: detokenizer.push(token))
            for event in events { if case .toolCall(let call) = event { calls.append(call) } }
            if !calls.isEmpty || tokenizer.stopTokenIDs.contains(token) { break }
            try await runner.produce(token: token, position: tokens.count + step, into: logits)
        }
        let elapsed = Date().timeIntervalSince(decodeStart)
        print("\(variable): serial full-logits smoke prefill \(tokens.count) tokens in \(prefillSeconds)s; decoded \(generated.count) tokens in \(elapsed)s; generated=\(tokenizer.decode(generated, skipSpecialTokens: false))")
        #expect(calls.count == 1)
        if let call = calls.first {
            #expect(call.name == "add")
            #expect(call.arguments == .object(["a": .integer(2), "b": .integer(2)]))
        }
    }
}
