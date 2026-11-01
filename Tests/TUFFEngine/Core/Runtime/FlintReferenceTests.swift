import Foundation
import Metal
import Testing
import TUFFValidationSupport
@testable import TUFFEngine

@Suite(.serialized) struct FlintReferenceTests {
    @Test func pinnedSingleTokenLogits() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["TUFF_FLINT_OUTPUT"], let reference = env["TUFF_FLINT_REFERENCE"] else { return }
        let object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: reference))) as? [String: Any])
        let ids = try #require(object["input_ids"] as? [Int])
        let expectedIDs = try #require(object["top_ids"] as? [Int])
        let expectedLogits = try #require(object["top_logits"] as? [Double])
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device, expecting: .flint40B)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 128,
                                           runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        for (position, token) in ids.enumerated() {
            try await runner.produce(token: Int32(token), position: position, into: logits)
        }
        let actual = Fp16Buffer.read(logits, count: model.config.vocabSize)
        let top = try #require(actual.indices.max(by: { actual[$0] < actual[$1] }))
        print("Flint actual top=\(top) value=\(actual[top]); reference top=\(expectedIDs[0]) value=\(expectedLogits[0])")
        #expect(actual.allSatisfy { $0.isFinite })
        #expect(top == expectedIDs[0])
        for (index, expected) in zip(expectedIDs, expectedLogits) {
            #expect(abs(Double(actual[index]) - expected) < 0.5)
        }
    }
    @Test func pinnedChatGreedyReference() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["TUFF_FLINT_OUTPUT"], let reference = env["TUFF_FLINT_REFERENCE_CHAT"] else { return }
        let object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: reference))) as? [String: Any])
        let ids = try #require(object["input_ids"] as? [Int])
        let steps = try #require(object["steps"] as? [[String: Any]])
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device, expecting: .flint40B)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 128,
                                           runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        for (position, token) in ids.enumerated() {
            try await runner.produce(token: Int32(token), position: position, into: logits)
        }
        for (step, expectedStep) in steps.enumerated() {
            let expectedIDs = try #require(expectedStep["top_ids"] as? [Int])
            let expectedLogits = try #require(expectedStep["top_logits"] as? [Double])
            let actual = Fp16Buffer.read(logits, count: model.config.vocabSize)
            let top = try #require(actual.indices.max(by: { actual[$0] < actual[$1] }))
            print("Flint step \(step) actual top=\(top) value=\(actual[top]); reference top=\(expectedIDs[0]) value=\(expectedLogits[0])")
            #expect(actual.allSatisfy { $0.isFinite })
            #expect(top == expectedIDs[0])
            for (index, expected) in zip(expectedIDs, expectedLogits) {
                #expect(abs(Double(actual[index]) - expected) < 0.5)
            }
            guard step + 1 < steps.count else { break }
            try await runner.produce(token: Int32(expectedIDs[0]), position: ids.count + step, into: logits)
        }
    }

}
