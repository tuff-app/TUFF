import Foundation
import Metal
import Testing
import TUFFValidationSupport
@testable import TUFFEngine

@Suite(.serialized) struct PrismReferenceTests {
    @Test func pinnedTwoBGreedyReference() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["TUFF_PRISM_2B_OUTPUT"], let reference = env["TUFF_PRISM_2B_REFERENCE"] else { return }
        try await compare(path: path, reference: reference, architecture: .bonsai2_2B)
    }

    @Test func pinned27BGreedyReference() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["TUFF_PRISM_27B_OUTPUT"], let reference = env["TUFF_PRISM_27B_REFERENCE"] else { return }
        try await compare(path: path, reference: reference, architecture: .bonsai2_27B)
    }

    private func compare(path: String, reference: String, architecture: ArchConfig) async throws {
        let raw = try Data(contentsOf: URL(fileURLWithPath: reference))
        let object = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        let ids = try #require(object["input_ids"] as? [Int])
        let steps = try #require(object["steps"] as? [[String: Any]])
        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device,
                                   expecting: architecture)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 128,
                                           runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        let chunkRunner = try RealForwardRunner(model: model, context: context, maxContext: 128,
                                                runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let chunkLogits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        #expect(chunkRunner.supportsChunkedPrefill)
        let tokens = ids.map(Int32.init)
        var start = 0
        while start < tokens.count {
            let end = min(start + 4, tokens.count)
            for position in start..<end {
                try await runner.produce(token: tokens[position], position: position, into: logits)
            }
            _ = try await chunkRunner.prefillChunked(tokens: tokens[start..<end], startPosition: start,
                outputMode: .logits, config: .production(chunkTokens: 32), into: chunkLogits, onProgress: { _ in })
            let serial = Fp16Buffer.read(logits, count: model.config.vocabSize)
            let chunk = Fp16Buffer.read(chunkLogits, count: model.config.vocabSize)
            #expect(chunk.allSatisfy { $0.isFinite })
            #expect(serial.indices.max(by: { serial[$0] < serial[$1] }) ==
                    chunk.indices.max(by: { chunk[$0] < chunk[$1] }))
            for index in serial.indices {
                #expect(abs(Float(serial[index]) - Float(chunk[index])) < 0.25)
            }
            start = end
        }
        #expect(runner.supportsHeadSkippingPrefill)
        for (step, referenceStep) in steps.enumerated() {
            let expectedIDs = try #require(referenceStep["top_ids"] as? [Int])
            let expectedLogits = try #require(referenceStep["top_logits"] as? [Double])
            let actual = Fp16Buffer.read(logits, count: model.config.vocabSize)
            let top = try #require(actual.indices.max(by: { actual[$0] < actual[$1] }))
            print("Prism step \(step) actual top=\(top) value=\(actual[top]); reference top=\(expectedIDs[0]) value=\(expectedLogits[0])")
            #expect(actual.allSatisfy { $0.isFinite })
            #expect(top == expectedIDs[0])
            for (index, expected) in zip(expectedIDs, expectedLogits) {
                #expect(abs(Double(actual[index]) - expected) < 0.25)
            }
            guard step + 1 < steps.count else { break }
            try await runner.produce(token: Int32(expectedIDs[0]), position: ids.count + step, into: logits)
            if step == 0 {
                try await chunkRunner.produce(token: Int32(expectedIDs[0]), position: ids.count, into: chunkLogits)
                let serial = Fp16Buffer.read(logits, count: model.config.vocabSize)
                let chunk = Fp16Buffer.read(chunkLogits, count: model.config.vocabSize)
                #expect(serial.indices.max(by: { serial[$0] < serial[$1] }) ==
                        chunk.indices.max(by: { chunk[$0] < chunk[$1] }))
                for index in serial.indices {
                    #expect(abs(Float(serial[index]) - Float(chunk[index])) < 0.25)
                }
            }
        }
    }
}
