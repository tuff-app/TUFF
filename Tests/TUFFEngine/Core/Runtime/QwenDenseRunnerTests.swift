import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFModelMetadata

@Suite(.serialized) struct QwenDenseRunnerTests {
    @Test func releasingPriorLayerExpertsPreservesViewsVerificationAndReadMetrics() async throws {
        let directory = try QwenToySynthetic.write(dense: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            expecting: .qwen36Toy(dense: false), streamingMode: .pread(slotCount: 8),
            denseResidency: .bounded(residentLayers: [])).withSlotCache()
        let first = try await model.fetchRoutedExperts(layer: 0, experts: [0])
        let view = try #require(first.first)
        let bytes = Data(bytes: view.buffer.contents(), count: view.buffer.length)
        _ = try await model.fetchRoutedExperts(layer: 1, experts: [0])
        let allocated = model.expertCacheAllocatedBytes
        let metrics = model.expertReadMetrics
        model.releaseExpertResidency(except: 1)
        #expect(model.openLayerFileCount() == 1)
        #expect(model.expertCacheAllocatedBytes < allocated)
        #expect(model.expertReadMetrics == metrics)
        #expect(model.streamersBox.layerVerified[0])
        #expect(Data(bytes: view.buffer.contents(), count: view.buffer.length) == bytes)
        let reopened = try await model.fetchRoutedExperts(layer: 0, experts: [0])
        let next = try #require(reopened.first)
        #expect(Data(bytes: next.buffer.contents(), count: next.buffer.length) == bytes)
    }

    @Test(arguments: [true, false]) func qwenProjectionResidencyPreservesDecodeAndPrefill(dense: Bool) async throws {
        let directory = try QwenToySynthetic.write(dense: dense)
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let output = try #require(context.device.makeBuffer(length: 1024 * 2, options: .storageModeShared))
        func values() -> [Float16] {
            Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float16.self), count: 1024))
        }
        let prompt: [Int32] = [7, 19, 4, 10]
        var reference: [[Float16]] = []
        var referencePrefill: [Float16] = []
        for policy: DenseResidencyPolicy in [.resident, .bounded(residentLayers: []), .bounded(residentLayers: [1, 3])] {
            let model = try Model.load(directoryURL: directory, device: context.device,
                expecting: .qwen36Toy(dense: dense), denseResidency: policy).withSlotCache()
            let runner = try RealForwardRunner(model: model, context: context, maxContext: 64,
                runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
            for (position, token) in prompt.enumerated() {
                try await runner.produce(token: token, position: position, into: output)
                let current = values()
                #expect(current.allSatisfy { $0.isFinite })
                if policy == .resident { reference.append(current) }
                else { #expect(current == reference[position]) }
            }
            if dense { #expect(model.openLayerFileCount() == 0) }
            runner.reset()
            _ = try await runner.prefillChunked(tokens: prompt[...], startPosition: 0,
                outputMode: .logits, config: .production(chunkTokens: 32), into: output,
                onProgress: { _ in })
            if policy == .resident { referencePrefill = values() }
            else { #expect(values() == referencePrefill) }
            #expect(values().allSatisfy { $0.isFinite })
        }
    }

    @Test func installedLayoutSummarizesActualPackedWeights() throws {
        let directory = try DenseGemmaToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let summary = try ModelWeightLayoutSummary.read(directory: directory, expecting: .gemma4E4BToy())
        #expect(summary.layout.denseGroupBytes.count == 4)
        #expect(summary.layout.fixedResidentBytes > 0)
        #expect(summary.minimumDenseStagingBytes > 0)
        #expect(summary.minimumDenseStagingBytes < summary.packedResidentPayloadBytes)
        #expect(summary.manifestSHA256.count == 64)
        #expect(summary.layout.expertLayers.isEmpty)
    }
}
