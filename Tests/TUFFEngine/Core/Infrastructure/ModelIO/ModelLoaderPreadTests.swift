import Testing
import Foundation
import Metal
@testable import TUFFEngine

/// `Model.load(streamingMode:).withSlotCache()` integration for the bounded pread cache.
@Suite struct ModelLoaderPreadTests {

    @Test func cachedFetchOverloadsPreserveRequestedOrderAndBuffers() async throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .gemma4Toy(),
                                   streamingMode: .pread(slotCount: 3)).withSlotCache()
        let warmed = try await model.fetchRoutedExperts(layer: 1, experts: [4, 2])
        let hits = try await model.fetchRoutedExperts(layer: 1, experts: [2, 4])
        #expect(hits[0].buffer === warmed[1].buffer)
        #expect(hits[1].buffer === warmed[0].buffer)
        let plan = try #require(try model.planRoutedExperts(layer: 1, experts: [4, 2]))
        #expect(plan.misses.isEmpty)
        let plannedHits = try await model.fetchRoutedExperts(plan: plan)
        for index in warmed.indices {
            #expect(plannedHits[index].buffer === warmed[index].buffer)
            #expect(Self.readBytes(plannedHits[index]) == Self.readBytes(warmed[index]))
        }
        let mixed = try await model.fetchRoutedExperts(layer: 1, experts: [4, 5])
        #expect(mixed[0].buffer === warmed[0].buffer)
        #expect(Self.readBytes(mixed[1])[1] == 5)
        #expect(try await model.fetchRoutedExperts(layer: 1, experts: []).isEmpty)
    }

    private static func readBytes(_ view: TensorView) -> [UInt8] {
        let base = view.buffer.contents().advanced(by: Int(view.offset))
        return [UInt8](UnsafeRawBufferPointer(start: base, count: Int(view.length)))
    }

    @Test func loadsUnderPread_routedExpertBytesAndLazyOpen() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .gemma4Toy(),
                                   streamingMode: .pread(slotCount: 2)).withSlotCache()

        #expect(model.openLayerFileCount() == 0)
        let view = try model.routedExpert(layer: 1, expert: 4)
        #expect(model.openLayerFileCount() == 1)

        // pread returns the scratch slot: offset is 0, not exp.offset.
        #expect(view.offset == 0)

        // Same tagged-byte contract as ModelLoaderTests.routedExpertBytesRoundTrip.
        let b = Self.readBytes(view)
        #expect(b[0] == 1)       // layer 1
        #expect(b[1] == 4)       // expert 4
        #expect(b[2] == 0xC1)
        #expect(b[3] == 0xC2)
    }

    @Test func routedExpertCacheSlotCountDoesNotOpenLayerFile() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .gemma4Toy(),
                                   streamingMode: .pread(slotCount: 2)).withSlotCache()

        #expect(model.openLayerFileCount() == 0)
        #expect(model.routedExpertCacheSlotCount(layer: 1) == 2)
        #expect(model.openLayerFileCount() == 0)
    }

    @Test func routedExpertCacheSlotCountCapsAtExpertsPerLayer() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .gemma4Toy(),
                                   streamingMode: .pread(slotCount: 128)).withSlotCache()

        #expect(model.openLayerFileCount() == 0)
        #expect(model.routedExpertCacheSlotCount(layer: 1) == 16)
        #expect(model.openLayerFileCount() == 0)
    }

    @Test func beginOpeningRoutedExpertStreamerIsCompatibleWithLazyFetch() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .gemma4Toy(),
                                   streamingMode: .pread(slotCount: 2)).withSlotCache()

        model.beginOpeningRoutedExpertStreamer(layer: 1)
        let view = try model.routedExpert(layer: 1, expert: 4)

        #expect(model.openLayerFileCount() == 1)
        let b = Self.readBytes(view)
        #expect(b[0] == 1)
        #expect(b[1] == 4)
    }

    @Test func visionResidencyModesReleaseOrRetainExpertStreamers() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .gemma4Toy(),
                                   streamingMode: .pread(slotCount: 2)).withSlotCache()
        _ = try model.routedExpert(layer: 1, expert: 4)
        let before = model.expertReadMetrics
        #expect(before.demandReads == 1)
        let expectedScratch = UInt64(2 * Int(getpagesize()))

        let retained = model.prepareExpertResidencyForVision(.keepReady)
        #expect(retained.releasedLayerCount == 0)
        #expect(retained.releasedSlotScratchBytes == 0)
        #expect(retained.remainingOpenLayerCount == 1)
        #expect(retained.remainingSlotScratchBytes == expectedScratch)

        let released = model.prepareExpertResidencyForVision(
            .onDemand, gpuDrainNanoseconds: 123)
        #expect(released.gpuDrainNanoseconds == 123)
        #expect(released.releasedLayerCount == 1)
        #expect(released.releasedSlotScratchBytes == expectedScratch)
        #expect(released.remainingOpenLayerCount == 0)
        #expect(released.remainingSlotScratchBytes == 0)
        #expect(model.openLayerFileCount() == 0)
        #expect(model.expertReadMetrics == before)

        let view = try model.routedExpert(layer: 1, expert: 4)
        #expect(model.openLayerFileCount() == 1)
        #expect(Self.readBytes(view)[1] == 4)
        #expect(model.expertReadMetrics.demandReads == before.demandReads + 1)
    }

    @Test func prefetchHandleRetainsExpertStorageAfterDroppingTheModelValue() async throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        var model: Model? = try Model.load(directoryURL: dir, device: device,
          expecting: .gemma4Toy(), streamingMode: .pread(slotCount: 3)).withSlotCache()
        let plan = try #require(try model!.planRoutedExperts(layer: 1, experts: [4, 2], purpose: .prefetch))
        let operation = try model!.expertPrefetchOperation(plan: plan)
        let views = try model!.routedExpertBuffers(for: plan)
        model = nil
        try await Task.detached { try operation() }.value
        for (index, expert) in [4, 2].enumerated() {
            let bytes = Self.readBytes(views[index])
            #expect(bytes[0] == 1)
            #expect(bytes[1] == UInt8(expert))
            #expect(bytes[2] == 0xC1)
            #expect(bytes[3] == 0xC2)
        }
    }

    @Test func modelPrefetchPurposeReachesPlanningReadsAndDemandDiagnostics() async throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
          expecting: .gemma4Toy(), streamingMode: .pread(slotCount: 3)).withSlotCache()
        let plan = try #require(try model.planRoutedExperts(layer: 1, experts: [4, 2], purpose: .prefetch))
        try model.expertPrefetchOperation(plan: plan)()
        #expect(model.expertReadMetrics.prefetchReads == 2)
        #expect(model.expertReadMetrics.demandReads == 0)
        let demanded = try await model.fetchRoutedExperts(layer: 1, experts: [2, 4])
        #expect(Self.readBytes(demanded[0])[1] == 2)
        #expect(model.expertReadMetrics.usefulPrefetchReads == 2)
        #expect(model.expertReadMetrics.demandReads == 0)
    }

}
