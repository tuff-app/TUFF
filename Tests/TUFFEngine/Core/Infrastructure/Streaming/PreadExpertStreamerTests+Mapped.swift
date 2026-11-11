import Testing
import Foundation
import Darwin
import Metal
@testable import TUFFEngine

/// The mapped mode: experts bound in place from the mapped layer file, the
/// file cache standing in for the slot cache.
extension PreadExpertStreamerTests {
  static func makeMapped(_ url: URL, slots: Int = 2) throws -> PreadExpertStreamer {
    let device = try #require(MTLCreateSystemDefaultDevice())
    return try PreadExpertStreamer(layout: makeLayout(path: url.path), device: device, slotCount: slots,
                                   fileDescriptor: nil, mapExperts: true)
  }

  @Test func mappedBuffersHoldEachExpertsBytesWithoutSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try Self.makeMapped(url)

    #expect(streamer.mapsExperts)
    #expect(streamer.diagnosticSlotScratchBytes == 0)
    for expert in 0..<Self.numExperts {
      let views = try streamer.loadExpertsCached(experts: [expert])
      #expect(views.count == 1)
      #expect(views[0].size == UInt64(Self.expertStride))
      let bytes = Self.bytes(of: views[0].buffer, offset: views[0].offset, count: Self.expertStride)
      #expect(bytes.allSatisfy { $0 == Self.tagByte(expert) })
    }
  }

  @Test func mappedPlansNeverRunOutOfSlotsAndCarryNoSlotLifetime() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try Self.makeMapped(url, slots: 2)

    // Every slot "in flight" would leave a slot cache nothing to evict.
    let plan = try #require(streamer.planExpertsCachedIfPossible(experts: [2, 3], avoidingSlots: [0, 1]))
    #expect(!plan.usesSlots)
    #expect(plan.hits + plan.misses.count == 2)
    let buffers = try streamer.executeExpertCachePlan(plan)
    #expect(Self.bytes(of: buffers[1].buffer, offset: 0, count: 1) == [Self.tagByte(3)])
    // Once read, the experts are in the file cache and plan as hits.
    #expect(streamer.planExpertsCached(experts: [2, 3]).misses.isEmpty)
  }

  @Test func mappedReadsCountOncePerExpert() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try Self.makeMapped(url, slots: 4)
    // Drop the file from the cache so every expert is a miss.
    let fd = open(url.path, O_RDONLY)
    defer { close(fd) }
    _ = fcntl(fd, F_NOCACHE, 1)
    let plan = streamer.planExpertsCached(experts: [0, 1, 2, 3])
    _ = try streamer.executeExpertCachePlan(plan)
    let metrics = streamer.readMetrics
    #expect(metrics.demandReads == UInt64(plan.misses.count))
    #expect(metrics.demandBytes == UInt64(plan.misses.count * Self.expertStride))
  }

  @Test func anExpertOffAPageBoundaryKeepsTheSlotCache() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let layout = StreamLayout(path: url.path, streamOffset: Self.streamOffset + 16,
                              streamSize: Self.streamSize - UInt64(Self.pageSize),
                              expertsPerLayer: Self.numExperts - 1,
                              expertStride: UInt64(Self.expertStride))
    let streamer = try PreadExpertStreamer(layout: layout, device: device, slotCount: 2,
                                           fileDescriptor: nil, mapExperts: true)
    #expect(!streamer.mapsExperts)
    #expect(streamer.diagnosticSlotScratchBytes > 0)
  }

  @Test func readsThatSkipTheFileCacheKeepTheSlotCache() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let streamer = try PreadExpertStreamer(layout: Self.makeLayout(path: url.path), device: device,
                                           slotCount: 2, fileDescriptor: nil,
                                           bypassFilesystemCache: true, mapExperts: true)
    #expect(!streamer.mapsExperts)
  }
}
