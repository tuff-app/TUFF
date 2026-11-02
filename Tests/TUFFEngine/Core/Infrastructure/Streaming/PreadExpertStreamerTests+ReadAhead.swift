import Foundation
import Testing

@testable import TUFFEngine

extension PreadExpertStreamerTests {
  private func waitForReadAhead(_ streamer: PreadExpertStreamer, bytes: UInt64) async -> Bool {
    for _ in 0..<500 {
      if streamer.readAheadBytesRead >= bytes { return true }
      try? await Task.sleep(for: .milliseconds(5))
    }
    return streamer.readAheadBytesRead >= bytes
  }

  /// Experts already in a cache slot are not read again; the rest are, once.
  @Test func readAheadReadsOnlyExpertsOutsideTheCache() async throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device, slotCount: 2)
    _ = try streamer.loadExpertsCached(experts: [0])
    streamer.readAheadExpertMisses(experts: [0, 1, 2])
    let stride = UInt64(Self.expertStride)
    #expect(await waitForReadAhead(streamer, bytes: 2 * stride))
    try await Task.sleep(for: .milliseconds(50))
    #expect(streamer.readAheadBytesRead == 2 * stride)
    // Reading ahead never fills a cache slot or changes what a fetch returns.
    let fetched = try streamer.loadExpertsCached(experts: [1])
    #expect(fetched.count == 1)
  }

  /// A streamer that bypasses the file cache has nothing to warm.
  @Test func readAheadIsSkippedWhenTheFileCacheIsBypassed() async throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device, slotCount: 2,
      fileDescriptor: nil, bypassFilesystemCache: true)
    streamer.readAheadExpertMisses(experts: [0, 1, 2, 3])
    try await Task.sleep(for: .milliseconds(100))
    #expect(streamer.readAheadBytesRead == 0)
  }

  /// The streamer stays usable while reads are in flight and after it is
  /// released: workers keep it alive until they finish.
  @Test func readAheadOutlivesTheCallerSafely() async throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    var streamer: PreadExpertStreamer? = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device, slotCount: 1)
    weak let weakStreamer = streamer
    streamer?.readAheadExpertMisses(experts: [0, 1, 2, 3])
    streamer = nil
    for _ in 0..<500 where weakStreamer != nil { try await Task.sleep(for: .milliseconds(5)) }
    #expect(weakStreamer == nil)
  }
}
