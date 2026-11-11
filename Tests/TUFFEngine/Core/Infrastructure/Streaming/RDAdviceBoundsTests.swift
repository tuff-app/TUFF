import Darwin
import Testing
@testable import TUFFEngine

@Suite struct RDAdviceBoundsTests {
    @Test func mergesDuplicateAndOverlappingAdviceWithinBudget() {
        let ranges = PreadExpertStreamer.boundedAdviceRanges([
            (0, 100), (0, 100), (50, 100), (200, 100)], maximumBytes: 180)
        #expect(ranges.count == 2)
        #expect(ranges[0].offset == 0 && ranges[0].count == 150)
        #expect(ranges[1].offset == 200 && ranges[1].count == 30)
    }
    @Test func capsCallsAndRejectsOverflowingRanges() {
        let ranges = PreadExpertStreamer.boundedAdviceRanges([
            (UInt64.max, 10), (0, 1), (10, 1)], maximumCalls: 1)
        #expect(ranges.count == 1)
        #expect(ranges[0].offset == 0)
        #expect(PreadExpertStreamer.boundedAdviceRanges([(0, 1)], maximumBytes: 0).isEmpty)
    }
    @Test func emptyAndUnrepresentableAdviceNeverIssuesIO() {
        #expect(RDAdvice.call(fd: -1, offset: 0, byteCount: 0).succeeded)
        let invalid = RDAdvice.call(fd: -1, offset: UInt64.max, byteCount: 1)
        #expect(invalid.errnoValue == EINVAL)
        #expect(invalid.requestedBytes == 0)
    }
}
