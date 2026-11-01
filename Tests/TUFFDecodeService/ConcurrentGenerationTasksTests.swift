import Foundation
import Testing
@testable import TUFFEngineAdapter

@Suite struct ConcurrentGenerationTasksTests {
    @Test func endingSessionBlocksReuseUntilProducerCleanup() async {
        let tasks = ConcurrentGenerationTasks()
        let firstSession = UUID(), secondSession = UUID()
        let first = UUID(), second = UUID()
        #expect(tasks.reserve(requestID: first, sessionID: firstSession))
        #expect(tasks.reserve(requestID: second, sessionID: secondSession))
        #expect(!tasks.reserve(requestID: UUID(), sessionID: firstSession))
        #expect(tasks.beginEnding(firstSession))
        tasks.clear(requestID: first)
        await tasks.waitUntilSessionIdle(firstSession)
        #expect(!tasks.reserve(requestID: UUID(), sessionID: firstSession))
        tasks.finishEnding(firstSession)
        let replacement = UUID()
        #expect(tasks.reserve(requestID: replacement, sessionID: firstSession))
        tasks.clear(requestID: second)
        tasks.clear(requestID: replacement)
        await tasks.waitUntilIdle()
    }
}
