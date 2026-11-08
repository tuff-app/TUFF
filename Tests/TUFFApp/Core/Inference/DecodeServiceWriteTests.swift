import Darwin
import Foundation
import Testing
import TUFFDecodeProtocol
@testable import TUFFAppCore

/// Frames are written by generations, cancels and teardown on different
/// threads. Before 10.2.2 only some writes took the lock, so two frames could
/// interleave, and teardown could close the handle under a writer.
@Suite struct DecodeServiceWriteTests {
    @Test func concurrentWritersNeverInterleaveFrames() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            input: commands.fileHandleForWriting, output: responses.fileHandleForReading)
        let ids = (0..<64).map { _ in UUID() }
        let reader = Task.detached { () -> [UUID] in
            var seen: [UUID] = []
            for _ in ids {
                let command = try DecodeFrameCodec.read(
                    DecodeServiceCommand.self, from: commands.fileHandleForReading)
                guard case .cancelRequest(let id) = command else {
                    Issue.record("unexpected frame \(command)")
                    return seen
                }
                seen.append(id)
            }
            return seen
        }
        await withTaskGroup(of: Void.self) { group in
            for id in ids {
                group.addTask {
                    try? client.send(.cancelRequest(id), to: commands.fileHandleForWriting)
                }
            }
        }
        let seen = try await reader.value
        #expect(Set(seen) == Set(ids))
        try? responses.fileHandleForWriting.close()
    }

    @Test func aWriteToAReplacedConnectionFailsInsteadOfWriting() throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            input: commands.fileHandleForWriting, output: responses.fileHandleForReading)
        let stale = Pipe()
        #expect(throws: AppInferenceError.self) {
            try client.send(.cancel, to: stale.fileHandleForWriting)
        }
        try? responses.fileHandleForWriting.close()
    }

    /// A service that stopped reading leaves the socket buffer full. Teardown
    /// used to write the shutdown frame with a blocking write and never return.
    @Test func teardownReturnsWhenTheServiceStoppedReading() async throws {
        var pair: [Int32] = [0, 0]
        try #require(socketpair(AF_UNIX, SOCK_STREAM, 0, &pair) == 0)
        defer { close(pair[1]) }
        let local = pair[0]
        let flags = fcntl(local, F_GETFL)
        _ = fcntl(local, F_SETFL, flags | O_NONBLOCK)
        let filler = [UInt8](repeating: 0, count: 64 * 1024)
        while filler.withUnsafeBytes({ write(local, $0.baseAddress, $0.count) }) > 0 {}
        _ = fcntl(local, F_SETFL, flags)
        let responses = Pipe()
        let finished = Task.detached { () -> Bool in
            var client: DecodeServiceInferenceClient? = DecodeServiceInferenceClient(
                input: FileHandle(fileDescriptor: local, closeOnDealloc: false),
                output: responses.fileHandleForReading)
            _ = client
            client = nil
            return true
        }
        let deadline = ContinuousClock.now + .seconds(5)
        let result = await withTaskGroup(of: Bool.self) { group in
            group.addTask { await finished.value }
            group.addTask {
                try? await Task.sleep(until: deadline)
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
        #expect(result, "teardown blocked on a full socket")
        try? responses.fileHandleForWriting.close()
    }
}
