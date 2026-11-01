import Foundation
import Darwin
import Testing
import TUFFEngine
@testable import TUFFAppCore
@testable import TUFFDecodeServiceCore
import TUFFDecodeProtocol

@Suite struct DecodeServiceOutboxTests {
    @Test func socketWriterDoesNotChangeSharedReadFlags() throws {
        var descriptors: [Int32] = [0, 0]
        #expect(socketpair(AF_UNIX, SOCK_STREAM, 0, &descriptors) == 0)
        let output = FileHandle(fileDescriptor: descriptors[0], closeOnDealloc: true)
        let peer = FileHandle(fileDescriptor: descriptors[1], closeOnDealloc: true)
        let before = fcntl(descriptors[0], F_GETFL)
        let writer = DecodeServiceFrameWriter(handle: output, timeout: 1)
        try writer.write(Data([1, 2, 3]))
        let after = fcntl(descriptors[0], F_GETFL)
        #expect((before & O_NONBLOCK) == (after & O_NONBLOCK))
        #expect(try peer.read(upToCount: 3) == Data([1, 2, 3]))
    }

    @Test func blockedFrameWriterFailsAndRefusesFurtherFrames() throws {
        let pipe = Pipe()
        let writer = DecodeServiceFrameWriter(handle: pipe.fileHandleForWriting, timeout: 0.05)
        do {
            try writer.write(Data(repeating: 1, count: 1_048_576))
            Issue.record("A stalled reader accepted an oversized frame")
        } catch {}
        do {
            try writer.write(Data([1]))
            Issue.record("A failed transport accepted another frame")
        } catch {}
        try? pipe.fileHandleForWriting.close()
        try? pipe.fileHandleForReading.close()
    }

    @Test func oversizedPendingOutputCancelsAndFailsOnce() throws {
        let id = UUID()
        let cancelled = DispatchSemaphore(value: 0)
        let outbox = DecodeServiceOutbox(generationID: id, maximumPendingBytes: 4,
                                         onOverflow: { cancelled.signal() })
        outbox.publish(.token(AppTokenEvent(index: 0, textDelta: "too long", elapsedDecodeSeconds: 0)))
        #expect(cancelled.wait(timeout: .now() + 1) == .success)
        outbox.publish(.token(AppTokenEvent(index: 1, textDelta: "more", elapsedDecodeSeconds: 0)))
        #expect(cancelled.wait(timeout: .now()) == .timedOut)
        let pipe = Pipe()
        try outbox.runWriter(to: pipe.fileHandleForWriting)
        let event = try DecodeFrameCodec.read(DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(event.kind == .failed)
        #expect(event.error == "output consumer is too slow")
    }

    @Test func cancellationFollowedByThrownCancellationWritesOneTerminal() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)

        let event = try firstTerminal(
            from: outbox,
            published: .cancelled(diagnostics(stopReason: .cancelled)),
            finishError: AppInferenceError.cancelled)

        #expect(event.kind == .cancelled)
        #expect(event.generationID == generationID)
    }

    @Test func failureFollowedByThrownErrorWritesOneTerminal() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)

        let event = try firstTerminal(
            from: outbox,
            published: .failed(.unknown("first"), partial: nil),
            finishError: AppInferenceError.unknown("second"))

        #expect(event.kind == .failed)
        #expect(event.generationID == generationID)
        #expect(event.error == "first")
    }

    /// Image encoding produces no progress or tokens, so the outbox must emit
    /// memory-only events during that otherwise silent interval.
    @Test func aSilentGenerationStillReportsMemory() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        let first = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(first.kind == .memory)
        #expect(first.generationID == generationID)
        #expect(try #require(first.currentMemoryBytes) > 0)

        let second = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(second.kind == .memory)

        outbox.publish(.prefillProgress(done: 4, total: 8))
        var event = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        while event.kind == .memory {
            event = try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        }
        #expect(event.kind == .prefill)
        #expect(event.prefillDone == 4)
        #expect(event.currentMemoryBytes != nil)

        outbox.finish(error: AppInferenceError.cancelled)
        #expect(writerFinished.wait(timeout: .now() + 5) == .success)
    }

    @Test func liveEventsCarryTheImageTowerFigure() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(
            generationID: generationID,
            towerBytes: { 1_144_373_248 })
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        let idle = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(idle.kind == .memory)
        #expect(idle.visionTowerMappedBytes == 1_144_373_248)

        outbox.publish(.prefillProgress(done: 1, total: 2))
        var event = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        while event.kind == .memory {
            event = try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        }
        #expect(event.kind == .prefill)
        #expect(event.visionTowerMappedBytes == 1_144_373_248)

        outbox.finish(error: AppInferenceError.cancelled)
        #expect(writerFinished.wait(timeout: .now() + 5) == .success)
    }

    @Test func thinkingAndVisibleTextStayOnSeparateIPCChannels() throws {
        let generationID = UUID()
        let outbox = DecodeServiceOutbox(generationID: generationID)
        outbox.publish(.thinking(AppTokenEvent(
            index: 0, textDelta: "private reasoning", elapsedDecodeSeconds: 0.1)))
        outbox.publish(.token(AppTokenEvent(
            index: 1, textDelta: "visible answer", elapsedDecodeSeconds: 0.2)))
        outbox.publish(.finished(diagnostics(stopReason: .eos)))
        outbox.finish()

        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        let snapshot = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(snapshot.kind == .snapshot)
        #expect(snapshot.thinkingDelta == "private reasoning")
        #expect(snapshot.textDelta == "visible answer")
        let terminal = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(terminal.kind == .finished)
        #expect(writerFinished.wait(timeout: .now() + 2) == .success)
    }

    private func firstTerminal(
        from outbox: DecodeServiceOutbox,
        published event: AppInferenceEvent,
        finishError: Error
    ) throws -> DecodeServiceEvent {
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        outbox.publish(event)
        let terminal = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        outbox.finish(error: finishError)

        #expect(writerFinished.wait(timeout: .now() + 2) == .success)
        #expect(pipe.fileHandleForReading.readDataToEndOfFile().isEmpty)
        return terminal
    }

    private func diagnostics(stopReason: AppStopReason) -> AppDiagnostics {
        AppDiagnostics(
            generatedTokens: 0,
            stopReason: stopReason,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
    }
}
