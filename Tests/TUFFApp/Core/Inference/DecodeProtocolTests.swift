import Foundation
import Testing
import TUFFEngine
import TUFFDecodeProtocol

@Suite struct DecodeProtocolTests {
    @Test func loadRequestRoundTripPreservesEveryPublicRuntimeOption() throws {
        let options = DecodeRuntimeOptions(
            expertCacheSlots: 32,
            expertCachePolicy: "lru",
            prefillEnabled: false,
            prefillChunkTokens: 64,
            rdadvisePolicy: "adaptive",
            modelVerification: "trusted-install")
        let request = DecodeLoadRequest(
            modelPath: "/tmp/model.gturbo",
            maxContextTokens: 8192,
            runtimeOptions: options,
            forceLogitsHead: true)

        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(request))
        try pipe.fileHandleForWriting.close()
        let decoded = try DecodeFrameCodec.read(
            DecodeLoadRequest.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.modelPath == request.modelPath)
        #expect(decoded.maxContextTokens == 8192)
        #expect(decoded.runtimeOptions == options)
        #expect(decoded.forceLogitsHead)
    }

    @Test func servingPreparationValidatesDecodedContextAndRuntimeOptions() throws {
        let valid = try ServingModelPreparation(
            modelID: "gemma-test", maxContextTokens: 4_096,
            runtimeOptions: DecodeRuntimeOptions(), forceLogitsHead: false)
        let encoded = try JSONEncoder().encode(valid)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var invalidContext = object
        invalidContext["maxContextTokens"] = 0
        let badContext = try JSONSerialization.data(withJSONObject: invalidContext)
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(ServingModelPreparation.self, from: badContext)
        }

        var invalidOptions = object
        var options = try #require(invalidOptions["runtimeOptions"] as? [String: Any])
        options["expertCacheSlots"] = -1
        invalidOptions["runtimeOptions"] = options
        let badOptions = try JSONSerialization.data(withJSONObject: invalidOptions)
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(ServingModelPreparation.self, from: badOptions)
        }
    }

    @Test func servingRequestRejectsMismatchedRequestAndGenerationIDsAfterDecode() throws {
        let generation = UUID()
        let payload = DecodeGenerationRequest(
            prompt: "hello", maxNewTokens: 8, maxContextTokens: 128,
            temperature: 0.2, generationID: generation)
        let request = try ServingGenerationRequest(
            modelID: "gemma-test", sessionID: UUID(), requestID: generation,
            payload: payload)
        let encoded = try JSONEncoder().encode(request)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var invalid = object
        invalid["requestID"] = UUID().uuidString
        let data = try JSONSerialization.data(withJSONObject: invalid)
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(ServingGenerationRequest.self, from: data)
        }
    }

    /// History carries images now, and the decode service is another process:
    /// an image that does not survive the wire is an image the model cannot see
    /// on the follow-up question.
    @Test func historyCarriesItsImagesAcrossTheWire() throws {
        let image = DecodeImageAttachment(
            id: UUID(),
            path: "/tmp/TUFF-Attachments/pid-1/a/photo.png",
            displayName: "photo.png",
            encodedBytes: 2_048,
            sha256: String(repeating: "c", count: 64))
        let request = DecodeGenerationRequest(
            prompt: "and the second line?",
            history: [
                DecodeChatTurn(prompt: "read this", response: "It says hello.",
                               images: [image]),
                DecodeChatTurn(prompt: "thanks", response: "Any time."),
            ],
            maxNewTokens: 128,
            maxContextTokens: 4_096,
            temperature: 0.2)

        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(
            DecodeGenerationRequest.self, from: data)

        #expect(decoded.history[0].images == [image])
        #expect(decoded.history[1].images == nil,
                "a text-only turn must not gain an empty image list")
    }

    /// A v2 client never sent the field. The service has to read that as "no
    /// images on this turn" rather than failing the whole request.
    @Test func aTurnWithoutTheImageFieldDecodesAsTextOnly() throws {
        let json = Data("""
        {"prompt":"hi","response":"hello"}
        """.utf8)

        let decoded = try JSONDecoder().decode(DecodeChatTurn.self, from: json)

        #expect(decoded.images == nil)
        #expect(decoded.prompt == "hi")
    }

    @Test func olderRunnerDiagnosticsRemainReadable() throws {
        let json = Data("""
        {"cb1MillisecondsPerToken":0,"ioMillisecondsPerToken":0,
         "cb2MillisecondsPerToken":0,"headMillisecondsPerToken":0,
         "rdadviseMillisecondsPerToken":0,"rdadviseCallsPerToken":0,
         "rdadviseMegabytesPerToken":0,"rdadviseSkippedPerToken":0,"rdadviseFailures":0}
        """.utf8)
        let runner = try JSONDecoder().decode(DecodeRunnerDiagnostics.self, from: json)
        #expect(runner.expertReads == nil)
        #expect(runner.exposedPrefetchWaitMillisecondsPerToken == nil)
    }

    @Test func terminalEventRoundTripPreservesDiagnosticsAndMemory() throws {
        var reads = ExpertReadMetrics()
        reads.demandReads = 2
        reads.demandBytes = 4096
        reads.prefetchReads = 3
        reads.prefetchBytes = 6144
        let runner = DecodeRunnerDiagnostics(
            cb1MillisecondsPerToken: 0.6,
            ioMillisecondsPerToken: 12,
            cb2MillisecondsPerToken: 0.4,
            headMillisecondsPerToken: 1.7,
            rdadviseMillisecondsPerToken: 0,
            rdadviseCallsPerToken: 0,
            rdadviseMegabytesPerToken: 0,
            rdadviseSkippedPerToken: 0,
            rdadviseFailures: 0,
            expertReads: reads, exposedPrefetchWaitMillisecondsPerToken: 1.2)
        let event = DecodeServiceEvent(
            kind: .finished,
            generationID: UUID(),
            tokenCount: 256,
            promptTokenCount: 1_017,
            prefillSeconds: 10.2,
            timeToFirstTokenSeconds: 0.04,
            decodeSeconds: 7.7,
            tokensPerSecond: 33.2,
            currentMemoryBytes: 2_000_000_000,
            peakMemoryBytes: 2_100_000_000,
            runner: runner)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(event))
        try pipe.fileHandleForWriting.close()
        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.tokenCount == 256)
        #expect(decoded.promptTokenCount == 1_017)
        #expect(decoded.currentMemoryBytes == 2_000_000_000)
        #expect(decoded.peakMemoryBytes == 2_100_000_000)
        #expect(decoded.runner == runner)
    }

    @Test func prefillEventRoundTripPreservesProgress() throws {
        let event = DecodeServiceEvent(
            kind: .prefill,
            generationID: UUID(),
            sequence: 7,
            prefillDone: 128,
            prefillTotal: 514)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(event))
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.kind == .prefill)
        #expect(decoded.sequence == 7)
        #expect(decoded.prefillDone == 128)
        #expect(decoded.prefillTotal == 514)
    }

    @Test func decoderAcceptsAFrameSplitAcrossSingleByteWrites() throws {
        let event = DecodeServiceEvent(
            kind: .snapshot,
            generationID: UUID(),
            sequence: 1,
            textDelta: "caf\u{00E9}",
            tokenCount: 1)
        let frame = try DecodeFrameCodec.encode(event)
        let pipe = Pipe()
        for byte in frame {
            try pipe.fileHandleForWriting.write(contentsOf: Data([byte]))
        }
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.sequence == 1)
        #expect(decoded.textDelta == "caf\u{00E9}")
    }

    @Test func oversizedPayloadIsRejectedBeforeEncoding() {
        let request = DecodeGenerationRequest(
            prompt: String(repeating: "x", count: DecodeFrameCodec.maximumPayloadBytes + 1),
            maxNewTokens: 1,
            maxContextTokens: 4_096,
            temperature: 0)

        #expect(throws: DecodeFrameError.self) {
            _ = try DecodeFrameCodec.encode(request)
        }
    }

    @Test func oversizedFrameIsRejectedBeforePayloadRead() throws {
        let pipe = Pipe()
        var count = UInt32(DecodeFrameCodec.maximumPayloadBytes + 1).littleEndian
        try pipe.fileHandleForWriting.write(contentsOf: withUnsafeBytes(of: &count) { Data($0) })
        try pipe.fileHandleForWriting.close()

        #expect(throws: DecodeFrameError.self) {
            _ = try DecodeFrameCodec.read(
                DecodeServiceEvent.self,
                from: pipe.fileHandleForReading)
        }
    }
    @Test func generationRequestRoundTripPreservesImageDescriptors() throws {
        let attachment = DecodeImageAttachment(
            id: UUID(),
            path: "/tmp/staged-image",
            displayName: "image.png",
            encodedBytes: 123,
            sha256: String(repeating: "b", count: 64))
        let request = DecodeGenerationRequest(
            prompt: "describe",
            history: [DecodeChatTurn(
                prompt: "earlier",
                response: "answer",
                thinking: "reasoning")],
            imageAttachments: [attachment],
            maxNewTokens: 16,
            maxContextTokens: 4096,
            reasoning: .on,
            temperature: 0)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(request))
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeGenerationRequest.self,
            from: pipe.fileHandleForReading)
        #expect(decoded.imageAttachments == [attachment])
        #expect(decoded.reasoning == .on)
        #expect(decoded.history.first?.thinking == "reasoning")
    }

    @Test func gptReasoningEffortRoundTrips() throws {
        let request = DecodeGenerationRequest(
            prompt: "solve",
            maxNewTokens: 16,
            maxContextTokens: 4_096,
            reasoningEffort: .high,
            temperature: 0)
        let encoded = try DecodeFrameCodec.encode(request)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(contentsOf: encoded)
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeGenerationRequest.self,
            from: pipe.fileHandleForReading)
        #expect(decoded.reasoningEffort == .high)
    }

    @Test func thinkingPreservationRoundTripsAndDefaultsOff() throws {
        let request = DecodeGenerationRequest(
            prompt: "continue",
            maxNewTokens: 16,
            maxContextTokens: 4_096,
            preserveThinking: true,
            temperature: 0)
        let encoded = try DecodeFrameCodec.encode(request)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(contentsOf: encoded)
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeGenerationRequest.self,
            from: pipe.fileHandleForReading)
        #expect(decoded.preserveThinking)

        var legacyObject = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request))
                as? [String: Any])
        legacyObject.removeValue(forKey: "preserveThinking")
        let legacy = try JSONDecoder().decode(
            DecodeGenerationRequest.self,
            from: JSONSerialization.data(withJSONObject: legacyObject))
        #expect(!legacy.preserveThinking)
    }

    @Test func seedRoundTripsAndRetiredServerKeysAreIgnored() throws {
        let request = DecodeGenerationRequest(
            prompt: "hello",
            maxNewTokens: 32,
            maxContextTokens: 8_192,
            temperature: 0.3,
            seed: 42)
        let encoded = try JSONEncoder().encode(request)
        #expect(try JSONDecoder().decode(
            DecodeGenerationRequest.self, from: encoded).seed == 42)

        // 7.0.0 clients and the release interface harness still send the keys
        // the app-hosted server used; the service ignores them.
        var older = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        older["structuredMessages"] = [["role": "user", "content": "x"]]
        older["tools"] = []
        older["stopStrings"] = ["END"]
        older["harmonyCurrentDate"] = "2026-08-25"
        let decoded = try JSONDecoder().decode(
            DecodeGenerationRequest.self,
            from: JSONSerialization.data(withJSONObject: older))
        #expect(decoded.prompt == "hello")
        #expect(decoded.seed == 42)
    }

}
