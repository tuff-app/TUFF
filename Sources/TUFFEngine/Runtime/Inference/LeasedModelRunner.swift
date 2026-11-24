import Metal

/// A session owns this output buffer exclusively until its forward returns.
private struct SessionOutputBuffer: @unchecked Sendable {
    let buffer: MTLBuffer
}

/// Borrowed only until the awaited prefill chunk returns.
private struct SessionProgress: @unchecked Sendable {
    let body: (Int) -> Void
}

/// The chunk is borrowed only until the awaited gate returns.
private struct SessionPrefillChunk: @unchecked Sendable {
    let body: () async throws -> Void
}

/// A session-local runner guarded at token boundaries against other sessions
/// using the same model's mutable staging and expert caches. Each instance
/// must own a distinct underlying runner and scratch/KV state.
///
/// Text prefill takes the lease one prefill chunk at a time, so a long
/// prompt never holds the shared model for its whole prefill and another
/// session's forward can run between its chunks. Until 11.6 this path read
/// text prompts a token at a time for that reason, which made every API
/// prompt run at decode speed.
public final class LeasedModelRunner: ChunkedPrefillRunner, ContinuableLogitProducer,
    ContextWindowReporting, GreedyHeadReporting, PrefillHeadSkippingRunner,
    MultimodalPrefillRunner,
    StateSnapshottingRunner, PrefixCheckpointingRunner, @unchecked Sendable {
    private let runner: ModelForwardRunner
    private let lease: FairForwardLease

    public init(runner: ModelForwardRunner, lease: FairForwardLease) {
        self.runner = runner
        self.lease = lease
    }

    public func reset() { runner.reset() }
    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        let output = SessionOutputBuffer(buffer: logits)
        try await lease.withLease {
            try await self.runner.produce(token: token, position: position, into: output.buffer)
        }
    }
    var supportsHeadSkippingPrefill: Bool { runner.supportsHeadSkippingPrefill }
    func consumePrefillToken(_ token: Int32, position: Int, scratchLogits: MTLBuffer) async throws {
        let output = SessionOutputBuffer(buffer: scratchLogits)
        try await lease.withLease {
            try await self.runner.consumePrefillToken(token, position: position, scratchLogits: output.buffer)
        }
    }
    var supportsChunkedPrefill: Bool { runner.supportsChunkedPrefill }
    func prefillBlockTokens(_ config: PrefillRuntimeConfig) -> Int { runner.prefillBlockTokens(config) }

    /// One lease per block. The runner splits a prompt into blocks of
    /// `prefillBlockTokens` from its start position anyway, so the slices
    /// are the blocks one call would have run, in the same order. Flint is
    /// the exception: it runs the last token of every call through decode,
    /// so each slice's last token does: the same state by another path, not
    /// necessarily byte for byte. Each slice also runs the output head,
    /// which only the last one needs: about 3 ms a block on Gemma E2B,
    /// against hundreds for the block itself.
    func prefillChunked(tokens: ArraySlice<Int32>, startPosition: Int,
                        outputMode: PrefillOutputMode, config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        let output = SessionOutputBuffer(buffer: logits)
        let step = max(1, runner.prefillBlockTokens(config))
        var result: PrefillResult?
        var start = tokens.startIndex
        while start < tokens.endIndex {
            let end = min(start + step, tokens.endIndex)
            let slice = tokens[start..<end]
            let position = startPosition + (start - tokens.startIndex)
            let done = start - tokens.startIndex
            result = try await withoutActuallyEscaping(onProgress) { progress in
                let report = SessionProgress(body: progress)
                return try await lease.withLease {
                    try await self.runner.prefillChunked(
                        tokens: slice, startPosition: position, outputMode: outputMode,
                        config: config, into: output.buffer) { report.body(done + $0) }
                }
            }
            start = end
        }
        guard let result else {
            throw PrefillError.chunkedUnsupported("an empty prompt has nothing to prefill")
        }
        return result
    }

    /// The lease covers one planned chunk (a text span or one image span) at
    /// a time. The vision runtime's chunks are independent forwards over this
    /// runner's private KV and scratch, so another session's forward can run
    /// between them and a long image prompt never holds the shared model for
    /// its whole prefill.
    func prefillMultimodal(input: MultimodalPrefillInput, startPosition: Int,
                           outputMode: PrefillOutputMode, config: PrefillRuntimeConfig,
                           into logits: MTLBuffer,
                           onProgress: (Int) -> Void) async throws -> PrefillResult {
        let output = SessionOutputBuffer(buffer: logits)
        let lease = self.lease
        let gate: PrefillChunkGate = { chunk in
            try await withoutActuallyEscaping(chunk) { escaping in
                let borrowed = SessionPrefillChunk(body: escaping)
                try await lease.withLease { try await borrowed.body() }
            }
        }
        return try await runner.prefillMultimodal(input: input, startPosition: startPosition,
            outputMode: outputMode, config: config, into: output.buffer,
            chunkGate: gate, onProgress: onProgress)
    }
    public var continuationPosition: Int { runner.continuationPosition }
    public var maxContext: Int { runner.maxContext }
    public func prepareForContinuation(expectedPosition: Int) throws {
        try runner.prepareForContinuation(expectedPosition: expectedPosition)
    }
    var usesFusedGreedyHead: Bool { runner.usesFusedGreedyHead }
    var lastGreedyToken: UInt32 { runner.lastGreedyToken }
    func selectHead(pureGreedy: Bool) { runner.selectHead(pureGreedy: pureGreedy) }
    public var snapshotOwner: ObjectIdentifier { runner.snapshotOwner }
    public var stateSnapshotByteEstimate: Int? { runner.stateSnapshotByteEstimate }
    public func captureState() throws -> RunnerStateSnapshot { try runner.captureState() }
    public func restoreState(_ snapshot: RunnerStateSnapshot) throws { try runner.restoreState(snapshot) }
    public var supportsPrefixCheckpoints: Bool { runner.supportsPrefixCheckpoints }
    public var prefixCheckpointByteEstimate: Int { runner.prefixCheckpointByteEstimate }
    public func capturePrefixCheckpoint() throws -> RunnerStateSnapshot { try runner.capturePrefixCheckpoint() }
    public func rewind(to checkpoint: RunnerStateSnapshot) throws { try runner.rewind(to: checkpoint) }
}
