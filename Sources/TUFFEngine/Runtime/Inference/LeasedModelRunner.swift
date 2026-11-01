import Metal

/// A session owns this output buffer exclusively until its forward returns.
private struct SessionOutputBuffer: @unchecked Sendable {
    let buffer: MTLBuffer
}

/// The chunk is borrowed only until the awaited gate returns.
private struct SessionPrefillChunk: @unchecked Sendable {
    let body: () async throws -> Void
}

/// A session-local runner guarded at token boundaries against other sessions
/// using the same model's mutable staging and expert caches. Each instance
/// must own a distinct underlying runner and scratch/KV state.
///
/// This initial path deliberately uses token-by-token text prefill, so a
/// long prompt cannot hold the shared lease for an entire generation.
public final class LeasedModelRunner: ContinuableLogitProducer,
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
