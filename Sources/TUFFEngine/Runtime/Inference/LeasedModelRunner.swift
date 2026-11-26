import Metal

/// A session owns this output buffer exclusively until its forward returns.
private struct SessionOutputBuffer: @unchecked Sendable {
    let buffer: MTLBuffer
}

/// Borrowed only until the awaited token forward returns.
private struct SessionEpilogue: @unchecked Sendable {
    let body: (MTLCommandBuffer) -> Void
}

/// The chunk is borrowed only until the awaited gate returns.
private struct SessionPrefillChunk: @unchecked Sendable {
    let body: () async throws -> Void
}

/// A session-local runner guarded at token boundaries against other sessions
/// using the same model's mutable staging and expert caches. Each instance
/// must own a distinct underlying runner and scratch/KV state.
///
/// Prefill takes the lease one backend block at a time so other sessions
/// can run between blocks.
public final class LeasedModelRunner: ChunkedPrefillRunner, ContinuableLogitProducer,
    ContextWindowReporting, GreedyHeadReporting, PrefillHeadSkippingRunner,
    MultimodalPrefillRunner, EpilogueFusingLogitProducer,
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
    public func produce(token: Int32, position: Int, into logits: MTLBuffer,
                        appendingToFinalCommandBuffer epilogue: (MTLCommandBuffer) -> Void)
        async throws -> Bool {
        let output = SessionOutputBuffer(buffer: logits)
        return try await withoutActuallyEscaping(epilogue) { escaping in
            let borrowed = SessionEpilogue(body: escaping)
            return try await lease.withLease {
                try await self.runner.produce(token: token, position: position, into: output.buffer,
                    appendingToFinalCommandBuffer: borrowed.body)
            }
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

    /// Let the backend choose the blocks and write the final head once.
    func prefillChunked(tokens: ArraySlice<Int32>, startPosition: Int,
                        outputMode: PrefillOutputMode, config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        try await runner.prefillChunked(tokens: tokens, startPosition: startPosition,
            outputMode: outputMode, config: config, into: logits,
            chunkGate: withPrefillLease, onProgress: onProgress)
    }

    func prefillMultimodal(input: MultimodalPrefillInput, startPosition: Int,
                           outputMode: PrefillOutputMode, config: PrefillRuntimeConfig,
                           into logits: MTLBuffer,
                           onProgress: (Int) -> Void) async throws -> PrefillResult {
        try await runner.prefillMultimodal(input: input, startPosition: startPosition,
            outputMode: outputMode, config: config, into: logits,
            chunkGate: withPrefillLease, onProgress: onProgress)
    }

    private func withPrefillLease(_ chunk: () async throws -> Void) async throws {
        try await withoutActuallyEscaping(chunk) { escaping in
            let borrowed = SessionPrefillChunk(body: escaping)
            try await lease.withLease { try await borrowed.body() }
        }
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

extension LeasedModelRunner: PrefillPhaseReporting {
    public var prefillPhaseTimes: PrefillPhaseTimes { runner.prefillPhaseTimes }
}
