import Metal
import TUFFModelCatalog

/// Family-neutral production runner used by the CLI, app, and server. The
/// existing Gemma/Qwen implementation remains untouched; GPT-OSS selects its
/// BF16/MXFP4 layer graph after the model has passed normal load validation.
public final class ModelForwardRunner: ChunkedPrefillRunner,
    MultimodalPrefillRunner, ContextWindowReporting, EpilogueFusingLogitProducer,
    ContinuableLogitProducer, SpeculativeVerificationRunner, PrefillHeadSkippingRunner,
    SpeculativeTargetCostReporting, GreedyHeadReporting,
    @unchecked Sendable {
    private enum Backend {
        #if !TUFF_ENGINE_GPT_OSS
        case affine(RealForwardRunner)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case gptOss(GPTOSSForwardRunner)
        #endif
        case inkling(InklingForwardRunner)
    }

    private let backend: Backend
    public let memoryPlan: InferenceMemoryPlan?

    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        memoryPlan = TUFFModelCatalog.all.first {
            $0.architecture.id.rawValue == model.config.variant.rawValue
        }.map {
            InferenceMemoryPlan(descriptor: $0, config: model.config,
                contextTokens: maxContext, expertCacheSlots: runtimeConfiguration.expertCacheSlots,
                prefillChunkTokens: runtimeConfiguration.prefillChunkTokens)
        }
        #if TUFF_ENGINE_GPT_OSS
        backend = .gptOss(try GPTOSSForwardRunner(model: model, context: context,
            maxContext: maxContext, runtimeConfiguration: runtimeConfiguration))
        #elseif TUFF_ENGINE_AFFINE_ONLY
        if model.config.family == .inkling2 {
            backend = .inkling(try InklingForwardRunner(model: model, context: context,
                maxContext: maxContext, runtimeConfiguration: runtimeConfiguration))
        } else {
            backend = .affine(try RealForwardRunner(model: model, context: context,
                maxContext: maxContext, runtimeConfiguration: runtimeConfiguration))
        }
        #else
        if model.config.family == .inkling2 {
            backend = .inkling(try InklingForwardRunner(model: model, context: context,
                maxContext: maxContext, runtimeConfiguration: runtimeConfiguration))
        } else if model.config.family == .gptOss {
            backend = .gptOss(try GPTOSSForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration))
        } else {
            backend = .affine(try RealForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration))
        }
        #endif
    }

    public func reset() {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): runner.reset()
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): runner.reset()
        #endif
        case .inkling(let runner): runner.reset()
        }
    }

    /// Decode expert lookahead counters, or nil for a runner without it.
    public var expertLookahead: (precision: Double, readsIssued: Int, enabled: Bool)? {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner):
            return (runner.lookaheadPrecision, runner.lookaheadReadsIssued,
                    runner.lookaheadEnabled)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner):
            return (runner.lookaheadPrecision, runner.lookaheadReadsIssued,
                    runner.lookaheadEnabled)
        #endif
        case .inkling: return nil
        }
    }

    public var exposedPrefetchWaitNanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.exposedPrefetchWaitNanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.exposedPrefetchWaitNanos
        #endif
        case .inkling: return 0
        }
    }

    public var expertReadMetrics: ExpertReadMetrics {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.expertReadMetrics
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.expertReadMetrics
        #endif
        case .inkling(let runner): return runner.expertReadMetrics
        }
    }

    public func produce(token: Int32, position: Int,
                        into logits: MTLBuffer) async throws {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner):
            try await runner.produce(token: token, position: position, into: logits)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner):
            try await runner.produce(token: token, position: position, into: logits)
        #endif
        case .inkling(let runner):
            try await runner.produce(token: token, position: position, into: logits)
        }
    }

    var supportsHeadSkippingPrefill: Bool {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): runner.supportsHeadSkippingPrefill
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: false
        #endif
        case .inkling: false
        }
    }

    func consumePrefillToken(_ token: Int32, position: Int, scratchLogits: MTLBuffer) async throws {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner):
            try await runner.consumePrefillToken(token, position: position, scratchLogits: scratchLogits)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss:
            preconditionFailure("GPT-OSS does not support this prefill path")
        #endif
        case .inkling:
            preconditionFailure("Inkling does not support this prefill path")
        }
    }

    public var supportsSpeculativeVerification: Bool {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.supportsSpeculativeVerification
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.supportsSpeculativeVerification
        #endif
        case .inkling: return false
        }
    }

    public func verifySpeculativeBlock(tokens: [Int32],
                                       startPosition: Int,
                                       into logits: MTLBuffer) async throws
        -> SpeculativeVerificationResult {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner):
            return try await runner.verifySpeculativeBlock(
                tokens: tokens, startPosition: startPosition, into: logits)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner):
            return try await runner.verifySpeculativeBlock(
                tokens: tokens, startPosition: startPosition, into: logits)
        #endif
        case .inkling:
            throw PrefillError.chunkedUnsupported("Inkling does not verify speculative blocks")
        }
    }

    public func commitSpeculativePrefix(_ count: Int) throws {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): try runner.commitSpeculativePrefix(count)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): try runner.commitSpeculativePrefix(count)
        #endif
        case .inkling:
            throw PrefillError.chunkedUnsupported("Inkling does not verify speculative blocks")
        }
    }

    public func speculativeBoundaryToken() async throws -> Int32? {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner):
            return try await runner.speculativeBoundaryToken()
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner):
            return try await runner.speculativeBoundaryToken()
        #endif
        case .inkling: return nil
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer,
                        appendingToFinalCommandBuffer epilogue: (MTLCommandBuffer) -> Void)
        async throws -> Bool {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner):
            return try await runner.produce(token: token, position: position,
                                            into: logits,
                                            appendingToFinalCommandBuffer: epilogue)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner):
            try await runner.produce(token: token, position: position, into: logits)
            return false
        #endif
        case .inkling(let runner):
            try await runner.produce(token: token, position: position, into: logits)
            return false
        }
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner):
            try runner.prepareForContinuation(expectedPosition: expectedPosition)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner):
            try runner.prepareForContinuation(expectedPosition: expectedPosition)
        #endif
        case .inkling(let runner):
            try runner.prepareForContinuation(expectedPosition: expectedPosition)
        }
    }

    public var continuationPosition: Int {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.continuationPosition
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.continuationPosition
        #endif
        case .inkling(let runner): return runner.continuationPosition
        }
    }

    public var maxContext: Int {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.maxContext
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.maxContext
        #endif
        case .inkling(let runner): return runner.maxContext
        }
    }

    var supportsChunkedPrefill: Bool {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): runner.supportsChunkedPrefill
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: true
        #endif
        case .inkling: true
        }
    }

    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner):
            return try await runner.prefillChunked(
                tokens: tokens,
                startPosition: startPosition,
                outputMode: outputMode,
                config: config,
                into: logits,
                onProgress: onProgress)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner):
            return try await runner.prefillChunked(
                tokens: tokens,
                startPosition: startPosition,
                outputMode: outputMode,
                config: config,
                into: logits,
                onProgress: onProgress)
        #endif
        case .inkling(let runner):
            return try await runner.prefillChunked(
                tokens: tokens,
                startPosition: startPosition,
                outputMode: outputMode,
                config: config,
                into: logits,
                onProgress: onProgress)
        }
    }

    func prefillMultimodal(input: MultimodalPrefillInput,
                           startPosition: Int,
                           outputMode: PrefillOutputMode,
                           config: PrefillRuntimeConfig,
                           into logits: MTLBuffer,
                           onProgress: (Int) -> Void) async throws -> PrefillResult {
        try await prefillMultimodal(
            input: input, startPosition: startPosition, outputMode: outputMode,
            config: config, into: logits, chunkGate: nil, onProgress: onProgress)
    }

    func prefillMultimodal(input: MultimodalPrefillInput,
                           startPosition: Int,
                           outputMode: PrefillOutputMode,
                           config: PrefillRuntimeConfig,
                           into logits: MTLBuffer,
                           chunkGate: PrefillChunkGate?,
                           onProgress: (Int) -> Void) async throws -> PrefillResult {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner):
            return try await runner.prefillMultimodal(
                input: input,
                startPosition: startPosition,
                outputMode: outputMode,
                config: config,
                into: logits,
                chunkGate: chunkGate,
                onProgress: onProgress)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss:
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS v2 does not support image input")
        #endif
        case .inkling:
            throw PrefillError.chunkedUnsupported(
                "Inkling image input is not supported yet")
        }
    }

    var usesFusedGreedyHead: Bool {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.usesFusedGreedyHead
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return false
        #endif
        case .inkling: return false
        }
    }

    var lastGreedyToken: UInt32 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.lastGreedyToken
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }

    var gptOssSplitExpertLayers: UInt64 {
        #if !TUFF_ENGINE_AFFINE_ONLY
        if case .gptOss(let runner) = backend { return runner.splitExpertLayers }
        #endif
        return 0
    }

    func selectHead(pureGreedy: Bool) {
        #if !TUFF_ENGINE_GPT_OSS
        if case .affine(let runner) = backend { runner.selectHead(pureGreedy: pureGreedy) }
        #endif
    }

    public var totalIoNanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalIoNanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.totalIoNanos
        #endif
        case .inkling(let runner): return runner.totalIoNanos
        }
    }
    public var totalCb1Nanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalCb1Nanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.totalCb1Nanos
        #endif
        case .inkling(let runner): return runner.totalCb1Nanos
        }
    }
    public var totalCb2Nanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalCb2Nanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.totalCb2Nanos
        #endif
        case .inkling(let runner): return runner.totalCb2Nanos
        }
    }
    public var totalHeadNanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalHeadNanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.totalHeadNanos
        #endif
        case .inkling(let runner): return runner.totalHeadNanos
        }
    }
    public var totalRoutedExpertReads: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalRoutedExpertReads
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.totalRoutedExpertReads
        #endif
        case .inkling(let runner): return runner.totalRoutedExpertReads
        }
    }
    public var totalRoutedExpertBytes: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalRoutedExpertBytes
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.totalRoutedExpertBytes
        #endif
        case .inkling(let runner): return runner.totalRoutedExpertBytes
        }
    }
    public var totalRoutedExpertCacheHits: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalRoutedExpertCacheHits
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.totalRoutedExpertCacheHits
        #endif
        case .inkling(let runner): return runner.totalRoutedExpertCacheHits
        }
    }
    public var totalRoutedExpertCacheMisses: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalRoutedExpertCacheMisses
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.totalRoutedExpertCacheMisses
        #endif
        case .inkling(let runner): return runner.totalRoutedExpertCacheMisses
        }
    }

    var gptOssPrefillExpertGroupCount: Int {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine: return 0
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.prefillExpertGroupCount
        #endif
        case .inkling: return 0
        }
    }
    var batchedPrefillProjections: Int {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.batchedPrefillProjections
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.batchedPrefillProjections
        #endif
        case .inkling: return 0
        }
    }
    /// Diagnostic kernel timing report; see `RealForwardRunner.benchmarkDecodeKernels`.
    public func benchmarkDecodeKernels() throws -> String {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return try runner.benchmarkDecodeKernels()
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return "kernel benchmark is not available for GPT-OSS\n"
        #endif
        case .inkling: return "kernel benchmark is not available for Inkling\n"
        }
    }
    public var totalGPULayerNanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalGPULayerNanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
    public var totalGPUHeadNanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalGPUHeadNanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
    public var totalGPURoutedNanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalGPURoutedNanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
    public var totalGPUSharedNanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalGPUSharedNanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
    public var totalGPULayerCommandBuffers: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalGPULayerCommandBuffers
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
    public var totalHeadFusedNanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalHeadFusedNanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
    public var totalRDAdviseNanos: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalRDAdviseNanos
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
    public var totalRDAdviseCalls: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalRDAdviseCalls
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
    public var totalRDAdviseBytes: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalRDAdviseBytes
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
    public var totalRDAdviseFailures: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalRDAdviseFailures
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
    public var totalRDAdviseSkipped: UInt64 {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.totalRDAdviseSkipped
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return 0
        #endif
        case .inkling: return 0
        }
    }
}

extension ModelForwardRunner: StateSnapshottingRunner {
    public var snapshotOwner: ObjectIdentifier {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return ObjectIdentifier(runner)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return ObjectIdentifier(runner)
        #endif
        case .inkling(let runner): return ObjectIdentifier(runner)
        }
    }

    public var stateSnapshotByteEstimate: Int? {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.stateSnapshotByteEstimate
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.stateSnapshotByteEstimate
        #endif
        case .inkling(let runner): return runner.stateSnapshotByteEstimate
        }
    }

    public func captureState() throws -> RunnerStateSnapshot {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return try runner.captureState()
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return try runner.captureState()
        #endif
        case .inkling(let runner): return try runner.captureState()
        }
    }

    public func restoreState(_ snapshot: RunnerStateSnapshot) throws {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): try runner.restoreState(snapshot)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): try runner.restoreState(snapshot)
        #endif
        case .inkling(let runner): try runner.restoreState(snapshot)
        }
    }
}

extension ModelForwardRunner: PrefixCheckpointingRunner {
    public var supportsPrefixCheckpoints: Bool {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.supportsPrefixCheckpoints
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss: return true
        #endif
        case .inkling: return true
        }
    }

    public var prefixCheckpointByteEstimate: Int {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.prefixCheckpointByteEstimate
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.prefixCheckpointByteEstimate
        #endif
        case .inkling(let runner): return runner.prefixCheckpointByteEstimate
        }
    }

    public func capturePrefixCheckpoint() throws -> RunnerStateSnapshot {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return try runner.capturePrefixCheckpoint()
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return try runner.capturePrefixCheckpoint()
        #endif
        case .inkling(let runner): return try runner.capturePrefixCheckpoint()
        }
    }

    public func rewind(to checkpoint: RunnerStateSnapshot) throws {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): try runner.rewind(to: checkpoint)
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): try runner.rewind(to: checkpoint)
        #endif
        case .inkling(let runner): try runner.rewind(to: checkpoint)
        }
    }
}

extension ModelForwardRunner: PrefillPhaseReporting {
    public var prefillPhaseTimes: PrefillPhaseTimes {
        switch backend {
        #if !TUFF_ENGINE_GPT_OSS
        case .affine(let runner): return runner.prefillPhaseTimes
        #endif
        #if !TUFF_ENGINE_AFFINE_ONLY
        case .gptOss(let runner): return runner.prefillPhaseTimes
        #endif
        case .inkling(let runner): return runner.prefillPhaseTimes
        }
    }
}
