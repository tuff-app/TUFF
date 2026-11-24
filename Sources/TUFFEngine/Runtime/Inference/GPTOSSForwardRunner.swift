import Foundation
import Metal

/// GPT-OSS forward path. Decode keeps resident BF16 projections mapped and
/// streams the four selected MXFP4 experts. Prefill works in bounded chunks,
/// groups routes by expert, and reads each selected expert once per layer and
/// chunk instead of once per token.
final class GPTOSSForwardRunner: ChunkedPrefillRunner, ContextWindowReporting,
    ContinuableLogitProducer, SpeculativeVerificationRunner, @unchecked Sendable {
    /// Rows every prefill buffer is sized for: the configured chunk, never
    /// below the size TUFF shipped with.
    private let prefillQueryCapacity: Int
    /// The prompt rows one block of `prefillChunked` runs under `config`.
    func prefillBlockTokens(_ config: PrefillRuntimeConfig) -> Int {
        max(1, min(config.chunkTokens, prefillQueryCapacity, PrefillRuntimeConfig.maxChunkTokens))
    }
    private struct LayerViews {
        let inputNorm: TensorView
        let postAttentionNorm: TensorView
        let qWeight: TensorView
        let qBias: TensorView
        let kWeight: TensorView
        let kBias: TensorView
        let vWeight: TensorView
        let vBias: TensorView
        let oWeight: TensorView
        let oBias: TensorView
        let sinks: TensorView
        let routerWeight: TensorView
        let routerBias: TensorView
        let expertOffsets: GPTOSSExpertOffsets
    }

    private let model: Model
    private let context: MetalContext
    private let config: ArchConfig
    private let kv: KVCacheManager
    private let layers: [LayerViews]

    private let bf16: BF16GEMV
    private let argmax: FP16Argmax
    private let rms: RMSNorm
    private let rope: RoPE
    private let attention: Attention
    private let elementwise: Elementwise
    private let moePrimitives: GPTOSSMoEPrimitives
    private let expertRuntime: GPTOSSExpertRuntime

    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let query: MTLBuffer
    private let attentionOutput: MTLBuffer
    private let projectedAttention: MTLBuffer
    private let routerLogits: MTLBuffer
    private let routedIndices: MTLBuffer
    private let expertScratch: GPTOSSExpertScratchBuffers
    /// Batched MPP experts for prefill; nil where MPP is unavailable.
    private let batchedExperts: GPTOSSBatchedExperts?
    /// Batched MPP BF16 projections for prefill; nil where MPP is unavailable.
    private let prefillProjection: MPPPrefillBF16MM?
    /// Decode lookahead: the next layer's predicted experts, read ahead.
    private var lookahead = ExpertLookahead()
    private var prefillClock = PrefillPhaseClock()
    var prefillPhaseTimes: PrefillPhaseTimes { prefillClock.times }
    private lazy var lookaheadIndices: MTLBuffer = context.device.makeBuffer(
        length: max(1, config.topKExperts) * MemoryLayout<UInt32>.stride,
        options: .storageModeShared)!
    private lazy var lookahead2Indices: MTLBuffer = context.device.makeBuffer(
        length: config.topKExperts * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
    private lazy var lookahead2Weights: MTLBuffer = context.device.makeBuffer(
        length: config.topKExperts * MemoryLayout<Float>.stride, options: .storageModeShared)!
    private lazy var lookaheadWeights: MTLBuffer = context.device.makeBuffer(
        length: max(1, config.topKExperts) * MemoryLayout<Float16>.stride,
        options: .storageModeShared)!
    var lookaheadPrecision: Double { lookahead.precision }
    var lookaheadReadsIssued: Int { lookahead.readsIssued }
    var exposedPrefetchWaitNanos: UInt64 { lookahead.exposedWaitNanos }
    /// Decoded layers whose cached experts started before the misses were read.
    private(set) var splitExpertLayers: UInt64 = 0
    var expertReadMetrics: ExpertReadMetrics { model.expertReadMetrics }
    var lookaheadEnabled: Bool { lookahead.enabled }
    /// Prefill K and V rows before they are copied into KV slots.
    private let kStage: MTLBuffer
    private let vStage: MTLBuffer
    private let speculativeTargetTokenBuffer: MTLBuffer
    private var lastLogitsBuffer: MTLBuffer?
    private var cachedSpeculativeBoundaryToken: Int32?
    private var speculativeStartPosition: Int?
    private var speculativeProcessedTokens = 0
    private var collectingSpeculativeMetrics = false
    private var speculativeExpertReads: UInt64 = 0
    private var speculativeExpertBytes: UInt64 = 0
    private var speculativeExpertCacheHits: UInt64 = 0
    private var speculativeExpertCacheMisses: UInt64 = 0

    let maxContext: Int
    /// Decode-phase totals only, matching `RealForwardRunner` and what
    /// `TUFF_PHASES=1` divides against. Accumulating the prefill passes into
    /// these as well made the phases sum past the decode window and printed a
    /// negative "unaccounted (GPU waits)" line on any prompt long enough to
    /// matter -- on GPT-OSS 120B it read -51,844.7 ms.
    private(set) var totalIoNanos: UInt64 = 0
    private(set) var totalCb1Nanos: UInt64 = 0
    private(set) var totalCb2Nanos: UInt64 = 0
    private(set) var totalHeadNanos: UInt64 = 0
    private(set) var prefillExpertGroupCount = 0
    /// Attention projections the batched MPP path has encoded.
    var batchedPrefillProjections: Int { prefillProjection?.encodedProjections ?? 0 }
    /// Decode-only routed-expert traffic. A read is a cache miss that required
    /// an SSD-backed expert fetch; cache hits are reported separately.
    private(set) var totalRoutedExpertReads: UInt64 = 0
    private(set) var totalRoutedExpertBytes: UInt64 = 0
    private(set) var totalRoutedExpertCacheHits: UInt64 = 0
    private(set) var totalRoutedExpertCacheMisses: UInt64 = 0

    init(model: Model, context: MetalContext, maxContext: Int,
         runtimeConfiguration: RuntimeConfiguration) throws {
        let config = model.config
        guard config.family == .gptOss,
              config.feedForwardKind == .mixtureOfExperts,
              config.topKExperts == 4,
              config.attentionSinks,
              let yarn = config.yarnRope,
              yarn.originalContextLength > 0 else {
            throw ModelError.archMismatch(
                field: "family",
                expected: ModelFamily.gptOss.rawValue,
                actual: config.family.rawValue)
        }
        guard maxContext > 0 else {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS maxContext must be positive")
        }

        try context.prepareKernelGroups(MetalKernelGroup.required(for: config))
        self.model = model
        self.context = context
        self.config = config
        self.maxContext = maxContext
        self.prefillQueryCapacity = max(PrefillRuntimeConfig.baselineChunkTokens,
                                        runtimeConfiguration.prefillChunkTokens)
        self.kv = try KVCacheManager(
            device: context.device,
            config: config,
            maxContext: maxContext,
            fp16RingEnabled: runtimeConfiguration.fp16RingEnabled,
            slidingWindow: config.slidingWindow,
            maxPrefillChunkTokens: prefillQueryCapacity,
            initialFullAttentionCapacityTokens: KVCacheManager.initialFullAttentionCapacity)

        bf16 = try BF16GEMV(context: context,
                            maxBatchRows: prefillQueryCapacity)
        argmax = try FP16Argmax(context: context)
        rms = try RMSNorm(context: context)
        rope = try RoPE(context: context)
        attention = try Attention(context: context)
        elementwise = try Elementwise(context: context)
        moePrimitives = try GPTOSSMoEPrimitives(context: context)
        expertRuntime = try GPTOSSExpertRuntime(context: context)

        var resolvedLayers: [LayerViews] = []
        resolvedLayers.reserveCapacity(config.numLayers)
        for layer in 0..<config.numLayers {
            resolvedLayers.append(LayerViews(
                inputNorm: try model.inputNorm(layer: layer),
                postAttentionNorm: try model.postAttnNorm(layer: layer),
                qWeight: try model.qProj(layer: layer),
                qBias: try model.qProjBias(layer: layer),
                kWeight: try model.kProj(layer: layer),
                kBias: try model.kProjBias(layer: layer),
                vWeight: try model.vProj(layer: layer),
                vBias: try model.vProjBias(layer: layer),
                oWeight: try model.oProj(layer: layer),
                oBias: try model.oProjBias(layer: layer),
                sinks: try model.attentionSinks(layer: layer),
                routerWeight: try model.router(layer: layer),
                routerBias: try model.routerBias(layer: layer),
                expertOffsets: try model.gptOssRoutedExpertOffsets(layer: layer)))
        }
        layers = resolvedLayers

        func sharedBuffer(elements: Int, stride: Int, label: String) throws -> MTLBuffer {
            guard let buffer = context.device.makeBuffer(
                length: max(1, elements) * stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }
        let hiddenSize = config.hiddenSize
        let querySize = config.numHeads * config.headDim
        let queryCapacity = prefillQueryCapacity
        hidden = try sharedBuffer(elements: queryCapacity * hiddenSize,
                                  stride: MemoryLayout<Float>.stride,
                                  label: "gptoss.hidden")
        normed = try sharedBuffer(elements: queryCapacity * hiddenSize,
                                  stride: MemoryLayout<Float16>.stride,
                                  label: "gptoss.normed")
        query = try sharedBuffer(elements: queryCapacity * querySize,
                                 stride: MemoryLayout<Float16>.stride,
                                 label: "gptoss.query")
        attentionOutput = try sharedBuffer(elements: queryCapacity * querySize,
                                           stride: MemoryLayout<Float16>.stride,
                                           label: "gptoss.attention")
        projectedAttention = try sharedBuffer(elements: queryCapacity * hiddenSize,
                                              stride: MemoryLayout<Float16>.stride,
                                              label: "gptoss.projectedAttention")
        routerLogits = try sharedBuffer(elements: queryCapacity * config.numExperts,
                                        stride: MemoryLayout<Float>.stride,
                                        label: "gptoss.routerLogits")
        routedIndices = try sharedBuffer(elements: queryCapacity * config.topKExperts,
                                         stride: MemoryLayout<UInt32>.stride,
                                         label: "gptoss.routedIndices")
        speculativeTargetTokenBuffer = try sharedBuffer(
            elements: 8, stride: MemoryLayout<UInt32>.stride,
            label: "gptoss.speculativeTargets")
        expertScratch = try GPTOSSExpertScratchBuffers.allocate(
            device: context.device,
            layout: GPTOSSExpertScratchLayout(
                hiddenSize: hiddenSize,
                intermediateSize: config.moeIntermediateSize,
                topK: config.topKExperts,
                queryCapacity: queryCapacity))
        kStage = try sharedBuffer(elements: queryCapacity * config.numKVHeads * config.headDim,
                                  stride: MemoryLayout<Float16>.stride,
                                  label: "gptoss.prefill.kStage")
        vStage = try sharedBuffer(elements: queryCapacity * config.numKVHeads * config.headDim,
                                  stride: MemoryLayout<Float16>.stride,
                                  label: "gptoss.prefill.vStage")
        batchedExperts = GPTOSSBatchedExperts(context: context,
                                              hiddenSize: hiddenSize,
                                              intermediateSize: config.moeIntermediateSize)
        prefillProjection = MPPPrefillBF16MM(context: context)
    }

    func reset() {
        _ = lookahead.drain()
        kv.reset()
        lastLogitsBuffer = nil
        cachedSpeculativeBoundaryToken = nil
        speculativeStartPosition = nil
        speculativeProcessedTokens = 0
    }

    var continuationPosition: Int { kv.position }

    // MARK: Retained conversation state

    /// GPT-OSS carries only KV rows between tokens; the logits buffer, the
    /// speculative boundary and expert prediction are per-request.
    var stateSnapshotByteEstimate: Int? {
        guard speculativeStartPosition == nil else { return nil }
        return kv.snapshotBytes(position: kv.position)
    }

    func captureState() throws -> RunnerStateSnapshot {
        guard speculativeStartPosition == nil else {
            throw RunnerStateSnapshotError.unsupported("a speculative verification is in progress")
        }
        var builder = RunnerStateSnapshotBuilder()
        kv.addSnapshotRanges(to: &builder, position: kv.position)
        return try builder.capture(owner: self, queue: context.queue,
                                   host: .init(position: kv.position, ngramContext: [], ropeDelta: 0))
    }

    func restoreState(_ snapshot: RunnerStateSnapshot) throws {
        guard snapshot.owner == ObjectIdentifier(self) else {
            throw RunnerStateSnapshotError.foreignSnapshot
        }
        reset()
        do {
            let position = snapshot.host.position
            guard position <= kv.maxContext else {
                throw RunnerStateSnapshotError.layoutMismatch("position exceeds context")
            }
            if position > 0 { try kv.ensureCapacity(through: position, on: context.queue) }
            var builder = RunnerStateSnapshotBuilder()
            kv.addSnapshotRanges(to: &builder, position: position)
            try builder.restore(snapshot, queue: context.queue)
            kv.restorePosition(position)
        } catch {
            reset()
            throw error
        }
    }

    // MARK: Prefix checkpoints

    /// GPT-OSS's full-attention rows are only appended to, so returning to an
    /// earlier position needs just the sliding-window rings.
    var supportsPrefixCheckpoints: Bool { true }

    var prefixCheckpointByteEstimate: Int {
        kv.snapshotBytes(position: kv.position, skipLayer: { [kv] in !kv.isRingLayer($0) })
    }

    func capturePrefixCheckpoint() throws -> RunnerStateSnapshot {
        guard speculativeStartPosition == nil else {
            throw RunnerStateSnapshotError.unsupported("a speculative verification is in progress")
        }
        var builder = RunnerStateSnapshotBuilder()
        kv.addSnapshotRanges(to: &builder, position: kv.position,
                             skipLayer: { [kv] in !kv.isRingLayer($0) })
        return try builder.capture(owner: self, queue: context.queue,
                                   host: .init(position: kv.position, ngramContext: [], ropeDelta: 0))
    }

    func rewind(to checkpoint: RunnerStateSnapshot) throws {
        guard checkpoint.owner == ObjectIdentifier(self) else {
            throw RunnerStateSnapshotError.foreignSnapshot
        }
        let position = checkpoint.host.position
        do {
            guard position > 0, position <= kv.position, speculativeStartPosition == nil else {
                throw RunnerStateSnapshotError.layoutMismatch(
                    "checkpoint at \(position) is past the sequence at \(kv.position)")
            }
            var builder = RunnerStateSnapshotBuilder()
            kv.addSnapshotRanges(to: &builder, position: position,
                                 skipLayer: { [kv] in !kv.isRingLayer($0) })
            try builder.restore(checkpoint, queue: context.queue)
            kv.rewind(to: position)
            lastLogitsBuffer = nil
            cachedSpeculativeBoundaryToken = nil
            speculativeProcessedTokens = 0
        } catch {
            reset()
            throw error
        }
    }

    func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "GPT-OSS continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        lastLogitsBuffer = nil
        cachedSpeculativeBoundaryToken = nil
        speculativeStartPosition = nil
        speculativeProcessedTokens = 0
    }

    var supportsSpeculativeVerification: Bool { true }

    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await executeToken(token: token, position: position,
                               emitHead: true, logits: logits)
    }

    func verifySpeculativeBlock(tokens: [Int32],
                                startPosition: Int,
                                into logits: MTLBuffer) async throws
        -> SpeculativeVerificationResult {
        guard (1...8).contains(tokens.count) else {
            throw SpeculativeDecodingError.invalidBlockSize(
                requested: tokens.count, maximum: 8)
        }
        guard speculativeStartPosition == nil else {
            throw PrefillError.chunkedRunnerDirty(
                "a speculative verification transaction is already active")
        }
        guard kv.position == startPosition else {
            throw SpeculativeDecodingError.invalidStartPosition(
                expected: kv.position, actual: startPosition)
        }
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < config.vocabSize }) else {
            throw GeneratorError.invalidGenerationConfig(
                "speculative token is outside the model vocabulary")
        }
        guard let currentLogits = lastLogitsBuffer else {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS has no target logits at the current boundary")
        }

        speculativeExpertReads = 0
        speculativeExpertBytes = 0
        speculativeExpertCacheHits = 0
        speculativeExpertCacheMisses = 0
        let prefillExpertGroupsBefore = prefillExpertGroupCount
        collectingSpeculativeMetrics = true
        defer { collectingSpeculativeMetrics = false }

        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let boundaryToken: Int32
        if let cachedSpeculativeBoundaryToken {
            boundaryToken = cachedSpeculativeBoundaryToken
        } else {
            boundaryToken = try currentGreedyToken(from: currentLogits)
            cachedSpeculativeBoundaryToken = boundaryToken
        }
        do {
            try await executePrefillChunk(
                tokens: tokens[...],
                startPosition: startPosition,
                emitHead: true,
                logits: logits,
                speculativeTargetTokens: speculativeTargetTokenBuffer)
        } catch {
            kv.rewind(to: startPosition)
            speculativeStartPosition = nil
            speculativeProcessedTokens = 0
            throw error
        }
        let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start
        let targetPointer = speculativeTargetTokenBuffer
            .contents().assumingMemoryBound(to: UInt32.self)
        var targetTokens = [boundaryToken]
        targetTokens.reserveCapacity(tokens.count + 1)
        for index in 0..<tokens.count {
            targetTokens.append(Int32(bitPattern: targetPointer[index]))
        }
        speculativeStartPosition = startPosition
        speculativeProcessedTokens = tokens.count
        let expertGroupCount = prefillExpertGroupCount - prefillExpertGroupsBefore
        return SpeculativeVerificationResult(
            startPosition: startPosition,
            proposedTokenIDs: tokens,
            targetTokenIDs: targetTokens,
            processedTokens: tokens.count,
            newPosition: startPosition + tokens.count,
            metrics: SpeculativeVerificationMetrics(
                wallNanos: wallNanos,
                // One embedding CB, one CB per layer, one CB per streamed
                // expert group, and one batched output-head CB.
                targetCommandBuffers: UInt64(2 + config.numLayers
                                              + expertGroupCount),
                expertReads: speculativeExpertReads,
                expertBytes: speculativeExpertBytes,
                expertCacheHits: speculativeExpertCacheHits,
                expertCacheMisses: speculativeExpertCacheMisses))
    }

    private func recordSpeculativeFetch(layer: Int,
                                        plan: RoutedExpertFetchPlan?) {
        guard collectingSpeculativeMetrics, let plan else { return }
        recordDecodeExpertFetch(layer: layer, plan: plan)
        let misses = plan.misses.count
        speculativeExpertReads &+= UInt64(misses)
        speculativeExpertCacheMisses &+= UInt64(misses)
        speculativeExpertCacheHits &+= UInt64(plan.hits)
        if let bytes = try? model.routedExpertAdviceByteEstimate(
            layer: layer, missCount: misses) {
            speculativeExpertBytes &+= bytes
        }
    }

    private func recordDecodeExpertFetch(layer: Int,
                                         plan: RoutedExpertFetchPlan) {
        let misses = plan.misses.count
        totalRoutedExpertReads &+= UInt64(misses)
        totalRoutedExpertCacheMisses &+= UInt64(misses)
        totalRoutedExpertCacheHits &+= UInt64(plan.hits)
        if let bytes = try? model.routedExpertAdviceByteEstimate(
            layer: layer, missCount: misses) {
            totalRoutedExpertBytes &+= bytes
        }
    }

    func commitSpeculativePrefix(_ count: Int) throws {
        guard let start = speculativeStartPosition else {
            throw SpeculativeDecodingError.noActiveTransaction
        }
        guard (0...speculativeProcessedTokens).contains(count) else {
            throw SpeculativeDecodingError.invalidCommitCount(
                requested: count, processed: speculativeProcessedTokens)
        }
        kv.rewind(to: start + count)
        speculativeStartPosition = nil
        speculativeProcessedTokens = 0
        lastLogitsBuffer = nil
        cachedSpeculativeBoundaryToken = nil
    }

    func speculativeBoundaryToken() async throws -> Int32? {
        guard let currentLogits = lastLogitsBuffer else { return nil }
        if let cachedSpeculativeBoundaryToken {
            return cachedSpeculativeBoundaryToken
        }
        let token = try currentGreedyToken(from: currentLogits)
        cachedSpeculativeBoundaryToken = token
        return token
    }

    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode _: PrefillOutputMode,
                        config prefillConfig: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        guard prefillConfig.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS chunked prefill requires chunked mode")
        }
        guard startPosition == kv.position else {
            throw PrefillError.prefillCursorMismatch(
                "GPT-OSS prefill cursor \(kv.position) != startPosition \(startPosition)")
        }
        guard startPosition >= 0,
              tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS prefill exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition,
                                 seed: .logitsWritten)
        }
        try kv.ensureCapacity(through: startPosition + tokens.count, on: context.queue)

        let spans = PrefillChunkPlanner.spans(
            tokenCount: tokens.count,
            startPosition: startPosition,
            config: prefillConfig.replacingChunkTokens(
                min(prefillConfig.chunkTokens, prefillQueryCapacity)))
        try await PrefillSpanIteration.forEachSpan(spans) { _, span in
            let lower = tokens.index(tokens.startIndex,
                                     offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                emitHead: span.completedCount == tokens.count,
                logits: logits)
            let firstCompleted = span.completedCount - span.tokenCount + 1
            for completed in firstCompleted...span.completedCount {
                onProgress(completed)
            }
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     emitHead: Bool,
                                     logits: MTLBuffer,
                                     speculativeTargetTokens: MTLBuffer? = nil)
        async throws {
        let queryCount = tokens.count
        guard queryCount > 0, queryCount <= prefillQueryCapacity else {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS prefill chunk has unsupported size \(queryCount)")
        }
        if emitHead && logits.length < config.vocabSize * MemoryLayout<Float16>.stride {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS logits buffer is smaller than the model vocabulary")
        }
        prefillClock.active = speculativeTargetTokens == nil
        defer { prefillClock.active = false }

        _ = lookahead.drain()
        let floatBytes = MemoryLayout<Float>.stride
        let embeddingCB = context.queue.makeCommandBuffer()!
        for (row, token) in tokens.enumerated() {
            guard token >= 0, Int(token) < config.vocabSize else {
                throw GeneratorError.invalidGenerationConfig(
                    "GPT-OSS token ID \(token) is outside vocab \(config.vocabSize)")
            }
            bf16.encodeFloatEmbedding(
                commandBuffer: embeddingCB,
                table: model.embedding,
                token: UInt32(token),
                output: hidden,
                outputOffset: row * config.hiddenSize * floatBytes,
                hiddenSize: config.hiddenSize)
        }
        embeddingCB.label = "gptoss prefill phase=embed_attention"
        embeddingCB.commit()
        try waitForCompletion(embeddingCB)
        for layer in 0..<config.numLayers {
            try Task.checkCancellation()
            try await executePrefillLayer(
                layer, startPosition: startPosition, queryCount: queryCount,
                speculativeVerification: speculativeTargetTokens != nil)
        }
        for _ in 0..<queryCount { kv.advance() }

        if emitHead {
            if let speculativeTargetTokens {
                try executeHeadRows(
                    hiddenOffset: 0,
                    normedOffset: 0,
                    hiddenStride: config.hiddenSize * floatBytes,
                    normedStride: config.hiddenSize * MemoryLayout<Float16>.stride,
                    rowCount: queryCount,
                    outputTokens: speculativeTargetTokens)
            } else {
                try executeHead(
                    hiddenOffset: (queryCount - 1) * config.hiddenSize * floatBytes,
                    normedOffset: (queryCount - 1) * config.hiddenSize
                        * MemoryLayout<Float16>.stride,
                    logits: logits)
            }
        }
    }

    private func executePrefillLayer(_ layer: Int,
                                     startPosition: Int,
                                     queryCount: Int,
                                     speculativeVerification: Bool) async throws {
        let views = layers[layer]
        let hiddenSize = config.hiddenSize
        let qRows = config.numHeads * config.headDim
        let kvRows = config.numKVHeads * config.headDim
        let halfBytes = MemoryLayout<Float16>.stride

        let cb1 = context.queue.makeCommandBuffer()!
        let batchedChunk = PrefillBatchingPolicy.batchesProjections(
            tokenCount: queryCount, speculativeVerification: speculativeVerification)

        func project(_ weights: TensorView, bias: TensorView, input: MTLBuffer,
                     output: MTLBuffer, rows: Int, columns: Int) {
            if batchedChunk, let mpp = prefillProjection,
               mpp.encode(commandBuffer: cb1, weights: weights, bias: bias,
                          x: input, y: output, m: queryCount, n: rows, k: columns) {
                return
            }
            bf16.encodeHalfRows(
                commandBuffer: cb1,
                weights: weights,
                input: input,
                inputStrideElements: columns,
                output: output,
                outputStrideElements: rows,
                bias: bias,
                batchCount: queryCount,
                rows: rows,
                columns: columns)
        }

        // Normalize and project all candidate rows through the resident
        // weights with bounded 2-D dispatches. K and V are projected into
        // staging rows in one dispatch each, then copied row by row into
        // their KV slots, because a ring-enabled KV cache may wrap each
        // position to a different physical slot.
        rms.encodeFloatBF16WRows(
            commandBuffer: cb1,
            x: hidden,
            xStrideElements: hiddenSize,
            weight: views.inputNorm.buffer,
            weightOffset: Int(views.inputNorm.offset),
            out: normed,
            outStrideElements: hiddenSize,
            rows: queryCount,
            d: UInt32(hiddenSize),
            eps: 1e-5)
        project(views.qWeight, bias: views.qBias, input: normed,
                output: query, rows: qRows, columns: hiddenSize)

        project(views.kWeight, bias: views.kBias, input: normed,
                output: kStage, rows: kvRows, columns: hiddenSize)
        project(views.vWeight, bias: views.vBias, input: normed,
                output: vStage, rows: kvRows, columns: hiddenSize)
        // RoPE runs over the whole chunk: Q in place, K in its staging rows
        // before they are copied to their slots.
        encodeRoPE(commandBuffer: cb1, data: query, dataOffset: 0,
                   position: startPosition, heads: config.numHeads, tokens: queryCount)
        encodeRoPE(commandBuffer: cb1, data: kStage, dataOffset: 0,
                   position: startPosition, heads: config.numKVHeads, tokens: queryCount)
        if let blit = cb1.makeBlitCommandEncoder() {
            let rowBytes = kvRows * halfBytes
            precondition(kv.stride(layer: layer) == rowBytes,
                         "GPT-OSS KV slot stride must be one K/V row")
            for row in 0..<queryCount {
                let kSlot = kv.kSlot(layer: layer, position: startPosition + row)
                let vSlot = kv.vSlot(layer: layer, position: startPosition + row)
                blit.copy(from: kStage, sourceOffset: row * rowBytes,
                          to: kSlot.buffer, destinationOffset: kSlot.offset, size: rowBytes)
                blit.copy(from: vStage, sourceOffset: row * rowBytes,
                          to: vSlot.buffer, destinationOffset: vSlot.offset, size: rowBytes)
            }
            blit.endEncoding()
        }

        for row in 0..<queryCount {
            let position = startPosition + row
            let queryOffset = row * qRows * halfBytes
            let kSlot = kv.kSlot(layer: layer, position: position)
            let vSlot = kv.vSlot(layer: layer, position: position)

            let sequenceLength = UInt32(position + 1)
            if config.layerIsFull(layer) {
                attention.encodeFull(
                    commandBuffer: cb1,
                    q: query, qOffset: queryOffset,
                    k: kSlot.buffer,
                    v: vSlot.buffer,
                    out: attentionOutput, outOffset: queryOffset,
                    headDim: UInt32(config.headDim),
                    numQHeads: UInt32(config.numHeads),
                    numKVHeads: UInt32(config.numKVHeads),
                    seqLen: sequenceLength,
                    scale: Float(config.attentionScale),
                    sinks: views.sinks.buffer,
                    sinksOffset: Int(views.sinks.offset))
            } else {
                attention.encodeSWA(
                    commandBuffer: cb1,
                    q: query, qOffset: queryOffset,
                    k: kSlot.buffer,
                    v: vSlot.buffer,
                    out: attentionOutput, outOffset: queryOffset,
                    headDim: UInt32(config.headDim),
                    numQHeads: UInt32(config.numHeads),
                    numKVHeads: UInt32(config.numKVHeads),
                    seqLen: sequenceLength,
                    window: UInt32(config.slidingWindow),
                    scale: Float(config.attentionScale),
                    sinks: views.sinks.buffer,
                    sinksOffset: Int(views.sinks.offset),
                    ringCapacity: UInt32(kv.ringCapacity(layer: layer)))
            }
        }

        project(views.oWeight, bias: views.oBias, input: attentionOutput,
                output: projectedAttention, rows: hiddenSize, columns: qRows)
        // Rows are contiguous in both buffers, so one dispatch covers the
        // chunk instead of one per token.
        elementwise.encodeFloatResidualAdd(
            commandBuffer: cb1,
            hidden: hidden, hiddenOffset: 0,
            delta: projectedAttention, deltaOffset: 0,
            count: queryCount * hiddenSize)
        rms.encodeFloatBF16WRows(
            commandBuffer: cb1,
            x: hidden,
            xStrideElements: hiddenSize,
            weight: views.postAttentionNorm.buffer,
            weightOffset: Int(views.postAttentionNorm.offset),
            out: normed,
            outStrideElements: hiddenSize,
            rows: queryCount,
            d: UInt32(hiddenSize),
            eps: 1e-5)
        bf16.encodeFloatRows(
            commandBuffer: cb1,
            weights: views.routerWeight,
            input: normed,
            inputStrideElements: hiddenSize,
            output: routerLogits,
            outputStrideElements: config.numExperts,
            bias: views.routerBias,
            batchCount: queryCount,
            rows: config.numExperts,
            columns: hiddenSize)
        precondition(config.topKExperts == 4, "GPT-OSS routes each token to 4 experts")
        moePrimitives.encodeRouterTop4(
            commandBuffer: cb1,
            logits: routerLogits,
            outputIndices: routedIndices,
            outputWeights: expertScratch.routeWeights,
            numExperts: UInt32(config.numExperts),
            rows: queryCount)
        cb1.label = "gptoss prefill layer=\(layer) phase=attention"
        cb1.commit()
        try waitForCompletion(cb1)

        let indexPointer = routedIndices.contents()
            .assumingMemoryBound(to: UInt32.self)
        var uniqueExperts = Set<Int>()
        for route in 0..<(queryCount * config.topKExperts) {
            uniqueExperts.insert(Int(indexPointer[route]))
        }
        let physicalOffsets = model.routedExpertPhysicalOffsets(layer: layer)
        let orderedExperts = uniqueExperts.sorted {
            physicalOffsets[$0] < physicalOffsets[$1]
        }
        let availableSlots = model.routedExpertCacheSlotCount(layer: layer)
            ?? orderedExperts.count
        let groupSize = max(1, min(availableSlots, orderedExperts.count))

        // The batched path wants each expert's pairs contiguous, in the same
        // order the fetch groups walk the experts.
        let batched = PrefillBatchingPolicy.batchesRoutedExperts(
            tokenCount: queryCount, speculativeVerification: speculativeVerification)
            ? batchedExperts : nil
        var pairsByExpert: [Int: (start: Int, count: Int)] = [:]
        var sortedPairsBuffer: MTLBuffer?
        if batched != nil {
            let order = Dictionary(uniqueKeysWithValues: orderedExperts.enumerated().map { ($1, $0) })
            var pairs: [PrefillTokenExpertPair] = []
            pairs.reserveCapacity(queryCount * config.topKExperts)
            for row in 0..<queryCount {
                for slot in 0..<config.topKExperts {
                    pairs.append(PrefillTokenExpertPair(
                        token: UInt32(row),
                        expert: indexPointer[row * config.topKExperts + slot],
                        rank: UInt32(slot),
                        weight: 0))
                }
            }
            pairs.sort {
                let l = order[Int($0.expert)]!, r = order[Int($1.expert)]!
                return l != r ? l < r : ($0.token, $0.rank) < ($1.token, $1.rank)
            }
            for (index, pair) in pairs.enumerated() {
                let expert = Int(pair.expert)
                if let run = pairsByExpert[expert] {
                    pairsByExpert[expert] = (run.start, run.count + 1)
                } else {
                    pairsByExpert[expert] = (index, 1)
                }
            }
            sortedPairsBuffer = pairs.withUnsafeBytes {
                context.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                          options: .storageModeShared)
            }
        }
        var groupStart = 0
        while groupStart < orderedExperts.count {
            try Task.checkCancellation()
            let groupEnd = min(orderedExperts.count, groupStart + groupSize)
            let expertIDs = Array(orderedExperts[groupStart..<groupEnd])
            prefillExpertGroupCount += 1
            let plannedFetch = try model.planRoutedExperts(
                layer: layer, experts: expertIDs)
            recordSpeculativeFetch(layer: layer, plan: plannedFetch)
            if !collectingSpeculativeMetrics, let plannedFetch {
                recordDecodeExpertFetch(layer: layer, plan: plannedFetch)
            }
            let blobs: [TensorView]
            let readStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if let plannedFetch {
                blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
            } else {
                blobs = try await model.fetchRoutedExperts(
                    layer: layer, experts: expertIDs)
            }
            prefillClock.expertRead(nanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - readStart)
            var blobByExpert: [Int: TensorView] = [:]
            for (expert, blob) in zip(expertIDs, blobs) {
                blobByExpert[expert] = blob
            }

            let cb2 = context.queue.makeCommandBuffer()!
            if let batched, let sortedPairsBuffer {
                let groups = expertIDs.compactMap { expert -> GPTOSSBatchedExperts.Group? in
                    guard let run = pairsByExpert[expert], let blob = blobByExpert[expert] else {
                        return nil
                    }
                    return GPTOSSBatchedExperts.Group(blob: blob, pairStart: run.start,
                                                      pairCount: run.count)
                }
                try batched.encode(commandBuffer: cb2, input: normed,
                                   sortedPairs: sortedPairsBuffer, groups: groups,
                                   offsets: views.expertOffsets,
                                   routePartials: expertScratch.routePartials,
                                   topK: config.topKExperts,
                                   swigluLimit: Float(config.swigluLimit))
            } else {
            for row in 0..<queryCount {
                for routeSlot in 0..<config.topKExperts {
                    let routeIndex = row * config.topKExperts + routeSlot
                    let expert = Int(indexPointer[routeIndex])
                    guard let blob = blobByExpert[expert] else { continue }
                    try expertRuntime.encodeExpert(
                        commandBuffer: cb2,
                        blob: blob,
                        offsets: views.expertOffsets,
                        input: normed,
                        queryIndex: row,
                        routeSlot: routeSlot,
                        scratch: expertScratch,
                        swigluLimit: Float(config.swigluLimit))
                }
            }
            }
            if groupEnd == orderedExperts.count {
                try expertRuntime.encodeFloatResidualReduce(
                    commandBuffer: cb2,
                    scratch: expertScratch,
                    residual: hidden,
                    output: hidden,
                    queryCount: queryCount)
            }
            cb2.label = "gptoss prefill layer=\(layer) phase=routed_tile"
            cb2.commit()
            try withExtendedLifetime(blobs) {
                try waitForCompletion(cb2)
            }
            groupStart = groupEnd
        }
    }

    private func executeToken(token: Int32, position: Int,
                              emitHead: Bool, logits: MTLBuffer) async throws {
        guard position == kv.position else {
            throw PrefillError.prefillCursorMismatch(
                "GPT-OSS token position \(position) != KV position \(kv.position)")
        }
        guard position >= 0, position < maxContext else {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS position \(position) exceeds maxContext \(maxContext)")
        }
        guard token >= 0, Int(token) < config.vocabSize else {
            throw GeneratorError.invalidGenerationConfig(
                "GPT-OSS token ID \(token) is outside vocab \(config.vocabSize)")
        }
        try kv.ensureCapacity(through: position + 1, on: context.queue)
        if emitHead && logits.length < config.vocabSize * MemoryLayout<Float16>.stride {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS logits buffer is smaller than the model vocabulary")
        }

        // A token that threw midway can leave the next layer's read in flight.
        _ = lookahead.drain()
        let embeddingCB = context.queue.makeCommandBuffer()!
        bf16.encodeFloatEmbedding(commandBuffer: embeddingCB,
                                  table: model.embedding,
                                  token: UInt32(token),
                                  output: hidden,
                                  hiddenSize: config.hiddenSize)
        embeddingCB.commit()
        try waitForCompletion(embeddingCB)

        for layer in 0..<config.numLayers {
            try Task.checkCancellation()
            try await executeLayer(layer, position: position)
        }
        kv.advance()

        guard emitHead else { return }
        try executeHead(hiddenOffset: 0, normedOffset: 0, logits: logits)
    }

    private func executeHead(hiddenOffset: Int,
                             normedOffset: Int,
                             logits: MTLBuffer) throws {
        let headStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let headCB = context.queue.makeCommandBuffer()!
        rms.encodeFloatBF16W(commandBuffer: headCB,
                             x: hidden, xOffset: hiddenOffset,
                             weight: model.finalNorm.buffer,
                             weightOffset: Int(model.finalNorm.offset),
                             out: normed, outOffset: normedOffset,
                             d: UInt32(config.hiddenSize),
                             eps: 1e-5)
        bf16.encodeHalf(commandBuffer: headCB,
                        weights: model.lmHead,
                        input: normed, inputOffset: normedOffset,
                        output: logits,
                        rows: config.vocabSize,
                        columns: config.hiddenSize)
        headCB.commit()
        try waitForCompletion(headCB)
        lastLogitsBuffer = logits
        cachedSpeculativeBoundaryToken = nil
        totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - headStart
    }

    private func executeHeadRows(hiddenOffset: Int,
                                 normedOffset: Int,
                                 hiddenStride: Int,
                                 normedStride: Int,
                                 rowCount: Int,
                                 outputTokens: MTLBuffer) throws {
        precondition((1...prefillQueryCapacity).contains(rowCount))
        let headStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let headCB = context.queue.makeCommandBuffer()!
        rms.encodeFloatBF16WRows(
            commandBuffer: headCB,
            x: hidden,
            xOffset: hiddenOffset,
            xStrideElements: hiddenStride / MemoryLayout<Float>.stride,
            weight: model.finalNorm.buffer,
            weightOffset: Int(model.finalNorm.offset),
            out: normed,
            outOffset: normedOffset,
            outStrideElements: normedStride / MemoryLayout<Float16>.stride,
            rows: rowCount,
            d: UInt32(config.hiddenSize),
            eps: 1e-5)
        bf16.encodeHalfArgmaxRows(
            commandBuffer: headCB,
            weights: model.lmHead,
            input: normed,
            inputOffset: normedOffset,
            inputStrideElements: normedStride / MemoryLayout<Float16>.stride,
            output: outputTokens,
            rowCount: rowCount,
            rows: config.vocabSize,
            columns: config.hiddenSize)
        headCB.commit()
        try waitForCompletion(headCB)
        totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - headStart
    }

    private func currentGreedyToken(from logits: MTLBuffer) throws -> Int32 {
        let cb = context.queue.makeCommandBuffer()!
        argmax.encode(commandBuffer: cb,
                      values: logits,
                      count: config.vocabSize,
                      output: speculativeTargetTokenBuffer)
        cb.commit()
        try waitForCompletion(cb)
        return Int32(bitPattern: speculativeTargetTokenBuffer.contents()
            .assumingMemoryBound(to: UInt32.self)[0])
    }

    private func executeLayer(_ layer: Int, position: Int) async throws {
        let views = layers[layer]
        let hiddenSize = config.hiddenSize
        let qRows = config.numHeads * config.headDim
        let kvRows = config.numKVHeads * config.headDim
        let kSlot = kv.kSlot(layer: layer, position: position)
        let vSlot = kv.vSlot(layer: layer, position: position)

        let cb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let cb1 = context.queue.makeCommandBuffer()!
        rms.encodeFloatBF16W(commandBuffer: cb1,
                             x: hidden,
                             weight: views.inputNorm.buffer,
                             weightOffset: Int(views.inputNorm.offset),
                             out: normed,
                             d: UInt32(hiddenSize),
                             eps: 1e-5)
        bf16.encodeHalf(commandBuffer: cb1,
                        weights: views.qWeight,
                        input: normed,
                        output: query,
                        bias: views.qBias,
                        rows: qRows,
                        columns: hiddenSize)
        bf16.encodeHalf(commandBuffer: cb1,
                        weights: views.kWeight,
                        input: normed,
                        output: kSlot.buffer,
                        outputOffset: kSlot.offset,
                        bias: views.kBias,
                        rows: kvRows,
                        columns: hiddenSize)
        bf16.encodeHalf(commandBuffer: cb1,
                        weights: views.vWeight,
                        input: normed,
                        output: vSlot.buffer,
                        outputOffset: vSlot.offset,
                        bias: views.vBias,
                        rows: kvRows,
                        columns: hiddenSize)
        encodeRoPE(commandBuffer: cb1, data: query, dataOffset: 0,
                   position: position, heads: config.numHeads)
        encodeRoPE(commandBuffer: cb1, data: kSlot.buffer,
                   dataOffset: kSlot.offset,
                   position: position, heads: config.numKVHeads)

        let sequenceLength = UInt32(position + 1)
        if config.layerIsFull(layer) {
            attention.encodeFull(
                commandBuffer: cb1,
                q: query,
                k: kSlot.buffer,
                v: vSlot.buffer,
                out: attentionOutput,
                headDim: UInt32(config.headDim),
                numQHeads: UInt32(config.numHeads),
                numKVHeads: UInt32(config.numKVHeads),
                seqLen: sequenceLength,
                scale: Float(config.attentionScale),
                sinks: views.sinks.buffer,
                sinksOffset: Int(views.sinks.offset))
        } else {
            attention.encodeSWA(
                commandBuffer: cb1,
                q: query,
                k: kSlot.buffer,
                v: vSlot.buffer,
                out: attentionOutput,
                headDim: UInt32(config.headDim),
                numQHeads: UInt32(config.numHeads),
                numKVHeads: UInt32(config.numKVHeads),
                seqLen: sequenceLength,
                window: UInt32(config.slidingWindow),
                scale: Float(config.attentionScale),
                sinks: views.sinks.buffer,
                sinksOffset: Int(views.sinks.offset),
                ringCapacity: UInt32(kv.ringCapacity(layer: layer)))
        }
        bf16.encodeHalf(commandBuffer: cb1,
                        weights: views.oWeight,
                        input: attentionOutput,
                        output: projectedAttention,
                        bias: views.oBias,
                        rows: hiddenSize,
                        columns: qRows)
        elementwise.encodeFloatResidualAdd(commandBuffer: cb1,
                                           hidden: hidden,
                                           delta: projectedAttention,
                                           count: hiddenSize)
        rms.encodeFloatBF16W(commandBuffer: cb1,
                             x: hidden,
                             weight: views.postAttentionNorm.buffer,
                             weightOffset: Int(views.postAttentionNorm.offset),
                             out: normed,
                             d: UInt32(hiddenSize),
                             eps: 1e-5)
        bf16.encodeFloat(commandBuffer: cb1,
                         weights: views.routerWeight,
                         input: normed,
                         output: routerLogits,
                         bias: views.routerBias,
                         rows: config.numExperts,
                         columns: hiddenSize)
        moePrimitives.encodeRouterTop4(
            commandBuffer: cb1,
            logits: routerLogits,
            outputIndices: routedIndices,
            outputWeights: expertScratch.routeWeights,
            numExperts: UInt32(config.numExperts))
        // Lookahead: layer + 1's router over this layer's input predicts its
        // experts, read ahead while this layer computes. See `ExpertLookahead`.
        var lookaheadEncoded = false
        if lookahead.enabled, layer + 1 < config.numLayers {
            let next = layers[layer + 1]
            bf16.encodeFloat(commandBuffer: cb1,
                             weights: next.routerWeight,
                             input: normed,
                             output: routerLogits,
                             bias: next.routerBias,
                             rows: config.numExperts,
                             columns: hiddenSize)
            moePrimitives.encodeRouterTop4(
                commandBuffer: cb1,
                logits: routerLogits,
                outputIndices: lookaheadIndices,
                outputWeights: lookaheadWeights,
                numExperts: UInt32(config.numExperts))
            lookaheadEncoded = true
        }
        // Two ahead: layer + 2's router over the same input. Its reads start
        // a layer earlier, at the cost of a less accurate guess.
        var lookahead2Encoded = false
        if lookahead.enabled, lookahead.depth >= 2, layer + 2 < config.numLayers {
            let next = layers[layer + 2]
            bf16.encodeFloat(commandBuffer: cb1,
                             weights: next.routerWeight,
                             input: normed,
                             output: routerLogits,
                             bias: next.routerBias,
                             rows: config.numExperts,
                             columns: hiddenSize)
            moePrimitives.encodeRouterTop4(
                commandBuffer: cb1,
                logits: routerLogits,
                outputIndices: lookahead2Indices,
                outputWeights: lookahead2Weights,
                numExperts: UInt32(config.numExperts))
            lookahead2Encoded = true
        }
        cb1.commit()
        try waitForCompletion(cb1)
        totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cb1Start

        let indexPointer = routedIndices.contents()
            .assumingMemoryBound(to: UInt32.self)
        let experts = (0..<config.topKExperts).map { Int(indexPointer[$0]) }

        if let finished = lookahead.drain(layer: layer) {
            lookahead.record(predicted: finished.experts, actual: experts)
        }
        var lookaheadDispatch: (() -> Void)?
        func fireLookahead() {
            lookaheadDispatch?()
            lookaheadDispatch = nil
        }
        // Every exit, including a throw, releases a registered read.
        defer { fireLookahead() }
        func predictions(_ buffer: MTLBuffer) -> [Int] {
            let pointer = buffer.contents().assumingMemoryBound(to: UInt32.self)
            return (0..<config.topKExperts).map { min(Int(pointer[$0]), config.numExperts - 1) }
        }
        // Reads are planned here on the runner's thread and run one after
        // another off it, the nearer layer first, once this layer's own fetch
        // is done: started together they would split the SSD's bandwidth.
        var reads: [(operation: @Sendable () throws -> Void, group: DispatchGroup)] = []
        func planRead(layer target: Int, experts predicted: [Int]) throws -> DispatchGroup? {
            guard let plan = try model.planRoutedExperts(layer: target, experts: predicted,
                                                         purpose: .prefetch),
                  !plan.misses.isEmpty else { return nil }
            let group = DispatchGroup()
            group.enter()
            reads.append((try model.expertPrefetchOperation(plan: plan), group))
            lookahead.noteRead()
            return group
        }
        if lookaheadEncoded {
            let predicted = predictions(lookaheadIndices)
            // A read two ahead may still be filling layer + 1's slots, and a
            // streamer is planned again only once its reads have landed. If
            // it has not, that layer's own fetch reads whatever is missing.
            if lookahead.isSettled(layer: layer + 1) {
                lookahead.expect(layer: layer + 1, experts: predicted,
                                 reads: try planRead(layer: layer + 1, experts: predicted))
            }
        }
        if lookahead2Encoded {
            let predicted = predictions(lookahead2Indices)
            lookahead.expect(layer: layer + 2, experts: predicted,
                             reads: try planRead(layer: layer + 2, experts: predicted))
        }
        if !reads.isEmpty {
            let queued = reads
            lookaheadDispatch = {
                DispatchQueue.global(qos: .userInitiated).async {
                    for read in queued {
                        // A failed read only loses the head start: the
                        // layer's own fetch reads whatever is missing.
                        try? read.operation()
                        read.group.leave()
                    }
                }
            }
        }

        let plannedFetch = try model.planRoutedExperts(
            layer: layer, experts: experts)
        // Experts already in the cache start on the GPU while the missing ones
        // are read. Each route writes its own scratch slot and the reduce
        // below sums them in the same order, so the result is unchanged.
        var cachedRoutes = Set<Int>()
        var cachedCB: MTLCommandBuffer?
        if let plannedFetch, plannedFetch.hits > 0, !plannedFetch.misses.isEmpty {
            let planned = try model.routedExpertBuffers(for: plannedFetch)
            let misses = Set(plannedFetch.misses)
            let cb = context.queue.makeCommandBuffer()!
            for route in 0..<config.topKExperts where !misses.contains(route) {
                try expertRuntime.encodeExpert(
                    commandBuffer: cb,
                    blob: planned[route],
                    offsets: views.expertOffsets,
                    input: normed,
                    queryIndex: 0,
                    routeSlot: route,
                    scratch: expertScratch,
                    swigluLimit: Float(config.swigluLimit))
                cachedRoutes.insert(route)
            }
            cb.commit()
            cachedCB = cb
            splitExpertLayers &+= 1
        }

        let ioStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let blobs: [TensorView]
        if let plannedFetch {
            recordDecodeExpertFetch(layer: layer, plan: plannedFetch)
            blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
        } else {
            blobs = try await model.fetchRoutedExperts(
                layer: layer, experts: experts)
        }
        fireLookahead()
        totalIoNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - ioStart

        let cb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let cb2 = context.queue.makeCommandBuffer()!
        for route in 0..<config.topKExperts where !cachedRoutes.contains(route) {
            try expertRuntime.encodeExpert(
                commandBuffer: cb2,
                blob: blobs[route],
                offsets: views.expertOffsets,
                input: normed,
                queryIndex: 0,
                routeSlot: route,
                scratch: expertScratch,
                swigluLimit: Float(config.swigluLimit))
        }
        try expertRuntime.encodeFloatResidualReduce(
            commandBuffer: cb2,
            scratch: expertScratch,
            residual: hidden,
            output: hidden,
            queryCount: 1)
        cb2.commit()
        try waitForCompletion(cb2)
        // Same queue, so it finished before cb2 started; this only surfaces
        // an error it hit.
        if let cachedCB { try checkCommandBufferError(cachedCB) }
        totalCb2Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cb2Start
    }

    private func encodeRoPE(commandBuffer: MTLCommandBuffer,
                            data: MTLBuffer,
                            dataOffset: Int,
                            position: Int,
                            heads: Int,
                            tokens: Int = 1) {
        let yarn = config.yarnRope!
        rope.encodeYaRNNeox(
            commandBuffer: commandBuffer,
            data: data,
            dataOffset: dataOffset,
            position: UInt32(position),
            headDim: UInt32(config.headDim),
            numHeads: UInt32(heads),
            numTokens: UInt32(tokens),
            theta: Float(config.ropeTheta),
            originalContextLength: UInt32(yarn.originalContextLength),
            scalingFactor: Float(yarn.scalingFactor),
            betaFast: Float(yarn.betaFast),
            betaSlow: Float(yarn.betaSlow))
    }

    private nonisolated func waitForCompletion(
        _ commandBuffer: MTLCommandBuffer
    ) throws {
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        commandBuffer.waitUntilCompleted()
        prefillClock.waited(commandBuffer, wallNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
        try checkCommandBufferError(commandBuffer)
    }
}
