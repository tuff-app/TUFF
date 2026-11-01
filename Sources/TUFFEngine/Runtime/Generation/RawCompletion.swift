import Foundation
import Metal

/// Streaming callbacks from `runRawCompletion`. `.prefill` reports monotonic
/// producer-defined prompt progress; scalar replay reports per token, while a
/// prefill-capable producer may report per internal chunk. `.token` fires per
/// decoded non-stop token; `.tail` carries the detokenizer flush remainder at a
/// stop boundary.
public enum RawDecodeProgress: Sendable {
    case prefill(done: Int, total: Int)
    case token(index: Int, id: Int32, delta: String)
    case tail(String)
}

public enum RawCompletionStart: Sendable, Equatable {
    case reset
    case resume(cachedPromptTokens: Int)
}

public struct RawDecodeResult: Sendable {
    public let prefillTokens: Int
    public let cachedPromptTokens: Int
    public let computedPrefillTokens: Int
    public let prefillSeconds: Double
    public let newTokens: Int
    public let decodeSeconds: Double
    public let reason: StopReason
    public let kvPosition: Int
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
    /// The part of `uncommittedBoundaryTokenIDs` that was never reported as
    /// `.token` progress: a stop token (EOS, EOT, tool response, Harmony
    /// `<|call|>`, or an extra stop token). A max-token, stop-string or
    /// cancelled boundary was already reported, so it is not listed here.
    /// A caller replaying the boundary into a structured decoder must use
    /// this list, or the decoder consumes the same token twice.
    public let undeliveredBoundaryTokenIDs: [Int32]
    public let speculative: SpeculativeDecodeMetrics
    /// The runner's checkpoints at the prompt positions the caller asked
    /// for, in order, when the runner can take them.
    public var prefixCheckpoints: [RunnerStateSnapshot] = []
    /// The runner's whole state at `stateSnapshotPosition`, when asked for,
    /// for a caller that keeps it beyond this process.
    public var stateSnapshot: RunnerStateSnapshot?
    /// Where this request's prompt reading spent its time, for a runner that
    /// times it by phase.
    public var prefillPhases: PrefillPhaseTimes?

    public init(prefillTokens: Int,
                cachedPromptTokens: Int,
                computedPrefillTokens: Int,
                prefillSeconds: Double,
                newTokens: Int,
                decodeSeconds: Double,
                reason: StopReason,
                kvPosition: Int,
                kvBackedTokenIDs: [Int32],
                uncommittedBoundaryTokenIDs: [Int32],
                undeliveredBoundaryTokenIDs: [Int32] = [],
                speculative: SpeculativeDecodeMetrics = .init()) {
        self.prefillTokens = prefillTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.computedPrefillTokens = computedPrefillTokens
        self.prefillSeconds = prefillSeconds
        self.newTokens = newTokens
        self.decodeSeconds = decodeSeconds
        self.reason = reason
        self.kvPosition = kvPosition
        self.kvBackedTokenIDs = kvBackedTokenIDs
        self.uncommittedBoundaryTokenIDs = uncommittedBoundaryTokenIDs
        self.undeliveredBoundaryTokenIDs = undeliveredBoundaryTokenIDs
        self.speculative = speculative
    }

    /// Compatibility overload retained for callers that do not consume
    /// speculative metrics.
    public init(prefillTokens: Int,
                cachedPromptTokens: Int,
                computedPrefillTokens: Int,
                prefillSeconds: Double,
                newTokens: Int,
                decodeSeconds: Double,
                reason: StopReason,
                kvPosition: Int,
                kvBackedTokenIDs: [Int32],
                uncommittedBoundaryTokenIDs: [Int32]) {
        self.init(prefillTokens: prefillTokens,
                  cachedPromptTokens: cachedPromptTokens,
                  computedPrefillTokens: computedPrefillTokens,
                  prefillSeconds: prefillSeconds,
                  newTokens: newTokens,
                  decodeSeconds: decodeSeconds,
                  reason: reason,
                  kvPosition: kvPosition,
                  kvBackedTokenIDs: kvBackedTokenIDs,
                  uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs,
                  speculative: .init())
    }
}

/// Preallocated per-generation buffers (two 512 KiB vocab buffers plus a token
/// slot) and sampler. A warm session reuses them for every token, avoiding
/// per-token Metal buffer allocation.
///
/// `@unchecked Sendable`: the buffers and sampler are exclusively owned by one
/// generation at a time — the single-in-flight guard upstream is the contract.
public struct RawCompletionScratch: @unchecked Sendable {
    let logits: MTLBuffer
    let probs: MTLBuffer
    let outToken: MTLBuffer
    let sampler: Sampler

    public init(context: MetalContext, vocab: Int, logitSoftcap: Float = 30.0) throws {
        guard let logits = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                     options: .storageModeShared),
              let probs = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                    options: .storageModeShared),
              let outToken = context.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                       options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logits = logits
        self.probs = probs
        self.outToken = outToken
        self.sampler = try Sampler(context: context, vocab: vocab,
                                   logitSoftcap: logitSoftcap)
    }
}

extension GenerationConfig {
    /// A pure-greedy config can use the fused head's GPU argmax
    /// (`RealForwardRunner.lastGreedyToken`) instead of sampling from the
    /// logits buffer. Anything else needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

}

protocol GreedyHeadReporting: Sendable {
    var usesFusedGreedyHead: Bool { get }
    var lastGreedyToken: UInt32 { get }
    /// Chooses the head for the next request. A runner loaded for the fused
    /// head writes logits while `pureGreedy` is false.
    func selectHead(pureGreedy: Bool)
}

#if !TUFF_ENGINE_GPT_OSS
extension RealForwardRunner: GreedyHeadReporting {}
#endif

/// Raw-completion prefill + decode loop shared by the CLI and the Mac app.
/// Consumes pre-encoded `promptIds` (BOS + verbatim encode upstream — no chat
/// template). Stop handling, detokenizer flush ordering, and history append
/// ordering are shared by both front ends.
///
/// When the producer runs the fused lm_head (`RealForwardRunner` default) the
/// logits buffer is never written and the loop reads `lastGreedyToken`. A
/// sampling config switches that runner to its logits head for the request.
public func runRawCompletion(producer: any LogitProducer,
                             tokenizer: GFTokenizer,
                             promptIds: [Int32],
                             multimodalInput: MultimodalPrefillInput? = nil,
                             config: GenerationConfig,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             start: RawCompletionStart = .reset,
                             draftProducer: (any DraftTokenProducer)? = nil,
                             prefixCheckpointPositions: [Int] = [],
                             stateSnapshotPosition: Int? = nil,
                             checkpointBudgetBytes: Int = .max,
                             shouldStop: () -> Bool = { false },
                             onProgress: (RawDecodeProgress) -> Void) async throws -> RawDecodeResult {
    try config.validate()
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    #if os(macOS)
    // Generation is user-requested work even when the decode helper has no
    // visible window. Keep App Nap from deferring it while a request is active;
    // still allow the user's normal system sleep policy. End on every exit,
    // including cancellation and errors, so idle runners remain idle.
    let activity = ProcessInfo.processInfo.beginActivity(
        options: .userInitiatedAllowingIdleSystemSleep,
        reason: "Generating a local model response")
    defer { ProcessInfo.processInfo.endActivity(activity) }
    #endif

    let fusedRunner = producer as? any GreedyHeadReporting
    fusedRunner?.selectHead(pureGreedy: config.isPureGreedy)
    let fusedGreedy = fusedRunner?.usesFusedGreedyHead == true
    guard !fusedGreedy || config.isPureGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }

    let cachedPromptTokens: Int
    switch start {
    case .reset:
        cachedPromptTokens = 0
    case .resume(let count):
        guard count > 0, count < promptIds.count else {
            throw GeneratorError.invalidContinuation(
                "cached prompt token count must be greater than zero and less than the effective prompt")
        }
        guard producer is any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        cachedPromptTokens = count
    }
    if let multimodalInput {
        // Under resume the multimodal input is the tail that still has to be
        // prefilled, so it must match the prompt suffix rather than the whole
        // prompt. `prefillMultimodal` already prefills from `startPosition`.
        guard multimodalInput.effectiveTokenIDs
            == Array(promptIds.dropFirst(cachedPromptTokens)) else {
            throw GeneratorError.invalidContinuation(
                "multimodal effective token IDs do not match the prompt")
        }
    }
    // Image spans are served only by the chunked prefill path, and a prefill
    // config that cannot serve them is a performance setting, not a decision to
    // drop the images.
    let prefillConfig = multimodalInput == nil
        ? prefillConfig
        : (prefillConfig.coercedForImagePrompt() ?? prefillConfig)
    let computedPrefillTokens = promptIds.count - cachedPromptTokens

    var detok = GFDetokenizer(tokenizer: tokenizer,
                              barrierTokenIDs: tokenizer.structuralMarkerIDs)
    var history = Array(promptIds.prefix(cachedPromptTokens))
    history.reserveCapacity(promptIds.count + config.maxNewTokens)

    if let context = producer as? any ContextWindowReporting,
       promptIds.count + config.maxNewTokens > context.maxContext {
        throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                             maxNew: config.maxNewTokens,
                                             maxContext: context.maxContext)
    }
    switch start {
    case .reset:
        producer.reset()
    case .resume:
        let continuable = producer as! any ContinuableLogitProducer
        try continuable.prepareForContinuation(expectedPosition: cachedPromptTokens)
    }
    let prefillStart = Date()
    let phaseReporter = producer as? any PrefillPhaseReporting
    let phasesBefore = phaseReporter?.prefillPhaseTimes
    var position = cachedPromptTokens
    var prefillSeed: PrefillSeed?
    let prefillTokens = promptIds[cachedPromptTokens...]

    // Checkpoints are taken as the prefill passes each requested position. A
    // checkpoint that cannot be taken only costs a later request its reuse.
    let checkpointing = (producer as? any PrefixCheckpointingRunner)
        .flatMap { $0.supportsPrefixCheckpoints ? $0 : nil }
    let checkpointPositions = checkpointing == nil || multimodalInput != nil ? [] : Array(Set(
        prefixCheckpointPositions.filter { $0 > cachedPromptTokens && $0 <= promptIds.count }
    )).sorted()
    var prefixCheckpoints: [RunnerStateSnapshot] = []
    var stateSnapshot: RunnerStateSnapshot?
    var remainingCheckpointBytes = max(0, checkpointBudgetBytes)
    func takeCheckpoint() {
        guard remainingCheckpointBytes > 0,
              let checkpointing,
              checkpointing.prefixCheckpointByteEstimate <= remainingCheckpointBytes else { return }
        if let snapshot = try? checkpointing.capturePrefixCheckpoint() {
            guard snapshot.byteCount <= remainingCheckpointBytes else { return }
            remainingCheckpointBytes -= snapshot.byteCount
            prefixCheckpoints.append(snapshot)
            if snapshot.position == stateSnapshotPosition {
                if let snapshotting = producer as? any StateSnapshottingRunner,
                   let bytes = snapshotting.stateSnapshotByteEstimate,
                   bytes <= remainingCheckpointBytes {
                    stateSnapshot = try? snapshotting.captureState()
                    remainingCheckpointBytes -= stateSnapshot?.byteCount ?? 0
                }
            }
        }
    }
    switch (multimodalInput, prefillConfig.mode) {
    case (.some(let input), .chunked) where producer is any MultimodalPrefillRunner:
        let multimodal = producer as! any MultimodalPrefillRunner
        let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
        let result = try await multimodal.prefillMultimodal(
            input: input,
            startPosition: position,
            outputMode: mode,
            config: prefillConfig,
            into: scratch.logits
        ) { done in
            // The suffix-local count plus what the KV already holds. Without the
            // offset a 300-token image turn on a 5,000-token KV showed 1 of 5,300.
            onProgress(.prefill(done: cachedPromptTokens + done,
                                total: promptIds.count))
        }
        if mode == .logits, result.seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion multimodal prefill requested logits but producer returned \(result.seed)")
        }
        if case .greedyToken = result.seed, !config.isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion multimodal prefill returned a greedy token for a sampling config")
        }
        position = result.newPosition
        prefillSeed = result.seed
        // Only the suffix: under resume the cached prefix is already in history,
        // and appending the whole prompt would duplicate it.
        history.append(contentsOf: prefillTokens)
    case (.some, _):
        // The coercion above leaves an image prompt in chunked mode, so the only
        // way here is a producer that cannot run image spans at all.
        throw PrefillError.chunkedUnsupported(
            "multimodal prefill requires a MultimodalPrefillRunner-backed runtime")
    case (.none, .chunked) where (producer as? any ChunkedPrefillRunner)?
        .supportsChunkedPrefill == true:
        let chunked = producer as! any ChunkedPrefillRunner
        let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
        // A checkpoint inside the prompt splits the prefill there, so the
        // runner is exactly at that position when it is taken.
        var segmentStart = cachedPromptTokens
        var segmentResult: PrefillResult?
        for segmentEnd in checkpointPositions.filter({ $0 < promptIds.count }) + [promptIds.count] {
            let offset = segmentStart
            segmentResult = try await chunked.prefillChunked(
                tokens: promptIds[segmentStart..<segmentEnd],
                startPosition: segmentStart,
                outputMode: mode,
                config: prefillConfig,
                into: scratch.logits) { done in
                onProgress(.prefill(done: offset + done, total: promptIds.count))
            }
            segmentStart = segmentEnd
            if segmentEnd < promptIds.count { takeCheckpoint() }
        }
        let result = segmentResult!
        if mode == .logits, result.seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill requested logits but producer returned \(result.seed)")
        }
        if case .greedyToken = result.seed, !config.isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill returned a greedy token for a sampling config")
        }
        position = result.newPosition
        prefillSeed = result.seed
        history.append(contentsOf: prefillTokens)
    case (.none, .chunked), (.none, .off):
        // A text prompt whose producer has no chunked path runs a token at a
        // time instead of refusing. Prefill mode is a performance choice — the
        // same reasoning `coercedForImagePrompt` applies in the other
        // direction — and a hyper-connection architecture has no chunked
        // kernels at all, so throwing here would mean the model cannot answer.
        for (index, t) in prefillTokens.enumerated() {
            try Task.checkCancellation()
            if index + 1 < prefillTokens.count,
               let skipping = producer as? any PrefillHeadSkippingRunner,
               skipping.supportsHeadSkippingPrefill {
                try await skipping.consumePrefillToken(t, position: position, scratchLogits: scratch.logits)
            } else {
                try await producer.produce(token: t, position: position, into: scratch.logits)
            }
            position += 1
            history.append(t)
            onProgress(.prefill(done: position, total: promptIds.count))
            if position < promptIds.count, checkpointPositions.contains(position) {
                takeCheckpoint()
            }
        }
    }

    // Sampling reads only what the token's own command buffer produced, so a
    // producer that can carry it there saves one submit-and-wait per token.
    // The repetition penalty rewrites the logits on the CPU between the head
    // and the sampler, so that configuration keeps its own command buffer.
    let epilogueProducer = config.repetitionPenalty == 1.0
        ? producer as? any EpilogueFusingLogitProducer
        : nil
    let targetCostReporter = producer as? any SpeculativeTargetCostReporting
    var fusedTokenReady = false
    let speculativeVerifier = producer as? any SpeculativeVerificationRunner
    let speculativeEnabled = config.speculative.isEnabled
        && config.isPureGreedy
        && multimodalInput == nil
        && draftProducer != nil
        && speculativeVerifier?.supportsSpeculativeVerification == true
    let speculativeUnavailableForAuto = config.speculative.mode == .auto
        && !speculativeEnabled
    var speculativeController = SpeculativeDecodeController(
        config: config.speculative)
    let speculativeDecoder = SpeculativeDecoder(
        maximumBlockTokens: speculativeController.maximumBlockTokens)
    if speculativeEnabled {
        draftProducer?.reset()
    }

    // The end of the prompt, before the first generated token touches the KV.
    if checkpointPositions.last == promptIds.count { takeCheckpoint() }

    let prefillPhases = phasesBefore.flatMap { before in
        phaseReporter.map { $0.prefillPhaseTimes - before }
    }
    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []
    var undeliveredBoundaryTokenIDs: [Int32] = []
    var speculativeRounds = 0
    var speculativeProposedTokens = 0
    var speculativeAcceptedTokens = 0
    var speculativeRejectedTokens = 0
    var speculativeCorrectionTokens = 0
    var speculativeVerificationBlocks = 0
    var speculativeVerificationTokens = 0
    var speculativeMinimumBlockTokens = 0
    var speculativeMaximumBlockTokens = 0
    var speculativeVerificationWallNanos: UInt64 = 0
    var speculativeVerificationCommandBuffers: UInt64 = 0
    var speculativeVerificationExpertReads: UInt64 = 0
    var speculativeVerificationExpertBytes: UInt64 = 0
    var speculativeVerificationExpertCacheHits: UInt64 = 0
    var speculativeVerificationExpertCacheMisses: UInt64 = 0
    var speculativeFastBoundaryChecks = 0
    var speculativeFastBoundaryRejects = 0
    var speculativeFastBoundaryCheckWallNanos: UInt64 = 0
    var speculativeDraftWallNanos: UInt64 = 0
    var speculativeFallbackDecodes = 0

    /// Apply exactly the existing visible-token semantics without advancing
    /// target state. A false result means the token is the uncommitted boundary
    /// for EOS/EOT/tool, stop string, cancellation, or max-token termination.
    func emitToken(_ tokenID: Int32) throws -> Bool {
        // A failed head or sampler must never feed an invalid ID into the
        // embedding lookup or publish it as reusable conversation state.
        guard tokenID >= 0, Int(tokenID) < scratch.sampler.vocab else {
            throw GeneratorError.invalidModelOutput
        }
        generated += 1
        uncommittedBoundaryTokenIDs = [tokenID]
        undeliveredBoundaryTokenIDs.removeAll(keepingCapacity: true)

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            // A stop token is never reported as `.token`; record that so a
            // caller can replay exactly this boundary and nothing else.
            undeliveredBoundaryTokenIDs = [tokenID]
            if tokenID == tokenizer.endOfTurnID {
                reason = .endOfTurn
            } else if tokenID == tokenizer.toolResponseID {
                reason = .toolCalls
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            return false
        }

        let delta = detok.push(tokenID)
        let visible = stopMatcher.push(delta)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))

        // Cancellation is not a stop-string match: reporting it as one made a
        // user pressing Stop indistinguishable from a configured stop string.
        let hitStopString = stopMatcher.isStopped
        let cancelled = !hitStopString && shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || cancelled || hitMax {
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = hitStopString ? .stopString : (cancelled ? .cancelled : .maxTokens)
            return false
        }

        history.append(tokenID)
        return true
    }

    /// Advance target state after a token has been made visible and committed
    /// to history. This is shared by the scalar path and the correction/bonus
    /// token after a speculative block.
    @discardableResult
    func advanceAfterCommittedToken(_ tokenID: Int32) async throws
        -> (wallNanos: UInt64, expertReads: UInt64, expertBytes: UInt64) {
        let advanceStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let expertReadsBefore = targetCostReporter?.totalRoutedExpertReads ?? 0
        let expertBytesBefore = targetCostReporter?.totalRoutedExpertBytes ?? 0
        prefillSeed = nil
        fusedTokenReady = false
        if !fusedGreedy, let epilogueProducer {
            let samplerPosition = generated
            fusedTokenReady = try await epilogueProducer.produce(
                token: tokenID, position: position, into: scratch.logits
            ) { commandBuffer in
                scratch.sampler.sample(commandBuffer: commandBuffer,
                                       logits: scratch.logits, probs: scratch.probs,
                                       history: history, config: config,
                                       position: samplerPosition,
                                       outToken: scratch.outToken)
            }
        } else {
            try await producer.produce(token: tokenID, position: position, into: scratch.logits)
        }
        position += 1
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
        let expertReadsAfter = targetCostReporter?.totalRoutedExpertReads ?? 0
        let expertBytesAfter = targetCostReporter?.totalRoutedExpertBytes ?? 0
        return (
            wallNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - advanceStart,
            expertReads: expertReadsAfter >= expertReadsBefore
                ? expertReadsAfter - expertReadsBefore : 0,
            expertBytes: expertBytesAfter >= expertBytesBefore
                ? expertBytesAfter - expertBytesBefore : 0)
    }

    func nextScalarToken() throws -> Int32 {
        if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                return Int32(bitPattern: token)
            case .logitsWritten:
                return try sampleOnce(scratch: scratch, context: context,
                                      history: history, config: config, position: generated)
            }
        } else if fusedGreedy {
            return Int32(bitPattern: fusedRunner!.lastGreedyToken)
        } else if fusedTokenReady {
            // The previous token's command buffer already ran the sampler.
            return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
        } else {
            return try sampleOnce(scratch: scratch, context: context,
                                  history: history, config: config, position: generated)
        }
    }

    generationLoop: while true {
        try Task.checkCancellation()

        if speculativeEnabled,
           speculativeController.isEnabled,
           generated < config.maxNewTokens,
           let draftProducer,
           let speculativeVerifier {
            let remaining = config.maxNewTokens - generated
            let blockTokens = speculativeController.blockSize(
                remainingTokens: remaining)
            guard blockTokens > 0 else { continue }
            let draftStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let proposal = try await draftProducer.propose(
                history: history,
                maxTokens: blockTokens,
                startPosition: position)
            let draftWallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                - draftStart
            speculativeDraftWallNanos &+= draftWallNanos
            try Task.checkCancellation()

            if !proposal.isEmpty {
                speculativeRounds += 1
                speculativeProposedTokens += proposal.count
                if speculativeMinimumBlockTokens == 0 {
                    speculativeMinimumBlockTokens = proposal.count
                } else {
                    speculativeMinimumBlockTokens = min(
                        speculativeMinimumBlockTokens, proposal.count)
                }
                speculativeMaximumBlockTokens = max(
                    speculativeMaximumBlockTokens, proposal.count)

                let boundaryCheckStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let boundaryToken = try await speculativeVerifier
                    .speculativeBoundaryToken()
                speculativeFastBoundaryCheckWallNanos &+=
                    clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - boundaryCheckStart
                if boundaryToken != nil {
                    speculativeFastBoundaryChecks += 1
                }
                if let boundaryToken,
                   proposal.tokenIDs[0] != boundaryToken {
                    // The target prediction at this boundary is already
                    // available. There is no reason to stream or write a
                    // speculative tail when the first candidate cannot be
                    // accepted. This is especially important for SSD-backed
                    // MoE targets, where a doomed block would otherwise pay
                    // for every routed expert before discovering the mismatch.
                    speculativeFastBoundaryRejects += 1
                    speculativeRejectedTokens += proposal.count
                    speculativeCorrectionTokens += 1

                    if try !emitToken(boundaryToken) {
                        speculativeController.record(
                            proposedTokens: proposal.count,
                            acceptedTokens: 0,
                            verificationWallNanos: 0,
                            draftWallNanos: draftWallNanos,
                            boundaryAdvanceNanos: nil)
                        break generationLoop
                    }
                    let boundaryAdvance = try await advanceAfterCommittedToken(
                        boundaryToken)
                    speculativeController.record(
                        proposedTokens: proposal.count,
                        acceptedTokens: 0,
                        verificationWallNanos: 0,
                        draftWallNanos: draftWallNanos,
                        boundaryAdvanceNanos: boundaryAdvance.wallNanos,
                        boundaryExpertBytes: boundaryAdvance.expertBytes)
                    continue
                }

                let roundStartPosition = position
                let plan: SpeculativeRoundPlan
                do {
                    plan = try await speculativeDecoder.verifyGreedyRound(
                        proposal: proposal,
                        startPosition: roundStartPosition,
                        verifier: speculativeVerifier,
                        into: scratch.logits)
                } catch {
                    // A cancellation or malformed verifier result must not
                    // strand a speculative tail in the target cache.
                    try? speculativeVerifier.commitSpeculativePrefix(0)
                    throw error
                }
                speculativeVerificationBlocks += 1
                speculativeVerificationTokens += plan.verification.processedTokens
                speculativeVerificationWallNanos &+= plan.verification.metrics.wallNanos
                speculativeVerificationCommandBuffers &+=
                    plan.verification.metrics.targetCommandBuffers
                speculativeVerificationExpertReads &+=
                    plan.verification.metrics.expertReads
                speculativeVerificationExpertBytes &+=
                    plan.verification.metrics.expertBytes
                speculativeVerificationExpertCacheHits &+=
                    plan.verification.metrics.expertCacheHits
                speculativeVerificationExpertCacheMisses &+=
                    plan.verification.metrics.expertCacheMisses
                speculativeAcceptedTokens += plan.decision.acceptedDraftCount
                speculativeRejectedTokens +=
                    proposal.count - plan.decision.acceptedDraftCount
                if plan.decision.firstMismatchIndex != nil {
                    speculativeCorrectionTokens += 1
                }

                var committedCandidateCount = 0
                do {
                    for index in 0..<plan.decision.acceptedDraftCount {
                        try Task.checkCancellation()
                        if try emitToken(proposal.tokenIDs[index]) {
                            committedCandidateCount += 1
                        } else {
                            // The boundary token was visible but not target-KV
                            // committed, exactly like scalar max/stop handling.
                            try speculativeVerifier.commitSpeculativePrefix(
                                committedCandidateCount)
                            position = roundStartPosition + committedCandidateCount
                            break generationLoop
                        }
                    }

                    try speculativeVerifier.commitSpeculativePrefix(
                        committedCandidateCount)
                    position = roundStartPosition + committedCandidateCount

                    // On mismatch this is the target correction. When every draft
                    // token matched it is the one-token bonus prediction.
                    let nextToken = plan.decision.nextTargetToken
                    if try !emitToken(nextToken) {
                        // The correction/bonus token is visible but is the
                        // uncommitted boundary for EOS, stop, cancellation,
                        // or max-token termination. The speculative tail was
                        // already committed to the accepted prefix above.
                        break generationLoop
                    }
                    let boundaryAdvance = try await advanceAfterCommittedToken(
                        nextToken)
                    speculativeController.record(
                        proposedTokens: proposal.count,
                        acceptedTokens: plan.decision.acceptedDraftCount,
                        verificationWallNanos: plan.verification.metrics.wallNanos,
                        draftWallNanos: draftWallNanos,
                        boundaryAdvanceNanos: boundaryAdvance.wallNanos,
                        verificationExpertBytes:
                            plan.verification.metrics.expertBytes,
                        boundaryExpertBytes: boundaryAdvance.expertBytes)
                    continue
                } catch {
                    // Tokens already emitted from the accepted prefix must stay
                    // backed by target KV even when cancellation arrives from a
                    // progress callback during the block.
                    try? speculativeVerifier.commitSpeculativePrefix(
                        committedCandidateCount)
                    position = roundStartPosition + committedCandidateCount
                    throw error
                }
            }
            speculativeFallbackDecodes += 1
        }

        let tokenID = try nextScalarToken()
        guard try emitToken(tokenID) else { break }
        _ = try await advanceAfterCommittedToken(tokenID)
    }

    var result = RawDecodeResult(prefillTokens: promptIds.count,
                           cachedPromptTokens: cachedPromptTokens,
                           computedPrefillTokens: computedPrefillTokens,
                           prefillSeconds: prefillSeconds,
                           newTokens: generated,
                           decodeSeconds: Date().timeIntervalSince(decodeStart),
                           reason: reason,
                           kvPosition: position,
                           kvBackedTokenIDs: history,
                           uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs,
                           undeliveredBoundaryTokenIDs: undeliveredBoundaryTokenIDs,
                           speculative: SpeculativeDecodeMetrics(
                               rounds: speculativeRounds,
                               proposedTokens: speculativeProposedTokens,
                               acceptedTokens: speculativeAcceptedTokens,
                               rejectedTokens: speculativeRejectedTokens,
                               correctionTokens: speculativeCorrectionTokens,
                               verificationBlocks: speculativeVerificationBlocks,
                               verificationTokens: speculativeVerificationTokens,
                               minimumVerificationBlockTokens:
                                   speculativeMinimumBlockTokens,
                               maximumVerificationBlockTokens:
                                   speculativeMaximumBlockTokens,
                               verificationWallNanos: speculativeVerificationWallNanos,
                               verificationCommandBuffers:
                                   speculativeVerificationCommandBuffers,
                               verificationExpertReads:
                                   speculativeVerificationExpertReads,
                               verificationExpertBytes:
                                   speculativeVerificationExpertBytes,
                               verificationExpertCacheHits:
                                   speculativeVerificationExpertCacheHits,
                               verificationExpertCacheMisses:
                                   speculativeVerificationExpertCacheMisses,
                               fastBoundaryChecks:
                                   speculativeFastBoundaryChecks,
                               fastBoundaryRejects:
                                   speculativeFastBoundaryRejects,
                               fastBoundaryCheckWallNanos:
                                   speculativeFastBoundaryCheckWallNanos,
                               draftWallNanos: speculativeDraftWallNanos,
                               normalFallbackDecodes: speculativeFallbackDecodes,
                               adaptiveDisabled: speculativeUnavailableForAuto
                                   || speculativeController.disabled))
    result.prefixCheckpoints = prefixCheckpoints
    result.prefillPhases = prefillPhases
    result.stateSnapshot = stateSnapshot
    return result
}

private func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                        history: [Int32], config: GenerationConfig, position: Int) throws -> Int32 {
    let cb = context.queue.makeCommandBuffer()!
    scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                           history: history, config: config, position: position,
                           outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    try checkCommandBufferError(cb)
    return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
}
