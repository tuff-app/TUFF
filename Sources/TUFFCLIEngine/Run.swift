@_exported import TUFFCLIArguments
import Foundation
import Metal
import TUFFEngine
import TUFFModelCatalog
import TUFFAppCore

/// One row of the `--messages-file` JSON.
private struct MessageJSON: Decodable {
    let role: String
    let content: MessageContentJSON
}

private enum MessageContentJSON: Decodable {
    case text(String)
    case parts([MessagePartJSON])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([MessagePartJSON].self))
        }
    }
}

private struct MessagePartJSON: Decodable {
    let type: String
    let text: String?
    let path: String?
}

enum PromptInput {
    case raw(String)
    case messages([GFTokenizer.Message])
    case multimodal(messages: [MultimodalMessage], images: [UUID: URL])

    /// Whether this prompt carries images. `--messages-file` reaches the image
    /// path with `args.images` empty, so the runtime checks that depend on
    /// images ask the parsed input, not the flag list.
    var hasImages: Bool {
        if case .multimodal = self { return true }
        return false
    }
}

/// `visionSupport` answers "can this Metal device run the image tower". It is a
/// parameter so the ordering below can be covered on a Mac that is not the one
/// the answer is about.
public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError,
                visionSupport: (MTLDevice) -> Bool = VisionRuntime.isSupported(on:)
) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        // What the Mac can do is settled before the model is opened. Reading
        // the manifest first meant an M1 asked for `--image` was told whatever
        // the model directory happened to fail on — a missing-file errno for a
        // run that this hardware was never going to be able to make.
        let input = try parseInput(args: args)
        var selectedDevice: MTLDevice?
        if input.hasImages {
            guard let device = MetalContext.makeSystemDefaultDevice() else {
                return errored(stderr, "no Metal device", 1)
            }
            try VisionRuntime.requireSupportedDevice(
                supportsApple8: visionSupport(device))
            selectedDevice = device
        }
        let architecture = try ManifestReader.resolveArchitecture(
            directoryURL: modelURL)
        let modelFamily = architecture.family
        if modelFamily == .gptOss, let policy = args.denseResidencyPolicy, policy != .resident {
            throw ArgsError.unsupported(flag: "--dense-residency", context: "GPT-OSS row-access projections require residency")
        }
        let effectiveThinking: ChatReasoning = modelFamily == .minimaxM2
            ? .on : args.thinking
        if modelFamily == .gptOss, args.thinking != .off {
            throw ArgsError.unsupported(
                flag: "--thinking", context: "GPT-OSS; use --reasoning")
        }
        if modelFamily != .gptOss, args.reasoningEffort != nil {
            throw ArgsError.unsupported(
                flag: "--reasoning", context: modelFamily.rawValue)
        }
        try validateImageSupport(
            hasImages: input.hasImages,
            family: modelFamily)
        let loadStarted = ContinuousClock.now
        // The shader compiler runs while the tokenizer and weights load.
        let context = try MetalContext(kernelGroups: [])
        context.prewarmKernels(forModelDirectory: modelURL, maxContext: args.maxContext,
                               includeVision: input.hasImages)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        let tokenizerLoaded = ContinuousClock.now
        let currentDate = HarmonyPromptRenderer.calendarDate()
        var promptIds: [Int32]
        var multimodalMessages: [MultimodalMessage]?
        var imageURLs: [UUID: URL] = [:]
        switch input {
        case .raw(let text):
            multimodalMessages = nil
            promptIds = tokenizer.encode(text, addBOS: true)
        case .messages(let messages):
            multimodalMessages = nil
            promptIds = try encodeTextChat(
                messages: messages,
                tokenizer: tokenizer,
                architecture: architecture,
                thinking: effectiveThinking,
                reasoningEffort: args.reasoningEffort,
                currentDate: currentDate)
        case .multimodal(let messages, let images):
            multimodalMessages = messages
            imageURLs = images
            // Filled in after the tower runs; the renderer needs the projected
            // features to lay out the placeholder spans.
            promptIds = []
        }
        if multimodalMessages == nil {
            guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        }
        guard multimodalMessages != nil || promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "the prompt is \(promptIds.count) tokens, which fills the \(args.maxContext)-token context; raise --max-context",
                2)
        }
        func makeConfig(maxNewTokens: Int) -> GenerationConfig {
            GenerationConfig(
                maxNewTokens: maxNewTokens,
                temperature: args.temperature,
                topK: args.topK,
                topP: args.topP,
                repetitionPenalty: args.repetitionPenalty,
                seed: args.seed,
                stopStrings: args.stops,
                extraStopTokens: [],
                speculative: SpeculativeDecodeConfig(
                    mode: args.speculativeMode,
                    draftTokens: args.speculativeDraftTokens))
        }
        // Hoisted above the `auto` estimate, which needs a device to read image
        // geometry. Planning never touches the GPU, but building the plan does
        // need the device the run will use.
        guard let device = selectedDevice ?? MetalContext.makeSystemDefaultDevice() else {
            return errored(stderr, "no Metal device", 1)
        }

        var effectiveArgs = args
        if args.prefillChunkTokensAuto, args.prefillPolicy != .off {
            let promptTokens = try estimatedPromptTokens(
                input: input, tokenizer: tokenizer, device: device,
                family: modelFamily,
                modelVariant: architecture.variant,
                reasoning: effectiveThinking,
                reasoningEffort: args.reasoningEffort,
                currentDate: currentDate)
            effectiveArgs.prefillChunkTokens =
                PrefillRuntimeConfig.autoChunkTokens(promptTokens: promptTokens,
                                                     cap: args.prefillChunkMax)
            if !args.quiet {
                let line = "[prefill chunk auto: "
                    + "\(effectiveArgs.prefillChunkTokens) tokens for a "
                    + "\(promptTokens)-token prompt]\n"
                stderr.write(Data(line.utf8))
            }
        }
        let args = effectiveArgs
        let runtime = try args.resolvedRuntimeConfiguration(
            forceLogitsHead: !makeConfig(maxNewTokens: args.maxNew).isPureGreedy,
            imagePrompt: input.hasImages,
            chunkedPrefillNeedsSlotFloor: modelFamily != .gptOss)

        if !args.quiet {
            let memory = TUFFModelCatalog.all.first {
                $0.architecture.id.rawValue == architecture.variant.rawValue
            }.map {
                InferenceMemoryPlan(descriptor: $0, config: architecture,
                    contextTokens: args.maxContext, expertCacheSlots: runtime.expertCacheSlots,
                    prefillChunkTokens: runtime.prefillChunkTokens)
            }
            let settings: [String: String] = [
                "context": String(args.maxContext),
                "expert_cache_slots": String(runtime.expertCacheSlots),
                "expert_cache_policy": runtime.expertCachePolicy.rawValue,
                "expert_lookahead": ProcessInfo.processInfo.environment["TUFF_EXPERT_LOOKAHEAD"] == "off" ? "off" : "auto",
                "small_block_prefill": SmallBlockPrefillPolicy.resolvedSetting(
                    environment: ProcessInfo.processInfo.environment,
                    variant: architecture.variant),
                "shared_expert_overlap": SharedExpertOverlapPolicy.resolvedSetting(
                    environment: ProcessInfo.processInfo.environment,
                    variant: architecture.variant),
                "prefill": runtime.prefillPolicy.rawValue,
                "prefill_chunk_tokens": String(runtime.prefillChunkTokens),
                "prefill_attention_path": runtime.prefillAttentionPath.rawValue,
                "head_path": runtime.headPath.rawValue,
                "model_integrity": "full-sha256",
                "rdadvise": runtime.rdadvisePolicy.rawValue,
                "temperature": String(args.temperature), "top_k": args.topK.map(String.init) ?? "off",
                "thinking": String(describing: effectiveThinking),
                "reasoning_effort": String(describing: args.reasoningEffort),
                "repetition_penalty": String(args.repetitionPenalty),
                "max_new_tokens": String(args.maxNew),
                "top_p": args.topP.map { String($0) } ?? "off", "seed": args.seed.map(String.init) ?? "random",
                "cache_tracking_metadata_reserve_bytes": memory.map { String($0.cacheTrackingMetadataReserveBytes) } ?? "unavailable",
                "catalog_runtime_allowance_bytes": memory.map { String($0.estimatedWorkingSetBytes) } ?? "unavailable"
            ]
            let json = try JSONSerialization.data(withJSONObject: settings, options: [.sortedKeys])
            stderr.write(Data("[resolved inference settings] ".utf8))
            stderr.write(json)
            stderr.write(Data("\n".utf8))
        }

        if !args.quiet,
           let notice = prefillCoercionNotice(
            hasImages: input.hasImages, config: runtime.prefillConfig) {
            stderr.write(Data((notice + "\n").utf8))
        }

        // An image prompt that cannot fit is decided by geometry, so decide it
        // before spending a model load and a GPU encode on it. The plan reads
        // metadata only, and the same plan is reused at encode time so each
        // image is opened and parsed once.
        var imagePlans: [UUID: VisionImagePlan] = [:]
        if let multimodalMessages {
            // Dictionary order depends on a per-process hash seed, so planning
            // straight from `imageURLs` named a different image on each run when
            // two were bad. The prompt's own part order is the stable one.
            let ordered = orderedImageIDs(messages: multimodalMessages, images: imageURLs)
            // The device from the guard above, not a fresh `MetalContext`:
            // building one compiles every shader module, and planning never
            // touches the GPU.
            let preprocessor = Gemma4ImagePreprocessor(
                device: device,
                config: VisionConfig(
                    family: modelFamily, modelVariant: architecture.variant))
            var projected = 0
            for id in ordered {
                guard let url = imageURLs[id] else { continue }
                let plan = try preprocessor.plan(fileURL: url)
                imagePlans[id] = plan
                projected += plan.geometry.softTokenCount
                    + VisionImageTokenBudget.markerTokensPerImage
            }
            // The text counts too. The tokenizer is already loaded, so this is
            // free and it closes the common case: a prompt whose text overflows
            // used to be refused only after the pack opened and every image had
            // been encoded. A lower bound - the template's framing tokens are
            // not counted - so it can only refuse what could never fit.
            for message in multimodalMessages {
                for part in message.content {
                    if case .text(let text) = part {
                        projected += tokenizer.encode(text, addBOS: false).count
                    }
                }
            }
            guard projected < args.maxContext else {
                return errored(
                    stderr,
                    "context overflow: prompt needs at least \(projected) tokens, "
                        + "which fills the \(args.maxContext)-token context; raise --max-context",
                    2)
            }
        }

        let densePolicy: DenseResidencyPolicy
        if let explicit = args.denseResidencyPolicy { densePolicy = explicit }
        else if architecture.feedForwardKind == .dense {
            guard let catalog = TUFFModelCatalog.all.first(where: { $0.architecture.id.rawValue == architecture.variant.rawValue }),
                  let selected = AppAutomaticMemoryPlanner.plan(for: AppModelInstallDescriptor(catalog: catalog),
                    on: .current(), profile: .speed, modelDirectory: modelURL, contextTokens: args.maxContext) else {
                return errored(stderr, "no safe automatic allocation at this context; lower --max-context", 1)
            }
            densePolicy = selected.denseResidencyPolicy
        } else { densePolicy = .resident }
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            denseResidency: densePolicy,
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        let weightsLoaded = ContinuousClock.now
        if !args.quiet {
            let estimate = AppAutomaticMemoryPlanner.estimatedAllocation(directory: modelURL,
                context: args.maxContext, options: AppRuntimeOptions(
                    expertCacheSlots: runtime.expertCacheSlots, prefillChunkTokens: runtime.prefillChunkTokens,
                    denseResidencyPolicy: model.stagesDenseLayers
                        ? .bounded(residentLayers: model.residentProjectionLayers) : .resident,
                    retainedConversationBytes: 0))
            if let estimate {
                let settings: [String: String] = [
                    "evidence": "calculated", "mode": estimate.mode.rawValue,
                    "resident_projection_layers": String(model.residentProjectionLayers.count),
                    "resident_weight_bytes": String(estimate.residentWeightBytes),
                    "state_bytes": String(estimate.stateBytes),
                    "estimated_peak_allocation_bytes": String(estimate.peakAllocatedBytes),
                    "streamed_projection_bytes_per_decode_token": String(estimate.streamedDenseBytesPerToken),
                ]
                stderr.write(Data("[model memory guidance] ".utf8))
                stderr.write(try JSONSerialization.data(withJSONObject: settings, options: [.sortedKeys]))
                stderr.write(Data("\n".utf8))
            }
        }
        let runner = try ModelForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let residentWarmup = model.startResidentWarmup()
        defer { residentWarmup?.cancel() }
        if ProcessInfo.processInfo.environment["TUFF_PHASES"] == "1" {
            let seconds = { (from: ContinuousClock.Instant, to: ContinuousClock.Instant) -> String in
                let d = from.duration(to: to)
                return String(format: "%.3f", Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18)
            }
            let runnerReady = ContinuousClock.now
            stderr.write(Data(("[load phases] tokenizer \(seconds(loadStarted, tokenizerLoaded))s, "
                + "weights \(seconds(tokenizerLoaded, weightsLoaded))s, "
                + "runner \(seconds(weightsLoaded, runnerReady))s, "
                + "total \(seconds(loadStarted, runnerReady))s\n").utf8))
        }
        if ProcessInfo.processInfo.environment["TUFF_KERNEL_BENCH"] == "1" {
            stderr.write(Data(try runner.benchmarkDecodeKernels().utf8))
            return RunResult(exitCode: 0)
        }
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let multimodalInput: MultimodalPrefillInput?
        if let multimodalMessages {
            let vision = try VisionRuntime.open(
                textModelURL: modelURL,
                context: context,
                visionPackURL: args.visionPack.map { URL(fileURLWithPath: $0) })
            var features: [UUID: VisionFeatures] = [:]
            features.reserveCapacity(imageURLs.count)
            let visionStarted = ContinuousClock.now
            for id in orderedImageIDs(messages: multimodalMessages, images: imageURLs) {
                guard let url = imageURLs[id] else { continue }
                try Task.checkCancellation()
                if let plan = imagePlans[id] {
                    features[id] = try vision.encodeImage(
                        plan: plan,
                        languageModel: model,
                        residencyPolicy: args.visionResidency,
                        checkCancellation: { try Task.checkCancellation() })
                } else {
                    features[id] = try vision.encodeImage(
                        at: url,
                        languageModel: model,
                        residencyPolicy: args.visionResidency,
                        checkCancellation: { try Task.checkCancellation() })
                }
            }
            if !args.quiet {
                let duration = visionStarted.duration(to: .now)
                let seconds = Double(duration.components.seconds)
                    + Double(duration.components.attoseconds) / 1e18
                stderr.write(Data(
                    "[vision images=\(imageURLs.count) encode=\(String(format: "%.3f", seconds))s]\n".utf8))
            }
            let rendered = try MultimodalPromptRenderer.render(
                messages: multimodalMessages,
                featuresByID: features,
                tokenizer: tokenizer,
                family: model.config.family,
                modelVariant: model.config.variant,
                reasoning: effectiveThinking)
            promptIds = rendered.effectiveTokenIDs
            multimodalInput = rendered
            guard promptIds.count < args.maxContext else {
                return errored(
                    stderr,
                    "the prompt is \(promptIds.count) tokens, which fills the \(args.maxContext)-token context; raise --max-context",
                    2)
            }
        } else {
            multimodalInput = nil
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = makeConfig(maxNewTokens: effectiveMaxNew)
        let draftProducer: (any DraftTokenProducer)? = args.speculativeMode != .off
            ? PromptLookupDraftTokenProducer()
            : nil
        let assistantDecoder = StructuredAssistantDecoder(
            tokenizer: tokenizer, allowedTools: [],
            promptOpensThinking: StructuredAssistantDecoder.promptOpensThinking(
                tokenizer: tokenizer, reasoning: effectiveThinking))
        var assistantDecodeError: Error?
        var lastWritten: Character?
        var thoughtTokens = false
        func writeAssistantEvents(_ events: [StructuredAssistantEvent]) {
            for event in events {
                switch event {
                case .content(let text) where !text.isEmpty:
                    stdout.write(Data(text.utf8))
                    lastWritten = text.last
                case .thinking(let text) where !text.isEmpty:
                    thoughtTokens = true
                default:
                    break
                }
            }
        }
        residentWarmup?.cancel()
        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            multimodalInput: multimodalInput,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig,
            draftProducer: draftProducer) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, let tokenID, let delta):
                    guard assistantDecodeError == nil else { break }
                    do {
                        writeAssistantEvents(try assistantDecoder.consume(
                            tokenID: tokenID, delta: delta))
                    } catch {
                        assistantDecodeError = error
                    }
                case .tail(let tail):
                    guard assistantDecodeError == nil else { break }
                    do {
                        writeAssistantEvents(
                            try assistantDecoder.consumeTail(tail))
                    } catch {
                        assistantDecodeError = error
                    }
                }
            }
        if let assistantDecodeError { throw assistantDecodeError }
        try assistantDecoder.finish()
        if lastWritten != nil, lastWritten != "\n", args.quiet {
            stdout.write(Data("\n".utf8))
        }
        // Said even with --quiet: an empty answer with exit code 0 otherwise
        // looks like the model had nothing to say.
        if case .maxTokens = stats.reason {
            if lastWritten == nil, thoughtTokens {
                stderr.write(Data(("[no answer: reasoning used all \(stats.newTokens) tokens; "
                    + "raise --max-new" + (modelFamily == .gptOss
                        ? " or use --reasoning low" : " or use --thinking off") + "]\n").utf8))
            } else if effectiveMaxNew < args.maxNew {
                stderr.write(Data(("[stopped at the context limit of \(args.maxContext) tokens; "
                    + "raise --max-context for a longer answer]\n").utf8))
            }
        }

        if let profiler = GPUOpProfiler.shared {
            stderr.write(Data(profiler.report().utf8))
        }
        if ProcessInfo.processInfo.environment["TUFF_PHASES"] == "1" {
            let ms = { (n: UInt64) in String(format: "%.1f", Double(n) / 1e6) }
            let total = stats.decodeSeconds * 1000
            var lines = "\n[phases over \(stats.newTokens) tokens, decode "
            lines += String(format: "%.0f", total) + " ms]\n"
            lines += "  cb1 encode+commit: " + ms(runner.totalCb1Nanos) + " ms\n"
            lines += "  expert io await:   " + ms(runner.totalIoNanos) + " ms\n"
            lines += "  cb2 encode+commit: " + ms(runner.totalCb2Nanos) + " ms\n"
            lines += "  exposed prefetch wait: " + ms(runner.exposedPrefetchWaitNanos) + " ms\n"
            lines += "  timings overlap; cumulative runner counters are not a wall-clock decomposition\n"
            lines += "  gpu layer cbs: " + ms(runner.totalGPULayerNanos) + " ms over "
            lines += "\(runner.totalGPULayerCommandBuffers) cbs\n"
            lines += "  gpu head cbs:  " + ms(runner.totalGPUHeadNanos) + " ms\n"
            lines += "  gpu routed cbs: " + ms(runner.totalGPURoutedNanos) + " ms\n"
            lines += "  gpu shared cbs: " + ms(runner.totalGPUSharedNanos) + " ms\n"
            lines += "  allocated expert slot bytes: \(model.expertCacheAllocatedBytes)\n"
            let reads = runner.expertReadMetrics
            lines += "  cache demand requests: \(reads.demandRequests), predictions: \(reads.predictionRequests)\n"
            lines += "  logical demand expert reads: \(reads.demandReads), bytes: \(reads.demandBytes), failures: \(reads.demandFailures)\n"
            lines += "  logical prefetch expert reads: \(reads.prefetchReads), bytes: \(reads.prefetchBytes), failures: \(reads.prefetchFailures)\n"
            lines += "  useful prefetched records: \(reads.usefulPrefetchReads), evicted unused: \(reads.unusedPrefetchEvictions)\n"
            lines += "  demand misses on experts a prediction pushed out: \(reads.pollutedDemandMisses)\n"
            lines += "  logical traffic includes prefill and OS-cache hits; physical SSD traffic is not measured\n"
            lines += "  routed expert cache hits: \(runner.totalRoutedExpertCacheHits)\n"
            lines += "  routed expert cache misses: \(runner.totalRoutedExpertCacheMisses)\n"
            if let phases = stats.prefillPhases {
                let s = { (value: Double) in String(format: "%.1f", value * 1000) }
                lines += "[prefill phases over \(stats.prefillTokens - stats.cachedPromptTokens) tokens, "
                    + String(format: "%.0f", stats.prefillSeconds * 1000) + " ms]\n"
                lines += "  attention gpu:      " + s(phases.attentionGPU) + " ms\n"
                lines += "  shared expert gpu:  " + s(phases.sharedExpertGPU) + " ms\n"
                lines += "  routed expert gpu:  " + s(phases.routedExpertGPU) + " ms\n"
                lines += "  other gpu:          " + s(phases.otherGPU) + " ms\n"
                lines += "  expert read wait:   " + s(phases.expertReadWait) + " ms\n"
                lines += "  waiting on gpu:     " + s(phases.gpuWait) + " ms\n"
                lines += "  gpu spans overlap the waits and each other; don't add them\n"
            }
            if let lookahead = runner.expertLookahead,
               lookahead.readsIssued > 0 || lookahead.precision > 0 {
                lines += "  expert lookahead: "
                    + String(format: "%.1f%% of predicted experts routed, ",
                             lookahead.precision * 100)
                    + "\(lookahead.readsIssued) prefetch batches"
                    + (lookahead.enabled ? "" : ", turned off") + "\n"
            }
            let pipelines = context.pipelineStatistics
            lines += "  pipelines created: \(pipelines.count) in "
                + String(format: "%.0f", pipelines.seconds * 1000) + " ms\n"
            stderr.write(Data(lines.utf8))
        }
        if let path = args.runReportPath,
           let data = try? JSONEncoder().encode(CLIRunReport(
            prefillTokens: stats.prefillTokens, newTokens: stats.newTokens,
            prefillSeconds: stats.prefillSeconds, decodeSeconds: stats.decodeSeconds)) {
            // Best effort: a missing report only means the run isn't learned from.
            try? data.write(to: URL(fileURLWithPath: path), options: .atomic)
        }
        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok/\(String(format: "%.2f", stats.prefillSeconds))s new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
            if args.speculativeMode != .off {
                let speculative = stats.speculative
                let rate = String(format: "%.3f", speculative.acceptanceRate)
                let verificationMs = String(
                    format: "%.2f",
                    Double(speculative.verificationWallNanos) / 1e6)
                let draftMs = String(
                    format: "%.2f",
                    Double(speculative.draftWallNanos) / 1e6)
                let boundaryMs = String(
                    format: "%.2f",
                    Double(speculative.fastBoundaryCheckWallNanos) / 1e6)
                let expertBytes = speculative.verificationExpertBytes
                let line = "[spec rounds=\(speculative.rounds) proposed=\(speculative.proposedTokens) "
                    + "accepted=\(speculative.acceptedTokens) acceptance=\(rate) "
                    + "rejected=\(speculative.rejectedTokens) corrections=\(speculative.correctionTokens) "
                    + "verifyTokens=\(speculative.verificationTokens) "
                    + "verifyMs=\(verificationMs) draftMs=\(draftMs) "
                    + "verifyReads=\(speculative.verificationExpertReads) "
                    + "verifyBytes=\(expertBytes) "
                    + "verifyCacheHits=\(speculative.verificationExpertCacheHits) "
                    + "verifyCacheMisses=\(speculative.verificationExpertCacheMisses) "
                    + "verifyCBs=\(speculative.verificationCommandBuffers) "
                    + "blockMin=\(speculative.minimumVerificationBlockTokens) "
                    + "blockMax=\(speculative.maximumVerificationBlockTokens) "
                    + "fallbacks=\(speculative.normalFallbackDecodes) "
                    + "boundaryMs=\(boundaryMs) "
                    + "boundaryChecks=\(speculative.fastBoundaryChecks) "
                    + "boundaryRejects=\(speculative.fastBoundaryRejects) "
                    + "autoDisabled=\(speculative.adaptiveDisabled)]\n"
                stderr.write(Data(line.utf8))
            }
        }
        return RunResult(exitCode: 0)
    } catch let error as ArgsError {
        return errored(stderr, "\(error)", 2)
    } catch let error as VisionImageError {
        return errored(stderr, "\(error)", 3)
    } catch let error as VisionPackError {
        return errored(stderr, "\(error)", 4)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

func validateImageSupport(
    hasImages: Bool,
    family: ModelFamily
) throws {
    guard !hasImages || family != .gptOss else {
        throw ArgsError.unsupported(
            flag: "image input",
            context: "GPT-OSS")
    }
}


/// The order the messages reference their images, so a repeated command plans
/// and encodes them the same way every run.
func orderedImageIDs(messages: [MultimodalMessage],
                     images: [UUID: URL]) -> [UUID] {
    var ordered: [UUID] = []
    var seen: Set<UUID> = []
    for message in messages {
        for part in message.content {
            if case .image(let id) = part, images[id] != nil, seen.insert(id).inserted {
                ordered.append(id)
            }
        }
    }
    for id in images.keys.sorted(by: { $0.uuidString < $1.uuidString })
    where !seen.contains(id) {
        ordered.append(id)
    }
    return ordered
}

/// The line to print when an image prompt overrides the requested prefill mode.
///
/// The override itself happens deep in `RawCompletion`, silently, while
/// `RUNTIME_CONTROLS.md` says `--prefill off` disables that path. A flag that is
/// ignored without a word is worse than one that is refused, so the CLI says it
/// before the model loads.

/// A prompt-length estimate good enough to choose a chunk size.
///
/// Deliberately cheap: it tokenises text and reads image geometry, and never
/// loads the model. Being a little low only costs one extra chunk.
func estimatedPromptTokens(
    input: PromptInput,
    tokenizer: GFTokenizer,
    device: MTLDevice,
    family: ModelFamily = .gemma4,
    modelVariant: ModelVariant? = nil,
    reasoning: ChatReasoning = .off,
    reasoningEffort: GPTOSSReasoningEffort? = nil,
    currentDate: String? = nil
) throws -> Int {
    switch input {
    case .raw(let text):
        return tokenizer.encode(text, addBOS: true).count
    case .messages(let messages):
        if family == .gptOss {
            return try tokenizer.encodeHarmonyChat(
                messages: messages,
                reasoningEffort: reasoningEffort ?? .medium,
                currentDate: currentDate ?? HarmonyPromptRenderer.calendarDate()).count
        }
        return tokenizer.encode(
            try tokenizer.applyChatTemplate(
                messages,
                modelVariant: modelVariant,
                reasoning: reasoning),
            addBOS: false).count
    case .multimodal(let messages, let images):
        var total = 0
        for message in messages {
            for part in message.content {
                if case .text(let text) = part {
                    total += tokenizer.encode(text, addBOS: false).count
                }
            }
        }
        let preprocessor = Gemma4ImagePreprocessor(
            device: device,
            config: VisionConfig(family: family, modelVariant: modelVariant))
        for url in images.values {
            // Not `try?`: an image that cannot be planned would count as zero
            // soft tokens, so `auto` would size the chunk for the text alone and
            // put the multimodal prompt back on the many-chunk path this flag
            // exists to avoid. An unplannable image fails the run anyway, and
            // failing here says why, before the load.
            total += try preprocessor.plan(fileURL: url).geometry.softTokenCount
            total += VisionImageTokenBudget.markerTokensPerImage
        }
        return total
    }
}

func encodeTextChat(
    messages: [GFTokenizer.Message],
    tokenizer: GFTokenizer,
    architecture: ArchConfig,
    thinking: ChatReasoning,
    reasoningEffort: GPTOSSReasoningEffort?,
    currentDate: String
) throws -> [Int32] {
    if architecture.family == .gptOss {
        return try tokenizer.encodeHarmonyChat(
            messages: messages,
            reasoningEffort: reasoningEffort ?? .medium,
            currentDate: currentDate)
    }
    return tokenizer.encode(
        try tokenizer.applyChatTemplate(
            messages,
            modelVariant: architecture.variant,
            reasoning: thinking),
        addBOS: false)
}

func prefillCoercionNotice(
    hasImages: Bool, config: PrefillRuntimeConfig
) -> String? {
    guard hasImages, config.coercedForImagePrompt() != nil else { return nil }
    return "[prefill coerced to chunked: image prompts require it]"
}

func parseInput(args: Args) throws -> PromptInput {
    if let prompt = args.prompt {
        return .raw(prompt)
    }
    if let chatPrompt = args.chatPrompt {
        let textSystemMessages = args.systemPrompt.map {
            [GFTokenizer.Message(role: .system, content: $0)]
        } ?? []
        guard !args.images.isEmpty else {
            return .messages(textSystemMessages + [
                GFTokenizer.Message(role: .user, content: chatPrompt)
            ])
        }
        var images: [UUID: URL] = [:]
        var content: [MultimodalContentPart] = []
        for path in args.images {
            let id = UUID()
            images[id] = URL(fileURLWithPath: path)
            content.append(.image(id: id))
        }
        if !chatPrompt.isEmpty { content.append(.text(chatPrompt)) }
        return .multimodal(
            messages: (args.systemPrompt.map {
                [MultimodalMessage(role: .system, content: [.text($0)])]
            } ?? []) + [MultimodalMessage(role: .user, content: content)],
            images: images)
    }
    guard let path = args.messagesFile else {
        throw ArgsError.modeMissing
    }
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
    let base = url.deletingLastPathComponent()
    var images: [UUID: URL] = [:]
    let messages: [MultimodalMessage] = try rows.map { row in
        guard let role = GFTokenizer.Role(rawValue: row.role) else {
            throw ArgsError.invalidValue(flag: "--messages-file", value: "unknown role: \(row.role)")
        }
        let parts: [MultimodalContentPart]
        switch row.content {
        case .text(let text):
            parts = [.text(text)]
        case .parts(let rows):
            parts = try rows.map { part in
                switch part.type {
                case "text":
                    guard let text = part.text else {
                        throw ArgsError.invalidValue(
                            flag: "--messages-file", value: "text part is missing text")
                    }
                    return .text(text)
                case "image_file":
                    guard role == .user, let path = part.path else {
                        throw ArgsError.invalidValue(
                            flag: "--messages-file", value: "image_file requires a user role and path")
                    }
                    let id = UUID()
                    // Test the string, not a URL: `URL(fileURLWithPath:)`
                    // resolves against the process directory, so its `path` is
                    // always absolute and the relative branch never ran. A
                    // relative path belongs to the messages file's directory.
                    images[id] = path.hasPrefix("/")
                        ? URL(fileURLWithPath: path)
                        : base.appendingPathComponent(path)
                    return .image(id: id)
                default:
                    throw ArgsError.invalidValue(
                        flag: "--messages-file", value: "unknown content type: \(part.type)")
                }
            }
        }
        return MultimodalMessage(role: role, content: parts)
    }
    if images.isEmpty {
        return .messages(messages.map { message in
            GFTokenizer.Message(
                role: message.role,
                content: message.content.compactMap {
                    if case .text(let text) = $0 { text } else { nil }
                }.joined())
        })
    }
    return .multimodal(messages: messages, images: images)
}

/// An image prompt's token count follows from geometry, so an oversized one can
/// be refused in milliseconds instead of after a model load and a GPU encode.
func validatePromptSize(
    tokens: Int,
    args: Args,
    stderr: FileHandle
) -> RunResult? {
    guard tokens + args.maxNew <= args.maxContext else {
        return errored(
            stderr,
            "context overflow: prompt \(tokens) + maxNew \(args.maxNew) exceeds maxContext \(args.maxContext)",
            2)
    }
    return nil
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
