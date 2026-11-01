import TUFFServerCore
import TUFFRuntimeManagement
import CryptoKit
import Foundation
import TUFFModelCatalog
import TUFFEngine

enum StructuredOutputFailureKind: String, Equatable, Sendable {
    case decoderConsume = "decoder_consume"
    case decoderFinish = "decoder_finish"
    case orphanToolResponse = "orphan_tool_response"
}

enum StructuredOutputFailureCause: String, Equatable, Sendable {
    case malformed
    case unknownTool = "unknown_tool"
    case oversized
    case unexpected
    case none

    static func classify(_ error: Error) -> Self {
        guard let parserError = error as? GemmaToolCallParserError else {
            return .unexpected
        }
        switch parserError {
        case .malformed: return .malformed
        case .unknownTool: return .unknownTool
        case .oversized: return .oversized
        }
    }
}

struct StructuredOutputFailureDiagnostics: Equatable, Sendable {
    let renderedPromptTokens: Int
    let effectivePromptTokens: Int
    let resultPromptTokens: Int
    let cachedPromptTokens: Int
    let computedPrefillTokens: Int
    let completionTokens: Int
    let maxCompletionTokens: Int
    let rawStop: String
    let kvPosition: Int
    let kvBackedTokens: Int
    let boundaryTokens: Int
    let decodedCalls: Int
    let visibleBytes: Int
    let stopStringMatched: Bool
    let toolStartCount: Int
    let toolEndCount: Int
    let toolResponseCount: Int
    let toolResponseEndCount: Int
    let lastToolStartOffset: Int
    let lastToolEndOffset: Int
    let lastToolResponseOffset: Int
    let lastToolResponseEndOffset: Int
    let effectiveCountMatchesResult: Bool
    let effectivePrefixMatchesKV: Bool
    let kvPositionMatchesHistory: Bool
    let completionCountMatchesHistory: Bool
    let prefillAccountingMatches: Bool
    let renderedPromptHash: String
    let effectivePromptHash: String
    let generatedHash: String

    init(
        renderedPromptIDs: [Int32],
        effectivePromptIDs: [Int32],
        result: RawDecodeResult,
        maxCompletionTokens: Int,
        decodedCalls: Int,
        visibleBytes: Int,
        stopStringMatched: Bool,
        toolStartID: Int32,
        toolEndID: Int32,
        toolResponseID: Int32,
        toolResponseEndID: Int32
    ) {
        let safePrefillCount = min(
            max(result.prefillTokens, 0),
            result.kvBackedTokenIDs.count)
        let committedGenerated = result.kvBackedTokenIDs.dropFirst(safePrefillCount)
        let boundary = result.uncommittedBoundaryTokenIDs[...]
        let generatedSegments = [committedGenerated, boundary]

        var toolStartCount = 0
        var toolEndCount = 0
        var toolResponseCount = 0
        var toolResponseEndCount = 0
        var lastToolStartOffset = -1
        var lastToolEndOffset = -1
        var lastToolResponseOffset = -1
        var lastToolResponseEndOffset = -1
        var offset = 0
        for segment in generatedSegments {
            for tokenID in segment {
                if tokenID == toolStartID {
                    toolStartCount += 1
                    lastToolStartOffset = offset
                }
                if tokenID == toolEndID {
                    toolEndCount += 1
                    lastToolEndOffset = offset
                }
                if tokenID == toolResponseID {
                    toolResponseCount += 1
                    lastToolResponseOffset = offset
                }
                if tokenID == toolResponseEndID {
                    toolResponseEndCount += 1
                    lastToolResponseEndOffset = offset
                }
                offset += 1
            }
        }

        let (prefillAccounted, prefillOverflow) = result.cachedPromptTokens
            .addingReportingOverflow(result.computedPrefillTokens)

        self.renderedPromptTokens = renderedPromptIDs.count
        self.effectivePromptTokens = effectivePromptIDs.count
        self.resultPromptTokens = result.prefillTokens
        self.cachedPromptTokens = result.cachedPromptTokens
        self.computedPrefillTokens = result.computedPrefillTokens
        self.completionTokens = result.newTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.rawStop = Self.rawStop(result.reason)
        self.kvPosition = result.kvPosition
        self.kvBackedTokens = result.kvBackedTokenIDs.count
        self.boundaryTokens = result.uncommittedBoundaryTokenIDs.count
        self.decodedCalls = decodedCalls
        self.visibleBytes = visibleBytes
        self.stopStringMatched = stopStringMatched
        self.toolStartCount = toolStartCount
        self.toolEndCount = toolEndCount
        self.toolResponseCount = toolResponseCount
        self.toolResponseEndCount = toolResponseEndCount
        self.lastToolStartOffset = lastToolStartOffset
        self.lastToolEndOffset = lastToolEndOffset
        self.lastToolResponseOffset = lastToolResponseOffset
        self.lastToolResponseEndOffset = lastToolResponseEndOffset
        self.effectiveCountMatchesResult = effectivePromptIDs.count == result.prefillTokens
        self.effectivePrefixMatchesKV = result.kvBackedTokenIDs.count >= effectivePromptIDs.count
            && result.kvBackedTokenIDs.prefix(effectivePromptIDs.count)
                .elementsEqual(effectivePromptIDs)
        self.kvPositionMatchesHistory = result.kvPosition == result.kvBackedTokenIDs.count
        self.completionCountMatchesHistory = offset == result.newTokens
        self.prefillAccountingMatches = !prefillOverflow
            && prefillAccounted == result.prefillTokens
        self.renderedPromptHash = Self.i32leSHA256([renderedPromptIDs[...]])
        self.effectivePromptHash = Self.i32leSHA256([effectivePromptIDs[...]])
        self.generatedHash = Self.i32leSHA256(generatedSegments)
    }

    static func i32leSHA256(_ segments: [ArraySlice<Int32>]) -> String {
        var hasher = SHA256()
        var bytes = Data()
        bytes.reserveCapacity(4_096)
        for segment in segments {
            for tokenID in segment {
                var littleEndian = UInt32(bitPattern: tokenID).littleEndian
                withUnsafeBytes(of: &littleEndian) {
                    bytes.append(contentsOf: $0)
                }
                if bytes.count == 4_096 {
                    hasher.update(data: bytes)
                    bytes.removeAll(keepingCapacity: true)
                }
            }
        }
        if !bytes.isEmpty { hasher.update(data: bytes) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    var logDescription: String {
        [
            "rendered_prompt_tokens=\(renderedPromptTokens)",
            "effective_prompt_tokens=\(effectivePromptTokens)",
            "result_prompt_tokens=\(resultPromptTokens)",
            "cached_prompt_tokens=\(cachedPromptTokens)",
            "computed_prefill_tokens=\(computedPrefillTokens)",
            "completion_tokens=\(completionTokens)",
            "max_completion_tokens=\(maxCompletionTokens)",
            "raw_stop=\(rawStop)",
            "kv_position=\(kvPosition)",
            "kv_backed_tokens=\(kvBackedTokens)",
            "boundary_tokens=\(boundaryTokens)",
            "decoded_calls=\(decodedCalls)",
            "visible_bytes=\(visibleBytes)",
            "stop_string_matched=\(stopStringMatched)",
            "tool_start_count=\(toolStartCount)",
            "tool_end_count=\(toolEndCount)",
            "tool_response_count=\(toolResponseCount)",
            "tool_response_end_count=\(toolResponseEndCount)",
            "last_tool_start_offset=\(lastToolStartOffset)",
            "last_tool_end_offset=\(lastToolEndOffset)",
            "last_tool_response_offset=\(lastToolResponseOffset)",
            "last_tool_response_end_offset=\(lastToolResponseEndOffset)",
            "effective_count_matches_result=\(effectiveCountMatchesResult)",
            "effective_prefix_matches_kv=\(effectivePrefixMatchesKV)",
            "kv_position_matches_history=\(kvPositionMatchesHistory)",
            "completion_count_matches_history=\(completionCountMatchesHistory)",
            "prefill_accounting_matches=\(prefillAccountingMatches)",
            "rendered_prompt_i32le_sha256=\(renderedPromptHash)",
            "effective_prompt_i32le_sha256=\(effectivePromptHash)",
            "generated_i32le_sha256=\(generatedHash)",
        ].joined(separator: " ")
    }

    private static func rawStop(_ reason: StopReason) -> String {
        switch reason {
        case .eos: "eos"
        case .endOfTurn: "end_of_turn"
        case .maxTokens: "max_tokens"
        case .stopString: "stop_string"
        case .cancelled: "cancelled"
        case .toolCalls: "tool_calls"
        }
    }
}

struct StructuredOutputFailure: Error, CustomDebugStringConvertible, Sendable {
    let kind: StructuredOutputFailureKind
    let cause: StructuredOutputFailureCause
    let diagnostics: StructuredOutputFailureDiagnostics

    var debugDescription: String {
        "structured_output_failure kind=\(kind.rawValue) "
            + "cause=\(cause.rawValue) \(diagnostics.logDescription)"
    }
}

/// How the server reads the images of a request: one plan per image, and the
/// encode reads the plan the count came from.
///
/// Planning opens the file, sniffs it, parses its metadata and — for a JPEG —
/// walks its markers. Counting with a plan and then letting `encodeImage(at:)`
/// re-plan internally did all of that a second time per image, and the two
/// opens saw two files: one rewritten in between produced a count the laid-out
/// span no longer matched, and the turn died on `placeholderMismatch` naming
/// nothing the client did.
///
/// Both multimodal paths go through here so the pattern has one implementation
/// to keep correct rather than one per path.
enum ServerRequestImages {
    /// An image planned once: the count that lays out its placeholder span,
    /// and — while the plan behind that count is still open — the bytes the
    /// count was taken from.
    struct Planned {
        let url: URL
        let softTokenCount: Int
        let plan: VisionImagePlan?
    }

    /// How many of a request's images keep their plan open at once.
    ///
    /// A plan owns the descriptor its ImageIO source reads through, and what
    /// bounds a request's image count is the context budget alone — an
    /// 8,192-token context admits some 2,700 single-token images. One
    /// descriptor each would exhaust the process table and take the listening
    /// socket down with it, so past this many the plan is released after
    /// counting and remade at encode time. Real requests carry a handful of
    /// images and never reach it.
    static let maximumOpenPlans = 32

    /// Plans every image before any of them is encoded.
    ///
    /// Two reasons for the order: the counts that lay out the placeholder spans
    /// then come from the same open files the encodes read, and a request whose
    /// last image cannot be read at all is refused before the tower has run on
    /// the images ahead of it.
    static func plans(
        for urls: [URL],
        with preprocessor: Gemma4ImagePreprocessor,
        maximumOpenPlans: Int = ServerRequestImages.maximumOpenPlans,
        checkCancellation: () throws -> Void = {}
    ) throws -> [Planned] {
        var planned: [Planned] = []
        planned.reserveCapacity(urls.count)
        for url in urls {
            // Planning a JPEG walks its whole scan, so an abandoned request
            // stops at the next image rather than reading all of them first.
            try checkCancellation()
            let plan = try preprocessor.plan(fileURL: url)
            planned.append(Planned(
                url: url,
                softTokenCount: plan.geometry.softTokenCount,
                plan: planned.count < maximumOpenPlans ? plan : nil))
        }
        return planned
    }

    /// Encodes an image from the plan its count was taken from. Only an image
    /// whose plan the open bound released is read a second time.
    static func encode<Features>(
        _ image: Planned,
        fromPlan: (VisionImagePlan) throws -> Features,
        byReopening: (URL) throws -> Features
    ) throws -> Features {
        guard let plan = image.plan else { return try byReopening(image.url) }
        return try fromPlan(plan)
    }

    /// Every image a full prefill needs, each read through one open file.
    ///
    /// A full prefill lays its spans out from the encoded features rather than
    /// from a count, so it asked for no plan of its own and let
    /// `encodeImage(at:)` make one per image internally — the third copy of the
    /// pattern, and the one that also encoded its way through the images ahead
    /// of one that could not be read at all.
    static func encodeAll<Features>(
        _ imageFiles: [UUID: URL],
        with preprocessor: Gemma4ImagePreprocessor,
        maximumOpenPlans: Int = ServerRequestImages.maximumOpenPlans,
        checkCancellation: () throws -> Void = {},
        encode: (Planned) throws -> Features
    ) throws -> [UUID: Features] {
        let entries = Array(imageFiles)
        let planned = try plans(
            for: entries.map { $0.value },
            with: preprocessor,
            maximumOpenPlans: maximumOpenPlans,
            checkCancellation: checkCancellation)
        var features: [UUID: Features] = [:]
        features.reserveCapacity(entries.count)
        for (entry, image) in zip(entries, planned) {
            try checkCancellation()
            features[entry.key] = try encode(image)
        }
        return features
    }
}

public actor ServerModelSession: ServerInferenceBackend {
    /// Chat dialect of the loaded tokenizer; drives request-validation rules.
    public nonisolated let chatDialect: ChatDialect
    public nonisolated let modelVariant: ModelVariant
    /// Registry-derived API model identifier used when --model-id is absent.
    public nonisolated var defaultModelID: String {
        registryModelID
    }
    private nonisolated let registryModelID: String

    private let context: MetalContext
    private let model: Model
    private let residentWarmup: ResidentWarmup?
    private let tokenizer: GFTokenizer
    private let runner: ModelForwardRunner
    private let scratch: RawCompletionScratch
    private let prefillConfig: PrefillRuntimeConfig
    private let maxContext: Int
    private let promptCacheMode: ServerPromptCacheMode
    private let promptCacheDomain: ServerPromptCacheDomain
    /// The conversation the runner holds plus any retained ones, within the
    /// retained-state budget the loader charged against the memory plan.
    private let conversations: ConversationStateStore
    /// Shared-prefix snapshots kept across restarts, or nil when off.
    private let prefixSnapshots: PrefixSnapshotDiskStore?
    public nonisolated let visionCapability: String
    private let visionRuntime: VisionRuntime?
    private let visionResidencyPolicy: VisionResidencyPolicy

    /// What the cached KV was actually produced by. Six configuration fields
    /// plus every `TUFF_*` variable in the environment: those select
    /// kernels - about nineteen vision switches alone - and one of them set to
    /// `0` silently picks a different attention path. Leaving them out meant a
    /// prefix built by one kernel set could be reused under another, with
    /// nothing recording that anything had changed.
    static func runtimeIdentityString(
        runtime: RuntimeConfiguration,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        let configuration = [
            GFTokenizer.toolChatTemplateIdentity,
            GFTokenizer.harmonyChatTemplateIdentity,
            GFTokenizer.minimaxChatTemplateIdentity,
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens),
            runtime.headPath.rawValue,
        ]
        let switches = environment
            .filter { $0.key.hasPrefix("TUFF_") }
            .map { "\($0.key)=\($0.value)" }
            .sorted()
        return (configuration + switches).joined(separator: ":")
    }

    static func registryModelID(
        for variant: ModelVariant,
        fallback: String
    ) -> String {
        if let checkpoint = TUFFModelCatalog.model(manifestModelID: fallback),
           checkpoint.architecture.id.rawValue == variant.rawValue {
            return checkpoint.apiModelID
        }
        return TUFFModelCatalog.all.first {
            $0.architecture.id.rawValue == variant.rawValue
        }?.apiModelID ?? fallback
    }

    static func assistantDecoder(
        tokenizer: GFTokenizer,
        tools: [GFTokenizer.FunctionDefinition],
        reasoning: ChatReasoning = .off
    ) -> StructuredAssistantDecoder {
        StructuredAssistantDecoder(
            tokenizer: tokenizer,
            allowedTools: Set(tools.map(\.name)),
            promptOpensThinking: StructuredAssistantDecoder.promptOpensThinking(
                tokenizer: tokenizer, reasoning: reasoning),
            parameterSchemas: Dictionary(tools.map { ($0.name, $0.parameters) },
                uniquingKeysWith: { first, _ in first }))
    }

    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            visionPackURL: URL? = nil,
                            visionResidencyPolicy: VisionResidencyPolicy = .onDemand,
                            promptCacheMode: ServerPromptCacheMode = .singlePrefix,
                            retainedConversationBytes: Int = 0,
                            runtimeConfiguration: RuntimeConfiguration,
                            context sharedContext: MetalContext? = nil,
                            integrityPolicy: ModelIntegrityPolicy = .fullSha256,
                            denseResidencyPolicy: DenseResidencyPolicy = .resident
    ) async throws -> ServerModelSession {
        let tokenizerFolder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw GFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            throw GFTokenizerError.missingToolTemplate
        }
        // A server that loads and unloads models on demand passes one context
        // in, so each load does not compile the shader library again. The
        // compiler runs while the tokenizer and weights load, and the tokenizer
        // is read while the weights load.
        let context = try sharedContext ?? MetalContext(kernelGroups: [])
        context.prewarmKernels(forModelDirectory: modelDirectory, maxContext: maxContext,
                               includeVision: VisionRuntime.isSupported(on: context.device))
        async let loadedTokenizer = GFTokenizer.load(from: tokenizerFolder)
        let runtime = runtimeConfiguration
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            denseResidency: denseResidencyPolicy,
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: integrityPolicy)
        let runner = try ModelForwardRunner(model: model,
                                           context: context,
                                           maxContext: maxContext,
                                           runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let tokenizer = try await loadedTokenizer
        let templateDigest = SHA256.hash(data: try Data(contentsOf: templateURL))
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = Self.runtimeIdentityString(runtime: runtime) + ":" + InferenceEngineEnvironment.computationIdentity
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: PrefillKVStorageMode.fp16.rawValue,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        let visionRuntime: VisionRuntime?
        let visionCapability: String
        // An explicit pack path is an operator's statement that the pack is
        // there; a typo must fail loudly at startup, not serve text with vision
        // silently missing.
        if let visionPackURL,
           !FileManager.default.fileExists(atPath: visionPackURL.path) {
            throw VisionPackError.packNotFound(visionPackURL.path)
        }
        // A model directory without the .gturbo suffix has no companion
        // location; that only means vision is missing, never that a
        // text-serving load must fail.
        let resolvedVisionPackURL = visionPackURL
            ?? (try? VisionPackLocation.companionURL(forTextModel: modelDirectory))
        if let hardwareCapability = Self.hardwareVisionCapability(
            supportsVisionRuntime: VisionRuntime.isSupported(on: context.device)) {
            visionRuntime = nil
            visionCapability = hardwareCapability
            ServerLog.visionRuntimeUnsupported()
        } else if let resolvedVisionPackURL,
           FileManager.default.fileExists(atPath: resolvedVisionPackURL.path) {
            do {
                visionRuntime = try VisionRuntime.open(
                    textModelURL: modelDirectory,
                    context: context,
                    visionPackURL: resolvedVisionPackURL)
                visionCapability = "ready"
            } catch {
                visionRuntime = nil
                visionCapability = Self.unavailableVisionCapability(for: error)
                if visionCapability == "unsupported" {
                    ServerLog.visionRuntimeUnsupported(
                        at: resolvedVisionPackURL, error: error)
                } else {
                    ServerLog.visionPackInvalid(at: resolvedVisionPackURL, error: error)
                }
            }
        } else {
            visionRuntime = nil
            visionCapability = "missing"
        }
        return ServerModelSession(context: context,
                                  model: model,
                                  tokenizer: tokenizer,
                                  runner: runner,
                                  scratch: scratch,
                                  prefillConfig: runtime.prefillConfig,
                                  maxContext: maxContext,
                                  promptCacheMode: promptCacheMode,
                                  promptCacheDomain: promptCacheDomain,
                                  retainedConversationBytes: retainedConversationBytes,
                                  visionRuntime: visionRuntime,
                                  visionCapability: visionCapability,
                                  visionResidencyPolicy: visionResidencyPolicy,
                                  prefixSnapshots: .standard(modelID: model.modelID,
                                                             buildIdentity: TUFFVersion.current))
    }

    static func hardwareVisionCapability(
        supportsVisionRuntime: Bool
    ) -> String? {
        supportsVisionRuntime ? nil : "unsupported"
    }

    static func unavailableVisionCapability(for error: Error) -> String {
        guard let runtimeError = error as? VisionRuntimeError else {
            return "invalid"
        }
        if case .unsupportedKernel = runtimeError {
            return "unsupported"
        }
        return "invalid"
    }

    private init(context: MetalContext,
                 model: Model,
                 tokenizer: GFTokenizer,
                 runner: ModelForwardRunner,
                 scratch: RawCompletionScratch,
                 prefillConfig: PrefillRuntimeConfig,
                 maxContext: Int,
                 promptCacheMode: ServerPromptCacheMode,
                 promptCacheDomain: ServerPromptCacheDomain,
                 retainedConversationBytes: Int,
                 visionRuntime: VisionRuntime?,
                 visionCapability: String,
                 visionResidencyPolicy: VisionResidencyPolicy,
                 prefixSnapshots: PrefixSnapshotDiskStore? = nil) {
        self.prefixSnapshots = promptCacheMode == .off ? nil : prefixSnapshots
        self.context = context
        self.model = model
        self.residentWarmup = model.startResidentWarmup()
        self.tokenizer = tokenizer
        self.chatDialect = tokenizer.dialect
        self.modelVariant = model.config.variant
        self.registryModelID = Self.registryModelID(
            for: model.config.variant,
            fallback: model.modelID)
        self.runner = runner
        self.scratch = scratch
        self.prefillConfig = prefillConfig
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheDomain = promptCacheDomain
        self.conversations = ConversationStateStore(
            budgetBytes: promptCacheMode == .off ? 0 : retainedConversationBytes)
        self.visionRuntime = visionRuntime
        self.visionResidencyPolicy = visionResidencyPolicy
        self.visionCapability = visionCapability
    }

    // An unload waits for a snapshot still being written, so an idle model
    // that is released right after reading a new system prompt keeps it.
    deinit { residentWarmup?.cancel(); prefixSnapshots?.flush() }

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let prepared = try await prepare(request)
        return try await generate(prepared, onEvent: onEvent)
    }

    public func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        residentWarmup?.cancel()
        var phases = ServerInferencePhaseTimings()
        var promptIDs: [Int32]?
        if !request.imageFiles.isEmpty {
            let imageStart = DispatchTime.now().uptimeNanoseconds
            let softTokens = try imageSoftTokenCounts(request)
            phases.record("image_admission", since: imageStart)
            let renderStart = DispatchTime.now().uptimeNanoseconds
            let rendered = try renderPrompt(request)
            phases.record("prompt_render_and_tokenization", since: renderStart)
            let budgetStart = DispatchTime.now().uptimeNanoseconds
            try validateImageTokenBudget(request, softTokens: softTokens, textTokens: rendered.count)
            phases.record("image_admission", since: budgetStart)
            if request.multimodalMessages == nil { promptIDs = rendered }
        } else if request.multimodalMessages == nil {
            let renderStart = DispatchTime.now().uptimeNanoseconds
            promptIDs = try renderPrompt(request)
            phases.record("prompt_render_and_tokenization", since: renderStart)
        }
        return ServerPreparedRequest(
            request: request,
            promptIDs: promptIDs, timingSeconds: phases.seconds)
    }

    private func imagePreprocessor(_ visionRuntime: VisionRuntime) -> Gemma4ImagePreprocessor {
        Gemma4ImagePreprocessor(device: context.device, config: visionRuntime.config)
    }

    /// The encode side of a planned image, bound to this session's runtime.
    private func encodeTurnImage(
        _ image: ServerRequestImages.Planned, visionRuntime: VisionRuntime
    ) throws -> VisionFeatures {
        try ServerRequestImages.encode(
            image,
            fromPlan: {
                try visionRuntime.encodeImage(
                    plan: $0,
                    languageModel: model,
                    residencyPolicy: visionResidencyPolicy,
                    checkCancellation: { try Task.checkCancellation() })
            },
            byReopening: {
                try visionRuntime.encodeImage(
                    at: $0,
                    languageModel: model,
                    residencyPolicy: visionResidencyPolicy,
                    checkCancellation: { try Task.checkCancellation() })
            })
    }

    /// Projected token count per image from headers alone; no pixel is decoded
    /// and no scan is walked.
    private func imageSoftTokenCounts(
        _ request: ValidatedChatRequest
    ) throws -> [Int] {
        guard let visionRuntime else {
            throw ServerRequestError.invalid(
                message: "image support is unavailable",
                param: "messages", code: "vision_unavailable")
        }
        let preprocessor = imagePreprocessor(visionRuntime)
        var counts: [Int] = []
        counts.reserveCapacity(request.imageFiles.count)
        for url in request.imageFiles.values {
            do {
                let geometry = try preprocessor.admissionGeometry(fileURL: url)
                counts.append(geometry.softTokenCount)
            } catch let error as ServerRequestError {
                throw error
            } catch {
                throw ServerRequestError.invalid(
                    message: "image could not be read: \(error)",
                    param: "messages", code: "invalid_image")
            }
        }
        return counts
    }

    /// Rejects an image request that cannot fit the context before any pixel is
    /// decoded or any tower runs.
    private func validateImageTokenBudget(
        _ request: ValidatedChatRequest, softTokens: [Int], textTokens: Int
    ) throws {
        guard ServerImageTokenBudget.fits(
            softTokenCounts: softTokens,
            textTokens: textTokens,
            maxContext: maxContext) else {
            throw ServerImageTokenBudget.rejection(
                imageCount: request.imageFiles.count,
                softTokenCounts: softTokens,
                textTokens: textTokens,
                maxContext: maxContext)
        }
    }

    /// The trailing user turn rendered as a continuation, with only its own
    /// images encoded. Token counts come from geometry, so the turn's shape is
    /// known before any encode.
    private func multimodalContinuation(
        request: ValidatedChatRequest,
        cachedMessageCount: Int
    ) async throws -> MultimodalPrefillInput {
        guard let messages = request.multimodalMessages, let visionRuntime,
              messages.count == cachedMessageCount + 2,
              let turn = messages.last, turn.role == .user,
              turn.toolCalls.isEmpty, turn.toolCallID == nil else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        var parts: [MultimodalContinuationPart] = []
        var urls: [URL] = []
        for part in turn.content {
            switch part {
            case .text(let value):
                parts.append(.text(value))
            case .image(let id):
                guard let url = request.imageFiles[id] else {
                    throw MultimodalPromptRendererError.missingImage(id)
                }
                parts.append(.image)
                urls.append(url)
            }
        }
        let images = try ServerRequestImages.plans(
            for: urls,
            with: imagePreprocessor(visionRuntime),
            checkCancellation: { try Task.checkCancellation() })
        let bridge = try tokenizer.encodeMultimodalUserContinuation(
            textAndImages: parts,
            imageTokenCounts: images.map(\.softTokenCount),
            family: model.config.family,
            modelVariant: model.config.variant,
            reasoning: request.reasoning)
        var spans: [MultimodalImageSpan] = []
        for (range, image) in zip(bridge.imageTokenRanges, images) {
            try Task.checkCancellation()
            let features = try encodeTurnImage(image, visionRuntime: visionRuntime)
            guard features.tokenCount == range.count else {
                // Geometry disagreed with the encode; refuse rather than inject
                // a span of the wrong length.
                throw MultimodalPromptRendererError.placeholderMismatch
            }
            spans.append(MultimodalImageSpan(tokenRange: range, features: features))
        }
        return try MultimodalPrefillInput(
            effectiveTokenIDs: bridge.effectiveTokenIDs,
            embeddingTokenIDs: bridge.embeddingTokenIDs,
            imageSpans: spans,
            family: model.config.family)
    }

    private func renderMultimodal(
        _ request: ValidatedChatRequest
    ) async throws -> MultimodalPrefillInput {
        guard let messages = request.multimodalMessages, let visionRuntime else {
            throw ServerRequestError.invalid(
                message: "image support is unavailable",
                param: "messages", code: "vision_unavailable")
        }
        do {
            let features = try ServerRequestImages.encodeAll(
                request.imageFiles,
                with: imagePreprocessor(visionRuntime),
                checkCancellation: { try Task.checkCancellation() },
                encode: { try encodeTurnImage($0, visionRuntime: visionRuntime) })
            return try MultimodalPromptRenderer.render(
                messages: messages,
                featuresByID: features,
                tokenizer: tokenizer,
                tools: request.tools,
                family: model.config.family,
                modelVariant: model.config.variant,
                reasoning: request.reasoning,
                preserveThinking: request.preserveThinking)
        } catch let error as MultimodalPromptRendererError {
            throw Self.clientError(for: error) ?? error
        }
    }

    /// Renderer failures a client can cause with its own message content are
    /// request rejections, not server errors; the rest stay opaque and surface
    /// as 500s because they mean the server's own layout went wrong.
    private static func clientError(
        for error: MultimodalPromptRendererError
    ) -> ServerRequestError? {
        switch error {
        case .reservedImageMarker:
            .invalid(
                message: "message text contains a reserved image placeholder token",
                param: "messages", code: "invalid_message")
        case .placeholderMismatch:
            .invalid(
                message: "message content could not be aligned with its images",
                param: "messages", code: "invalid_message")
        case .emptyMessages, .emptyContent:
            .invalid(
                message: "multimodal messages must not be empty",
                param: "messages", code: "invalid_message")
        case .missingImage, .unexpectedImage:
            nil
        }
    }

    public func generate(
        _ prepared: ServerPreparedRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        residentWarmup?.cancel()
        let request = prepared.request
        var phases = ServerInferencePhaseTimings(prepared.timingSeconds)
        var completionStarted = false
        var completed = false
        var plannedEntry: ServerPromptCacheEntry?
        var cancelCheckpoint: RunnerStateSnapshot?
        defer {
            // Rendering, planning and image encoding touch no KV state, so a
            // request that fails or is cancelled before the completion starts
            // leaves the cache exactly as the last completed request left it.
            // Image encoding is seconds long, so this window is not rare.
            if completionStarted, !completed {
                // A request that continued a conversation goes back to where
                // it started, so the conversation survives a client giving
                // up mid-answer; anything else starts over.
                if let checkpoint = cancelCheckpoint, let entry = plannedEntry,
                   (try? runner.rewind(to: checkpoint)) != nil {
                    conversations.restoreAfterCancel(entry, runnerPosition: runner.continuationPosition)
                } else {
                    conversations.invalidateActive()
                    runner.reset()
                }
            }
        }
        let needsToolTemplate = usesToolTemplate(request)
        let isMultimodal = request.multimodalMessages != nil
        if isMultimodal, visionRuntime == nil {
            throw ServerRequestError.invalid(
                message: "image support is unavailable",
                param: "messages", code: "vision_unavailable")
        }

        // Matched before any image is encoded. A multimodal hit continues from
        // the cached KV, so the images already inside that prefix are never
        // re-encoded; only a full-prefill miss needs them.
        let renderedTextIDs: [Int32]?
        if isMultimodal {
            renderedTextIDs = nil
        } else if let promptIDs = prepared.promptIDs {
            renderedTextIDs = promptIDs
        } else {
            let renderStart = DispatchTime.now().uptimeNanoseconds
            renderedTextIDs = try renderPrompt(request)
            phases.record("prompt_render_and_tokenization", since: renderStart)
        }
        var cacheMatch: ServerPromptCacheMatch = .miss
        if promptCacheMode == .singlePrefix {
            let planStart = DispatchTime.now().uptimeNanoseconds
            let plan = conversations.plan(
                domain: promptCacheDomain,
                transcript: request.conversationTranscript,
                renderedPromptIDs: renderedTextIDs,
                tokenizer: tokenizer,
                modelVariant: model.config.variant,
                conversationKey: request.promptCacheKey,
                runner: runner,
                allowsTextBridge: request.allowsTextBridge)
            phases.record("cache_plan", since: planStart)
            phases.recordCachePlan(conversations.statistics)
            cacheMatch = plan.match
            plannedEntry = plan.match.isHit ? conversations.active : nil
            if plan.match.missReason == .bridgeRenderFailed {
                ServerLog.promptCacheBridgeFailed(error: GFTokenizerError.invalidChatTemplate(
                    "tool-result bridge failed to encode"))
            }
        }
        // Nothing in memory matched: a system prompt read by an earlier
        // process may still be on disk.
        var diskCheckpoint: ConversationPrefixCheckpoint?
        if conversations.budgetBytes > 0, !cacheMatch.isHit, let store = prefixSnapshots, let renderedTextIDs,
           runner.supportsPrefixCheckpoints,
           let instructions = instructionPrefixLength(request: request, promptIDs: renderedTextIDs) {
            let tokens = Array(renderedTextIDs.prefix(instructions))
            let loadStart = DispatchTime.now().uptimeNanoseconds
            if let snapshot = store.load(tokens: tokens, domain: promptCacheDomain,
                                         runner: runner, device: context.device) {
                do {
                    try runner.restoreState(snapshot)
                    diskCheckpoint = ConversationPrefixCheckpoint(
                        tokenIDs: tokens, snapshot: try runner.capturePrefixCheckpoint())
                    cacheMatch = .hit(effectivePromptIDs: renderedTextIDs,
                                      cachedPromptTokens: instructions)
                } catch {
                    // The runner reset itself; read the prompt from scratch.
                    diskCheckpoint = nil
                }
            }
            phases.record("prefix_snapshot_load", since: loadStart)
        }
        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        var multimodalInput: MultimodalPrefillInput?
        let multimodalStart = DispatchTime.now().uptimeNanoseconds
        switch cacheMatch {
        case .hit(let effective, let cached):
            effectivePromptIDs = effective
            completionStart = .resume(cachedPromptTokens: cached)
            multimodalInput = nil
        case .renderThenResume(let cached):
            // Build the new turn as a continuation on the cached KV rather than
            // re-rendering history: the KV holds tokens the model generated, and
            // a fresh render re-tokenises that assistant text, so the two cannot
            // be compared. Only this turn's images are encoded.
            do {
                let bridge = try await multimodalContinuation(
                    request: request,
                    cachedMessageCount: conversations.active?.inputMessages.count ?? 0)
                effectivePromptIDs = (conversations.active?.kvBackedTokenIDs ?? [])
                    + bridge.effectiveTokenIDs
                multimodalInput = bridge
                completionStart = .resume(cachedPromptTokens: cached)
            } catch is CancellationError {
                // A disconnect mid-plan or mid-encode is not a bridge failure:
                // the cached prefix is untouched and stays valid for the retry.
                throw CancellationError()
            } catch {
                ServerLog.promptCacheBridgeFailed(error: error)
                conversations.invalidateActive()
                let rendered = try await renderMultimodal(request)
                effectivePromptIDs = rendered.effectiveTokenIDs
                multimodalInput = rendered
                completionStart = .reset
            }
        case .miss:
            if let renderedTextIDs {
                effectivePromptIDs = renderedTextIDs
                multimodalInput = nil
            } else {
                let rendered = try await renderMultimodal(request)
                effectivePromptIDs = rendered.effectiveTokenIDs
                multimodalInput = rendered
            }
            completionStart = .reset
        }
        if multimodalInput != nil {
            phases.record("multimodal_render_and_encode", since: multimodalStart)
        }
        // What the template renders versus what actually prefills: on a text
        // cache hit the two genuinely differ, and that difference is the point
        // of the rendered/effective split in the failure diagnostics. A
        // multimodal resume has no full render to compare against, so the
        // effective ids stand in.
        let renderedPromptIDs = renderedTextIDs ?? effectivePromptIDs
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        config.stopStrings = []

        // Reasoning channels are part of ordinary text generation too. The v1
        // path only needed structured decoding when tools were present because
        // thinking-off prompts started after an empty thought channel. Native
        // thinking opens a real thought channel, so every response must pass
        // through the same channel-aware decoder.
        let decoder = Self.assistantDecoder(
            tokenizer: tokenizer, tools: request.tools,
            reasoning: request.reasoning)

        let checkpointPositions = promptCacheMode == .singlePrefix && multimodalInput == nil
            ? checkpointPositions(request: request, effectivePromptIDs: effectivePromptIDs)
            : []
        // A system prompt read for the first time is also written to disk,
        // once, so a later process can skip it.
        let diskSnapshotPosition = prefixSnapshots.flatMap { store in
            instructionPrefixLength(request: request, promptIDs: effectivePromptIDs).flatMap {
                !store.contains(tokens: Array(effectivePromptIDs.prefix($0)),
                                domain: promptCacheDomain) ? $0 : nil
            }
        }

        // Where a continuation starts, so a cancelled request can return here.
        // Taken only when the copy fits the conversation budget.
        if case .resume = completionStart, plannedEntry != nil, runner.supportsPrefixCheckpoints,
           runner.prefixCheckpointByteEstimate <= max(0, conversations.budgetBytes
                - conversations.retainedBytes - conversations.activeCheckpointBytes) {
            cancelCheckpoint = try? runner.capturePrefixCheckpoint()
        }

        completionStarted = true
        let decoded = try await Self.decodeStructuredCompletion(
            producer: runner,
            tokenizer: tokenizer,
            decoder: decoder,
            renderedPromptIDs: renderedPromptIDs,
            effectivePromptIDs: effectivePromptIDs,
            multimodalInput: multimodalInput,
            config: config,
            stopStrings: request.generationConfig.stopStrings,
            needsToolTemplate: needsToolTemplate,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: completionStart,
            prefixCheckpointPositions: checkpointPositions,
            stateSnapshotPosition: diskSnapshotPosition,
            checkpointBudgetBytes: max(0, conversations.budgetBytes - conversations.retainedBytes
                - conversations.activeCheckpointBytes - (cancelCheckpoint?.byteCount ?? 0)
                - (diskCheckpoint?.snapshot.byteCount ?? 0)),
            onEvent: onEvent)
        let result = decoded.result
        if let snapshot = result.stateSnapshot, let store = prefixSnapshots {
            store.save(snapshot, tokens: Array(effectivePromptIDs.prefix(snapshot.position)),
                       domain: promptCacheDomain)
        }
        let reason: String
        if !decoded.calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        // Publishing an image turn is safe now that the entry carries per-message
        // image identity: a later turn whose images differ is refused by
        // `imagesDiverged` rather than resumed onto a KV built from a different
        // picture.
        if promptCacheMode == .singlePrefix {
            conversations.publish(ConversationCache.entry(
                domain: promptCacheDomain,
                transcript: request.conversationTranscript,
                content: decoded.content,
                thinking: decoded.reasoning,
                calls: decoded.calls,
                result: result,
                stopStringFiltered: decoded.stopStringFiltered,
                conversationKey: request.promptCacheKey,
                prefixCheckpoints: result.prefixCheckpoints.compactMap {
                    ConversationPrefixCheckpoint(
                        tokenIDs: Array(effectivePromptIDs.prefix($0.position)), snapshot: $0)
                } + (plannedEntry?.prefixCheckpoints ?? []) + (diskCheckpoint.map { [$0] } ?? [])))
        } else {
            conversations.invalidateActive()
        }
        completed = true
        var timingSeconds = phases.seconds
        timingSeconds["prefill"] = result.prefillSeconds
        timingSeconds["decode"] = result.decodeSeconds
        if let phases = result.prefillPhases {
            timingSeconds.merge(phases.timingSeconds, uniquingKeysWith: { _, new in new })
        }
        return ServerCompletion(
            content: decoded.content,
            reasoning: decoded.reasoning,
            toolCalls: decoded.calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: result.prefillTokens,
                               completionTokens: result.newTokens,
                               totalTokens: result.prefillTokens + result.newTokens,
                               cachedTokens: result.cachedPromptTokens,
                               reasoningTokens: decoded.reasoning.isEmpty
                                ? nil : min(decoded.reasoningTokens, result.newTokens)),
            timingSeconds: timingSeconds)
    }

    /// Where a request takes checkpoints, for a runner that can return to
    /// one. Where the instructions and tools end lets a new conversation with
    /// the same system prompt skip them; an agent's is often thousands of
    /// tokens. Where a user turn's prompt ends lets the next message resume
    /// there when the template rewrites the finished turn, as Harmony always
    /// does and Gemma and Qwen do with thinking on. Tool rounds keep both.
    /// Each costs a copy of the runner's sliding-window rows, which the
    /// conversation store charges against its memory budget.
    func checkpointPositions(request: ValidatedChatRequest,
                             effectivePromptIDs: [Int32]) -> [Int] {
        var positions: [Int] = []
        if let instructions = instructionPrefixLength(request: request,
                                                      promptIDs: effectivePromptIDs) {
            positions.append(instructions)
        }
        // Without that rewrite a follow-up already continues from the end of
        // the KV. The copy is still what a retry of the same request, or a
        // changed answer to it, goes back to: those render the conversation
        // only up to here, short of the KV's end. So it is kept for prompts
        // long enough that reading them again costs more than the copy.
        // One short of the prompt's end at most: a retry renders exactly this
        // prompt, and a checkpoint is only used with at least one token left
        // to read for the next token's logits.
        if request.messages.last?.role == .user,
           let agreeing = agreeingTokens(
            with: effectivePromptIDs, request: request,
            probe: request.messages + [.init(role: .assistant, content: Self.probeText)]),
           case let turn = min(agreeing, effectivePromptIDs.count - 1),
           turn > (positions.last ?? 0),
           !request.allowsTextBridge || turn >= Self.minimumSharedPrefixTokens {
            positions.append(turn)
        }
        return positions
    }

    /// Where the instructions and tools end in `promptIDs`: everything a new
    /// conversation with the same system prompt also starts with. Nil when
    /// there are none, or too few to be worth a checkpoint.
    func instructionPrefixLength(request: ValidatedChatRequest, promptIDs: [Int32]) -> Int? {
        let leading = request.messages.prefix { $0.role == .system || $0.role == .developer }
        guard !leading.isEmpty || !request.tools.isEmpty,
              let instructions = agreeingTokens(
                with: promptIDs, request: request,
                probe: Array(leading) + [.init(role: .user, content: Self.probeText)]),
              instructions >= Self.minimumSharedPrefixTokens,
              promptIDs.count > instructions else { return nil }
        return instructions
    }

    /// How many of the prompt's tokens a render of `probe` shares. With a
    /// placeholder user message after the instructions, that is everything a
    /// new conversation with the same system prompt and tools also starts
    /// with. With a placeholder answer after the last user message, it is
    /// everything the next request renders the same way, which excludes an
    /// opening such as Qwen's `<think>` that a finished turn drops.
    private func agreeingTokens(with prompt: [Int32], request: ValidatedChatRequest,
                                probe: [GFTokenizer.Message]) -> Int? {
        guard let rendered = try? renderPromptIDs(messages: probe, request: request) else {
            return nil
        }
        return zip(rendered, prompt).prefix { $0 == $1 }.count
    }

    /// Text no real message starts with, so a probe render stops agreeing
    /// with the prompt exactly where the probe message's content begins.
    static let probeText = "\u{1}"

    /// Shorter instructions cost less to read than the checkpoint saves.
    static let minimumSharedPrefixTokens = 256

    private func renderPrompt(_ request: ValidatedChatRequest) throws -> [Int32] {
        let promptIDs = try renderPromptIDs(messages: request.messages, request: request)
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return promptIDs
    }

    /// `messages` rendered with `request`'s tools, template and settings.
    private func renderPromptIDs(messages: [GFTokenizer.Message],
                                 request: ValidatedChatRequest) throws -> [Int32] {
        let promptIDs: [Int32]
        if chatDialect == .harmony {
            guard let reasoningEffort = request.reasoningEffort,
                  let currentDate = request.harmonyCurrentDate else {
                throw ServerRequestError.invalid(
                    message: "Harmony reasoning metadata is missing",
                    param: "reasoning_effort",
                    code: "invalid_request")
            }
            promptIDs = try tokenizer.encodeHarmonyChat(
                messages: messages,
                tools: request.tools,
                reasoningEffort: reasoningEffort,
                currentDate: currentDate)
        } else if usesToolTemplate(request) {
            promptIDs = try tokenizer.encodeToolChat(
                messages: messages,
                tools: request.tools,
                reasoning: request.reasoning,
                preserveThinking: request.preserveThinking)
        } else {
            let rendered = try tokenizer.applyChatTemplate(
                messages,
                modelVariant: model.config.variant,
                reasoning: request.reasoning,
                preserveThinking: request.preserveThinking)
            promptIDs = tokenizer.encode(rendered, addBOS: false)
        }
        return promptIDs
    }

    private func usesToolTemplate(_ request: ValidatedChatRequest) -> Bool {
        !request.tools.isEmpty || request.messages.contains {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }
    }
}

/// What one structured completion produced: the raw decode result plus the
/// visible content and tool calls the assistant decoder accepted.
struct ServerStructuredDecodeOutcome: Sendable {
    let result: RawDecodeResult
    let content: String
    var reasoning: String = ""
    var reasoningTokens: Int = 0
    let calls: [ParsedToolCall]
    let stopStringFiltered: Bool
}

extension ServerModelSession {
    /// Runs the raw completion loop through the structured assistant decoder.
    /// This is the server's production decode path; it is static so tests can
    /// drive it with a scripted producer and a real tokenizer.
    static func decodeStructuredCompletion(
        producer: any LogitProducer,
        tokenizer: GFTokenizer,
        decoder: StructuredAssistantDecoder,
        renderedPromptIDs: [Int32],
        effectivePromptIDs: [Int32],
        multimodalInput: MultimodalPrefillInput?,
        config: GenerationConfig,
        stopStrings: [String],
        needsToolTemplate: Bool,
        context: MetalContext,
        scratch: RawCompletionScratch,
        prefillConfig: PrefillRuntimeConfig,
        start: RawCompletionStart,
        prefixCheckpointPositions: [Int] = [],
        stateSnapshotPosition: Int? = nil,
        checkpointBudgetBytes: Int = .max,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerStructuredDecodeOutcome {
        let state = ServerDecodeState(
            stopMatcher: StreamingStopMatcher(stops: stopStrings))

        let handle: @Sendable ([StructuredAssistantEvent]) -> Void = { events in
            for event in events {
                switch event {
                case .content(let text):
                    var visible = state.stopMatcher.push(text)
                    // Qwen opens its answer with blank lines after reasoning;
                    // they reached clients as the start of `content`.
                    if state.content.isEmpty {
                        visible = String(visible.drop { $0.isNewline })
                    }
                    if !visible.isEmpty {
                        state.content += visible
                        onEvent(.content(visible))
                    }
                    if state.stopMatcher.isStopped { state.shouldStop = true }
                case .thinking(let text):
                    // Reasoning is reported in its own field and never passes
                    // through the stop-string matcher, which applies to the
                    // answer a client shows.
                    state.reasoning += text
                    onEvent(.reasoning(text))
                case .toolCall(let call):
                    state.calls.append(call)
                    onEvent(.toolCall(call))
                }
            }
        }

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: effectivePromptIDs,
            multimodalInput: multimodalInput,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: start,
            prefixCheckpointPositions: prefixCheckpointPositions,
            stateSnapshotPosition: stateSnapshotPosition,
            checkpointBudgetBytes: checkpointBudgetBytes,
            shouldStop: { state.shouldStop }) { @Sendable progress in
                guard state.decodingError == nil else { return }
                do {
                    switch progress {
                    case .prefill:
                        break
                    case .token(_, let tokenID, let delta):
                        let events = try decoder.consume(
                            tokenID: tokenID, delta: delta)
                        handle(events)
                    case .tail(let text):
                        // The flush tail is not tied to a token ID, so it must
                        // go through the decoder's channel state explicitly;
                        // appending it directly would leak text held back
                        // inside the thought channel or a tool call.
                        let events = try decoder.consumeTail(text)
                        handle(events)
                    }
                } catch {
                    state.decodingError = error
                    state.shouldStop = true
                }
        }
        // Stop tokens are intentionally not emitted as `.token` progress by
        // the raw loop. Most dialects finish their structure before that
        // boundary; Harmony's `<|call|>` is both the tool-payload terminator and
        // a stop token, so the structured decoder must consume it explicitly
        // before `finish()` validates its state. Only the withheld boundary is
        // replayed: a token that ended generation at the length limit, a stop
        // string or a cancellation was already consumed through `.token`, and
        // a second pass would close a Gemma, Qwen or MiniMax call twice.
        if state.decodingError == nil {
            do {
                for tokenID in result.undeliveredBoundaryTokenIDs {
                    handle(try decoder.consume(tokenID: tokenID, delta: ""))
                }
            } catch {
                state.decodingError = error
            }
        }
        func structuredFailure(
            kind: StructuredOutputFailureKind,
            cause: StructuredOutputFailureCause
        ) -> StructuredOutputFailure {
            StructuredOutputFailure(
                kind: kind,
                cause: cause,
                diagnostics: StructuredOutputFailureDiagnostics(
                    renderedPromptIDs: renderedPromptIDs,
                    effectivePromptIDs: effectivePromptIDs,
                    result: result,
                    maxCompletionTokens: config.maxNewTokens,
                    decodedCalls: state.calls.count,
                    visibleBytes: state.content.utf8.count,
                    stopStringMatched: state.stopMatcher.isStopped,
                    toolStartID: tokenizer.toolCallStartID,
                    toolEndID: tokenizer.toolCallEndID,
                    toolResponseID: tokenizer.toolResponseID,
                    toolResponseEndID: tokenizer.toolResponseEndID))
        }
        if let decodingError = state.decodingError {
            throw structuredFailure(
                kind: .decoderConsume,
                cause: .classify(decodingError))
        }
        do {
            try decoder.finish()
        } catch {
            throw structuredFailure(
                kind: .decoderFinish,
                cause: .classify(error))
        }
        if needsToolTemplate, result.reason == .toolCalls, state.calls.isEmpty {
            throw structuredFailure(kind: .orphanToolResponse, cause: .none)
        }
        let tail = state.stopMatcher.finish()
        if !tail.isEmpty {
            state.content += tail
            onEvent(.content(tail))
        }
        return ServerStructuredDecodeOutcome(
            result: result,
            content: state.content,
            reasoning: state.reasoning,
            reasoningTokens: decoder.reasoningTokenCount,
            calls: state.calls,
            stopStringFiltered: state.stopMatcher.isStopped)
    }
}

/// Access is sequential: callbacks run synchronously inside `runRawCompletion`
/// while the actor method awaits it; only then does the actor read the state.
private final class ServerDecodeState: @unchecked Sendable {
    var stopMatcher: StreamingStopMatcher
    var content = ""
    var reasoning = ""
    var calls: [ParsedToolCall] = []
    var decodingError: Error?
    var shouldStop = false

    init(stopMatcher: StreamingStopMatcher) {
        self.stopMatcher = stopMatcher
    }
}
