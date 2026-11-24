import Accelerate
import Foundation
import Metal
import TUFFFormat

/// Argument of an Inkling kernel, bound in declaration order.
enum InklingArgument {
    case buffer(MTLBuffer, Int = 0)
    case u32(Int)
    case f32(Float)
    /// A short constant array, copied when the kernel is encoded.
    case floats([Float])
}

/// A command buffer and the one compute pass its kernels are encoded into.
/// A token is some three thousand small dispatches; a pass apiece costs more
/// than the work in most of them. Dispatches in a pass run in order.
final class InklingPass {
    let commandBuffer: MTLCommandBuffer
    private var encoder: MTLComputeCommandEncoder?

    init(_ queue: MTLCommandQueue) throws {
        guard let commandBuffer = queue.makeCommandBuffer() else { throw MetalError.noQueue }
        self.commandBuffer = commandBuffer
    }

    var compute: MTLComputeCommandEncoder? {
        if encoder == nil { encoder = commandBuffer.makeComputeCommandEncoder() }
        return encoder
    }

    /// Before anything else encodes into the command buffer, and before it
    /// is committed.
    func endEncoding() {
        encoder?.endEncoding()
        encoder = nil
    }

    /// A pass abandoned by an error or a cancelled block is never committed,
    /// but Metal requires its encoder to be ended.
    deinit { encoder?.endEncoding() }
}

/// The compute pipelines of `inkling.metal` and the two dispatch shapes they
/// use: one threadgroup per grid cell, or one thread per grid cell.
struct InklingKernels {
    static let names = [
        "ink_rms_norm", "ink_depth_attention", "ink_gate_scales", "ink_gate_down", "ink_gate_mix",
        "ink_affine4_gemv", "ink_affine4_gemv_half", "ink_affine4_lookup", "ink_conv_static",
        "ink_conv_dynamic", "ink_conv_advance", "ink_head_clip", "ink_relative_bias",
        "ink_cache_store", "ink_attention_scores", "ink_attention_mix", "ink_sigmoid_gate",
        "ink_add", "ink_zero", "ink_expert_gemv_nvfp4", "ink_expert_gemv_mxfp8",
        "ink_expert_activation", "ink_to_half", "ink_to_float", "ink_gather_rows_half",
        "ink_expert_activation_half", "ink_expert_reduce",
        "ink_expert_phase1_nvfp4", "ink_expert_phase1_mxfp8",
        "ink_expert_phase2_nvfp4", "ink_expert_phase2_mxfp8",
    ]
    /// The several-rows expert kernels. Optional: a GPU that cannot build one
    /// keeps the one-row kernels.
    static let rowNames = [
        "ink_expert_phase1_rows_nvfp4", "ink_expert_phase1_rows_mxfp8",
        "ink_expert_phase2_rows_nvfp4", "ink_expert_phase2_rows_mxfp8",
    ]
    /// Rows one threadgroup of a `_rows` kernel covers: kInkRows x kInkRowGroups.
    static let rowsPerGroup = 8
    static let rowGroupThreads = 64
    private let pipelines: [String: MTLComputePipelineState]
    let hasRowKernels: Bool

    init(context: MetalContext) throws {
        var pipelines: [String: MTLComputePipelineState] = [:]
        for name in Self.names { pipelines[name] = try context.pipeline(name) }
        let rows = Self.rowNames.compactMap { name in (try? context.pipeline(name)).map { (name, $0) } }
        hasRowKernels = rows.count == Self.rowNames.count
        if hasRowKernels { for (name, pipeline) in rows { pipelines[name] = pipeline } }
        self.pipelines = pipelines
    }

    private func bind(_ encoder: MTLComputeCommandEncoder, _ name: String,
                      _ arguments: [InklingArgument]) -> MTLComputePipelineState {
        let pipeline = pipelines[name]!
        encoder.setComputePipelineState(pipeline)
        for (index, argument) in arguments.enumerated() {
            switch argument {
            case .buffer(let buffer, let offset):
                encoder.setBuffer(buffer, offset: offset, index: index)
            case .u32(let value):
                var value = UInt32(value)
                encoder.setBytes(&value, length: MemoryLayout<UInt32>.stride, index: index)
            case .f32(let value):
                var value = value
                encoder.setBytes(&value, length: MemoryLayout<Float>.stride, index: index)
            case .floats(let values):
                encoder.setBytes(values, length: values.count * MemoryLayout<Float>.stride, index: index)
            }
        }
        return pipeline
    }

    func groups(_ pass: InklingPass, _ name: String, _ arguments: [InklingArgument],
                grid: (Int, Int), threads: Int) {
        guard let encoder = pass.compute else { return }
        _ = bind(encoder, name, arguments)
        encoder.dispatchThreadgroups(MTLSize(width: grid.0, height: grid.1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    func threads(_ pass: InklingPass, _ name: String, _ arguments: [InklingArgument],
                 grid: (Int, Int, Int)) {
        guard let encoder = pass.compute else { return }
        let pipeline = bind(encoder, name, arguments)
        encoder.dispatchThreads(MTLSize(width: grid.0, height: grid.1, depth: grid.2),
                                threadsPerThreadgroup: MTLSize(
                                    width: min(pipeline.maxTotalThreadsPerThreadgroup, 64),
                                    height: 1, depth: 1))
    }
}

/// What an Inkling sequence carries from one token to the next: keys and
/// values for every layer visit, and the last three inputs of each visit's
/// four short convolutions.
///
/// A full-attention visit keeps one row per position. A sliding visit keeps a
/// ring of `ringCapacity` rows, the window plus room for one prefill chunk,
/// so a row is still there for every query that can see it.
final class InklingSequenceState {
    let visits: Int
    let rowBytes: Int
    let ringCapacity: Int
    let maxContext: Int
    private let device: MTLDevice
    private let isRing: [Bool]
    private let convChannels: [Int]
    private(set) var keys: [MTLBuffer] = []
    private(set) var values: [MTLBuffer] = []
    private(set) var conv: [[MTLBuffer]] = []
    private(set) var fullCapacity: Int
    private(set) var position = 0

    static let initialFullCapacity = 2_048

    init(device: MTLDevice, visits: Int, isRing: [Bool], keyValueWidth: Int, hiddenSize: Int,
         window: Int, chunkCapacity: Int, maxContext: Int) throws {
        self.device = device
        self.visits = visits
        self.isRing = isRing
        self.maxContext = maxContext
        rowBytes = keyValueWidth * MemoryLayout<Float16>.stride
        // A context shorter than that never wraps the ring.
        ringCapacity = min(maxContext, window + chunkCapacity)
        fullCapacity = min(maxContext, Self.initialFullCapacity)
        convChannels = [keyValueWidth, keyValueWidth, hiddenSize, hiddenSize]
        for visit in 0..<visits {
            let rows = isRing[visit] ? ringCapacity : fullCapacity
            keys.append(try Self.allocate(device, rows * rowBytes, "inkling.keys.\(visit)"))
            values.append(try Self.allocate(device, rows * rowBytes, "inkling.values.\(visit)"))
            conv.append(try convChannels.enumerated().map { index, channels in
                try Self.allocate(device, 3 * channels * MemoryLayout<Float>.stride,
                                  "inkling.conv.\(visit).\(index)")
            })
        }
    }

    private static func allocate(_ device: MTLDevice, _ length: Int, _ label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: max(16, length), options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        buffer.label = label
        memset(buffer.contents(), 0, buffer.length)
        return buffer
    }

    func isRing(_ visit: Int) -> Bool { isRing[visit] }
    func capacity(_ visit: Int) -> Int { isRing[visit] ? ringCapacity : fullCapacity }
    /// Rows a query at the last written position can find in a visit's cache.
    func slots(_ visit: Int, through lastPosition: Int) -> Int {
        min(lastPosition + 1, capacity(visit))
    }
    func slot(_ visit: Int, position: Int) -> Int {
        isRing[visit] ? position % ringCapacity : position
    }

    /// The convolution taps as they were when `saveConvolutions` ran. A
    /// block overwrites them visit by visit, and they are all of the
    /// sequence a block changes that the tokens after it still need: ring
    /// slots it writes held keys already outside the window, and
    /// full-attention rows are only appended.
    private var savedConvolutions: [UInt8] = []

    /// Call only with no GPU work pending.
    func saveConvolutions() {
        let total = conv.reduce(0) { $0 + $1.reduce(0) { $0 + $1.length } }
        if savedConvolutions.count != total { savedConvolutions = [UInt8](repeating: 0, count: total) }
        var offset = 0
        savedConvolutions.withUnsafeMutableBytes { saved in
            for buffer in conv.joined() {
                memcpy(saved.baseAddress! + offset, buffer.contents(), buffer.length)
                offset += buffer.length
            }
        }
    }

    func restoreConvolutions() {
        var offset = 0
        savedConvolutions.withUnsafeBytes { saved in
            for buffer in conv.joined() {
                memcpy(buffer.contents(), saved.baseAddress! + offset, buffer.length)
                offset += buffer.length
            }
        }
    }

    /// Starts a new sequence. The convolutions begin from zeros.
    func reset() {
        position = 0
        for buffers in conv { for buffer in buffers { memset(buffer.contents(), 0, buffer.length) } }
    }

    func advance(by count: Int) { position += count }
    func setPosition(_ value: Int) { position = value }

    /// Grows the full-attention caches to hold `end` rows, keeping what is
    /// there. Call only with no GPU work pending.
    func ensureCapacity(through end: Int, queue: MTLCommandQueue) throws {
        guard end <= maxContext else {
            throw PrefillError.chunkedUnsupported("Inkling position \(end) exceeds maxContext \(maxContext)")
        }
        guard end > fullCapacity else { return }
        var grown = fullCapacity
        while grown < end { grown = min(maxContext, grown * 2) }
        let copyBytes = min(position, fullCapacity) * rowBytes
        guard let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else { throw MetalError.noQueue }
        var replacements: [(Int, MTLBuffer, MTLBuffer)] = []
        for visit in 0..<visits where !isRing[visit] {
            let newKeys = try Self.allocate(device, grown * rowBytes, "inkling.keys.\(visit)")
            let newValues = try Self.allocate(device, grown * rowBytes, "inkling.values.\(visit)")
            if copyBytes > 0 {
                blit.copy(from: keys[visit], sourceOffset: 0, to: newKeys, destinationOffset: 0, size: copyBytes)
                blit.copy(from: values[visit], sourceOffset: 0, to: newValues, destinationOffset: 0, size: copyBytes)
            }
            replacements.append((visit, newKeys, newValues))
        }
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer)
        for (visit, newKeys, newValues) in replacements {
            keys[visit] = newKeys
            values[visit] = newValues
        }
        fullCapacity = grown
    }

    /// The ranges a sequence at `position` has written. With `ringsOnly`, only
    /// what later tokens overwrite in place: the rings and the convolution
    /// taps. Full-attention rows are appended and stay where they are.
    func addSnapshotRanges(to builder: inout RunnerStateSnapshotBuilder, position: Int,
                           ringsOnly: Bool = false) {
        for visit in 0..<visits {
            if isRing[visit] || !ringsOnly {
                let rows = min(position, capacity(visit))
                builder.add("keys.\(visit)", keys[visit], length: rows * rowBytes)
                builder.add("values.\(visit)", values[visit], length: rows * rowBytes)
            }
            for (index, buffer) in conv[visit].enumerated() {
                builder.add("conv.\(visit).\(index)", buffer, length: buffer.length)
            }
        }
    }

    func snapshotBytes(position: Int, ringsOnly: Bool = false) -> Int {
        var builder = RunnerStateSnapshotBuilder()
        addSnapshotRanges(to: &builder, position: position, ringsOnly: ringsOnly)
        return builder.byteCount
    }
}

/// The batched products chunked prefill runs on: TUFF's affine 4-bit matmul
/// for the projections, and one matmul per expert format. Nil where the
/// tensor pipeline does not build (macOS 15); prompts are then read through
/// the per-row kernels, which give the same results more slowly.
struct InklingBatchedProducts {
    typealias Tile = (rows: Int, columns: Int, pipeline: MTLComputePipelineState)
    let affine: MPPPrefillAffineQMM
    let nvfp4: [Tile]
    let mxfp8: [Tile]

    init?(context: MetalContext) {
        guard let affine = MPPPrefillAffineQMM(context: context, bits: 4, groupSize: 64),
              let library = try? MetalContext.privateLibrary(device: context.device, module: "tensorops")
        else { return nil }
        func tiles(_ format: String) -> [Tile]? {
            var tiles: [Tile] = []
            for (suffix, rows, columns) in [("_m8", 8, 64), ("_m16", 16, 64), ("_m32", 32, 32), ("", 64, 32)] {
                guard let function = library.makeFunction(name: "mpp_prefill_\(format)_qmm_f32\(suffix)"),
                      let pipeline = try? context.device.makeComputePipelineState(function: function),
                      pipeline.maxTotalThreadsPerThreadgroup >= pipeline.threadExecutionWidth * 4
                else { return nil }
                tiles.append((rows, columns, pipeline))
            }
            return tiles
        }
        guard let nvfp4 = tiles("nvfp4"), let mxfp8 = tiles("mxfp8") else { return nil }
        self.affine = affine
        self.nvfp4 = nvfp4
        self.mxfp8 = mxfp8
    }

    /// Y[m, n] = X[m, k] * W[n, k]^T for one expert matrix. X is FP16, Y FP32.
    func expert(_ pass: InklingPass, nvfp4 format: Bool,
                codes: (MTLBuffer, Int), scales: (MTLBuffer, Int),
                x: (MTLBuffer, Int), y: (MTLBuffer, Int), m: Int, n: Int, k: Int, factor: Float) {
        let tiles = format ? nvfp4 : mxfp8
        let tile = tiles.first { m <= $0.rows } ?? tiles[tiles.count - 1]
        guard let encoder = pass.compute else { return }
        encoder.setComputePipelineState(tile.pipeline)
        encoder.setBuffer(codes.0, offset: codes.1, index: 0)
        encoder.setBuffer(scales.0, offset: scales.1, index: 1)
        encoder.setBuffer(x.0, offset: x.1, index: 2)
        encoder.setBuffer(y.0, offset: y.1, index: 3)
        var mValue = UInt32(m), nValue = UInt32(n), kValue = UInt32(k), factorValue = factor
        encoder.setBytes(&mValue, length: 4, index: 4)
        encoder.setBytes(&nValue, length: 4, index: 5)
        encoder.setBytes(&kValue, length: 4, index: 6)
        encoder.setBytes(&factorValue, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + tile.columns - 1) / tile.columns, height: (m + tile.rows - 1) / tile.rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tile.pipeline.threadExecutionWidth * 4, height: 1, depth: 1))
    }
}

/// Inkling 2's forward path.
///
/// A token passes through every layer once and through the looped layers
/// `passes` times: 48 visits for Inkling 2 Mini. A visit is one decoder
/// layer applied with its own keys, values and convolution taps:
///
/// 1. The layer's inputs are read by depth attention over the blocks before
///    it. A looped layer after the first pass also mixes in messages from the
///    previous pass through low-rank gates.
/// 2. Attention: 4-bit projections, short convolutions on keys and values,
///    a relative position bias instead of RoPE, a sink logit per head, and an
///    output gate.
/// 3. The feed-forward block routes to eight experts. Routing is 128 FP32 dot
///    products a token and runs on the CPU, which is also where the expert
///    read is started, so the visit splits there into two command buffers.
///    The second half is encoded together with the first half of the next
///    visit.
///
/// The same code runs one token (decode) or a block of prompt tokens. Every
/// kernel takes a row count; a block's projections and expert products go
/// through batched matmuls where the system has them, so a weight tile is
/// read once for the whole block instead of once per token.
///
/// The batched matmuls work in FP16, the per-row kernels in FP32. The two
/// agree to about a part in a thousand, which is enough to change which
/// expert wins a near tie between the eighth and ninth. A changed expert
/// moves that token's state by a few percent and later tokens read it, so a
/// long prompt read in blocks is not the same numbers as the prompt read a
/// token at a time, only an equally good reading of it.
final class InklingForwardRunner: ChunkedPrefillRunner, ContextWindowReporting,
    ContinuableLogitProducer, @unchecked Sendable {
    private struct AffineView {
        let view: TensorView
        var weights: InklingArgument { .buffer(view.buffer, Int(view.offset)) }
        var scales: InklingArgument { .buffer(view.buffer, Int(view.scaleOffset)) }
        var biases: InklingArgument { .buffer(view.buffer, Int(view.biasOffset)) }
    }

    private struct MessageGate {
        let down: InklingArgument
        let forget: InklingArgument
        let gain: InklingArgument
    }

    private struct LayerViews {
        let full: Bool
        let looped: Bool
        let attentionQuery: InklingArgument?
        let mlpQuery: InklingArgument
        /// One norm for a sliding layer; query, key and value norms for a full one.
        let inputNorms: [InklingArgument]
        let mlpNorm: InklingArgument
        let q, k, v, r, gate, sink, o: AffineView
        let qClip: InklingArgument
        let relClip: InklingArgument
        let relProjection: InklingArgument
        /// Sliding layers: the key and value convolution taps.
        let keyValueTaps: [InklingArgument]
        /// Full layers: the fixed taps and the projection of the moving ones.
        let dynamicTaps: [(fixed: InklingArgument, projection: AffineView)]
        let attentionTaps: InklingArgument
        let mlpTaps: InklingArgument
        /// Query, key, value and feed-forward gates of a looped layer.
        let messageGates: [MessageGate]
        let routerWeight: [Float]
        let routerBias: [Float]
        let nvfp4: Bool
        /// Per-expert NVFP4 factors, already divided by the format's range.
        let gateUpFactors: [Float]
        let downFactors: [Float]
        let gateUpCodes, gateUpScales, downCodes, downScales: Int
    }

    /// A row of some buffer: where row 0 starts and how far apart rows are,
    /// in bytes and in floats.
    private struct Rows {
        let buffer: MTLBuffer
        var offset = 0
        let stride: Int
        var argument: InklingArgument { .buffer(buffer, offset) }
    }

    private let model: Model
    private let context: MetalContext
    private let config: ArchConfig
    private let loop: InklingLoopConfig
    private let kernels: InklingKernels
    private let batched: InklingBatchedProducts?
    private let layers: [LayerViews]
    private let state: InklingSequenceState
    private let embedding: AffineView
    private let embeddingNorm: InklingArgument
    private let finalNorm: InklingArgument
    private let finalQuery: InklingArgument
    private let messageQuery: TensorView
    private let head: AffineView

    let maxContext: Int
    /// Positions a sliding layer's query can see, its own included.
    private let window: Int
    private let fusedDecode: Bool
    /// A decoded token's experts through the several-rows kernels, after the
    /// load-time comparison found them byte-identical to the one-row kernels.
    let rowKernels: Bool
    /// Token rows one forward pass can hold.
    private let rowCapacity: Int
    /// The prompt rows one block of `prefillChunked` runs, whatever the
    /// requested chunk size.
    var prefillBlockTokens: Int { rowCapacity }
    /// Scores one attention dispatch may write, in floats.
    private let scoreCapacity: Int

    // FP32 scratch, `rowCapacity` rows each unless noted.
    private let hidden, normed, blocks, attentionInputs, messages, bankBlocks, bankPartial, preLoopHidden: MTLBuffer
    private let gatedInputs, queryInput, keyInput, valueInput: MTLBuffer
    private let queryRows, keyRows, valueRows, keyConv, valueConv, relative, outputGate, sinks: MTLBuffer
    private let movingKeyTaps, movingValueTaps, bias, scores: MTLBuffer
    private let attention, gatedAttention, projected, attentionConv, residual: MTLBuffer
    private let mlpPartial, mlpInput, mlpGated, mlpNormed, mlpOutput, mlpConv: MTLBuffer
    private let finalMix, finalNormed: MTLBuffer
    /// One expert's gate/up and hidden rows on the per-row path.
    private let gateUp, activation: MTLBuffer
    /// FP16 staging for the batched projections.
    private let halfInput, halfOutput: MTLBuffer
    /// Expert-sorted (token, expert) pairs of a block: gathered inputs, gate/up
    /// products, hidden rows and down products.
    private let pairRows, pairGateUp, pairActivation, pairDown: MTLBuffer
    private let pairRowIndex, pairWeights, tokenPairs, tokenWeights: MTLBuffer
    private let gateScales, gateLow, logScale, one, tokenBuffer: MTLBuffer
    private var depth = 0

    private(set) var totalIoNanos: UInt64 = 0
    private(set) var totalCb1Nanos: UInt64 = 0
    private(set) var totalCb2Nanos: UInt64 = 0
    private(set) var totalHeadNanos: UInt64 = 0
    /// GPU time of every committed command buffer, and how many.
    private(set) var totalGPULayerNanos: UInt64 = 0
    private(set) var totalGPULayerCommandBuffers: UInt64 = 0
    private(set) var totalRoutedExpertReads: UInt64 = 0
    private(set) var totalRoutedExpertBytes: UInt64 = 0
    private(set) var totalRoutedExpertCacheHits: UInt64 = 0
    private(set) var totalRoutedExpertCacheMisses: UInt64 = 0
    /// Blocks of more than one row whose products ran on the batched path.
    private(set) var batchedBlocks = 0
    private var prefillClock = PrefillPhaseClock()
    var prefillPhaseTimes: PrefillPhaseTimes { prefillClock.times }

    /// Decode lookahead: once a visit's own experts are read, the next
    /// visit's router over this visit's feed-forward input guesses the next
    /// visit's experts, and storage is asked to start reading them while
    /// this visit and the next one's attention run. 76% of the guessed
    /// experts are the ones routed; on a 16 GB M2 decode waited on reads 35
    /// to 45% less and ran 8% faster on average from a cold file cache. The
    /// next visit still routes on its own input; a wrong guess only costs
    /// bandwidth. See `ExpertLookahead`. `TUFF_EXPERT_LOOKAHEAD=off` turns it off.
    private var lookahead = ExpertLookahead()
    /// The layer of the visit after each visit within a token, by visit.
    private let nextLayerOfVisit: [Int?]
    var lookaheadPrecision: Double { lookahead.precision }
    var lookaheadReadsIssued: Int { lookahead.readsIssued }
    var lookaheadEnabled: Bool { lookahead.enabled }
    /// Prompt blocks read ahead too, with background `pread`s: the rest of a
    /// visit's experts while its first batch is read, and the next visit's
    /// while this one computes. On a 16 GB M2 a 1,062-token prompt took
    /// 32.8 s instead of 46.6. (Storage hints, which decode uses, saved
    /// little here.) `TUFF_INKLING_PREFILL_READAHEAD=off` turns it off.
    private let blockReadAhead = ProcessInfo.processInfo.environment["TUFF_INKLING_PREFILL_READAHEAD"] != "off"
    var expertReadMetrics: ExpertReadMetrics { model.expertReadMetrics }

    /// `slidingWindow` is for tests: a shorter window lets a short sequence
    /// cover keys leaving it. The relative bias table keeps its real extent.
    /// `batchedProducts` false reads blocks with the per-row kernels, as a
    /// system without the tensor pipeline does; `TUFF_INKLING_BATCHED_PREFILL=off`
    /// does the same for a whole process. `fusedDecode` false sends a single
    /// token through the per-row kernels too, which makes it the same
    /// arithmetic as a block read that way.
    init(model: Model, context: MetalContext, maxContext: Int,
         runtimeConfiguration: RuntimeConfiguration, slidingWindow: Int? = nil,
         batchedProducts: Bool? = nil, fusedDecode: Bool = true) throws {
        let config = model.config
        let loop = config.inklingLoop
        guard config.family == .inkling2, loop.isEnabled, loop.convKernelSize == 4,
              config.feedForwardKind == .mixtureOfExperts,
              config.headDim == config.fullHeadDim, config.numKVHeads == config.numFullKVHeads,
              config.headDim <= 256, config.numHeads % config.numKVHeads == 0,
              loop.relativeExtent == config.slidingWindow,
              (config.numLayers + loop.blockLayers - 1) / loop.blockLayers <= 8,
              config.topKExperts <= 16 else {
            throw ModelError.archMismatch(field: "family", expected: ModelFamily.inkling2.rawValue,
                                          actual: config.family.rawValue)
        }
        guard maxContext > 0, (slidingWindow ?? 1) > 0,
              slidingWindow ?? config.slidingWindow <= config.slidingWindow else {
            throw PrefillError.chunkedUnsupported("Inkling maxContext and window must be positive")
        }
        window = slidingWindow ?? config.slidingWindow
        self.fusedDecode = fusedDecode
        nextLayerOfVisit = Self.nextLayers(numLayers: config.numLayers, loop: loop)
        try context.prepareKernelGroups(MetalKernelGroup.required(for: config))
        self.model = model
        self.context = context
        self.config = config
        self.loop = loop
        self.maxContext = maxContext
        kernels = try InklingKernels(context: context)
        rowKernels = ProcessInfo.processInfo.environment["TUFF_INKLING_ROW_KERNELS"] != "off"
            && kernels.hasRowKernels
            && config.hiddenSize % InklingKernels.rowsPerGroup == 0
            && config.moeIntermediateSize % InklingKernels.rowsPerGroup == 0
            && Self.rowKernelsMatch(kernels, device: context.device)
        if ProcessInfo.processInfo.environment["TUFF_PHASES"] == "1" {
            FileHandle.standardError.write(Data("[inkling] row kernels \(rowKernels ? "on" : "off")\n".utf8))
        }
        let useBatched = batchedProducts
            ?? (ProcessInfo.processInfo.environment["TUFF_INKLING_BATCHED_PREFILL"] != "off")
        batched = useBatched ? InklingBatchedProducts(context: context) : nil
        rowCapacity = runtimeConfiguration.prefillPolicy == .chunked
            ? min(InklingLoopConfig.maximumBlockRows, max(1, runtimeConfiguration.prefillChunkTokens)) : 1

        let hiddenSize = config.hiddenSize
        let qDim = config.numHeads * config.headDim
        let kvDim = config.numKVHeads * config.headDim
        let visits = loop.visits(numLayers: config.numLayers)
        state = try InklingSequenceState(
            device: context.device, visits: visits,
            isRing: (0..<visits).map { !config.layerIsFull(loop.layer(ofVisit: $0, numLayers: config.numLayers)) },
            keyValueWidth: kvDim, hiddenSize: hiddenSize, window: window,
            chunkCapacity: rowCapacity, maxContext: maxContext)

        func plain(_ name: String) throws -> InklingArgument {
            let view = try model.resident(name: name)
            return .buffer(view.buffer, Int(view.offset))
        }
        func floats(_ name: String) throws -> [Float] {
            let view = try model.resident(name: name)
            var values = [Float](repeating: 0, count: Int(view.length) / MemoryLayout<Float>.stride)
            values.withUnsafeMutableBytes {
                $0.copyMemory(from: UnsafeRawBufferPointer(
                    start: view.buffer.contents() + Int(view.offset), count: Int(view.length)))
            }
            return values
        }
        func affine(_ name: String) throws -> AffineView { AffineView(view: try model.resident(name: name)) }

        let root = "language_model.model."
        embedding = try affine(root + "embed_tokens.weight")
        embeddingNorm = try plain(root + "embed_norm.weight")
        finalNorm = try plain(root + "norm.weight")
        finalQuery = try plain(root + "final_input.query")
        messageQuery = try model.resident(name: root + "message_input.query")
        head = try affine("language_model.lm_head.weight")

        var resolved: [LayerViews] = []
        for layer in 0..<config.numLayers {
            let prefix = "\(root)layers.\(layer)."
            let attention = prefix + "self_attn."
            let full = config.layerIsFull(layer)
            let looped = loop.loopLayers.contains(layer)
            guard let expert = model.packedExpertsLayout.layers.first(where: { $0.layer == layer })?
                .experts.first,
                  let gateUpCodes = expert.subTensors["gate_up"],
                  let gateUpScales = expert.subTensors["gate_up_scales"],
                  let downCodes = expert.subTensors["down"],
                  let downScales = expert.subTensors["down_scales"] else {
                throw ModelError.indexCorrupt(detail: "Inkling layer \(layer) has no routed experts")
            }
            let nvfp4 = gateUpCodes.bits == 4
            // An NVFP4 value is code x block scale x amax / 2688, and the
            // conversion stored amax x 2688 / denominator.
            let range: Float = 6 * 448
            resolved.append(LayerViews(
                full: full, looped: looped,
                attentionQuery: layer > 0 ? try plain(prefix + "attention_input.query") : nil,
                mlpQuery: try plain(prefix + "mlp_input.query"),
                inputNorms: try full
                    ? ["q", "k", "v"].map { try plain(prefix + "\($0)_input_layernorm.weight") }
                    : [try plain(prefix + "input_layernorm.weight")],
                mlpNorm: try plain(prefix + "post_attention_layernorm.weight"),
                q: try affine(attention + "q_proj.weight"), k: try affine(attention + "k_proj.weight"),
                v: try affine(attention + "v_proj.weight"), r: try affine(attention + "r_proj.weight"),
                gate: try affine(attention + "gate_proj.weight"),
                sink: try affine(attention + "sink_proj.weight"), o: try affine(attention + "o_proj.weight"),
                qClip: try plain(attention + "q_l2_clip.scale"),
                relClip: try plain(attention + "rel_l2_clip.scale"),
                relProjection: try plain(attention + "rel_logits_proj.proj"),
                keyValueTaps: try full ? [] : ["k", "v"].map { try plain(attention + "\($0)_sconv.conv1d.weight") },
                dynamicTaps: try full ? ["k", "v"].map {
                    (fixed: try plain(attention + "\($0)_sconv.static_weight"),
                     projection: try affine(attention + "\($0)_sconv.weight_proj.weight"))
                } : [],
                attentionTaps: try plain(prefix + "post_attention_convolution.conv1d.weight"),
                mlpTaps: try plain(prefix + "post_mlp_convolution.conv1d.weight"),
                messageGates: try looped ? ["q", "k", "v", "mlp"].map {
                    let gate = prefix + "\($0)_message_gate."
                    return MessageGate(down: try plain(gate + "down_proj.weight"),
                                       forget: try plain(gate + "forget_gate_proj.weight"),
                                       gain: try plain(gate + "message_gate_proj.weight"))
                } : [],
                routerWeight: try floats(prefix + "mlp.gate.weight"),
                routerBias: try floats(prefix + "mlp.gate.e_score_correction_bias"),
                nvfp4: nvfp4,
                gateUpFactors: nvfp4
                    ? try floats(prefix + "mlp.experts.gate_up_proj.global_scale").map { $0 / range } : [],
                downFactors: nvfp4
                    ? try floats(prefix + "mlp.experts.down_proj.global_scale").map { $0 / range } : [],
                gateUpCodes: Int(gateUpCodes.offset), gateUpScales: Int(gateUpScales.offset),
                downCodes: Int(downCodes.offset), downScales: Int(downScales.offset)))
        }
        layers = resolved

        let device = context.device
        func buffer(_ count: Int, _ label: String, stride: Int = MemoryLayout<Float>.stride) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(4, count) * stride, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "inkling." + label
            memset(buffer.contents(), 0, buffer.length)
            return buffer
        }
        let rows = rowCapacity
        let half = MemoryLayout<Float16>.stride
        let maxDepth = 8
        let intermediate = config.moeIntermediateSize
        let pairs = rows * config.topKExperts
        // A decoded token scores every key of a full layer in one dispatch;
        // a block scores as many rows at a time as fit in the same space.
        scoreCapacity = max(config.numHeads * max(maxContext, state.ringCapacity), 8 << 20)
        hidden = try buffer(rows * hiddenSize, "hidden")
        normed = try buffer(rows * hiddenSize, "normed")
        blocks = try buffer(maxDepth * rows * hiddenSize, "blocks")
        attentionInputs = try buffer(rows * 3 * hiddenSize, "attentionInputs")
        messages = try buffer(rows * 4 * hiddenSize, "messages")
        bankBlocks = try buffer(maxDepth * rows * hiddenSize, "bankBlocks")
        bankPartial = try buffer(rows * hiddenSize, "bankPartial")
        preLoopHidden = try buffer(rows * hiddenSize, "preLoopHidden")
        gatedInputs = try buffer(rows * 3 * hiddenSize, "gatedInputs")
        queryInput = try buffer(rows * hiddenSize, "queryInput")
        keyInput = try buffer(rows * hiddenSize, "keyInput")
        valueInput = try buffer(rows * hiddenSize, "valueInput")
        queryRows = try buffer(rows * qDim, "query")
        keyRows = try buffer(rows * kvDim, "key")
        valueRows = try buffer(rows * kvDim, "value")
        keyConv = try buffer(rows * kvDim, "keyConv")
        valueConv = try buffer(rows * kvDim, "valueConv")
        relative = try buffer(rows * config.numHeads * loop.relativeDim, "relative")
        outputGate = try buffer(rows * qDim, "outputGate")
        sinks = try buffer(rows * config.numHeads, "sinks")
        movingKeyTaps = try buffer(rows * config.numKVHeads * 4, "movingKeyTaps")
        movingValueTaps = try buffer(rows * config.numKVHeads * 4, "movingValueTaps")
        bias = try buffer(rows * config.numHeads * loop.relativeExtent, "bias")
        scores = try buffer(scoreCapacity, "scores")
        attention = try buffer(rows * qDim, "attention")
        gatedAttention = try buffer(rows * qDim, "gatedAttention")
        projected = try buffer(rows * hiddenSize, "projected")
        attentionConv = try buffer(rows * hiddenSize, "attentionConv")
        residual = try buffer(rows * hiddenSize, "residual")
        mlpPartial = try buffer(rows * hiddenSize, "mlpPartial")
        mlpInput = try buffer(rows * hiddenSize, "mlpInput")
        mlpGated = try buffer(rows * hiddenSize, "mlpGated")
        mlpNormed = try buffer(rows * hiddenSize, "mlpNormed")
        mlpOutput = try buffer(rows * hiddenSize, "mlpOutput")
        mlpConv = try buffer(rows * hiddenSize, "mlpConv")
        finalMix = try buffer(rows * hiddenSize, "finalMix")
        finalNormed = try buffer(hiddenSize, "finalNormed")
        gateUp = try buffer(rows * 2 * intermediate, "gateUp")
        // A block's rows one expert at a time, or a decoded token's eight experts at once.
        activation = try buffer(max(rows, config.topKExperts) * intermediate, "activation")
        let batchedRows = batched == nil || rows == 1 ? 1 : rows
        halfInput = try buffer(batchedRows * max(qDim, hiddenSize), "halfInput", stride: half)
        halfOutput = try buffer(batchedRows * max(qDim, hiddenSize), "halfOutput", stride: half)
        let batchedPairs = batched == nil || rows == 1 ? 1 : pairs
        pairRows = try buffer(batchedPairs * hiddenSize, "pairRows", stride: half)
        pairGateUp = try buffer(batchedPairs * 2 * intermediate, "pairGateUp")
        pairActivation = try buffer(batchedPairs * intermediate, "pairActivation", stride: half)
        pairDown = try buffer(batchedPairs * hiddenSize, "pairDown")
        pairRowIndex = try buffer(pairs, "pairRowIndex")
        pairWeights = try buffer(pairs, "pairWeights")
        tokenPairs = try buffer(pairs, "tokenPairs")
        tokenWeights = try buffer(pairs, "tokenWeights")
        gateScales = try buffer(rows * 2, "gateScales")
        gateLow = try buffer(rows * loop.gateRank, "gateLow")
        logScale = try buffer(rows, "logScale")
        one = try buffer(rows, "one")
        let ones = one.contents().assumingMemoryBound(to: Float.self)
        for row in 0..<rows { ones[row] = 1 }
        tokenBuffer = try buffer(rows, "token")
    }

    /// Visits in the order a token makes them, as (visit, layer): the layers
    /// before the loop, the loop `passes` times, then the layers after it.
    /// Matches the numbering in `run`.
    static func visitOrder(numLayers: Int, loop: InklingLoopConfig) -> [(visit: Int, layer: Int)] {
        var order = (0..<loop.loopStart).map { ($0, $0) }
        order += loop.loopLayers.map { ($0, $0) }
        for loopPass in 1..<max(1, loop.passes) {
            let base = numLayers + (loopPass - 1) * loop.loopLayers.count - loop.loopStart
            order += loop.loopLayers.map { ($0 + base, $0) }
        }
        order += (loop.loopEnd..<numLayers).map { ($0, $0) }
        return order
    }

    static func nextLayers(numLayers: Int, loop: InklingLoopConfig) -> [Int?] {
        let order = visitOrder(numLayers: numLayers, loop: loop)
        var next = [Int?](repeating: nil, count: loop.visits(numLayers: numLayers))
        for index in order.indices.dropLast() { next[order[index].visit] = order[index + 1].layer }
        return next
    }

    /// Test seam: called when a visit's attention has run, while
    /// `attentionRows` still holds its output (before the output gate).
    var afterAttention: ((_ visit: Int) -> Void)?
    func attentionRows(_ rows: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: attention.contents().assumingMemoryBound(to: Float.self),
                                  count: rows * config.numHeads * config.headDim))
    }

    // MARK: Sequence

    func reset() {
        lookahead.pending = nil
        model.cancelRoutedExpertReadAhead()
        state.reset()
    }

    var continuationPosition: Int { state.position }

    func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition > 0, state.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "Inkling continuation expected position \(expectedPosition), current \(state.position)")
        }
    }

    var stateSnapshotByteEstimate: Int? { state.snapshotBytes(position: state.position) }

    func captureState() throws -> RunnerStateSnapshot {
        var builder = RunnerStateSnapshotBuilder()
        state.addSnapshotRanges(to: &builder, position: state.position)
        return try builder.capture(owner: self, queue: context.queue,
                                   host: .init(position: state.position, ngramContext: [], ropeDelta: 0))
    }

    func restoreState(_ snapshot: RunnerStateSnapshot) throws {
        guard snapshot.owner == ObjectIdentifier(self) else { throw RunnerStateSnapshotError.foreignSnapshot }
        reset()
        do {
            let position = snapshot.host.position
            guard position <= maxContext else {
                throw RunnerStateSnapshotError.layoutMismatch("position exceeds context")
            }
            if position > 0 { try state.ensureCapacity(through: position, queue: context.queue) }
            var builder = RunnerStateSnapshotBuilder()
            state.addSnapshotRanges(to: &builder, position: position)
            try builder.restore(snapshot, queue: context.queue)
            state.setPosition(position)
        } catch {
            reset()
            throw error
        }
    }

    /// Full-attention rows are only appended, so returning to an earlier
    /// position needs the rings and the convolution taps as they were.
    var prefixCheckpointByteEstimate: Int { state.snapshotBytes(position: state.position, ringsOnly: true) }

    func capturePrefixCheckpoint() throws -> RunnerStateSnapshot {
        var builder = RunnerStateSnapshotBuilder()
        state.addSnapshotRanges(to: &builder, position: state.position, ringsOnly: true)
        return try builder.capture(owner: self, queue: context.queue,
                                   host: .init(position: state.position, ngramContext: [], ropeDelta: 0))
    }

    func rewind(to checkpoint: RunnerStateSnapshot) throws {
        guard checkpoint.owner == ObjectIdentifier(self) else { throw RunnerStateSnapshotError.foreignSnapshot }
        let position = checkpoint.host.position
        do {
            guard position > 0, position <= state.position else {
                throw RunnerStateSnapshotError.layoutMismatch(
                    "checkpoint at \(position) is past the sequence at \(state.position)")
            }
            var builder = RunnerStateSnapshotBuilder()
            state.addSnapshotRanges(to: &builder, position: position, ringsOnly: true)
            try builder.restore(checkpoint, queue: context.queue)
            state.setPosition(position)
        } catch {
            reset()
            throw error
        }
    }

    // MARK: Entry points

    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await execute(tokens: [token], startPosition: position, emitHead: true, logits: logits)
    }

    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode _: PrefillOutputMode,
                        config prefillConfig: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        guard prefillConfig.mode == .chunked else {
            throw PrefillError.chunkedUnsupported("Inkling chunked prefill requires chunked mode")
        }
        guard startPosition == state.position else {
            throw PrefillError.prefillCursorMismatch(
                "Inkling prefill cursor \(state.position) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported("Inkling prefill exceeds maxContext \(maxContext)")
        }
        prefillClock.active = true
        // However it ends, nothing read ahead for a block is wanted once the
        // prompt is done: the token's last visit has nothing after it, and a
        // cancel between blocks never reaches `execute`. Decode's lookahead
        // only hints storage and is not affected.
        defer {
            prefillClock.active = false
            model.cancelRoutedExpertReadAhead()
        }
        let all = Array(tokens)
        var done = 0
        while done < all.count {
            try Task.checkCancellation()
            let count = min(rowCapacity, all.count - done)
            try await execute(tokens: Array(all[done..<(done + count)]), startPosition: startPosition + done,
                              emitHead: done + count == all.count, logits: logits)
            for completed in (done + 1)...(done + count) { onProgress(completed) }
            done += count
        }
        return PrefillResult(newPosition: startPosition + all.count, seed: .logitsWritten)
    }

    // MARK: Encoding helpers

    /// output[rows, outRows] = input[rows, columns] * W^T, FP32 in and out.
    /// A block goes through the batched matmul on FP16 copies.
    private func project(_ pass: InklingPass, _ weights: AffineView, input: MTLBuffer,
                         output: MTLBuffer, outRows: Int, columns: Int, rows: Int) {
        if rows > 1, let batched {
            let view = weights.view
            if batched.affine.accepts(scalesOffset: Int(view.scaleOffset), biasesOffset: Int(view.biasOffset),
                                      xOffset: 0, yOffset: 0, m: rows, n: outRows, k: columns) {
                kernels.threads(pass, "ink_to_half",
                                [.buffer(input), .buffer(halfInput), .u32(rows * columns)],
                                grid: (rows * columns, 1, 1))
                pass.endEncoding()
                batched.affine.encode(commandBuffer: pass.commandBuffer,
                                      weights: view.buffer, weightsOffset: Int(view.offset),
                                      scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                                      biases: view.buffer, biasesOffset: Int(view.biasOffset),
                                      x: halfInput, y: halfOutput, m: rows, n: outRows, k: columns)
                kernels.threads(pass, "ink_to_float",
                                [.buffer(halfOutput), .buffer(output), .u32(rows * outRows)],
                                grid: (rows * outRows, 1, 1))
                return
            }
        }
        kernels.groups(pass, "ink_affine4_gemv",
                       [weights.weights, weights.scales, weights.biases, .buffer(input), .buffer(output),
                        .u32(columns), .u32(outRows), .u32(columns)],
                       grid: (outRows, rows), threads: 32)
    }

    private func norm(_ pass: InklingPass, _ input: Rows, weight: InklingArgument?,
                      output: MTLBuffer, width: Int, rows: Int) {
        kernels.groups(pass, "ink_rms_norm",
                       [input.argument, weight ?? .buffer(one), .buffer(output),
                        .u32(width), .f32(1e-6), .u32(weight == nil ? 0 : 1), .u32(input.stride)],
                       grid: (rows, 1), threads: 256)
    }

    private func depthAttention(_ pass: InklingPass, blocks: MTLBuffer, depth: Int,
                                partial: MTLBuffer, query: InklingArgument, queries: Int,
                                output: MTLBuffer, rows: Int) {
        kernels.groups(pass, "ink_depth_attention",
                       [.buffer(blocks), .buffer(partial), query, .buffer(output),
                        .u32(config.hiddenSize), .u32(depth), .u32(queries), .u32(rows)],
                       grid: (rows, 1), threads: 256)
    }

    /// out = (1 - tanh(F g)) state + tanh(G g) message, with g from the two
    /// rows at unit RMS. `message` is which of a row's four messages.
    private func messageGate(_ pass: InklingPass, _ gate: MessageGate,
                             state: Rows, message: Int, output: Rows, rows: Int) {
        let width = config.hiddenSize
        let floatBytes = MemoryLayout<Float>.stride
        let messageArgument = InklingArgument.buffer(messages, message * width * floatBytes)
        kernels.groups(pass, "ink_gate_scales",
                       [state.argument, messageArgument, .buffer(gateScales), .u32(width), .u32(state.stride),
                        .u32(4 * width), .f32(1e-6)], grid: (rows, 1), threads: 256)
        kernels.groups(pass, "ink_gate_down",
                       [gate.down, state.argument, messageArgument, .buffer(gateScales), .buffer(gateLow),
                        .u32(width), .u32(loop.gateRank), .u32(state.stride), .u32(4 * width)],
                       grid: (loop.gateRank, rows), threads: 32)
        kernels.threads(pass, "ink_gate_mix",
                        [gate.forget, gate.gain, .buffer(gateLow), state.argument, messageArgument,
                         output.argument, .u32(width), .u32(loop.gateRank), .u32(state.stride),
                         .u32(4 * width), .u32(output.stride)], grid: (width, rows, 1))
    }

    private func staticConv(_ pass: InklingPass, input: MTLBuffer, taps: InklingArgument,
                            state: MTLBuffer, output: MTLBuffer, channels: Int, rows: Int) {
        kernels.threads(pass, "ink_conv_static",
                        [.buffer(input), .buffer(state), taps, .buffer(output), .u32(channels)],
                        grid: (channels, rows, 1))
        kernels.threads(pass, "ink_conv_advance",
                        [.buffer(input), .buffer(state), .u32(channels), .u32(rows)], grid: (channels, 1, 1))
    }

    private func copy(_ pass: InklingPass, from: MTLBuffer, fromOffset: Int = 0,
                      to: MTLBuffer, toOffset: Int = 0, count: Int) {
        kernels.threads(pass, "ink_add",
                        [.buffer(from, fromOffset), .buffer(from, fromOffset), .buffer(to, toOffset),
                         .u32(count), .u32(0)], grid: (count, 1, 1))
    }

    private func commitAndWait(_ pass: InklingPass) throws -> UInt64 {
        pass.endEncoding()
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        pass.commandBuffer.commit()
        pass.commandBuffer.waitUntilCompleted()
        let elapsed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start
        prefillClock.waited(pass.commandBuffer, wallNanos: elapsed)
        let gpu = pass.commandBuffer.gpuEndTime - pass.commandBuffer.gpuStartTime
        if gpu > 0 { totalGPULayerNanos &+= UInt64(gpu * 1e9) }
        totalGPULayerCommandBuffers &+= 1
        try checkCommandBufferError(pass.commandBuffer)
        return elapsed
    }

    // MARK: Forward

    /// A block writes its keys, values and convolution taps visit by visit.
    /// A block that is cancelled part way puts the taps back and leaves the
    /// sequence at the block's first position, so the conversation can go on
    /// from there. Any other failure drops the sequence: the next request
    /// reads its prompt again.
    private func execute(tokens: [Int32], startPosition: Int, emitHead: Bool,
                         logits: MTLBuffer) async throws {
        state.saveConvolutions()
        do {
            try await run(tokens: tokens, startPosition: startPosition, emitHead: emitHead, logits: logits)
        } catch let error as CancellationError {
            // Every committed command buffer has been waited for. Reads
            // started for the rest of the block are no longer wanted.
            model.cancelRoutedExpertReadAhead()
            lookahead.pending = nil
            state.restoreConvolutions()
            throw error
        } catch let error as PrefillError {
            // Argument errors are raised before anything is written.
            throw error
        } catch let error as GeneratorError {
            throw error
        } catch {
            state.reset()
            throw error
        }
    }

    private func run(tokens: [Int32], startPosition: Int, emitHead: Bool,
                     logits: MTLBuffer) async throws {
        let rows = tokens.count
        guard startPosition == state.position else {
            throw PrefillError.prefillCursorMismatch(
                "Inkling token position \(startPosition) != sequence position \(state.position)")
        }
        guard rows >= 1, rows <= rowCapacity, startPosition >= 0, startPosition + rows <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "Inkling positions \(startPosition)..<\(startPosition + rows) exceed maxContext \(maxContext)")
        }
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < loop.paddedVocabSize }) else {
            throw GeneratorError.invalidGenerationConfig(
                "Inkling token ID is outside vocab \(loop.paddedVocabSize)")
        }
        if emitHead, logits.length < config.vocabSize * MemoryLayout<Float16>.stride {
            throw PrefillError.chunkedUnsupported("Inkling logits buffer is smaller than the vocabulary")
        }
        try state.ensureCapacity(through: startPosition + rows, queue: context.queue)
        if rows > 1, batched != nil { batchedBlocks += 1 }

        let width = config.hiddenSize
        let floatBytes = MemoryLayout<Float>.stride
        // Full-attention queries and their relative states grow slowly with
        // position. The values are written before any buffer that reads them
        // is committed.
        let scales = logScale.contents().assumingMemoryBound(to: Float.self)
        let ids = tokenBuffer.contents().assumingMemoryBound(to: UInt32.self)
        for row in 0..<rows {
            scales[row] = 1 + Float(loop.logScalingAlpha)
                * log(max(Float(startPosition + row + 1) / Float(loop.logScalingFloor), 1))
            ids[row] = UInt32(tokens[row])
        }

        var pass = try InklingPass(context.queue)
        kernels.threads(pass, "ink_affine4_lookup",
                        [embedding.weights, embedding.scales, embedding.biases, .buffer(tokenBuffer),
                         .buffer(hidden), .u32(width)], grid: (width, rows, 1))
        norm(pass, Rows(buffer: hidden, stride: width), weight: embeddingNorm, output: hidden,
             width: width, rows: rows)

        depth = 0
        for layer in 0..<loop.loopStart {
            pass = try await visit(layer, layer: layer, rows: rows, startPosition: startPosition,
                                            hasMessages: false, pass: pass)
        }
        copy(pass, from: hidden, to: preLoopHidden, count: rows * width)
        let preLoopDepth = depth
        for layer in loop.loopLayers {
            pass = try await visit(layer, layer: layer, rows: rows, startPosition: startPosition,
                                            hasMessages: false, pass: pass)
        }
        for loopPass in 1..<loop.passes {
            // What the next pass reads from this one: the blocks it added
            // after its first, and its final state.
            let bankDepth = depth - preLoopDepth - 1
            for index in 0..<bankDepth {
                copy(pass, from: blocks, fromOffset: (preLoopDepth + 1 + index) * rows * width * floatBytes,
                     to: bankBlocks, toOffset: index * rows * width * floatBytes, count: rows * width)
            }
            norm(pass, Rows(buffer: hidden, stride: width), weight: nil, output: bankPartial,
                 width: width, rows: rows)
            copy(pass, from: preLoopHidden, to: hidden, count: rows * width)
            depth = preLoopDepth
            let base = config.numLayers + (loopPass - 1) * loop.loopLayers.count - loop.loopStart
            for layer in loop.loopLayers {
                kernels.groups(pass, "ink_depth_attention",
                               [.buffer(bankBlocks), .buffer(bankPartial),
                                .buffer(messageQuery.buffer,
                                        Int(messageQuery.offset) + (layer - loop.loopStart) * 4 * width * floatBytes),
                                .buffer(messages), .u32(width), .u32(bankDepth), .u32(4), .u32(rows)],
                               grid: (rows, 1), threads: 256)
                pass = try await visit(layer + base, layer: layer, rows: rows,
                                                startPosition: startPosition, hasMessages: true,
                                                pass: pass)
            }
        }
        for layer in loop.loopEnd..<config.numLayers {
            pass = try await visit(layer, layer: layer, rows: rows, startPosition: startPosition,
                                            hasMessages: false, pass: pass)
        }

        guard emitHead else {
            totalCb2Nanos &+= try commitAndWait(pass)
            state.advance(by: rows)
            return
        }
        // Only the last row needs the vocabulary head.
        norm(pass, Rows(buffer: hidden, stride: width), weight: nil, output: normed,
             width: width, rows: rows)
        depthAttention(pass, blocks: blocks, depth: depth, partial: normed,
                       query: finalQuery, queries: 1, output: finalMix, rows: rows)
        norm(pass, Rows(buffer: finalMix, offset: (rows - 1) * width * floatBytes, stride: width),
             weight: finalNorm, output: finalNormed, width: width, rows: 1)
        kernels.groups(pass, "ink_affine4_gemv_half",
                       [head.weights, head.scales, head.biases, .buffer(finalNormed), .buffer(logits),
                        .u32(width), .u32(config.vocabSize), .u32(width)],
                       grid: (config.vocabSize, 1), threads: 32)
        totalHeadNanos &+= try commitAndWait(pass)
        state.advance(by: rows)
    }

    /// One layer visit over `rows` token rows. `pass` already holds
    /// the end of the visit before; the returned buffer holds this visit's
    /// experts and tail and is committed by whatever comes next.
    private func visit(_ visit: Int, layer: Int, rows: Int, startPosition: Int, hasMessages: Bool,
                       pass start: InklingPass) async throws -> InklingPass {
        var pass = start
        // A long block reads most of the model; stopping should not wait for it.
        if rows > 1 { try Task.checkCancellation() }
        let views = layers[layer]
        let width = config.hiddenSize
        let floatBytes = MemoryLayout<Float>.stride
        let qDim = config.numHeads * config.headDim
        let kvDim = config.numKVHeads * config.headDim
        let lastPosition = startPosition + rows - 1

        // Inputs: depth attention over the blocks so far, then the loop's
        // messages, then the layer norm.
        norm(pass, Rows(buffer: hidden, stride: width), weight: nil, output: normed,
             width: width, rows: rows)
        var sources = [Rows](repeating: Rows(buffer: hidden, stride: width), count: 3)
        if let query = views.attentionQuery {
            let queries = views.full ? 3 : 1
            depthAttention(pass, blocks: blocks, depth: depth, partial: normed,
                           query: query, queries: queries, output: attentionInputs, rows: rows)
            sources = (0..<3).map {
                Rows(buffer: attentionInputs, offset: (views.full ? $0 : 0) * width * floatBytes,
                     stride: queries * width)
            }
        }
        if hasMessages {
            for index in 0..<3 {
                messageGate(pass, views.messageGates[index], state: sources[index], message: index,
                            output: Rows(buffer: gatedInputs, offset: index * width * floatBytes, stride: 3 * width),
                            rows: rows)
            }
            sources = (0..<3).map {
                Rows(buffer: gatedInputs, offset: $0 * width * floatBytes, stride: 3 * width)
            }
        }
        // A sliding layer outside the loop gives query, key and value the
        // same input.
        let shared = !views.full && !hasMessages
        let inputs = shared ? [queryInput, queryInput, queryInput] : [queryInput, keyInput, valueInput]
        for index in 0..<(shared ? 1 : 3) {
            norm(pass, sources[index], weight: views.inputNorms[views.full ? index : 0],
                 output: inputs[index], width: width, rows: rows)
        }

        project(pass, views.q, input: inputs[0], output: queryRows, outRows: qDim, columns: width, rows: rows)
        project(pass, views.gate, input: inputs[0], output: outputGate, outRows: qDim, columns: width, rows: rows)
        project(pass, views.r, input: inputs[0], output: relative,
                outRows: config.numHeads * loop.relativeDim, columns: width, rows: rows)
        project(pass, views.sink, input: inputs[0], output: sinks, outRows: config.numHeads,
                columns: width, rows: rows)
        project(pass, views.k, input: inputs[1], output: keyRows, outRows: kvDim, columns: width, rows: rows)
        project(pass, views.v, input: inputs[2], output: valueRows, outRows: kvDim, columns: width, rows: rows)
        let conv = state.conv[visit]
        if views.full {
            for (index, projectedRows, moving, output) in [(0, keyRows, movingKeyTaps, keyConv),
                                                           (1, valueRows, movingValueTaps, valueConv)] {
                let taps = views.dynamicTaps[index]
                project(pass, taps.projection, input: inputs[index + 1], output: moving,
                        outRows: config.numKVHeads * 4, columns: width, rows: rows)
                kernels.threads(pass, "ink_conv_dynamic",
                                [.buffer(projectedRows), .buffer(conv[index]), taps.fixed, .buffer(moving),
                                 .buffer(output), .u32(kvDim), .u32(config.numKVHeads)],
                                grid: (kvDim, rows, 1))
                kernels.threads(pass, "ink_conv_advance",
                                [.buffer(projectedRows), .buffer(conv[index]), .u32(kvDim), .u32(rows)],
                                grid: (kvDim, 1, 1))
            }
        } else {
            staticConv(pass, input: keyRows, taps: views.keyValueTaps[0], state: conv[0],
                       output: keyConv, channels: kvDim, rows: rows)
            staticConv(pass, input: valueRows, taps: views.keyValueTaps[1], state: conv[1],
                       output: valueConv, channels: kvDim, rows: rows)
        }
        let rowScale = views.full ? logScale : one
        kernels.groups(pass, "ink_head_clip",
                       [.buffer(queryRows), views.qClip, .buffer(rowScale), .u32(config.headDim),
                        .u32(config.numHeads), .f32(Float(loop.queryKeyClip)), .u32(1)],
                       grid: (config.numHeads, rows), threads: 32)
        kernels.groups(pass, "ink_head_clip",
                       [.buffer(keyConv), .buffer(one), .buffer(one), .u32(config.headDim),
                        .u32(config.numKVHeads), .f32(Float(loop.queryKeyClip)), .u32(0)],
                       grid: (config.numKVHeads, rows), threads: 32)
        norm(pass, Rows(buffer: valueConv, stride: config.headDim), weight: nil, output: valueConv,
             width: config.headDim, rows: rows * config.numKVHeads)
        kernels.groups(pass, "ink_head_clip",
                       [.buffer(relative), views.relClip, .buffer(rowScale), .u32(loop.relativeDim),
                        .u32(config.numHeads), .f32(Float(loop.relativeClip)), .u32(1)],
                       grid: (config.numHeads, rows), threads: 32)
        kernels.threads(pass, "ink_relative_bias",
                        [.buffer(relative), views.relProjection, .buffer(bias), .u32(loop.relativeDim),
                         .u32(loop.relativeExtent)], grid: (loop.relativeExtent, config.numHeads * rows, 1))

        // Attention over this visit's own keys and values. Every row of the
        // block is stored first; a row never sees the ones after it.
        let ring = state.isRing(visit)
        let capacity = state.capacity(visit)
        let slots = state.slots(visit, through: lastPosition)
        for (source, cache) in [(keyConv, state.keys[visit]), (valueConv, state.values[visit])] {
            kernels.threads(pass, "ink_cache_store",
                            [.buffer(source), .buffer(cache), .u32(kvDim),
                             .u32(state.slot(visit, position: startPosition)), .u32(capacity)],
                            grid: (kvDim, rows, 1))
        }
        let perDispatch = max(1, min(rows, scoreCapacity / (config.numHeads * slots)))
        var first = 0
        while first < rows {
            let count = min(perDispatch, rows - first)
            kernels.threads(pass, "ink_attention_scores",
                            [.buffer(queryRows, first * qDim * floatBytes), .buffer(state.keys[visit]),
                             .buffer(bias, first * config.numHeads * loop.relativeExtent * floatBytes),
                             .buffer(scores), .u32(config.headDim), .u32(config.numHeads),
                             .u32(config.numKVHeads), .u32(slots), .u32(capacity), .u32(ring ? 1 : 0),
                             .u32(startPosition + first), .u32(lastPosition),
                             .u32(ring ? window : 0), .u32(loop.relativeExtent),
                             .f32(Float(config.attentionScale))],
                            grid: (slots, config.numHeads, count))
            kernels.groups(pass, "ink_attention_mix",
                           [.buffer(scores), .buffer(state.values[visit]),
                            .buffer(sinks, first * config.numHeads * floatBytes),
                            .buffer(attention, first * qDim * floatBytes),
                            .u32(config.headDim), .u32(config.numHeads), .u32(config.numKVHeads), .u32(slots)],
                           grid: (config.numHeads, count), threads: 256)
            first += count
        }
        kernels.threads(pass, "ink_sigmoid_gate",
                        [.buffer(attention), .buffer(outputGate), .buffer(gatedAttention), .u32(rows * qDim)],
                        grid: (rows * qDim, 1, 1))
        project(pass, views.o, input: gatedAttention, output: projected, outRows: width,
                columns: qDim, rows: rows)
        staticConv(pass, input: projected, taps: views.attentionTaps, state: conv[2],
                   output: attentionConv, channels: width, rows: rows)

        // The first layer of a block stores its input for the layers after
        // it and starts a fresh residual.
        if layer % loop.blockLayers == 0 {
            copy(pass, from: normed, to: blocks, toOffset: depth * rows * width * floatBytes,
                 count: rows * width)
            depth += 1
            copy(pass, from: attentionConv, to: residual, count: rows * width)
        } else {
            kernels.threads(pass, "ink_add",
                            [.buffer(hidden), .buffer(attentionConv), .buffer(residual), .u32(rows * width),
                             .u32(1)], grid: (rows * width, 1, 1))
        }
        norm(pass, Rows(buffer: residual, stride: width), weight: nil, output: mlpPartial,
             width: width, rows: rows)
        depthAttention(pass, blocks: blocks, depth: depth, partial: mlpPartial,
                       query: views.mlpQuery, queries: 1, output: mlpInput, rows: rows)
        var mlpSource = mlpInput
        if hasMessages {
            messageGate(pass, views.messageGates[3], state: Rows(buffer: mlpInput, stride: width),
                        message: 3, output: Rows(buffer: mlpGated, stride: width), rows: rows)
            mlpSource = mlpGated
        }
        norm(pass, Rows(buffer: mlpSource, stride: width), weight: views.mlpNorm, output: mlpNormed,
             width: width, rows: rows)
        totalCb1Nanos &+= try commitAndWait(pass)
        afterAttention?(visit)

        pass = try await experts(visit: visit, layer: layer, views: views, rows: rows)
        staticConv(pass, input: mlpOutput, taps: views.mlpTaps, state: conv[3],
                   output: mlpConv, channels: width, rows: rows)
        kernels.threads(pass, "ink_add",
                        [.buffer(residual), .buffer(mlpConv), .buffer(hidden), .u32(rows * width), .u32(1)],
                        grid: (rows * width, 1, 1))
        return pass
    }

    /// Reads a layer's experts, in the order given. They stay bound until the
    /// layer's next read.
    private func fetch(layer: Int, experts: [Int]) async throws -> [TensorView] {
        let ioStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        guard let plan = try model.planRoutedExperts(layer: layer, experts: experts) else {
            throw ModelError.indexCorrupt(detail: "Inkling layer \(layer) could not plan its experts")
        }
        let blobs = try await model.fetchRoutedExperts(plan: plan)
        let ioElapsed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - ioStart
        totalIoNanos &+= ioElapsed
        prefillClock.expertRead(nanos: ioElapsed)
        let misses = plan.misses.count
        totalRoutedExpertReads &+= UInt64(misses)
        totalRoutedExpertCacheMisses &+= UInt64(misses)
        totalRoutedExpertCacheHits &+= UInt64(plan.hits)
        if let bytes = try? model.routedExpertAdviceByteEstimate(layer: layer, missCount: misses) {
            totalRoutedExpertBytes &+= bytes
        }
        return blobs
    }

    /// The top-K experts of one feed-forward input row under a layer's
    /// router, best first, and the sigmoid scores of all experts. Scores
    /// are sigmoids; the correction bias only affects which experts are
    /// chosen, not their weights.
    private func route(_ views: LayerViews, input: UnsafePointer<Float>,
                       logit: inout [Float]) -> (chosen: ArraySlice<Int>, score: [Float]) {
        let expertCount = config.numExperts
        vDSP_mmul(views.routerWeight, 1, input, 1, &logit, 1,
                  vDSP_Length(expertCount), 1, vDSP_Length(config.hiddenSize))
        let score = logit.map { 1 / (1 + exp(-$0)) }
        let chosen = (0..<expertCount).sorted {
            let a = score[$0] + views.routerBias[$0], b = score[$1] + views.routerBias[$1]
            return a != b ? a > b : $0 < $1
        }.prefix(config.topKExperts)
        return (chosen, score)
    }

    /// Guesses the next visit's experts from this visit's single
    /// feed-forward input row and asks storage to start reading them.
    private func readAheadNextVisit(after visit: Int) {
        guard lookahead.enabled, let next = nextLayerOfVisit[visit] else { return }
        var logit = [Float](repeating: 0, count: config.numExperts)
        let predicted = Array(route(layers[next], input: mlpNormed.contents().assumingMemoryBound(to: Float.self),
                                    logit: &logit).chosen)
        // A hint, not a read: storage starts on the pages and nothing waits
        // for them; the visit reads whatever is still missing when it
        // routes. Background `pread`s gained nothing here, because copying
        // the guesses out cost as much GPU time as the reads saved.
        _ = try? model.adviseRoutedExperts(layer: next, experts: predicted)
        lookahead.noteRead()
        lookahead.pending = ExpertLookahead.Pending(layer: next, experts: predicted, reads: nil)
    }

    /// A block's guess at the next visit's experts: each row through the
    /// next visit's router, or, once the block picks at least twice as many
    /// experts as there are, all of them. Started at utility priority after
    /// this visit's own reads, so it never competes with them.
    private func readAheadNextVisit(after visit: Int, rows: Int) {
        guard blockReadAhead, let next = nextLayerOfVisit[visit] else { return }
        let expertCount = config.numExperts
        var experts: [Int]
        if rows * config.topKExperts >= 2 * expertCount {
            experts = Array(0..<expertCount)
        } else {
            var chosen = Set<Int>()
            var logit = [Float](repeating: 0, count: expertCount)
            let input = mlpNormed.contents().assumingMemoryBound(to: Float.self)
            for row in 0..<rows {
                chosen.formUnion(route(layers[next], input: input + row * config.hiddenSize, logit: &logit).chosen)
            }
            experts = chosen.sorted()
        }
        try? model.readAheadRoutedExperts(layer: next, experts: experts, lowPriority: true)
    }

    /// Routes every row, reads the experts they chose, and encodes the
    /// feed-forward block into a new command buffer that leaves the weighted
    /// sum in `mlpOutput`.
    private func experts(visit: Int, layer: Int, views: LayerViews, rows: Int) async throws -> InklingPass {
        let width = config.hiddenSize
        let floatBytes = MemoryLayout<Float>.stride
        let intermediate = config.moeIntermediateSize
        let topK = config.topKExperts
        let expertCount = config.numExperts

        // Routing.
        struct Pair { let row: Int; let slot: Int; let weight: Float }
        var byExpert: [Int: [Pair]] = [:]
        let input = mlpNormed.contents().assumingMemoryBound(to: Float.self)
        var logit = [Float](repeating: 0, count: expertCount)
        for row in 0..<rows {
            let (chosen, score) = route(views, input: input + row * width, logit: &logit)
            let total = chosen.reduce(Float(0)) { $0 + score[$1] } + 1e-20
            for (slot, expert) in chosen.enumerated() {
                byExpert[expert, default: []].append(
                    Pair(row: row, slot: slot, weight: score[expert] / total * Float(loop.routeScale)))
            }
        }
        let chosenExperts = byExpert.keys.sorted()
        // Only decode's guesses are scored; a block's cover most experts.
        if let guess = lookahead.pending {
            lookahead.pending = nil
            if rows == 1, guess.layer == layer {
                lookahead.record(predicted: guess.experts, actual: chosenExperts)
            }
        }
        // Pairs in expert order, and where each token's pairs landed.
        let pairRow = pairRowIndex.contents().assumingMemoryBound(to: UInt32.self)
        let pairWeight = pairWeights.contents().assumingMemoryBound(to: Float.self)
        let tokenPair = tokenPairs.contents().assumingMemoryBound(to: UInt32.self)
        let tokenWeight = tokenWeights.contents().assumingMemoryBound(to: Float.self)
        var groups: [(expert: Int, start: Int, count: Int)] = []
        var cursor = 0
        for expert in chosenExperts {
            let pairs = byExpert[expert]!
            groups.append((expert, cursor, pairs.count))
            for pair in pairs {
                pairRow[cursor] = UInt32(pair.row)
                pairWeight[cursor] = pair.weight
                tokenPair[pair.row * topK + pair.slot] = UInt32(cursor)
                tokenWeight[pair.row * topK + pair.slot] = pair.weight
                cursor += 1
            }
        }

        let useBatched = rows > 1 && batched != nil
        // One read covers at most as many experts as the cache has slots.
        // Mapped layer files stay bound, so a block's reads share a command
        // buffer; slots are reused, so each batch runs before the next read.
        let mapped = try model.routedExpertsAreMapped(layer: layer)
        let slotLimit = max(1, model.routedExpertCacheSlotCount(layer: layer) ?? topK)
        var pass = try InklingPass(context.queue)
        // A decoded token: its eight experts in two dispatches.
        if fusedDecode, rows == 1, topK == 8, groups.count == 8, slotLimit >= 8 {
            let blobs = try await fetch(layer: layer, experts: groups.map(\.expert))
            readAheadNextVisit(after: visit)
            let bound = blobs.map { InklingArgument.buffer($0.buffer, Int($0.offset)) }
            let format = views.nvfp4 ? "nvfp4" : "mxfp8"
            let span = rowKernels ? InklingKernels.rowsPerGroup : 1
            let threads = rowKernels ? InklingKernels.rowGroupThreads : 32
            let rowsName = rowKernels ? "rows_" : ""
            kernels.groups(pass, "ink_expert_phase1_" + rowsName + format,
                           bound + [.buffer(mlpNormed), .buffer(activation),
                                    .floats(groups.map { views.nvfp4 ? views.gateUpFactors[$0.expert] : 1 }),
                                    .u32(views.gateUpCodes), .u32(views.gateUpScales), .u32(width),
                                    .u32(intermediate), .f32(Float(loop.gateSoftcap)), .f32(Float(loop.upSoftcap))],
                           grid: (intermediate / span, 8), threads: threads)
            kernels.groups(pass, "ink_expert_phase2_" + rowsName + format,
                           bound + [.buffer(activation), .buffer(mlpOutput),
                                    .floats(groups.enumerated().map {
                                        (views.nvfp4 ? views.downFactors[$1.expert] : 1) * pairWeight[$0]
                                    }),
                                    .u32(views.downCodes), .u32(views.downScales), .u32(intermediate)],
                           grid: (width / span, 1), threads: threads)
            return pass
        }
        if useBatched {
            kernels.threads(pass, "ink_gather_rows_half",
                            [.buffer(mlpNormed), .buffer(pairRowIndex), .buffer(pairRows), .u32(width)],
                            grid: (width, cursor, 1))
        } else {
            kernels.threads(pass, "ink_zero", [.buffer(mlpOutput), .u32(rows * width)],
                            grid: (rows * width, 1, 1))
        }
        // The batches below each wait for their own reads; the rest of the
        // block's experts start now, so the drive always has work queued.
        if rows > 1, blockReadAhead, groups.count > slotLimit {
            try? model.readAheadRoutedExperts(layer: layer, experts: groups[slotLimit...].map(\.expert))
        }
        var next = 0
        while next < groups.count {
            let batch = Array(groups[next..<min(groups.count, next + slotLimit)])
            if next > 0, !mapped {
                totalCb2Nanos &+= try commitAndWait(pass)
                pass = try InklingPass(context.queue)
            }
            let blobs = try await fetch(layer: layer, experts: batch.map(\.expert))

            if useBatched, let batched {
                let half = MemoryLayout<Float16>.stride
                for (group, blob) in zip(batch, blobs) {
                    let base = Int(blob.offset)
                    batched.expert(pass, nvfp4: views.nvfp4,
                                   codes: (blob.buffer, base + views.gateUpCodes),
                                   scales: (blob.buffer, base + views.gateUpScales),
                                   x: (pairRows, group.start * width * half),
                                   y: (pairGateUp, group.start * 2 * intermediate * floatBytes),
                                   m: group.count, n: 2 * intermediate, k: width,
                                   factor: views.nvfp4 ? views.gateUpFactors[group.expert] : 1)
                }
                let first = batch[0].start
                let count = batch.reduce(0) { $0 + $1.count }
                kernels.threads(pass, "ink_expert_activation_half",
                                [.buffer(pairGateUp, first * 2 * intermediate * floatBytes),
                                 .buffer(pairActivation, first * intermediate * half), .u32(intermediate),
                                 .f32(Float(loop.gateSoftcap)), .f32(Float(loop.upSoftcap))],
                                grid: (intermediate, count, 1))
                for (group, blob) in zip(batch, blobs) {
                    let base = Int(blob.offset)
                    batched.expert(pass, nvfp4: views.nvfp4,
                                   codes: (blob.buffer, base + views.downCodes),
                                   scales: (blob.buffer, base + views.downScales),
                                   x: (pairActivation, group.start * intermediate * half),
                                   y: (pairDown, group.start * width * floatBytes),
                                   m: group.count, n: width, k: intermediate,
                                   factor: views.nvfp4 ? views.downFactors[group.expert] : 1)
                }
            } else {
                let kernel = views.nvfp4 ? "ink_expert_gemv_nvfp4" : "ink_expert_gemv_mxfp8"
                for (group, blob) in zip(batch, blobs) {
                    let base = Int(blob.offset)
                    let indexOffset = group.start * MemoryLayout<UInt32>.stride
                    kernels.groups(pass, kernel,
                                   [.buffer(blob.buffer, base + views.gateUpCodes),
                                    .buffer(blob.buffer, base + views.gateUpScales),
                                    .buffer(mlpNormed), .buffer(gateUp), .buffer(pairRowIndex, indexOffset),
                                    .buffer(one), .u32(width), .u32(2 * intermediate),
                                    .f32(views.nvfp4 ? views.gateUpFactors[group.expert] : 1), .u32(0)],
                                   grid: (2 * intermediate, group.count), threads: 32)
                    kernels.threads(pass, "ink_expert_activation",
                                    [.buffer(gateUp), .buffer(activation), .u32(intermediate),
                                     .f32(Float(loop.gateSoftcap)), .f32(Float(loop.upSoftcap))],
                                    grid: (intermediate, group.count, 1))
                    kernels.groups(pass, kernel,
                                   [.buffer(blob.buffer, base + views.downCodes),
                                    .buffer(blob.buffer, base + views.downScales),
                                    .buffer(activation), .buffer(mlpOutput), .buffer(pairRowIndex, indexOffset),
                                    .buffer(pairWeights, group.start * floatBytes), .u32(intermediate),
                                    .u32(width), .f32(views.nvfp4 ? views.downFactors[group.expert] : 1),
                                    .u32(1)],
                                   grid: (width, group.count), threads: 32)
                }
            }
            next += batch.count
        }
        if rows == 1 { readAheadNextVisit(after: visit) } else { readAheadNextVisit(after: visit, rows: rows) }
        if useBatched {
            kernels.threads(pass, "ink_expert_reduce",
                            [.buffer(pairDown), .buffer(tokenPairs), .buffer(tokenWeights), .buffer(mlpOutput),
                             .u32(width), .u32(topK)], grid: (width, rows, 1))
        }
        return pass
    }
}
