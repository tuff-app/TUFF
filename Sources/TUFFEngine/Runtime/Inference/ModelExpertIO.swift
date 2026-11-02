import Foundation
import Metal
import TUFFFormat

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }

    public init(layer: Int, cachePlan: ExpertCachePlan) {
        self.layer = layer
        self.cachePlan = cachePlan
    }
}

extension Model {
    public var expertCacheAllocatedBytes: UInt64 {
        streamersQueue.sync {
            streamersBox.streamers.compactMap { $0 }.reduce(0) {
                $0 + $1.diagnosticSlotScratchBytes
            }
        }
    }

    public var expertReadMetrics: ExpertReadMetrics {
        streamersQueue.sync {
            var total = streamersBox.retiredReadMetrics
            for streamer in streamersBox.streamers.compactMap({ $0 }) { total.add(streamer.readMetrics) }
            return total
        }
    }

    func gptOssRoutedExpertOffsets(layer: Int) throws -> GPTOSSExpertOffsets {
        let experts = packedExpertsLayout.layers[layer].experts
        guard let first = experts.first else {
            throw ModelError.indexCorrupt(detail: "GPT-OSS layer has no routed experts")
        }
        let names = [
            "mlp1", "mlp1_scales", "mlp1_bias",
            "mlp2", "mlp2_scales", "mlp2_bias",
        ]
        var resolved: [String: Int] = [:]
        for name in names {
            guard let tensor = first.subTensors[name],
                  let offset = Int(exactly: tensor.offset) else {
                throw ModelError.indexCorrupt(
                    detail: "GPT-OSS routed expert is missing \(name)")
            }
            resolved[name] = offset
        }
        for expert in experts.dropFirst() {
            for name in names where expert.subTensors[name]?.offset
                != first.subTensors[name]?.offset {
                throw ModelError.indexCorrupt(
                    detail: "GPT-OSS routed expert \(name) offsets differ within layer")
            }
        }
        return GPTOSSExpertOffsets(
            mlp1Weights: resolved["mlp1"]!,
            mlp1Scales: resolved["mlp1_scales"]!,
            mlp1Bias: resolved["mlp1_bias"]!,
            mlp2Weights: resolved["mlp2"]!,
            mlp2Scales: resolved["mlp2_scales"]!,
            mlp2Bias: resolved["mlp2_bias"]!)
    }

    public func routedExpertOffsets(layer: Int) -> MoEExpertOffsets {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            guard let tensor = expert.subTensors[role],
                  let offset = UInt32(exactly: tensor.offset) else {
                preconditionFailure("invalid routed expert metadata for role \(role)")
            }
            return offset
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"))
    }

    func routedExpertBitWidths(layer: Int) -> MoEExpertBitWidths {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func bits(_ role: String) -> UInt32 {
            guard let tensor = expert.subTensors[role] else {
                preconditionFailure("missing routed expert metadata for role \(role)")
            }
            let width = tensor.bits ?? manifest.quant?.routedExpert.weightBits ?? 4
            guard width == 4 || width == 8 else {
                preconditionFailure("unsupported routed expert width \(width) for \(role)")
            }
            return UInt32(width)
        }
        return MoEExpertBitWidths(gate: bits("gate"), up: bits("up"), down: bits("down"))
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        packedExpertsLayout.layers[layer].experts.map(\.offset)
    }

    public func adviseRoutedExperts(layer: Int,
                                    experts: [Int]) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.adviseExpertMisses(experts: experts)
    }

    /// Starts reading a layer's uncached experts into the file cache in the
    /// background. See `PreadExpertStreamer.readAheadExpertMisses`.
    public func readAheadRoutedExperts(layer: Int, experts: [Int], concurrency: Int = 8,
                                       lowPriority: Bool = false) throws {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        streamer.readAheadExpertMisses(experts: experts, concurrency: concurrency,
                                       lowPriority: lowPriority)
    }

    public func routedExpertAdviceByteEstimate(layer: Int,
                                               missCount: Int) throws -> UInt64 {
        guard missCount > 0 else { return 0 }
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return UInt64(missCount) * streamer.layout.expertStride
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan? {
        try planRoutedExperts(layer: layer, experts: experts,
                              avoidingSlots: avoidingSlots, purpose: .demand)
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = [],
                                  purpose: ExpertReadPurpose) throws -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        return RoutedExpertFetchPlan(
            layer: layer,
            cachePlan: streamer.planExpertsCached(experts: experts, avoidingSlots: validSlots, purpose: purpose))
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: validSlots)
        else {
            return nil
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    public func routedExpertCacheSlotCount(layer _: Int) -> Int? {
        // Match the allocation in `openLayerLocked`. Prefill uses this value
        // to size overlapping fetch plans, so reporting the requested value
        // could otherwise exceed the actual streamer's capacity.
        effectiveExpertCacheSlotCount
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return try await fetchRoutedExperts(
            streamer: streamer, layer: plan.layer, cachePlan: plan.cachePlan)
    }

    /// Retains only the synchronized streamer and immutable plan across the
    /// background read. Resolve model ownership on the runner's thread. The
    /// caller must drain this operation before planning the same layer again.
    func expertPrefetchOperation(plan: RoutedExpertFetchPlan) throws
        -> @Sendable () throws -> Void {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        let cachePlan = plan.cachePlan
        return {
            _ = try streamer.executeExpertCachePlan(cachePlan, purpose: .prefetch)
        }
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let cachePlan = streamer.planExpertsCached(experts: experts)
        return try await fetchRoutedExperts(
            streamer: streamer, layer: layer, cachePlan: cachePlan)
    }

    private func fetchRoutedExperts(streamer: PreadExpertStreamer,
                                    layer: Int,
                                    cachePlan: ExpertCachePlan) async throws -> [TensorView] {
        // Cache hits are already GPU-visible. Avoid suspending the runner
        // and dispatching an I/O worker when there is nothing to read.
        if cachePlan.misses.isEmpty {
            return Self.makeExpertViews(
                streamer.expertCachePlanBuffers(cachePlan),
                layer: layer, experts: cachePlan.experts)
        }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.executeExpertCachePlan(cachePlan)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: layer,
                        experts: cachePlan.experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: GTurboFormatV1.DType.u32.rawValue)
        }
    }
}
