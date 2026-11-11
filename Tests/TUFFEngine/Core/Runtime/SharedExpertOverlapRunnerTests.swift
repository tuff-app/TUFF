import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// The shared-expert overlap and the routed phase's failure cleanup, driven
/// through real toy runners. The observer reports each step of a layer's
/// routed phase in order and can throw at any of them, which the runner
/// treats as a failure at that step.
///
/// Flash Next's toy streams its experts through 16 pread slots, so a second
/// routed tile is fetched while the first is still on the GPU. Qwen 3.6's toy
/// keeps its experts resident and drains before each new tile. There is no
/// runnable Gemma 4 26B-A4B toy, so its sandwich-norm shared expert is covered
/// only by the real-model comparison.
@Suite(.serialized) struct SharedExpertOverlapRunnerTests {

    enum Fixture: String, CaseIterable {
        case flashNext
        case qwen36
    }

    private struct InjectedFailure: Error, Equatable {}

    /// Records steps and throws at the chosen one. Optionally cancels the
    /// running task instead of throwing.
    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var recorded: [PrefillRoutedPhaseStep] = []
        var failAt: PrefillRoutedPhaseStep?
        var cancelAt: PrefillRoutedPhaseStep?

        var steps: [PrefillRoutedPhaseStep] { lock.withLock { recorded } }

        func observe(_ step: PrefillRoutedPhaseStep) throws {
            lock.withLock { recorded.append(step) }
            if step == cancelAt {
                withUnsafeCurrentTask { $0?.cancel() }
            }
            if step == failAt {
                throw InjectedFailure()
            }
        }
    }

    private struct Runner: @unchecked Sendable {
        let directory: URL
        let context: MetalContext
        let runner: RealForwardRunner
        let logits: MTLBuffer
        let vocab: Int
    }

    private static let chunkTokens = 32
    /// 40 tokens: a 32-token chunk on the batched routed kernels and an
    /// 8-token chunk on the per-pair kernels.
    private static let prompt = (0..<40).map { Int32(($0 * 7 + 11) % 31) }

    private func makeRunner(_ fixture: Fixture, overlap: Bool) throws -> Runner {
        let context = try MetalContext()
        let directory: URL
        let model: Model
        switch fixture {
        case .flashNext:
            let config = ArchConfig.qwen4ExpToy()
            directory = try Qwen4ExpToySynthetic.write(config: config)
            model = try Model.load(directoryURL: directory, device: context.device,
                                   expecting: config, streamingMode: .pread(slotCount: 16)).withSlotCache()
        case .qwen36:
            directory = try QwenToySynthetic.write()
            model = try Model.load(directoryURL: directory, device: context.device,
                                   expecting: .qwen36Toy()).withSlotCache()
        }
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 128,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true),
            smallBlockPrefill: .disabled,
            sharedExpertOverlap: SharedExpertOverlapPolicy(enabled: overlap))
        let vocab = model.config.vocabSize
        let logits = try #require(context.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride, options: .storageModeShared))
        return Runner(directory: directory, context: context, runner: runner,
                      logits: logits, vocab: vocab)
    }

    private func prefill(_ runner: Runner) async throws -> [Float] {
        runner.runner.reset()
        _ = try await runner.runner.prefillChunked(
            tokens: Self.prompt[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: Self.chunkTokens),
            into: runner.logits, onProgress: { _ in })
        return Fp16Buffer.read(runner.logits, count: runner.vocab)
    }

    private static func argmax(_ values: [Float]) -> Int32 {
        Int32(values.indices.max { values[$0] < values[$1] } ?? 0)
    }

    /// Prefill logits, then greedy tokens decoded after them.
    private func prefillThenGreedy(_ runner: Runner, steps: Int) async throws -> ([Float], [Int32]) {
        let logits = try await prefill(runner)
        var tokens = [Self.argmax(logits)]
        for step in 0..<(steps - 1) {
            try await runner.runner.produce(token: tokens.last!, position: Self.prompt.count + step,
                                            into: runner.logits)
            tokens.append(Self.argmax(Fp16Buffer.read(runner.logits, count: runner.vocab)))
        }
        return (logits, tokens)
    }

    private func remove(_ runners: Runner...) {
        for runner in runners { try? FileManager.default.removeItem(at: runner.directory) }
    }

    @Test func failedDecodeWaitsForRegisteredReadBeforeReleasingLease() async throws {
        let fixture = try makeRunner(.flashNext, overlap: false)
        defer { remove(fixture) }
        let registered = DispatchSemaphore(value: 0)
        let returned = DispatchSemaphore(value: 0)
        let releaseRead = DispatchSemaphore(value: 0)
        func wait(_ semaphore: DispatchSemaphore, timeout: DispatchTimeInterval) async -> Bool {
            await withCheckedContinuation { continuation in
                DispatchQueue.global().async {
                    continuation.resume(returning: semaphore.wait(timeout: .now() + timeout) == .success)
                }
            }
        }
        fixture.runner.decodeLookaheadObserver = { group in
            group.enter()
            DispatchQueue.global().async {
                releaseRead.wait()
                group.leave()
            }
            registered.signal()
            throw InjectedFailure()
        }
        let lease = FairForwardLease()
        let task = Task.detached {
            defer { returned.signal() }
            return try await lease.withLease {
                try await fixture.runner.produce(token: 1, position: 0, into: fixture.logits)
            }
        }
        let wasRegistered = await wait(registered, timeout: .seconds(30))
        #expect(wasRegistered, "toy decode must register a streamed expert read")
        if wasRegistered {
            #expect(!(await wait(returned, timeout: .milliseconds(100))),
                    "the failing decode must not release its lease while the read is pending")
        }
        releaseRead.signal()
        do {
            try await task.value
            Issue.record("expected injected decode failure")
        } catch {
            #expect(error is InjectedFailure)
        }
    }

    private func index(_ step: PrefillRoutedPhaseStep, in steps: [PrefillRoutedPhaseStep],
                       sourceLocation: SourceLocation = #_sourceLocation) -> Int {
        guard let index = steps.firstIndex(of: step) else {
            Issue.record("missing \(step)", sourceLocation: sourceLocation)
            return -1
        }
        return index
    }

    private func moeLayers(_ steps: [PrefillRoutedPhaseStep]) -> [Int] {
        var layers: [Int] = []
        for case .sharedSubmitted(let layer) in steps where !layers.contains(layer) {
            layers.append(layer)
        }
        return layers
    }

    // MARK: Numerics

    @Test(arguments: Fixture.allCases)
    func overlapMatchesTheSerializedSchedule(fixture: Fixture) async throws {
        let serialized = try makeRunner(fixture, overlap: false)
        let overlapped = try makeRunner(fixture, overlap: true)
        defer { remove(serialized, overlapped) }
        let reference = try await prefillThenGreedy(serialized, steps: 6)
        let actual = try await prefillThenGreedy(overlapped, steps: 6)
        // Same kernels in the same GPU order, so the results are identical.
        #expect(actual.0 == reference.0)
        #expect(actual.1 == reference.1)
        #expect(actual.0.allSatisfy { $0.isFinite })
    }

    // MARK: Ordering

    @Test(arguments: Fixture.allCases)
    func overlapPreparesBeforeTheJoinAndDispatchesAfterIt(fixture: Fixture) async throws {
        let runner = try makeRunner(fixture, overlap: true)
        defer { remove(runner) }
        let recorder = Recorder()
        runner.runner.routedPhaseObserver = recorder.observe
        _ = try await prefill(runner)
        let steps = recorder.steps
        let layers = moeLayers(steps)
        #expect(!layers.isEmpty)
        for layer in layers {
            let joined = index(.sharedJoined(layer: layer), in: steps)
            #expect(index(.sharedSubmitted(layer: layer), in: steps) < index(.metadataPrepared(layer: layer), in: steps))
            #expect(index(.metadataPrepared(layer: layer), in: steps) < joined)
            #expect(index(.fetchStarting(layer: layer, tile: 0), in: steps) < joined)
            #expect(index(.tileBound(layer: layer, tile: 0), in: steps) < joined)
            #expect(joined < index(.tileDispatched(layer: layer, tile: 0), in: steps))
        }
        #expect(!steps.contains { if case .drainedAfterFailure = $0 { true } else { false } })
    }

    @Test func theSerializedScheduleJoinsBeforeAnyPreparation() async throws {
        let runner = try makeRunner(.flashNext, overlap: false)
        defer { remove(runner) }
        let recorder = Recorder()
        runner.runner.routedPhaseObserver = recorder.observe
        _ = try await prefill(runner)
        let steps = recorder.steps
        for layer in moeLayers(steps) {
            #expect(index(.sharedJoined(layer: layer), in: steps)
                    < index(.metadataPrepared(layer: layer), in: steps))
        }
    }

    @Test func speculativeVerificationNeverOverlaps() {
        let policy = SharedExpertOverlapPolicy(enabled: true)
        #expect(policy.admits(speculativeVerification: false))
        #expect(!policy.admits(speculativeVerification: true))
        #expect(!SharedExpertOverlapPolicy.disabled.admits(speculativeVerification: false))
    }

    // MARK: Failure cleanup

    /// Fails at `step`, checks the original error escaped after cleanup, then
    /// reuses the same runner and expects the reference logits.
    private func failThenReuse(_ fixture: Fixture, overlap: Bool,
                               at step: PrefillRoutedPhaseStep,
                               expectedDrained: Int?,
                               sourceLocation: SourceLocation = #_sourceLocation) async throws
        -> [PrefillRoutedPhaseStep] {
        let reference = try makeRunner(fixture, overlap: false)
        let runner = try makeRunner(fixture, overlap: overlap)
        defer { remove(reference, runner) }
        let expected = try await prefill(reference)

        let recorder = Recorder()
        recorder.failAt = step
        runner.runner.routedPhaseObserver = recorder.observe
        await #expect(throws: InjectedFailure.self, sourceLocation: sourceLocation) {
            _ = try await prefill(runner)
        }
        let steps = recorder.steps
        let drained = steps.compactMap { step -> Int? in
            if case .drainedAfterFailure(_, let buffers) = step { buffers } else { nil }
        }
        #expect(drained.count == 1, "cleanup ran once", sourceLocation: sourceLocation)
        if let expectedDrained {
            #expect(drained.first == expectedDrained, sourceLocation: sourceLocation)
        }
        // Every buffer submitted before the failure was joined either in order
        // or by the cleanup. A join step is recorded after its wait, so a
        // failure reported there counts as joined.
        let dispatched = steps.filter { if case .tileDispatched = $0 { true } else { false } }.count
        let joined = steps.filter { if case .tileJoined = $0 { true } else { false } }.count
        let sharedLeft = steps.filter { if case .sharedSubmitted = $0 { true } else { false } }.count
            - steps.filter { if case .sharedJoined = $0 { true } else { false } }.count
        #expect(dispatched - joined + sharedLeft == (drained.first ?? -1),
                "dispatched \(dispatched), joined \(joined), shared left \(sharedLeft)",
                sourceLocation: sourceLocation)

        runner.runner.routedPhaseObserver = nil
        let reused = try await prefill(runner)
        #expect(reused == expected, "a runner reused after a failure matches", sourceLocation: sourceLocation)
        return steps
    }

    @Test func aFetchFailureDrainsThePendingTile() async throws {
        _ = try await failThenReuse(.flashNext, overlap: true,
                                    at: .fetchStarting(layer: 1, tile: 1),
                                    expectedDrained: 1)
    }

    @Test func aBindingOrAllocationFailureDrainsThePendingTile() async throws {
        _ = try await failThenReuse(.flashNext, overlap: false,
                                    at: .tileBound(layer: 2, tile: 1),
                                    expectedDrained: 1)
    }

    /// Qwen 3.6's toy has one routed tile per layer.
    @Test func singleTileFailuresDrainOnlyWhatIsInFlight() async throws {
        _ = try await failThenReuse(.qwen36, overlap: true,
                                    at: .fetchStarting(layer: 1, tile: 0),
                                    expectedDrained: 1)
        _ = try await failThenReuse(.qwen36, overlap: false,
                                    at: .tileBound(layer: 2, tile: 0),
                                    expectedDrained: 0)
        _ = try await failThenReuse(.qwen36, overlap: true,
                                    at: .tileDispatched(layer: 3, tile: 0),
                                    expectedDrained: 1)
    }

    @Test func aFailureAfterDispatchDrainsTheNewTile() async throws {
        _ = try await failThenReuse(.flashNext, overlap: true,
                                    at: .tileDispatched(layer: 0, tile: 0),
                                    expectedDrained: 1)
    }

    @Test func aFailedOlderJoinStillDrainsTheNewerTile() async throws {
        // With a prefetched second tile, the first is joined after the second
        // is dispatched; a failure reported by that join leaves the second.
        let steps = try await failThenReuse(.flashNext, overlap: true,
                                            at: .tileJoined(layer: 0, tile: 0),
                                            expectedDrained: 1)
        #expect(steps.contains(.tileDispatched(layer: 0, tile: 1)))
    }

    @Test func metadataOrFetchFailureDrainsTheInFlightSharedExpert() async throws {
        for step in [PrefillRoutedPhaseStep.metadataPrepared(layer: 0),
                     .fetchStarting(layer: 0, tile: 0),
                     .tileBound(layer: 1, tile: 0)] {
            let steps = try await failThenReuse(.flashNext, overlap: true, at: step,
                                                expectedDrained: 1)
            #expect(!steps.contains { if case .tileDispatched(let layer, _) = $0 { layer == step.layer } else { false } })
        }
    }

    @Test func aSharedExpertFailurePreventsRoutedDispatch() async throws {
        let steps = try await failThenReuse(.flashNext, overlap: true,
                                            at: .sharedJoined(layer: 1),
                                            expectedDrained: 0)
        #expect(!steps.contains(.tileDispatched(layer: 1, tile: 0)))
        #expect(steps.contains(.tileBound(layer: 1, tile: 0)))
    }

    @Test(arguments: [false, true])
    func cancellationMidLayerDrainsAndLeavesTheRunnerReusable(overlap: Bool) async throws {
        let reference = try makeRunner(.flashNext, overlap: false)
        let runner = try makeRunner(.flashNext, overlap: overlap)
        defer { remove(reference, runner) }
        let expected = try await prefill(reference)

        let recorder = Recorder()
        recorder.cancelAt = .tileDispatched(layer: 1, tile: 0)
        runner.runner.routedPhaseObserver = recorder.observe
        let task = Task { try await prefill(runner) }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        let steps = recorder.steps
        #expect(steps.contains(.drainedAfterFailure(layer: 1, buffers: 1)))
        #expect(!steps.contains(.fetchStarting(layer: 1, tile: 1)))

        runner.runner.routedPhaseObserver = nil
        let reused = try await prefill(runner)
        #expect(reused == expected)
    }

    @Test func cleanupJoinsEveryBufferPastAFailedJoin() {
        var joined: [Int] = []
        struct JoinFailure: Error {}
        let failures = PrefillSubmittedWorkDrain.joinAll([
            { joined.append(0) },
            { joined.append(1); throw JoinFailure() },
            { joined.append(2) },
            { joined.append(3); throw JoinFailure() },
        ])
        #expect(joined == [0, 1, 2, 3])
        #expect(failures.count == 2)
    }

    @Test(arguments: [false, true])
    func cancellationAfterFetchPreventsDispatch(overlap: Bool) async throws {
        let reference = try makeRunner(.flashNext, overlap: false)
        let runner = try makeRunner(.flashNext, overlap: overlap)
        defer { remove(reference, runner) }
        let expected = try await prefill(reference)
        let recorder = Recorder()
        recorder.cancelAt = .tileBound(layer: 1, tile: 0)
        runner.runner.routedPhaseObserver = recorder.observe
        let task = Task { try await prefill(runner) }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(!recorder.steps.contains(.tileDispatched(layer: 1, tile: 0)))
        #expect(recorder.steps.contains(.drainedAfterFailure(layer: 1, buffers: overlap ? 1 : 0)))
        runner.runner.routedPhaseObserver = nil
        #expect(try await prefill(runner) == expected)
    }
}

@Suite struct SharedExpertOverlapPolicyTests {
    private let variants: [ModelVariant] = [.gemma4_E2B, .gemma4_E4B, .gemma4_12B_QAT,
        .gemma4_26B_A4B, .qwen36_35B_A3B, .gptOss_20B, .gptOss_120B,
        .minimaxM27, .qwen38FlashNext]
    @Test func defaultAndOffStaySerialized() {
        for variant in variants {
            #expect(!SharedExpertOverlapPolicy(environment: [:], variant: variant).enabled)
            #expect(!SharedExpertOverlapPolicy(environment: [SharedExpertOverlapPolicy.environmentKey: "off"], variant: variant).enabled)
            #expect(SharedExpertOverlapPolicy.resolvedSetting(environment: [:], variant: variant) == "off")
        }
    }

    @Test func explicitOnAdmitsOnlyQualifiedModels() {
        let environment = [SharedExpertOverlapPolicy.environmentKey: "on"]
        for variant in variants {
            let qualified = variant == .gemma4_26B_A4B || variant == .qwen38FlashNext
            let policy = SharedExpertOverlapPolicy(environment: environment, variant: variant)
            #expect(policy.enabled == qualified)
            #expect(policy.admits(speculativeVerification: false) == qualified)
            #expect(!policy.admits(speculativeVerification: true))
            #expect(SharedExpertOverlapPolicy.resolvedSetting(environment: environment, variant: variant)
                    == (qualified ? "on" : "off (model not qualified)"))
        }
    }
}

private extension PrefillRoutedPhaseStep {
    var layer: Int {
        switch self {
        case .sharedSubmitted(let layer), .metadataPrepared(let layer),
             .sharedJoined(let layer), .drainedAfterFailure(let layer, _):
            layer
        case .fetchStarting(let layer, _), .tileBound(let layer, _),
             .tileDispatched(let layer, _), .tileJoined(let layer, _):
            layer
        }
    }
}
