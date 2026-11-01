import Foundation
import Metal
import Testing
import TUFFValidationSupport
@testable import TUFFEngine

/// Position-by-position check of Flint against the publisher's BF16 execution,
/// with a higher-precision (FP32 activations, same weights) execution as arbiter.
///
/// Why an arbiter: Flint routes each token to 6 of 256 experts by sigmoid score.
/// On the pinned 136-token tool prompt several tokens have a 6th/7th router
/// score gap below 3e-4 (layer 14 position 31: 2.6e-4; layer 3 position 131:
/// 2e-5), so any activation precision difference flips an expert and moves later
/// logits by up to ~2. The publisher's BF16 run is therefore itself unfaithful
/// to the FP32 result at 13 of 136 positions. Matching BF16 argmax exactly is
/// not a property any other precision can satisfy, so the test asks that the
/// native runner is no less faithful to the FP32 arbiter than the publisher's
/// BF16 run is, and that every disagreement with BF16 is classified.
@Suite(.serialized) struct FlintPositionReferenceTests {
    /// BF16 spacing at logit magnitudes 16...32 is 0.125; two steps.
    static let bf16NearTieMargin = 0.25
    /// Native token within this FP32 logit gap of the FP32 maximum is an FP32 near tie.
    static let fp32NearTieGap = 0.5

    private struct TopK {
        let ids: [[Int]]
        let logits: [[Double]]
        let top: [Int]

        init(file: String) throws {
            let object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: file))) as? [String: Any])
            let step = try #require((object["steps"] as? [[String: Any]])?.first)
            ids = try #require(step["position_topk_ids"] as? [[Int]])
            logits = try #require(step["position_topk_logits"] as? [[Double]])
            top = ids.map { $0[0] }
        }

        /// Gap from the position maximum to `token`; a token outside the stored
        /// top list is charged the gap to the last stored entry (a lower bound).
        func gap(position: Int, token: Int) -> Double {
            let row = logits[position]
            if let index = ids[position].firstIndex(of: token) { return row[0] - row[index] }
            return row[0] - row[row.count - 1]
        }

        func margin(position: Int) -> Double { logits[position][0] - logits[position][1] }
    }

    private struct Run {
        let ids: [Int]
        let native: [Int]
        let bf16: TopK
        let fp32: TopK
        var bf16Gaps: [Double] { ids.indices.map { fp32.gap(position: $0, token: bf16.top[$0]) } }
        var nativeGaps: [Double] { ids.indices.map { fp32.gap(position: $0, token: native[$0]) } }
    }

    private func run() async throws -> Run? {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["TUFF_FLINT_OUTPUT"], let bf16File = env["TUFF_FLINT_BF16_TOPK_REFERENCE"],
              let fp32File = env["TUFF_FLINT_FP32_TOPK_REFERENCE"], let promptFile = env["TUFF_FLINT_TOOL_REFERENCE"]
        else { return nil }
        let prompt = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: promptFile))) as? [String: Any])
        let ids = try #require(prompt["input_ids"] as? [Int])
        let bf16 = try TopK(file: bf16File), fp32 = try TopK(file: fp32File)
        #expect(ids.count == bf16.top.count && ids.count == fp32.top.count)

        let context = try MetalContext()
        let model = try Model.load(directoryURL: URL(fileURLWithPath: path), device: context.device, expecting: .flint40B)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 1024,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        let logits = try #require(context.device.makeBuffer(length: model.config.vocabSize * 2, options: .storageModeShared))
        var native: [Int] = []
        for (position, token) in ids.enumerated() {
            try await runner.produce(token: Int32(token), position: position, into: logits)
            let values = Fp16Buffer.read(logits, count: model.config.vocabSize)
            #expect(values.allSatisfy { $0.isFinite }, "non-finite logit at position \(position)")
            native.append(try #require(values.indices.max(by: { values[$0] < values[$1] })))
        }
        return Run(ids: ids, native: native, bf16: bf16, fp32: fp32)
    }

    /// Passes: native is no less faithful to the FP32 arbiter than the publisher's own BF16 run.
    @Test func toolPromptIsNoLessFaithfulThanPublisherBF16() async throws {
        guard let r = try await run() else { return }
        let nativeGaps = r.nativeGaps, bf16Gaps = r.bf16Gaps
        // Publisher's own worst deviation from the FP32 arbiter bounds any position.
        let publisherWorst = bf16Gaps.max() ?? 0
        let nativeBeyondTie = nativeGaps.filter { $0 > Self.fp32NearTieGap }.count
        let bf16BeyondTie = bf16Gaps.filter { $0 > Self.fp32NearTieGap }.count
        let nativeSum = nativeGaps.reduce(0, +), bf16Sum = bf16Gaps.reduce(0, +)
        print("Flint vs BF16 argmax mismatches: \(r.ids.indices.filter { r.native[$0] != r.bf16.top[$0] }.count)/\(r.ids.count); vs FP32 argmax: \(r.ids.indices.filter { r.native[$0] != r.fp32.top[$0] }.count); BF16 vs FP32 argmax: \(r.ids.indices.filter { r.bf16.top[$0] != r.fp32.top[$0] }.count)")
        print("Flint FP32-gap beyond near tie: native \(nativeBeyondTie), publisher BF16 \(bf16BeyondTie); gap sum native \(nativeSum), BF16 \(bf16Sum); max native \(nativeGaps.max() ?? 0), publisher worst \(publisherWorst)")
        #expect(r.native.last == r.bf16.top.last && r.native.last == r.fp32.top.last, "final greedy token differs from BF16 or FP32")
        for position in r.ids.indices {
            #expect(nativeGaps[position] <= publisherWorst,
                    "position \(position): native token \(r.native[position]) is \(nativeGaps[position]) below the FP32 maximum, worse than the publisher's own worst \(publisherWorst)")
        }
        #expect(nativeBeyondTie <= bf16BeyondTie, "native has more positions beyond an FP32 near tie than the publisher BF16 run")
        #expect(nativeSum <= bf16Sum, "native total FP32 gap exceeds the publisher BF16 run's")
    }

    /// Literal BF16 gate, deliberately not relaxed: every disagreement with the
    /// publisher's BF16 argmax must be a BF16 near tie (margin <= 0.25), an
    /// FP32 near tie (native token within 0.5 of the FP32 maximum), or the FP32
    /// argmax itself. The original FP16 residual failed at position 31; FP32
    /// residual with BF16 routed intermediates failed at positions 95 and 104.
    /// FP32 routed intermediates pass this unchanged gate on the pinned prompt.
    @Test func everyBF16DisagreementIsANearTieOrMatchesFP32() async throws {
        guard let r = try await run() else { return }
        let nativeGaps = r.nativeGaps, bf16Gaps = r.bf16Gaps
        var unclassified: [Int] = []
        for position in r.ids.indices where r.native[position] != r.bf16.top[position] {
            let reason: String
            if r.bf16.margin(position: position) <= Self.bf16NearTieMargin { reason = "bf16-near-tie" }
            else if r.native[position] == r.fp32.top[position] { reason = "native-matches-fp32" }
            else if nativeGaps[position] <= Self.fp32NearTieGap { reason = "fp32-near-tie" }
            else { reason = "UNCLASSIFIED"; unclassified.append(position) }
            print("Flint disagreement position=\(position) native=\(r.native[position]) bf16=\(r.bf16.top[position]) fp32=\(r.fp32.top[position]) bf16_margin=\(r.bf16.margin(position: position)) native_fp32_gap=\(nativeGaps[position]) bf16_fp32_gap=\(bf16Gaps[position]) -> \(reason)")
        }
        #expect(unclassified.isEmpty, "disagreements with BF16 not explained by a near tie or the FP32 arbiter: \(unclassified)")
    }
}
