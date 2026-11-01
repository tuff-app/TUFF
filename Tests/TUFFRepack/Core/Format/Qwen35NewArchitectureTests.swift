import Foundation
import Testing
@testable import TUFFRepackCore

/// Actual pinned publisher/config snapshots, rather than shapes manufactured
/// from the runtime registry. Provenance is recorded in IMPLEMENTATION_9.0.0.
@Suite struct Qwen35NewArchitectureTests {
    @Test func parsesAllFiveArchitectureSnapshots() throws {
        let cases: [(String, RepackModelVariant, Int, Int, Int)] = [
            ("ornith9", .ornith15_9B, 4096, 32, 0),
            ("ornith35", .ornith15_35B_A3B, 2048, 40, 256),
            ("ornith122", .ornith15_122B, 3072, 48, 256),
            ("ornith397", .ornith15_397B, 4096, 60, 512),
            ("qwen27", .qwen38_27B, 5120, 64, 0),
        ]
        for (file, variant, hidden, layers, experts) in cases {
            let path = try #require(Bundle.module.url(forResource: file, withExtension: "json", subdirectory: "NewModels"))
            let arch = try ArchInfo.load(configPath: path.path, variant: variant)
            #expect(arch.variant == variant)
            #expect(arch.hiddenSize == hidden)
            #expect(arch.numLayers == layers)
            #expect(arch.numExperts == experts)
            #expect(arch.feedForwardKind == (experts == 0 ? .dense : .mixtureOfExperts))
            #expect(arch.fullAttentionLayerMask.filter { $0 == 1 }.count == layers / 4)
        }
    }

    @Test func rejectsAnAlteredProductionContract() throws {
        let source = try #require(Bundle.module.url(forResource: "ornith9", withExtension: "json", subdirectory: "NewModels"))
        var root = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: source)) as? [String: Any])
        var text = try #require(root["text_config"] as? [String: Any])
        text["linear_num_value_heads"] = 48
        root["text_config"] = text
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("tuff-arch-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try JSONSerialization.data(withJSONObject: root).write(to: temporary)
        #expect(throws: RepackError.self) {
            try ArchInfo.load(configPath: temporary.path, variant: .ornith15_9B)
        }
    }

    @Test func rejectsUnsupportedRotatedAndFlintArchitecturesBeforeFallback() throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("tuff-arch-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: temporary) }
        for modelType in ["prism_hadamard_qwen35", "flint"] {
            let root: [String: Any] = ["model_type": modelType, "text_config": [:]]
            try JSONSerialization.data(withJSONObject: root).write(to: temporary)
            do {
                _ = try ArchInfo.load(configPath: temporary.path)
                Issue.record("Expected \(modelType) to be rejected")
            } catch RepackError.configJsonInvalid(_, let detail) {
                #expect(detail.contains(modelType == "flint" ? "Flint" : "Hadamard"))
            }
        }
    }
}
