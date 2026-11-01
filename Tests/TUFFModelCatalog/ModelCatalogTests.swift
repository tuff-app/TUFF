import Foundation
import Testing
@testable import TUFFModelCatalog

@Suite struct ModelCatalogTests {
    @Test func localPreferenceDomainsAreStableAndSeparateFromNormalPreferences() {
        let first = TUFFApplicationDirectories.preferencesSuiteName(environment:
            ["TUFF_APPLICATION_SUPPORT_ROOT": "/tmp/tuff-one"])
        #expect(first != "TUFF")
        #expect(first == TUFFApplicationDirectories.preferencesSuiteName(environment:
            ["TUFF_APPLICATION_SUPPORT_ROOT": "/tmp/tuff-one/./"]))
        #expect(first != TUFFApplicationDirectories.preferencesSuiteName(environment:
            ["TUFF_APPLICATION_SUPPORT_ROOT": "/tmp/tuff-two"]))
        #expect(TUFFApplicationDirectories.preferencesSuiteName(environment: [:]) == "TUFF")
        #expect(TUFFApplicationDirectories.preferencesSuiteName(environment:
            ["TUFF_APPLICATION_SUPPORT_ROOT": "relative"]) == "TUFF")
    }
    @Test func olderCatalogJSONLoadsWithTheCurrentMaintenancePolicy() throws {
        for model in TUFFModelCatalog.all {
            let encoded = try JSONEncoder().encode(model)
            var oldObject = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
            oldObject.removeValue(forKey: "maintenanceStatus")
            let oldData = try JSONSerialization.data(withJSONObject: oldObject)
            let decoded = try JSONDecoder().decode(TUFFModelDescriptor.self, from: oldData)
            #expect(decoded == model)
        }
    }

    @Test func legacyStatusDoesNotRemoveModelsOrChangeTheirIdentity() {
        let legacy = TUFFModelCatalog.all.filter { $0.maintenanceStatus == .legacy }
        #expect(Set(legacy.map(\.id)) == [.gptOss_20B, .gptOss_120B, .minimaxM27])
        for model in legacy {
            #expect(TUFFModelCatalog.model(selector: model.selector)?.id == model.id)
            #expect(model.qualification == .qualified)
        }
        #expect(TUFFModelCatalog.default.maintenanceStatus == .maintained)
        #expect(TUFFModelCatalog.gemma4_12B_QAT.displayName == "Gemma 4 12B QAT")
        #expect(TUFFModelCatalog.gemma4_12B_QAT.summary.contains("8-bit QAT"))
    }

    @Test func currentCatalogOrderAndSelectorsAreStable() {
        #expect(TUFFModelCatalog.all.map(\.selector) == [
            "gemma4-e2b", "gemma4-e4b", "gemma4-12b-qat", "gemma4", "qwen36",
            "gpt-oss-20b", "gpt-oss-120b", "minimax-m2.7", "qwen38-flash-next", "ornith-9b", "ornith-35b", "ornith-122b", "ornith-397b", "qwen38-27b", "gemma4-heretic",
            "flint-40b", "bonsai-2-2b", "bonsai-2-27b",
        ])
        #expect(TUFFModelCatalog.model(selector: "e2b")?.id == .gemma4_E2B)
        #expect(TUFFModelCatalog.default.id == .gemma4_26B_A4B)
        #expect(TUFFModelCatalog.model(selector: "gemma4")?.id == .gemma4_26B_A4B)
        #expect(TUFFModelCatalog.model(selector: "e4b")?.id == .gemma4_E4B)
        #expect(TUFFModelCatalog.model(selector: "12b")?.id == .gemma4_12B_QAT)
        #expect(TUFFModelCatalog.model(selector: "qwen36")?.id == .qwen36_35B_A3B)
        #expect(TUFFModelCatalog.model(selector: "gpt-oss")?.id == .gptOss_20B)
        #expect(TUFFModelCatalog.model(selector: "gpt-oss-120b")?.id == .gptOss_120B)
        #expect(TUFFModelCatalog.model(selector: "minimax")?.id == .minimaxM27)
        #expect(TUFFModelCatalog.model(selector: "flash-next")?.id == .qwen38FlashNext)
        #expect(TUFFModelCatalog.model(selector: "qwen-flash")?.id == .qwen38FlashNext)
        #expect(TUFFModelCatalog.model(selector: "unknown") == nil)
    }

    @Test func flintOptiQAndBonsaiAreAdvertisedTextOnly() {
        let flint = TUFFModelCatalog.flint40B
        #expect(flint.qualification == .requiresRealModelValidation)
        #expect(flint.architecture.family == .flint)
        // The OptiQ promotion descriptor replaces `flint40B` when promoted: same
        // identity, install directory and architecture, its own pinned source.
        let optiq = TUFFModelCatalog.flint40BOptiQ
        #expect(optiq.id == flint.id && optiq.installDirectoryName == flint.installDirectoryName)
        #expect(optiq.source.repoID == "mlx-community/flint-OptiQ-4bit")
        #expect(optiq.qualification == .qualified)
        #expect(TUFFModelCatalog.model(selector: flint.selector)?.source.repoID == optiq.source.repoID)
        #expect(optiq.hardware.minimumUnifiedMemoryBytes == 16 * 1024 * 1024 * 1024)
        #expect(TUFFModelCatalog.model(manifestModelID: optiq.source.manifestModelID)?.source.revision
                == optiq.source.revision)
        for model in [flint, optiq, TUFFModelCatalog.bonsai2_2B, TUFFModelCatalog.bonsai2_27B] {
            #expect(model.source.revision.count == 40)
            #expect(model.source.sourceIndexSHA256.count == 64)
            #expect(model.capabilities.contains(.textGeneration))
            #expect(!model.capabilities.contains(.imageInput))
            #expect(model.addons.isEmpty)
        }
        for model in [TUFFModelCatalog.bonsai2_2B, TUFFModelCatalog.bonsai2_27B] {
            #expect(model.qualification == .qualified)
            #expect(TUFFModelCatalog.model(selector: model.selector)?.id == model.id)
            #expect(model.architecture.weightLayout == .prismQ2)
        }
        #expect(TUFFModelCatalog.all.count == 18)
    }

    @Test func sourcesRemainPinnedAndUnique() {
        #expect(Set(TUFFModelCatalog.all.map(\.source.repoID)).count
                == TUFFModelCatalog.all.count)
        for model in TUFFModelCatalog.all {
            #expect(model.source.revision.count == 40)
            #expect(model.source.sourceIndexSHA256.count == 64)
            #expect(model.source.approximateDownloadBytes > 0)
            #expect(model.source.installedBytes > 0)
            #expect(model.installDirectoryName.hasSuffix(".gturbo"))
            #expect(TUFFModelCatalog.model(manifestModelID:
                    model.source.manifestModelID)?.id == model.id)
        }
    }

    @Test func imageAddonsRemainSeparateAndM2Gated() {
        // GPT-OSS ships no image add-on; the Gemma and Qwen checkpoints
        // each carry a vision tower packaged separately.
        #expect(TUFFModelCatalog.gptOss_20B.addons.isEmpty)
        #expect(TUFFModelCatalog.gptOss_120B.addons.isEmpty)
        #expect(TUFFModelCatalog.minimaxM27.addons.isEmpty)
        for model in [TUFFModelCatalog.gemma4_E2B,
                      TUFFModelCatalog.gemma4_E4B,
                      TUFFModelCatalog.gemma4_12B_QAT,
                      TUFFModelCatalog.gemma4_26B_A4B,
                      TUFFModelCatalog.qwen36_35B_A3B,
                      TUFFModelCatalog.qwen38FlashNext] {
            let addon = model.addons[0]
            #expect(addon.kind == .imageInput)
            #expect(addon.hardware.minimumAppleSiliconGeneration == 2)
            #expect(addon.source.repoID == model.source.repoID)
        }
    }

    @Test func architectureProfilesAreIndependentFromCheckpointCapabilities() {
        let gemma = TUFFModelCatalog.gemma4_26B_A4B
        let qwen = TUFFModelCatalog.qwen36_35B_A3B
        let gptOss = TUFFModelCatalog.gptOss_20B
        let gptOss120B = TUFFModelCatalog.gptOss_120B
        let minimax = TUFFModelCatalog.minimaxM27

        #expect(TUFFArchitectureProfile.gemma4E4B.feedForwardKind == .dense)
        #expect(TUFFArchitectureProfile.gemma4E4B.weightLayout == .affine)
        #expect(gemma.architecture.id == .gemma4_26B_A4B)
        #expect(qwen.architecture.id == .qwen36_35B_A3B)
        #expect(gptOss.architecture.id == .gptOss_20B)
        #expect(gptOss120B.architecture.id == .gptOss_120B)
        #expect(minimax.architecture.id == .minimaxM27)
        #expect(gemma.architecture.feedForwardKind == .mixtureOfExperts)
        #expect(qwen.architecture.feedForwardKind == .mixtureOfExperts)
        #expect(gemma.architecture.weightLayout == .affine)
        #expect(qwen.architecture.weightLayout == .affine)
        #expect(gptOss.architecture.weightLayout == .mxfp4)
        #expect(gptOss120B.architecture.weightLayout == .mxfp4)
        #expect(minimax.architecture.weightLayout == .affine)
        #expect(minimax.hardware.minimumUnifiedMemoryBytes == 16 * TUFFModelCatalog.oneGiB)
        #expect(minimax.reasoningControl == .alwaysOn)
        #expect(gemma.capabilities.contains(.imageInput))
        #expect(qwen.capabilities.contains(.reasoning))
        #expect(gptOss.reasoningControl == .graded)
        #expect(gptOss.qualification == .qualified)
        #expect(gptOss.addons.isEmpty)
        #expect(gptOss120B.reasoningControl == .graded)
        #expect(gptOss120B.qualification == .qualified)
        #expect(gptOss120B.addons.isEmpty)
    }
}
