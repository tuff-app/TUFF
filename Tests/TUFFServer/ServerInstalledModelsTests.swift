import Foundation
import Testing
import TUFFModelCatalog
import TUFFEngine
import TUFFAppCore
import TUFFDecodeProtocol
@testable import TUFFServerCore

/// Installed-model discovery, routing and serving plans, which the host-backed
/// provider takes from `ServerInstalledModels` exactly as the old in-process
/// router did.
@Suite(.serialized) struct ServerInstalledModelsTests {
    @Test func servingRuntimeUsesTheResolvedAllocationSnapshot() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let installed = provider(root).installed
        let resolved = AppAutomaticMemoryPlan(contextTokens: 2_048, expertCacheSlots: 10,
            estimatedWorkingSetBytes: 7 << 30, safeBudgetBytes: 8 << 30,
            denseResidencyPolicy: .bounded(residentLayers: []), prefillChunkTokens: 32,
            retainedConversationBytes: 0)
        let runtime = installed.runtimeConfiguration(for: TUFFModelCatalog.ornith15_397B, resolvedPlan: resolved)
        #expect(runtime.expertCacheSlots == 10)
        #expect(runtime.prefillChunkTokens == 32)
        #expect(runtime.prefillPolicy == .off)
    }
    @Test func servingPlansGrowOnlyWithinTheMemoryBudget() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let installed = provider(root).installed
        for model in TUFFModelCatalog.all {
            let plan = installed.inferencePlan(for: model)
            #expect(plan.contextTokens >= model.runtimeDefaults.contextTokens)
            if plan.contextTokens != model.runtimeDefaults.contextTokens
                || plan.expertCacheSlots != model.runtimeDefaults.expertCacheSlots {
                #expect(model.estimatedInferenceWorkingSetBytes(
                    contextTokens: plan.contextTokens, expertCacheSlots: plan.expertCacheSlots,
                    prefillChunkTokens: model.recommendedPrefillChunkTokens(on: installed.device))
                    <= installed.device.safeAppMemoryBudgetBytes)
            }
            #expect(installed.runtimeConfiguration(for: model).expertCacheSlots == plan.expertCacheSlots)
            if model.family == .gptOss {
                #expect(installed.runtimeConfiguration(for: model).prefillPolicy == .chunked)
            }
        }
        let small = provider(root, memory: 1 << 30).installed
        for model in TUFFModelCatalog.all {
            #expect(small.inferencePlan(for: model).contextTokens == model.runtimeDefaults.contextTokens)
            #expect(small.inferencePlan(for: model).expertCacheSlots == model.runtimeDefaults.expertCacheSlots)
            if model.family == .gptOss {
                #expect(small.runtimeConfiguration(for: model).prefillPolicy == .chunked)
            }
        }
    }

    @Test func modelDiscoveryAdvertisesTheServingPlan() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        for model in TUFFModelCatalog.all { try install(model, root: root) }
        let p = provider(root)
        let encoded = try JSONEncoder().encode(p.modelList())
        let list = try JSONDecoder().decode(OpenAIModelList.self, from: encoded)
        for row in list.data {
            let model = try #require(ServerInstalledModels.descriptor(named: row.id))
            #expect(row.contextLength == p.installed.inferencePlan(for: model).contextTokens)
            #expect(row.maxOutputTokens == min(4_096, (row.contextLength ?? 0) / 4))
        }
    }
    private func root() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TUFFRouterTests-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    private func install(_ descriptor: TUFFModelDescriptor, root: URL) throws {
        try installFakeModel(descriptor, root: root)
    }
    private func provider(_ root: URL, memory: UInt64 = 16 << 30) -> HostRoutedModelProvider {
        let refuse: HostServingLink.Connector = { throw PrivateServingIPCError.unsafeSocketPath }
        return HostRoutedModelProvider(installed: .init(modelsRoot: root, device: .init(
            unifiedMemoryBytes: memory, macOSMajorVersion: 26, appleSiliconGeneration: 2)),
            settings: .init(url: nil, fallback: .init(defaultModel: "gemma4-e2b")),
            link: HostServingLink(launching: refuse, existing: refuse))
    }

    /// An imported checkpoint is offered under a name made from its own,
    /// planned as its template, and loaded under the template's identity.
    @Test func customCheckpointsAreServedByName() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E2B, root: root)
        let id = "custom-" + UUID().uuidString.lowercased()
        let template = TUFFModelCatalog.gemma4_E2B
        let custom = TUFFModelDescriptor(id: template.id, selector: "x", apiModelID: "x",
            displayName: "x", shortName: "x", summary: "", family: template.family,
            architecture: template.architecture, installDirectoryName: "\(id).gturbo",
            source: TUFFModelSource(repoID: "local/\(id)", revision: "r", sourceIndexSHA256: "s",
                manifestModelID: "local/my-tune", approximateDownloadBytes: 0,
                installedBytes: 1, reserveBytes: 0),
            hardware: template.hardware, memory: template.memory,
            capabilities: template.capabilities, reasoningControl: template.reasoningControl)
        try install(custom, root: root)
        let registry: [String: Any] = ["schemaVersion": 1, "checkpoints": [[
            "id": id, "name": "My Gemma Tune!", "sourcePath": "/tmp/source.gturbo",
            "templateID": template.id.rawValue,
            "sourceIndexSHA256": String(repeating: "a", count: 64),
            "manifestSHA256": String(repeating: "b", count: 64), "installedBytes": 1]]]
        try JSONSerialization.data(withJSONObject: registry)
            .write(to: root.appendingPathComponent("custom-checkpoints.json"))

        let p = provider(root)
        #expect(p.modelList().data.map(\.id) == [template.apiModelID, "my-gemma-tune"])
        let routed = try p.route("my-gemma-tune")
        #expect(routed.id == "my-gemma-tune")
        let descriptor = try #require(p.installed.descriptor(named: "my-gemma-tune"))
        #expect(descriptor.displayName == "My Gemma Tune!")
        #expect(!descriptor.capabilities.contains(.imageInput))
        #expect(p.installed.engineReportedModelID(for: descriptor) == template.apiModelID)
        #expect(p.installed.engineReportedModelID(for: template) == template.apiModelID)
        #expect(p.installed.planningDescriptor(for: descriptor).isCustomCheckpoint)
        #expect(p.installed.directory(for: descriptor).lastPathComponent == "\(id).gturbo")
    }

    @Test func customCheckpointNamesAreStableAndNeverShadowCatalogModels() throws {
        func checkpoint(_ name: String) -> AppCustomCheckpoint {
            AppCustomCheckpoint(id: "custom-" + UUID().uuidString.lowercased(), name: name,
                sourcePath: "/x", templateID: .gemma4_E2B,
                sourceIndexSHA256: String(repeating: "a", count: 64),
                manifestSHA256: String(repeating: "b", count: 64), installedBytes: 1)
        }
        let list = [checkpoint("Tuned Model"), checkpoint("tuned model"), checkpoint("gemma4"),
                    checkpoint("!!!"), checkpoint("default")]
        let names = AppCustomCheckpointRegistry.apiNames(for: list)
        #expect(list.map { names[$0.id] } == ["tuned-model", "tuned-model-2", "gemma4-custom",
                                               "custom", "default-custom"])
    }

    @Test func mismatchedPackCannotMasqueradeAsCatalogModel() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E4B, root: root)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent(TUFFModelCatalog.gemma4_E2B.installDirectoryName),
            withDestinationURL: root.appendingPathComponent(TUFFModelCatalog.gemma4_E4B.installDirectoryName))
        let p = provider(root)
        #expect(throws: ServerRequestError.self) { try p.route("default") }
        #expect(p.modelList().data.map(\.id) == [TUFFModelCatalog.gemma4_E4B.apiModelID])
    }
    @Test func aliasesModelsListUnavailableAndIneligible() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E2B, root: root)
        let p = provider(root)
        for name in ["default", "gemma4-e2b", "gemma-4-e2b-it"] {
            #expect(try p.route(name).id == TUFFModelCatalog.gemma4_E2B.apiModelID)
            #expect(try p.route(name).requestedID == name)
        }
        #expect(p.modelList().data.map(\.id) == [TUFFModelCatalog.gemma4_E2B.apiModelID])
        #expect(throws: ServerRequestError.self) { try p.route("gemma4") }
        #expect(throws: ServerRequestError.unknownModel) { try p.route("bogus") }
        #expect(throws: ServerRequestError.self) { try provider(root, memory: 1 << 30).route("default") }
    }
    @Test func legacyManifestsUseTheRuntimesExactArchitectureResolution() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let p = provider(root)
        for model in [TUFFModelCatalog.gemma4_26B_A4B, TUFFModelCatalog.qwen36_35B_A3B,
                      TUFFModelCatalog.minimaxM27, TUFFModelCatalog.gemma4_E2B] {
            try install(model, root: root)
            let url = p.installed.directory(for: model).appendingPathComponent("manifest.json")
            var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
            var arch = manifest["arch"] as! [String: Any]
            arch.removeValue(forKey: "variant")
            if model.family == .gemma4 { arch.removeValue(forKey: "family") }
            manifest["arch"] = arch; manifest["versionMinor"] = 0
            try JSONSerialization.data(withJSONObject: manifest).write(to: url)
            if model.id == .gemma4_E2B {
                #expect(!p.installed.isInstalled(model))
            } else {
                #expect(try p.route(model.apiModelID).id == model.apiModelID)
            }
        }
    }
    @Test func argumentsValidateAndResolveOverrides() throws {
        let a = try RouterServerArguments.parse(["--all-models", "--port", "9999", "--default-model", "gemma4-e2b", "--unload-after", "immediately", "--queue-limit", "2"])
        #expect(a.resolvedSettings(saved: nil).port == 9999)
        #expect(a.resolvedSettings(saved: nil).unloadDelay == .immediately)
        #expect(try RouterServerArguments.parse(["--background"]).background)
        for input in [["--all-models", "--port", "0"], ["--background", "--queue-limit", "17"], ["--all-models", "--unload-after", "-1"], ["--all-models", "--default-model", "missing"]] {
            #expect(throws: ServerArgumentError.self) { try RouterServerArguments.parse(input) }
        }
    }

    @Test func installedModelsUseCatalogRuntimeDefaultsForThisMac() {
        let sixteen = TUFFDeviceCapabilities(unifiedMemoryBytes: 16 * TUFFModelCatalog.oneGiB,
                                             macOSMajorVersion: 26, appleSiliconGeneration: 2)
        let installed = ServerInstalledModels(modelsRoot: URL(fileURLWithPath: "/models"), device: sixteen)
        for descriptor in [TUFFModelCatalog.minimaxM27, TUFFModelCatalog.default] {
            let runtime = installed.runtimeConfiguration(for: descriptor)
            #expect(runtime.expertCacheSlots == descriptor.runtimeDefaults.expertCacheSlots)
            #expect(runtime.prefillChunkTokens == descriptor.recommendedPrefillChunkTokens(on: sixteen))
        }
    }

    @Test func routingNeedsNoModeFlag() throws {
        let plain = try RouterServerArguments.parse([])
        #expect(!plain.background)
        #expect(plain.resolvedSettings(saved: nil) == TUFFBackgroundServerSettings())
        // 7.0.0 scripts passed --all-models; it still parses and changes nothing.
        #expect(try RouterServerArguments.parse(["--all-models"]) == plain)
        #expect(!RouterServerArguments.usage.contains("--all-models"))
    }

    @Test(arguments: ["--model", "--max-context", "--expert-cache-slots", "--prefill-chunk-tokens", "--vision-pack"])
    func removedFixedModelFlagsExplainTheReplacement(flag: String) {
        #expect {
            try RouterServerArguments.parse([flag, "x"])
        } throws: { error in
            guard case ServerArgumentError.invalid(let message) = error else { return false }
            return message.hasPrefix("\(flag) was removed in TUFF 7.1")
        }
    }
}

/// A model folder the installed-model checks accept: a manifest naming the
/// architecture and a verified-install receipt. Weights are absent.
func installFakeModel(_ descriptor: TUFFModelDescriptor, root: URL) throws {
    let dir = root.appendingPathComponent(descriptor.installDirectoryName)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let c = ArchConfig.registeredArchitectures[ModelVariant(rawValue: descriptor.architecture.id.rawValue)!]!
    let arch: [String: Any] = ["variant": c.variant.rawValue, "family": c.family.rawValue,
        "hiddenSize": c.hiddenSize, "ffnIntermediate": c.intermediateSize,
        "moeIntermediateSize": c.moeIntermediateSize, "numHeads": c.numHeads,
        "numKVHeads": c.numKVHeads, "numFullKVHeads": c.numFullKVHeads,
        "headDim": c.headDim, "fullHeadDim": c.fullHeadDim, "vocabSize": c.vocabSize,
        "slidingWindow": c.slidingWindow, "finalLogitSoftcap": c.finalLogitSoftcap,
        "ropeTheta": c.ropeTheta, "fullRopeTheta": c.fullRopeTheta,
        "partialRotaryFactor": c.partialRotaryFactor, "numLayers": c.numLayers,
        "numExperts": c.numExperts, "topKExperts": c.topKExperts,
        "tieWordEmbeddings": c.tieWordEmbeddings, "attentionKEqV": c.attentionKEqV,
        "hiddenActivation": c.hiddenActivation, "fullAttentionLayerMask": c.fullAttentionLayerMask.map { Int($0) }]
    try JSONSerialization.data(withJSONObject: ["magic": "GTURBO", "versionMajor": 1,
        "versionMinor": 1, "flags": [:], "modelID": descriptor.source.manifestModelID,
        "arch": arch, "files": [:], "expertsPerLayer": c.numExperts,
        "numLayers": c.numLayers, "expertStride": 1]).write(to: dir.appendingPathComponent("manifest.json"))
    try Data("{}".utf8).write(to: dir.appendingPathComponent("verified-install.json"))
}
