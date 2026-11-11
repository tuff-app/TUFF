import Foundation
import Synchronization
import Testing
import TUFFEngine
import TUFFModelCatalog
@testable import TUFFAppCore

@Suite struct MacAppSettingsTests {
    @Test(arguments: [false, true])
    func transparencyMigratesTheOldToggle(enabled: Bool) throws {
        let data = try JSONEncoder().encode(MacAppSettings())
        var legacy = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        legacy.removeValue(forKey: "windowTransparency")
        legacy["translucentWindow"] = enabled
        let restored = try JSONDecoder().decode(MacAppSettings.self,
            from: JSONSerialization.data(withJSONObject: legacy))
        #expect(restored.windowTransparency == (enabled ? 0.7 : 0))
    }

    @MainActor
    @Test func transparencyStepsPersistAcrossRelaunchIncludingOpaque() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = makeAppModel(modelDirectory: directory, settingsPersistenceEnabled: true)
        for percentage in [0.3, 0.8, 0.0, 1.0] {
            model.windowTransparency = percentage
            model.flushPendingSettings()
            let restored = makeAppModel(modelDirectory: directory, settingsPersistenceEnabled: true)
            #expect(restored.windowTransparency == percentage)
            #expect(restored.translucentWindow == (percentage > 0))
        }
    }

    @Test func workspacePermissionsDefaultOffAndPersist() throws {
        let data = try JSONEncoder().encode(MacAppSettings())
        var old = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for key in ["codingWriteEnabled", "codingCommandEnabled", "codingNetworkEnabled", "codingFolderID"] { old.removeValue(forKey: key) }
        var settings = try JSONDecoder().decode(MacAppSettings.self, from: JSONSerialization.data(withJSONObject: old))
        #expect(!settings.codingWriteEnabled && !settings.codingCommandEnabled && !settings.codingNetworkEnabled)
        settings.codingWriteEnabled = true
        settings.codingCommandEnabled = true
        settings.codingNetworkEnabled = true
        settings.codingFolderID = UUID().uuidString
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: JSONEncoder().encode(settings))
        #expect(restored == settings)
    }

    @Test func offlinePauseMigratesAndRoundTripsWithoutChangingWebIntent() throws {
        let original = MacAppSettings()
        let data = try JSONEncoder().encode(original)
        var legacy = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        legacy.removeValue(forKey: "pauseWebSearchWhenOffline")
        legacy["webSearchEnabled"] = true
        let migrated = try JSONDecoder().decode(MacAppSettings.self,
            from: JSONSerialization.data(withJSONObject: legacy))
        #expect(migrated.webSearchEnabled)
        #expect(!migrated.pauseWebSearchWhenOffline)
        var updated = migrated
        updated.pauseWebSearchWhenOffline = true
        let decoded = try JSONDecoder().decode(MacAppSettings.self,
            from: JSONEncoder().encode(updated))
        #expect(decoded == updated)
    }

    @MainActor
    @Test func offlinePausePersistsAcrossAppRelaunch() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = makeAppModel(modelDirectory: directory, settingsPersistenceEnabled: true)
        model.webSearchEnabled = true
        model.pauseWebSearchWhenOffline = true
        model.flushPendingSettings()
        let relaunched = makeAppModel(modelDirectory: directory, settingsPersistenceEnabled: true)
        #expect(relaunched.webSearchEnabled)
        #expect(relaunched.pauseWebSearchWhenOffline)
    }

    /// `tuff serve` reads the app's memory settings without ever writing the
    /// file, so the Background API uses the Context chosen in Settings (#55).
    @Test func serverReadsSavedProfileWithoutTouchingTheFile() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let key = TUFFModelID.ornith15_35B_A3B.rawValue
        #expect(AppModelSettingsProfile.saved(forKey: key, modelsRoot: root) == nil)
        var manual = AppModelSettingsProfile.defaults(for: key)
        manual.automaticMemory = false
        manual.contextTokens = AppContextLengthOption.sixtyFourK.tokens
        let directory = root.appendingPathComponent("ornith-35b.gturbo", isDirectory: true)
        try MacAppSettingsFileStore.save(MacAppSettings(modelProfiles: [key: manual]),
                                         forModelDirectory: directory)
        let file = MacAppSettingsFileStore.fileURL(forModelDirectory: directory)
        let written = try Data(contentsOf: file)
        #expect(AppModelSettingsProfile.saved(forKey: key, modelsRoot: root) == manual)
        #expect(AppModelSettingsProfile.saved(forKey: "gemma4-26b-a4b", modelsRoot: root) == nil)
        #expect(try Data(contentsOf: file) == written)

        var newer = try #require(JSONSerialization.jsonObject(with: written) as? [String: Any])
        newer["version"] = MacAppSettings.currentVersion + 1
        let newerData = try JSONSerialization.data(withJSONObject: newer)
        try newerData.write(to: file)
        #expect(AppModelSettingsProfile.saved(forKey: key, modelsRoot: root) == nil)
        #expect(try Data(contentsOf: file) == newerData)
    }

    @Test func settingsFileLivesBesideModelDirectory() {
        let model = URL(fileURLWithPath: "/tmp/TUFF/gemma4.gturbo",
                        isDirectory: true)
        #expect(MacAppSettingsFileStore.fileURL(forModelDirectory: model).path
            == "/tmp/TUFF/mac-app-settings.json")
    }

    @Test func missingFileCreatesReadableDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)

        #expect(settings == MacAppSettings())
        #expect(FileManager.default.fileExists(atPath: fileURL.path))
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: Data(contentsOf: fileURL))
        #expect(decoded == MacAppSettings())
    }

    @Test func malformedFileIsReplacedWithDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        try Data("not json".utf8).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        #expect(settings == MacAppSettings())
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: Data(contentsOf: fileURL))
        #expect(decoded == MacAppSettings())
    }

    @Test func invalidValuesAreReplacedWithDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let invalid = MacAppSettings(contextTokens: 123)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        try JSONEncoder().encode(invalid).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        #expect(settings == MacAppSettings())
    }

    @Test func legacySettingsDefaultToPlainReturnSendWithoutLosingValues() throws {
        let data = Data("""
        {
          "version": 1,
          "contextTokens": 8192,
          "expertCacheSlots": 24,
          "temperature": 0.4,
          "topKEnabled": false,
          "topK": 32,
          "topPEnabled": false,
          "topP": 0.8,
          "prefillEnabled": false
        }
        """.utf8)

        let settings = try JSONDecoder().decode(MacAppSettings.self, from: data)

        #expect(settings.contextTokens == 8_192)
        #expect(settings.expertCacheSlots == 24)
        #expect(settings.temperature == 0.4)
        #expect(!settings.topKEnabled)
        #expect(settings.topK == 32)
        #expect(!settings.topPEnabled)
        #expect(settings.topP == 0.8)
        #expect(!settings.prefillEnabled)
        #expect(settings.newlineShortcut == .shiftReturn)
        // A settings file predating these keys picks up the current defaults:
        // examples hidden, and the prompt cleared once it has been sent.
        #expect(!settings.showPromptExamples)
        #expect(settings.sentPromptBehavior == .clear)
    }

    @Test func legacySettingsMigrateIntoTheActiveModelProfile() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("qwen36.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        try Data("""
        {
          "version": 2,
          "contextTokens": 16384,
          "expertCacheSlots": 24,
          "temperature": 0.4,
          "topKEnabled": false,
          "topK": 32,
          "topPEnabled": false,
          "topP": 0.8,
          "prefillEnabled": false,
          "newlineShortcut": "shift-return",
          "showPromptExamples": false,
          "sentPromptBehavior": "clear",
          "visionResidencyPolicy": "on-demand",
          "rdadvisePolicy": "bounded",
          "loadModelOnLaunch": true
        }
        """.utf8).write(to: fileURL)

        let key = AppModelInstallDescriptor.qwen36.settingsProfileKey
        let settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: model,
            profileKey: key)
        let profile = settings.profile(for: key)

        #expect(settings.version == MacAppSettings.currentVersion)
        #expect(settings.modelProfiles.count == 1)
        #expect(profile.contextTokens == 16_384)
        #expect(profile.expertCacheSlots == 24)
        #expect(profile.temperature == 0.4)
        #expect(!profile.topKEnabled)
        #expect(profile.topK == 32)
        #expect(!profile.topPEnabled)
        #expect(profile.topP == 0.8)
        #expect(!profile.prefillEnabled)
        #expect(profile.rdadvisePolicy == .bounded)
        #expect(settings.newlineShortcut == .shiftReturn)
        #expect(!settings.showPromptExamples)
        #expect(settings.sentPromptBehavior == .clear)
        #expect(settings.loadModelOnLaunch)

        let object = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: fileURL))
                as? [String: Any])
        #expect(object["modelProfiles"] != nil)
        #expect(object["contextTokens"] == nil)
    }

    @Test func modelProfilesRoundTripWithoutOverwritingEachOther() throws {
        let gemmaKey = AppModelInstallDescriptor.default.settingsProfileKey
        let qwenKey = AppModelInstallDescriptor.qwen36.settingsProfileKey
        var settings = MacAppSettings()
        settings.setProfile(
            MacModelSettings(contextTokens: 4_096, temperature: 0.1),
            for: gemmaKey)
        settings.setProfile(
            MacModelSettings(contextTokens: 32_768, temperature: 0.7),
            for: qwenKey)

        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(settings))

        #expect(decoded.profile(for: gemmaKey).contextTokens == 4_096)
        #expect(decoded.profile(for: gemmaKey).temperature == 0.1)
        #expect(decoded.profile(for: qwenKey).contextTokens == 32_768)
        #expect(decoded.profile(for: qwenKey).temperature == 0.7)
    }

    @Test func versionThreeProfilesUpgradeWithoutBeingCollapsed() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        try Data("""
        {
          "version": 3,
          "modelProfiles": {
            "gemma4-26b-a4b": {
              "contextTokens": 8192,
              "expertCacheSlots": 16,
              "temperature": 0.1,
              "topKEnabled": true,
              "topK": 64,
              "topPEnabled": true,
              "topP": 0.95,
              "prefillEnabled": true,
              "visionResidencyPolicy": "on-demand",
              "rdadvisePolicy": "off"
            },
            "qwen36-35b-a3b": {
              "contextTokens": 16384,
              "expertCacheSlots": 24,
              "temperature": 0.7,
              "topKEnabled": false,
              "topK": 32,
              "topPEnabled": false,
              "topP": 0.8,
              "prefillEnabled": false,
              "visionResidencyPolicy": "on-demand",
              "rdadvisePolicy": "bounded"
            }
          }
        }
        """.utf8).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: model)

        #expect(settings.version == MacAppSettings.currentVersion)
        #expect(settings.modelProfiles.count == 2)
        #expect(settings.profile(for: "gemma4-26b-a4b").temperature == 0.1)
        #expect(settings.profile(for: "qwen36-35b-a3b").contextTokens == 16_384)
        #expect(settings.profile(for: "qwen36-35b-a3b").defaultReasoning == .off)
        #expect(settings.profile(for: "qwen36-35b-a3b").defaultReasoningEffort == .medium)
        #expect(!settings.profile(for: "qwen36-35b-a3b").preserveThinking)
        #expect(MacModelSettings.defaults(
            for: "minimax-m2.7").defaultReasoning == .on)
        #expect(settings.profile(for: "gemma4-26b-a4b").systemPrompt
            == TUFFModelCatalog.gemma4_26B_A4B.defaultSystemPrompt)
        #expect(settings.profile(for: "qwen36-35b-a3b").systemPrompt
            == TUFFModelCatalog.qwen36_35B_A3B.defaultSystemPrompt)
    }

    @Test func versionThreeLowSlotPrefillMigratesToAValidProfile() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        try Data("""
        {
          "version": 3,
          "modelProfiles": {
            "gemma4-26b-a4b": {
              "contextTokens": 8192,
              "expertCacheSlots": 8,
              "temperature": 0.2,
              "topKEnabled": true,
              "topK": 64,
              "topPEnabled": true,
              "topP": 0.95,
              "prefillEnabled": true,
              "visionResidencyPolicy": "on-demand",
              "rdadvisePolicy": "off"
            }
          }
        }
        """.utf8).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: model)

        #expect(settings.version == MacAppSettings.currentVersion)
        #expect(settings.profile(for: "gemma4-26b-a4b").expertCacheSlots == 8)
        #expect(!settings.profile(for: "gemma4-26b-a4b").prefillEnabled)
        #expect(settings.isValid())
    }

    @Test func reasoningDefaultsRoundTripPerModel() throws {
        let qwenKey = AppModelInstallDescriptor.qwen36.settingsProfileKey
        let gptKey = AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gptOss_20B)!.settingsProfileKey
        var settings = MacAppSettings()
        settings.setProfile(AppModelSettingsProfile(
            defaultReasoning: .on,
            preserveThinking: true), for: qwenKey)
        settings.setProfile(AppModelSettingsProfile(
            contextTokens: 4_096,
            expertCacheSlots: 4,
            topKEnabled: false,
            defaultReasoningEffort: .high), for: gptKey)

        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(settings))

        #expect(decoded.profile(for: qwenKey).defaultReasoning == .on)
        #expect(decoded.profile(for: qwenKey).preserveThinking)
        #expect(decoded.profile(for: gptKey).defaultReasoningEffort == .high)
    }

    @Test func systemPromptRoundTripsAndDefaultsForOlderProfiles() throws {
        let profile = AppModelSettingsProfile(systemPrompt: "Answer briefly.")
        let encoded = try JSONEncoder().encode(profile)
        let decoded = try JSONDecoder().decode(
            AppModelSettingsProfile.self, from: encoded)

        #expect(decoded.systemPrompt == "Answer briefly.")

        var object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "systemPrompt")
        let legacyData = try JSONSerialization.data(withJSONObject: object)
        let legacy = try JSONDecoder().decode(
            AppModelSettingsProfile.self, from: legacyData)
        #expect(legacy.systemPrompt.isEmpty)

        let minimax = AppModelSettingsProfile.defaults(for: "minimax-m2.7")
        #expect(minimax.systemPrompt
            == "You are MiniMax M2.7, a helpful AI assistant. "
                + "You are running locally on a Mac in an app called TUFF.")
    }

    @Test func versionFourAddsDefaultsWithoutReplacingCustomSystemPrompts() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let gemmaKey = TUFFModelCatalog.gemma4_26B_A4B.id.rawValue
        let minimaxKey = TUFFModelCatalog.minimaxM27.id.rawValue
        let old = MacAppSettings(version: 4, modelProfiles: [
            gemmaKey: AppModelSettingsProfile(systemPrompt: "Custom."),
            minimaxKey: AppModelSettingsProfile(systemPrompt: ""),
        ])
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try JSONEncoder().encode(old).write(to: fileURL)

        let migrated = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: model,
            profileKey: gemmaKey)

        #expect(migrated.version == MacAppSettings.currentVersion)
        #expect(migrated.profile(for: gemmaKey).systemPrompt == "Custom.")
        #expect(migrated.profile(for: minimaxKey).systemPrompt
            == TUFFModelCatalog.minimaxM27.defaultSystemPrompt)
    }

    @Test func versionFiveProfilesEnableAutomaticMemoryByDefault() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let key = TUFFModelCatalog.gemma4_26B_A4B.id.rawValue
        var object = try #require(JSONSerialization.jsonObject(with:
            JSONEncoder().encode(MacAppSettings(modelProfiles: [
                key: AppModelSettingsProfile(
                    automaticMemory: false,
                    contextTokens: 16_384,
                    expertCacheSlots: 24)
            ]))) as? [String: Any])
        object["version"] = 5
        var profiles = try #require(object["modelProfiles"] as? [String: Any])
        var profile = try #require(profiles[key] as? [String: Any])
        profile.removeValue(forKey: "automaticMemory")
        profiles[key] = profile
        object["modelProfiles"] = profiles
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: object).write(to: fileURL)

        let migrated = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: model,
            profileKey: key)

        #expect(migrated.version == MacAppSettings.currentVersion)
        #expect(migrated.profile(for: key).automaticMemory)
        #expect(migrated.profile(for: key).contextTokens == 16_384)
        #expect(migrated.profile(for: key).expertCacheSlots == 24)
    }

    /// Every model defaults to Balanced, saved archives included. The
    /// profiles differ in context alone and the extra context is free, so
    /// there is nothing to preserve by leaving an older archive on Speed.
    @Test func everyProfileDefaultsToBalanced() throws {
        #expect(AppModelSettingsProfile().automaticMemoryProfile == .balanced)
        #expect(AppModelSettingsProfile.defaults(for: MacAppSettings.defaultProfileKey)
            .automaticMemoryProfile == .balanced)

        var object = try #require(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(AppModelSettingsProfile()))
            as? [String: Any])
        object.removeValue(forKey: "automaticMemoryProfile")
        let decoded = try JSONDecoder().decode(
            AppModelSettingsProfile.self,
            from: try JSONSerialization.data(withJSONObject: object))

        #expect(decoded.automaticMemoryProfile == .balanced)
    }

    @Test(arguments: AppAutomaticMemoryProfile.allCases)
    func theAutoProfileRoundTrips(_ profile: AppAutomaticMemoryProfile) throws {
        var written = AppModelSettingsProfile()
        written.automaticMemoryProfile = profile
        let decoded = try JSONDecoder().decode(
            AppModelSettingsProfile.self,
            from: try JSONEncoder().encode(written))

        #expect(decoded.automaticMemoryProfile == profile)
        #expect(decoded == written)
    }

    @Test func bypassingRestrictionsRoundTripsAndDefaultsOff() throws {
        #expect(!MacAppSettings().bypassModelRestrictions)

        var written = MacAppSettings()
        written.bypassModelRestrictions = true
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self, from: try JSONEncoder().encode(written))

        #expect(decoded.bypassModelRestrictions)
    }

    @MainActor
    @Test func systemPromptSurvivesAnAppModelRelaunch() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent(
            "gemma4.gturbo", isDirectory: true)

        let first = makeAppModel(
            modelDirectory: modelDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)
        first.modelSystemPrompt = "Persist this instruction."

        let relaunched = makeAppModel(
            modelDirectory: modelDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)
        #expect(relaunched.modelSystemPrompt == "Persist this instruction.")
    }

    @Test func gptDefaultsRepresentDisabledTopKWithAValidStoredValue() throws {
        let key = try #require(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gptOss_20B)?.settingsProfileKey)
        let profile = AppModelSettingsProfile.defaults(for: key)

        #expect(!profile.topKEnabled)
        #expect(profile.topK == 1)
        #expect(profile.isValid())
    }

    @MainActor
    @Test func selectingModelsSavesAndLoadsTheirOwnProfiles() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let gemmaDirectory = root.appendingPathComponent(
            "gemma4.gturbo", isDirectory: true)
        let qwenDirectory = root.appendingPathComponent(
            "qwen36.gturbo", isDirectory: true)
        let gemmaKey = AppModelInstallDescriptor.default.settingsProfileKey
        let qwenKey = AppModelInstallDescriptor.qwen36.settingsProfileKey
        var settings = MacAppSettings()
        settings.setProfile(MacModelSettings(temperature: 0.1), for: gemmaKey)
        settings.setProfile(MacModelSettings(temperature: 0.7), for: qwenKey)
        try MacAppSettingsFileStore.save(
            settings, forModelDirectory: gemmaDirectory)

        let qwen = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: qwenDirectory,
            client: MockModelInstallerClient(descriptor: .qwen36))
        let model = makeAppModel(
            modelDirectory: gemmaDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwen],
            settingsPersistenceEnabled: true)
        let gemma = try #require(
            model.installs.first { $0.descriptor == .default })

        #expect(model.temperature == 0.1)
        model.temperature = 0.55
        model.selectModel(qwen)
        #expect(model.temperature == 0.7)
        model.selectModel(gemma)
        #expect(model.temperature == 0.55)

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: gemmaDirectory,
            profileKey: gemmaKey)
        #expect(saved.profile(for: gemmaKey).temperature == 0.55)
        #expect(saved.profile(for: qwenKey).temperature == 0.7)
    }

    @MainActor
    @Test func editingANonselectedProfileDoesNotChangeTheActiveModel() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let gemmaDirectory = root.appendingPathComponent(
            "gemma4.gturbo", isDirectory: true)
        let qwenDirectory = root.appendingPathComponent(
            "qwen36.gturbo", isDirectory: true)
        let qwen = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: qwenDirectory,
            client: MockModelInstallerClient(descriptor: .qwen36))
        let model = makeAppModel(
            modelDirectory: gemmaDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwen],
            settingsPersistenceEnabled: true)
        var qwenProfile = model.settingsProfile(for: qwen)
        qwenProfile.automaticMemory = false
        qwenProfile.contextTokens = 32_768
        qwenProfile.temperature = 0.65
        qwenProfile.defaultReasoning = .on
        qwenProfile.preserveThinking = true

        model.updateSettingsProfile(qwenProfile, for: qwen)

        #expect(model.selectedModelID != qwen.id)
        // Gemma's own resolved context, not the 32K just written to Qwen.
        #expect(model.maxContextTokens != 32_768)
        #expect(model.temperature == 0.2)
        let saved = model.settingsProfile(for: qwen)
        #expect(saved.contextTokens == 32_768)
        #expect(saved.temperature == 0.65)
        #expect(saved.defaultReasoning == .on)
        #expect(saved.preserveThinking)
    }

    @Test(arguments: AppNewlineShortcut.allCases)
    func newlineShortcutRoundTrips(_ shortcut: AppNewlineShortcut) throws {
        let initial = MacAppSettings(newlineShortcut: shortcut)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @Test func sendMessageOptionsUseUserFacingOrderAndLabels() {
        #expect(AppNewlineShortcut.sendMessageOptions == [.shiftReturn, .return])
        #expect(AppNewlineShortcut.shiftReturn.sendMessageLabel == "Return")
        #expect(AppNewlineShortcut.return.sendMessageLabel == "Command-Return")
    }

    @Test(arguments: [true, false])
    func showPromptExamplesRoundTrips(_ show: Bool) throws {
        let initial = MacAppSettings(showPromptExamples: show)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @Test(arguments: AppSentPromptBehavior.allCases)
    func sentPromptBehaviorRoundTrips(_ behavior: AppSentPromptBehavior) throws {
        let initial = MacAppSettings(sentPromptBehavior: behavior)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @Test func invalidNewlineShortcutIsReplacedWithDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let invalid = Data("""
        {
          "version": 1,
          "contextTokens": 4096,
          "expertCacheSlots": 16,
          "temperature": 0.2,
          "topKEnabled": true,
          "topK": 64,
          "topPEnabled": true,
          "topP": 0.95,
          "prefillEnabled": true,
          "newlineShortcut": "invalid"
        }
        """.utf8)
        try invalid.write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        #expect(settings == MacAppSettings())
    }

    @MainActor
    @Test func appModelLoadsAndSavesPersistedSettings() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory,
            withIntermediateDirectories: true)
        let initial = MacAppSettings(
            automaticMemory: false,
            contextTokens: 8_192,
            expertCacheSlots: 24,
            temperature: 0.4,
            topKEnabled: false,
            topK: 32,
            topPEnabled: false,
            topP: 0.8,
            prefillEnabled: false,
            newlineShortcut: .shiftReturn,
            showPromptExamples: false,
            sentPromptBehavior: .clear)
        try MacAppSettingsFileStore.save(initial, forModelDirectory: modelDirectory)

        let model = makeAppModel(
            modelDirectory: modelDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)
        #expect(model.maxContextTokens == 8_192)
        #expect(model.runtimeOptions.expertCacheSlots == 24)
        #expect(model.temperature == 0.4)
        #expect(!model.topKEnabled)
        #expect(model.topK == 32)
        #expect(!model.topPEnabled)
        #expect(model.topP == 0.8)
        #expect(!model.runtimeOptions.prefillEnabled)
        #expect(model.newlineShortcut == .shiftReturn)
        #expect(!model.showPromptExamples)
        #expect(model.sentPromptBehavior == .clear)

        model.temperature = 0.6
        model.runtimeOptions.expertCacheSlots = 32
        model.runtimeOptions.prefillEnabled = true
        let beforeGenerate = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(beforeGenerate == initial)

        model.loadState = .ready(modelDirectory: modelDirectory, loadSeconds: 0)
        model.promptText = "Save these settings"
        model.run()
        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(saved.temperature == 0.6)
        #expect(saved.expertCacheSlots == 32)
        #expect(saved.prefillEnabled)
        #expect(saved.newlineShortcut == .shiftReturn)
        #expect(!saved.showPromptExamples)
        #expect(saved.sentPromptBehavior == .clear)
        model.cancel()
    }

    @MainActor
    @Test func autoPreservesAndRestoresTheManualMemoryProfile() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent(
            "gemma4.gturbo", isDirectory: true)
        let key = AppModelInstallDescriptor.default.settingsProfileKey
        var initial = MacAppSettings()
        initial.setProfile(AppModelSettingsProfile(
            automaticMemory: true,
            contextTokens: 32_768,
            expertCacheSlots: 24,
            temperature: 0.4,
            prefillEnabled: false, denseResidentLayerCount: 3), for: key)
        try MacAppSettingsFileStore.save(
            initial, forModelDirectory: modelDirectory)
        let model = makeAppModel(
            modelDirectory: modelDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)
        let selected = try #require(
            model.installs.first { $0.id == model.selectedModelID })

        #expect(model.automaticMemory)
        // Auto chooses a legal allocation from all-layer layout counts on
        // this fixture's small device, while retaining the manual profile.
        #expect(model.maxContextTokens != 32_768)
        #expect(AppRuntimeOptions.allowedSlotCounts.contains(model.runtimeOptions.expertCacheSlots))
        #expect(model.runtimeOptions.expertCacheSlots != 24)
        #expect(model.runtimeOptions.prefillEnabled == (model.runtimeOptions.expertCacheSlots >= 16))

        var effective = model.settingsProfile(for: selected)
        effective.temperature = 0.6
        model.updateSettingsProfile(effective, for: selected)
        // Profile edits are coalesced — the sliders and the prompt editor call
        // this per tick and per keystroke — so the write is asked for before
        // the file is read back.
        model.flushPendingSettings()
        let whileAutomatic = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory,
            profileKey: key).profile(for: key)
        #expect(whileAutomatic.automaticMemory)
        #expect(whileAutomatic.contextTokens == 32_768)
        #expect(whileAutomatic.expertCacheSlots == 24)
        #expect(!whileAutomatic.prefillEnabled)
        #expect(whileAutomatic.denseResidentLayerCount == 3)
        #expect(whileAutomatic.temperature == 0.6)

        model.setAutomaticMemory(false, for: selected)

        #expect(!model.automaticMemory)
        #expect(model.maxContextTokens == 32_768)
        #expect(model.runtimeOptions.expertCacheSlots == 24)
        #expect(!model.runtimeOptions.prefillEnabled)
        #expect(model.runtimeOptions.denseResidencyPolicy == .bounded(residentLayers: [0, 1, 2]))
    }

    /// Edits made through the profile controls are coalesced rather than
    /// written per keystroke, so what matters is that the flush on quit still
    /// gets them to disk.
    @MainActor
    @Test func aCoalescedProfileEditIsOnDiskAfterAFlush() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = makeAppModel(
            modelDirectory: modelDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)
        let install = try #require(model.installs.first { $0.descriptor == .default })
        var profile = model.settingsProfile(for: install)
        profile.temperature = 0.77

        model.updateSettingsProfile(profile, for: install)
        model.flushPendingSettings()

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory,
            profileKey: install.descriptor.settingsProfileKey)
        #expect(saved.profile(for: install.descriptor.settingsProfileKey).temperature == 0.77)
    }

    @MainActor
    @Test func zoomLevelPersistsImmediately() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = makeAppModel(
            modelDirectory: modelDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)

        model.zoomLevel = .percent150

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(saved.zoomLevel == .percent150)
    }

    /// A file written before zoom existed has no key for it, and must open at
    /// 100% rather than failing validation and being replaced by defaults.
    @Test func settingsWrittenBeforeZoomExistedOpenAtActualSize() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: modelDirectory)
        var previous = MacAppSettings()
        previous.newlineShortcut = .shiftReturn
        try MacAppSettingsFileStore.save(previous, forModelDirectory: modelDirectory)
        var json = try JSONSerialization.jsonObject(
            with: Data(contentsOf: fileURL)) as? [String: Any] ?? [:]
        json.removeValue(forKey: "zoomLevel")
        try JSONSerialization.data(withJSONObject: json).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)

        #expect(settings.zoomLevel == .percent100)
        #expect(settings.newlineShortcut == .shiftReturn,
                "the rest of the file must survive the missing key")
    }

    @MainActor
    @Test func newlineShortcutPersistsImmediately() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = makeAppModel(
            modelDirectory: modelDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)

        model.newlineShortcut = .shiftReturn

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(saved.newlineShortcut == .shiftReturn)
    }

    @MainActor
    @Test func showPromptExamplesPersistsImmediately() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = makeAppModel(
            modelDirectory: modelDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)

        model.showPromptExamples = false

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(!saved.showPromptExamples)
    }

    @MainActor
    @Test func sentPromptBehaviorPersistsImmediately() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = makeAppModel(
            modelDirectory: modelDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)

        model.sentPromptBehavior = .clear

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(saved.sentPromptBehavior == .clear)
    }

    @MainActor
    @Test func changingModelDirectoryLoadsItsNewlineShortcut() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first/model.gturbo", isDirectory: true)
        let second = root.appendingPathComponent("second/model.gturbo", isDirectory: true)
        try MacAppSettingsFileStore.save(
            MacAppSettings(newlineShortcut: .return, showPromptExamples: true),
            forModelDirectory: first)
        try MacAppSettingsFileStore.save(
            MacAppSettings(newlineShortcut: .shiftReturn, showPromptExamples: false),
            forModelDirectory: second)
        let model = makeAppModel(
            modelDirectory: first,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)
        #expect(model.newlineShortcut == .return)
        #expect(model.showPromptExamples)

        model.setModelURL(second)

        #expect(model.newlineShortcut == .shiftReturn)
        #expect(!model.showPromptExamples)
    }

    /// Phase D item 15. The newer-version branch says "Every key decodes with
    /// `decodeIfPresent`, so it reads cleanly", and that is false: required keys
    /// still use a hard `decode`. So a newer-version file whose schema moved one
    /// throws inside `JSONDecoder().decode` *before* the version guard is
    /// reached, and lands in the `catch` that deletes the file - destroying a
    /// newer build's settings, which is the exact outcome that branch exists to
    /// prevent. A version bump that cannot change the schema protects nothing.
    @Test func aNewerSettingsFileSurvivesAKeyThisBuildDoesNotKnow() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: modelDirectory)

        // A plausible future version with a schema this build does not
        // understand. Derived from the current version rather than written as
        // a literal, so bumping the schema cannot quietly turn this case into
        // "the build reads its own version" — which is what happened when the
        // zoom setting took version 7.
        let newer = """
        {
          "version": \(MacAppSettings.currentVersion + 1),
          "profilesByModel": {}
        }
        """
        try Data(newer.utf8).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: modelDirectory)

        #expect(settings == MacAppSettings(), "this build must run on defaults")
        #expect(FileManager.default.fileExists(atPath: fileURL.path),
                "a newer build's settings file was deleted by an older build")
        #expect(try String(contentsOf: fileURL, encoding: .utf8) == newer,
                "a newer build's settings file was rewritten by an older build")
    }

    /// Phase D item 16. RDADVISE is a Picker in `InspectorView` bound to
    /// `runtimeOptions.rdadvisePolicy`, and `RUNTIME_CONTROLS.md` lists it as an
    /// app control - but it had no `CodingKey`, so every relaunch silently
    /// reverted it while the UI kept offering it as a persistent setting.
    @Test func rdadviseSurvivesARelaunch() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)

        try MacAppSettingsFileStore.save(
            MacAppSettings(rdadvisePolicy: .bounded),
            forModelDirectory: modelDirectory)
        let reloaded = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: modelDirectory)

        #expect(reloaded.rdadvisePolicy == .bounded)
    }


    // MARK: - Live editing of a model that is not selected

    /// Two models installed side by side, so a profile can be edited while its
    /// model is not the one the app is focused on.
    @MainActor
    private func makeTwoModelHost(
        root: URL,
        persistence: Bool = true,
        conversationStore: AppConversationStore = AppConversationStore()
    ) -> (model: AppModel, gemma: ModelInstallCoordinator, qwen: ModelInstallCoordinator) {
        let gemmaDirectory = root.appendingPathComponent(
            "gemma4.gturbo", isDirectory: true)
        let qwenDirectory = root.appendingPathComponent(
            "qwen36.gturbo", isDirectory: true)
        let qwen = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: qwenDirectory,
            client: MockModelInstallerClient(descriptor: .qwen36))
        let model = makeAppModel(
            modelDirectory: gemmaDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwen],
            conversationStore: conversationStore,
            settingsPersistenceEnabled: persistence)
        let gemma = model.installs.first { $0.descriptor == .default } ?? qwen
        return (model, gemma, qwen)
    }

    /// A sibling view must observe edits made through another control.
    @MainActor
    @Test func editingANonselectedProfileMutatesObservableState() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let host = makeTwoModelHost(root: root)
        // `onChange` is @Sendable, so the flag it sets cannot be a local var.
        let observedChange = Mutex(false)
        withObservationTracking {
            _ = host.model.settingsProfile(for: host.qwen).temperature
        } onChange: {
            observedChange.withLock { $0 = true }
        }

        var profile = host.model.settingsProfile(for: host.qwen)
        profile.temperature = 0.65
        host.model.updateSettingsProfile(profile, for: host.qwen)

        #expect(observedChange.withLock { $0 },
                "a sibling view reading this profile has nothing to recompute from")
        #expect(host.model.settingsProfile(for: host.qwen).temperature == 0.65)
        #expect(host.model.selectedModelID != host.qwen.id)
    }

    /// The same edit, on a model with no settings file behind it at all. What
    /// the screen shows must come from memory, not from whether a write
    /// happened to succeed.
    @MainActor
    @Test func aNonselectedProfileEditAppliesWithoutPersistence() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let host = makeTwoModelHost(root: root, persistence: false)

        var profile = host.model.settingsProfile(for: host.qwen)
        profile.automaticMemory = false
        profile.contextTokens = AppContextLengthOption.thirtyTwoK.tokens
        host.model.updateSettingsProfile(profile, for: host.qwen)

        #expect(host.model.settingsProfile(for: host.qwen).contextTokens
            == AppContextLengthOption.thirtyTwoK.tokens)
    }

    /// The derived text under the memory controls — the working-set estimate,
    /// the eligibility warning — is computed from the same profile, so it has
    /// to move with the edit rather than after a relaunch.
    @MainActor
    @Test func aNonselectedProfileEditMovesTheDerivedMemoryEstimate() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let host = makeTwoModelHost(root: root)
        var profile = host.model.settingsProfile(for: host.qwen)
        profile.automaticMemory = false
        profile.contextTokens = AppContextLengthOption.fourK.tokens
        host.model.updateSettingsProfile(profile, for: host.qwen)
        let before = host.model.contextEligibility(
            for: host.qwen,
            contextTokens: host.model.settingsProfile(for: host.qwen).contextTokens,
            expertCacheSlots: host.model.settingsProfile(for: host.qwen).expertCacheSlots)

        profile.contextTokens = AppContextLengthOption.thirtyTwoK.tokens
        host.model.updateSettingsProfile(profile, for: host.qwen)
        let after = host.model.contextEligibility(
            for: host.qwen,
            contextTokens: host.model.settingsProfile(for: host.qwen).contextTokens,
            expertCacheSlots: host.model.settingsProfile(for: host.qwen).expertCacheSlots)

        #expect(after.estimatedWorkingSetBytes > before.estimatedWorkingSetBytes)
    }

    /// Immediate in memory, coalesced on disk — the same bargain the selected
    /// model's own edits make.
    @MainActor
    @Test func aNonselectedProfileEditIsOnDiskAfterAFlush() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let host = makeTwoModelHost(root: root)
        let qwenKey = AppModelInstallDescriptor.qwen36.settingsProfileKey
        var profile = host.model.settingsProfile(for: host.qwen)
        profile.temperature = 0.65
        profile.defaultReasoning = .on

        host.model.updateSettingsProfile(profile, for: host.qwen)
        host.model.flushPendingSettings()

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: host.qwen.directoryURL,
            profileKey: qwenKey).profile(for: qwenKey)
        #expect(saved.temperature == 0.65)
        #expect(saved.defaultReasoning == .on)
    }

    /// Editing one model must not disturb the model that is loaded, and the
    /// selected model's own edits must not be lost to a sibling's write —
    /// they share one settings file.
    @MainActor
    @Test func editsToBothModelsSurviveOneAnother() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let host = makeTwoModelHost(root: root)
        let gemmaKey = AppModelInstallDescriptor.default.settingsProfileKey
        let qwenKey = AppModelInstallDescriptor.qwen36.settingsProfileKey

        var qwenProfile = host.model.settingsProfile(for: host.qwen)
        qwenProfile.temperature = 0.65
        host.model.updateSettingsProfile(qwenProfile, for: host.qwen)
        var gemmaProfile = host.model.settingsProfile(for: host.gemma)
        gemmaProfile.temperature = 0.15
        host.model.updateSettingsProfile(gemmaProfile, for: host.gemma)
        host.model.flushPendingSettings()

        #expect(host.model.temperature == 0.15, "the selected model still runs its own")
        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: host.gemma.directoryURL,
            profileKey: gemmaKey)
        #expect(saved.profile(for: gemmaKey).temperature == 0.15)
        #expect(saved.profile(for: qwenKey).temperature == 0.65)
    }

    /// Selecting a model whose edit has not been written yet must adopt the
    /// edit, not the file it is still on its way to. Reading the file here is
    /// what would put a stale snapshot on screen.
    @MainActor
    @Test func selectingAModelAdoptsItsUnwrittenEdit() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let host = makeTwoModelHost(root: root)
        var profile = host.model.settingsProfile(for: host.qwen)
        profile.temperature = 0.65
        profile.automaticMemory = false
        profile.contextTokens = AppContextLengthOption.thirtyTwoK.tokens

        // No flush: the write is still sitting behind the debounce.
        host.model.updateSettingsProfile(profile, for: host.qwen)
        host.model.selectModel(host.qwen)

        #expect(host.model.selectedModelID == host.qwen.id)
        #expect(host.model.temperature == 0.65)
        #expect(host.model.maxContextTokens == AppContextLengthOption.thirtyTwoK.tokens)
    }

    /// Switching away and back keeps each model on its own values.
    @MainActor
    @Test func switchingBetweenProfilesShowsEachModelsOwnValues() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let host = makeTwoModelHost(root: root)
        var qwenProfile = host.model.settingsProfile(for: host.qwen)
        qwenProfile.temperature = 0.65
        host.model.updateSettingsProfile(qwenProfile, for: host.qwen)
        host.model.temperature = 0.15

        host.model.selectModel(host.qwen)
        #expect(host.model.temperature == 0.65)
        #expect(host.model.settingsProfile(for: host.gemma).temperature == 0.15)

        host.model.selectModel(host.gemma)
        #expect(host.model.temperature == 0.15)
        #expect(host.model.settingsProfile(for: host.qwen).temperature == 0.65)
    }

    /// Auto on a model that is not selected has to behave exactly as it does
    /// on the selected one: its plan shows through, and the manual context and
    /// cache underneath survive to be handed back.
    @MainActor
    @Test func automaticMemoryOnANonselectedModelKeepsItsManualSettings() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let qwenKey = AppModelInstallDescriptor.qwen36.settingsProfileKey
        var settings = MacAppSettings()
        settings.setProfile(AppModelSettingsProfile(
            automaticMemory: false,
            contextTokens: AppContextLengthOption.thirtyTwoK.tokens,
            expertCacheSlots: 24,
            prefillEnabled: false), for: qwenKey)
        try MacAppSettingsFileStore.save(
            settings,
            forModelDirectory: root.appendingPathComponent(
                "gemma4.gturbo", isDirectory: true))
        let host = makeTwoModelHost(root: root)

        host.model.setAutomaticMemory(true, for: host.qwen)

        let automatic = host.model.settingsProfile(for: host.qwen)
        let plan = try #require(host.model.automaticMemoryPlan(for: host.qwen))
        #expect(automatic.automaticMemory)
        #expect(automatic.contextTokens == plan.contextTokens,
                "Auto's plan is what the screen shows")
        #expect(automatic.expertCacheSlots == plan.expertCacheSlots)

        // Whatever Auto resolved, the manual choices are still what is stored.
        host.model.flushPendingSettings()
        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: host.qwen.directoryURL,
            profileKey: qwenKey).profile(for: qwenKey)
        #expect(saved.automaticMemory)
        #expect(saved.contextTokens == AppContextLengthOption.thirtyTwoK.tokens)
        #expect(saved.expertCacheSlots == 24)
        #expect(!saved.prefillEnabled)

        host.model.setAutomaticMemory(false, for: host.qwen)

        let manual = host.model.settingsProfile(for: host.qwen)
        #expect(!manual.automaticMemory)
        #expect(manual.contextTokens == AppContextLengthOption.thirtyTwoK.tokens)
        #expect(manual.expertCacheSlots == 24)
        #expect(!manual.prefillEnabled)
    }

    @MainActor
    @Test func anInvalidProfileIsRefusedForANonselectedModelToo() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let host = makeTwoModelHost(root: root)
        var profile = host.model.settingsProfile(for: host.qwen)
        profile.automaticMemory = false
        profile.temperature = 0.42
        host.model.updateSettingsProfile(profile, for: host.qwen)

        profile.temperature = 9
        host.model.updateSettingsProfile(profile, for: host.qwen)

        #expect(host.model.settingsProfile(for: host.qwen).temperature == 0.42)
    }

    /// The app-wide settings own their persistence now, so the Settings pane
    /// and the Settings menu can both bind straight to the property. Both
    /// still have to reach disk on the write itself, not on a flush.
    @MainActor
    @Test func appWideSettingsPersistOnAssignment() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let model = makeAppModel(
            modelDirectory: modelDirectory,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)

        model.newlineShortcut = .return
        model.showPromptExamples = true
        model.sentPromptBehavior = .keep
        model.loadModelOnLaunch = true
        model.accentColorMode = .system
        model.zoomLevel = .percent150

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(saved.newlineShortcut == .return)
        #expect(saved.showPromptExamples)
        #expect(saved.sentPromptBehavior == .keep)
        #expect(saved.loadModelOnLaunch)
        #expect(saved.accentColorMode == .system)
        #expect(saved.zoomLevel == .percent150)
    }

    @MainActor
    @Test func aMalformedAccentHexIsIgnored() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = makeAppModel(
            modelDirectory: root.appendingPathComponent("gemma4.gturbo", isDirectory: true),
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)
        model.setCustomAccentColorHex("#123456")

        // Four digits: what "#6F4DFF" looks like halfway through being typed,
        // and not one of the two lengths that parse.
        model.setCustomAccentColorHex("#6F4D")

        #expect(model.customAccentColorHex == "#123456")
    }

    @MainActor
    @Test func deletingSelectedChatSavesItsPendingProfile() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AppConversationStore()
        let gemmaKey = AppModelInstallDescriptor.default.settingsProfileKey
        let qwenKey = AppModelInstallDescriptor.qwen36.settingsProfileKey
        store.recordCompletedTurn(AppChatTurn(prompt: "other", response: "answer"),
                                  attachments: [], modelID: qwenKey)
        store.startNewConversation(modelID: gemmaKey)
        store.recordCompletedTurn(AppChatTurn(prompt: "selected", response: "answer"),
                                  attachments: [], modelID: gemmaKey)
        let host = makeTwoModelHost(root: root, conversationStore: store)
        var profile = host.model.settingsProfile(for: host.gemma)
        profile.temperature = 0.15
        host.model.updateSettingsProfile(profile, for: host.gemma)

        host.model.deleteConversation(try #require(store.selectedConversation))
        host.model.flushPendingSettings()

        #expect(host.model.selectedModelID == host.qwen.id)
        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: host.gemma.directoryURL)
        #expect(saved.profile(for: gemmaKey).temperature == 0.15)
    }

    @MainActor
    @Test func changingDirectorySavesPendingEditsAtTheOldLocation() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let host = makeTwoModelHost(root: root)
        let oldDirectory = host.gemma.directoryURL
        var profile = host.model.settingsProfile(for: host.gemma)
        profile.temperature = 0.15
        host.model.updateSettingsProfile(profile, for: host.gemma)

        host.model.setModelURL(root.appendingPathComponent("new/model.gturbo"))
        host.model.flushPendingSettings()

        let saved = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: oldDirectory)
        #expect(saved.profile(for: host.gemma.descriptor.settingsProfileKey).temperature == 0.15)
        #expect(host.model.temperature != 0.15)
    }

    @MainActor
    @Test func failedNonselectedProfileWriteCanBeRetried() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let host = makeTwoModelHost(root: root)
        let blockedParent = root.appendingPathComponent("blocked")
        try Data("not a directory".utf8).write(to: blockedParent)
        host.qwen.setDirectory(blockedParent.appendingPathComponent("model.gturbo"))
        var profile = host.model.settingsProfile(for: host.qwen)
        profile.temperature = 0.65
        host.model.updateSettingsProfile(profile, for: host.qwen)
        host.model.flushPendingSettings()

        try FileManager.default.removeItem(at: blockedParent)
        host.model.flushPendingSettings()

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: host.qwen.directoryURL,
            profileKey: host.qwen.descriptor.settingsProfileKey)
        #expect(saved.profile(for: host.qwen.descriptor.settingsProfileKey).temperature == 0.65)
    }

    private func makeTemporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("MacAppSettingsTests-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        return root
    }
    @Test func visionResidencyRoundTrips() throws {
        let initial = MacAppSettings(visionResidencyPolicy: .keepReady)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @MainActor
    @Test func changingModelDirectoryStillReleasesTheImageTower() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first/model.gturbo", isDirectory: true)
        let second = root.appendingPathComponent("second/model.gturbo", isDirectory: true)
        try MacAppSettingsFileStore.save(
            MacAppSettings(visionResidencyPolicy: .onDemand),
            forModelDirectory: first)
        try MacAppSettingsFileStore.save(
            MacAppSettings(visionResidencyPolicy: .keepReady),
            forModelDirectory: second)
        let model = makeAppModel(
            modelDirectory: first,
            installer: MockModelInstallerClient(descriptor: .default),
            settingsPersistenceEnabled: true)
        #expect(model.runtimeOptions.visionResidencyPolicy == .onDemand)

        model.setModelURL(second)

        #expect(model.runtimeOptions.visionResidencyPolicy == .onDemand,
                "a persisted keep-ready came back through the model path change")
    }

    @Test func editingDefaultsCannotOverwriteNewerSettings() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model.gturbo")
        let url = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let data = Data(#"{"version":99,"future":"preserve"}"#.utf8)
        try data.write(to: url)
        var settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)
        settings.temperature = 0.7
        #expect(throws: MacAppSettingsFileStore.NewerSettingsOnDisk.self) {
            try MacAppSettingsFileStore.save(settings, forModelDirectory: model)
        }
        #expect(try Data(contentsOf: url) == data)
    }

}
