import Foundation
import Testing
import TUFFModelCatalog
import TUFFEngine
import TUFFRepackCore

@testable import TUFFAppCore

/// The catalog's contract: every model has its own install, and starting,
/// cancelling, or finishing one never touches another. Selection is a separate
/// concern from downloading — picking a model decides what loads, and must not
/// start or stop any transfer.
@Suite struct ModelCatalogTests {
    /// Every image pack must come from the *same* checkpoint as the text model
    /// it attaches to. Routing by family alone handed every Gemma the 26B pack,
    /// which installs cleanly and then fails verification against the text
    /// manifest it was never built for.
    @Test func everyImagePackComesFromItsOwnCheckpoint() {
        for model in TUFFModelCatalog.all {
            let companion = AppModelInstallDescriptor(catalog: model).imageCompanion
            guard model.capabilities.contains(.imageInput) else {
                #expect(companion == nil,
                        "\(model.selector) has no image input but offers a pack")
                continue
            }
            // Ornith 9B and 35B are the exceptions: their MLX conversions
            // dropped the tower, so the pack comes from Ornith's BF16 release
            // of the same model, pinned by revision and hash.
            if let repo = separateImageSources[model.id] {
                #expect(companion?.repoID == repo)
                #expect(companion?.revision.count == 40)
                #expect(companion?.sourceIndexSHA256 != model.source.sourceIndexSHA256)
                continue
            }
            #expect(companion?.repoID == model.source.repoID,
                    "\(model.selector) image pack points at \(companion?.repoID ?? "nil")")
            #expect(companion?.revision == model.source.revision)
            #expect(companion?.sourceIndexSHA256 == model.source.sourceIndexSHA256)
        }
    }

    private let separateImageSources: [TUFFModelID: String] = [
        .ornith15_9B: "ornith-ai/Ornith-1.5-9B",
        .ornith15_35B_A3B: "ornith-ai/Ornith-1.5-35B-A3B",
        .ornith15_122B: "ornith-ai/Ornith-1.5-122B",
    ]

    /// Every image pack's advertised download, pinned to the coalesced range
    /// total measured against the pinned checkpoint's real tensor layout.
    ///
    /// Two of these were wrong because they were derived rather than measured:
    /// E4B reused E2B's figure plus the payload delta, and Qwen used the bare
    /// vision payload sum. Both checkpoints interleave their vision tensors
    /// with language tensors, so the bytes actually pulled over the wire
    /// include the gaps a 64 MiB coalescing window spans.
    @Test func imagePackDownloadsArePinnedToMeasuredRangeTotals() {
        let measured: [TUFFModelID: UInt64] = [
            .gemma4_E2B: 1_169_504_854,
            .gemma4_E4B: 1_313_596_274,
            .gemma4_12B_QAT: 102_556_672,
            .gemma4_26B_A4B: 1_539_478_890,
            .gemma4_26B_Heretic: 1_580_286_570,
            .qwen36_35B_A3B: 1_137_999_008,
            // Measured by the installer's own planner against this pin:
            // "planned: 1009.1 MiB to download, 859.6 MiB to write".
            .qwen38FlashNext: 1_058_209_792,
            // From the installer's resume checkpoint (`totalSourceBytes`)
            // against this pin.
            .qwen38_27B: 1_929_719_232,
            // Ornith's towers sit contiguously in their sources, so the range
            // download is exactly the payload.
            .ornith15_9B: 912_020_960,
            .ornith15_35B_A3B: 893_142_496,
            .ornith15_122B: 902_581_728,
            .ornith15_397B: 912_020_960,
        ]
        for model in TUFFModelCatalog.all {
            guard let addon = model.addons.first(where: { $0.kind == .imageInput })
            else { continue }
            #expect(measured[model.id] == addon.source.approximateDownloadBytes,
                    "\(model.selector) image pack download drifted")
            // A pack can never download less than the tensors it installs.
            #expect(addon.source.approximateDownloadBytes >= addon.source.installedBytes,
                    "\(model.selector) claims to download less than it installs")
        }
        #expect(measured.count == TUFFModelCatalog.all.filter {
            $0.addons.contains { $0.kind == .imageInput }
        }.count)
    }

    /// Selecting a text-only model on a Mac whose *device* supports the vision
    /// runtime used to reach a `preconditionFailure` through the inspector's
    /// download line, because that line is gated on the device, not the model.
    @Test func textOnlyModelsReportNoImagePackInsteadOfTrapping() {
        #expect(AppModelInstallDescriptor.visionCompanion(for: .gptOss_20B) == nil)
        #expect(AppModelInstallDescriptor.visionCompanion(for: .gptOss_120B) == nil)
        #expect(AppModelInstallDescriptor.visionCompanion(for: ModelFamily.gptOss) == nil)
        #expect(AppModelInstallDescriptor.visionCompanion(for: .minimaxM27) == nil)
        #expect(AppModelInstallDescriptor.visionCompanion(for: ModelFamily.minimaxM2) == nil)
        // Flash Next is no longer text-only: it carries the same vision tower
        // as Qwen 3.6, so the companion resolves rather than returning nil.
        #expect(AppModelInstallDescriptor.visionCompanion(for: .qwen38FlashNext) != nil)
        #expect(AppModelInstallDescriptor.visionCompanion(for: ModelFamily.qwen4Exp) != nil)
    }

    @Test func catalogExposesStableAPIIDsAndPromptDialects() {
        for descriptor in AppModelInstallDescriptor.catalog {
            #expect(!descriptor.apiModelID.isEmpty)
            #expect(descriptor.defaultSystemPrompt.hasPrefix(
                "You are \(descriptor.shortName), a helpful AI assistant."))
            switch descriptor.family {
            case .gemma4: #expect(descriptor.chatDialect == .gemma)
            case .qwen36, .qwen4Exp: #expect(descriptor.chatDialect == .chatml)
            case .flint: #expect(descriptor.chatDialect == .flint)
            case .gptOss: #expect(descriptor.chatDialect == .harmony)
            case .minimaxM2: #expect(descriptor.chatDialect == .minimax)
            }
        }
    }


    private func temporaryInstallPath(_ tag: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "tuff-catalog-\(tag)-\(UUID().uuidString).gturbo")
    }

    @MainActor
    private func makeModel(
        selectedDirectory: URL,
        selectedInstaller: MockModelInstallerClient,
        otherDirectory: URL,
        otherInstaller: MockModelInstallerClient
    ) -> AppModel {
        let other = ModelInstallCoordinator(
            descriptor: otherInstaller.descriptor,
            directoryURL: otherDirectory,
            client: otherInstaller)
        return makeAppModel(
            modelDirectory: selectedDirectory,
            client: MockLifecycleInferenceClient(),
            installer: selectedInstaller,
            otherInstalls: [other])
    }

    @MainActor
    @Test func catalogListsEveryModelAndSelectsTheInjectedOne() {
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("catalog-gemma"),
            selectedInstaller: MockModelInstallerClient(descriptor: .default),
            otherDirectory: temporaryInstallPath("catalog-qwen"),
            otherInstaller: MockModelInstallerClient(descriptor: .qwen36))

        #expect(model.installs.count == 2)
        #expect(model.selectedModelID == AppModelInstallDescriptor.default.id)
        #expect(model.selectedDescriptor == .default)
        #expect(model.installs.map(\.descriptor) == [.default, .qwen36])
    }

    @Test func appCatalogKeepsStableRegistryIdentities() {
        #expect(AppModelInstallDescriptor.catalog.map(\.settingsProfileKey) == [
            "gemma4-e2b",
            "gemma4-e4b",
            "gemma4-12b-qat",
            "gemma4-26b-a4b",
            "qwen36-35b-a3b",
            "gpt-oss-20b",
            "gpt-oss-120b",
            "minimax-m2.7",
            "qwen3.8-flash-next",
            "ornith-1.5-9b",
            "ornith-1.5-35b-a3b",
            "ornith-1.5-122b",
            "ornith-1.5-397b",
            "qwen3.8-27b",
            "gemma4-26b-heretic",
            "flint-40b-a2.5b",
            "ternary-bonsai-2-2b",
            "ternary-bonsai-2-27b",
        ])
        #expect(AppModelInstallDescriptor.catalog.map(\.installDirectoryName) == [
            "gemma4-e2b.gturbo",
            "gemma4-e4b.gturbo",
            "gemma4-12b-qat.gturbo",
            "gemma4.gturbo",
            "qwen36.gturbo",
            "gpt-oss-20b.gturbo",
            "gpt-oss-120b.gturbo",
            "minimax-m2.7.gturbo",
            "qwen38-flash-next.gturbo",
            "ornith-9b.gturbo",
            "ornith-35b.gturbo",
            "ornith-122b.gturbo",
            "ornith-397b.gturbo",
            "qwen38-27b.gturbo",
            "gemma4-heretic.gturbo",
            "flint-40b.gturbo",
            "bonsai-2-2b.gturbo",
            "bonsai-2-27b.gturbo",
        ])
        #expect(AppModelInstallDescriptor.gemma4E4B.supportsImageInput)
        #expect(MacModelSettings.defaults(
            for: AppModelInstallDescriptor.gemma4E4B.settingsProfileKey).isValid())
        #expect(AppModelInstallDescriptor.default.supportsImageInput)
        #expect(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gemma4_E4B) == .gemma4E4B)
        #expect(AppModelInstallDescriptor.visionCompanion(
            for: .gemma4_E2B)?.repoID == "mlx-community/gemma-4-e2b-it-4bit")
        #expect(AppModelInstallDescriptor.visionCompanion(
            for: .gemma4_12B_QAT)?.repoID == "mlx-community/gemma-4-12B-it-qat-4bit")
        #expect(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.qwen36_35B_A3B) == .qwen36)
        #expect(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gptOss_20B)?.settingsProfileKey == "gpt-oss-20b")
        #expect(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gptOss_120B)?.settingsProfileKey == "gpt-oss-120b")
        #expect(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.minimaxM27) == .minimaxM27)
        #expect(AppModelInstallDescriptor.minimaxM27.reasoningControl == .alwaysOn)
        #expect(AppModelInstallDescriptor.gemma4E4B.reasoningControl == .toggle)
        #expect(AppModelInstallDescriptor.qwen36.reasoningControl
            == .toggleWithPreservation)
        #expect(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gptOss_20B)?.reasoningControl == .graded)
        #expect(AppModelInstallDescriptor.default.isRecommended)
        #expect(!AppModelInstallDescriptor.gemma4E4B.isRecommended)
    }

    @MainActor
    @Test func incompatibleModelsCannotDownloadOrSelect() {
        let gpt120 = AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gptOss_120B)!
        let gptInstaller = MockModelInstallerClient(descriptor: gpt120)
        let gptCoordinator = ModelInstallCoordinator(
            descriptor: gpt120,
            directoryURL: temporaryInstallPath("incompatible-120b"),
            client: gptInstaller)
        let device = TUFFDeviceCapabilities(
            unifiedMemoryBytes: 8 * TUFFModelCatalog.oneGiB,
            macOSMajorVersion: 26,
            appleSiliconGeneration: 2)
        let model = makeAppModel(
            modelDirectory: temporaryInstallPath("compatible-gemma"),
            client: MockLifecycleInferenceClient(),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [gptCoordinator],
            deviceCapabilities: device)

        let eligibility = model.hardwareEligibility(for: gptCoordinator)
        #expect(!eligibility.isCompatible)
        #expect(eligibility.explanation
            == "Requires 16 GB unified memory; this Mac has 8 GB.")
        #expect(!model.canInstallModel(gptCoordinator))
        #expect(!model.canSelectModel(gptCoordinator))

        model.installModel(gptCoordinator)
        model.selectModel(gptCoordinator)
        #expect(!gptCoordinator.isInstalling)
        #expect(model.selectedDescriptor == .default)
    }

    @MainActor
    @Test func qualified120BIsAvailableOnSixteenGB() {
        let gpt120 = AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gptOss_120B)!
        let coordinator = ModelInstallCoordinator(
            descriptor: gpt120,
            directoryURL: temporaryInstallPath("qualified-120b"),
            client: MockModelInstallerClient(descriptor: gpt120))
        let device = TUFFDeviceCapabilities(
            unifiedMemoryBytes: 16 * TUFFModelCatalog.oneGiB,
            macOSMajorVersion: 26,
            appleSiliconGeneration: 2)
        let model = makeAppModel(
            modelDirectory: temporaryInstallPath("selected-gemma"),
            client: MockLifecycleInferenceClient(),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [coordinator],
            deviceCapabilities: device)

        #expect(model.hardwareEligibility(for: coordinator).isCompatible)
        #expect(model.canInstallModel(coordinator))
        #expect(model.canSelectModel(coordinator))
    }

    @MainActor
    @Test func downloadingOneModelLeavesTheOtherAlone() async {
        let selected = MockModelInstallerClient(descriptor: .default, holdOpen: true)
        let other = MockModelInstallerClient(descriptor: .qwen36, holdOpen: true)
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("independent-gemma"),
            selectedInstaller: selected,
            otherDirectory: temporaryInstallPath("independent-qwen"),
            otherInstaller: other)
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        qwen.install()
        await Task.yield()

        #expect(qwen.isInstalling)
        #expect(model.isInstallingAnyModel)
        // The selected model is untouched: its own state, and the app-level
        // single-model properties that report it, stay idle.
        #expect(!model.selectedInstall.isInstalling)
        #expect(model.installState == .idle)
        #expect(!model.isInstallingModel)
        #expect(!selected.cancelCalled)
    }

    @MainActor
    @Test func bothModelsCanDownloadAtOnce() async {
        let selected = MockModelInstallerClient(descriptor: .default, holdOpen: true)
        let other = MockModelInstallerClient(descriptor: .qwen36, holdOpen: true)
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("both-gemma"),
            selectedInstaller: selected,
            otherDirectory: temporaryInstallPath("both-qwen"),
            otherInstaller: other)
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        model.installModel()
        qwen.install()
        await Task.yield()

        #expect(model.selectedInstall.isInstalling)
        #expect(qwen.isInstalling)
    }

    @MainActor
    @Test func cancellingOneDownloadLeavesTheOtherRunning() async {
        let selected = MockModelInstallerClient(descriptor: .default, holdOpen: true)
        let other = MockModelInstallerClient(descriptor: .qwen36, holdOpen: true)
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("cancel-gemma"),
            selectedInstaller: selected,
            otherDirectory: temporaryInstallPath("cancel-qwen"),
            otherInstaller: other)
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        model.installModel()
        qwen.install()
        await Task.yield()
        qwen.cancel()
        // `.cancelling` is still an installing state; wait for it to settle.
        try? await waitUntil { !qwen.isInstalling }

        // `cancelCalled` is not the signal here: starting an install calls
        // cancel on its own client first, to clear any earlier attempt. What
        // matters is that only the cancelled model left the installing state.
        #expect(qwen.state == .cancelled)
        #expect(model.selectedInstall.isInstalling)
        #expect(model.isInstallingModel)
    }

    @MainActor
    @Test func selectingAnotherModelSwitchesTheActiveDirectory() {
        let qwenDirectory = temporaryInstallPath("select-qwen")
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("select-gemma"),
            selectedInstaller: MockModelInstallerClient(descriptor: .default),
            otherDirectory: qwenDirectory,
            otherInstaller: MockModelInstallerClient(descriptor: .qwen36))
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        #expect(model.canSelectModel(qwen))
        model.selectModel(qwen)

        #expect(model.selectedModelID == AppModelInstallDescriptor.qwen36.id)
        #expect(model.modelPathText == qwenDirectory.standardizedFileURL.path)
        #expect(model.installDescriptor == .qwen36)
        // Selecting the model already selected is a no-op, not a reset.
        #expect(!model.canSelectModel(qwen))
    }

    @MainActor
    @Test func installedQwenOffersItsOwnOptionalVisionPack() throws {
        let directory = try makeCompleteModelInstall(
            "vision-qwen", stampedAs: .qwen36)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = makeAppModel(
            modelDirectory: directory,
            client: MockLifecycleInferenceClient(),
            installer: MockModelInstallerClient(descriptor: .qwen36),
            visionInstaller: MockVisionPackInstallerClient(
                descriptor: .qwen36VisionCompanion))

        #expect(model.selectedDescriptor == .qwen36)
        #expect(model.visionInstallDescriptor == .qwen36VisionCompanion)
        #expect(!model.isImageInputAvailable)
        #expect(model.canInstallVisionPack)
        #expect(model.visionInstallReadiness.requirement?.canInstall == true)

        // Preparing the optional companion is a separate download. It may run
        // while the text model stays loaded; only activation needs an unload.
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        #expect(model.canInstallVisionPack)
        model.visionInstallState = .readyToActivate(directory)
        #expect(!model.canActivateVisionPack)
        model.loadState = .notLoaded
        #expect(model.canActivateVisionPack)
    }

    @MainActor
    @Test func catalogOffersSeparateVisionDownloadsWithoutSwitchingModels() async throws {
        let gemmaDirectory = try makeCompleteModelInstall(
            "separate-vision-gemma", stampedAs: .default)
        let qwenDirectory = try makeCompleteModelInstall(
            "separate-vision-qwen", stampedAs: .qwen36)
        defer {
            try? FileManager.default.removeItem(at: gemmaDirectory)
            try? FileManager.default.removeItem(at: qwenDirectory)
        }
        let qwenInstall = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: qwenDirectory,
            client: MockModelInstallerClient(descriptor: .qwen36))
        let visionInstaller = MockVisionPackInstallerClient(holdOpen: true)
        let model = makeAppModel(
            modelDirectory: gemmaDirectory,
            client: MockLifecycleInferenceClient(),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwenInstall],
            visionInstaller: visionInstaller)
        let gemma = try #require(model.installs.first { $0.descriptor == .default })
        let qwen = try #require(model.installs.first { $0.descriptor == .qwen36 })

        #expect(model.visionDownloadButtonLabel(for: gemma)
            == "Download Gemma 4 Image Support")
        #expect(model.visionDownloadButtonLabel(for: qwen)
            == "Download Qwen 3.6 Image Support")
        #expect(model.canInstallVisionPack(for: gemma))
        #expect(model.canInstallVisionPack(for: qwen))

        model.installVisionPack(for: qwen)
        await Task.yield()

        #expect(model.selectedModelID == gemma.id)
        #expect(model.visionInstallTargetModelID == qwen.id)
        #expect(!model.selectedModelOwnsVisionInstallState)
        #expect(visionInstaller.installedTextModelDirectories == [
            qwenDirectory.standardizedFileURL,
        ])
        #expect(!model.canSelectModel(qwen))

        model.cancelVisionInstall()
        try await waitUntil { !model.isInstallingVisionPack }
    }

    @MainActor
    @Test func preparedVisionDownloadFollowsOnlyItsMatchingModel() async throws {
        let gemmaDirectory = try makeCompleteModelInstall(
            "prepared-vision-gemma", stampedAs: .default)
        let qwenDirectory = try makeCompleteModelInstall(
            "prepared-vision-qwen", stampedAs: .qwen36)
        defer {
            try? FileManager.default.removeItem(at: gemmaDirectory)
            try? FileManager.default.removeItem(at: qwenDirectory)
        }
        let qwenInstall = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: qwenDirectory,
            client: MockModelInstallerClient(descriptor: .qwen36))
        let preparedDirectory = qwenDirectory.deletingPathExtension()
            .appendingPathExtension("vision.gturbo")
        let model = makeAppModel(
            modelDirectory: gemmaDirectory,
            client: MockLifecycleInferenceClient(),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwenInstall],
            visionInstaller: MockVisionPackInstallerClient(events: [
                .readyToActivate(preparedDirectory),
            ]))
        let qwen = try #require(model.installs.first { $0.descriptor == .qwen36 })

        model.installVisionPack(for: qwen)
        try await waitUntil {
            if case .readyToActivate = model.visionInstallState { return true }
            return false
        }

        #expect(model.selectedDescriptor == .default)
        #expect(model.canSelectModel(qwen))
        model.selectModel(qwen)

        #expect(model.selectedDescriptor == .qwen36)
        #expect(model.modelPathText == qwenDirectory.standardizedFileURL.path)
        #expect(model.visionInstallTargetModelID == qwen.id)
        #expect(model.selectedModelOwnsVisionInstallState)
        #expect(model.canActivateVisionPack)
    }

    @MainActor
    @Test func selectingAModelDoesNotDisturbAnyDownload() async {
        let selected = MockModelInstallerClient(descriptor: .default, holdOpen: true)
        let other = MockModelInstallerClient(descriptor: .qwen36, holdOpen: true)
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("switch-gemma"),
            selectedInstaller: selected,
            otherDirectory: temporaryInstallPath("switch-qwen"),
            otherInstaller: other)
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        model.installModel()
        qwen.install()
        await Task.yield()

        model.selectModel(qwen)
        await Task.yield()

        #expect(model.installs.allSatisfy { $0.isInstalling })
        #expect(model.selectedModelID == AppModelInstallDescriptor.qwen36.id)
    }

    /// Free space is shared, so a running download has to shrink what the other
    /// model believes it can use. Without this, both models pass their own
    /// check on a disk that fits only one, and the second fails deep into a
    /// multi-gigabyte transfer instead of before it starts.
    @MainActor
    @Test func aRunningDownloadReservesSpaceAgainstTheOtherModel() async {
        // Room for one model but not both.
        let diskBytes = AppModelInstallDescriptor.default.requiredFreeBytes
            + AppModelInstallDescriptor.qwen36.installedBytes / 2
        func installer(_ descriptor: AppModelInstallDescriptor)
            -> MockModelInstallerClient {
            MockModelInstallerClient(
                requirement: AppModelInstallRequirement(
                    requiredBytes: descriptor.requiredFreeBytes,
                    availableBytes: diskBytes),
                descriptor: descriptor,
                holdOpen: true)
        }
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("reserve-gemma"),
            selectedInstaller: installer(.default),
            otherDirectory: temporaryInstallPath("reserve-qwen"),
            otherInstaller: installer(.qwen36))
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        // On an idle disk each model fits on its own.
        #expect(model.selectedInstall.canInstall)
        #expect(qwen.canInstall)

        qwen.install()
        await Task.yield()
        #expect(qwen.isInstalling)

        // Qwen now owes its whole installed size, which is no longer free.
        model.installModel()
        #expect(!model.selectedInstall.isInstalling)
        if case .insufficientSpace(let requirement) = model.installReadiness {
            #expect(requirement.shortfallBytes > 0)
        } else {
            Issue.record("expected insufficientSpace, got \(model.installReadiness)")
        }

        // Cancelling the first download releases the reservation.
        qwen.cancel()
        try? await waitUntil { !qwen.isInstalling }
        model.refreshInstallReadiness()
        #expect(model.selectedInstall.canInstall)
    }

    @MainActor
    @Test func textAndAddOnDownloadsReserveDiskAgainstEachOther() async throws {
        let gemmaDirectory = try makeCompleteModelInstall(
            "cross-reservation-gemma", stampedAs: .default)
        defer { try? FileManager.default.removeItem(at: gemmaDirectory) }

        let vision = AppModelInstallDescriptor.visionCompanion
        let qwen = AppModelInstallDescriptor.qwen36
        let qwenInstaller = MockModelInstallerClient(
            requirement: AppModelInstallRequirement(
                requiredBytes: qwen.requiredFreeBytes,
                availableBytes: qwen.requiredFreeBytes + vision.installedBytes / 2),
            descriptor: qwen,
            holdOpen: true)
        let qwenCoordinator = ModelInstallCoordinator(
            descriptor: qwen,
            directoryURL: temporaryInstallPath("cross-reservation-qwen"),
            client: qwenInstaller)
        let visionInstaller = MockVisionPackInstallerClient(
            requirement: AppModelInstallRequirement(
                requiredBytes: vision.requiredFreeBytes,
                availableBytes: vision.requiredFreeBytes + qwen.installedBytes / 2),
            events: [.copyingPayload(
                reusedBytes: 0,
                downloadedThisRunBytes: 0,
                totalBytes: vision.installedBytes)],
            holdOpen: true)
        let model = makeAppModel(
            modelDirectory: gemmaDirectory,
            client: MockLifecycleInferenceClient(),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwenCoordinator],
            visionInstaller: visionInstaller)
        let gemma = try #require(model.installs.first { $0.descriptor == .default })

        #expect(qwenCoordinator.canInstall)
        #expect(model.canInstallVisionPack(for: gemma))

        model.installVisionPack(for: gemma)
        try await waitUntil("add-on reservation appears") {
            model.visionInstallOutstandingBytes > 0
        }
        qwenCoordinator.refresh()
        #expect(!qwenCoordinator.canInstall)

        model.cancelVisionInstall()
        try await waitUntil("add-on cancellation settles") {
            !model.isInstallingVisionPack
        }
        qwenCoordinator.refresh()
        #expect(qwenCoordinator.canInstall)

        qwenCoordinator.install()
        try await waitUntil("text download starts") { qwenCoordinator.isInstalling }
        #expect(model.visionInstallRequirement(for: gemma)?.canInstall == false)
        #expect(!model.canInstallVisionPack(for: gemma))

        qwenCoordinator.cancel()
        try await waitUntil("text cancellation settles") {
            !qwenCoordinator.isInstalling
        }
    }

    @MainActor
    @Test func nonselectedAddOnCanBeRemovedWhileSelectedModelStaysLoaded() async throws {
        let gemmaDirectory = try makeCompleteModelInstall(
            "remove-other-gemma", stampedAs: .default)
        let qwenDirectory = try makeCompleteModelInstall(
            "remove-other-qwen", stampedAs: .qwen36)
        try installVisionCompanion(forTextModel: qwenDirectory)
        defer {
            try? FileManager.default.removeItem(at: gemmaDirectory)
            try? FileManager.default.removeItem(at: qwenDirectory)
        }
        let qwen = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: qwenDirectory,
            client: MockModelInstallerClient(descriptor: .qwen36))
        let visionInstaller = MockVisionPackInstallerClient()
        let model = makeAppModel(
            modelDirectory: gemmaDirectory,
            client: MockLifecycleInferenceClient(),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwen],
            visionInstaller: visionInstaller)
        model.loadState = .ready(modelDirectory: gemmaDirectory, loadSeconds: 0)

        #expect(model.canRemoveVisionPack(for: qwen))
        model.removeVisionPack(for: qwen)
        try await waitUntil("nonselected add-on removal runs") {
            !visionInstaller.removedTextModelDirectories.isEmpty
        }

        #expect(visionInstaller.removedTextModelDirectories == [
            qwenDirectory.standardizedFileURL,
        ])
        #expect(model.selectedDescriptor == .default)
        #expect(model.loadState.isReady)
    }

    /// A client that serves one model at a time cannot switch models under a
    /// running chat. (A pooled client can; see AppModelPooledClientTests.)
    @MainActor
    @Test func singleModelClientBlocksSelectionWhileGenerating() async throws {
        let selectedDirectory = try makeCompleteModelInstall("busy-gemma")
        let otherDirectory = try makeCompleteModelInstall("busy-qwen", stampedAs: .qwen36)
        defer {
            try? FileManager.default.removeItem(at: selectedDirectory)
            try? FileManager.default.removeItem(at: otherDirectory)
        }
        let other = ModelInstallCoordinator(
            descriptor: .qwen36, directoryURL: otherDirectory,
            client: MockModelInstallerClient(descriptor: .qwen36))
        let client = MockInferenceClient(response: "one two three", tokenDelayNanos: 200_000_000)
        let model = makeAppModel(modelDirectory: selectedDirectory, client: client,
                                 installer: MockModelInstallerClient(descriptor: .default),
                                 otherInstalls: [other])
        model.loadState = .ready(modelDirectory: selectedDirectory, loadSeconds: 1)
        model.promptText = "go"
        model.run()
        #expect(model.isRunning)

        #expect(!model.canSelectModel(other))
        model.selectModel(other)
        #expect(model.selectedModelID == AppModelInstallDescriptor.default.id)
        model.cancel()
        for _ in 0..<400 where model.isRunning { try? await Task.sleep(nanoseconds: 5_000_000) }
    }

    @MainActor
    @Test func aFinishedDownloadSelectsItsModelWhenNothingIsInstalled() async throws {
        let qwenDirectory = temporaryInstallPath("finish-qwen")
        let selected = MockModelInstallerClient(descriptor: .default)
        let installed = try makeCompleteModelInstall(
            "finish-selects", stampedAs: .qwen36)
        defer { try? FileManager.default.removeItem(at: installed) }
        let other = MockModelInstallerClient(
            events: [.installed(installed)],
            descriptor: .qwen36)
        let coordinator = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: qwenDirectory,
            client: other)
        let model = makeAppModel(
            modelDirectory: temporaryInstallPath("finish-gemma"),
            client: MockLifecycleInferenceClient(),
            installer: selected,
            otherInstalls: [coordinator])

        #expect(!model.isModelInstalled)
        coordinator.install()
        try await waitUntil { coordinator.isInstalled }

        #expect(model.selectedModelID == AppModelInstallDescriptor.qwen36.id)
        #expect(model.isModelInstalled)
        #expect(model.modelPathText == installed.standardizedFileURL.path)
    }
}

/// Poll the main actor until `condition` holds or the budget runs out.
@MainActor
private func waitUntil(_ condition: () -> Bool) async throws {
    for _ in 0..<200 {
        if condition() { return }
        try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("condition did not become true")
}

@MainActor
private func waitUntil(_ label: String, _ condition: () -> Bool) async throws {
    for _ in 0..<200 {
        if condition() { return }
        try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("condition did not become true: \(label)")
}
