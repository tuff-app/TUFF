import Foundation
import TUFFRepackCore
import TUFFModelCatalog
import TUFFRuntimeManagement

private let supportedModelSelectors = SupportedModelSource.all
    .map(\.name).joined(separator: "|")

private let usage = """
Usage:
  TUFFRepack [--model <\(supportedModelSelectors)>] --output <model.gturbo> [--overwrite] [--resume] [--dry-run] [--layout-summary-output <json>] [--repo-id <owner/repo>] [--revision <ref>]
  TUFFRepack --discard-partial --output <model.gturbo>
  TUFFRepack --verify-install --input-gturbo <model.gturbo>
  TUFFRepack --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo> [--overwrite] [--resume]
  TUFFRepack --verify-vision-install
                       --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo>
  TUFFRepack --activate-vision-install
                       --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo>
  TUFFRepack --remove-vision-install
                       --vision-output <model.vision.gturbo>
  TUFFRepack --discard-partial
                       --vision-output <model.vision.gturbo>
  TUFFRepack --help

The installer streams the selected checkpoint (default: the supported Gemma 4
checkpoint) from Hugging Face and repackages it without materializing the
source checkpoint on disk. --model picks which supported architecture to
install; --repo-id and optionally --revision override where that architecture's
weights come from. Overriding the source waives the pinned-fingerprint check.
Set HF_TOKEN only if Hugging Face requests authentication. A cancelled or
interrupted download can be continued with --resume or removed with
--discard-partial.

An optional compatible image companion pack installs beside its matching text
model and is bound to it. The architecture is inferred from the text
manifest (or selected explicitly with --model). Without the pack the text
runtime is unchanged; image input is unavailable.
"""

private struct Arguments {
    var model = SupportedModelSource.default
    var modelWasExplicit = false
    var output: String?
    var overwrite = false
    var resume = false
    var dryRun = false
    var layoutSummaryOutput: String?
    var discardPartial = false
    var verifyInstall = false
    var inputGTurbo: String?
    /// Optional override to fetch from a different Hugging Face repo (owner/repo)
    var repoID: String?
    /// Optional revision (branch, tag, or commit). If omitted and repoID is set, defaults to "main".
    var revision: String?
    var visionOutput: String?
    var textModel: String?
    var verifyVisionInstall = false
    var activateVisionInstall = false
    var removeVisionInstall = false

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--dry-run":
                parsed.dryRun = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--verify-vision-install":
                parsed.verifyVisionInstall = true
                index += 1
            case "--activate-vision-install":
                parsed.activateVisionInstall = true
                index += 1
            case "--remove-vision-install":
                parsed.removeVisionInstall = true
                index += 1
            case "--vision-output":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.visionOutput = values[index + 1]
                index += 2
            case "--text-model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.textModel = values[index + 1]
                index += 2
            case "--model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                guard let source = SupportedModelSource.named(values[index + 1]) else {
                    throw ParseError.invalidMode(
                        "unknown model \"\(values[index + 1])\"; supported: "
                        + SupportedModelSource.all.map(\.name).joined(separator: ", "))
                }
                parsed.model = source
                parsed.modelWasExplicit = true
                index += 2
            case "--layout-summary-output", "--output", "--input-gturbo", "--repo-id", "--revision":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                let value = values[index + 1]
                switch flag {
                case "--layout-summary-output": parsed.layoutSummaryOutput = value
                case "--output": parsed.output = value
                case "--input-gturbo": parsed.inputGTurbo = value
                case "--repo-id": parsed.repoID = value
                case "--revision": parsed.revision = value
                default: break
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        let visionModes = [parsed.verifyVisionInstall,
                           parsed.activateVisionInstall,
                           parsed.removeVisionInstall].filter { $0 }.count
        guard visionModes <= 1 else {
            throw ParseError.invalidMode("vision install modes are mutually exclusive")
        }
        if visionModes == 1 || parsed.visionOutput != nil {
            guard parsed.visionOutput != nil else {
                throw ParseError.missingRequired("--vision-output")
            }
            guard parsed.output == nil, parsed.inputGTurbo == nil,
                  !parsed.verifyInstall else {
                throw ParseError.invalidMode(
                    "vision install operations do not accept text install arguments")
            }
            // Discard runs first below, so accepting it alongside another mode
            // would silently perform the discard and exit 0 without ever doing
            // what was asked.
            guard !(parsed.discardPartial && visionModes == 1) else {
                throw ParseError.invalidMode(
                    "--discard-partial is mutually exclusive with the other vision "
                        + "install operations")
            }
            if parsed.removeVisionInstall || parsed.discardPartial {
                guard parsed.textModel == nil, !parsed.overwrite, !parsed.resume else {
                    throw ParseError.invalidMode(
                        "this vision operation only accepts --vision-output")
                }
            } else if parsed.verifyVisionInstall || parsed.activateVisionInstall {
                guard parsed.textModel != nil else {
                    throw ParseError.missingRequired("--text-model")
                }
                // Neither reads a download, so a transfer flag here is a
                // request this mode cannot honour rather than a no-op.
                guard !parsed.overwrite, !parsed.resume else {
                    throw ParseError.invalidMode(
                        "this vision operation only accepts --vision-output and "
                            + "--text-model")
                }
            } else {
                guard parsed.textModel != nil else {
                    throw ParseError.missingRequired("--text-model")
                }
            }
            return parsed
        }
        guard parsed.textModel == nil else {
            throw ParseError.invalidMode("--text-model requires --vision-output")
        }
        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume else {
                throw ParseError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func runVisionInstall(_ arguments: Arguments) async -> Int32? {
    guard let visionOutput = arguments.visionOutput else { return nil }

    if arguments.discardPartial {
        do {
            try RemoteVisionPackInstaller.discardPartial(outputDirectory: visionOutput)
            print("Discarded saved image-pack download for \(visionOutput)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.removeVisionInstall {
        do {
            try RemoteVisionPackInstaller.removeInstalled(outputDirectory: visionOutput)
            print("Removed image pack \(visionOutput)")
            return 0
        } catch {
            printError("remove-vision-install failed: \(error)")
            return 1
        }
    }

    guard let textModel = arguments.textModel else { return 2 }

    if arguments.verifyVisionInstall {
        do {
            let verification = try VisionPackVerifier.verify(
                directory: URL(fileURLWithPath: visionOutput, isDirectory: true),
                installedDirectory: URL(fileURLWithPath: visionOutput, isDirectory: true),
                textModelDirectory: URL(fileURLWithPath: textModel, isDirectory: true),
                verifyWeights: true)
            print("Verified image pack \(visionOutput)")
            print("Bound to text model \(textModel)")
            print("Text manifest sha256 \(verification.compatibleTextManifestSha256)")
            return 0
        } catch {
            printError("verify-vision-install failed: \(error)")
            return 1
        }
    }

    // Resume and activation are local-state operations first. Preserve their
    // fail-fast contract before reading the text manifest, so a missing saved
    // transaction is never misreported as an unidentifiable model family.
    if arguments.resume || arguments.activateVisionInstall {
        do {
            let paths = try RemoteInstallPaths(outputDirectory: visionOutput)
            guard FileManager.default.fileExists(atPath: paths.checkpointFile) else {
                throw RepackError.installStateMissing(path: paths.checkpointFile)
            }
        } catch {
            let operation = arguments.activateVisionInstall
                ? "activate-vision-install"
                : "vision install"
            printError("\(operation) failed: \(error)")
            return 1
        }
    }

    let textSource: SupportedModelSource
    do {
        textSource = try visionSource(arguments: arguments, textModel: textModel)
    } catch {
        printError("could not identify text-model family: \(error)")
        return 2
    }
    // The catalog's image add-on names where the pack comes from. Usually the
    // text model's own checkpoint; for Ornith 9B, 35B and 122B, Ornith's BF16
    // release.
    let addon = TUFFModelCatalog.model(manifestModelID: textSource.modelID)?
        .addons.first { $0.kind == .imageInput }?.source
    let source = (repoID: addon?.repoID ?? textSource.repoID,
                  revision: addon?.revision ?? textSource.revision)

    if arguments.activateVisionInstall {
        do {
            try RemoteVisionPackInstaller.activatePrepared(
                outputDirectory: visionOutput,
                textModelDirectory: textModel,
                repoID: source.repoID,
                requestedRevision: source.revision)
            print("Activated image pack \(visionOutput)")
            return 0
        } catch {
            printError("activate-vision-install failed: \(error)")
            return 1
        }
    }

    let options = RemoteVisionPackInstallOptions(
        repoID: source.repoID,
        revision: source.revision,
        textModelDirectory: textModel,
        outputDirectory: visionOutput,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        overwrite: arguments.overwrite,
        resume: arguments.resume)
    do {
        let progress = InstallProgressReporter()
        _ = try await RemoteVisionPackInstaller(options: options).run(
            progress: { progress($0) })
        print("Installed image pack \(visionOutput)")
        print("Text model: \(textModel)")
        return 0
    } catch {
        printError("vision install failed: \(error)")
        return 1
    }
}

private func visionSource(
    arguments: Arguments,
    textModel: String
) throws -> SupportedModelSource {
    if arguments.modelWasExplicit { return arguments.model }
    let manifest = URL(fileURLWithPath: textModel, isDirectory: true)
        .appendingPathComponent("manifest.json")
    let data = try Data(contentsOf: manifest)
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let arch = root["arch"] as? [String: Any] else {
        throw ParseError.invalidMode("text manifest has no architecture")
    }
    if let modelID = root["modelID"] as? String,
       let exact = SupportedModelSource.all.first(where: { $0.modelID == modelID }) {
        return exact
    }
    return (arch["family"] as? String) == "qwen36" ? .qwen36 : .gemma4
}

/// Bonsai checkpoints are repacked by the Prism installer, which only installs
/// the pinned revision; an explicit repo or revision cannot be honoured.
private func prismSize(_ arguments: Arguments) -> PrismRemoteInstaller.Size? {
    guard arguments.repoID == nil, arguments.revision == nil,
          let model = TUFFModelCatalog.model(selector: arguments.model.name) else { return nil }
    return PrismRemoteInstaller.Size(modelID: model.id)
}

private func installPrism(_ arguments: Arguments, size: PrismRemoteInstaller.Size, output: String) async -> Int32 {
    let source = arguments.model
    guard let model = TUFFModelCatalog.model(selector: source.name) else { return 2 }
    guard !arguments.resume else {
        printError("install failed: \(source.displayName) downloads cannot be resumed; discard the partial with --discard-partial and start again")
        return 1
    }
    guard !arguments.overwrite else {
        printError("install failed: \(source.displayName) does not support --overwrite; choose a new output directory")
        return 1
    }
    let directory = URL(fileURLWithPath: output)
    var engine: InferenceEngineInstallation?
    do {
        if arguments.dryRun {
            let needed = size.snapshotBytes + size.outputReservationBytes + 8 * 1024 * 1024
            let space = try DiskSpaceChecker.assess(path: directory.path, bytes: needed,
                                                     reserveBytes: source.reserveBytes)
            guard space.canInstall else {
                throw RepackError.diskSpaceInsufficient(path: space.path, required: space.requiredBytes,
                                                        available: space.availableBytes)
            }
            print("Planned \(source.displayName)")
            print("Download: \(size.snapshotBytes) bytes; peak disk: \(needed) bytes; available: \(space.availableBytes) bytes")
            return 0
        }
        printError("[install] preparing inference engine")
        engine = try await InferenceEngineEnvironment.acquire(architectureID: model.architecture.id.rawValue,
            quantizationLayout: model.architecture.weightLayout.rawValue)
        let progress = InstallProgressReporter()
        let result = try await PrismRemoteInstaller.install(size: size, outputDirectory: directory,
            token: ProcessInfo.processInfo.environment["HF_TOKEN"], reserveBytes: source.reserveBytes,
            progress: { progress($0) })
        if let engine {
            try await InferenceEngineEnvironment.bind(engine, to: result)
            await engine.release()
        }
        print("Installed \(source.displayName)")
        print("Source revision: \(source.revision)")
        print("Model: \(result.path)")
        return 0
    } catch {
        await engine?.release()
        printError("install failed: \(error)")
        return 1
    }
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if let code = await runVisionInstall(arguments) {
        return code
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            if prismSize(arguments) != nil {
                try PrismRemoteInstaller.discardPartial(outputDirectory: URL(fileURLWithPath: output))
            } else {
                try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            }
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    let source = arguments.model
    if let size = prismSize(arguments) { return await installPrism(arguments, size: size, output: output) }
    var options = source.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume, dryRun: arguments.dryRun)
    // An explicit repo or revision replaces the architecture's pinned source,
    // which also waives the fingerprint check the pin exists to enforce.
    if arguments.repoID != nil || arguments.revision != nil {
        options = RemoteStreamingRepackOptions(
            repoID: arguments.repoID ?? source.repoID,
            revision: arguments.revision ?? source.revision,
            outputDir: URL(fileURLWithPath: output).path,
            token: ProcessInfo.processInfo.environment["HF_TOKEN"],
            requireKnownSource: false,
            minFreeReserveBytes: source.reserveBytes,
            overwrite: arguments.overwrite,
            resume: arguments.resume, dryRunSpaceCheck: arguments.dryRun)
    }
    let installedName = arguments.repoID == nil && arguments.revision == nil
        ? source.displayName
        : "\(source.displayName) from \(options.repoID)@\(options.revision)"
    var engine: InferenceEngineInstallation?
    do {
        // An absent or incompatible checkpoint is a local error. Report it
        // before acquiring an engine or contacting its publisher.
        if arguments.resume {
            guard try RemoteStreamingRepacker.inspectPersistentInstall(outputDirectory: options.outputDir,
                repoID: options.repoID, requestedRevision: options.revision) != nil else {
                throw RepackError.installStateMissing(path:
                    try RemoteInstallPaths(outputDirectory: options.outputDir).checkpointFile)
            }
        }
        if !arguments.dryRun, arguments.repoID == nil, arguments.revision == nil,
           let model = TUFFModelCatalog.model(selector: source.name) {
            printError("[install] preparing inference engine")
            engine = try await InferenceEngineEnvironment.acquire(architectureID: model.architecture.id.rawValue,
                quantizationLayout: model.architecture.weightLayout.rawValue)
        }
        let progress = InstallProgressReporter()
        let result = try await RemoteStreamingRepacker(options: options).run(
            progress: { progress($0) })
        if let output = arguments.layoutSummaryOutput {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(result.weightLayoutSummary).write(to: URL(fileURLWithPath: output), options: .atomic)
        }
        if let engine {
            try await InferenceEngineEnvironment.bind(engine, to: URL(fileURLWithPath: result.outputDir))
            await engine.release()
        }
        print("\(arguments.dryRun ? "Planned" : "Installed") \(installedName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        await engine?.release()
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
