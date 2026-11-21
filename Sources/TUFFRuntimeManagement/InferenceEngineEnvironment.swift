import Foundation
import Synchronization
import TUFFModelCatalog
import TUFFModelMetadata
import TUFFRuntimeDescriptions
import Darwin

public enum InferenceEngineEnvironmentError: Error, CustomStringConvertible {
    case missingCatalog
    case invalidCatalogFile
    public var description: String {
        switch self {
        case .missingCatalog: "Inference engine catalogue is missing. Use the local 9.0 build or a packaged TUFF app."
        case .invalidCatalogFile: "Inference engine catalogue is unreadable or too large."
        }
    }
}

/// One loan per host use. Helpers take their own file-lock lease before serving
/// requests, so release of this host loan cannot remove a running engine.
public final class InferenceEngineInstallation: @unchecked Sendable {
    public let lease: InferenceEngineLease
    public let controller: InferenceEngineDistributionController
    private let released = Mutex(false)
    public var executableURL: URL { lease.executableURL }
    public var lockFileURL: URL { lease.lockFileURL }
    public var identity: String { lease.manifest.archiveSHA256 }

    init(lease: InferenceEngineLease, controller: InferenceEngineDistributionController) {
        self.lease = lease
        self.controller = controller
    }
    public func release() async {
        guard released.withLock({ if $0 { return false }; $0 = true; return true }) else { return }
        try? await controller.release(lease)
    }
    deinit {
        if released.withLock({ if $0 { return false }; $0 = true; return true }) {
            let controller = controller, lease = lease
            Task { try? await controller.release(lease) }
        }
    }
}

/// The app bundle supplies the catalogue and publisher keys. Local qualification
/// can explicitly select the generated catalogue with TUFF_ENGINE_CATALOG_PATH.
public enum InferenceEngineEnvironment {
    /// Cache identity only. The host/store authenticates code before launch.
    /// Development helpers use a process identity so persisted state cannot
    /// survive an unversioned source change.
    public static let computationIdentity: String = {
        let fallback = "development:" + UUID().uuidString
        guard let executable = Bundle.main.executableURL else { return fallback }
        let receipt = executable.deletingLastPathComponent().appendingPathComponent(".engine-receipt.json")
        let fd = open(receipt.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { return fallback }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_uid == getuid(),
              info.st_size > 0, info.st_size <= 1_048_576,
              let data = try? FileHandle(fileDescriptor: fd, closeOnDealloc: false).read(upToCount: 1_048_577),
              let signed = try? JSONDecoder().decode(SignedInferenceEngineManifest.self, from: data),
              let manifest = try? JSONDecoder().decode(InferenceEngineManifest.self, from: signed.payload),
              (try? manifest.validate()) != nil else { return fallback }
        return "\(manifest.packageID):\(manifest.version.directoryName):\(manifest.archiveSHA256)"
    }()
    private static let cached = Mutex<(key: String, value: InferenceEngineDistributionController)?>(nil)
    public static func identity(architectureID: String) throws -> String {
        let controller = try controller()
        let release = try controller.catalog.release(architectureID: architectureID,
            macOSMajorVersion: ProcessInfo.processInfo.operatingSystemVersion.majorVersion)
        let fallback = release.packageID + ":" + release.version.directoryName
        let receipt = TUFFApplicationDirectories.applicationSupportURL()
            .appendingPathComponent("TUFF/Engines/\(release.packageID)/\(release.version.directoryName)/.engine-receipt.json")
        let fd = open(receipt.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { return fallback }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == getuid(), info.st_size > 0, info.st_size <= 1 << 20 else { return fallback }
        let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: false)
        guard let data = try handle.read(upToCount: (1 << 20) + 1), data.count <= 1 << 20,
              let signed = try? JSONDecoder().decode(SignedInferenceEngineManifest.self, from: data),
              let manifest = try? signed.verifiedManifest(trustedPublisherKeys: controller.catalog.publisherKeys),
              manifest.packageID == release.packageID, manifest.version == release.version else { return fallback }
        return fallback + ":" + manifest.archiveSHA256
    }
    public static func catalogURL() throws -> URL {
        if let path = ProcessInfo.processInfo.environment["TUFF_ENGINE_CATALOG_PATH"], path.hasPrefix("/"), !path.contains("\0") {
            return URL(fileURLWithPath: path).standardizedFileURL
        }
        if let url = Bundle.main.url(forResource: "engine-catalog", withExtension: "json") { return url }
        if let executable = Bundle.main.executableURL {
            let sibling = executable.deletingLastPathComponent().appendingPathComponent("engine-catalog.json")
            if FileManager.default.fileExists(atPath: sibling.path) { return sibling }
            var parent = executable.deletingLastPathComponent()
            while parent.path != "/" {
                if parent.pathExtension == "app" {
                    let resource = parent.appendingPathComponent("Contents/Resources/engine-catalog.json")
                    if FileManager.default.fileExists(atPath: resource.path) { return resource }
                    break
                }
                parent.deleteLastPathComponent()
            }
        }
        throw InferenceEngineEnvironmentError.missingCatalog
    }

    public static func controller() throws -> InferenceEngineDistributionController {
        let url = try catalogURL()
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let bytes = (attributes[.size] as? NSNumber)?.uint64Value, bytes > 0, bytes <= 1_048_576 else {
            throw InferenceEngineEnvironmentError.invalidCatalogFile
        }
        let root = TUFFApplicationDirectories.applicationSupportURL().appendingPathComponent("TUFF/Engines", isDirectory: true)
        let key = "\(url.path):\(root.path):\(attributes[.modificationDate] ?? ""):\(bytes)"
        return try cached.withLock { state in
            if let state, state.key == key { return state.value }
            let catalog = try JSONDecoder().decode(InferenceEngineCatalog.self, from: Data(contentsOf: url))
            let controller = try InferenceEngineDistributionController(root: root, catalog: catalog,
                transport: InferenceEngineSourceTransport(localRoot: catalog.localPackageRoot))
            state = (key, controller)
            return controller
        }
    }

    public static func acquire(architectureID: String, quantizationLayout: String,
                               protocolVersion: Int = 1) async throws -> InferenceEngineInstallation {
        let controller = try controller()
        let lease = try await controller.acquire(architectureID: architectureID,
            quantizationLayout: quantizationLayout, protocolVersion: protocolVersion,
            macOSMajorVersion: ProcessInfo.processInfo.operatingSystemVersion.majorVersion)
        return InferenceEngineInstallation(lease: lease, controller: controller)
    }

    public static func acquireModel(at directory: URL, protocolVersion: Int = 1) async throws -> InferenceEngineInstallation {
        let arch = try ManifestReader.resolveArchitecture(directoryURL: directory)
        let installation: InferenceEngineInstallation
        do {
            installation = try await acquire(architectureID: arch.variant.rawValue,
                quantizationLayout: arch.family.engineQuantizationLayout, protocolVersion: protocolVersion)
        } catch {
            try Task.checkCancellation()
            let controller = try controller()
            guard let lease = try? await controller.acquireKnownGood(bindingID(for: directory),
                architectureID: arch.variant.rawValue, quantizationLayout: arch.family.engineQuantizationLayout,
                protocolVersion: protocolVersion, macOSMajorVersion: ProcessInfo.processInfo.operatingSystemVersion.majorVersion) else { throw error }
            installation = InferenceEngineInstallation(lease: lease, controller: controller)
        }
        do {
            try await installation.controller.bindModel(bindingID(for: directory), engine: installation.lease, directory: directory)
        } catch {
            await installation.release()
            throw error
        }
        return installation
    }

    public static func bind(_ installation: InferenceEngineInstallation, to directory: URL) async throws {
        try await installation.controller.bindModel(bindingID(for: directory), engine: installation.lease, directory: directory)
    }
    public static func acquireKnownGoodModel(at directory: URL,
                                            excluding identity: String? = nil) async throws -> InferenceEngineInstallation? {
        let arch = try ManifestReader.resolveArchitecture(directoryURL: directory)
        let controller = try controller()
        guard let lease = try await controller.acquireKnownGood(bindingID(for: directory),
            architectureID: arch.variant.rawValue, quantizationLayout: arch.family.engineQuantizationLayout,
            protocolVersion: 1, macOSMajorVersion: ProcessInfo.processInfo.operatingSystemVersion.majorVersion) else { return nil }
        if lease.manifest.archiveSHA256 == identity {
            try await controller.release(lease)
            return nil
        }
        return InferenceEngineInstallation(lease: lease, controller: controller)
    }
    public static func markKnownGood(_ installation: InferenceEngineInstallation, for directory: URL) async {
        try? await installation.controller.markKnownGood(bindingID(for: directory), engine: installation.lease)
    }
    public static func unbindModel(at directory: URL) async throws {
        try await controller().unbindModel(bindingID(for: directory))
    }
    public static func bindingID(for directory: URL) -> String {
        "model-" + Sha256Verifier.hashData(Data(directory.standardizedFileURL.resolvingSymlinksInPath().path.utf8))
    }
}
