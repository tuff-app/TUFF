import Foundation
import CryptoKit
import Synchronization
import Testing
import TUFFInferenceTypes
@testable import TUFFRuntimeManagement

@Suite struct InferenceEngineArchiveTests {
    private let helper = Data("engine".utf8)

    private func tar(_ entries: [(String, Data, UInt8)]) -> Data {
        var result = Data()
        for (path, body, type) in entries {
            var header = [UInt8](repeating: 0, count: 512)
            func write(_ text: String, at offset: Int) {
                for (index, byte) in text.utf8.enumerated() { header[offset + index] = byte }
            }
            write(path, at: 0)
            write("0000755\0", at: 100)
            write(String(format: "%011o", body.count) + "\0", at: 124)
            for index in 148..<156 { header[index] = 32 }
            header[156] = type
            write("ustar\0", at: 257)
            write("00", at: 263)
            let checksum = header.reduce(0) { $0 + Int($1) }
            write(String(format: "%06o", checksum) + "\0 ", at: 148)
            result.append(contentsOf: header)
            result.append(body)
            result.append(Data(repeating: 0, count: (512 - body.count % 512) % 512))
        }
        result.append(Data(repeating: 0, count: 1024))
        return result
    }

    private func signed(_ archive: Data, archiveBytes: UInt64? = nil)
        throws -> (SignedInferenceEngineManifest, [String: Data]) {
        let key = Curve25519.Signing.PrivateKey()
        func hash(_ bytes: Data) -> String {
            SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        }
        let manifest = InferenceEngineManifest(packageID: "gemma",
            version: .init(major: 9, minor: 0, patch: 0), minimumMacOSMajorVersion: 15,
            minimumProtocolVersion: 3, maximumProtocolVersion: 3,
            architectureIDs: ["gemma4-e4b"], modelFormatVersions: [1],
            quantizationLayouts: ["affine"], residencyModes: [.resident, .streamed],
            executablePath: "bin/helper", archiveBytes: archiveBytes ?? UInt64(archive.count),
            archiveSHA256: hash(archive), files: [
                .init(path: "bin/helper", bytes: UInt64(helper.count), sha256: hash(helper), executable: true),
            ])
        let payload = try JSONEncoder().encode(manifest)
        return (SignedInferenceEngineManifest(publisherKeyID: "test", payload: payload,
            signature: try key.signature(for: SignedInferenceEngineManifest.signatureMessage(for: payload))),
            ["test": key.publicKey.rawRepresentation])
    }

    private func root() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("TUFF-EngineArchiveTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        return url
    }

    @Test func concurrentModelsShareDownloadsAndInstalledEnginesWorkOffline() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let archive = tar([("bin/helper", helper, 48)])
        let (signed, keys) = try signed(archive)
        let request = request()
        let release = InferenceEngineRelease(packageID: request.packageID, version: request.version,
            architectureIDs: [request.architectureID], minimumMacOSMajorVersion: 15,
            manifestURL: request.manifestURL, archiveURL: request.archiveURL,
            archiveBytes: UInt64(archive.count), installedBytes: UInt64(helper.count))
        let catalog = InferenceEngineCatalog(publisherKeys: keys, releases: [release])
        let transport = MockEngineDownloadTransport(root: root, contents: [
            request.manifestURL: try JSONEncoder().encode(signed), request.archiveURL: archive,
        ])
        let controller = try InferenceEngineDistributionController(root: root.appendingPathComponent("engines"),
            catalog: catalog, transport: transport)
        async let first = controller.acquire(architectureID: request.architectureID,
            quantizationLayout: "affine", protocolVersion: 3, macOSMajorVersion: 15)
        async let second = controller.acquire(architectureID: request.architectureID,
            quantizationLayout: "affine", protocolVersion: 3, macOSMajorVersion: 15)
        let (a, b) = try await (first, second)
        #expect(a.executableURL == b.executableURL)
        #expect(await transport.requests.count == 2)
        try await controller.bindModel("model-a", engine: a)
        try await controller.release(a)
        try await controller.release(b)
        let offline = MockEngineDownloadTransport(root: root, contents: [:])
        let restarted = try InferenceEngineDistributionController(root: root.appendingPathComponent("engines"),
            catalog: catalog, transport: offline)
        let lease = try await restarted.acquire(architectureID: request.architectureID,
            quantizationLayout: "affine", protocolVersion: 3, macOSMajorVersion: 15)
        #expect(await offline.requests.isEmpty)
        try await restarted.release(lease)
        #expect(try await restarted.removeUnused(.init(packageID: request.packageID, version: request.version)) == false)
        try await restarted.unbindModel("model-a")
        #expect(try await restarted.removeUnused(.init(packageID: request.packageID, version: request.version)))
    }

    /// One catalog per engine release, as an app update ships it. Every
    /// version is signed by the same publisher key and served by one transport.
    private struct ReleaseSeries {
        let keys: [String: Data]
        let transport: MockEngineDownloadTransport
        let releases: [InferenceEngineRelease]

        func controller(root: URL, current: Int) throws -> InferenceEngineDistributionController {
            try InferenceEngineDistributionController(root: root.appendingPathComponent("engines"),
                catalog: InferenceEngineCatalog(publisherKeys: keys, releases: [releases[current]]),
                transport: transport)
        }
    }

    private func series(root: URL, versions: [InferenceEngineVersion]) throws -> ReleaseSeries {
        let key = Curve25519.Signing.PrivateKey()
        func hash(_ bytes: Data) -> String {
            SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        }
        var contents: [URL: Data] = [:]
        var releases: [InferenceEngineRelease] = []
        for version in versions {
            // A different payload per version, as real releases have.
            let helper = Data("engine \(version.directoryName)".utf8)
            let archive = tar([("bin/helper", helper, 48)])
            let manifest = InferenceEngineManifest(packageID: "gemma",
                version: version, minimumMacOSMajorVersion: 15,
                minimumProtocolVersion: 3, maximumProtocolVersion: 3,
                architectureIDs: ["gemma4-e4b"], modelFormatVersions: [1],
                quantizationLayouts: ["affine"], residencyModes: [.resident, .streamed],
                executablePath: "bin/helper", archiveBytes: UInt64(archive.count),
                archiveSHA256: hash(archive), files: [
                    .init(path: "bin/helper", bytes: UInt64(helper.count), sha256: hash(helper), executable: true),
                ])
            let payload = try JSONEncoder().encode(manifest)
            let signed = SignedInferenceEngineManifest(publisherKeyID: "test", payload: payload,
                signature: try key.signature(for: SignedInferenceEngineManifest.signatureMessage(for: payload)))
            let base = "https://github.com/tuff-app/TUFF/releases/download/v\(version.directoryName)/gemma"
            let manifestURL = URL(string: base + ".json")!, archiveURL = URL(string: base + ".tar")!
            contents[manifestURL] = try JSONEncoder().encode(signed)
            contents[archiveURL] = archive
            releases.append(InferenceEngineRelease(packageID: "gemma", version: version,
                architectureIDs: ["gemma4-e4b"], minimumMacOSMajorVersion: 15,
                manifestURL: manifestURL, archiveURL: archiveURL,
                archiveBytes: UInt64(archive.count), installedBytes: UInt64(helper.count)))
        }
        return ReleaseSeries(keys: ["test": key.publicKey.rawRepresentation],
            transport: MockEngineDownloadTransport(root: root, contents: contents), releases: releases)
    }

    private func installedVersions(_ root: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(
            atPath: root.appendingPathComponent("engines/gemma").path)
            .filter { !$0.hasPrefix(".") }.sorted()
    }

    private func acquire(_ controller: InferenceEngineDistributionController) async throws -> InferenceEngineLease {
        try await controller.acquire(architectureID: "gemma4-e4b",
            quantizationLayout: "affine", protocolVersion: 3, macOSMajorVersion: 15)
    }

    /// What an app update leaves behind: the engine the previous release
    /// used. It goes once no model is bound to it; the current one stays even
    /// before any model has used it. `minimumAge: 0` here; the waiting period
    /// has its own test.
    @Test func supersededEnginesAreRemovedOnceNoModelIsBoundToThem() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let series = try series(root: root, versions: [
            .init(major: 9, minor: 0, patch: 0), .init(major: 9, minor: 1, patch: 0),
        ])

        // The previous app release: one model loaded on 9.0.0 and proven good.
        let before = try series.controller(root: root, current: 0)
        let old = try await acquire(before)
        try await before.bindModel("model-a", engine: old)
        try await before.markKnownGood("model-a", engine: old)
        try await before.release(old)
        // Nothing is superseded yet: the only version is the current one.
        #expect(await before.removeSupersededPackages(minimumAge: 0) == 0)

        // The update. 9.1.0 is fetched at launch, before any model uses it.
        let after = try series.controller(root: root, current: 1)
        let new = try await acquire(after)
        try await after.release(new)
        #expect(await after.removeSupersededPackages(minimumAge: 0) == 0)
        #expect(try installedVersions(root) == ["9.0.0", "9.1.0"])

        // The model's first load on 9.1.0 keeps 9.0.0 bound for rollback.
        let first = try await acquire(after)
        try await after.bindModel("model-a", engine: first)
        #expect(await after.removeSupersededPackages(minimumAge: 0) == 0)
        #expect(try installedVersions(root) == ["9.0.0", "9.1.0"])

        // Proven good on 9.1.0 and bound again, as the next load does: now
        // nothing can load 9.0.0. A live lease on 9.1.0 does not matter.
        try await after.markKnownGood("model-a", engine: first)
        try await after.bindModel("model-a", engine: first)
        #expect(await after.removeSupersededPackages(minimumAge: 0) == 1)
        #expect(try installedVersions(root) == ["9.1.0"])
        try await after.release(first)

        // The current version stays although no model is bound to it; the
        // Settings button is what removes an engine nothing uses at all.
        try await after.unbindModel("model-a")
        #expect(await after.removeSupersededPackages(minimumAge: 0) == 0)
        #expect(try installedVersions(root) == ["9.1.0"])
        #expect(try await after.removeUnusedPackages() == 1)
    }

    /// A superseded engine a process is still running is left for a later launch.
    @Test func aSupersededEngineInUseIsKept() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let series = try series(root: root, versions: [
            .init(major: 9, minor: 0, patch: 0), .init(major: 9, minor: 1, patch: 0),
        ])
        let before = try series.controller(root: root, current: 0)
        let running = try await acquire(before)

        let after = try series.controller(root: root, current: 1)
        try await after.release(try await acquire(after))
        #expect(await after.removeSupersededPackages(minimumAge: 0) == 0)
        #expect(try installedVersions(root) == ["9.0.0", "9.1.0"])

        try await before.release(running)
        #expect(await after.removeSupersededPackages(minimumAge: 0) == 1)
        #expect(try installedVersions(root) == ["9.1.0"])
    }

    /// A version put in place a moment ago may be one another build is still
    /// installing, between its installation and its lease, so it waits a day.
    @Test func aSupersededEngineInstalledRecentlyIsKept() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let series = try series(root: root, versions: [
            .init(major: 9, minor: 0, patch: 0), .init(major: 9, minor: 1, patch: 0),
        ])
        let before = try series.controller(root: root, current: 0)
        try await before.release(try await acquire(before))
        let after = try series.controller(root: root, current: 1)
        try await after.release(try await acquire(after))

        #expect(await after.removeSupersededPackages() == 0)
        #expect(await after.removeSupersededPackages(now: Date().addingTimeInterval(86_000)) == 0)
        #expect(try installedVersions(root) == ["9.0.0", "9.1.0"])
        #expect(await after.removeSupersededPackages(now: Date().addingTimeInterval(86_500)) == 1)
        #expect(try installedVersions(root) == ["9.1.0"])
    }

    /// Two builds can share the folder. The older one must not take the
    /// newer one's engine, bound or not, however long it has been there.
    @Test func anOlderBuildNeverRemovesANewerBuildsEngine() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let series = try series(root: root, versions: [
            .init(major: 9, minor: 0, patch: 0), .init(major: 9, minor: 1, patch: 0),
            .init(major: 10, minor: 0, patch: 0),
        ])
        for index in 0..<3 {
            let build = try series.controller(root: root, current: index)
            try await build.release(try await acquire(build))
        }
        let middle = try series.controller(root: root, current: 1)
        #expect(await middle.removeSupersededPackages(minimumAge: 0) == 1)
        #expect(try installedVersions(root) == ["10.0.0", "9.1.0"])
        let oldest = try series.controller(root: root, current: 0)
        #expect(await oldest.removeSupersededPackages(minimumAge: 0) == 0)
        #expect(try installedVersions(root) == ["10.0.0", "9.1.0"])
    }

    /// A model on a drive that is not plugged in keeps its engine. Only the
    /// Settings button drops the bindings of folders that are gone.
    @Test func theEngineOfAModelWhoseFolderIsMissingIsKept() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let series = try series(root: root, versions: [
            .init(major: 9, minor: 0, patch: 0), .init(major: 9, minor: 1, patch: 0),
        ])
        let before = try series.controller(root: root, current: 0)
        let old = try await acquire(before)
        try await before.bindModel("model-away", engine: old,
            directory: root.appendingPathComponent("unplugged/model.gturbo"))
        try await before.release(old)

        let after = try series.controller(root: root, current: 1)
        try await after.release(try await acquire(after))
        #expect(await after.removeSupersededPackages(minimumAge: 0) == 0)
        #expect(try installedVersions(root) == ["9.0.0", "9.1.0"])
        #expect(try await after.removeUnusedPackages() == 2)
    }

    /// The app, the CLI and the server share the bindings file. A process
    /// that found another one reading it used to fail its model load with
    /// `packageBusy`; it now waits for the short time the other needs.
    @Test func bindingWaitsForAnotherProcessReadingTheBindings() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let series = try series(root: root, versions: [.init(major: 9, minor: 0, patch: 0)])
        let controller = try series.controller(root: root, current: 0)
        let lease = try await acquire(controller)
        try await controller.bindModel("model-a", engine: lease)

        // Another process's shared hold, as a status or cleanup pass takes
        // it. It is released only after the store has been seen to find the
        // lock busy, so the bind below can only succeed by waiting.
        let lock = root.appendingPathComponent("engines/.locks/model-bindings-0.0.0.lock")
        let fd = open(lock.path, O_RDWR)
        try #require(fd >= 0)
        try #require(flock(fd, LOCK_SH | LOCK_NB) == 0)
        let waitsBefore = controller.bindingsLockWaits
        let bind = Task { try await controller.bindModel("model-b", engine: lease) }
        while controller.bindingsLockWaits == waitsBefore {
            try await Task.sleep(for: .milliseconds(2))
        }
        flock(fd, LOCK_UN)
        close(fd)
        try await bind.value
        try await controller.release(lease)
    }

    /// A version whose interrupted repair is still waiting is left for the
    /// build that started it, and so is one that build has just recovered,
    /// however old its files are.
    @Test func aHalfRepairedOrJustRecoveredSupersededEngineIsKept() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let series = try series(root: root, versions: [
            .init(major: 9, minor: 0, patch: 0), .init(major: 9, minor: 1, patch: 0),
        ])
        let before = try series.controller(root: root, current: 0)
        try await before.release(try await acquire(before))
        let after = try series.controller(root: root, current: 1)
        try await after.release(try await acquire(after))
        let package = root.appendingPathComponent("engines/gemma")
        let old = package.appendingPathComponent("9.0.0")
        let backup = package.appendingPathComponent(".repair-backup-9.0.0")
        let longAgo = Date().addingTimeInterval(-30 * 86_400)
        let later = Date().addingTimeInterval(86_500)
        func age(_ url: URL) throws {
            try FileManager.default.setAttributes([.modificationDate: longAgo], ofItemAtPath: url.path)
        }

        // An interrupted repair left its backup beside the version. Nothing
        // here is recent, and the version still stays.
        try FileManager.default.createDirectory(at: backup, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        try age(old); try age(package)
        #expect(await after.removeSupersededPackages() == 0)
        #expect(try installedVersions(root) == ["9.0.0", "9.1.0"])

        // The older build comes back and recovers: the version is missing
        // and the backup holds its month-old files.
        try FileManager.default.removeItem(at: backup)
        try FileManager.default.moveItem(at: old, to: backup)
        try age(backup); try age(package)
        try await before.release(try await acquire(before))
        #expect(!FileManager.default.fileExists(atPath: backup.path))
        #expect(await after.removeSupersededPackages() == 0)

        // A build from before this change recovers without marking the
        // version as newly placed. The rename still shows in the package
        // folder, and that alone keeps the version.
        try age(old)
        #expect(await after.removeSupersededPackages() == 0)
        #expect(try installedVersions(root) == ["9.0.0", "9.1.0"])

        // A day later, with nobody bound to it or running it, it goes.
        #expect(await after.removeSupersededPackages(now: later) == 1)
        #expect(try installedVersions(root) == ["9.1.0"])
    }

    @Test func verifiedArchiveInstallsThroughThePackageStore() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let bytes = tar([("bin/", Data(), 53), ("bin/helper", helper, 48)])
        let (signed, keys) = try signed(bytes)
        let archive = root.appendingPathComponent("engine.tar")
        try bytes.write(to: archive)
        let destination = root.appendingPathComponent("extracted")
        let manifest = try InferenceEngineArchive.extract(archiveURL: archive, destination: destination,
            signedManifest: signed, trustedPublisherKeys: keys)
        #expect(try Data(contentsOf: destination.appendingPathComponent("bin/helper")) == helper)
        #expect(FileManager.default.isExecutableFile(atPath: destination.appendingPathComponent("bin/helper").path))
        let store = InferenceEngineStore(root: root.appendingPathComponent("engines"), trustedPublisherKeys: keys)
        try await store.install(extractedDirectory: destination, signedManifest: signed)
        let lease = try await store.acquire(packageID: manifest.packageID, version: manifest.version,
            architectureID: "gemma4-e4b", modelFormatVersion: 1, quantizationLayout: "affine",
            protocolVersion: 3, macOSMajorVersion: 15)
        #expect(try Data(contentsOf: lease.executableURL) == helper)
        try await store.release(lease)
    }

    @Test func linksTraversalAndDuplicatesNeverEscapeTheStagingDirectory() throws {
        let invalidArchives = [
            tar([("../escape", helper, 48)]),
            tar([("bin/helper", Data(), 50)]),
            tar([("bin/helper", helper, 48), ("bin/helper", helper, 48)]),
            tar([("bin/helper", helper, 48), ("unexpected", helper, 48)]),
        ]
        for bytes in invalidArchives {
            let root = try root()
            defer { try? FileManager.default.removeItem(at: root) }
            let (signed, keys) = try signed(bytes)
            let archive = root.appendingPathComponent("engine.tar")
            try bytes.write(to: archive)
            let destination = root.appendingPathComponent("extracted")
            #expect(throws: InferenceEngineArchiveError.invalidArchive) {
                try InferenceEngineArchive.extract(archiveURL: archive, destination: destination,
                    signedManifest: signed, trustedPublisherKeys: keys)
            }
            #expect(!FileManager.default.fileExists(atPath: destination.path))
            #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("escape").path))
        }
    }

    @Test func tamperedArchivesAreRejectedBeforeAnyExtraction() throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        var bytes = tar([("bin/helper", helper, 48)])
        let (signed, keys) = try signed(bytes)
        bytes[512] ^= 1
        let archive = root.appendingPathComponent("engine.tar")
        try bytes.write(to: archive)
        let destination = root.appendingPathComponent("extracted")
        #expect(throws: InferenceEngineArchiveError.archiveDigestMismatch) {
            try InferenceEngineArchive.extract(archiveURL: archive, destination: destination,
                signedManifest: signed, trustedPublisherKeys: keys)
        }
        #expect(!FileManager.default.fileExists(atPath: destination.path))
    }

    @Test func badChecksumsMissingTerminatorsAndOversizedArchivesFailClosed() throws {
        let valid = tar([("bin/helper", helper, 48)])
        var badChecksum = valid
        badChecksum[148] = 49
        for bytes in [badChecksum, Data(valid.dropLast(1024))] {
            let root = try root()
            defer { try? FileManager.default.removeItem(at: root) }
            let (signed, keys) = try signed(bytes)
            let archive = root.appendingPathComponent("engine.tar")
            try bytes.write(to: archive)
            #expect(throws: InferenceEngineArchiveError.invalidArchive) {
                try InferenceEngineArchive.extract(archiveURL: archive,
                    destination: root.appendingPathComponent("extracted"),
                    signedManifest: signed, trustedPublisherKeys: keys)
            }
        }
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let (signed, keys) = try signed(valid, archiveBytes: InferenceEngineArchive.maximumArchiveBytes + 512)
        let archive = root.appendingPathComponent("engine.tar")
        try valid.write(to: archive)
        #expect(throws: InferenceEngineArchiveError.archiveTooLarge) {
            try InferenceEngineArchive.extract(archiveURL: archive,
                destination: root.appendingPathComponent("extracted"),
                signedManifest: signed, trustedPublisherKeys: keys)
        }
    }

    private func request(architecture: String = "gemma4-e4b",
                         version: InferenceEngineVersion = .init(major: 9, minor: 0, patch: 0))
        -> InferenceEngineInstallRequest {
        .init(packageID: "gemma", version: version,
            manifestURL: URL(string: "https://github.com/tuff-app/TUFF/releases/download/v9.0.0/gemma.json")!,
            archiveURL: URL(string: "https://github.com/tuff-app/TUFF/releases/download/v9.0.0/gemma.tar")!,
            architectureID: architecture, modelFormatVersion: 1, quantizationLayout: "affine",
            protocolVersion: 3, macOSMajorVersion: 15)
    }

    @Test func downloadPipelineVerifiesBeforeActivatingTheEngine() async throws {
        let root = try root()
        defer { try? FileManager.default.removeItem(at: root) }
        let archive = tar([("bin/helper", helper, 48)])
        let (signed, keys) = try signed(archive)
        let request = request()
        let transport = MockEngineDownloadTransport(root: root, contents: [
            request.manifestURL: try JSONEncoder().encode(signed), request.archiveURL: archive,
        ])
        let store = InferenceEngineStore(root: root.appendingPathComponent("engines"), trustedPublisherKeys: keys)
        let installer = InferenceEngineInstaller(store: store, trustedPublisherKeys: keys, transport: transport)
        let manifest = try await installer.install(request)
        #expect(manifest.packageID == "gemma")
        let requests = await transport.requests
        #expect(requests.count == 2)
        #expect(requests[1].maximumBytes == UInt64(archive.count))
        let lease = try await store.acquire(packageID: "gemma", version: manifest.version,
            architectureID: request.architectureID, modelFormatVersion: 1,
            quantizationLayout: "affine", protocolVersion: 3, macOSMajorVersion: 15)
        #expect(try Data(contentsOf: lease.executableURL) == helper)
        try await store.release(lease)
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: root.path)
            .filter { $0.hasPrefix("download-") }
        #expect(leftovers.isEmpty)
    }

    @Test func unsignedOrIncompatibleManifestsNeverDownloadAnArchive() async throws {
        for variant in ["signature", "architecture", "version"] {
            let root = try root()
            defer { try? FileManager.default.removeItem(at: root) }
            let archive = tar([("bin/helper", helper, 48)])
            let (original, keys) = try signed(archive)
            let signed = variant == "signature"
                ? SignedInferenceEngineManifest(publisherKeyID: original.publisherKeyID,
                    payload: original.payload, signature: Data(repeating: 0, count: 64)) : original
            let request = variant == "architecture" ? request(architecture: "gpt-oss-20b")
                : (variant == "version" ? request(version: .init(major: 9, minor: 1, patch: 0)) : request())
            let transport = MockEngineDownloadTransport(root: root, contents: [
                request.manifestURL: try JSONEncoder().encode(signed), request.archiveURL: archive,
            ])
            let store = InferenceEngineStore(root: root.appendingPathComponent("engines"), trustedPublisherKeys: keys)
            let installer = InferenceEngineInstaller(store: store, trustedPublisherKeys: keys, transport: transport)
            if variant == "signature" {
                await #expect(throws: InferenceEngineManifestError.invalidSignature) { try await installer.install(request) }
            } else if variant == "architecture" {
                await #expect(throws: InferenceEngineManifestError.incompatibleEngine) { try await installer.install(request) }
            } else {
                await #expect(throws: InferenceEngineStoreError.conflictingVersion) { try await installer.install(request) }
            }
            #expect(await transport.requests.count == 1)
            #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("engines").path))
        }
    }

    @Test func releaseDownloadsRefusePrivateOrCredentialBearingURLs() {
        for text in ["http://github.com/example", "https://127.0.0.1/example",
                     "https://github.com:8080/example", "https://user:secret@github.com/example",
                     "file:///tmp/engine.tar", "https://github.com.evil.example/engine.tar"] {
            #expect(!InferenceEngineHTTPTransport.permits(URL(string: text)!))
        }
        #expect(InferenceEngineHTTPTransport.permits(URL(string: "https://github.com/tuff-app/TUFF/releases")!))
        #expect(InferenceEngineHTTPTransport.permits(URL(string: "https://release-assets.githubusercontent.com/asset")!))
    }
}

private actor MockEngineDownloadTransport: InferenceEngineDownloadTransport {
    struct Request: Sendable {
        let url: URL
        let maximumBytes: UInt64
    }
    private let root: URL
    private let contents: [URL: Data]
    private(set) var requests: [Request] = []

    init(root: URL, contents: [URL: Data]) {
        self.root = root
        self.contents = contents
    }

    func download(from url: URL, maximumBytes: UInt64) async throws -> URL {
        requests.append(Request(url: url, maximumBytes: maximumBytes))
        let data = contents[url] ?? Data()
        guard UInt64(data.count) <= maximumBytes else { throw InferenceEngineDownloadError.oversizedDownload }
        let file = root.appendingPathComponent("download-\(UUID().uuidString)")
        try data.write(to: file)
        return file
    }
}
