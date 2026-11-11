import Foundation
import CryptoKit
import Darwin
import TUFFRuntimeManagement
import TUFFModelMetadata

struct PackageFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
func command(_ executable: String, _ arguments: [String]) throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    process.standardOutput = FileHandle.standardError
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { throw PackageFailure("\(executable) failed") }
}
func encode<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(value).write(to: url, options: .atomic)
}

func writeArchive(files: [InferenceEngineFile], root: URL, destination: URL) throws {
    let temporary = destination.deletingLastPathComponent().appendingPathComponent(".archive-\(UUID().uuidString)")
    guard FileManager.default.createFile(atPath: temporary.path, contents: nil) else { throw PackageFailure("cannot create archive") }
    defer { try? FileManager.default.removeItem(at: temporary) }
    let output = try FileHandle(forWritingTo: temporary)
    defer { try? output.close() }
    for file in files {
        var header = [UInt8](repeating: 0, count: 512)
        func field(_ text: String, offset: Int, width: Int) throws {
            guard text.utf8.count <= width else { throw PackageFailure("ustar field exceeds limit") }
            for (index, byte) in text.utf8.enumerated() { header[offset + index] = byte }
        }
        var name = file.path, prefix = ""
        if name.utf8.count > 100 {
            guard let separator = name.lastIndex(of: "/") else { throw PackageFailure("ustar name exceeds limit") }
            prefix = String(name[..<separator]); name = String(name[name.index(after: separator)...])
        }
        try field(name, offset: 0, width: 100)
        try field(prefix, offset: 345, width: 155)
        try field(file.executable ? "0000755\0" : "0000644\0", offset: 100, width: 8)
        try field("0000000\0", offset: 108, width: 8)
        try field("0000000\0", offset: 116, width: 8)
        try field(String(format: "%011llo", file.bytes) + "\0", offset: 124, width: 12)
        try field("00000000000\0", offset: 136, width: 12)
        for index in 148..<156 { header[index] = 32 }
        header[156] = 48
        try field("ustar\0", offset: 257, width: 6)
        try field("00", offset: 263, width: 2)
        let checksum = header.reduce(0) { $0 + Int($1) }
        try field(String(format: "%06o", checksum) + "\0 ", offset: 148, width: 8)
        try output.write(contentsOf: Data(header))
        let input = try FileHandle(forReadingFrom: root.appendingPathComponent(file.path))
        defer { try? input.close() }
        var remaining = file.bytes
        while remaining > 0 {
            guard let data = try input.read(upToCount: Int(min(remaining, 1 << 20))), !data.isEmpty else { throw PackageFailure("short engine input") }
            try output.write(contentsOf: data); remaining -= UInt64(data.count)
        }
        try output.write(contentsOf: Data(repeating: 0, count: Int((512 - file.bytes % 512) % 512)))
    }
    try output.write(contentsOf: Data(repeating: 0, count: 1024))
    try output.synchronize()
    if FileManager.default.fileExists(atPath: destination.path) {
        guard try Sha256Verifier.hashFile(at: temporary) == Sha256Verifier.hashFile(at: destination) else {
            throw PackageFailure("engine version already exists with different bytes; increment its version")
        }
    } else { try FileManager.default.moveItem(at: temporary, to: destination) }
}

// Private keys stay outside the distribution root and
// must never be copied into app resources or an engine archive.
do {
    let arguments = Array(CommandLine.arguments.dropFirst())
    guard (3...5).contains(arguments.count) else {
        throw PackageFailure("usage: TUFFEnginePackage <SwiftPM binary directory> <output directory> <private key file> [engine version] [HTTPS download base]")
    }
    let downloadBase: URL?
    if arguments.count == 5 {
        guard let url = URL(string: arguments[4]), InferenceEngineHTTPTransport.permits(url),
              url.query == nil, url.fragment == nil else {
            throw PackageFailure("download base must use a supported GitHub HTTPS host without credentials, query or fragment")
        }
        downloadBase = url
    } else { downloadBase = nil }
    let components = (arguments.count >= 4 ? arguments[3] : "1.0.0").split(separator: ".", omittingEmptySubsequences: false)
    let parts = components.compactMap { Int($0) }
    guard components.count == 3, parts.count == 3,
          components.allSatisfy({ !$0.isEmpty && $0.allSatisfy({ "0123456789".contains($0) }) }),
          parts.allSatisfy({ $0 >= 0 }) else { throw PackageFailure("invalid engine version") }
    let version = InferenceEngineVersion(major: parts[0], minor: parts[1], patch: parts[2])
    let fm = FileManager.default
    let binary = URL(fileURLWithPath: arguments[0]).standardizedFileURL.resolvingSymlinksInPath()
    // Directory enumeration resolves /tmp to /private/tmp on macOS. Keep
    // the root canonical too, so relative archive paths use the same prefix.
    let output = URL(fileURLWithPath: arguments[1]).standardizedFileURL.resolvingSymlinksInPath()
    let keyURL = URL(fileURLWithPath: arguments[2]).standardizedFileURL.resolvingSymlinksInPath()
    guard !keyURL.path.hasPrefix(output.path + "/") else { throw PackageFailure("private key must be outside output directory") }
    try fm.createDirectory(at: output, withIntermediateDirectories: true)
    try fm.createDirectory(at: keyURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    let key: Curve25519.Signing.PrivateKey
    if fm.fileExists(atPath: keyURL.path) {
        guard let attrs = try? fm.attributesOfItem(atPath: keyURL.path),
              (attrs[.type] as? FileAttributeType) == .typeRegular,
              (attrs[.posixPermissions] as? NSNumber)?.intValue == 0o600 else {
            throw PackageFailure("private key must be a regular file with mode 0600")
        }
        key = try .init(rawRepresentation: Data(contentsOf: keyURL))
    } else {
        guard downloadBase == nil else { throw PackageFailure("release engines need the existing release signing key") }
        key = .init()
        guard fm.createFile(atPath: keyURL.path, contents: key.rawRepresentation,
                            attributes: [.posixPermissions: 0o600]) else { throw PackageFailure("cannot create signing key") }
    }
    // Published engines are always signed by the one release key, so an app
    // update never stops trusting the engines users already have.
    let releasePublicKey = "NhjXokBnXf80eKZd3Mkg/p3ZKtZ71UTzK4ujiljgzuA="
    guard downloadBase == nil || key.publicKey.rawRepresentation.base64EncodedString() == releasePublicKey else {
        throw PackageFailure("release engines must be signed by the release key (public key \(releasePublicKey))")
    }
    let families: [(String, Set<String>, Set<String>)] = [
        ("gemma-affine", ["gemma4-e2b", "gemma4-e4b", "gemma4-12b-qat", "gemma4-26b-a4b"], ["affine"]),
        ("qwen-hybrid-affine", ["qwen36-35b-a3b", "qwen3.8-flash-next", "ornith-1.5-9b", "ornith-1.5-35b-a3b", "ornith-1.5-122b", "ornith-1.5-397b", "qwen3.8-27b", "ternary-bonsai-2-2b", "ternary-bonsai-2-27b"], ["affine", "prism-q2"]),
        ("gpt-oss-mxfp4", ["gpt-oss-20b", "gpt-oss-120b"], ["mxfp4"]),
        ("minimax-affine", ["minimax-m2.7"], ["affine"]),
        ("flint-affine", ["flint-40b-a2.5b"], ["affine"])
    ]
    let selectedFamilies = ProcessInfo.processInfo.environment["TUFF_ENGINE_FAMILIES"]
        .map { Set($0.split(whereSeparator: { $0.isWhitespace }).map(String.init)) }
    if let selectedFamilies {
        guard !selectedFamilies.isEmpty, selectedFamilies.isSubset(of: Set(families.map { $0.0 })) else {
            throw PackageFailure("invalid engine family selection")
        }
    }
    var releases: [InferenceEngineRelease] = []
    for (id, architectures, layouts) in families {
        if let selectedFamilies, !selectedFamilies.contains(id) { continue }
        let familyBinary = binary.appendingPathComponent(id)
        guard fm.fileExists(atPath: familyBinary.path) else { throw PackageFailure("missing separately built engine family: \(id)") }
        let stage = output.appendingPathComponent(".package-staging-\(UUID().uuidString)")
        try fm.createDirectory(at: stage, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: stage) }
        let executable = stage.appendingPathComponent("TUFFInferenceEngine")
        try fm.copyItem(at: familyBinary.appendingPathComponent("TUFFInferenceEngine"), to: executable)
        let symbols = output.appendingPathComponent("symbols/\(id)/\(version.directoryName)")
        try fm.createDirectory(at: symbols, withIntermediateDirectories: true)
        let symbolBinary = symbols.appendingPathComponent("TUFFInferenceEngine")
        if !fm.fileExists(atPath: symbolBinary.path) {
            try fm.copyItem(at: executable, to: symbolBinary)
        }
        // Retain the unstripped build inputs for local symbolication. Packages
        // ship the smaller runtime; external symbols and dynamic linkage stay.
        try command("/usr/bin/strip", ["-x", executable.path])
        try command("/usr/bin/codesign", ["--force", "--sign", "-", "--timestamp=none", executable.path])
        try command("/usr/bin/codesign", ["--verify", "--strict", executable.path])
        for name in ["TUFF_TUFFEngine.bundle", "TUFF_TUFFAppCore.bundle", "swift-transformers_Hub.bundle", "swift-crypto_Crypto.bundle"] {
            try fm.copyItem(at: familyBinary.appendingPathComponent(name), to: stage.appendingPathComponent(name))
        }
        guard let enumeration = fm.enumerator(at: stage, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]) else { throw PackageFailure("cannot enumerate engine files") }
        var files: [InferenceEngineFile] = []
        for case let url as URL in enumeration {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
            guard values.isSymbolicLink != true else { throw PackageFailure("engine resources contain a symbolic link") }
            if values.isRegularFile == true {
                let rootComponents = stage.resolvingSymlinksInPath().pathComponents
                let fileComponents = url.resolvingSymlinksInPath().pathComponents
                guard fileComponents.starts(with: rootComponents) else { throw PackageFailure("engine file escaped staging root") }
                let path = fileComponents.dropFirst(rootComponents.count).joined(separator: "/")
                files.append(.init(path: path, bytes: UInt64(values.fileSize!), sha256: try Sha256Verifier.hashFile(at: url), executable: path == "TUFFInferenceEngine"))
            }
        }
        files.sort { $0.path < $1.path }
        let archive = output.appendingPathComponent("\(id)-\(version.directoryName).tar")
        try writeArchive(files: files, root: stage, destination: archive)
        let bytes = UInt64((try fm.attributesOfItem(atPath: archive.path)[.size] as! NSNumber).uint64Value)
        guard bytes <= InferenceEngineArchive.maximumArchiveBytes else { throw PackageFailure("engine archive exceeds bounded installer limit") }
        let manifest = InferenceEngineManifest(packageID: id, version: version,
            minimumMacOSMajorVersion: 15, minimumProtocolVersion: 1, maximumProtocolVersion: 1,
            architectureIDs: architectures, modelFormatVersions: [1], quantizationLayouts: layouts,
            residencyModes: [.resident, .streamed, .partial], executablePath: "TUFFInferenceEngine",
            archiveBytes: bytes, archiveSHA256: try Sha256Verifier.hashFile(at: archive), files: files)
        try manifest.validate()
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let payload = try encoder.encode(manifest)
        let signed = SignedInferenceEngineManifest(publisherKeyID: "local-development", payload: payload,
            signature: try key.signature(for: SignedInferenceEngineManifest.signatureMessage(for: payload)))
        let verification = output.appendingPathComponent(".verification-\(UUID().uuidString)")
        _ = try InferenceEngineArchive.extract(archiveURL: archive, destination: verification,
            signedManifest: signed, trustedPublisherKeys: ["local-development": key.publicKey.rawRepresentation])
        try fm.removeItem(at: verification)
        let manifestURL = output.appendingPathComponent("\(id)-\(version.directoryName).json")
        try encode(signed, to: manifestURL)
        releases.append(.init(packageID: id, version: version, architectureIDs: architectures,
            minimumMacOSMajorVersion: 15,
            manifestURL: downloadBase?.appendingPathComponent(manifestURL.lastPathComponent) ?? manifestURL,
            archiveURL: downloadBase?.appendingPathComponent(archive.lastPathComponent) ?? archive,
            archiveBytes: bytes, installedBytes: manifest.installedBytes!))
    }
    let catalog = InferenceEngineCatalog(publisherKeys: ["local-development": key.publicKey.rawRepresentation],
        releases: releases, localPackageRoot: downloadBase == nil ? output : nil)
    try catalog.validate()
    let catalogURL = output.appendingPathComponent("engine-catalog.json")
    try encode(catalog, to: catalogURL)
    print(catalogURL.path)
} catch {
    FileHandle.standardError.write(Data("engine packaging failed: \(error)\n".utf8))
    exit(1)
}
