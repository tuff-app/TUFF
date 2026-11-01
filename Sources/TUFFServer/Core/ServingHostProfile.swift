import CryptoKit
import Darwin
import Foundation
import Security
import TUFFModelCatalog

/// Stable identity and private filesystem paths for one installed TUFF
/// Application Support root. Local preview builds naturally get separate hosts.
public struct ServingHostProfile: Sendable, Equatable {
    public let applicationSupportURL: URL
    public let profileIdentity: String
    public let serverDirectoryURL: URL
    public let socketURL: URL
    public let lockURL: URL
    public let tokenURL: URL
    public let clientIDURL: URL
    public let clientDirectoryURL: URL
    public let identityURL: URL
    /// The identifier the profile was derived from; a host launched for this
    /// profile must be given the same one.
    public let bundleIdentifier: String

    public init(applicationSupportURL: URL, bundleIdentifier: String) throws {
        let support = applicationSupportURL.resolvingSymlinksInPath().standardizedFileURL
        guard support.isFileURL, !bundleIdentifier.isEmpty else {
            throw ServingHostProfileError.invalidProfile
        }
        self.applicationSupportURL = support
        self.bundleIdentifier = bundleIdentifier
        let material = "\(getuid())\n\(bundleIdentifier)\n\(support.path)"
        let digest = SHA256.hash(data: Data(material.utf8))
            .map { String(format: "%02x", $0) }.joined()
        profileIdentity = digest
        let base = support.appendingPathComponent("TUFF/Server", isDirectory: true)
        serverDirectoryURL = base
        lockURL = base.appendingPathComponent("serving-host.lock")
        tokenURL = base.appendingPathComponent("serving-host-token")
        clientIDURL = base.appendingPathComponent("serving-client-id")
        clientDirectoryURL = base.appendingPathComponent("client-identities", isDirectory: true)
        identityURL = base.appendingPathComponent("serving-profile-identity")

        // macOS limits sun_path to a little over 100 bytes. Keep the live socket
        // in a private temporary directory named by a digest of this profile;
        // persistent ownership and identity remain in Application Support.
        let shortID = String(digest.prefix(24))
        let socketDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-serving-\(getuid())-\(shortID)", isDirectory: true)
        socketURL = socketDirectory.appendingPathComponent("s")
        guard socketURL.path.utf8.count < MemoryLayout<sockaddr_un>.size else {
            throw ServingHostProfileError.socketPathTooLong
        }
    }

    /// The profile of the Application Support root this process runs against.
    /// The bundle identifier comes from, in order: the caller, the environment,
    /// the main bundle, or the `.app` that contains this executable. A local
    /// preview therefore never shares a host with another installation.
    public static func forCurrentProcess(
        bundleIdentifier: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ServingHostProfile {
        let identifier = bundleIdentifier
            ?? environment["TUFF_SERVING_BUNDLE_ID"]
            ?? Bundle.main.bundleIdentifier
            ?? enclosingAppBundleIdentifier()
            ?? "tuff.unbundled"
        return try ServingHostProfile(
            applicationSupportURL: TUFFApplicationDirectories.applicationSupportURL(
                environment: environment),
            bundleIdentifier: identifier)
    }

    private static func enclosingAppBundleIdentifier() -> String? {
        var url = (Bundle.main.executableURL
            ?? URL(fileURLWithPath: CommandLine.arguments.first ?? "")).resolvingSymlinksInPath()
        while url.path != "/" {
            if url.pathExtension == "app" { return Bundle(url: url)?.bundleIdentifier }
            url.deleteLastPathComponent()
        }
        return nil
    }

    /// Creates or verifies the private profile directory and its identity file.
    /// An identity mismatch or a symlink substitution fails closed.
    public func prepare() throws {
        try Self.makePrivateDirectory(serverDirectoryURL)
        try Self.verifyOwnedDirectory(serverDirectoryURL.deletingLastPathComponent())
        try Self.verifyPrivateDirectory(serverDirectoryURL)
        try Self.makePrivateDirectory(clientDirectoryURL)
        try Self.verifyPrivateDirectory(clientDirectoryURL)
        if let stored = try Self.readPrivateFile(identityURL) {
            guard stored == profileIdentity else {
                throw ServingHostProfileError.profileIdentityMismatch
            }
        } else {
            try Self.createPrivateFile(identityURL, contents: profileIdentity)
            guard try Self.readPrivateFile(identityURL) == profileIdentity else {
                throw ServingHostProfileError.profileIdentityMismatch
            }
        }
        let socketDirectory = socketURL.deletingLastPathComponent()
        try Self.makePrivateDirectory(socketDirectory)
        try Self.verifyPrivateDirectory(socketDirectory)
    }

    /// Returns the stable ID used by this app client across host restarts.
    public func clientID() throws -> UUID {
        try clientID(for: "gui")
    }

    /// Stable identity per client surface in this profile. Distinct clients
    /// can attach concurrently, while reconnects retain their session namespace.
    public func clientID(for clientName: String) throws -> UUID {
        try prepare()
        guard !clientName.isEmpty, clientName.utf8.count <= 32,
              clientName.utf8.allSatisfy({
                  (48...57).contains($0) || (97...122).contains($0) || $0 == 45
              }) else { throw ServingHostProfileError.invalidClientName }
        let url = clientName == "gui" ? clientIDURL
            : clientDirectoryURL.appendingPathComponent("\(clientName).id")
        if let text = try Self.readPrivateFile(url), let id = UUID(uuidString: text) {
            return id
        }
        let id = UUID()
        try Self.createPrivateFile(url, contents: id.uuidString)
        guard let saved = try Self.readPrivateFile(url),
              let resolved = UUID(uuidString: saved) else {
            throw ServingHostProfileError.unsafeFile(url.lastPathComponent)
        }
        return resolved
    }

    /// Reads or creates the 256-bit authentication token without following a
    /// symlink and verifies owner, file type, mode and link count.
    public func token() throws -> String {
        try prepare()
        if let existing = try Self.readPrivateFile(tokenURL) {
            guard existing.utf8.count == 64,
                  existing.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
            else { throw ServingHostProfileError.invalidToken }
            return existing
        }
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw ServingHostProfileError.randomUnavailable
        }
        let value = bytes.map { String(format: "%02x", $0) }.joined()
        try Self.createPrivateFile(tokenURL, contents: value)
        guard let saved = try Self.readPrivateFile(tokenURL), saved == value
                || (saved.utf8.count == 64
                    && saved.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }))
        else { throw ServingHostProfileError.invalidToken }
        return saved
    }

    public func acquireHostLease() throws -> ServingHostLease? {
        try prepare()
        let fd = lockURL.path.withCString {
            open($0, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, mode_t(0o600))
        }
        guard fd >= 0 else { throw ServingHostProfileError.systemCall("open host lock", errno) }
        var info = stat()
        guard fstat(fd, &info) == 0,
              (info.st_mode & S_IFMT) == S_IFREG,
              info.st_uid == getuid(), info.st_nlink == 1,
              (info.st_mode & 0o077) == 0 else {
            close(fd)
            throw ServingHostProfileError.unsafeFile("serving-host.lock")
        }
        if flock(fd, LOCK_EX | LOCK_NB) == 0 { return ServingHostLease(fd: fd) }
        let code = errno
        close(fd)
        if code == EWOULDBLOCK || code == EAGAIN { return nil }
        throw ServingHostProfileError.systemCall("lock host", code)
    }

    /// Creates `url` (and any missing parents) so that it is never visible with
    /// looser permissions: concurrent processes preparing one profile must not
    /// observe a half-configured directory. Existing directories are verified
    /// by the caller.
    fileprivate static func makePrivateDirectory(_ url: URL) throws {
        let parent = url.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: parent.path) {
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        }
        if url.path.withCString({ mkdir($0, 0o700) }) != 0, errno != EEXIST {
            throw ServingHostProfileError.systemCall("create \(url.lastPathComponent)", errno)
        }
    }

    /// TUFF 9.x created `TUFF/Server` with the default 0755 for its control
    /// token and Background API settings, so an updated profile already has
    /// it. A real directory this user owns is tightened to 0700 in place
    /// (through a descriptor, so a swapped-in symlink is never followed);
    /// anything else is still refused.
    fileprivate static func verifyPrivateDirectory(_ url: URL) throws {
        var info = stat()
        guard url.path.withCString({ lstat($0, &info) }) == 0,
              (info.st_mode & S_IFMT) == S_IFDIR,
              info.st_uid == getuid() else {
            throw ServingHostProfileError.unsafeDirectory(url.lastPathComponent)
        }
        if (info.st_mode & 0o077) == 0 { return }
        let fd = url.path.withCString { open($0, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC) }
        guard fd >= 0 else { throw ServingHostProfileError.unsafeDirectory(url.lastPathComponent) }
        defer { close(fd) }
        var opened = stat()
        guard fstat(fd, &opened) == 0,
              opened.st_dev == info.st_dev, opened.st_ino == info.st_ino,
              opened.st_uid == getuid(), fchmod(fd, 0o700) == 0,
              fstat(fd, &opened) == 0, (opened.st_mode & 0o077) == 0 else {
            throw ServingHostProfileError.unsafeDirectory(url.lastPathComponent)
        }
    }

    private static func verifyOwnedDirectory(_ url: URL) throws {
        var info = stat()
        guard url.path.withCString({ lstat($0, &info) }) == 0,
              (info.st_mode & S_IFMT) == S_IFDIR,
              info.st_uid == getuid() else {
            throw ServingHostProfileError.unsafeDirectory(url.lastPathComponent)
        }
    }

    fileprivate static func readPrivateFile(_ url: URL) throws -> String? {
        let fd = url.path.withCString { open($0, O_RDONLY | O_CLOEXEC | O_NOFOLLOW) }
        if fd < 0 {
            if errno == ENOENT { return nil }
            throw ServingHostProfileError.systemCall("open \(url.lastPathComponent)", errno)
        }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0,
              (info.st_mode & S_IFMT) == S_IFREG,
              info.st_uid == getuid(), info.st_nlink == 1,
              (info.st_mode & 0o077) == 0,
              info.st_size <= 4096 else {
            throw ServingHostProfileError.unsafeFile(url.lastPathComponent)
        }
        var data = Data(count: Int(info.st_size))
        var offset = 0
        while offset < data.count {
            let count = data.withUnsafeMutableBytes { buffer in
                Darwin.read(fd, buffer.baseAddress!.advanced(by: offset), buffer.count - offset)
            }
            if count < 0, errno == EINTR { continue }
            guard count > 0 else { throw ServingHostProfileError.unsafeFile(url.lastPathComponent) }
            offset += count
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw ServingHostProfileError.unsafeFile(url.lastPathComponent)
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Publishes `contents` atomically: the file appears complete or not at
    /// all, so a process racing to read the profile never sees an empty or
    /// half-written identity, token or client ID. An existing file wins.
    fileprivate static func createPrivateFile(_ url: URL, contents: String) throws {
        let temporary = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).\(getpid()).\(UInt32.random(in: 0...UInt32.max))")
        let fd = temporary.path.withCString {
            open($0, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, mode_t(0o600))
        }
        guard fd >= 0 else {
            throw ServingHostProfileError.systemCall("create \(url.lastPathComponent)", errno)
        }
        var published = false
        defer {
            close(fd)
            if !published { _ = temporary.path.withCString { unlink($0) } }
        }
        let bytes = Array(contents.utf8)
        var offset = 0
        while offset < bytes.count {
            let count = bytes.withUnsafeBytes {
                Darwin.write(fd, $0.baseAddress!.advanced(by: offset), bytes.count - offset)
            }
            if count < 0, errno == EINTR { continue }
            guard count > 0 else {
                throw ServingHostProfileError.systemCall("write \(url.lastPathComponent)", errno)
            }
            offset += count
        }
        guard fsync(fd) == 0 else {
            throw ServingHostProfileError.systemCall("sync \(url.lastPathComponent)", errno)
        }
        let result = temporary.path.withCString { source in
            url.path.withCString { renamex_np(source, $0, UInt32(RENAME_EXCL)) }
        }
        if result == 0 { published = true; return }
        if errno == EEXIST { return }
        throw ServingHostProfileError.systemCall("publish \(url.lastPathComponent)", errno)
    }
}

public final class ServingHostLease: @unchecked Sendable {
    private let lock = NSLock()
    private var descriptor: Int32?

    fileprivate init(fd: Int32) { descriptor = fd }

    public func release() {
        lock.lock()
        let fd = descriptor
        descriptor = nil
        lock.unlock()
        if let fd { close(fd) }
    }

    deinit { release() }
}

public enum ServingHostProfileError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalidProfile
    case socketPathTooLong
    case profileIdentityMismatch
    case invalidToken
    case invalidClientName
    case randomUnavailable
    case unsafeDirectory(String)
    case unsafeFile(String)
    case systemCall(String, Int32)

    public var description: String {
        switch self {
        case .invalidProfile: "Invalid TUFF serving profile"
        case .socketPathTooLong: "TUFF private socket path exceeds the macOS Unix socket limit"
        case .profileIdentityMismatch: "TUFF serving profile identity does not match this support root"
        case .invalidToken: "TUFF serving authentication token is invalid"
        case .invalidClientName: "TUFF serving client name is invalid"
        case .randomUnavailable: "Could not create a TUFF serving authentication token"
        case .unsafeDirectory(let name): "TUFF serving directory is unsafe: \(name)"
        case .unsafeFile(let name): "TUFF serving file is unsafe: \(name)"
        case .systemCall(let operation, let code): "\(operation) failed (errno \(code))"
        }
    }
}
