import Foundation
import NIOHTTP1
import Security

/// Status the background server reports at `GET /tuff/v1/status`.
public struct ServerControlStatus: Codable, Equatable, Sendable {
    public var version: String
    public var mode: String
    public var residentModel: String?
    public var transition: String?
    public var activeRequests: Int
    public var queuedRequests: Int
    /// Seconds until an idle model unloads, when a timer is running.
    public var idleUnloadInSeconds: Int?
    public var defaultModel: String
    public var unloadDelaySeconds: Int

    public init(version: String, mode: String, residentModel: String?, transition: String?,
                activeRequests: Int, queuedRequests: Int, idleUnloadInSeconds: Int?,
                defaultModel: String, unloadDelaySeconds: Int) {
        self.version = version
        self.mode = mode
        self.residentModel = residentModel
        self.transition = transition
        self.activeRequests = activeRequests
        self.queuedRequests = queuedRequests
        self.idleUnloadInSeconds = idleUnloadInSeconds
        self.defaultModel = defaultModel
        self.unloadDelaySeconds = unloadDelaySeconds
    }

    enum CodingKeys: String, CodingKey {
        case version, mode, transition
        case residentModel = "resident_model"
        case activeRequests = "active_requests"
        case queuedRequests = "queued_requests"
        case idleUnloadInSeconds = "idle_unload_in_seconds"
        case defaultModel = "default_model"
        case unloadDelaySeconds = "unload_delay_seconds"
    }
}

/// The loopback control routes of a server that manages its own models.
///
/// `POST /tuff/v1/unload` lets the TUFF app reclaim memory from an idle
/// background server before it loads a model itself. It requires the bearer
/// token stored in a file only this user can read, and it never interrupts a
/// running or queued request: a busy server answers 409.
public struct ServerControl: Sendable {
    public static let statusPath = "/tuff/v1/status"
    public static let unloadPath = "/tuff/v1/unload"

    public let token: String
    public let status: @Sendable () async -> ServerControlStatus
    public let unloadIfIdle: @Sendable () async -> Bool

    public init(token: String,
                status: @escaping @Sendable () async -> ServerControlStatus,
                unloadIfIdle: @escaping @Sendable () async -> Bool) {
        self.token = token
        self.status = status
        self.unloadIfIdle = unloadIfIdle
    }

    public func authorizes(_ header: String?) -> Bool {
        guard let header, header.hasPrefix("Bearer ") else { return false }
        return Self.matches(String(header.dropFirst("Bearer ".count)), token)
    }

    /// Constant time, so a secret cannot be recovered byte by byte.
    static func matches(_ presented: String, _ expected: String) -> Bool {
        let presented = Array(presented.utf8)
        let expected = Array(expected.utf8)
        guard presented.count == expected.count else { return false }
        var difference: UInt8 = 0
        for (lhs, rhs) in zip(presented, expected) { difference |= lhs ^ rhs }
        return difference == 0
    }

    public static func status(provider: HostRoutedModelProvider,
                              version: String) async -> ServerControlStatus {
        await provider.controlStatus(version: version)
    }
}

/// The token file the app and the background server share.
public enum ServerControlToken {
    public static func fileURL(applicationSupport: URL) -> URL {
        applicationSupport.appendingPathComponent("TUFF/Server/control-token", isDirectory: false)
    }

    /// Creates a fresh random token readable only by this user.
    public static func create(at url: URL) throws -> String {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw CocoaError(.fileWriteUnknown)
        }
        let token = bytes.map { String(format: "%02x", $0) }.joined()
        try Data(token.utf8).write(to: url, options: [.atomic])
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        return token
    }

    public static func read(at url: URL) -> String? {
        guard let data = try? Data(contentsOf: url),
              let token = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              token.count == 64 else { return nil }
        return token
    }
}

/// The key callers on other devices send when the server listens beyond
/// loopback. It is created the first time it is needed and kept, so clients
/// stay configured across restarts. Deleting the file makes a new one.
public enum ServerAPIKey {
    public static func fileURL(applicationSupport: URL) -> URL {
        applicationSupport.appendingPathComponent("TUFF/Server/api-key", isDirectory: false)
    }

    public static func readOrCreate(at url: URL) throws -> String {
        try ServerControlToken.read(at: url) ?? ServerControlToken.create(at: url)
    }

    /// `Authorization: Bearer <key>` as OpenAI clients send it, or
    /// `x-api-key: <key>` as Anthropic clients do.
    static func presented(in headers: HTTPHeaders, matches key: String) -> Bool {
        if let header = headers.first(name: "authorization"), header.hasPrefix("Bearer "),
           ServerControl.matches(String(header.dropFirst("Bearer ".count)), key) {
            return true
        }
        if let header = headers.first(name: "x-api-key"), ServerControl.matches(header, key) {
            return true
        }
        return false
    }
}
