import Darwin
import Foundation

/// How long the background server keeps a model in memory after its last
/// request finishes. Requests that are running or queued always keep it.
public enum TUFFModelUnloadDelay: Codable, Equatable, Hashable, Sendable {
    /// Unload as soon as a response finishes and nothing else is queued.
    case immediately
    case seconds(Int)

    public static let choices: [TUFFModelUnloadDelay] = [
        .immediately, .seconds(60), .seconds(300), .seconds(900), .seconds(3_600),
    ]
    public static let `default`: TUFFModelUnloadDelay = .seconds(300)

    public var seconds: Int {
        switch self {
        case .immediately: 0
        case .seconds(let value): max(0, value)
        }
    }

    public var label: String {
        switch self {
        case .immediately: return "Immediately after each response"
        case .seconds(let value):
            if value <= 0 { return "Immediately after each response" }
            if value % 3_600 == 0 {
                let hours = value / 3_600
                return hours == 1 ? "After 1 hour" : "After \(hours) hours"
            }
            if value % 60 == 0 {
                let minutes = value / 60
                return minutes == 1 ? "After 1 minute" : "After \(minutes) minutes"
            }
            return "After \(value) seconds"
        }
    }

    /// Stored as a number of seconds so the file stays readable and a later
    /// build can add choices without a schema change.
    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer().decode(Int.self)
        self = value <= 0 ? .immediately : .seconds(value)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(seconds)
    }
}

/// Settings shared by the app, which writes them, and the background server,
/// which reads them. A newer file this build does not understand is left
/// untouched and the server keeps its defaults, the same rule the app's own
/// settings follow.
public struct TUFFBackgroundServerSettings: Codable, Equatable, Sendable {
    public static let currentVersion = 1
    public static let defaultPort = 8_080
    public static let defaultQueueLimit = 4
    public static let loopbackHost = "127.0.0.1"
    /// Every interface, which still includes loopback.
    public static let networkHost = "0.0.0.0"

    public var version: Int
    public var enabled: Bool
    /// The address the server listens on. Only loopback by default; any
    /// other address requires the API key from callers on other devices.
    public var host: String
    public var port: Int
    /// Catalog selector of the model used when a request names `default`, or
    /// a model this server does not know by its own identifier.
    public var defaultModel: String
    public var unloadDelay: TUFFModelUnloadDelay
    public var queueLimit: Int

    public init(version: Int = currentVersion,
                enabled: Bool = false,
                host: String = loopbackHost,
                port: Int = defaultPort,
                defaultModel: String = TUFFModelCatalog.default.selector,
                unloadDelay: TUFFModelUnloadDelay = .default,
                queueLimit: Int = defaultQueueLimit) {
        self.version = version
        self.enabled = enabled
        self.host = host
        self.port = port
        self.defaultModel = defaultModel
        self.unloadDelay = unloadDelay
        self.queueLimit = queueLimit
    }

    enum CodingKeys: String, CodingKey {
        case version, enabled, host, port, defaultModel, unloadDelay, queueLimit
    }

    /// `host` arrived in 10.2.1 without a version change: an older file has
    /// none and stays on loopback, and an older build ignores it, which also
    /// means loopback.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        enabled = try container.decode(Bool.self, forKey: .enabled)
        host = try container.decodeIfPresent(String.self, forKey: .host) ?? Self.loopbackHost
        port = try container.decode(Int.self, forKey: .port)
        defaultModel = try container.decode(String.self, forKey: .defaultModel)
        unloadDelay = try container.decode(TUFFModelUnloadDelay.self, forKey: .unloadDelay)
        queueLimit = try container.decode(Int.self, forKey: .queueLimit)
    }

    public var isValid: Bool {
        (1...65_535).contains(port) && (1...16).contains(queueLimit)
            && !defaultModel.isEmpty && Self.isAddress(host)
    }

    public var listensBeyondLoopback: Bool { !Self.isLoopback(host) }

    /// A numeric IPv4 or IPv6 address. Names are not accepted, so the bound
    /// address never depends on DNS.
    public static func isAddress(_ host: String) -> Bool {
        var v4 = in_addr()
        var v6 = in6_addr()
        return inet_pton(AF_INET, host, &v4) == 1 || inet_pton(AF_INET6, host, &v6) == 1
    }

    public static func isLoopback(_ host: String) -> Bool {
        var v4 = in_addr()
        if inet_pton(AF_INET, host, &v4) == 1 {
            return UInt32(bigEndian: v4.s_addr) >> 24 == 127
        }
        var v6 = in6_addr()
        return inet_pton(AF_INET6, host, &v6) == 1
            && withUnsafeBytes(of: &v6) { $0.dropLast().allSatisfy { $0 == 0 } && $0.last == 1 }
    }
}

public enum TUFFBackgroundServerSettingsStore {
    public enum SaveError: Error, Equatable, Sendable {
        case invalidSettings
        /// The file on disk was written by a newer TUFF.
        case newerVersionOnDisk(Int)
    }

    public enum LoadResult: Equatable, Sendable {
        case missing
        case loaded(TUFFBackgroundServerSettings)
        /// Written by a newer TUFF. Leave it alone.
        case newer(version: Int)
        case unreadable
    }

    public static func directory(applicationSupport: URL) -> URL {
        applicationSupport.appendingPathComponent("TUFF/Server", isDirectory: true)
    }

    public static func fileURL(applicationSupport: URL) -> URL {
        directory(applicationSupport: applicationSupport)
            .appendingPathComponent("background-server.json", isDirectory: false)
    }

    public static func load(from url: URL) -> LoadResult {
        guard let data = try? Data(contentsOf: url) else {
            return FileManager.default.fileExists(atPath: url.path) ? .unreadable : .missing
        }
        struct Stamp: Decodable { let version: Int }
        guard let stamp = try? JSONDecoder().decode(Stamp.self, from: data) else {
            return .unreadable
        }
        guard stamp.version <= TUFFBackgroundServerSettings.currentVersion else {
            return .newer(version: stamp.version)
        }
        guard let settings = try? JSONDecoder().decode(
                  TUFFBackgroundServerSettings.self, from: data),
              settings.isValid else { return .unreadable }
        return .loaded(settings)
    }

    /// Refuses to overwrite a file written by a newer build.
    public static func save(_ settings: TUFFBackgroundServerSettings, to url: URL) throws {
        guard settings.isValid else { throw SaveError.invalidSettings }
        if case .newer(let version) = load(from: url) {
            throw SaveError.newerVersionOnDisk(version)
        }
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(settings)
        data.append(0x0A)
        try data.write(to: url, options: .atomic)
    }
}
