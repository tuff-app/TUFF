import Darwin
import Foundation
import NIOCore
import NIOHTTP1
import Testing
@testable import TUFFEngine
@testable import TUFFServerCore

private actor AnsweringBackend: ServerInferenceBackend {
    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("ok"))
        return ServerCompletion(
            content: "ok", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

// Counts unloadIfIdle calls that got past the host check, so a test can verify that a
// refused request does not trigger an unload.
private final class UnloadCount: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func add() { lock.withLock { count += 1 } }
    var value: Int { lock.withLock { count } }
}

// A web page can point its own domain at 127.0.0.1 (DNS rebinding), and the
// browser then sends that domain as Host and Origin. Only this Mac's own
// loopback names are served.
@Suite("Server host check", .serialized)
struct ServerHostCheckTests {
    private static let chatBody =
        #"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#
    private static let messagesBody =
        #"{"model":"test-model","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}"#
    private static let responsesBody = #"{"model":"test-model","input":"hi","store":false}"#
    private static let token = String(repeating: "a", count: 64)

    private static func body(for path: String) -> String {
        switch path {
        case "/v1/messages": messagesBody
        case "/v1/responses": responsesBody
        default: chatBody
        }
    }

    private static let apiKey = String(repeating: "k", count: 64)

    private static func makeServer(root: URL, unloads: UnloadCount,
                                   apiKey: String? = nil) -> TUFFHTTPServer {
        let provider = FixedServerModelProvider(
            modelID: "test-model", backend: AnsweringBackend(), dialect: .gemma,
            visionCapability: "ready", queueLimit: 1)
        let control = ServerControl(
            token: token,
            status: {
                ServerControlStatus(
                    version: "test", mode: "test", residentModel: nil, transition: nil,
                    activeRequests: 0, queuedRequests: 0, idleUnloadInSeconds: nil,
                    defaultModel: "test-model", unloadDelaySeconds: 0)
            },
            unloadIfIdle: {
                unloads.add()
                return true
            })
        return TUFFHTTPServer(provider: provider, control: control, apiKey: apiKey,
                              attachmentRoot: root)
    }

    private static func makeRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("host-check-\(UUID().uuidString)", isDirectory: true)
    }

    private static func entryCount(_ root: URL) -> Int {
        FileManager.default.enumerator(atPath: root.path)?.allObjects.count ?? 0
    }

    private static func stagedFileCount(_ root: URL) -> Int {
        guard let walker = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey]) else { return 0 }
        var count = 0
        for case let url as URL in walker
        where (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true {
            count += 1
        }
        return count
    }

    private static func request(method: String, path: String, headers: [String],
                                body: String = "", version: String = "HTTP/1.1",
                                connection: String = "close") -> String {
        var text = "\(method) \(path) \(version)\r\n"
        for line in headers { text += line + "\r\n" }
        if method == "POST" {
            text += "Content-Type: application/json\r\n"
                + "Content-Length: \(body.utf8.count)\r\n"
        }
        return text + "Connection: \(connection)\r\n\r\n" + body
    }

    private static func readResponse(_ socket: Int32) throws -> String {
        try readUntil(socket: socket, timeoutMilliseconds: 4_000) {
            guard let split = $0.range(of: "\r\n\r\n") else { return false }
            let body = $0[split.upperBound...]
            let head = $0[..<split.lowerBound].lowercased()
            if let line = head.split(separator: "\r\n")
                .first(where: { $0.hasPrefix("content-length:") }),
               let length = Int(line.dropFirst("content-length:".count)
                .trimmingCharacters(in: .whitespaces)) {
                return body.utf8.count >= length
            }
            return body.hasSuffix("}")
        }
    }

    private static func send(port: Int, method: String, path: String,
                             headers: [String], version: String = "HTTP/1.1",
                             address: String = "127.0.0.1") throws -> String {
        let socket = try connectedSocket(port: port, host: address)
        defer { _ = Darwin.close(socket) }
        try writeAll(socket: socket, text: request(
            method: method, path: path, headers: headers,
            body: method == "POST" ? body(for: path) : "", version: version))
        return try readResponse(socket)
    }

    private static let routes = [
        ("POST", "/v1/chat/completions", [String]()),
        ("POST", "/v1/messages", []),
        ("POST", "/v1/responses", []),
        ("GET", "/v1/models", []),
        ("GET", "/health", []),
        ("GET", ServerControl.statusPath, []),
        ("POST", ServerControl.unloadPath, ["Authorization: Bearer \(token)"]),
    ]

    @Test func rebindingHostsAndForeignOriginsAreRefused() async throws {
        let root = Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let unloads = UnloadCount()
        let server = Self.makeServer(root: root, unloads: unloads)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let local = "Host: 127.0.0.1:\(port)"

        let refused: [[String]] = [
            ["Host: attacker.example:\(port)"],
            ["Host: attacker.example"],
            ["Host: 127.0.0.1.attacker.example:\(port)"],
            ["Host: localhost.attacker.example"],
            ["Host: x.localhost:\(port)"],
            ["Host: 0.0.0.0:\(port)"],
            ["Host: 192.168.1.20:\(port)"],
            ["Host: "],
            ["Host: 127.0.0.1:abc"],
            [local, "Origin: http://attacker.example"],
            [local, "Origin: null"],
            [local, "Origin: http://localhost.attacker.example:3000"],
            [local, "Origin: http://localhost@attacker.example"],
            [local, "Origin: http://127.0.0.1:80@attacker.example"],
            [local, "Origin: http://localhost:3000", "Origin: http://attacker.example"],
        ]
        for headers in refused {
            for (method, path, extra) in Self.routes {
                let response = try Self.send(port: port, method: method, path: path,
                                             headers: headers + extra)
                #expect(response.hasPrefix("HTTP/1.1 403"), "\(headers) \(path)")
                if path == "/v1/messages" {
                    #expect(response.contains(#""type":"permission_error""#),
                            "\(headers) \(path)")
                    #expect(response.contains("is not a loopback name"),
                            "\(headers) \(path)")
                } else {
                    #expect(response.contains("forbidden_host"), "\(headers) \(path)")
                }
                #expect(!response.contains("test-model"), "\(headers) \(path)")
            }
        }
        #expect(unloads.value == 0)
        try await server.shutdown()
    }

    // Two Host lines must reach the handler and be refused there. If the
    // HTTP parser ever rejects them first, this fails with a 400 and shows it.
    @Test func aSecondHostLineIsRefused() async throws {
        let root = Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let server = Self.makeServer(root: root, unloads: UnloadCount())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        for hosts in [["Host: 127.0.0.1:\(port)", "Host: attacker.example"],
                      ["Host: attacker.example", "Host: 127.0.0.1:\(port)"]] {
            let response = try Self.send(port: port, method: "POST",
                                         path: "/v1/chat/completions", headers: hosts)
            #expect(response.hasPrefix("HTTP/1.1 403"), "\(response.prefix(40))")
            #expect(response.contains("forbidden_host"))
        }
        try await server.shutdown()
    }

    @Test func loopbackNamesAndOriginsAreServed() async throws {
        let root = Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let unloads = UnloadCount()
        let server = Self.makeServer(root: root, unloads: unloads)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let accepted: [[String]] = [
            ["Host: 127.0.0.1:\(port)"],
            ["Host: 127.0.0.1"],
            ["Host: localhost:\(port)"],
            ["Host: LocalHost."],
            ["Host: [::1]:\(port)"],
            ["Host: 127.0.0.1:\(port)", "Origin: http://localhost:3000"],
            ["Host: 127.0.0.1:\(port)", "Origin: https://127.0.0.1"],
        ]
        for headers in accepted {
            let response = try Self.send(port: port, method: "POST",
                                         path: "/v1/chat/completions", headers: headers)
            #expect(response.hasPrefix("HTTP/1.1 200"), "\(headers)")
            #expect(response.contains(#""content":"ok""#), "\(headers)")
        }
        for path in ["/v1/messages", "/v1/responses"] {
            for headers in [["Host: 127.0.0.1:\(port)"],
                            ["Host: [::1]:\(port)", "Origin: http://localhost:3000"]] {
                let response = try Self.send(port: port, method: "POST", path: path,
                                             headers: headers)
                #expect(response.hasPrefix("HTTP/1.1 200"), "\(headers) \(path)")
                #expect(response.contains(#""text":"ok""#), "\(headers) \(path)")
            }
        }
        for path in ["/v1/models", "/health"] {
            let response = try Self.send(
                port: port, method: "GET", path: path,
                headers: ["Host: 127.0.0.1:\(port)", "Origin: http://localhost:3000"])
            #expect(response.hasPrefix("HTTP/1.1 200"), "\(path)")
        }
        let status = try Self.send(port: port, method: "GET", path: ServerControl.statusPath,
                                   headers: ["Host: localhost:\(port)"])
        #expect(status.hasPrefix("HTTP/1.1 200"))
        let unload = try Self.send(port: port, method: "POST", path: ServerControl.unloadPath,
                                   headers: ["Host: 127.0.0.1:\(port)",
                                             "Authorization: Bearer \(Self.token)"])
        #expect(unload.hasPrefix("HTTP/1.1 200"))
        #expect(unloads.value == 1)

        let old = try Self.send(port: port, method: "POST", path: "/v1/chat/completions",
                                headers: [], version: "HTTP/1.0")
        #expect(old.hasPrefix("HTTP/1.1 200") || old.hasPrefix("HTTP/1.0 200"),
                "\(old.prefix(40))")
        #expect(old.contains(#""content":"ok""#))

        for headers in [["Host: attacker.example"], ["Origin: http://attacker.example"]] {
            let refused = try Self.send(port: port, method: "POST",
                                        path: "/v1/chat/completions",
                                        headers: headers, version: "HTTP/1.0")
            #expect(refused.contains(" 403 "), "\(headers) \(refused.prefix(40))")
            #expect(refused.contains("forbidden_host"), "\(headers)")
        }
        try await server.shutdown()
    }

    // A request that is refused does not stage any files, and the connection remains usable
    // for a later request.
    @Test func aRefusalStagesNothingAndKeepsTheConnectionUsable() async throws {
        let root = Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let server = Self.makeServer(root: root, unloads: UnloadCount())
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let socket = try connectedSocket(port: port)
        defer { _ = Darwin.close(socket) }

        let pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        let image = #"{"model":"test-model","messages":[{"role":"user","content":["#
            + #"{"type":"image_url","image_url":{"url":"data:image/png;base64,\#(pixel)"}}]}]}"#
        // All but the last byte, so the image has arrived in full while the request is
        // still open: a server that staged it would have written it by now.
        let text = Self.request(
            method: "POST", path: "/v1/chat/completions",
            headers: ["Host: attacker.example:\(port)"], body: image,
            connection: "keep-alive")
        try writeAll(socket: socket, text: String(text.dropLast()))
        try await Task.sleep(for: .milliseconds(200))
        #expect(Self.stagedFileCount(root) == 0)
        try writeAll(socket: socket, text: String(text.suffix(1)))
        let refused = try Self.readResponse(socket)
        #expect(refused.hasPrefix("HTTP/1.1 403"))
        #expect(Self.stagedFileCount(root) == 0)
        #expect(Self.entryCount(root) == 0)
        #expect(!FileManager.default.fileExists(atPath: root.path))

        try writeAll(socket: socket, text: Self.request(
            method: "POST", path: "/v1/chat/completions",
            headers: ["Host: 127.0.0.1:\(port)"], body: Self.chatBody,
            connection: "keep-alive"))
        let served = try Self.readResponse(socket)
        #expect(served.hasPrefix("HTTP/1.1 200"))
        #expect(served.contains(#""content":"ok""#))
        try await server.shutdown()
    }

    // This Mac's own IPv4 address on a network interface. A connection to it
    // arrives from that address, not loopback, as another device's would.
    private static func networkAddress() -> String? {
        var first: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&first) == 0, let first else { return nil }
        defer { freeifaddrs(first) }
        for entry in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let flags = Int32(entry.pointee.ifa_flags)
            guard let address = entry.pointee.ifa_addr,
                  address.pointee.sa_family == sa_family_t(AF_INET),
                  flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0 else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard getnameinfo(address, socklen_t(address.pointee.sa_len), &host,
                              socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 else { continue }
            return String(cString: host)
        }
        return nil
    }

    @Test func callersOffThisMacNeedTheKey() {
        let head = { (headers: [(String, String)]) in
            HTTPRequestHead(version: .http1_1, method: .GET, uri: "/v1/models",
                            headers: HTTPHeaders(headers))
        }
        let lan = [("Host", "192.168.1.20:8080")]
        #expect(ServerHTTPHandler.refusal(head(lan), fromLoopback: false, apiKey: nil)?
            .envelope.error.code == "invalid_api_key")
        #expect(ServerHTTPHandler.refusal(head(lan), fromLoopback: false, apiKey: Self.apiKey)?
            .envelope.error.code == "invalid_api_key")
        #expect(ServerHTTPHandler.refusal(
            head(lan + [("Authorization", "Bearer \(Self.apiKey)x")]),
            fromLoopback: false, apiKey: Self.apiKey) != nil)
        #expect(ServerHTTPHandler.refusal(
            head(lan + [("Authorization", "Bearer \(Self.apiKey)")]),
            fromLoopback: false, apiKey: nil) != nil)
        for credential in [("Authorization", "Bearer \(Self.apiKey)"), ("x-api-key", Self.apiKey)] {
            #expect(ServerHTTPHandler.refusal(
                head(lan + [credential, ("Origin", "http://webui.example")]),
                fromLoopback: false, apiKey: Self.apiKey) == nil)
            // A rebound name on this Mac is served too, since a browser
            // cannot attach the key to a request it was tricked into making.
            #expect(ServerHTTPHandler.refusal(
                head([("Host", "attacker.example"), credential]),
                fromLoopback: true, apiKey: Self.apiKey) == nil)
        }
        #expect(ServerHTTPHandler.refusal(head([("Host", "localhost")]),
                                          fromLoopback: true, apiKey: Self.apiKey) == nil)
        #expect(ServerHTTPHandler.refusal(head([("Host", "attacker.example")]),
                                          fromLoopback: true, apiKey: Self.apiKey)?
            .envelope.error.code == "forbidden_host")
    }

    @Test func loopbackPeers() throws {
        #expect(ServerHTTPHandler.isLoopback(peer: try SocketAddress(ipAddress: "127.0.0.1", port: 1)))
        #expect(ServerHTTPHandler.isLoopback(peer: try SocketAddress(ipAddress: "127.3.0.1", port: 1)))
        #expect(ServerHTTPHandler.isLoopback(peer: try SocketAddress(ipAddress: "::1", port: 1)))
        #expect(ServerHTTPHandler.isLoopback(peer: try SocketAddress(ipAddress: "::ffff:127.0.0.1", port: 1)))
        for address in ["192.168.1.20", "10.0.0.2", "::", "fe80::1", "::ffff:10.0.0.1"] {
            #expect(!ServerHTTPHandler.isLoopback(peer: try SocketAddress(ipAddress: address, port: 1)),
                    "\(address)")
        }
        #expect(!ServerHTTPHandler.isLoopback(peer: nil))
    }

    @Test(.enabled(if: networkAddress() != nil, "needs a network interface"))
    func aServerOnEveryInterfaceServesTheNetworkWithTheKey() async throws {
        let address = try #require(Self.networkAddress())
        let root = Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let unloads = UnloadCount()
        let server = Self.makeServer(root: root, unloads: unloads, apiKey: Self.apiKey)
        let channel = try await server.start(host: "0.0.0.0", port: 0)
        let port = try #require(channel.localAddress?.port)
        let lanHost = "Host: \(address):\(port)"

        for headers in [[lanHost], [lanHost, "Authorization: Bearer \(String(repeating: "x", count: 64))"],
                        [lanHost, "x-api-key: wrong"], ["Host: 127.0.0.1:\(port)"]] {
            for (method, path, extra) in Self.routes {
                let response = try Self.send(port: port, method: method, path: path,
                                             headers: headers + extra, address: address)
                #expect(response.hasPrefix("HTTP/1.1 401"), "\(headers) \(path)")
                if path == "/v1/messages" {
                    #expect(response.contains(#""type":"authentication_error""#), "\(headers)")
                } else {
                    #expect(response.contains("invalid_api_key"), "\(headers) \(path)")
                }
                #expect(!response.contains("test-model"), "\(headers) \(path)")
            }
        }
        #expect(unloads.value == 0)

        let bearer = try Self.send(
            port: port, method: "POST", path: "/v1/chat/completions",
            headers: [lanHost, "Origin: http://webui.example", "Authorization: Bearer \(Self.apiKey)"],
            address: address)
        #expect(bearer.hasPrefix("HTTP/1.1 200"))
        #expect(bearer.contains(#""content":"ok""#))
        let anthropic = try Self.send(port: port, method: "POST", path: "/v1/messages",
                                      headers: [lanHost, "x-api-key: \(Self.apiKey)"],
                                      address: address)
        #expect(anthropic.hasPrefix("HTTP/1.1 200"))
        let models = try Self.send(port: port, method: "GET", path: "/v1/models",
                                   headers: ["Host: tuff-mac.local:\(port)",
                                             "Authorization: Bearer \(Self.apiKey)"],
                                   address: address)
        #expect(models.hasPrefix("HTTP/1.1 200") && models.contains("test-model"))

        // This Mac still needs no key, and is still protected from rebinding.
        let local = try Self.send(port: port, method: "POST", path: "/v1/chat/completions",
                                  headers: ["Host: 127.0.0.1:\(port)"])
        #expect(local.hasPrefix("HTTP/1.1 200"))
        let rebound = try Self.send(port: port, method: "POST", path: "/v1/chat/completions",
                                    headers: ["Host: attacker.example"])
        #expect(rebound.hasPrefix("HTTP/1.1 403"))
        try await server.shutdown()
    }

    // Without a key nothing off this Mac is served, whatever the address.
    @Test(.enabled(if: networkAddress() != nil, "needs a network interface"))
    func aServerWithoutAKeyRefusesTheNetwork() async throws {
        let address = try #require(Self.networkAddress())
        let root = Self.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let server = Self.makeServer(root: root, unloads: UnloadCount())
        let channel = try await server.start(host: "0.0.0.0", port: 0)
        let port = try #require(channel.localAddress?.port)
        let response = try Self.send(port: port, method: "POST", path: "/v1/chat/completions",
                                     headers: ["Host: \(address):\(port)",
                                               "Authorization: Bearer \(Self.apiKey)"],
                                     address: address)
        #expect(response.hasPrefix("HTTP/1.1 401"))
        try await server.shutdown()
    }
}
