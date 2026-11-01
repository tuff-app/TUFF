import Darwin
import Foundation
import Testing
import TUFFServerCore

/// Exercises the real `TUFFServer --private-host` process (no model, no GPU).
/// Guard-returns unless `TUFF_SERVING_HOST_EXECUTABLE` names the built binary.
@Suite(.serialized) struct ServingHostLaunchTests {
    private final class Collected: @unchecked Sendable {
        private let lock = NSLock()
        private var hosts: [UUID] = []
        private var pids: [pid_t] = []
        func add(host: UUID, pid: pid_t?) {
            lock.withLock { hosts.append(host); if let pid { pids.append(pid) } }
        }
        var snapshot: (hosts: [UUID], pids: [pid_t]) { lock.withLock { (hosts, pids) } }
    }

    @Test(.timeLimit(.minutes(3)))
    func racingClientsShareOneIndependentHostThatIdlesOutAfterTheLastClientLeaves() async throws {
        guard let path = ProcessInfo.processInfo.environment["TUFF_SERVING_HOST_EXECUTABLE"] else { return }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-host-launch-\(UUID().uuidString.prefix(8))", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: root) }
        setenv("TUFF_APPLICATION_SUPPORT_ROOT", root.path, 1)
        let bundle = "test.tuff.launch-race"
        setenv("TUFF_SERVING_BUNDLE_ID", bundle, 1)
        defer { unsetenv("TUFF_SERVING_BUNDLE_ID") }
        let profile = try ServingHostProfile.forCurrentProcess()
        let executable = URL(fileURLWithPath: path)
        let collected = Collected()

        // Eight surfaces start at once: exactly one process wins the profile
        // lease and every client attaches to it.
        let connections = try await withThrowingTaskGroup(of: PrivateServingIPCConnection.self) { group in
            for index in 0..<8 {
                group.addTask {
                    let attachment = try ServingHostRuntime.connectOrLaunch(
                        profile: profile, clientID: UUID(), role: index == 0 ? .api : .gui,
                        hostExecutable: executable)
                    collected.add(host: attachment.connection.host.hostID, pid: attachment.launchedHostPID)
                    return attachment.connection
                }
            }
            var all: [PrivateServingIPCConnection] = []
            for try await connection in group { all.append(connection) }
            return all
        }
        let snapshot = collected.snapshot
        #expect(Set(snapshot.hosts).count == 1, "all clients share one host")
        #expect(!snapshot.pids.isEmpty)
        // Callers that lost the lease race launched hosts that exit at once;
        // exactly one launched process remains.
        try await Task.sleep(for: .seconds(1))
        let alive = snapshot.pids.filter { kill($0, 0) == 0 }
        print("[launch-race] pids=\(snapshot.pids) alive=\(alive) hosts=\(Set(snapshot.hosts))")
        fflush(stdout)
        let hostPID = try #require(alive.first)

        // Closing every client (a quitting GUI and a stopping API adapter)
        // does not stop the host immediately; once idle it exits by itself and
        // releases the profile so the next client launches a fresh one.
        let client = PrivateServingIPCClient(connection: connections[0])
        for connection in connections.dropFirst() { connection.close() }
        try await Task.sleep(for: .seconds(2))
        #expect(kill(hostPID, 0) == 0, "the host outlives all but one client")
        client.close()
        var exited = false
        for _ in 0..<(50 * 40) {
            if kill(hostPID, 0) != 0 { exited = true; break }
            try await Task.sleep(for: .milliseconds(25))
        }
        #expect(exited, "an idle host with no clients exits")
        let again = try ServingHostRuntime.connectOrLaunch(
            profile: profile, clientID: UUID(), role: .gui,
            hostExecutable: executable)
        #expect(again.launchedHost)
        #expect(again.connection.host.hostID != snapshot.hosts[0])
        again.connection.close()
        if let pid = again.launchedHostPID { kill(pid, SIGTERM) }
    }
}
