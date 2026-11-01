#if os(macOS)
import Darwin
import Foundation
import Synchronization
import Testing
import TUFFInferenceTypes
@testable import TUFFAppCore

@Suite struct AppWorkspaceTests {
    private func fixture() throws -> (URL, URL) {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("tuff-workspace-test-\(UUID())")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let root = base.appendingPathComponent("workspace")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return (base, URL(fileURLWithPath: canonicalPath(root.path)!))
    }

    private func compileC(_ source: String, name: String, in root: URL) throws -> URL {
        let sourceURL = root.appendingPathComponent("\(name).c")
        let binaryURL = root.appendingPathComponent(name)
        try source.write(to: sourceURL, atomically: true, encoding: .utf8)
        let compiler = Process()
        compiler.executableURL = URL(fileURLWithPath: "/usr/bin/clang")
        compiler.arguments = ["-Wall", "-Werror", sourceURL.path, "-o", binaryURL.path]
        let output = Pipe()
        compiler.standardOutput = output
        compiler.standardError = output
        try compiler.run()
        compiler.waitUntilExit()
        guard compiler.terminationStatus == 0 else {
            let diagnostics = output.fileHandleForReading.readDataToEndOfFile()
            throw AppWorkspaceFiles.Failure("Could not compile \(name): \(String(decoding: diagnostics, as: UTF8.self))")
        }
        try? FileManager.default.removeItem(at: sourceURL)
        return binaryURL
    }

    /// A non-loopback IPv4 address of this Mac, if it has one.
    private func lanIPv4() -> String? {
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0 else { return nil }
        defer { freeifaddrs(list) }
        var cursor = list
        while let entry = cursor?.pointee {
            cursor = entry.ifa_next
            let wanted = UInt32(IFF_UP | IFF_RUNNING)
            guard let address = entry.ifa_addr, address.pointee.sa_family == sa_family_t(AF_INET),
                  entry.ifa_flags & wanted == wanted, entry.ifa_flags & UInt32(IFF_LOOPBACK) == 0 else { continue }
            var sin = address.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { $0.pointee }
            var buffer = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
            inet_ntop(AF_INET, &sin.sin_addr, &buffer, socklen_t(buffer.count))
            let text = String(cString: buffer)
            if !text.hasPrefix("169.254.") { return text }
        }
        return nil
    }

    @Test func fileOperationsRejectEscapes() throws {
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        let outside = base.appendingPathComponent("outside")
        try "fixture".write(to: outside, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("link"), withDestinationURL: outside)
        try FileManager.default.linkItem(at: outside, to: root.appendingPathComponent("hardlink"))
        let files = AppWorkspaceFiles(root: root.path)
        #expect(throws: (any Error).self) { try files.read("../outside") }
        #expect(throws: (any Error).self) { try files.read("link") }
        #expect(throws: (any Error).self) { try files.write("link", content: "bad") }
        #expect(throws: (any Error).self) { try files.read("hardlink") }
        #expect(throws: (any Error).self) { try files.write("hardlink", content: "bad") }
        try files.write("file", content: "before")
        try files.edit("file", old: "before", new: "after")
        #expect(try files.read("file") == "after")
        #expect(try String(contentsOf: outside, encoding: .utf8) == "fixture")
    }

    @Test func oldCapabilitiesDecodeWithNewPermissionsOff() throws {
        let saved = Data(#"{"web":true,"files":true}"#.utf8)
        let capabilities = try JSONDecoder().decode(AppChatCapabilities.self, from: saved)
        #expect(capabilities.files && capabilities.web)
        #expect(!capabilities.workspaceRead && !capabilities.fileWrite
                && !capabilities.command && !capabilities.commandNetwork)
    }

    @Test func localFilesAloneGrantNoRawReads() throws {
        // Legacy settings: Local Files on, no workspace ever chosen.
        let capabilities = try JSONDecoder().decode(
            AppChatCapabilities.self, from: Data(#"{"files":true}"#.utf8))
        let names = AppToolCatalog.definitions(for: capabilities).map(\.name)
        #expect(names == ["search_files"])
        for tool in ["read_file", "list_files"] {
            let call = AppToolCall(id: "r", name: tool, arguments: .object(["path": .string(".")]))
            #expect(throws: (any Error).self) { try AppToolCatalog.validate(call, capabilities: capabilities) }
        }
        let withWorkspace = AppChatCapabilities(files: true, workspaceRead: true)
        #expect(AppToolCatalog.definitions(for: withWorkspace).map(\.name)
                == ["search_files", "list_files", "read_file"])
    }

    @Test func gitDirectoryIsNeverWritten() throws {
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        try FileManager.default.createDirectory(at: root.appendingPathComponent(".git"), withIntermediateDirectories: false)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("sub/.GIT"), withIntermediateDirectories: true)
        let files = AppWorkspaceFiles(root: root.path)
        for path in [".git/config", ".git/hooks/pre-commit", "sub/.GIT/config", ".GIT/config", "sub/.git"] {
            #expect(throws: (any Error).self) { try files.write(path, content: "[core]\nfsmonitor = x") }
        }
        #expect(throws: (any Error).self) { try files.edit(".git/config", old: "a", new: "b") }
        try files.write(".gitignore", content: "build\n")
        try files.write("sub/.github", content: "ok")
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent(".git/config").path))
    }

    @Test func fileCapabilitiesAreIndependent() throws {
        let write = AppToolCall(id: "w", name: "write_file", arguments: .object([
            "path": .string("x"), "content": .string("content")]))
        let run = AppToolCall(id: "r", name: "run_command", arguments: .object(["command": .string("true")]))
        #expect(throws: (any Error).self) { try AppToolCatalog.validate(write, capabilities: .init(files: true)) }
        #expect(throws: (any Error).self) { try AppToolCatalog.validate(run, capabilities: .init(files: true, fileWrite: true)) }
        _ = try AppToolCatalog.validate(write, capabilities: .init(files: true, fileWrite: true))
        _ = try AppToolCatalog.validate(run, capabilities: .init(files: true, command: true))
    }

    @Test func readOnlyCommandsCannotWriteAndHardlinksAreRefused() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1" else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        let runner = AppWorkspaceCommand(root: root.path, allowWrite: false, allowNetwork: false)
        var failure: AppCommandFailure?
        do { _ = try await runner.run("echo bad > prohibited") }
        catch let error as AppCommandFailure { failure = error }
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("prohibited").path))
        #expect(failure?.reason == .exit(1))
        #expect(failure?.output.contains("Operation not permitted") == true)
        let outside = base.appendingPathComponent("outside")
        try "fixture".write(to: outside, atomically: true, encoding: .utf8)
        try FileManager.default.linkItem(at: outside, to: root.appendingPathComponent("linked\u{202E}name"))
        do {
            _ = try await runner.run("cat linked*")
            Issue.record("a command ran with a hard-linked file present")
        } catch {
            // The file name is shown to the model; it must not carry control or bidi characters.
            #expect(!"\(error)".contains("\u{202E}"))
            #expect("\(error)".contains("hard-linked"))
        }
    }

    @Test func commandsCannotCreateHardLinksToOutsideFiles() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1" else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        try "outside-secret".write(to: base.appendingPathComponent("outside"), atomically: true, encoding: .utf8)
        let runner = AppWorkspaceCommand(root: root.path, allowWrite: true, allowNetwork: false)
        var output = ""
        do { output = try await runner.run("ln ../outside x 2>&1; cat x 2>&1; cat ../outside 2>&1; true") }
        catch let failure as AppCommandFailure { output = failure.output }
        #expect(!output.contains("outside-secret"))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("x").path))
    }

    @Test func commandsCannotReadOtherProcessEnvironments() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1" else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        let secret = "tuff-env-secret-\(UUID().uuidString)"
        let other = Process()
        other.executableURL = URL(fileURLWithPath: "/bin/sleep")
        other.arguments = ["30"]
        other.environment = ["TUFF_PROBE_SECRET": secret, "PATH": "/usr/bin:/bin"]
        try other.run()
        defer { other.terminate(); other.waitUntilExit() }
        let runner = AppWorkspaceCommand(root: root.path, allowWrite: true, allowNetwork: false)
        var output = ""
        do { output = try await runner.run("ps -Aww -E -o pid=,command= 2>&1 | grep -c sleep; ps -Aww -E -o pid=,command= 2>&1 | grep tuff-env-secret; true") }
        catch let failure as AppCommandFailure { output = failure.output }
        #expect(!output.contains(secret), "another process's environment was readable: \(output)")
    }

    @Test func commandNetworkRequiresOptInAndNeverReachesThisMac() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1" else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        // A listener on all interfaces counts connections. Nothing outside this
        // Mac is contacted: the allowed case is a connected UDP socket to a
        // reserved documentation address, which only asks the kernel for a
        // route and sends no packet.
        let listener = socket(AF_INET, SOCK_STREAM, 0)
        guard listener >= 0 else { throw AppWorkspaceFiles.Failure("Could not create fixture socket.") }
        defer { close(listener) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr.s_addr = INADDR_ANY
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bound == 0, listen(listener, 5) == 0 else { throw AppWorkspaceFiles.Failure("Could not start fixture listener.") }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(listener, $0, &length) }
        }
        _ = fcntl(listener, F_SETFL, O_NONBLOCK)
        let acceptedConnections = Mutex(0)
        let server = Task.detached {
            while !Task.isCancelled {
                let client = accept(listener, nil, nil)
                if client >= 0 {
                    acceptedConnections.withLock { $0 += 1 }
                    close(client)
                }
                usleep(10_000)
            }
        }
        let port = UInt16(bigEndian: address.sin_port)
        let source = """
        #include <arpa/inet.h>
        #include <errno.h>
        #include <netinet/in.h>
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <sys/socket.h>
        #include <unistd.h>
        int main(int argc, char **argv) {
          if (argc != 4) return 64;
          int udp = strcmp(argv[1], "udp") == 0;
          int fd = socket(AF_INET, udp ? SOCK_DGRAM : SOCK_STREAM, 0);
          if (fd < 0) { fprintf(stderr, "socket: %s\\n", strerror(errno)); return 65; }
          struct sockaddr_in address = {0};
          address.sin_len = sizeof(address);
          address.sin_family = AF_INET;
          address.sin_port = htons((unsigned short)atoi(argv[3]));
          if (inet_pton(AF_INET, argv[2], &address.sin_addr) != 1) return 66;
          if (connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
            int code = errno;
            fprintf(stderr, "connect: %s (%d)\\n", strerror(code), code);
            close(fd);
            return (code == EPERM || code == EACCES) ? 77 : 1;
          }
          puts("connected");
          close(fd);
          return 0;
        }
        """
        guard FileManager.default.isExecutableFile(atPath: "/usr/bin/clang") else {
            throw AppWorkspaceFiles.Failure("The local C compiler is unavailable for the network fixture.")
        }
        let binaryURL = try compileC(source, name: "socket-probe", in: root)
        /// The probe's exit status, or nil when the command itself failed to run.
        func exitStatus(_ mode: String, _ target: String, port: UInt16, network: Bool) async -> AppCommandFailure.Reason? {
            do {
                let output = try await AppWorkspaceCommand(root: root.path, allowWrite: false, allowNetwork: network)
                    .run("\(binaryURL.path) \(mode) \(target) \(port)")
                return output.contains("connected") ? .exit(0) : nil
            } catch { return (error as? AppCommandFailure)?.reason }
        }
        let documentation = "192.0.2.1"
        // Off: every destination is denied.
        var targets = ["127.0.0.1"]
        if let lan = lanIPv4() { targets.append(lan) }
        for target in targets {
            let status = await exitStatus("tcp", target, port: port, network: false)
            #expect(status == .exit(77), "network was reachable with command network off: \(target)")
        }
        let offUDP = await exitStatus("udp", documentation, port: 9, network: false)
        #expect(offUDP == .exit(77))
        // On: a routable destination is no longer denied by the sandbox ...
        let onUDP = await exitStatus("udp", documentation, port: 9, network: true)
        #expect(onUDP != .exit(77), "command network on did not allow outbound UDP")
        // ... while this Mac's loopback and own addresses stay denied.
        for target in targets {
            let status = await exitStatus("tcp", target, port: port, network: true)
            #expect(status == .exit(77), "local service reachable with command network on: \(target)")
        }
        let loopbackUDP = await exitStatus("udp", "127.0.0.1", port: port, network: true)
        #expect(loopbackUDP == .exit(77))
        try await Task.sleep(for: .milliseconds(200))
        server.cancel()
        await server.value
        #expect(acceptedConnections.withLock { $0 } == 0)
    }

    @Test func developerToolsWorkInsideWorkspace() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1",
              FileManager.default.fileExists(atPath: "/Library/Developer/CommandLineTools/usr/bin/python3") else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        // Writes to .git are denied inside the command, so the repository is
        // created by the app side of the fixture.
        let git = Process()
        git.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        git.arguments = ["init", "-q"]
        git.currentDirectoryURL = root
        try git.run(); git.waitUntilExit()
        #expect(git.terminationStatus == 0)
        let runner = AppWorkspaceCommand(root: root.path, allowWrite: true, allowNetwork: false)
        let output = try await runner.run("python3 -c 'print(123)' && git status --short")
        #expect(output.contains("123"))
        #expect(output.contains("Exit status 0"))
    }

    @Test func commandsCannotWriteGitDirectories() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1" else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        try FileManager.default.createDirectory(at: root.appendingPathComponent(".git/hooks"), withIntermediateDirectories: true)
        try "[core]\n".write(to: root.appendingPathComponent(".git/config"), atomically: true, encoding: .utf8)
        let runner = AppWorkspaceCommand(root: root.path, allowWrite: true, allowNetwork: false)
        let output: String
        do {
            output = try await runner.run("""
                echo '[core] fsmonitor=x' >> .git/config 2>&1
                echo '#!/bin/sh' > .git/hooks/pre-commit 2>&1
                mkdir .GIT 2>&1; mkdir -p sub/.git 2>&1
                mv .git moved 2>&1
                echo ok > .gitignore
                true
                """)
        } catch let failure as AppCommandFailure { output = failure.output }
        #expect(output.contains("Operation not permitted"))
        #expect(try String(contentsOf: root.appendingPathComponent(".git/config"), encoding: .utf8) == "[core]\n")
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent(".git/hooks/pre-commit").path))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("sub/.git").path))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("moved").path))
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent(".gitignore").path))
    }

    /// A child that asks the kernel for a new session or group through
    /// posix_spawn flags is outside the shell's group. It must still be killed.
    private func escapeProbe(in root: URL) throws -> URL {
        let source = """
        #include <spawn.h>
        #include <stdio.h>
        #include <string.h>
        #include <unistd.h>
        #ifndef POSIX_SPAWN_SETSID
        #define POSIX_SPAWN_SETSID 0x0400
        #endif
        int main(int argc, char **argv) {
          if (argc != 2) return 64;
          posix_spawnattr_t attr;
          posix_spawnattr_init(&attr);
          short flags = strcmp(argv[1], "setsid") == 0 ? POSIX_SPAWN_SETSID : POSIX_SPAWN_SETPGROUP;
          posix_spawnattr_setflags(&attr, flags);
          if (flags & POSIX_SPAWN_SETPGROUP) posix_spawnattr_setpgroup(&attr, 0);
          char *args[] = {"/bin/sh", "-c", "sleep 2; echo escaped > escaped", NULL};
          char *env[] = {NULL};
          pid_t child;
          int rc = posix_spawn(&child, "/bin/sh", NULL, &attr, args, env);
          if (rc != 0) { fprintf(stderr, "spawn: %d\\n", rc); return 65; }
          printf("own group %d, child %d, child group %d\\n", (int)getpgrp(), (int)child, (int)getpgid(child));
          return (getpgid(child) == child && getpgrp() != child) ? 0 : 66;
        }
        """
        return try compileC(source, name: "escape-probe", in: root)
    }

    @Test(arguments: ["setsid", "setpgroup"])
    func spawnFlagEscapesAreKilledWhenTheCommandEnds(flag: String) async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1" else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        let probe = try escapeProbe(in: root)
        let runner = AppWorkspaceCommand(root: root.path, allowWrite: true, allowNetwork: false)
        let output = try await runner.run("\(probe.path) \(flag)")
        #expect(output.contains("Exit status 0"), "the probe did not leave the process group: \(output)")
        try await Task.sleep(for: .milliseconds(3000))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("escaped").path),
                "a process that escaped the group outlived the command")
    }

    @Test func spawnFlagEscapesAreKilledOnTimeoutAndCancel() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1" else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        let probe = try escapeProbe(in: root)
        let timed = AppWorkspaceCommand(root: root.path, allowWrite: true, allowNetwork: false, timeout: 0.7)
        var failure: AppCommandFailure?
        do { _ = try await timed.run("\(probe.path) setsid; sleep 5") }
        catch let error as AppCommandFailure { failure = error }
        #expect(failure?.reason == .timedOut(1))
        #expect(failure?.output.contains("child group") == true)
        try await Task.sleep(for: .milliseconds(3000))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("escaped").path))

        let cancellable = AppWorkspaceCommand(root: root.path, allowWrite: true, allowNetwork: false, timeout: 10)
        let task = Task { try await cancellable.run("\(probe.path) setpgroup; sleep 8") }
        try await Task.sleep(for: .milliseconds(600))
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        try await Task.sleep(for: .milliseconds(3000))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("escaped").path))
    }

    @Test func cancellationEndsDescendants() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1" else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        let runner = AppWorkspaceCommand(root: root.path, allowWrite: true, allowNetwork: false, timeout: 5)
        let task = Task { try await runner.run("(sleep 1; echo escaped > escaped) & wait") }
        try await Task.sleep(for: .milliseconds(200))
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        try await Task.sleep(for: .milliseconds(1100))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("escaped").path))
    }

    @Test func sandboxedCommandAndTimeout() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1" else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        try "fixture".write(to: base.appendingPathComponent("outside"), atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("link"), withDestinationURL: base.appendingPathComponent("outside"))
        let runner = AppWorkspaceCommand(root: root.path, allowWrite: true, allowNetwork: false, timeout: 0.5)
        var failure: AppCommandFailure?
        do { _ = try await runner.run("echo inside > file; cat file; cat link; sleep 5") }
        catch let error as AppCommandFailure { failure = error }
        #expect(failure?.reason == .timedOut(1))
        #expect(failure?.output.contains("inside") == true)
        #expect(failure?.output.contains("fixture") == false)
        #expect(failure?.output.contains("timed out after 1 second") == true)
        #expect(try String(contentsOf: root.appendingPathComponent("file"), encoding: .utf8).contains("inside"))
    }

    @Test func commandExitStatusAndCapturedOutputReachToolActivity() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_COMMAND_SANDBOX_TEST"] == "1" else { return }
        let (base, root) = try fixture(); defer { try? FileManager.default.removeItem(at: base) }
        let runner = AppWorkspaceCommand(root: root.path, allowWrite: false, allowNetwork: false)
        let queries = Mutex<[String]>([])
        let toolbox = AppToolbox(
            webSearch: { query, _ in
                queries.withLock { $0.append(query) }
                throw AppToolValidationError("A private query should not reach the provider.")
            },
            readWebpage: { _ in throw AppToolValidationError("Not used in this test.") },
            searchFiles: { _, _ in [] },
            coding: { call in
                guard case .runCommand(let command) = call else {
                    throw AppToolValidationError("Expected a command call.")
                }
                return try await runner.run(command)
            })
        let session = AppToolAnswerSession(capabilities: .init(web: true, command: true),
                                           toolbox: toolbox, userText: "")
        let states = Mutex<[String: AppToolActivity.State]>([:])
        let round = await session.execute(calls: [
            AppToolCall(id: "fail", name: "run_command",
                        arguments: .object(["command": .string("printf 'failure marker\\n'; exit 7")])),
            ToolLoopFixtures.search("failure marker", id: "search"),
            AppToolCall(id: "pass", name: "run_command",
                        arguments: .object(["command": .string("printf 'success marker\\n'; exit 0")]))
        ], thinking: nil, content: "", characterBudget: 20_000, onActivity: { activity in
            states.withLock { $0[activity.id] = activity.state }
        })

        #expect(round.results.map(\.status) == [.failed, .refused, .succeeded])
        #expect(round.results[0].modelText.contains("failure marker"))
        #expect(round.results[0].modelText.contains("[Exit status 7.]"))
        #expect(round.results[0].summary == "Exit status 7")
        #expect(round.results[1].modelText.contains("This chat contains local files"))
        #expect(queries.withLock { $0.isEmpty })
        #expect(round.results[2].modelText.contains("success marker"))
        #expect(round.results[2].modelText.contains("[Exit status 0.]"))
        #expect(states.withLock { $0["fail"] } == .failed)
        #expect(states.withLock { $0["search"] } == .refused)
        #expect(states.withLock { $0["pass"] } == .succeeded)
    }
}
#endif
