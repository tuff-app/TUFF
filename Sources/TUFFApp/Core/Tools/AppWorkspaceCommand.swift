#if os(macOS)
import Darwin
import Foundation

/// Commands run in a new process group, under a default-deny macOS seatbelt.
/// `setsid`/`setpgid` system calls are denied to descendants, but
/// `posix_spawn(POSIX_SPAWN_SETSID | POSIX_SPAWN_SETPGROUP)` applies the change
/// inside the kernel and is not filtered, so a descendant can leave the group.
/// Every run therefore also carries a unique profile tag (an `allow` for one
/// file that exists only for that run); after the shell ends, is cancelled or
/// times out, every same-user process whose sandbox allows the tag file but
/// denies a sibling decoy file is killed, wherever it sits in the process tree.
struct AppWorkspaceCommand: Sendable {
    let root: String
    let allowWrite: Bool
    let allowNetwork: Bool
    var timeout: TimeInterval = 30
    private static let outputLimit = 64 * 1024

    func run(_ command: String) async throws -> String {
        try preflight()
        let state = CommandCancellation()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    do { continuation.resume(returning: try execute(command, state: state)) }
                    catch { continuation.resume(throwing: error) }
                }
            }
        } onCancel: { state.cancel() }
    }

    private func preflight() throws {
        guard canonicalPath(root) == root, root != "/" else { throw AppWorkspaceFiles.Failure("The workspace is unavailable.") }
        var traversalFailed = false
        guard let e = FileManager.default.enumerator(at: URL(fileURLWithPath: root),
            includingPropertiesForKeys: nil, options: [], errorHandler: { _, _ in traversalFailed = true; return false }) else {
            throw AppWorkspaceFiles.Failure("Could not inspect the workspace.")
        }
        var count = 0
        for case let url as URL in e {
            if Task.isCancelled { throw CancellationError() }
            count += 1
            guard count <= 200_000 else { throw AppWorkspaceFiles.Failure("Workspace has too many items to check safely.") }
            var info = stat()
            guard lstat(url.path, &info) == 0 else { throw AppWorkspaceFiles.Failure("Could not inspect a workspace item.") }
            if (info.st_mode & S_IFMT) == S_IFREG && info.st_nlink > 1 {
                throw AppWorkspaceFiles.Failure("Commands cannot run while the workspace contains hard-linked files: \(AppFileNameText.safe(url.lastPathComponent)).")
            }
        }
        guard !traversalFailed else { throw AppWorkspaceFiles.Failure("Could not inspect the entire workspace.") }
    }

    private func execute(_ command: String, state: CommandCancellation) throws -> String {
        if state.isCancelled { throw CancellationError() }
        guard let runTag = CommandRunTag.make() else {
            throw AppWorkspaceFiles.Failure("Could not prepare command isolation.")
        }
        defer { runTag.remove() }
        var profile = """
        (version 1)
        (deny default)
        (allow process-exec process-fork)
        (allow syscall-unix)
        (deny syscall-unix (syscall-number 82 147))
        (allow sysctl-read)
        (allow file-read-metadata (literal "/var") (literal "/etc")
          (literal "/Library") (literal "/Library/Developer") (literal "/Applications"))
        (allow file-read-metadata (literal "/private") (literal "/private/etc")
          (literal "/private/etc/ssl") (literal "/private/var")
          (literal "/private/var/run"))
        (allow file-read* (literal "/") (literal "/private/var/select/sh")
          (subpath "/usr/bin") (subpath "/usr/lib") (subpath "/usr/share")
          (subpath "/System/Library") (subpath "/System/Volumes/Preboot/Cryptexes") (subpath "/bin")
          (subpath "/Library/Developer/CommandLineTools")
          (subpath "/Applications/Xcode.app/Contents/Developer")
          (literal "/private/var/select/developer_dir") (literal "/dev/null")
          (literal "/dev/random") (literal "/dev/urandom") (subpath (param "ROOT"))
          (literal (param "RUNTAG")))
        (allow file-write* (literal "/dev/null"))
        """
        // Directory ancestors are needed for getcwd, but cannot be listed or read.
        var args = ["-D", "ROOT=\(root)", "-D", "RUNTAG=\(runTag.tag)"]
        var ancestor = (root as NSString).deletingLastPathComponent
        var index = 0
        while ancestor != "/" && !ancestor.isEmpty {
            profile += "\n(allow file-read-metadata (literal (param \"P\(index)\")))"
            args += ["-D", "P\(index)=\(ancestor)"]
            ancestor = (ancestor as NSString).deletingLastPathComponent
            index += 1
        }
        if allowWrite {
            profile += "\n(allow file-write* (subpath (param \"ROOT\")))"
            // Git runs hooks, filters and fsmonitor commands from .git outside
            // any sandbox, so planting them would persist as code execution.
            // The volume is case-insensitive by default.
            profile += "\n(deny file-write* (regex #\"/\\.[gG][iI][tT](/|$)\"))"
        }
        if allowNetwork {
            // Loopback is denied after the broad allow (last match wins):
            // TUFF's own server and local development services must not be
            // reachable. Services listening on all interfaces remain reachable
            // through the Mac's LAN address.
            profile += "\n(allow network-outbound (remote tcp) (remote udp))\n(deny network-outbound (remote ip \"localhost:*\"))\n(allow file-read* (literal \"/private/etc/ssl/openssl.cnf\") (literal \"/private/etc/ssl/cert.pem\") (literal \"/private/etc/hosts\") (literal \"/private/var/run/resolv.conf\"))\n(allow mach-lookup (global-name \"com.apple.system.opendirectoryd.libinfo\") (global-name \"com.apple.SystemConfiguration.configd\") (global-name \"com.apple.mDNSResponder\"))"
        }
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_UID, Int32(getuid())]
        var processBytes = 0
        let processCountKnown = mib.withUnsafeMutableBufferPointer {
            sysctl($0.baseAddress, u_int($0.count), nil, &processBytes, nil, 0) == 0
        }
        // KERN_PROC_UID includes the rest of this user's apps. Keep a small
        // additional budget rather than imposing a low user-wide ceiling.
        let processCeiling = processCountKnown
            ? min(4096, max(128, processBytes / MemoryLayout<kinfo_proc>.stride + 64)) : 1024
        // Lower both soft and hard limits before evaluating model-supplied
        // shell code. RLIMIT_NPROC counts this user's processes, not just this
        // group, so a very busy account may be unable to launch more children.
        let limits = """
        n=$(ulimit -Hu); case "$n" in unlimited) n=\(processCeiling);; *[!0-9]*|'') exit 125;; *) [ "$n" -gt \(processCeiling) ] && n=\(processCeiling);; esac
        t=$(ulimit -Ht); case "$t" in unlimited) t=30;; *[!0-9]*|'') exit 125;; *) [ "$t" -gt 30 ] && t=30;; esac
        f=$(ulimit -Hf); case "$f" in unlimited) f=1048576;; *[!0-9]*|'') exit 125;; *) [ "$f" -gt 1048576 ] && f=1048576;; esac
        d=$(ulimit -Hn); case "$d" in unlimited) d=256;; *[!0-9]*|'') exit 125;; *) [ "$d" -gt 256 ] && d=256;; esac
        ulimit -Su "$n" && ulimit -Hu "$n" && ulimit -St "$t" && ulimit -Ht "$t" && ulimit -Sf "$f" && ulimit -Hf "$f" && ulimit -Sn "$d" && ulimit -Hn "$d" || exit 125
        """
        args += ["-p", profile, "/bin/sh", "-c", limits + "\n" + command]
        var outputPipe: [Int32] = [0, 0]
        guard pipe(&outputPipe) == 0 else { throw AppWorkspaceFiles.Failure("Could not create command output pipe.") }
        defer { close(outputPipe[0]); close(outputPipe[1]) }
        guard fcntl(outputPipe[0], F_SETFL, O_NONBLOCK) == 0 else {
            throw AppWorkspaceFiles.Failure("Could not configure bounded command output.")
        }
        var actions: posix_spawn_file_actions_t?
        guard posix_spawn_file_actions_init(&actions) == 0 else { throw AppWorkspaceFiles.Failure("Could not configure command isolation.") }
        defer { posix_spawn_file_actions_destroy(&actions) }
        guard posix_spawn_file_actions_adddup2(&actions, outputPipe[1], STDOUT_FILENO) == 0,
              posix_spawn_file_actions_adddup2(&actions, outputPipe[1], STDERR_FILENO) == 0,
              posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0) == 0,
              posix_spawn_file_actions_addclose(&actions, outputPipe[0]) == 0,
              posix_spawn_file_actions_addchdir_np(&actions, root) == 0 else {
            throw AppWorkspaceFiles.Failure("Could not configure command isolation.")
        }
        var attr: posix_spawnattr_t?
        guard posix_spawnattr_init(&attr) == 0 else { throw AppWorkspaceFiles.Failure("Could not configure command isolation.") }
        defer { posix_spawnattr_destroy(&attr) }
        guard posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT)) == 0,
              posix_spawnattr_setpgroup(&attr, 0) == 0 else {
            throw AppWorkspaceFiles.Failure("Could not configure command isolation.")
        }
        let arguments = ["/usr/bin/sandbox-exec"] + args
        let environment = ["PATH=/Library/Developer/CommandLineTools/usr/bin:/Applications/Xcode.app/Contents/Developer/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin", "HOME=\(root)", "TMPDIR=\(root)", "LANG=C"]
        let argv = arguments.map { strdup($0) } + [nil]
        let env = environment.map { strdup($0) } + [nil]
        defer { for p in argv { free(p) }; for p in env { free(p) } }
        var pid: pid_t = 0
        let result = argv.withUnsafeBufferPointer { ap in
            env.withUnsafeBufferPointer { ep in
                posix_spawn(&pid, "/usr/bin/sandbox-exec", &actions, &attr,
                            UnsafeMutablePointer(mutating: ap.baseAddress!),
                            UnsafeMutablePointer(mutating: ep.baseAddress!))
            }
        }
        guard result == 0 else { throw AppWorkspaceFiles.Failure("Could not start the command sandbox (\(result)).") }
        // The group is killed while the shell is still unreaped (its pid cannot
        // be reused), then tagged processes that left the group are swept.
        func terminateEverything() {
            kill(-pid, SIGKILL)
            CommandRunTag.sweep(runTag)
        }
        close(outputPipe[1]); outputPipe[1] = -1
        var output = Data(), dropped = false, status: Int32 = 0
        let deadline = ProcessInfo.processInfo.systemUptime + timeout
        var timedOut = false, cancelled = false
        func drainOutput(waitForWriters: Bool) {
            let drainDeadline = ProcessInfo.processInfo.systemUptime + (waitForWriters ? 0.25 : 0)
            var buffer = [UInt8](repeating: 0, count: 8192)
            while true {
                let n = Darwin.read(outputPipe[0], &buffer, buffer.count)
                if n > 0 {
                    let room = max(0, Self.outputLimit - output.count)
                    output.append(contentsOf: buffer.prefix(min(room, n)))
                    if n > room { dropped = true }
                    continue
                }
                if n == 0 { break }
                if errno == EINTR { continue }
                if waitForWriters && errno == EAGAIN
                    && ProcessInfo.processInfo.systemUptime < drainDeadline {
                    usleep(1_000)
                    continue
                }
                break
            }
        }
        while true {
            var buffer = [UInt8](repeating: 0, count: 8192)
            let n = Darwin.read(outputPipe[0], &buffer, buffer.count)
            if n > 0 {
                let room = max(0, Self.outputLimit - output.count)
                output.append(contentsOf: buffer.prefix(min(room, n)))
                if n > room { dropped = true }
            }
            if state.isCancelled || ProcessInfo.processInfo.systemUptime >= deadline {
                cancelled = state.isCancelled; timedOut = !cancelled
                terminateEverything()
                _ = waitpid(pid, &status, 0)
                drainOutput(waitForWriters: true)
                break
            }
            // WNOWAIT leaves the shell as a zombie, so kill(-pid) below cannot
            // hit a recycled process group.
            var exited = siginfo_t()
            if waitid(P_PID, id_t(pid), &exited, WEXITED | WNOHANG | WNOWAIT) == -1 {
                let waitError = errno
                if waitError == EINTR { continue }
                // Do not turn an unknown child state into a successful exit.
                // Kill any remaining sandbox descendants and keep diagnostics
                // while reporting that no trustworthy status was available.
                terminateEverything()
                _ = waitpid(pid, &status, 0)
                drainOutput(waitForWriters: true)
                var text = String(decoding: output, as: UTF8.self)
                if dropped { text += "\n[Output truncated.]" }
                let reason = AppCommandFailure.Reason.statusUnavailable(waitError)
                text += "\n[\(reason.summary).]"
                throw AppCommandFailure(output: text, reason: reason)
            }
            if exited.si_pid == pid {
                // End background descendants even when the shell exits first.
                terminateEverything()
                _ = waitpid(pid, &status, 0)
                drainOutput(waitForWriters: true)
                break
            }
            usleep(20_000)
        }
        if cancelled { throw CancellationError() }
        var text = String(decoding: output, as: UTF8.self)
        if dropped { text += "\n[Output truncated.]" }
        if timedOut {
            let timeoutSeconds = max(1, Int(timeout.rounded(.up)))
            text += "\n[\(AppCommandFailure.Reason.timedOut(timeoutSeconds).summary).]"
            throw AppCommandFailure(output: text, reason: .timedOut(timeoutSeconds))
        }
        if status & 0x7f != 0 {
            let signal = status & 0x7f
            text += "\n[Terminated by signal \(signal).]"
            throw AppCommandFailure(output: text, reason: .signal(signal))
        }
        let exitCode = (status >> 8) & 0xff
        text += "\n[Exit status \(exitCode).]"
        if exitCode != 0 {
            throw AppCommandFailure(output: text, reason: .exit(exitCode))
        }
        return text
    }
}

private final class CommandCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    var isCancelled: Bool { lock.withLock { cancelled } }
    func cancel() { lock.withLock { cancelled = true } }
}

/// A per-run marker in the command's seatbelt profile. The profile allows
/// reading `tag` (a file that exists only for this run) and nothing allows
/// `decoy`, so `sandbox_check` against another pid tells this run's
/// descendants (tag allowed, decoy denied) from unsandboxed processes (both
/// allowed) and from other sandboxes (tag denied).
private struct CommandRunTag {
    let directory: String
    var tag: String { directory + "/tag" }
    var decoy: String { directory + "/decoy" }

    static func make() -> CommandRunTag? {
        guard sandboxCheck != nil, let temporary = canonicalPath(NSTemporaryDirectory()) else { return nil }
        let directory = temporary + (temporary.hasSuffix("/") ? "" : "/") + "tuff-run-" + UUID().uuidString
        guard mkdir(directory, 0o700) == 0 else { return nil }
        let value = CommandRunTag(directory: directory)
        for path in [value.tag, value.decoy] {
            let fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard fd >= 0 else { value.remove(); return nil }
            close(fd)
        }
        return value
    }

    func remove() {
        unlink(tag); unlink(decoy); rmdir(directory)
    }

    // sandbox_check is private and variadic. On arm64 the variadic arguments
    // are passed on the stack, so the path follows five unused register
    // arguments; the other architecture passes it in the fourth register.
    #if arch(arm64)
    private typealias Check = @convention(c) (Int32, UnsafePointer<CChar>, Int32, Int, Int, Int, Int, Int, UnsafePointer<CChar>) -> Int32
    #else
    private typealias Check = @convention(c) (Int32, UnsafePointer<CChar>, Int32, UnsafePointer<CChar>) -> Int32
    #endif
    private static let sandboxCheck: Check? = {
        guard let handle = dlopen("/usr/lib/libsandbox.1.dylib", RTLD_LAZY),
              let symbol = dlsym(handle, "sandbox_check") else { return nil }
        return unsafeBitCast(symbol, to: Check.self)
    }()

    /// 0 from `sandbox_check` means the operation is allowed.
    private static func allowsRead(_ pid: pid_t, _ path: String) -> Bool {
        guard let check = sandboxCheck else { return false }
        // SANDBOX_CHECK_NO_REPORT | SANDBOX_FILTER_PATH
        let type: Int32 = 0x4000_0000 | 1
        return "file-read-data".withCString { operation in
            path.withCString { path in
                #if arch(arm64)
                check(pid, operation, type, 0, 0, 0, 0, 0, path) == 0
                #else
                check(pid, operation, type, path) == 0
                #endif
            }
        }
    }

    private static func userProcessIDs() -> [pid_t] {
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_UID, Int32(getuid())]
        var size = 0
        guard mib.withUnsafeMutableBufferPointer({ sysctl($0.baseAddress, u_int($0.count), nil, &size, nil, 0) }) == 0 else { return [] }
        var processes = [kinfo_proc](repeating: kinfo_proc(), count: size / MemoryLayout<kinfo_proc>.stride + 64)
        size = processes.count * MemoryLayout<kinfo_proc>.stride
        guard mib.withUnsafeMutableBufferPointer({ m in
            processes.withUnsafeMutableBytes { sysctl(m.baseAddress, u_int(m.count), $0.baseAddress, &size, nil, 0) }
        }) == 0 else { return [] }
        return processes.prefix(size / MemoryLayout<kinfo_proc>.stride).map { $0.kp_proc.p_pid }
    }

    /// Kills every process of this user that carries `run`'s tag, repeating
    /// until a pass finds none that were not already killed, so descendants
    /// forked during the sweep are caught too.
    static func sweep(_ run: CommandRunTag) {
        var killed = Set<pid_t>()
        for _ in 0..<50 {
            let found = userProcessIDs().filter { pid in
                pid > 1 && pid != getpid() && !killed.contains(pid)
                    && allowsRead(pid, run.tag) && !allowsRead(pid, run.decoy)
            }
            if found.isEmpty { return }
            for pid in found { kill(pid, SIGKILL); killed.insert(pid) }
            usleep(5_000)
        }
    }
}
#endif
