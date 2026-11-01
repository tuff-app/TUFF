import Darwin
import Foundation
import TUFFDecodeProtocol
import TUFFModelCatalog
import TUFFRuntimeManagement

/// The per-profile serving host owns every model handle and the private
/// socket. It runs as its own `TUFFServer --private-host` process so no GUI or
/// API process lifetime decides whether another surface's work survives.
/// Clients, including the one that launched it, always attach over IPC.
public final class ServingHostRuntime: @unchecked Sendable {
    public let profile: ServingHostProfile
    public let coordinator: ServingModelCoordinator
    public let server: PrivateServingIPCServer
    private let shutdownLock = NSLock()
    private var hasShutdown = false

    /// Shared by the host and every client. A host from another build is
    /// never attached to, so an updated app cannot talk to an old pool.
    public static var buildIdentity: String { "tuff-\(TUFFVersion.current)" }

    private init(profile: ServingHostProfile, buildIdentity: String,
                 installed: ServerInstalledModels, helperExecutable: URL?) throws {
        self.profile = profile
        let resolver = ServingModelHandleResolver(
            installed: installed, helperExecutable: helperExecutable)
        let coordinator = ServingModelCoordinator(resolve: { modelID in
            try await resolver.resolve(modelID: modelID)
        })
        self.coordinator = coordinator
        self.server = PrivateServingIPCServer(
            profile: profile,
            token: try profile.token(),
            buildIdentity: buildIdentity,
            statusSnapshot: { await coordinator.snapshot() }) { context, command, emit in
                await coordinator.handle(context: context, command: command, emit: emit)
            }
    }

    /// Acquires the profile lease and binds the socket. Throws
    /// `PrivateServingIPCError.hostAlreadyRunning` when another process owns it.
    public static func startOwner(
        profile: ServingHostProfile,
        buildIdentity: String = ServingHostRuntime.buildIdentity,
        installed: ServerInstalledModels,
        helperExecutable: URL? = nil
    ) throws -> ServingHostRuntime {
        let runtime = try ServingHostRuntime(
            profile: profile, buildIdentity: buildIdentity, installed: installed,
            helperExecutable: helperExecutable)
        try runtime.server.start()
        return runtime
    }

    /// Runs the dedicated host process until a signal arrives or every client
    /// has stayed away for `idleGraceSeconds` with no work in flight. Returns
    /// the process exit status.
    public static func runPrivateHost(
        profile: ServingHostProfile,
        installed: ServerInstalledModels,
        buildIdentity: String = ServingHostRuntime.buildIdentity,
        idleGraceSeconds: Int = 20,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) async -> Int32 {
        // A directly launched helper is a local-build/test override and is
        // ignored unless the process is explicitly a local build.
        let helper = environment["TUFF_LOCAL_BUILD"] == "1"
            ? environment["TUFF_SERVING_HELPER_EXECUTABLE"].map { URL(fileURLWithPath: $0) } : nil
        let runtime: ServingHostRuntime
        do {
            runtime = try startOwner(profile: profile, buildIdentity: buildIdentity,
                                     installed: installed, helperExecutable: helper)
        } catch PrivateServingIPCError.hostAlreadyRunning {
            // Losing the start race is success: the winner serves the profile.
            return 0
        } catch {
            FileHandle.standardError.write(Data("serving host could not start: \(error)\n".utf8))
            return 1
        }
        let signals = ServerTerminationSignals()
        let idle = Task {
            var quietSeconds = 0
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                if await runtime.isIdle { quietSeconds += 1 } else { quietSeconds = 0 }
                if quietSeconds >= idleGraceSeconds {
                    // Idle was observed, not guaranteed: a client may have
                    // connected since. Only the atomic commit decides.
                    if await runtime.retireIfIdle() {
                        await signals.cancel()
                        return
                    }
                    quietSeconds = 0
                }
            }
        }
        _ = await signals.wait()
        idle.cancel()
        await runtime.shutdown()
        await signals.cancel()
        return 0
    }

    /// Drains active sessions and releases model helpers before dropping the
    /// profile lease and socket.
    public func shutdown() async {
        let shouldShutdown = shutdownLock.withLock { () -> Bool in
            guard !hasShutdown else { return false }
            hasShutdown = true
            return true
        }
        guard shouldShutdown else { return }
        await coordinator.shutdown()
        server.shutdown()
    }

    /// No client is attached (handshaking ones count) and no request, load,
    /// unload or cleanup is in flight.
    public var isIdle: Bool {
        get async {
            guard server.connectedClientCount == 0 else { return false }
            return !(await coordinator.isBusy)
        }
    }

    /// Commits this host to exiting if, at this instant, nothing is attached or
    /// running. On success every later connection is refused at its handshake
    /// with a retiring marker (`PrivateServingIPCError.hostRetiring`), so a
    /// client that races the exit relaunches a fresh host instead of attaching
    /// to one that is about to drop its work.
    public func retireIfIdle() async -> Bool {
        guard !(await coordinator.isBusy) else { return false }
        return server.retireIfIdle()
    }

    /// Evicts only idle model handles; active requests keep their leases.
    @discardableResult
    public func evictIdleModels() async -> [String] {
        await coordinator.evictIdleModels()
    }

    // MARK: - Client side

    public struct Attachment: @unchecked Sendable {
        public let connection: PrivateServingIPCConnection
        /// True when this call started the host process.
        public let launchedHost: Bool
        /// The process this call started, if any. Other callers' hosts are
        /// not theirs to signal.
        public let launchedHostPID: pid_t?
    }

    /// Connects to the profile's host, launching the dedicated process when
    /// none answers. Safe to call from any number of racing processes: a
    /// second host exits at the profile lease, and every caller keeps retrying
    /// the connection, re-launching if the host it started has exited (for
    /// example one that was idling out at the moment of connection).
    public static func connectOrLaunch(
        profile: ServingHostProfile,
        clientID: UUID,
        role: ServingClientRole,
        buildIdentity: String = ServingHostRuntime.buildIdentity,
        hostExecutable: URL? = nil,
        timeoutSeconds: Double = 30
    ) throws -> Attachment {
        func connect() throws -> PrivateServingIPCConnection {
            try PrivateServingIPCConnection.connect(
                profile: profile, clientID: clientID, role: role,
                expectedBuildIdentity: buildIdentity)
        }
        do { return Attachment(connection: try connect(), launchedHost: false, launchedHostPID: nil) }
        catch PrivateServingIPCError.authentication { throw PrivateServingIPCError.incompatibleProtocol }
        catch let error as PrivateServingIPCError {
            switch error {
            case .rejected, .incompatibleBuild, .incompatibleProtocol: throw error
            default: break
            }
        }
        catch {}

        let executable = try hostExecutable ?? locateHostExecutable()
        var launched = false
        var lastLaunch = Date.distantPast
        var launchedPID: pid_t?
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        var lastError: (any Error)?
        while Date() < deadline {
            let needsLaunch = launchedPID.map { pid in
                // `kill(pid, 0)` succeeds for zombies, so the spawn thread
                // marks exit through `HostChildren`.
                HostChildren.hasExited(pid)
            } ?? true
            if needsLaunch, Date().timeIntervalSince(lastLaunch) > 0.5 {
                launchedPID = try spawnDetached(
                    executable: executable,
                    arguments: ["--private-host", "--serving-bundle-id", profile.bundleIdentifier])
                launched = true
                lastLaunch = Date()
            }
            do { return Attachment(connection: try connect(), launchedHost: launched, launchedHostPID: launchedPID) }
            catch PrivateServingIPCError.authentication {
                throw PrivateServingIPCError.incompatibleProtocol
            } catch let error as PrivateServingIPCError {
                switch error {
                case .rejected, .incompatibleBuild, .incompatibleProtocol: throw error
                default:
                    lastError = error
                    usleep(50_000)
                }
            } catch {
                lastError = error
                usleep(50_000)
            }
        }
        throw lastError ?? PrivateServingIPCError.socket("connect", ETIMEDOUT)
    }

    public static func locateHostExecutable(bundle: Bundle = .main) throws -> URL {
        if let resourceURL = bundle.resourceURL {
            let bundled = resourceURL.appendingPathComponent("bin/TUFFServer")
            if FileManager.default.isExecutableFile(atPath: bundled.path) { return bundled }
        }
        // The packaged server is a role-selecting symlink to the app binary.
        // Preserve argv[0] until after checking its sibling: resolving it first
        // moves from Resources/bin to MacOS, which has no TUFFServer entry.
        let invoked = URL(fileURLWithPath: CommandLine.arguments.first ?? "")
        let sibling = invoked.deletingLastPathComponent().appendingPathComponent("TUFFServer")
        if FileManager.default.isExecutableFile(atPath: sibling.path) { return sibling }
        throw PrivateServingIPCError.socket("locate host executable", ENOENT)
    }

    /// Starts a process in its own session so closing the launching app (or
    /// its terminal) does not signal it, and reaps it when it exits.
    private static func spawnDetached(executable: URL, arguments: [String]) throws -> pid_t {
        var attributes: posix_spawnattr_t?
        posix_spawnattr_init(&attributes)
        defer { posix_spawnattr_destroy(&attributes) }
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETSID | POSIX_SPAWN_CLOEXEC_DEFAULT))
        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        for descriptor in 0...2 {
            posix_spawn_file_actions_addopen(&actions, Int32(descriptor), "/dev/null",
                                             descriptor == 0 ? O_RDONLY : O_WRONLY, 0)
        }
        let argv = ([executable.path] + arguments).map { strdup($0) } + [nil]
        defer { for pointer in argv { free(pointer) } }
        let environment = ProcessInfo.processInfo.environment.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { for pointer in environment { free(pointer) } }
        var pid: pid_t = 0
        let status = posix_spawn(&pid, executable.path, &actions, &attributes, argv, environment)
        guard status == 0 else { throw PrivateServingIPCError.socket("spawn host", status) }
        HostChildren.reap(pid)
        return pid
    }
}

/// Tracks hosts this process started so zombies are reaped and "has it
/// exited" is answerable.
private enum HostChildren {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var exited: Set<pid_t> = []

    static func reap(_ pid: pid_t) {
        Thread.detachNewThread {
            var status: Int32 = 0
            while waitpid(pid, &status, 0) < 0, errno == EINTR {}
            lock.lock(); exited.insert(pid); lock.unlock()
        }
    }

    static func hasExited(_ pid: pid_t) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return exited.contains(pid)
    }
}
