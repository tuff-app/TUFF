import Darwin
import Foundation

/// Lets a newer TUFF replace a serving host left running by an older one.
///
/// Updating TUFF replaces the app while the old Background API, or an old
/// `tuff serve` in a terminal, can stay attached to the old build's host, so
/// that host never goes idle and the updated app could not load any model.
/// A client from a newer build asks the older host to shut down (the same
/// SIGTERM drain as quitting it) and then starts its own. An older client
/// never stops a newer host, so two versions can't take turns replacing
/// each other.
enum ServingHostTakeover {
    /// The version in a `tuff-MAJOR.MINOR.PATCH` build identity.
    static func version(_ buildIdentity: String) -> [Int]? {
        guard buildIdentity.hasPrefix("tuff-") else { return nil }
        let parts = buildIdentity.dropFirst(5).split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 3 else { return nil }
        let numbers = parts.compactMap { Int($0) }
        return numbers.count == 3 ? numbers : nil
    }

    static func isOlder(_ host: String, than client: String) -> Bool {
        guard let host = version(host), let client = version(client) else { return false }
        return host.lexicographicallyPrecedes(client)
    }

    /// The process on the other end of a connected Unix socket.
    static func peerPID(_ descriptor: Int32) -> pid_t? {
        var pid: pid_t = 0
        var length = socklen_t(MemoryLayout<pid_t>.size)
        guard getsockopt(descriptor, SOL_LOCAL, LOCAL_PEERPID, &pid, &length) == 0, pid > 0 else {
            return nil
        }
        return pid
    }

    /// Asks `pid` to shut down if it is still a serving host owned by this
    /// user. The socket lives in a private directory, so its peer is this
    /// user's; the checks guard against the pid having been reused since.
    /// A packaged host runs the app's shared `TUFF` executable through the
    /// `bin/TUFFServer` link, so the name alone is not enough: the host is
    /// the only TUFF process started with `--private-host`.
    @discardableResult
    static func retire(pid: pid_t) -> Bool {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout<proc_bsdinfo>.size)
        guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size,
              info.pbi_uid == getuid() else { return false }
        var path = [CChar](repeating: 0, count: Int(MAXPATHLEN))
        guard proc_pidpath(pid, &path, UInt32(path.count)) > 0,
              ["TUFF", "TUFFServer"].contains(
                URL(fileURLWithPath: String(cString: path)).lastPathComponent),
              arguments(of: pid)?.contains("--private-host") == true
        else { return false }
        return kill(pid, SIGTERM) == 0
    }

    /// The command-line arguments of a process this user owns, or nil.
    static func arguments(of pid: pid_t) -> [String]? {
        var mib: [Int32] = [CTL_KERN, KERN_PROCARGS2, pid]
        var size = 0
        guard sysctl(&mib, 3, nil, &size, nil, 0) == 0, size > MemoryLayout<Int32>.size else {
            return nil
        }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctl(&mib, 3, &buffer, &size, nil, 0) == 0, size > MemoryLayout<Int32>.size else {
            return nil
        }
        // argc, then the executable path, NUL padding, then argv strings.
        let argc = buffer.withUnsafeBytes { Int($0.loadUnaligned(as: Int32.self)) }
        var index = MemoryLayout<Int32>.size
        while index < size, buffer[index] != 0 { index += 1 }
        while index < size, buffer[index] == 0 { index += 1 }
        var arguments: [String] = []
        while arguments.count < argc, index < size {
            let start = index
            while index < size, buffer[index] != 0 { index += 1 }
            arguments.append(String(decoding: buffer[start..<index], as: UTF8.self))
            index += 1
        }
        return arguments
    }
}
