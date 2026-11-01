#if os(macOS)
import Darwin
import Foundation

/// File operations relative to a selected directory. Every component is opened
/// without following symlinks; an open directory descriptor anchors the walk.
struct AppWorkspaceFiles: Sendable {
    let root: String
    private static let maximumBytes = 128 * 1024

    private func openFile(_ path: String, write: Bool) throws -> Int32 {
        let components = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard !path.hasPrefix("/"), !components.isEmpty,
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\0") }),
              canonicalPath(root) == root else { throw Failure("Use a relative path inside the selected folder, without symlinks.") }
        var parent = Darwin.open(root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parent >= 0 else { throw Failure("The selected folder is unavailable.") }
        defer { close(parent) }
        for component in components.dropLast() {
            let next = component.withCString { openat(parent, $0, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC) }
            guard next >= 0 else { throw Failure("A directory is missing or is a symbolic link.") }
            close(parent); parent = next
        }
        let name = components.last!
        var fd = name.withCString { openat(parent, $0, (write ? O_WRONLY : O_RDONLY) | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC) }
        if write && fd < 0 && errno == ENOENT {
            fd = name.withCString { openat(parent, $0, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600) }
        }
        guard fd >= 0 else { throw Failure("The file could not be opened; check its path and permissions.") }
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG, info.st_nlink == 1 else {
            close(fd); throw Failure("Only regular files without hard links can be accessed.")
        }
        return fd
    }

    func list(_ path: String) throws -> String {
        guard canonicalPath(root) == root else { throw Failure("The selected folder is unavailable.") }
        var directory = Darwin.open(root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard directory >= 0 else { throw Failure("The selected folder is unavailable.") }
        if path != "." {
            let components = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
            guard !path.hasPrefix("/"), components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\0") }) else {
                close(directory); throw Failure("Use a relative directory inside the selected folder.")
            }
            for component in components {
                let next = component.withCString { openat(directory, $0, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC) }
                close(directory)
                guard next >= 0 else { throw Failure("A directory is missing or is a symbolic link.") }
                directory = next
            }
        }
        guard let stream = fdopendir(directory) else { close(directory); throw Failure("Could not list directory.") }
        defer { closedir(stream) }
        var entries: [String] = []
        while let entry = readdir(stream) {
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) { String(cString: $0) }
            }
            if name == "." || name == ".." { continue }
            entries.append(AppFileNameText.safe(name))
            if entries.count >= 500 { break }
        }
        return entries.sorted().joined(separator: "\n") + (entries.count == 500 ? "\n[Listing limited to 500 entries.]" : "")
    }

    func read(_ path: String) throws -> String {
        let fd = try openFile(path, write: false)
        defer { close(fd) }
        var bytes = [UInt8](repeating: 0, count: Self.maximumBytes + 1)
        var count = 0
        let capacity = bytes.count
        while count < capacity {
            let n = bytes.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress!.advanced(by: count), capacity - count) }
            if n < 0 { if errno == EINTR { continue }; throw Failure("Could not read file.") }
            if n == 0 { break }; count += n
        }
        guard count <= Self.maximumBytes else { throw Failure("File exceeds the 128 KiB read limit.") }
        guard let text = String(bytes: bytes.prefix(count), encoding: .utf8) else { throw Failure("File is not UTF-8 text.") }
        return text
    }

    func write(_ path: String, content: String) throws {
        let bytes = Array(content.utf8)
        guard bytes.count <= Self.maximumBytes else { throw Failure("Content exceeds the 128 KiB write limit.") }
        let components = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard !path.hasPrefix("/"), !components.isEmpty,
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\0") }),
              canonicalPath(root) == root else { throw Failure("Use a relative path inside the selected folder, without symlinks.") }
        // Git runs configuration, hooks and filters outside any sandbox, so a
        // planted .git entry would persist as code execution. The volume is
        // case-insensitive by default.
        guard !components.contains(where: { $0.lowercased() == ".git" }) else {
            throw Failure("Writing inside .git is not allowed.")
        }
        var parent = Darwin.open(root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parent >= 0 else { throw Failure("The selected folder is unavailable.") }
        defer { close(parent) }
        for component in components.dropLast() {
            let next = component.withCString { openat(parent, $0, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC) }
            guard next >= 0 else { throw Failure("A directory is missing or is a symbolic link.") }
            close(parent); parent = next
        }
        let name = components.last!
        var info = stat()
        let found = name.withCString { fstatat(parent, $0, &info, AT_SYMLINK_NOFOLLOW) }
        if found == 0 {
            guard (info.st_mode & S_IFMT) == S_IFREG, info.st_nlink == 1 else {
                throw Failure("Only regular files without hard links can be written.")
            }
        } else if errno != ENOENT { throw Failure("Could not inspect the destination.") }
        let temporary = ".tuff-write-" + UUID().uuidString
        let fd = temporary.withCString { openat(parent, $0, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600) }
        guard fd >= 0 else { throw Failure("Could not create the replacement file.") }
        defer { close(fd); _ = temporary.withCString { unlinkat(parent, $0, 0) } }
        if found == 0 { _ = fchmod(fd, info.st_mode & 0o777) }
        var offset = 0
        while offset < bytes.count {
            let n = bytes.withUnsafeBytes { Darwin.write(fd, $0.baseAddress!.advanced(by: offset), bytes.count - offset) }
            if n < 0 { if errno == EINTR { continue }; throw Failure("Could not write file.") }
            guard n > 0 else { throw Failure("Could not write file.") }; offset += n
        }
        let renamed = temporary.withCString { source in name.withCString { target in renameat(parent, source, parent, target) } }
        guard renamed == 0 else { throw Failure("Could not replace the file.") }
    }

    func edit(_ path: String, old: String, new: String) throws {
        guard !old.isEmpty else { throw Failure("The old text must not be empty.") }
        let text = try read(path)
        let parts = text.components(separatedBy: old)
        guard parts.count == 2 else { throw Failure("The old text must occur exactly once; read the file and try again.") }
        try write(path, content: parts[0] + new + parts[1])
    }

    struct Failure: Error, CustomStringConvertible { let description: String; init(_ s: String) { description = s } }
}
#endif
