import Darwin
import Foundation

/// The canonical path of an existing item: symbolic links resolved and
/// `/private` kept, so two paths compare equal exactly when they name the same
/// place. Foundation's symlink resolution strips `/private`, which made a
/// folder under /var and the files found in it disagree.
func canonicalPath(_ path: String) -> String? {
    guard let resolved = realpath(path, nil) else { return nil }
    defer { free(resolved) }
    return String(cString: resolved)
}

/// A folder the user chose in the open panel. Search reads only inside these.
public struct AppSearchFolder: Codable, Equatable, Hashable, Identifiable, Sendable {
    public let id: UUID
    /// Standardized, symlink-resolved path at the time it was chosen.
    public var path: String
    /// A bookmark, so a folder that moves is found again.
    public var bookmark: Data?
    public var addedAt: Date

    public init(id: UUID = UUID(), path: String, bookmark: Data? = nil, addedAt: Date = Date()) {
        self.id = id
        self.path = path
        self.bookmark = bookmark
        self.addedAt = addedAt
    }

    public var url: URL { URL(fileURLWithPath: path, isDirectory: true) }
    public var displayName: String { AppFileNameText.safe(url.lastPathComponent) }
}

/// File and folder names are chosen by whoever made the file, so they are
/// shown to the person and the model with invisible and direction-changing
/// characters written out (`\u{202E}`), not applied: a name such as
/// `report\u{202E}fdp.exe` cannot pass for a PDF.
public enum AppFileNameText {
    public static func safe(_ text: String) -> String {
        var result = ""
        for scalar in text.unicodeScalars {
            switch scalar.properties.generalCategory {
            case .control, .format, .lineSeparator, .paragraphSeparator:
                result += "\\u{" + String(scalar.value, radix: 16, uppercase: true) + "}"
            default:
                result.unicodeScalars.append(scalar)
            }
        }
        return result
    }
}

/// The chosen folders, stored in Application Support so clearing caches does
/// not forget them. The index itself is in Caches and can be rebuilt.
public struct AppSearchFolderStore: Sendable {
    public let fileURL: URL

    public init(fileURL: URL) { self.fileURL = fileURL }

    public static func standard(applicationSupport: URL = AppSupportMigration.applicationSupportURL())
        -> AppSearchFolderStore {
        AppSearchFolderStore(fileURL: applicationSupport
            .appendingPathComponent("TUFF/LocalSearch/folders.json"))
    }

    public func load() -> [AppSearchFolder] {
        guard let data = try? Data(contentsOf: fileURL),
              let folders = try? JSONDecoder().decode([AppSearchFolder].self, from: data) else {
            return []
        }
        var seen = Set<UUID>()
        return folders.compactMap { saved in
            guard seen.insert(saved.id).inserted else { return nil }
            var folder = saved
            if let bookmark = saved.bookmark {
                var stale = false
                if let resolved = try? URL(resolvingBookmarkData: bookmark,
                                           options: [.withoutUI, .withoutMounting],
                                           relativeTo: nil, bookmarkDataIsStale: &stale),
                   let path = canonicalPath(resolved.path), path != "/" {
                    folder.path = path
                    if stale {
                        folder.bookmark = try? resolved.bookmarkData(options: [],
                            includingResourceValuesForKeys: nil, relativeTo: nil)
                    }
                }
            }
            return folder
        }
    }

    public func save(_ folders: [AppSearchFolder]) throws {
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(folders).write(to: fileURL, options: [.atomic])
    }

    /// A folder chosen in the open panel, resolved and bookmarked.
    public static func folder(for url: URL) throws -> AppSearchFolder {
        var isDirectory = ObjCBool(false)
        guard let path = canonicalPath(url.path),
              FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw AppLocalSearchError.notAFolder(url.path)
        }
        let resolved = URL(fileURLWithPath: path, isDirectory: true)
        guard resolved.path != "/" else { throw AppLocalSearchError.tooBroad(resolved.path) }
        let bookmark = try? resolved.bookmarkData(options: [], includingResourceValuesForKeys: nil,
                                                  relativeTo: nil)
        return AppSearchFolder(path: resolved.path, bookmark: bookmark)
    }
}

public enum AppLocalSearchError: Error, Equatable, Sendable, CustomStringConvertible {
    case notAFolder(String)
    case tooBroad(String)
    case noFolders
    case index(String)

    public var description: String {
        switch self {
        case .notAFolder(let path): "\(AppFileNameText.safe((path as NSString).lastPathComponent)) is not a folder."
        case .tooBroad: "Choose a specific folder rather than the whole disk."
        case .noFolders: "No folders are selected for file search. Add one with the Files button."
        case .index(let message): message
        }
    }
}

/// One ranked passage from a local file.
public struct AppFilePassage: Equatable, Sendable {
    public let filePath: String
    public let folderID: UUID
    public let page: Int?
    public let lineStart: Int?
    public let lineEnd: Int?
    public let text: String
    public let score: Double

    public var displayName: String { AppFileNameText.safe((filePath as NSString).lastPathComponent) }

    public var location: String? {
        if let page { return "page \(page)" }
        if let lineStart, let lineEnd {
            return lineStart == lineEnd ? "line \(lineStart)" : "lines \(lineStart)-\(lineEnd)"
        }
        return nil
    }
}

public struct AppLocalIndexLimits: Equatable, Sendable {
    public var maximumFilesPerFolder = 20_000
    public var maximumTextFileBytes = 2 * 1_024 * 1_024
    public var maximumPDFBytes = 64 * 1_024 * 1_024
    public var maximumPDFPages = 500
    public var maximumDepth = 24
    public var passageCharacters = 1_200

    public init() {}
    public static let standard = AppLocalIndexLimits()
}

/// Folders and files the scan never enters: version control, dependency and
/// build output, which are large and rarely what someone means by "my files".
let skippedDirectoryNames: Set<String> = [
    ".git", ".hg", ".svn", "node_modules", ".build", "DerivedData", "Pods", "build",
    "dist", ".venv", "venv", "__pycache__", ".tox", ".gradle", "target", ".next",
]

public struct AppIndexFolderStatus: Equatable, Sendable {
    public var indexedFiles: Int = 0
    public var skippedFiles: Int = 0
    public var passages: Int = 0
    public var pendingFiles: Int = 0
    public var reachedFileLimit = false
    public var lastError: String?
    public var lastCompleted: Date?
}

/// SQLite FTS5 index of passages from the chosen folders, ranked with BM25.
/// No embedding model, nothing loaded beside the chat model.
public actor AppLocalSearchIndex {
    public static func standardURL() -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return caches.appendingPathComponent("TUFF/LocalSearch/index-v1.sqlite")
    }

    private let database: AppSQLiteDatabase
    public let databaseURL: URL

    public init(databaseURL: URL = AppLocalSearchIndex.standardURL()) throws {
        try FileManager.default.createDirectory(
            at: databaseURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        self.databaseURL = databaseURL
        database = try AppSQLiteDatabase(path: databaseURL.path)
        try database.execute("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS files(
                path TEXT PRIMARY KEY, folder TEXT NOT NULL, size INTEGER NOT NULL,
                mtime REAL NOT NULL, status TEXT NOT NULL, detail TEXT);
            CREATE INDEX IF NOT EXISTS files_folder ON files(folder);
            CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(
                text, path UNINDEXED, page UNINDEXED, line_start UNINDEXED,
                line_end UNINDEXED, tokenize='porter unicode61');
            """)
    }

    struct IndexedFile: Equatable {
        let size: Int
        let mtime: Double
        let status: String
    }

    func indexedFiles(folder: UUID) throws -> [String: IndexedFile] {
        var result: [String: IndexedFile] = [:]
        for row in try database.query(
            "SELECT path, size, mtime, status FROM files WHERE folder = ?",
            [.text(folder.uuidString)]) {
            guard let path = row[0].string, let size = row[1].int, let mtime = row[2].double,
                  let status = row[3].string else { continue }
            result[path] = IndexedFile(size: size, mtime: mtime, status: status)
        }
        return result
    }

    /// Replaces a file's passages in one transaction.
    func store(path: String, folder: UUID, size: Int, mtime: Double,
               passages: [(page: Int?, lineStart: Int?, lineEnd: Int?, text: String)]) throws {
        try database.execute("BEGIN IMMEDIATE")
        do {
            try database.query("DELETE FROM passages WHERE path = ?", [.text(path)])
            for passage in passages {
                try database.query(
                    "INSERT INTO passages(text, path, page, line_start, line_end) VALUES(?,?,?,?,?)",
                    [.text(passage.text), .text(path),
                     passage.page.map { .integer(Int64($0)) } ?? .null,
                     passage.lineStart.map { .integer(Int64($0)) } ?? .null,
                     passage.lineEnd.map { .integer(Int64($0)) } ?? .null])
            }
            try database.query(
                "INSERT OR REPLACE INTO files(path, folder, size, mtime, status, detail) VALUES(?,?,?,?,?,NULL)",
                [.text(path), .text(folder.uuidString), .integer(Int64(size)), .real(mtime),
                 .text("indexed")])
            try database.execute("COMMIT")
        } catch {
            try? database.execute("ROLLBACK")
            throw error
        }
    }

    /// Records a file that could not be indexed, so it is not retried until
    /// it changes.
    func skip(path: String, folder: UUID, size: Int, mtime: Double, reason: String) throws {
        try database.query("DELETE FROM passages WHERE path = ?", [.text(path)])
        try database.query(
            "INSERT OR REPLACE INTO files(path, folder, size, mtime, status, detail) VALUES(?,?,?,?,?,?)",
            [.text(path), .text(folder.uuidString), .integer(Int64(size)), .real(mtime),
             .text("skipped"), .text(reason)])
    }

    /// Forgets the files that were skipped, so the next update reads them
    /// again. Used when the reader changes: a file one reader could not
    /// handle may be fine for the other.
    func forgetSkipped() throws {
        try database.query("DELETE FROM files WHERE status != ?", [.text("indexed")])
    }

    func forget(paths: [String]) throws {
        guard !paths.isEmpty else { return }
        try database.execute("BEGIN IMMEDIATE")
        do {
            for path in paths {
                try database.query("DELETE FROM passages WHERE path = ?", [.text(path)])
                try database.query("DELETE FROM files WHERE path = ?", [.text(path)])
            }
            try database.execute("COMMIT")
        } catch {
            try? database.execute("ROLLBACK")
            throw error
        }
    }

    /// Deletes everything indexed for a folder. Used when access is removed.
    public func remove(folder: UUID) throws {
        let paths = try indexedFiles(folder: folder).keys
        try forget(paths: Array(paths))
    }

    public func removeAll() throws {
        try database.execute("DELETE FROM passages; DELETE FROM files;")
        try database.execute("VACUUM")
    }

    public func counts(folder: UUID) throws -> (indexed: Int, skipped: Int, passages: Int) {
        let files = try database.query(
            "SELECT status, COUNT(*) FROM files WHERE folder = ? GROUP BY status",
            [.text(folder.uuidString)])
        var indexed = 0, skipped = 0
        for row in files {
            if row[0].string == "indexed" { indexed = row[1].int ?? 0 } else { skipped += row[1].int ?? 0 }
        }
        let passages = try database.query(
            "SELECT COUNT(*) FROM passages WHERE path IN (SELECT path FROM files WHERE folder = ?)",
            [.text(folder.uuidString)]).first?.first?.int ?? 0
        return (indexed, skipped, passages)
    }

    /// An FTS5 expression for free text: each word quoted, so punctuation and
    /// operators in a query are plain text, and joined with OR so BM25 ranks
    /// passages by how many terms they contain and how rare those are.
    static func matchExpression(_ query: String) -> String? {
        var terms: [String] = []
        var seen = Set<String>()
        for word in query.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }) {
            let term = String(word.prefix(64))
            guard term.count > 1 || term.allSatisfy(\.isNumber), seen.insert(term).inserted else { continue }
            terms.append("\"\(term)\"")
            if terms.count == 16 { break }
        }
        return terms.isEmpty ? nil : terms.joined(separator: " OR ")
    }

    /// Ranked passages from `folders`. A candidate is returned only if its
    /// file still exists inside a chosen folder, is not a symbolic link, and
    /// has not changed since it was indexed; a changed file is left for the
    /// next index pass rather than quoted from stale text.
    public func search(_ query: String, folders: [AppSearchFolder],
                       limit: Int) throws -> [AppFilePassage] {
        guard !folders.isEmpty else { throw AppLocalSearchError.noFolders }
        guard let match = Self.matchExpression(query) else { return [] }
        let folderByID = Dictionary(uniqueKeysWithValues: folders.map { ($0.id.uuidString, $0) })
        let placeholders = folders.map { _ in "?" }.joined(separator: ",")
        let rows = try database.query("""
            SELECT p.text, p.path, p.page, p.line_start, p.line_end, bm25(passages), f.folder,
                   f.size, f.mtime
            FROM passages p JOIN files f ON f.path = p.path
            WHERE passages MATCH ? AND f.folder IN (\(placeholders))
            ORDER BY bm25(passages) LIMIT ?
            """, [.text(match)] + folders.map { .text($0.id.uuidString) }
                + [.integer(Int64(max(1, limit) * 4))])
        var passages: [AppFilePassage] = []
        var perFile: [String: Int] = [:]
        for row in rows {
            guard let text = row[0].string, let path = row[1].string,
                  let folderID = row[6].string, let folder = folderByID[folderID],
                  let size = row[7].int, let mtime = row[8].double else { continue }
            guard Self.isInside(path, folder: folder),
                  let current = Self.fileStamp(path), current.size == size,
                  abs(current.mtime - mtime) < 0.001 else { continue }
            guard perFile[path, default: 0] < 2 else { continue }
            perFile[path, default: 0] += 1
            passages.append(AppFilePassage(
                filePath: path, folderID: folder.id, page: row[2].int,
                lineStart: row[3].int, lineEnd: row[4].int, text: text,
                score: -(row[5].double ?? 0)))
            if passages.count == limit { break }
        }
        return passages
    }

    /// Whether `path` resolves to a regular file inside `folder` without
    /// passing through a symbolic link.
    static func isInside(_ path: String, folder: AppSearchFolder) -> Bool {
        let url = URL(fileURLWithPath: path)
        guard let values = try? url.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey]),
              values.isSymbolicLink != true, values.isRegularFile == true,
              let resolved = canonicalPath(path), let root = canonicalPath(folder.path) else {
            return false
        }
        // A path that resolves elsewhere passes through a link; one outside
        // the root escaped it.
        return resolved == path && resolved.hasPrefix(root.hasSuffix("/") ? root : root + "/")
    }

    static func fileStamp(_ path: String) -> (size: Int, mtime: Double)? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
              let size = (attributes[.size] as? NSNumber)?.intValue,
              let date = attributes[.modificationDate] as? Date else { return nil }
        return (size, date.timeIntervalSinceReferenceDate)
    }
}

/// Where the indexer gets the files of a folder and their passages from. The
/// default is the Mac process itself (`AppLocalIndexer.scan` and `passages`);
/// with the sandbox setting on it is a VM that has the folders mounted
/// read-only, so a file the app cannot parse safely is never opened here.
protocol AppLocalFileReading: Sendable {
    /// The files to index, in the form `AppLocalIndexer.scan` gives them.
    func scan(_ folder: AppSearchFolder, limits: AppLocalIndexLimits) async throws
        -> (files: [AppLocalIndexer.FoundFile], reachedLimit: Bool)
    /// The passages of one file. Throws `AppLocalFileSkip` for a file that
    /// cannot be indexed and `AppLocalFileChanged` for one that changed
    /// while it was read; any other error stops the folder's update.
    func passages(for file: AppLocalIndexer.FoundFile, in folder: AppSearchFolder,
                  limits: AppLocalIndexLimits) async throws -> [(Int?, Int?, Int?, String)]
}

/// This file cannot be indexed (too large, unreadable, no text); `reason` is
/// remembered so it is not tried again until it changes.
struct AppLocalFileSkip: Error, Equatable {
    let reason: String
}

/// The file changed while it was read: leave it for the next update.
struct AppLocalFileChanged: Error, Equatable {}

/// Walks the chosen folders and keeps the index current: new and changed
/// files are read, deleted ones dropped, unchanged ones left alone. It runs at
/// background priority, checks for cancellation between files, and waits
/// while `shouldPause` is true so it does not compete with inference.
public struct AppLocalIndexer: Sendable {
    public let index: AppLocalSearchIndex
    public let limits: AppLocalIndexLimits
    /// nil: the app reads the folders itself.
    let reader: (any AppLocalFileReading)?

    public init(index: AppLocalSearchIndex, limits: AppLocalIndexLimits = .standard) {
        self.init(index: index, limits: limits, reader: nil)
    }

    init(index: AppLocalSearchIndex, limits: AppLocalIndexLimits = .standard,
         reader: (any AppLocalFileReading)?) {
        self.index = index
        self.limits = limits
        self.reader = reader
    }

    public func update(_ folder: AppSearchFolder,
                       shouldPause: @escaping @Sendable () -> Bool = { false },
                       progress: @escaping @Sendable (AppIndexFolderStatus) -> Void = { _ in })
        async throws -> AppIndexFolderStatus {
        var status = AppIndexFolderStatus()
        let root = folder.url
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            status.lastError = "The folder is missing or TUFF can no longer read it."
            return status
        }
        let found: (files: [FoundFile], reachedLimit: Bool)
        if let reader {
            found = try await reader.scan(folder, limits: limits)
        } else {
            found = try scan(folder)
        }
        status.reachedFileLimit = found.reachedLimit
        let known = try await index.indexedFiles(folder: folder.id)
        let present = Set(found.files.map(\.path))
        try Task.checkCancellation()
        try await index.forget(paths: known.keys.filter { !present.contains($0) })

        let pending = found.files.filter { file in
            guard let previous = known[file.path] else { return true }
            return previous.size != file.size || abs(previous.mtime - file.mtime) >= 0.001
        }
        status.indexedFiles = known.values.filter { $0.status == "indexed" }.count
        status.skippedFiles = known.values.filter { $0.status != "indexed" }.count
        status.pendingFiles = pending.count
        progress(status)

        // Files whose stamp on the Mac disagreed with the one the reader gave.
        var stampMismatches = 0
        for file in pending {
            try Task.checkCancellation()
            while shouldPause() {
                try await Task.sleep(for: .milliseconds(500))
            }
            await Task.yield()
            let passages: [(Int?, Int?, Int?, String)]
            do {
                guard AppLocalSearchIndex.isInside(file.path, folder: folder),
                      let stamp = AppLocalSearchIndex.fileStamp(file.path),
                      stamp.size == file.size, abs(stamp.mtime - file.mtime) < 0.001 else {
                    stampMismatches += 1
                    continue
                }
                if let reader {
                    passages = try await reader.passages(for: file, in: folder, limits: limits)
                } else {
                    passages = try Self.passages(for: file, in: folder, limits: limits)
                }
            } catch is AppLocalFileChanged {
                continue
            } catch {
                // A problem with the sandbox itself says nothing about the
                // file: it stops the update instead of marking files skipped.
                if reader != nil, !(error is AppLocalFileSkip) { throw error }
                let reason = (error as? AppLocalFileSkip)?.reason ?? "\(error)"
                try await index.skip(path: file.path, folder: folder.id, size: file.size,
                                     mtime: file.mtime, reason: reason)
                status.skippedFiles += 1
                status.pendingFiles -= 1
                progress(status)
                continue
            }
            try Task.checkCancellation()
            guard AppLocalSearchIndex.isInside(file.path, folder: folder),
                  let stamp = AppLocalSearchIndex.fileStamp(file.path),
                  stamp.size == file.size, abs(stamp.mtime - file.mtime) < 0.001 else { continue }
            try await index.store(path: file.path, folder: folder.id, size: file.size,
                                  mtime: file.mtime,
                                  passages: passages.map { (page: $0.0, lineStart: $0.1,
                                                            lineEnd: $0.2, text: $0.3) })
            status.indexedFiles += 1
            status.pendingFiles -= 1
            if status.pendingFiles % 20 == 0 { progress(status) }
        }
        // A reader whose modification times never match the Mac's (a file
        // system that rounds them) would index nothing and say nothing.
        if reader != nil, pending.count >= 2, stampMismatches == pending.count {
            status.lastError = "The files' modification times as the sandbox reports them do not match the Mac's, "
                + "so nothing could be indexed."
        }
        let counts = try await index.counts(folder: folder.id)
        status.indexedFiles = counts.indexed
        status.skippedFiles = counts.skipped
        status.passages = counts.passages
        status.pendingFiles = 0
        status.lastCompleted = Date()
        progress(status)
        return status
    }

    struct FoundFile: Equatable {
        let path: String
        let size: Int
        let mtime: Double
        /// Identity the Mac's own scan saw; the read must open the same file.
        var device: Int32? = nil
        var inode: UInt64? = nil
    }

    /// Regular, readable-type files under the folder, without following
    /// symbolic links, entering packages or hidden directories, or leaving
    /// the folder.
    func scan(_ folder: AppSearchFolder) throws -> (files: [FoundFile], reachedLimit: Bool) {
        guard let canonicalRoot = canonicalPath(folder.path) else {
            throw AppLocalSearchError.index("The folder could not be read.")
        }
        let root = URL(fileURLWithPath: canonicalRoot, isDirectory: true)
        let rootPath = canonicalRoot.hasSuffix("/") ? canonicalRoot : canonicalRoot + "/"
        let keys: [URLResourceKey] = [.isSymbolicLinkKey, .isRegularFileKey, .isDirectoryKey,
                                      .fileSizeKey, .contentModificationDateKey]
        guard let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
            throw AppLocalSearchError.index("The folder could not be read.")
        }
        var files: [FoundFile] = []
        while let url = enumerator.nextObject() as? URL {
            try Task.checkCancellation()
            guard let values = try? url.resourceValues(forKeys: Set(keys)) else { continue }
            if values.isSymbolicLink == true {
                if values.isDirectory == true { enumerator.skipDescendants() }
                continue
            }
            if values.isDirectory == true {
                if skippedDirectoryNames.contains(url.lastPathComponent)
                    || enumerator.level > limits.maximumDepth {
                    enumerator.skipDescendants()
                }
                continue
            }
            guard values.isRegularFile == true,
                  DocumentTextExtractor.canExtract(from: url), !url.pathExtension.isEmpty
                    || url.lastPathComponent.lowercased().hasPrefix("readme") else { continue }
            let path = url.path
            guard path.hasPrefix(rootPath), canonicalPath(path) == path else { continue }
            var identity = stat()
            guard lstat(path, &identity) == 0, (identity.st_mode & S_IFMT) == S_IFREG else { continue }
            files.append(FoundFile(
                path: path, size: values.fileSize ?? 0,
                mtime: values.contentModificationDate?.timeIntervalSinceReferenceDate ?? 0,
                device: identity.st_dev, inode: identity.st_ino))
            if files.count >= limits.maximumFilesPerFolder { return (files, true) }
        }
        return (files, false)
    }

    /// Opens `file` below the folder one component at a time without following
    /// symbolic links, requires the descriptor to be the regular, singly
    /// linked file the scan saw, and returns its bytes. A path swapped for a
    /// link or another file after the scan reads nothing from outside.
    static func readVerified(_ file: FoundFile, in folder: AppSearchFolder,
                             maximumBytes: Int) throws -> Data {
        guard let canonicalRoot = canonicalPath(folder.path) else { throw AppLocalFileChanged() }
        let rootPath = canonicalRoot.hasSuffix("/") ? canonicalRoot : canonicalRoot + "/"
        guard file.path.hasPrefix(rootPath) else { throw AppLocalFileSkip(reason: "outside the folder") }
        let components = file.path.dropFirst(rootPath.count)
            .split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard !components.isEmpty,
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && !$0.contains("\0") }) else {
            throw AppLocalFileSkip(reason: "unexpected path")
        }
        var parent = Darwin.open(canonicalRoot, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parent >= 0 else { throw AppLocalFileChanged() }
        defer { close(parent) }
        for component in components.dropLast() {
            let next = component.withCString { openat(parent, $0, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC) }
            guard next >= 0 else { throw AppLocalFileChanged() }
            close(parent); parent = next
        }
        let fd = components.last!.withCString { openat(parent, $0, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC) }
        guard fd >= 0 else { throw AppLocalFileChanged() }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else { throw AppLocalFileChanged() }
        guard info.st_nlink == 1 else { throw AppLocalFileSkip(reason: "hard-linked file") }
        if let device = file.device, let inode = file.inode {
            guard info.st_dev == device, info.st_ino == inode else { throw AppLocalFileChanged() }
        }
        guard Int(info.st_size) == file.size else { throw AppLocalFileChanged() }
        guard file.size <= maximumBytes else { throw AppLocalSearchError.index("larger than the indexing limit") }
        var data = Data(count: file.size)
        var offset = 0
        while offset < file.size {
            let n = data.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress!.advanced(by: offset), file.size - offset) }
            if n < 0 { if errno == EINTR { continue }; throw AppLocalFileChanged() }
            if n == 0 { throw AppLocalFileChanged() }
            offset += n
        }
        return data
    }

    /// Passages of about `passageCharacters`, split on line boundaries, with
    /// their line range, or their PDF page.
    static func passages(for file: FoundFile, in folder: AppSearchFolder,
                         limits: AppLocalIndexLimits) throws
        -> [(Int?, Int?, Int?, String)] {
        let url = URL(fileURLWithPath: file.path)
        let isPDF = url.pathExtension.lowercased() == "pdf"
        let maximum = isPDF ? limits.maximumPDFBytes : limits.maximumTextFileBytes
        if file.size > maximum {
            throw AppLocalSearchError.index("larger than the indexing limit")
        }
        let data = try readVerified(file, in: folder, maximumBytes: maximum)
        let sections = try DocumentTextExtractor.extractSections(
            data: data, name: url, maximumPDFPages: limits.maximumPDFPages)
        var passages: [(Int?, Int?, Int?, String)] = []
        for section in sections {
            var buffer: [Substring] = []
            var count = 0
            var startLine = 1
            var line = 0
            func flush() {
                let text = buffer.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty {
                    passages.append(section.page == nil
                        ? (nil, startLine, line, text) : (section.page, nil, nil, text))
                }
                buffer.removeAll()
                count = 0
                startLine = line + 1
            }
            for raw in section.text.split(separator: "\n", omittingEmptySubsequences: false) {
                line += 1
                var piece = raw
                // A single enormous line (minified code, a data dump) is cut.
                while piece.count > limits.passageCharacters {
                    buffer.append(piece.prefix(limits.passageCharacters))
                    piece = piece.dropFirst(limits.passageCharacters)
                    flush()
                    startLine = line
                }
                buffer.append(piece)
                count += piece.count + 1
                if count >= limits.passageCharacters { flush() }
            }
            flush()
        }
        return passages
    }
}
