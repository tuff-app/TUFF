import AppKit
import Foundation
import PDFKit
import Testing
@testable import TUFFAppCore

/// Local search against a temporary folder of invented files. Nothing reads
/// the person's documents.
@Suite(.serialized) struct AppLocalSearchTests {
    struct Sandbox {
        let root: URL
        let folder: AppSearchFolder
        let index: AppLocalSearchIndex
        let indexer: AppLocalIndexer

        func write(_ path: String, _ text: String) throws {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try text.write(to: url, atomically: true, encoding: .utf8)
        }

        func cleanUp() { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }
    }

    static func sandbox(limits: AppLocalIndexLimits = .standard) throws -> Sandbox {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-local-search-\(UUID().uuidString)")
        let root = base.appendingPathComponent("Notes")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let index = try AppLocalSearchIndex(databaseURL: base.appendingPathComponent("index.sqlite"))
        return Sandbox(root: root, folder: try AppSearchFolderStore.folder(for: root), index: index,
                       indexer: AppLocalIndexer(index: index, limits: limits))
    }

    /// A two-page PDF with distinct text on each page.
    static func writePDF(_ url: URL, pages: [String]) throws {
        let data = NSMutableData()
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let consumer = CGDataConsumer(data: data as CFMutableData),
              let context = CGContext(consumer: consumer, mediaBox: &box, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        for text in pages {
            context.beginPDFPage(nil)
            let graphics = NSGraphicsContext(cgContext: context, flipped: false)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = graphics
            NSAttributedString(string: text, attributes: [.font: NSFont.systemFont(ofSize: 14)])
                .draw(in: CGRect(x: 72, y: 72, width: 468, height: 648))
            NSGraphicsContext.restoreGraphicsState()
            context.endPDFPage()
        }
        context.closePDF()
        try (data as Data).write(to: url)
    }

    @Test func passagesAreRankedWithTheirLinesAndPages() async throws {
        let box = try Self.sandbox()
        defer { box.cleanUp() }
        try box.write("garden.md", (1...30).map { "Line \($0) about tomatoes and basil." }
            .joined(separator: "\n") + "\nThe compost bin needs turning every Tuesday.\n")
        try box.write("code/Pump.swift", "struct Pump {\n    func prime() {}\n}\n")
        try box.write("ignored.bin.png", "not text")
        try Self.writePDF(box.root.appendingPathComponent("manual.pdf"), pages: [
            "Chapter one covers installation of the water pump.",
            "Chapter two explains the irrigation timer and its schedule.",
        ])
        let status = try await box.indexer.update(box.folder)
        #expect(status.indexedFiles == 3)
        #expect(status.lastError == nil)

        let compost = try await box.index.search("When does the compost get turned?",
                                                 folders: [box.folder], limit: 3)
        let first = try #require(compost.first)
        #expect(first.displayName == "garden.md")
        #expect(first.text.contains("compost bin"))
        #expect(first.lineStart != nil && first.lineEnd != nil)
        #expect(first.page == nil)

        let timer = try await box.index.search("irrigation timer schedule",
                                               folders: [box.folder], limit: 3)
        let pdf = try #require(timer.first)
        #expect(pdf.displayName == "manual.pdf")
        #expect(pdf.page == 2)
        #expect(pdf.location == "page 2")

        let code = try await box.index.search("prime", folders: [box.folder], limit: 3)
        #expect(code.first?.displayName == "Pump.swift")
    }

    @Test func changedAndDeletedFilesAreUpdatedIncrementally() async throws {
        let box = try Self.sandbox()
        defer { box.cleanUp() }
        try box.write("a.txt", "The meeting moved to Thursday.")
        try box.write("b.txt", "Bring the projector cable.")
        _ = try await box.indexer.update(box.folder)

        // Unchanged files are not read again.
        let again = try await box.indexer.update(box.folder)
        #expect(again.indexedFiles == 2)

        // A changed file is not quoted from its stale text, even before the
        // index catches up.
        try await Task.sleep(for: .milliseconds(20))
        try box.write("a.txt", "The meeting moved to Friday afternoon.")
        let stale = try await box.index.search("Thursday", folders: [box.folder], limit: 3)
        #expect(stale.isEmpty)
        _ = try await box.indexer.update(box.folder)
        #expect(try await box.index.search("Friday", folders: [box.folder], limit: 3).count == 1)
        #expect(try await box.index.search("Thursday", folders: [box.folder], limit: 3).isEmpty)

        try FileManager.default.removeItem(at: box.root.appendingPathComponent("b.txt"))
        #expect(try await box.index.search("projector", folders: [box.folder], limit: 3).isEmpty)
        let afterDelete = try await box.indexer.update(box.folder)
        #expect(afterDelete.indexedFiles == 1)
    }

    @Test func symbolicLinksCannotEscapeTheFolder() async throws {
        let box = try Self.sandbox()
        defer { box.cleanUp() }
        let outside = box.root.deletingLastPathComponent().appendingPathComponent("Private")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try "The vault code is 7741.".write(to: outside.appendingPathComponent("secret.txt"),
                                             atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            at: box.root.appendingPathComponent("linked"), withDestinationURL: outside)
        try FileManager.default.createSymbolicLink(
            at: box.root.appendingPathComponent("secret-link.txt"),
            withDestinationURL: outside.appendingPathComponent("secret.txt"))
        try box.write("inside.txt", "Nothing secret here about the vault.")
        let status = try await box.indexer.update(box.folder)
        #expect(status.indexedFiles == 1)
        let results = try await box.index.search("vault code", folders: [box.folder], limit: 5)
        #expect(results.allSatisfy { $0.filePath.hasPrefix(box.folder.path + "/") })
        #expect(!results.contains { $0.text.contains("7741") })

        // A directory replaced by a link after indexing is refused at search.
        try box.write("docs/plan.txt", "Plan the vault refurbishment.")
        _ = try await box.indexer.update(box.folder)
        let docs = box.root.appendingPathComponent("docs")
        try FileManager.default.removeItem(at: docs)
        try FileManager.default.createSymbolicLink(at: docs, withDestinationURL: outside)
        try "Plan the vault refurbishment.".write(to: outside.appendingPathComponent("plan.txt"),
                                                  atomically: true, encoding: .utf8)
        let swapped = try await box.index.search("refurbishment", folders: [box.folder], limit: 5)
        #expect(swapped.isEmpty)
    }

    @Test func oversizedUnreadableAndExcludedFilesAreSkipped() async throws {
        var limits = AppLocalIndexLimits.standard
        limits.maximumTextFileBytes = 1_000
        let box = try Self.sandbox(limits: limits)
        defer { box.cleanUp() }
        try box.write("big.txt", String(repeating: "elephant ", count: 500))
        try box.write("small.txt", "A small elephant.")
        try box.write(".hidden/notes.txt", "hidden elephant")
        try box.write("node_modules/pkg/readme.md", "dependency elephant")
        try Data([0xff, 0xfe, 0x00, 0xd8]).write(to: box.root.appendingPathComponent("broken.txt"))
        let status = try await box.indexer.update(box.folder)
        #expect(status.indexedFiles == 1)
        #expect(status.skippedFiles == 2)
        let results = try await box.index.search("elephant", folders: [box.folder], limit: 5)
        #expect(results.map(\.displayName) == ["small.txt"])
    }

    @Test func aFileLimitIsReported() async throws {
        var limits = AppLocalIndexLimits.standard
        limits.maximumFilesPerFolder = 3
        let box = try Self.sandbox(limits: limits)
        defer { box.cleanUp() }
        for index in 0..<6 { try box.write("f\(index).txt", "note \(index)") }
        let status = try await box.indexer.update(box.folder)
        #expect(status.reachedFileLimit)
        #expect(status.indexedFiles == 3)
    }

    @Test func removingAFolderDeletesItsIndexAndSearchNeedsAFolder() async throws {
        let box = try Self.sandbox()
        defer { box.cleanUp() }
        try box.write("a.txt", "Lighthouse keeper's log.")
        _ = try await box.indexer.update(box.folder)
        try await box.index.remove(folder: box.folder.id)
        #expect(try await box.index.counts(folder: box.folder.id).indexed == 0)
        #expect(try await box.index.search("lighthouse", folders: [box.folder], limit: 3).isEmpty)
        await #expect(throws: AppLocalSearchError.noFolders) {
            _ = try await box.index.search("lighthouse", folders: [], limit: 3)
        }
    }

    @Test func indexingPausesWhileAskedAndStopsWhenCancelled() async throws {
        let box = try Self.sandbox()
        defer { box.cleanUp() }
        for index in 0..<5 { try box.write("f\(index).txt", "paragraph \(index)") }
        let pause = AppInferenceActivityFlag()
        pause.set(true)
        let task = Task { try await box.indexer.update(box.folder, shouldPause: { pause.isActive }) }
        try await Task.sleep(for: .milliseconds(700))
        #expect(try await box.index.counts(folder: box.folder.id).indexed == 0)
        task.cancel()
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        pause.set(false)
        let finished = try await box.indexer.update(box.folder)
        #expect(finished.indexedFiles == 5)
    }

    @Test func aCancelledScanPreservesAlreadyIndexedFiles() async throws {
        let box = try Self.sandbox()
        defer { box.cleanUp() }
        try box.write("kept.txt", "The lighthouse is still indexed.")
        _ = try await box.indexer.update(box.folder)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await box.indexer.update(box.folder)
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(try await box.index.counts(folder: box.folder.id).indexed == 1)
        #expect(try await box.index.search("lighthouse", folders: [box.folder], limit: 3).count == 1)
    }

    @Test func aSavedFolderBookmarkFollowsARename() throws {
        let box = try Self.sandbox()
        defer { box.cleanUp() }
        let store = AppSearchFolderStore(fileURL: box.root.deletingLastPathComponent()
            .appendingPathComponent("folders.json"))
        try store.save([box.folder])
        let renamed = box.root.deletingLastPathComponent().appendingPathComponent("Renamed")
        try FileManager.default.moveItem(at: box.root, to: renamed)
        let loaded = try #require(store.load().first)
        #expect(loaded.id == box.folder.id)
        #expect(loaded.path == canonicalPath(renamed.path))
    }

    @Test func queriesAreTreatedAsPlainWords() {
        // Operators become words; single letters are too common to rank on.
        #expect(AppLocalSearchIndex.matchExpression("NEAR(\"x\" y) OR *") == #""near" OR "or""#)
        #expect(AppLocalSearchIndex.matchExpression("??!") == nil)
        #expect(AppLocalSearchIndex.matchExpression("a 7 bb") == #""7" OR "bb""#)
    }

    @Test func theWholeDiskIsRefusedAsAFolder() {
        #expect(throws: AppLocalSearchError.tooBroad("/")) {
            _ = try AppSearchFolderStore.folder(for: URL(fileURLWithPath: "/"))
        }
    }

    @Test func theFolderListPersistsWithoutTheIndex() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-folders-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let store = AppSearchFolderStore(fileURL: file)
        let folder = AppSearchFolder(path: "/tmp/example")
        try store.save([folder])
        #expect(store.load() == [folder])
    }

    @Test func theIndexerReadsOnlyTheFileTheScanSaw() throws {
        let box = try Self.sandbox()
        defer { box.cleanUp() }
        try box.write("note.md", "plain note text")
        let secret = box.root.deletingLastPathComponent().appendingPathComponent("secret.md")
        try "outside secret".write(to: secret, atomically: true, encoding: .utf8)
        let found = try #require(box.indexer.scan(box.folder).files.first)
        #expect(found.device != nil && found.inode != nil)
        let limits = AppLocalIndexLimits.standard
        let passages = try AppLocalIndexer.passages(for: found, in: box.folder, limits: limits)
        #expect(passages.first?.3.contains("plain note text") == true)

        // The name is swapped for a symbolic link to an outside file with the
        // same size, as a racing command could do between scan and read.
        let path = URL(fileURLWithPath: found.path)
        try FileManager.default.removeItem(at: path)
        try FileManager.default.createSymbolicLink(at: path, withDestinationURL: secret)
        #expect(throws: AppLocalFileChanged.self) {
            _ = try AppLocalIndexer.passages(for: found, in: box.folder, limits: limits)
        }
        // A directory component swapped for a link is refused too.
        try box.write("dir/inner.md", "inner text")
        let inner = try #require(box.indexer.scan(box.folder).files.first { $0.path.hasSuffix("inner.md") })
        let outsideDirectory = box.root.deletingLastPathComponent().appendingPathComponent("elsewhere")
        try FileManager.default.createDirectory(at: outsideDirectory, withIntermediateDirectories: true)
        try "inner text".write(to: outsideDirectory.appendingPathComponent("inner.md"), atomically: true, encoding: .utf8)
        let directory = box.root.appendingPathComponent("dir")
        try FileManager.default.removeItem(at: directory)
        try FileManager.default.createSymbolicLink(at: directory, withDestinationURL: outsideDirectory)
        #expect(throws: AppLocalFileChanged.self) {
            _ = try AppLocalIndexer.passages(for: inner, in: box.folder, limits: limits)
        }
    }

    @Test func aReplacedOrHardLinkedFileIsNotRead() throws {
        let box = try Self.sandbox()
        defer { box.cleanUp() }
        try box.write("a.md", "original text")
        let found = try #require(box.indexer.scan(box.folder).files.first)
        let limits = AppLocalIndexLimits.standard
        // Same size, different file (new inode).
        let path = URL(fileURLWithPath: found.path)
        try FileManager.default.removeItem(at: path)
        try "replaced text".write(to: path, atomically: false, encoding: .utf8)
        #expect(throws: AppLocalFileChanged.self) {
            _ = try AppLocalIndexer.passages(for: found, in: box.folder, limits: limits)
        }
        // A hard link to an outside file is skipped, not extracted.
        let outside = box.root.deletingLastPathComponent().appendingPathComponent("outside.md")
        try "outside text".write(to: outside, atomically: true, encoding: .utf8)
        try FileManager.default.removeItem(at: path)
        try FileManager.default.linkItem(at: outside, to: path)
        var info = stat()
        #expect(lstat(path.path, &info) == 0)
        let linked = AppLocalIndexer.FoundFile(path: found.path, size: Int(info.st_size),
                                               mtime: found.mtime, device: info.st_dev, inode: info.st_ino)
        #expect(throws: AppLocalFileSkip.self) {
            _ = try AppLocalIndexer.passages(for: linked, in: box.folder, limits: limits)
        }
    }
}
