import Foundation
import PDFKit
import UniformTypeIdentifiers

/// Text pulled out of a file the user attached, with enough context to show what
/// happened before anything is sent.
///
/// Documents are not a model capability: every model reads text, so a document
/// becomes prompt text rather than a companion pack. What matters is that the
/// user can see how much text arrived, because a long file will not fit the
/// context window and silently trimming it would answer a different question
/// than the one asked.
public struct ExtractedDocument: Equatable, Sendable {
    public let displayName: String
    public let text: String
    /// Characters divided by four. Deliberately an estimate, and labelled as one
    /// wherever it is shown: the real count depends on the model's tokenizer.
    public var estimatedTokens: Int { max(1, text.count / 4) }

    public init(displayName: String, text: String) {
        self.displayName = displayName
        self.text = text
    }
}

public enum DocumentExtractionError: Error, Equatable, CustomStringConvertible {
    case unsupportedType(String)
    case unreadable(String)
    case empty(String)

    public var description: String {
        switch self {
        case .unsupportedType(let ext):
            return "TUFF cannot read .\(ext) files. Plain text and source files, "
                + "Markdown, CSV, JSON, YAML, and PDF are supported."
        case .unreadable(let name):
            return "\(name) could not be read."
        case .empty(let name):
            return "\(name) has no extractable text. A scanned PDF needs OCR first."
        }
    }
}

public enum DocumentTextExtractor {
    public static let supportedExtensions: Set<String> = [
        "txt", "md", "markdown", "text", "csv", "tsv", "json", "yaml", "yml", "pdf",
    ]

    /// What the open panel and a drop are allowed to offer, derived from the
    /// extensions above rather than listed a second time.
    ///
    /// The picker used to carry its own hand-written list, which omitted
    /// Markdown and TSV: the extractor accepted a `.md` file that the panel had
    /// already greyed out, so the one format most worth attaching could not be
    /// chosen. One list, and the two cannot drift apart again.
    public static let allowedContentTypes: [UTType] = {
        var types = supportedExtensions.compactMap {
            UTType(filenameExtension: $0)
        }
        // A file with no extension, or one the system has no declaration for,
        // is still plain text as far as the extractor is concerned.
        if !types.contains(.plainText) { types.append(.plainText) }
        return types
    }()

    /// Whether a dropped, pasted, or chosen URL is one this extractor will
    /// accept.
    ///
    /// The named extensions are the documented set; beyond them, anything the
    /// system calls *plain* text is read too. The open panel offers
    /// `public.plain-text` because `.txt` maps to it, so a `.swift`, `.log`, or
    /// `.py` file was selectable and then refused on the grounds that TUFF
    /// "cannot read" it — a file it reads perfectly well. The panel and this
    /// now agree on the same set.
    ///
    /// `public.plain-text` and not `public.text`: RTF, HTML and XML conform to
    /// the latter while being markup containers, and reading one as UTF-8 hands
    /// the model `{\rtf1\ansi…` and calls it the document. JSON and YAML also
    /// fail the plain-text test, which is why they stay named above.
    public static func canExtract(from url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        if ext.isEmpty || supportedExtensions.contains(ext) { return true }
        guard let type = UTType(filenameExtension: ext)
        else { return false }
        return type.conforms(to: .plainText)
    }

    public static func extract(from url: URL) throws -> ExtractedDocument {
        let name = url.lastPathComponent
        let ext = url.pathExtension.lowercased()
        guard canExtract(from: url) else {
            throw DocumentExtractionError.unsupportedType(ext)
        }

        let text: String
        if ext == "pdf" {
            guard let document = PDFDocument(url: url) else {
                throw DocumentExtractionError.unreadable(name)
            }
            var pages: [String] = []
            for index in 0..<document.pageCount {
                if let page = document.page(at: index), let content = page.string {
                    pages.append(content)
                }
            }
            text = pages.joined(separator: "\n\n")
        } else {
            guard let content = try? String(contentsOf: url, encoding: .utf8) else {
                throw DocumentExtractionError.unreadable(name)
            }
            text = content
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw DocumentExtractionError.empty(name) }
        return ExtractedDocument(displayName: name, text: trimmed)
    }

    /// Text in sections that keep their location: one per PDF page, or one
    /// for a text file. Used by local search, which cites pages.
    public struct Section: Equatable, Sendable {
        /// One-based PDF page, or nil for a text file.
        public let page: Int?
        public let text: String
    }

    public static func extractSections(from url: URL,
                                       maximumPDFPages: Int = 1_000) throws -> [Section] {
        let name = url.lastPathComponent
        guard canExtract(from: url) else {
            throw DocumentExtractionError.unsupportedType(url.pathExtension.lowercased())
        }
        if url.pathExtension.lowercased() == "pdf" {
            guard let document = PDFDocument(url: url) else {
                throw DocumentExtractionError.unreadable(name)
            }
            var sections: [Section] = []
            for index in 0..<min(document.pageCount, maximumPDFPages) {
                guard let content = document.page(at: index)?.string?
                    .trimmingCharacters(in: .whitespacesAndNewlines), !content.isEmpty else { continue }
                sections.append(Section(page: index + 1, text: content))
            }
            guard !sections.isEmpty else { throw DocumentExtractionError.empty(name) }
            return sections
        }
        guard let content = try? String(contentsOf: url, encoding: .utf8) else {
            throw DocumentExtractionError.unreadable(name)
        }
        guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DocumentExtractionError.empty(name)
        }
        return [Section(page: nil, text: content)]
    }

    /// Like `extractSections(from:)` for bytes the caller already read through
    /// a descriptor it verified; `name` only supplies the type.
    public static func extractSections(data: Data, name: URL,
                                       maximumPDFPages: Int = 1_000) throws -> [Section] {
        let displayName = name.lastPathComponent
        guard canExtract(from: name) else {
            throw DocumentExtractionError.unsupportedType(name.pathExtension.lowercased())
        }
        if name.pathExtension.lowercased() == "pdf" {
            guard let document = PDFDocument(data: data) else {
                throw DocumentExtractionError.unreadable(displayName)
            }
            var sections: [Section] = []
            for index in 0..<min(document.pageCount, maximumPDFPages) {
                guard let content = document.page(at: index)?.string?
                    .trimmingCharacters(in: .whitespacesAndNewlines), !content.isEmpty else { continue }
                sections.append(Section(page: index + 1, text: content))
            }
            guard !sections.isEmpty else { throw DocumentExtractionError.empty(displayName) }
            return sections
        }
        guard let content = String(data: data, encoding: .utf8) else {
            throw DocumentExtractionError.unreadable(displayName)
        }
        guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DocumentExtractionError.empty(displayName)
        }
        return [Section(page: nil, text: content)]
    }

    /// The document as it is added to a prompt. The delimiters are plain text so
    /// every model reads them the same way, and the name is kept so the answer
    /// can refer to the file.
    public static func promptBlock(for document: ExtractedDocument) -> String {
        documentPromptBlock(displayName: document.displayName, text: document.text)
    }
}
