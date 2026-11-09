import AppKit
import Foundation

/// The text storage of the one answer still being generated.
///
/// Only the live answer goes through here. Completed messages are ordinary
/// views that lay out once, so this controller holds a single response rather
/// than the whole conversation: the transcript is a list of messages, and the
/// streaming fast path — append the suffix, format once at the end — applies to
/// the last of them.
@MainActor
public final class StreamingResponseDocumentController {
    public enum Mutation: Equatable {
        case none
        case rebuilt
        case appended
        case finalized
    }

    public struct UpdateResult {
        public let mutation: Mutation
        public let assistantRange: NSRange

        public init(mutation: Mutation, assistantRange: NSRange) {
            self.mutation = mutation
            self.assistantRange = assistantRange
        }
    }

    public private(set) var response = ""
    public private(set) var isFinalized = false
    public private(set) var showsPrefillPlaceholder = false
    public private(set) var assistantRange = NSRange(location: 0, length: 0)
    private var prefillPlaceholderRange: NSRange?
    private var prefillDotCount = 0
    /// How many characters of `response` are shown formatted. Paragraphs
    /// that are finished are rendered as they complete; only the one still
    /// being written stays raw, so headings and bold no longer show their
    /// `###` and `**` until the very end.
    private var formattedCharacterCount = 0

    private let renderer: ResponseMarkdownRenderer
    /// The zoom this document is set at. The streamed text is an
    /// `NSAttributedString`, so the size is baked in here as well as in the
    /// renderer, and both have to agree or the answer changes size as it
    /// finishes.
    private let scale: CGFloat

    public init(
        renderer: ResponseMarkdownRenderer? = nil,
        scale: CGFloat = 1
    ) {
        self.scale = scale
        self.renderer = renderer ?? ResponseMarkdownRenderer(scale: scale)
    }

    public static func clampedRanges(
        _ ranges: [NSRange],
        toLength length: Int
    ) -> [NSRange] {
        ranges.map { range in
            let location = min(max(range.location, 0), length)
            let available = max(0, length - location)
            return NSRange(location: location, length: min(max(range.length, 0), available))
        }
    }

    public static func shouldRunPrefillAnimation(
        response: String,
        isTerminal: Bool,
        requested: Bool
    ) -> Bool {
        requested && response.isEmpty && !isTerminal
    }

    @discardableResult
    public func synchronize(
        storage: NSMutableAttributedString,
        response: String,
        isTerminal: Bool,
        showsPrefillPlaceholder: Bool = false
    ) -> UpdateResult {
        let responseChanged = !response.utf8.elementsEqual(self.response.utf8)
        let displaysPrefillPlaceholder = Self.shouldRunPrefillAnimation(
            response: response,
            isTerminal: isTerminal,
            requested: showsPrefillPlaceholder)
        // A byte prefix, not `hasPrefix`: canonically equivalent text can
        // differ in length, and the append below slices at the old length.
        let needsRebuild = !response.utf8.starts(with: self.response.utf8)
            || (isFinalized && !isTerminal)
            || displaysPrefillPlaceholder != self.showsPrefillPlaceholder

        var mutation: Mutation = .none
        if needsRebuild
            || storage.length == 0
                && (!response.isEmpty || displaysPrefillPlaceholder) {
            rebuild(
                storage: storage,
                response: response,
                showsPrefillPlaceholder: displaysPrefillPlaceholder)
            mutation = .rebuilt
        } else if response.utf8.count > self.response.utf8.count {
            // `response` extends the previous one here, so the new part starts
            // at its byte length; counting characters ran on every frame.
            let delta = String(response[response.utf8.index(
                response.utf8.startIndex, offsetBy: self.response.utf8.count)...])
            storage.append(NSAttributedString(
                string: delta,
                attributes: Self.responseAttributes(scale: scale)))
            assistantRange.length += (delta as NSString).length
            mutation = .appended
            if !isTerminal, delta.contains("\n"),
               formatFinishedParagraphs(storage: storage, response: response) {
                mutation = .rebuilt
            }
        }

        self.response = response
        self.showsPrefillPlaceholder = displaysPrefillPlaceholder

        if isTerminal && (!isFinalized || responseChanged) {
            let rendered = renderer.render(response).attributedString
            storage.replaceCharacters(in: assistantRange, with: rendered)
            assistantRange.length = rendered.length
            isFinalized = true
            mutation = .finalized
        } else if !isTerminal {
            isFinalized = false
        }

        return UpdateResult(mutation: mutation, assistantRange: assistantRange)
    }

    @discardableResult
    public func advancePrefillAnimation(
        storage: NSMutableAttributedString
    ) -> Bool {
        guard showsPrefillPlaceholder, var range = prefillPlaceholderRange else {
            return false
        }
        prefillDotCount = (prefillDotCount + 1) % 4
        let replacement = NSAttributedString(
            string: Self.prefillPlaceholder(dotCount: prefillDotCount),
            attributes: Self.prefillPlaceholderAttributes(scale: scale))
        storage.replaceCharacters(in: range, with: replacement)
        range.length = replacement.length
        prefillPlaceholderRange = range
        return true
    }

    /// Renders everything up to the last finished paragraph and keeps the rest
    /// raw. Returns whether the storage changed. A paragraph inside an open
    /// code fence or display math isn't finished, since rendering half of one
    /// would show it wrong until the rest arrived.
    private func formatFinishedParagraphs(
        storage: NSMutableAttributedString,
        response: String
    ) -> Bool {
        let boundary = Self.finishedParagraphBoundary(in: response)
        guard boundary > formattedCharacterCount else { return false }
        let finished = String(response.prefix(boundary))
            .trimmingCharacters(in: .newlines)
        let document = NSMutableAttributedString(
            attributedString: renderer.render(finished).attributedString)
        document.append(NSAttributedString(
            string: "\n" + String(response.dropFirst(boundary)),
            attributes: Self.responseAttributes(scale: scale)))
        storage.replaceCharacters(in: assistantRange, with: document)
        assistantRange.length = document.length
        formattedCharacterCount = boundary
        return true
    }

    /// The character offset just past the last blank line that closes a
    /// paragraph outside code fences and `$$` math, or 0 when there is none.
    static func finishedParagraphBoundary(in response: String) -> Int {
        var boundary = 0
        var offset = 0
        var inFence = false
        var inMath = false
        var previousLineBlank = false
        for line in response.split(separator: "\n", omittingEmptySubsequences: false)
            .dropLast() {
            offset += line.count + 1
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") { inFence.toggle() }
            if !inFence, trimmed.components(separatedBy: "$$").count % 2 == 0 {
                inMath.toggle()
            }
            let blank = trimmed.isEmpty
            if blank, !previousLineBlank, !inFence, !inMath, offset > 1 {
                boundary = offset
            }
            previousLineBlank = blank
        }
        return boundary
    }

    private func rebuild(
        storage: NSMutableAttributedString,
        response: String,
        showsPrefillPlaceholder: Bool
    ) {
        formattedCharacterCount = 0
        let document = NSMutableAttributedString()
        assistantRange = NSRange(location: 0, length: 0)
        document.append(NSAttributedString(
            string: response,
            attributes: Self.responseAttributes(scale: scale)))
        assistantRange.length = (response as NSString).length
        prefillDotCount = 0
        prefillPlaceholderRange = nil
        if showsPrefillPlaceholder {
            let placeholder = NSAttributedString(
                string: Self.prefillPlaceholder(dotCount: prefillDotCount),
                attributes: Self.prefillPlaceholderAttributes(scale: scale))
            prefillPlaceholderRange = NSRange(
                location: document.length,
                length: placeholder.length)
            document.append(placeholder)
        }
        storage.setAttributedString(document)
        isFinalized = false
    }

    public static func responseAttributes(scale: CGFloat = 1) -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3 * scale
        style.paragraphSpacing = 6 * scale
        return [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize * scale),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: style,
        ]
    }

    private static func prefillPlaceholderAttributes(scale: CGFloat = 1) -> [NSAttributedString.Key: Any] {
        var attributes = responseAttributes(scale: scale)
        attributes[.foregroundColor] = NSColor.secondaryLabelColor
        return attributes
    }

    private static func prefillPlaceholder(dotCount: Int) -> String {
        "Processing your prompt" + String(repeating: ".", count: dotCount)
    }
}
