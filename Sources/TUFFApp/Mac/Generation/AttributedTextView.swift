import AppKit
import SwiftUI
import TUFFMacPresentation

/// An `NSTextView` that has no scroll view of its own and reports the height it
/// needs instead.
///
/// The conversation is one scrollable surface. A message that carried its own
/// scroll view put a scroller inside a scroller — the wheel stopped at whichever
/// message the pointer happened to be over — so every message here sizes to its
/// content and lets the transcript do the scrolling.
final class MessageTextView: NSTextView {
    var onHeightChange: ((CGFloat) -> Void)?
    private var reportedHeight: CGFloat = -1
    private var measuredWidth: CGFloat = -1

    func fittingHeight(width: CGFloat) -> CGFloat? {
        guard width.isFinite, width > 0,
              let container = textContainer, let layoutManager else { return nil }
        container.containerSize = NSSize(width: width, height: .greatestFiniteMagnitude)
        layoutManager.ensureLayout(for: container)
        layoutCodeCopyButtons()
        return max(1, ceil(layoutManager.usedRect(for: container).height))
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        // A narrower window rewraps the text, and nothing else asks the view to
        // measure again: without this, resizing left every message at the
        // height its old width needed.
        guard abs(newSize.width - measuredWidth) > 0.5 else { return }
        defer { layoutCodeCopyButtons() }
        measuredWidth = newSize.width
        reportHeight()
    }

    /// Measures and, when it changed, reports. The report is deferred because
    /// callers run inside SwiftUI's layout pass, where writing state directly is
    /// a "modifying state during view update" violation.
    func reportHeight() {
        guard let container = textContainer, let layoutManager else { return }
        layoutManager.ensureLayout(for: container)
        layoutCodeCopyButtons()
        let measured = ceil(layoutManager.usedRect(for: container).height)
        guard measured > 0, abs(measured - reportedHeight) > 0.5 else { return }
        reportedHeight = measured
        let handler = onHeightChange
        DispatchQueue.main.async { handler?(measured) }
    }

    // MARK: Code blocks

    /// One Copy button per code card, at the right of its header row. Always
    /// shown, as in other chat apps; a hover-only button was easy to miss and
    /// a small target.
    private var codeCopyButtons: [CodeCopyButton] = []

    override func menu(for event: NSEvent) -> NSMenu? {
        let menu = super.menu(for: event) ?? NSMenu()
        if let block = codeBlock(at: convert(event.locationInWindow, from: nil)) {
            let item = NSMenuItem(title: "Copy Code", action: #selector(copyCodeFromMenu(_:)),
                                  keyEquivalent: "")
            item.target = self
            item.representedObject = block.code
            menu.insertItem(item, at: 0)
            menu.insertItem(.separator(), at: 1)
        }
        return menu
    }

    private func codeBlock(at point: NSPoint) -> (range: NSRange, code: String)? {
        guard let textStorage, textStorage.length > 0, let layoutManager, let textContainer else {
            return nil
        }
        var fraction: CGFloat = 0
        let glyph = layoutManager.glyphIndex(for: point, in: textContainer,
                                             fractionOfDistanceThroughGlyph: &fraction)
        let line = layoutManager.lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil)
        guard line.minY <= point.y, point.y <= line.maxY else { return nil }
        let character = layoutManager.characterIndexForGlyph(at: glyph)
        guard character < textStorage.length,
              textStorage.attribute(.tuffCodeBlock, at: character, effectiveRange: nil) != nil else {
            return nil
        }
        return ResponseMarkdownRenderer.codeBlocks(in: textStorage)
            .first { NSLocationInRange(character, $0.range) }
    }

    /// Puts a Copy button in each card's header. Called whenever the text or
    /// width changes, after layout.
    func layoutCodeCopyButtons() {
        let cards = (layoutManager as? CodeBlockLayoutManager)?.codeCards() ?? []
        while codeCopyButtons.count > cards.count { codeCopyButtons.removeLast().removeFromSuperview() }
        while codeCopyButtons.count < cards.count {
            let button = CodeCopyButton()
            addSubview(button)
            codeCopyButtons.append(button)
        }
        for (card, button) in zip(cards, codeCopyButtons) {
            button.code = card.code
            button.fontSize = card.codeFont.pointSize
            let size = button.fittingSize
            let inset = card.codeFont.pointSize * ResponseMarkdownRenderer.codeInset / 12.5
            button.frame = NSRect(
                x: card.rect.maxX + textContainerOrigin.x - size.width - inset + 8,
                y: card.rect.minY + textContainerOrigin.y + (card.headerHeight - size.height) / 2,
                width: size.width, height: size.height)
        }
    }

    @objc private func copyCodeFromMenu(_ sender: NSMenuItem) {
        guard let code = sender.representedObject as? String else { return }
        copyToPasteboard(code)
    }

    private func copyToPasteboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    /// Applies the standard message text-view configuration.
    static func make() -> MessageTextView {
        // TextKit 1 with the layout manager that draws code cards.
        let storage = NSTextStorage()
        let layoutManager = CodeBlockLayoutManager()
        storage.addLayoutManager(layoutManager)
        let container = NSTextContainer(size: NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude))
        layoutManager.addTextContainer(container)
        let textView = MessageTextView(frame: .zero, textContainer: container)
        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = true
        textView.drawsBackground = false
        textView.textContainerInset = .zero
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.heightTracksTextView = false
        textView.isAutomaticLinkDetectionEnabled = false
        textView.isAutomaticDataDetectionEnabled = false
        // SwiftUI owns the frame. AppKit resizing it during measurement can
        // reset the container to the previous width and produce a zero height.
        textView.isVerticallyResizable = false
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.setContentHuggingPriority(.defaultHigh, for: .vertical)
        return textView
    }
}

/// A code card's Copy control: icon and label, a full-height hit area, and a
/// brief "Copied" confirmation.
final class CodeCopyButton: NSButton {
    var code = ""
    var fontSize: CGFloat = 12 { didSet { if fontSize != oldValue { refresh(copied: false) } } }
    private var resetWork: DispatchWorkItem?

    init() {
        super.init(frame: .zero)
        isBordered = false
        bezelStyle = .regularSquare
        imagePosition = .imageLeading
        imageHugsTitle = true
        contentTintColor = TUFFDesign.secondaryText
        target = self
        action = #selector(copyCode)
        setAccessibilityLabel("Copy code")
        refresh(copied: false)
    }

    required init?(coder: NSCoder) { fatalError() }

    override var fittingSize: NSSize {
        let base = super.fittingSize
        // Padding around the label makes the whole pill the target.
        return NSSize(width: ceil(base.width) + 16, height: max(ceil(base.height) + 10, 26))
    }

    override func resetCursorRects() { addCursorRect(bounds, cursor: .pointingHand) }

    override func draw(_ dirtyRect: NSRect) {
        if isHighlighted {
            NSColor.labelColor.withAlphaComponent(0.08).setFill()
            NSBezierPath(roundedRect: bounds, xRadius: 7, yRadius: 7).fill()
        }
        super.draw(dirtyRect)
    }

    private func refresh(copied: Bool) {
        let configuration = NSImage.SymbolConfiguration(pointSize: fontSize - 1, weight: .medium)
        image = NSImage(systemSymbolName: copied ? "checkmark" : "doc.on.doc",
                        accessibilityDescription: nil)?.withSymbolConfiguration(configuration)
        attributedTitle = NSAttributedString(string: copied ? " Copied" : " Copy", attributes: [
            .font: NSFont.systemFont(ofSize: fontSize, weight: .medium),
            .foregroundColor: TUFFDesign.secondaryText,
        ])
    }

    @objc private func copyCode() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)
        refresh(copied: true)
        resetWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.refresh(copied: false) }
        resetWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5, execute: work)
    }
}

/// A static, selectable attributed-text block that reports its own height.
///
/// Rendered responses carry math as `NSTextAttachment`, which SwiftUI's `Text`
/// cannot draw, so a completed message still needs a text view. This one holds
/// a single message: it lays out once and never streams, which is what lets the
/// transcript be a list of messages.
struct AttributedTextView: NSViewRepresentable {
    let attributed: NSAttributedString
    /// Handles a click on a link the renderer made clickable: only validated
    /// citations are. Nil leaves links inert.
    var onLink: ((URL) -> Void)?

    final class Coordinator: NSObject, NSTextViewDelegate {
        var onLink: ((URL) -> Void)?
        /// The string last applied. The renderer caches its results, so an
        /// unchanged message hands back the same object and the comparison
        /// below is skipped on every token of a reply elsewhere in the chat.
        weak var applied: NSAttributedString?

        func textView(_ textView: NSTextView, clickedOnLink link: Any, at charIndex: Int) -> Bool {
            guard let url = link as? URL ?? (link as? String).flatMap(URL.init(string:)) else {
                return true
            }
            onLink?(url)
            // Handled either way: nothing is passed on to open by default.
            return true
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> MessageTextView {
        let textView = MessageTextView.make()
        textView.delegate = context.coordinator
        return textView
    }

    func updateNSView(_ textView: MessageTextView, context: Context) {
        context.coordinator.onLink = onLink
        guard context.coordinator.applied !== attributed else { return }
        if textView.textStorage?.isEqual(to: attributed) != true {
            textView.textStorage?.setAttributedString(attributed)
        }
        context.coordinator.applied = attributed
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: MessageTextView, context: Context) -> CGSize? {
        let width = proposal.width ?? nsView.bounds.width
        guard let height = nsView.fittingHeight(width: width) else { return nil }
        return CGSize(width: width, height: height)
    }
}
