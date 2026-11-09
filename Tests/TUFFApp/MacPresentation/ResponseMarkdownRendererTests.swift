import AppKit
import Foundation
import Testing
@testable import TUFFMacPresentation

@MainActor
@Suite struct ResponseMarkdownRendererTests {
    /// A composed "é" replaced by a decomposed "e" + accent is equal as a
    /// Swift string but not byte for byte; the streaming append must rebuild
    /// rather than slice the new text at the old byte length.
    @Test func streamingAppendRebuildsWhenOnlyCanonicallyEqual() {
        let controller = StreamingResponseDocumentController()
        let storage = NSMutableAttributedString()
        controller.synchronize(storage: storage, response: "caf\u{E9}", isTerminal: false)
        controller.synchronize(storage: storage, response: "cafe\u{301} au lait", isTerminal: false)
        #expect(storage.string.utf8.elementsEqual("cafe\u{301} au lait".utf8))
        // An accent arriving after its letter is a plain append.
        controller.synchronize(storage: storage, response: "cafe\u{301} au lait e", isTerminal: false)
        controller.synchronize(storage: storage, response: "cafe\u{301} au lait e\u{301}", isTerminal: false)
        #expect(storage.string.utf8.elementsEqual("cafe\u{301} au lait e\u{301}".utf8))
    }

    @Test func eachCodeBlockCanBeCopiedOnItsOwn() {
        let source = """
        Install it:

        ```sh
        brew install tuff
        tuff serve
        ```

        Then call `tuff prompt`:

        ```python
        print("hi")
        ```
        """
        let rendered = ResponseMarkdownRenderer().render(source).attributedString
        let blocks = ResponseMarkdownRenderer.codeBlocks(in: rendered)
        #expect(blocks.map(\.code) == ["brew install tuff\ntuff serve", "print(\"hi\")"])
        // Each block is one card: room for the header above its first line
        // only, and the language it was fenced with.
        func spacingBefore(at index: Int) -> CGFloat {
            (rendered.attribute(.paragraphStyle, at: index, effectiveRange: nil)
                as? NSParagraphStyle)?.paragraphSpacingBefore ?? 0
        }
        func language(at index: Int) -> String? {
            rendered.attribute(.tuffCodeLanguage, at: index, effectiveRange: nil) as? String
        }
        let first = blocks[0].range
        let secondLine = (rendered.string as NSString).range(of: "tuff serve").location
        #expect(spacingBefore(at: first.location) > 0)
        #expect(spacingBefore(at: secondLine) == 0)
        #expect(language(at: first.location) == "sh")
        #expect(language(at: blocks[1].range.location) == "python")
        // Markup inside a fence is code, not raw HTML in the answer.
        let html = ResponseMarkdownRenderer().render("Page:\n\n```html\n<p>hi</p>\n```")
        #expect(!html.usedFallback)
        #expect(ResponseMarkdownRenderer.codeBlocks(in: html.attributedString).map(\.code) == ["<p>hi</p>"])
        // Blank lines the author put at the end of a block are kept.
        let padded = ResponseMarkdownRenderer().render("```\nfirst\n\n\n```").attributedString
        let paddedCode = ResponseMarkdownRenderer.codeBlocks(in: padded).first?.code
        #expect(paddedCode?.hasPrefix("first\n") == true)
        // Inline code is not a block.
        #expect(!blocks.contains { $0.code.contains("tuff prompt") })
    }

    @Test func rendersSupportedMarkdownWithNativeAttributes() throws {
        let source = """
        # Heading

        A **bold** and *italic* sentence with ~~obsolete~~ text, `inlineCode`, and a [link](https://example.com).

        - first
        - second

        > quoted text

        ```swift
        let answer = 42
        ```

        ---
        """

        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text.contains("Heading"))
        #expect(text.contains("bold"))
        #expect(text.contains("italic"))
        #expect(text.contains("•\tfirst\n•\tsecond"))
        #expect(text.contains("│\tquoted text"))
        #expect(text.contains("let answer = 42"))
        #expect(text.contains("────────────────"))
        #expect(!text.contains("**"))
        #expect(!text.contains("```"))

        let linkRange = (text as NSString).range(of: "link")
        #expect(result.attributedString.attribute(.link, at: linkRange.location,
                                                  effectiveRange: nil) == nil)
        let linkColor = result.attributedString.attribute(
            .foregroundColor, at: linkRange.location, effectiveRange: nil) as? NSColor
        #expect(linkColor?.isEqual(NSColor.linkColor) == true)
        #expect(result.attributedString.attribute(.underlineStyle,
                                                  at: linkRange.location,
                                                  effectiveRange: nil) as? Int
            == NSUnderlineStyle.single.rawValue)

        let codeRange = (text as NSString).range(of: "inlineCode")
        let codeFont = try #require(result.attributedString.attribute(
            .font, at: codeRange.location, effectiveRange: nil) as? NSFont)
        #expect(codeFont.fontDescriptor.symbolicTraits.contains(.monoSpace))
        #expect(result.attributedString.attribute(
            .backgroundColor, at: codeRange.location, effectiveRange: nil) != nil)

        let strikeRange = (text as NSString).range(of: "obsolete")
        #expect(result.attributedString.attribute(
            .strikethroughStyle, at: strikeRange.location, effectiveRange: nil) != nil)
    }

    /// Every message re-renders each time a token arrives elsewhere in the
    /// chat, so an unchanged message must come back from the cache as the same
    /// object rather than being parsed again.
    @Test func renderingTheSameTextReturnsTheCachedResult() {
        let renderer = ResponseMarkdownRenderer()
        let source = "# Cached \(UUID().uuidString)\n\n**bold** and `code`"
        let first = renderer.render(source).attributedString
        #expect(renderer.render(source).attributedString === first)
        #expect(ResponseMarkdownRenderer(scale: 2).render(source).attributedString !== first)
    }

    @Test func unfinishedFenceStillGetsItsCodeCard() {
        // An answer cut off inside a code block: the text before it keeps
        // its formatting and the code its card, as if the fence had closed.
        let source = "**Before**\n\n```python\nprint('unfinished')"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(!result.attributedString.string.contains("```"))
        #expect(!result.attributedString.string.contains("**"))
        let blocks = ResponseMarkdownRenderer.codeBlocks(in: result.attributedString)
        #expect(blocks.map(\.code) == ["print('unfinished')"])
    }

    @Test func tablesRenderAsTextTablesAndKeepTheRestFormatted() throws {
        let source = """
        **Numbers**

        ```python
        print(2)
        ```

        | Prime | Note |
        | :--- | ---: |
        | 2 | `even` |
        | 3 |  |

        After.
        """
        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString

        #expect(!result.usedFallback)
        #expect(!text.string.contains("|"))
        #expect(!text.string.contains("```"))
        #expect(ResponseMarkdownRenderer.codeBlocks(in: text).map(\.code) == ["print(2)"])

        func cell(_ string: String) throws -> NSTextTableBlock {
            let range = (text.string as NSString).range(of: string)
            try #require(range.location != NSNotFound)
            let style = try #require(text.attribute(.paragraphStyle, at: range.location,
                                                    effectiveRange: nil) as? NSParagraphStyle)
            return try #require(style.textBlocks.first as? NSTextTableBlock)
        }
        let header = try cell("Prime")
        let note = try cell("even")
        #expect(header.table === note.table)
        #expect(header.table.numberOfColumns == 2)
        #expect((header.startingRow, header.startingColumn) == (0, 0))
        #expect((note.startingRow, note.startingColumn) == (1, 1))
        #expect(header.backgroundColor != nil)
        #expect(note.backgroundColor == nil)
        // A table this small is as wide as what is in it, not the whole line.
        #expect(header.contentWidthValueType == .absoluteValueType)
        #expect(header.contentWidth > 0 && header.contentWidth < 120)
        #expect(header.table.contentWidthValueType == .absoluteValueType)

        // Every cell has a paragraph of its own, the empty one included: two
        // columns by three rows.
        var cells = Set<String>()
        text.enumerateAttribute(.paragraphStyle, in: NSRange(location: 0, length: text.length)) { value, _, _ in
            guard let block = (value as? NSParagraphStyle)?.textBlocks.first as? NSTextTableBlock else { return }
            cells.insert("\(block.startingRow),\(block.startingColumn)")
        }
        #expect(cells.count == 6)
        #expect(try #require(text.string.range(of: "After.")) != nil)
        let after = (text.string as NSString).range(of: "After.")
        let afterStyle = text.attribute(.paragraphStyle, at: after.location, effectiveRange: nil) as? NSParagraphStyle
        #expect(afterStyle?.textBlocks.isEmpty != false)
        // Copied as plain text, the layout-only characters are gone.
        let plain = ResponseMarkdownRenderer().plainText(source)
        #expect(!plain.contains("\u{200B}"))
        #expect(plain.contains("Prime\tNote\n2\teven\n3\t\n"))
    }

    @Test func lineBreakTagsBreakLinesOutsideCodeOnly() {
        let source = "| A |\n|---|\n| one<br>two |\n\n`<br>` stays.\n\n```html\n<br/>\n```"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string.contains("one\u{2028}two"))
        #expect(result.attributedString.string.contains("<br> stays."))
        #expect(ResponseMarkdownRenderer.codeBlocks(in: result.attributedString).map(\.code) == ["<br/>"])

        // Code the Markdown parser knows as code, however it is fenced.
        for source in ["``<br>`` stays", "~~~\n<br>\n~~~", "> ```html\n> <br>\n> ```"] {
            let rendered = ResponseMarkdownRenderer().render(source)
            #expect(!rendered.usedFallback)
            #expect(rendered.attributedString.string.contains("<br>"))
            #expect(!rendered.attributedString.string.contains("\u{2028}"))
        }
    }

    @Test func aTagInASentenceIsShownAsWrittenWithoutLosingTheFormatting() {
        // A generic type in prose is inline HTML to the parser. It used to
        // turn the whole answer back into raw Markdown.
        let result = ResponseMarkdownRenderer().render("**Returns** a Vec<String> of names.")

        #expect(!result.usedFallback)
        #expect(result.attributedString.string == "Returns a Vec<String> of names.")
    }

    @Test func anOpenFenceOfAnyLengthIsCodeToItsEnd() {
        for source in ["````python\nprint(1)", "~~~python\nprint(1)", "Text with ``` inline.\n\n```python\nprint(1)"] {
            let result = ResponseMarkdownRenderer().render(source)
            #expect(!result.usedFallback)
            #expect(ResponseMarkdownRenderer.codeBlocks(in: result.attributedString).map(\.code) == ["print(1)"])
        }
    }

    @Test func unsupportedHTMLAndImageStayReadableAsRawText() {
        let renderer = ResponseMarkdownRenderer()
        let samples = [
            "<div>Never execute this</div>",
            "![remote](https://example.com/image.png)",
        ]

        for source in samples {
            let result = renderer.render(source)
            #expect(result.usedFallback)
            #expect(result.attributedString.string == source)
        }
    }

    @Test func latexRendersAsNativeMathAlongsideMarkdown() throws {
        let source = "Cosine is **$\\frac{u \\cdot v}{\\lVert u\\rVert \\lVert v\\rVert}$**."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        let attachmentRange = (result.attributedString.string as NSString)
            .range(of: "\u{fffc}")
        try #require(attachmentRange.location != NSNotFound)
        let attachment = try #require(result.attributedString.attribute(
            .attachment,
            at: attachmentRange.location,
            effectiveRange: nil) as? NSTextAttachment)
        #expect(attachment.image?.size.width ?? 0 > 0)
        #expect(attachment.image?.size.height ?? 0 > 0)
        #expect(attachment.image?.accessibilityDescription?.contains("\\frac") == true)
        #expect(result.attributedString.attribute(
            .tuffLaTeXSource,
            at: attachmentRange.location,
            effectiveRange: nil) as? String == "$\\frac{u \\cdot v}{\\lVert u\\rVert \\lVert v\\rVert}$")
    }

    @Test func supportsDisplayAndBracketedLaTeXDelimiters() {
        let source = """
        Inline \\(x^2 + y^2 = z^2\\).

        $$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$

        \\[\\int_0^1 x^2 \\, dx = \\frac{1}{3}\\]
        """
        let result = ResponseMarkdownRenderer().render(source)
        let attachments = result.attributedString.string.filter { $0 == "\u{fffc}" }

        #expect(!result.usedFallback)
        #expect(attachments.count == 3)
    }

    @Test func codeAndEscapedCurrencyNeverBecomeMath() {
        let source = """
        `let price = "$5"`

        ```text
        $x^2$
        ```

        A ticket costs \\$20.
        """
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.attributedString.string.contains("\u{fffc}"))
        #expect(result.attributedString.string.contains("$x^2$"))
        #expect(result.attributedString.string.contains("$20"))
    }

    @Test func malformedLaTeXFallsBackToItsExactSource() {
        let source = "Before $\\frac{1}{2$ after"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(result.usedFallback)
        #expect(result.attributedString.string == source)
    }

    @Test func boldOnlyModelHeadingStaysOnItsOwnLine() {
        let source = "**Origins**\nFieldfares arrive from northern Europe."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string
            == "Origins\n\nFieldfares arrive from northern Europe.")
    }
}

@MainActor
@Suite struct StreamingResponseDocumentControllerTests {
    @Test func appAccentMatchesProductRGB() {
        let color = TUFFMacTheme.accentNSColor
            .usingColorSpace(.sRGB)
        #expect(color != nil)
        #expect(abs((color?.redComponent ?? 0) - 111.0 / 255.0) < 0.000_001)
        #expect(abs((color?.greenComponent ?? 0) - 77.0 / 255.0) < 0.000_001)
        #expect(abs((color?.blueComponent ?? 0) - 255.0 / 255.0) < 0.000_001)
    }

    @Test func appAccentKeepsWhiteControlLabelsReadable() {
        let components = [111.0, 77.0, 255.0].map { value in
            let channel = value / 255.0
            return channel <= 0.03928
                ? channel / 12.92
                : pow((channel + 0.055) / 1.055, 2.4)
        }
        let luminance = 0.2126 * components[0]
            + 0.7152 * components[1]
            + 0.0722 * components[2]
        let contrast = 1.05 / (luminance + 0.05)

        #expect(contrast >= 4.5)
    }

    /// The streaming fast path: the first token rebuilds, every token after it
    /// appends only what is new. The document holds the answer alone — the
    /// prompt is a separate view now, so nothing else may end up in here.
    @Test func rebuildsThenAppendsOnlyNewResponseSuffix() {
        let storage = NSMutableAttributedString()
        let controller = StreamingResponseDocumentController()

        let first = controller.synchronize(
            storage: storage, response: "Hel", isTerminal: false)
        let second = controller.synchronize(
            storage: storage, response: "Hello", isTerminal: false)

        #expect(first.mutation == .rebuilt)
        #expect(second.mutation == .appended)
        #expect(storage.string == "Hello")
        #expect(controller.response == "Hello")
        #expect((storage.string as NSString).substring(with: second.assistantRange)
            == "Hello")
    }

    /// Finished paragraphs are formatted while the answer streams; the one
    /// still being written, and anything inside an open code fence, stays raw.
    @Test func finishedParagraphsAreFormattedWhileStreaming() {
        let storage = NSMutableAttributedString()
        let controller = StreamingResponseDocumentController()
        _ = controller.synchronize(storage: storage, response: "### Day", isTerminal: false)
        #expect(storage.string == "### Day")

        let formatted = controller.synchronize(
            storage: storage, response: "### Day 1\n\n**Morning** walk", isTerminal: false)
        #expect(formatted.mutation == .rebuilt)
        #expect(storage.string.hasPrefix("Day 1"))
        #expect(storage.string.hasSuffix("\n**Morning** walk"))
        #expect(!storage.string.contains("###"))

        let appended = controller.synchronize(
            storage: storage, response: "### Day 1\n\n**Morning** walk.", isTerminal: false)
        #expect(appended.mutation == .appended)

        #expect(StreamingResponseDocumentController.finishedParagraphBoundary(
            in: "```swift\nlet a = 1\n\nlet b = 2\n") == 0)
        #expect(StreamingResponseDocumentController.finishedParagraphBoundary(
            in: "Intro\n\n```swift\nlet a = 1\n\n") == 7)

        let final = controller.synchronize(
            storage: storage, response: "### Day 1\n\n**Morning** walk.", isTerminal: true)
        #expect(final.mutation == .finalized)
        #expect(storage.string == "Day 1\nMorning walk." || !storage.string.contains("**"))
    }

    @Test func animatedPrefillPlaceholderIsPresentationOnlyAndFirstResponseRemovesIt() {
        let storage = NSMutableAttributedString()
        let controller = StreamingResponseDocumentController()

        let prefilling = controller.synchronize(
            storage: storage,
            response: "",
            isTerminal: false,
            showsPrefillPlaceholder: true)

        #expect(prefilling.mutation == .rebuilt)
        #expect(controller.showsPrefillPlaceholder)
        #expect(storage.string == "Processing your prompt")
        #expect(controller.response.isEmpty)
        #expect(controller.assistantRange.length == 0)

        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt.."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt..."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt"))

        let responding = controller.synchronize(
            storage: storage,
            response: "Hello",
            isTerminal: false,
            showsPrefillPlaceholder: true)

        #expect(responding.mutation == .rebuilt)
        #expect(!controller.showsPrefillPlaceholder)
        #expect(storage.string == "Hello")
        #expect(!storage.string.contains("Processing your prompt"))
        #expect((storage.string as NSString).substring(with: responding.assistantRange)
            == "Hello")
    }

    @Test func processingAnimationPolicyStopsForTextAndTerminalStates() {
        #expect(StreamingResponseDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: false, requested: true))
        #expect(!StreamingResponseDocumentController.shouldRunPrefillAnimation(
            response: "First token", isTerminal: false, requested: true))
        #expect(!StreamingResponseDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: true, requested: true))
        #expect(!StreamingResponseDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: false, requested: false))
    }

    /// A new run reuses the view, and with it this controller. Its response is
    /// not a prefix of the last one, which is what has to force the rebuild —
    /// otherwise the next answer appends to the previous answer.
    @Test func aNewRunRebuildsRatherThanAppendingToTheLastAnswer() {
        let storage = NSMutableAttributedString()
        let controller = StreamingResponseDocumentController()
        _ = controller.synchronize(
            storage: storage, response: "Long response", isTerminal: true)

        let result = controller.synchronize(
            storage: storage, response: "Short", isTerminal: false)

        #expect(result.mutation == .rebuilt)
        #expect(storage.string == "Short")
        #expect(!storage.string.contains("Long response"))
    }

    @Test func terminalUpdateFormatsTheAnswerAndKeepsRawResponse() {
        let storage = NSMutableAttributedString()
        let controller = StreamingResponseDocumentController()
        _ = controller.synchronize(
            storage: storage,
            response: "**Bold answer**",
            isTerminal: false)

        let result = controller.synchronize(
            storage: storage,
            response: "**Bold answer**",
            isTerminal: true)

        #expect(result.mutation == .finalized)
        #expect(controller.isFinalized)
        #expect(controller.response == "**Bold answer**")
        #expect(storage.string == "Bold answer")
        #expect((storage.string as NSString).substring(with: result.assistantRange)
            == "Bold answer")

        let unchanged = storage.copy() as! NSAttributedString
        let repeated = controller.synchronize(
            storage: storage,
            response: "**Bold answer**",
            isTerminal: true)
        #expect(repeated.mutation == .none)
        #expect(storage.isEqual(to: unchanged))
    }

    @Test func terminalPartialOutputIsReadableAndNextRunRestoresStreamingSource() {
        let storage = NSMutableAttributedString()
        let controller = StreamingResponseDocumentController()
        _ = controller.synchronize(
            storage: storage,
            response: "Partial **answer**",
            isTerminal: true)
        #expect(storage.string.hasSuffix("Partial answer"))

        let result = controller.synchronize(
            storage: storage,
            response: "Partial **answer**",
            isTerminal: false)
        #expect(result.mutation == .rebuilt)
        #expect(!controller.isFinalized)
        #expect(storage.string.hasSuffix("Partial **answer**"))
    }

    @Test func terminalResponseRendersAgainWhenClosingFenceArrivesLate() {
        let storage = NSMutableAttributedString()
        let controller = StreamingResponseDocumentController()
        let partial = "```cpp\nkernel void matmul() {}"
        let complete = partial + "\n```"

        let first = controller.synchronize(
            storage: storage,
            response: partial,
            isTerminal: true)
        #expect(first.mutation == .finalized)
        // Already a card: an open fence is rendered as if it had closed.
        #expect(!storage.string.contains("```"))

        let updated = controller.synchronize(
            storage: storage,
            response: complete,
            isTerminal: true)

        #expect(updated.mutation == .finalized)
        #expect(controller.response == complete)
        // A card that is the whole answer gets hairline spacer lines around
        // it for layout; the code itself is unchanged.
        #expect((storage.string as NSString).substring(with: updated.assistantRange)
            .trimmingCharacters(in: .newlines) == "kernel void matmul() {}")
        #expect(!storage.string.contains("```"))
    }

    @Test func selectionRangesClampToCurrentStorage() {
        let ranges = StreamingResponseDocumentController.clampedRanges([
            NSRange(location: 3, length: 20),
            NSRange(location: 50, length: 2),
        ], toLength: 10)

        #expect(ranges == [
            NSRange(location: 3, length: 7),
            NSRange(location: 10, length: 0),
        ])
    }
}
