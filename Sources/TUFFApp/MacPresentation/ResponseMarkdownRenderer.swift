import AppKit
import Foundation
import SwiftMath

public extension NSAttributedString.Key {
    static let tuffLaTeXSource = NSAttributedString.Key("TUFFLaTeXSource")
    /// Marks the text of one fenced code block, so the transcript can offer
    /// to copy that block alone. The value tells adjacent blocks apart.
    static let tuffCodeBlock = NSAttributedString.Key("TUFFCodeBlock")
    /// The language a fenced code block was marked with, as written after the
    /// fence. Empty when it had none.
    static let tuffCodeLanguage = NSAttributedString.Key("TUFFCodeLanguage")
    /// Layout-only text the renderer adds (the hairline lines around a code
    /// card at the answer's edges); left out of copied text.
    static let tuffLayoutSpacer = NSAttributedString.Key("TUFFLayoutSpacer")
}

@MainActor
public struct ResponseMarkdownRenderer {
    public struct Result {
        public let attributedString: NSAttributedString
        public let usedFallback: Bool

        public init(attributedString: NSAttributedString, usedFallback: Bool) {
            self.attributedString = attributedString
            self.usedFallback = usedFallback
        }
    }

    private enum BlockKind: Equatable {
        case paragraph
        case heading(Int)
        case code
        case quote
        case unorderedList(indent: Int)
        case orderedList(ordinal: Int, indent: Int)
        case thematicBreak
        case tableCell(TableCell)

        var isList: Bool {
            switch self {
            case .unorderedList, .orderedList: true
            default: false
            }
        }

        var tableCell: TableCell? {
            if case .tableCell(let cell) = self { cell } else { nil }
        }
    }

    /// One cell of a Markdown table. The header is row 0.
    private struct TableCell: Equatable {
        let table: Int
        let row: Int
        let column: Int
        let columns: Int
        var alignment: NSTextAlignment = .left

        /// The cell's place counting along the rows.
        var index: Int { row * columns + column }

        func cell(at index: Int) -> TableCell {
            TableCell(table: table, row: index / columns, column: index % columns, columns: columns)
        }
    }

    /// A table being laid out: its text table, and the paragraph style of
    /// each cell so far. Every run in a cell has to carry the same one.
    private final class TableLayout {
        let table = NSTextTable()
        var styles: [Int: NSParagraphStyle] = [:]
        var blocks: [Int: NSTextTableBlock] = [:]

        init(columns: Int) {
            table.numberOfColumns = columns
            table.collapsesBorders = true
            table.hidesEmptyCells = false
        }
    }

    private struct Block: Equatable {
        let identity: Int
        let kind: BlockKind
    }

    /// How much larger than the system default this transcript is set.
    /// Passed in rather than read from the environment: the text is built as
    /// an `NSAttributedString`, so the size has to be baked into the fonts.
    private let scale: CGFloat

    /// The body size everything else is derived from, at the current zoom.
    private var baseSize: CGFloat { NSFont.systemFontSize * scale }

    public init(scale: CGFloat = 1) {
        self.scale = scale
    }

    /// The scheme `AppCitations` gives a validated citation.
    public static let citationScheme = "tuff-source"

    /// Rendered results by zoom, appearance and source. SwiftUI rebuilds
    /// every message in the transcript each time a token arrives, so without
    /// this a chat with a few long answers re-parsed all of them per token,
    /// and the app fell minutes behind the model. Math is drawn as images for
    /// the current appearance, which is why that is part of the key.
    private static var cache: [String: Result] = [:]
    private static var cacheOrder: [String] = []
    private static let cacheLimit = 256

    public func render(_ source: String) -> Result {
        guard !source.isEmpty else { return renderUncached(source) }
        let appearance = NSApp?.effectiveAppearance.name.rawValue ?? ""
        let key = "\(scale)|\(appearance)|\(source)"
        if let cached = Self.cache[key] { return cached }
        let result = renderUncached(source)
        if Self.cacheOrder.count >= Self.cacheLimit {
            Self.cache.removeValue(forKey: Self.cacheOrder.removeFirst())
        }
        Self.cache[key] = result
        Self.cacheOrder.append(key)
        return result
    }

    private func renderUncached(_ source: String) -> Result {
        guard !source.isEmpty else {
            return Result(attributedString: NSAttributedString(), usedFallback: false)
        }
        let presentationSource = source.replacingOccurrences(
            of: #"(?m)^([ \t]*\*\*[^*\n]+\*\*[ \t]*)\n(?=\S)"#,
            with: "$1\n\n",
            options: .regularExpression)
        let extraction = LaTeXMathExtractor.extract(from: presentationSource)

        do {
            let parsed = try AttributedString(
                markdown: extraction.markdownSource,
                options: .init(
                    interpretedSyntax: .full,
                    failurePolicy: .returnPartiallyParsedIfPossible))
            guard !containsUnsupportedContent(in: parsed) else { return fallback(source) }

            let output = NSMutableAttributedString()
            var tables: [Int: TableLayout] = [:]
            var previousBlock: Block?
            for run in parsed.runs {
                let block = block(for: run.presentationIntent)
                if block != previousBlock {
                    appendSeparator(to: output, previous: previousBlock, next: block, tables: &tables)
                    appendPrefix(for: block.kind, to: output)
                    if block.kind == .thematicBreak {
                        output.append(NSAttributedString(
                            string: "────────────────",
                            attributes: attributes(
                                inlineIntent: nil,
                                link: nil,
                                block: block.kind)))
                    }
                    previousBlock = block
                }

                guard block.kind != .thematicBreak else { continue }
                var text = String(parsed[run.range].characters)
                guard !text.isEmpty else { continue }
                // A tag in the middle of a sentence is shown as written,
                // except a line break, which models put in table cells.
                if run.inlinePresentationIntent?.contains(.inlineHTML) == true,
                   text.range(of: #"^<br\s*/?>$"#, options: [.regularExpression, .caseInsensitive]) != nil {
                    text = "\u{2028}"
                }
                var values = attributes(
                    inlineIntent: run.inlinePresentationIntent,
                    link: run.link,
                    block: block.kind)
                if block.kind == .code {
                    values[.tuffCodeBlock] = block.identity
                    values[.tuffCodeLanguage] = Self.languageHint(run.presentationIntent) ?? ""
                }
                if let cell = block.kind.tableCell {
                    values[.paragraphStyle] = paragraphStyle(for: cell, tables: &tables)
                }
                output.append(NSAttributedString(string: text, attributes: values))
            }
            // A table that ends the answer on a short row.
            if let cell = previousBlock?.kind.tableCell {
                appendEmptyCells(after: cell, through: (cell.row + 1) * cell.columns - 1,
                                 to: output, tables: &tables)
            }

            guard output.length > 0 else { return fallback(source) }
            styleCodeBlocks(in: output)
            padCodeAtEdges(output)
            let usedMathFallback = replaceLaTeX(
                extraction.formulas, in: output)
            // After the math is in place: a formula is measured as drawn,
            // not as the marker word that stood in for it.
            sizeTables(tables, in: output)
            return Result(
                attributedString: output,
                usedFallback: usedMathFallback)
        } catch {
            return fallback(source)
        }
    }

    /// `source` rendered, with each citation number in `ranges` made a link.
    /// A number inside code is left alone. Links are added here, after the
    /// Markdown is rendered, so citations never pass through the Markdown or
    /// math parsers.
    public func render(_ source: String, citations: (String) -> [(range: NSRange, url: URL)])
        -> NSAttributedString {
        let rendered = NSMutableAttributedString(attributedString: render(source).attributedString)
        for citation in citations(rendered.string) {
            guard NSMaxRange(citation.range) <= rendered.length,
                  rendered.attribute(.backgroundColor, at: citation.range.location,
                                     effectiveRange: nil) == nil,
                  rendered.attribute(.tuffCodeBlock, at: citation.range.location,
                                     effectiveRange: nil) == nil else { continue }
            rendered.addAttributes([
                .link: citation.url,
                .foregroundColor: NSColor.linkColor,
                .toolTip: "Open source \(citation.url.absoluteString.dropFirst(Self.citationScheme.count + 1))",
            ], range: citation.range)
        }
        return rendered
    }

    /// The code blocks of a rendered answer, in order, without the newline
    /// the Markdown parser leaves at the end of each.
    public nonisolated static func codeBlocks(in attributed: NSAttributedString) -> [(range: NSRange, code: String)] {
        var blocks: [(range: NSRange, code: String)] = []
        attributed.enumerateAttribute(
            .tuffCodeBlock, in: NSRange(location: 0, length: attributed.length)
        ) { value, range, _ in
            guard value != nil else { return }
            let code = (attributed.string as NSString).substring(with: range)
            blocks.append((range, code.hasSuffix("\n") ? String(code.dropLast()) : code))
        }
        return blocks
    }

    public func plainText(_ source: String) -> String {
        let rendered = NSMutableAttributedString(attributedString: render(source).attributedString)
        var spacers: [NSRange] = []
        rendered.enumerateAttribute(.tuffLayoutSpacer, in: NSRange(location: 0, length: rendered.length)) {
            value, range, _ in if value != nil { spacers.append(range) }
        }
        // A table row reads across as plain text, its cells apart by tabs,
        // rather than one cell to a line.
        let text = rendered.string as NSString
        rendered.enumerateAttribute(.paragraphStyle, in: NSRange(location: 0, length: rendered.length)) {
            value, range, _ in
            guard let block = (value as? NSParagraphStyle)?.textBlocks.first as? NSTextTableBlock,
                  block.startingColumn < block.table.numberOfColumns - 1,
                  text.character(at: NSMaxRange(range) - 1) == 10 else { return }
            rendered.mutableString.replaceCharacters(
                in: NSRange(location: NSMaxRange(range) - 1, length: 1), with: "\t")
        }
        for range in spacers.reversed() { rendered.deleteCharacters(in: range) }
        return rendered.string
    }

    /// What the transcript does not draw: a block of HTML, which is never
    /// interpreted, and an image, which would be a request to somewhere the
    /// model chose. Either shows the answer as written. Asked of the parsed
    /// answer, so a tag or `![…](…)` inside code of any kind is just code.
    private func containsUnsupportedContent(in parsed: AttributedString) -> Bool {
        parsed.runs.contains { run in
            run.inlinePresentationIntent?.contains(.blockHTML) == true || run.imageURL != nil
        }
    }

    private func block(for intent: PresentationIntent?) -> Block {
        guard let components = intent?.components, let leaf = components.first else {
            return Block(identity: 0, kind: .paragraph)
        }

        var headingLevel: Int?
        var code = false
        var quote = false
        var thematicBreak = false
        var ordinal: Int?
        var ordered = false
        var unordered = false
        var listDepth = 0
        var table: (identity: Int, columns: [PresentationIntent.TableColumn])?
        var tableRow: Int?
        var tableColumn: Int?

        for component in components {
            switch component.kind {
            case .table(let columns): table = (component.identity, columns)
            case .tableHeaderRow: tableRow = 0
            case .tableRow(let index): tableRow = index
            case .tableCell(let index): tableColumn = index
            case .header(let level): headingLevel = level
            case .codeBlock: code = true
            case .blockQuote: quote = true
            case .thematicBreak: thematicBreak = true
            case .listItem(let itemOrdinal): ordinal = itemOrdinal
            case .orderedList:
                ordered = true
                listDepth += 1
            case .unorderedList:
                unordered = true
                listDepth += 1
            default: break
            }
        }

        let kind: BlockKind
        if let table, let tableRow, let tableColumn, table.columns.indices.contains(tableColumn) {
            let alignment: NSTextAlignment = switch table.columns[tableColumn].alignment {
            case .center: .center
            case .right: .right
            default: .left
            }
            kind = .tableCell(TableCell(
                table: table.identity, row: tableRow, column: tableColumn,
                columns: table.columns.count, alignment: alignment))
        } else if thematicBreak {
            kind = .thematicBreak
        } else if let headingLevel {
            kind = .heading(headingLevel)
        } else if code {
            kind = .code
        } else if ordered, let ordinal {
            kind = .orderedList(ordinal: ordinal, indent: max(0, listDepth - 1))
        } else if unordered {
            kind = .unorderedList(indent: max(0, listDepth - 1))
        } else if quote {
            kind = .quote
        } else {
            kind = .paragraph
        }
        return Block(identity: leaf.identity, kind: kind)
    }

    private func appendSeparator(
        to output: NSMutableAttributedString,
        previous: Block?,
        next: Block,
        tables: inout [Int: TableLayout]
    ) {
        let nextCell = next.kind.tableCell
        if let cell = previous?.kind.tableCell {
            if let nextCell, nextCell.table == cell.table {
                // The next cell of the same table: skipped ones were empty.
                appendEmptyCells(after: cell, through: nextCell.index - 1, to: output, tables: &tables)
                appendCellEnd(cell.cell(at: nextCell.index - 1), to: output, tables: &tables)
                return
            }
            // The table is over: finish its last row and its last cell.
            let last = (cell.row + 1) * cell.columns - 1
            appendEmptyCells(after: cell, through: last, to: output, tables: &tables)
            appendCellEnd(cell.cell(at: last), to: output, tables: &tables)
        }
        defer {
            // A table whose first cells are empty.
            if let nextCell, nextCell.index > 0, previous?.kind.tableCell?.table != nextCell.table {
                appendEmptyCell(nextCell.cell(at: 0), to: output, tables: &tables)
                appendEmptyCells(after: nextCell.cell(at: 0), through: nextCell.index - 1,
                                 to: output, tables: &tables)
                appendCellEnd(nextCell.cell(at: nextCell.index - 1), to: output, tables: &tables)
            }
        }
        guard let previous else { return }
        let adjacentListItems = previous.kind.isList && next.kind.isList
        let requiredNewlines = adjacentListItems ? 1 : 2
        let trailingNewlines = output.string.reversed().prefix { $0 == "\n" }.count
        guard trailingNewlines < requiredNewlines else { return }
        var separator = baseAttributes()
        if previous.kind == .code || next.kind == .code
            || previous.kind.tableCell != nil || nextCell != nil {
            // A short blank line beside a code card or a table: both have
            // their own padding, and a full body-height line left a gap
            // twice the size of the one between paragraphs.
            separator[.font] = NSFont.systemFont(ofSize: 6 * scale)
            let style = NSMutableParagraphStyle()
            style.paragraphSpacing = 4 * scale
            separator[.paragraphStyle] = style
        }
        output.append(NSAttributedString(
            string: String(repeating: "\n", count: requiredNewlines - trailingNewlines),
            attributes: separator))
    }

    /// The paragraph style that puts text in `cell`: its place in the text
    /// table, a hairline around it, and a tinted header row.
    private func paragraphStyle(for cell: TableCell, tables: inout [Int: TableLayout]) -> NSParagraphStyle {
        let layout = tables[cell.table] ?? TableLayout(columns: cell.columns)
        tables[cell.table] = layout
        if let style = layout.styles[cell.index] { return style }
        let block = NSTextTableBlock(
            table: layout.table, startingRow: cell.row, rowSpan: 1,
            startingColumn: cell.column, columnSpan: 1)
        // A rule under each row rather than a grid: lines on every side of
        // every cell read as a spreadsheet.
        block.setWidth(1, type: .absoluteValueType, for: .border, edge: .maxY)
        block.setBorderColor(.separatorColor)
        for edge in [NSRectEdge.minY, .maxY] {
            block.setWidth(Self.tableCellPadding.height * scale, type: .absoluteValueType, for: .padding, edge: edge)
        }
        for edge in [NSRectEdge.minX, .maxX] {
            block.setWidth(Self.tableCellPadding.width * scale, type: .absoluteValueType, for: .padding, edge: edge)
        }
        if cell.row == 0 { block.backgroundColor = TUFFDesign.inset }
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 2
        style.alignment = cell.alignment
        style.textBlocks = [block]
        layout.styles[cell.index] = style
        layout.blocks[cell.index] = block
        return style
    }

    private static let tableCellPadding = CGSize(width: 10, height: 6)
    /// The widest a column is taken to want. A cell holding a sentence wraps
    /// rather than claim the whole table.
    private static let tableColumnLimit: CGFloat = 300
    /// The narrowest the transcript's text gets, at the window's minimum
    /// size. A table no wider than this is laid out at its own width.
    private static let narrowestTranscript: CGFloat = 360

    /// Gives each table its column widths. Left alone, a text table spreads
    /// across the whole line in equal columns, so two short columns sat a
    /// window apart. A table that fits is as wide as what is in it; a wider
    /// one fills the line, shared out by how much each column holds.
    private func sizeTables(_ tables: [Int: TableLayout], in output: NSAttributedString) {
        guard !tables.isEmpty else { return }
        var widths: [ObjectIdentifier: [Int: CGFloat]] = [:]
        output.enumerateAttribute(.paragraphStyle, in: NSRange(location: 0, length: output.length)) { value, range, _ in
            guard let block = (value as? NSParagraphStyle)?.textBlocks.first as? NSTextTableBlock else { return }
            // Measured without the cell's own style: with it, the size is
            // that of the whole table laid out on an endless line.
            let text = NSMutableAttributedString(attributedString: output.attributedSubstring(from: range))
            text.removeAttribute(.paragraphStyle, range: NSRange(location: 0, length: text.length))
            let width = ceil(text.size().width) + 2 * Self.tableCellPadding.width * scale
            let table = ObjectIdentifier(block.table)
            widths[table, default: [:]][block.startingColumn] = max(
                widths[table]?[block.startingColumn] ?? 0, width)
        }
        for layout in tables.values {
            let natural = widths[ObjectIdentifier(layout.table)] ?? [:]
            let columns = (0..<layout.table.numberOfColumns).map {
                min(max(natural[$0] ?? 0, 44 * scale), Self.tableColumnLimit * scale)
            }
            let total = columns.reduce(0, +)
            let fits = total <= Self.narrowestTranscript
            if fits {
                layout.table.setContentWidth(total, type: .absoluteValueType)
            }
            for block in layout.blocks.values {
                let width = columns[block.startingColumn]
                let padding = 2 * Self.tableCellPadding.width * scale
                if fits {
                    block.setContentWidth(width - padding, type: .absoluteValueType)
                } else {
                    block.setContentWidth(width / total * 100, type: .percentageValueType)
                }
            }
        }
    }

    /// Ends `cell`'s paragraph, which is what closes a cell.
    private func appendCellEnd(_ cell: TableCell, to output: NSMutableAttributedString,
                               tables: inout [Int: TableLayout]) {
        var values = baseAttributes()
        values[.paragraphStyle] = paragraphStyle(for: cell, tables: &tables)
        output.append(NSAttributedString(string: "\n", attributes: values))
    }

    /// A cell the Markdown left empty. It still needs a paragraph of its
    /// own, or the table is drawn with a hole where it should be.
    private func appendEmptyCell(_ cell: TableCell, to output: NSMutableAttributedString,
                                 tables: inout [Int: TableLayout]) {
        var values = baseAttributes()
        values[.paragraphStyle] = paragraphStyle(for: cell, tables: &tables)
        values[.tuffLayoutSpacer] = true
        output.append(NSAttributedString(string: "\u{200B}", attributes: values))
    }

    /// The empty cells from the one after `cell` through `last`, each after
    /// the end of the cell before it.
    private func appendEmptyCells(after cell: TableCell, through last: Int,
                                  to output: NSMutableAttributedString,
                                  tables: inout [Int: TableLayout]) {
        guard last > cell.index else { return }
        for index in (cell.index + 1)...last {
            appendCellEnd(cell.cell(at: index - 1), to: output, tables: &tables)
            appendEmptyCell(cell.cell(at: index), to: output, tables: &tables)
        }
    }

    private func appendPrefix(for block: BlockKind, to output: NSMutableAttributedString) {
        let prefix: String
        switch block {
        case .unorderedList:
            prefix = "•\t"
        case .orderedList(let ordinal, _):
            prefix = "\(ordinal).\t"
        case .quote:
            prefix = "│\t"
        default:
            return
        }
        output.append(NSAttributedString(
            string: prefix,
            attributes: attributes(inlineIntent: nil, link: nil, block: block)))
    }

    private func attributes(
        inlineIntent: InlinePresentationIntent?,
        link: URL?,
        block: BlockKind
    ) -> [NSAttributedString.Key: Any] {
        var values = baseAttributes()
        values[.paragraphStyle] = paragraphStyle(for: block)
        values[.font] = font(for: block, inlineIntent: inlineIntent)

        if block == .quote {
            values[.foregroundColor] = NSColor.secondaryLabelColor
        }
        // Fenced blocks already have one continuous background from codeBox.
        // Per-run backgrounds paint contrasting stripes behind each line.
        if block != .code && inlineIntent?.contains(.code) == true {
            values[.backgroundColor] = TUFFDesign.inset
        }
        if inlineIntent?.contains(.strikethrough) == true {
            values[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
        }
        if let link {
            values[.foregroundColor] = NSColor.linkColor
            values[.underlineStyle] = NSUnderlineStyle.single.rawValue
            // Only citations the app validated are clickable. A link the
            // model wrote stays styled text: its address is unverified.
            if link.scheme == Self.citationScheme {
                values[.link] = link
                values[.underlineStyle] = nil
                values[.toolTip] = "Open source \(link.absoluteString.dropFirst(Self.citationScheme.count + 1))"
            }
        }
        return values
    }

    private func baseAttributes() -> [NSAttributedString.Key: Any] {
        [
            .font: NSFont.systemFont(ofSize: baseSize),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: paragraphStyle(for: .paragraph),
        ]
    }

    private func font(
        for block: BlockKind,
        inlineIntent: InlinePresentationIntent?
    ) -> NSFont {
        if block == .code || inlineIntent?.contains(.code) == true {
            return NSFont.monospacedSystemFont(
                ofSize: baseSize - 0.5 * scale,
                weight: .regular)
        }

        let size: CGFloat
        let baseWeight: NSFont.Weight
        switch block {
        case .heading(let level):
            size = max(baseSize + scale, (22 - CGFloat(level - 1) * 2) * scale)
            baseWeight = .semibold
        case .tableCell(let cell) where cell.row == 0:
            size = baseSize
            baseWeight = .semibold
        default:
            size = baseSize
            baseWeight = .regular
        }

        let stronglyEmphasized = inlineIntent?.contains(.stronglyEmphasized) == true
        let emphasized = inlineIntent?.contains(.emphasized) == true
        var font = NSFont.systemFont(
            ofSize: size,
            weight: stronglyEmphasized ? .semibold : baseWeight)
        if emphasized {
            let descriptor = font.fontDescriptor.withSymbolicTraits(.italic)
            font = NSFont(descriptor: descriptor, size: size) ?? font
        }
        return font
    }

    private static func languageHint(_ intent: PresentationIntent?) -> String? {
        for component in intent?.components ?? [] {
            if case .codeBlock(let hint) = component.kind { return hint }
        }
        return nil
    }

    /// Lays out each fenced block as a card: text inset from both edges, room
    /// at the top for the header (language and Copy) that the text view draws,
    /// padding at the bottom, and syntax colors. The card itself is drawn by
    /// `CodeBlockLayoutManager` behind these lines.
    private func styleCodeBlocks(in output: NSMutableAttributedString) {
        let inset = Self.codeInset * scale
        let header = Self.codeHeaderHeight * scale
        for block in Self.codeBlocks(in: output) {
            let text = output.string as NSString
            var paragraphs: [NSRange] = []
            text.enumerateSubstrings(in: block.range, options: [.byParagraphs, .substringNotRequired]) {
                _, _, enclosing, _ in paragraphs.append(enclosing)
            }
            for (index, paragraph) in paragraphs.enumerated() {
                let style = NSMutableParagraphStyle()
                style.lineSpacing = 3 * scale
                style.firstLineHeadIndent = inset
                style.headIndent = inset
                style.tailIndent = -inset
                style.lineBreakMode = .byCharWrapping
                style.paragraphSpacingBefore = index == 0 ? header : 0
                style.paragraphSpacing = index == paragraphs.count - 1 ? inset : 0
                output.addAttribute(.paragraphStyle, value: style, range: paragraph)
            }
            let language = output.attribute(.tuffCodeLanguage, at: block.range.location,
                                            effectiveRange: nil) as? String ?? ""
            let code = NSRange(location: block.range.location, length: (block.code as NSString).length)
            for token in CodeSyntaxHighlighter.tokens(in: block.code, language: language) {
                output.addAttribute(.foregroundColor, value: token.kind.color,
                                    range: NSRange(location: code.location + token.range.location,
                                                   length: token.range.length))
            }
        }
    }

    /// A hairline paragraph before a card that opens the answer and after one
    /// that ends it. TextKit drops the space above a text view's first line
    /// and below its last, which put the header on top of the code.
    private func padCodeAtEdges(_ output: NSMutableAttributedString) {
        guard output.length > 0 else { return }
        let style = NSMutableParagraphStyle()
        style.minimumLineHeight = 1
        style.maximumLineHeight = 1
        let spacer = NSAttributedString(string: "\n", attributes: [
            .font: NSFont.systemFont(ofSize: 1), .paragraphStyle: style,
            .tuffLayoutSpacer: true])
        if output.attribute(.tuffCodeBlock, at: output.length - 1, effectiveRange: nil) != nil {
            output.append(spacer)
        }
        if output.attribute(.tuffCodeBlock, at: 0, effectiveRange: nil) != nil {
            output.insert(spacer, at: 0)
        }
    }

    /// Space from the card's edges to its code, in points at 100%.
    public nonisolated static let codeInset: CGFloat = 16
    /// The header row above the code, in points at 100%.
    public nonisolated static let codeHeaderHeight: CGFloat = 40

    private func paragraphStyle(for block: BlockKind) -> NSParagraphStyle {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 6

        switch block {
        case .heading:
            style.paragraphSpacingBefore = 8
            style.paragraphSpacing = 4
        case .code:
            // Replaced per line by styleCodeBlocks once the block is complete.
            style.paragraphSpacingBefore = 0
            style.paragraphSpacing = 0
            style.lineSpacing = 3
        case .quote:
            style.firstLineHeadIndent = 4
            style.headIndent = 20
            style.tailIndent = -8
            style.tabStops = [NSTextTab(textAlignment: .left, location: 16)]
        case .unorderedList(let indent), .orderedList(_, let indent):
            let base = CGFloat(22 + indent * 18)
            style.firstLineHeadIndent = CGFloat(indent * 18)
            style.headIndent = base
            style.tabStops = [NSTextTab(textAlignment: .left, location: base)]
            style.paragraphSpacing = 2
        case .thematicBreak:
            style.alignment = .center
            style.paragraphSpacingBefore = 8
            style.paragraphSpacing = 8
        case .paragraph, .tableCell:
            // A cell's style is made with its table, in the render loop.
            break
        }
        return style
    }

    /// Replaces parser-safe marker words after Markdown has been styled. This
    /// keeps Markdown syntax out of equations and equations out of code spans,
    /// while still producing one native attributed transcript.
    private func replaceLaTeX(
        _ formulas: [LaTeXFormula],
        in output: NSMutableAttributedString
    ) -> Bool {
        var usedFallback = false
        for formula in formulas.reversed() {
            let range = (output.string as NSString).range(of: formula.token)
            guard range.location != NSNotFound else {
                usedFallback = true
                continue
            }
            let inherited = output.attributes(
                at: range.location, effectiveRange: nil)
            guard let rendered = renderLaTeX(formula, inherited: inherited) else {
                output.replaceCharacters(
                    in: range,
                    with: NSAttributedString(
                        string: formula.source,
                        attributes: inherited))
                usedFallback = true
                continue
            }
            output.replaceCharacters(in: range, with: rendered)
        }
        return usedFallback
    }

    private func renderLaTeX(
        _ formula: LaTeXFormula,
        inherited: [NSAttributedString.Key: Any]
    ) -> NSAttributedString? {
        let supportedBody = formula.body
            .replacingOccurrences(of: #"\lVert"#, with: #"\Vert"#)
            .replacingOccurrences(of: #"\rVert"#, with: #"\Vert"#)
        var math = MathImage(
            latex: supportedBody,
            fontSize: formula.isDisplay ? 17 * scale : baseSize,
            textColor: NSColor.labelColor,
            labelMode: formula.isDisplay ? .display : .text,
            textAlignment: .left)
        let (error, image, layout) = math.asImage()
        guard error == nil, let image, let layout,
              image.size.width > 0, image.size.height > 0 else { return nil }

        image.accessibilityDescription = "Math formula: \(formula.body)"
        let attachment = NSTextAttachment()
        attachment.image = image
        attachment.bounds = NSRect(
            x: 0,
            y: -layout.descent,
            width: image.size.width,
            height: image.size.height)
        let rendered = NSMutableAttributedString(attachment: attachment)
        rendered.addAttributes(
            inherited,
            range: NSRange(location: 0, length: rendered.length))
        rendered.addAttribute(
            .tuffLaTeXSource,
            value: formula.source,
            range: NSRange(location: 0, length: rendered.length))
        if formula.isDisplay {
            let style = (inherited[.paragraphStyle] as? NSParagraphStyle)?
                .mutableCopy() as? NSMutableParagraphStyle
                ?? NSMutableParagraphStyle()
            style.alignment = .center
            style.paragraphSpacingBefore = max(style.paragraphSpacingBefore, 6)
            style.paragraphSpacing = max(style.paragraphSpacing, 8)
            rendered.addAttribute(
                .paragraphStyle,
                value: style,
                range: NSRange(location: 0, length: rendered.length))
        }
        return rendered
    }

    private func fallback(_ source: String) -> Result {
        Result(
            attributedString: NSAttributedString(
                string: source,
                attributes: baseAttributes()),
            usedFallback: true)
    }
}
