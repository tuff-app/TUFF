import AppKit
import Foundation

// MARK: - Syntax colors

/// A small, language-tolerant highlighter for fenced code in answers.
///
/// It does not parse: one pass over the text finds comments, strings, numbers
/// and words, and words are sorted into keywords, types and calls by shape.
/// That is right for the common cases in every language a model writes, and
/// it never fails in a way worse than leaving some text uncolored.
public enum CodeSyntaxHighlighter {
    public enum Kind {
        case keyword, string, number, comment, type, function

        /// Xcode's default palette, so code reads as it does in the editor.
        public var color: NSColor {
            switch self {
            case .keyword: TUFFDesign.dynamic(light: 0x9B2393, dark: 0xFF7AB2)
            case .string: TUFFDesign.dynamic(light: 0xC41A16, dark: 0xFF8170)
            case .number: TUFFDesign.dynamic(light: 0x1C00CF, dark: 0xD9C97C)
            case .comment: TUFFDesign.dynamic(light: 0x5D6C79, dark: 0x7F8C98)
            case .type: TUFFDesign.dynamic(light: 0x3900A0, dark: 0xDABAFF)
            case .function: TUFFDesign.dynamic(light: 0x326D74, dark: 0x67B7A4)
            }
        }
    }

    public struct Token {
        public let range: NSRange
        public let kind: Kind
    }

    private static let hashCommentLanguages: Set<String> = [
        "", "python", "py", "ruby", "rb", "sh", "bash", "zsh", "shell", "console",
        "yaml", "yml", "toml", "perl", "r", "dockerfile", "makefile", "make",
        "elixir", "nim", "powershell", "ps1", "conf", "ini",
    ]

    // Built from an array: the lists overlap, and a Set literal with a
    // repeated element traps at launch.
    private static let keywords = Set<String>([
        // Shared by most C-family languages
        "if", "else", "for", "while", "do", "switch", "case", "default", "break",
        "continue", "return", "goto", "try", "catch", "finally", "throw", "throws",
        "new", "delete", "this", "self", "super", "null", "nil", "true", "false",
        "class", "struct", "enum", "interface", "extends", "implements", "import",
        "export", "from", "package", "public", "private", "protected", "internal",
        "static", "final", "const", "let", "var", "void", "async", "await", "yield",
        "in", "is", "as", "typeof", "instanceof", "where", "with",
        // Swift
        "func", "guard", "defer", "repeat", "protocol", "extension", "init",
        "deinit", "inout", "some", "any", "mutating", "override", "fileprivate",
        "open", "lazy", "weak", "unowned", "rethrows", "subscript", "typealias",
        "associatedtype", "operator", "actor", "nonisolated", "Self",
        // Python
        "def", "lambda", "pass", "raise", "except", "elif", "not", "and", "or",
        "None", "True", "False", "global", "nonlocal", "assert", "del",
        // JavaScript / TypeScript
        "function", "undefined", "type", "namespace", "declare", "readonly",
        "keyof", "of",
        // Rust / Go
        "fn", "mut", "impl", "trait", "pub", "use", "mod", "crate", "match",
        "loop", "move", "ref", "unsafe", "dyn", "go", "chan", "select", "defer",
        "range", "map", "fallthrough",
        // C / C++ / Java / Kotlin
        "int", "long", "short", "char", "float", "double", "bool", "boolean",
        "unsigned", "signed", "sizeof", "template", "typename", "virtual",
        "auto", "namespace", "using", "fun", "val", "when", "object", "companion",
        // Shell
        "then", "fi", "esac", "done", "echo", "export", "local", "function",
        // Ruby
        "end", "begin", "rescue", "ensure", "unless", "until", "module", "require",
    ])

    private static let scanner = try! NSRegularExpression(pattern: [
        #"(?<blockcomment>/\*[\s\S]*?(?:\*/|$))"#,
        #"(?<linecomment>//[^\n]*|--(?=\s)[^\n]*)"#,
        #"(?<hash>#[^\n]*)"#,
        #"(?<string>"""[\s\S]*?(?:"""|$)|"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?|`(?:\\.|[^`\\])*`?)"#,
        #"(?<number>\b(?:0[xX][0-9a-fA-F_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)\b)"#,
        #"(?<word>[A-Za-z_][A-Za-z0-9_]*)"#,
    ].joined(separator: "|"))

    private static let markupLanguages: Set<String> = [
        "html", "htm", "xhtml", "xml", "svg", "vue", "svelte", "plist", "xaml", "erb",
    ]
    private static let jsxLanguages: Set<String> = ["jsx", "tsx"]
    private static let styleLanguages: Set<String> = ["css", "scss", "sass", "less"]

    private static let markupScanner = try! NSRegularExpression(pattern: [
        #"(?<comment><!--[\s\S]*?(?:-->|$))"#,
        #"(?<tag></?[A-Za-z][\w:.-]*|/?>)"#,
        #"(?<attribute>\b[A-Za-z_:@][\w:.-]*(?=\s*=))"#,
        #"(?<string>"[^"]*"?|'[^']*'?)"#,
        #"(?<entity>&#?\w+;)"#,
    ].joined(separator: "|"))

    private static let cssPropertyScanner = try! NSRegularExpression(
        pattern: #"(?m)(?<=[{;\s])[a-z-]+(?=\s*:[^:])|#[0-9a-fA-F]{3,8}\b"#)

    public static func tokens(in code: String, language: String) -> [Token] {
        let lang = language.lowercased().trimmingCharacters(in: .whitespaces)
        if markupLanguages.contains(lang) { return markupTokens(in: code) }
        var tokens = codeTokens(in: code, language: lang)
        if jsxLanguages.contains(lang) {
            // Tags inside JSX, where the generic pass sees only words.
            tokens += markupTokens(in: code).filter { $0.kind == .keyword }
        }
        if styleLanguages.contains(lang) {
            let text = code as NSString
            cssPropertyScanner.enumerateMatches(in: code, range: NSRange(location: 0, length: text.length)) {
                match, _, _ in
                guard let match else { return }
                let isColor = text.character(at: match.range.location) == 0x23
                tokens.append(Token(range: match.range, kind: isColor ? .number : .type))
            }
        }
        return tokens
    }

    /// Tags as keywords, attributes as calls, values as strings: how Xcode
    /// colors HTML and XML.
    private static func markupTokens(in code: String) -> [Token] {
        var tokens: [Token] = []
        markupScanner.enumerateMatches(in: code, range: NSRange(location: 0, length: (code as NSString).length)) {
            match, _, _ in
            guard let match else { return }
            func has(_ name: String) -> Bool { match.range(withName: name).location != NSNotFound }
            let kind: Kind = has("comment") ? .comment : has("tag") ? .keyword
                : has("attribute") ? .function : has("string") ? .string : .number
            tokens.append(Token(range: match.range, kind: kind))
        }
        return tokens
    }

    private static func codeTokens(in code: String, language lang: String) -> [Token] {
        let hashComments = hashCommentLanguages.contains(lang)
        let text = code as NSString
        var tokens: [Token] = []
        scanner.enumerateMatches(in: code, range: NSRange(location: 0, length: text.length)) {
            match, _, _ in
            guard let match else { return }
            func has(_ name: String) -> Bool { match.range(withName: name).location != NSNotFound }
            let range = match.range
            if has("blockcomment") || has("linecomment") {
                tokens.append(Token(range: range, kind: .comment))
            } else if has("hash") {
                // `#` starts a comment in scripting languages; elsewhere it is
                // a directive (`#include`, `#if`), shown as a keyword.
                tokens.append(Token(range: range, kind: hashComments ? .comment : .keyword))
            } else if has("string") {
                tokens.append(Token(range: range, kind: .string))
            } else if has("number") {
                tokens.append(Token(range: range, kind: .number))
            } else if has("word") {
                let word = text.substring(with: range)
                if keywords.contains(word) {
                    tokens.append(Token(range: range, kind: .keyword))
                } else if let next = nextNonSpace(in: text, after: NSMaxRange(range)), next == "(" {
                    tokens.append(Token(range: range, kind: word.first!.isUppercase ? .type : .function))
                } else if word.first!.isUppercase, word.count > 1 {
                    tokens.append(Token(range: range, kind: .type))
                }
            }
        }
        return tokens
    }

    private static func nextNonSpace(in text: NSString, after index: Int) -> Character? {
        var i = index
        while i < text.length {
            let c = text.character(at: i)
            if c != 0x20 && c != 0x09 { return Character(UnicodeScalar(c) ?? " ") }
            i += 1
        }
        return nil
    }

    /// The name shown in a code block's header.
    public static func displayName(for language: String) -> String {
        let lang = language.lowercased().trimmingCharacters(in: .whitespaces)
        switch lang {
        case "": return "Code"
        case "js", "javascript", "jsx": return "JavaScript"
        case "ts", "typescript", "tsx": return "TypeScript"
        case "py", "python", "python3": return "Python"
        case "sh", "bash", "zsh", "shell", "console": return "Shell"
        case "cpp", "c++", "cc", "hpp": return "C++"
        case "cs", "csharp", "c#": return "C#"
        case "objc", "objective-c": return "Objective-C"
        case "rb", "ruby": return "Ruby"
        case "rs", "rust": return "Rust"
        case "go", "golang": return "Go"
        case "kt", "kotlin": return "Kotlin"
        case "md", "markdown": return "Markdown"
        case "yml", "yaml": return "YAML"
        case "json", "html", "css", "sql", "xml", "toml", "php": return lang.uppercased()
        default: return lang.prefix(1).uppercased() + lang.dropFirst()
        }
    }
}

// MARK: - The card

public extension TUFFDesign {
    /// Behind a code block: a step off the page, like the composer.
    static let codeBackground = dynamic(light: 0xF5F5F8, dark: 0x252528)
}

/// Draws each fenced code block as one rounded card behind its lines, with
/// the language in the header row the renderer leaves room for. The Copy
/// button in that row is a real control, placed by `MessageTextView`.
public final class CodeBlockLayoutManager: NSLayoutManager {
    public struct Card {
        public let rect: NSRect
        public let headerHeight: CGFloat
        public let language: String
        public let code: String
        public let codeFont: NSFont
    }

    /// Every code card in the text, in container coordinates.
    public func codeCards() -> [Card] {
        guard let storage = textStorage, let container = textContainers.first,
              storage.length > 0 else { return [] }
        return ResponseMarkdownRenderer.codeBlocks(in: storage).compactMap { block in
            let glyphs = glyphRange(forCharacterRange: block.range, actualCharacterRange: nil)
            guard glyphs.length > 0 else { return nil }
            var rect = NSRect.null
            enumerateLineFragments(forGlyphRange: glyphs) { fragment, _, _, _, _ in
                rect = rect.union(fragment)
            }
            guard !rect.isNull else { return nil }
            rect.origin.x = 0
            rect.size.width = container.size.width
            let style = storage.attribute(.paragraphStyle, at: block.range.location,
                                          effectiveRange: nil) as? NSParagraphStyle
            let font = storage.attribute(.font, at: block.range.location, effectiveRange: nil)
                as? NSFont ?? .monospacedSystemFont(ofSize: 12, weight: .regular)
            let language = storage.attribute(.tuffCodeLanguage, at: block.range.location,
                                             effectiveRange: nil) as? String ?? ""
            return Card(rect: rect, headerHeight: style?.paragraphSpacingBefore ?? 0,
                        language: language, code: block.code, codeFont: font)
        }
    }

    public override func drawBackground(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
        for card in codeCards() {
            let rect = card.rect.offsetBy(dx: origin.x, dy: origin.y).insetBy(dx: 0.5, dy: 0.5)
            let radius = min(14, card.headerHeight * 0.35)
            let path = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
            TUFFDesign.codeBackground.setFill()
            path.fill()
            TUFFDesign.hairline.setStroke()
            path.lineWidth = 1
            path.stroke()
            drawHeader(card, in: rect)
        }
        super.drawBackground(forGlyphRange: glyphsToShow, at: origin)
    }

    private func drawHeader(_ card: Card, in rect: NSRect) {
        guard card.headerHeight > 0 else { return }
        let size = card.codeFont.pointSize
        let inset = size * ResponseMarkdownRenderer.codeInset / 12.5
        let label = NSAttributedString(
            string: CodeSyntaxHighlighter.displayName(for: card.language),
            attributes: [
                .font: NSFont.systemFont(ofSize: size, weight: .medium),
                .foregroundColor: TUFFDesign.secondaryText,
            ])
        let labelSize = label.size()
        var x = rect.minX + inset
        let midY = rect.minY + card.headerHeight / 2
        let configuration = NSImage.SymbolConfiguration(pointSize: size, weight: .medium)
            .applying(.init(paletteColors: [TUFFDesign.secondaryText]))
        if let symbol = NSImage(systemSymbolName: "chevron.left.forwardslash.chevron.right",
                                accessibilityDescription: nil)?
            .withSymbolConfiguration(configuration) {
            let symbolSize = symbol.size
            symbol.draw(in: NSRect(x: x, y: midY - symbolSize.height / 2,
                                   width: symbolSize.width, height: symbolSize.height),
                        from: .zero, operation: .sourceOver, fraction: 1,
                        respectFlipped: true, hints: nil)
            x += symbolSize.width + 7
        }
        label.draw(at: NSPoint(x: x, y: midY - labelSize.height / 2))
    }
}
