import Foundation

/// Reveals a streaming answer at a steady pace.
///
/// The engine sends text in batches about every 100 ms, and the window draws
/// on its own clock, so an answer arrived as runs of several words with pauses
/// between them: at 70 tokens a second it looked like chunks, not writing.
/// This holds what has arrived and gives back a little more of it on every
/// frame, at the rate text has recently been arriving.
///
/// It never falls far behind: when more is waiting than `maximumLag` seconds
/// of it at the current rate, it speeds up to catch up. Text is revealed one
/// character (grapheme) at a time, so an emoji or accented letter is never
/// split. The caller shows everything at once when the answer ends, stops or
/// fails.
public struct StreamingTextPacer: Sendable {
    /// How far behind the arrived text the display may run, in seconds.
    public var maximumLag: TimeInterval

    public private(set) var target = ""
    public private(set) var displayed = ""
    private var targetCount = 0
    private var displayedCount = 0
    /// Recent arrival rate, characters per second.
    private var rate: Double = 0
    private var lastArrival: TimeInterval?
    private var lastFrame: TimeInterval?
    /// Fractional characters owed from earlier frames.
    private var carry: Double = 0
    /// Set when text arrives: the pace that shows all of it within
    /// `maximumLag`. A pace recomputed from the shrinking backlog on every
    /// frame only approaches the end and never quite gets there.
    private var catchUpRate: Double = 0

    public init(maximumLag: TimeInterval = 0.4) {
        self.maximumLag = maximumLag
    }

    public var isCaughtUp: Bool { displayedCount >= targetCount }

    /// Takes the complete text so far. Text that does not extend what came
    /// before (a new answer) starts the reveal over.
    public mutating func receive(_ text: String, at time: TimeInterval) {
        guard !text.utf8.elementsEqual(target.utf8) else { return }
        // Bytes, not characters: `hasPrefix` compares graphemes, so an accent
        // or emoji joiner arriving in the next batch would look like a new
        // answer and replay the text from the start.
        guard text.utf8.starts(with: target.utf8) else {
            self = StreamingTextPacer(maximumLag: maximumLag)
            receive(text, at: time)
            return
        }
        // The shown text may now end inside a character the new bytes
        // extended (an "e" that became "é"). Step back to the character
        // boundary at or before it, so nothing half-drawn stays on screen.
        if !displayed.isEmpty {
            let raw = text.utf8.index(text.utf8.startIndex, offsetBy: displayed.utf8.count)
            let aligned = text.index(raw, offsetBy: 0)
            if aligned != raw { displayed = String(text[..<aligned]) }
        }
        let previousCount = targetCount
        targetCount = text.count
        displayedCount = displayed.count
        let added = max(0, targetCount - previousCount)
        if let lastArrival {
            let interval = min(1, max(0.03, time - lastArrival))
            let instant = Double(added) / interval
            rate = rate == 0 ? instant : rate * 0.7 + instant * 0.3
        } else {
            // The first batch has no rate yet; reveal it over about one batch.
            rate = Double(added) / 0.1
        }
        lastArrival = time
        target = text
        catchUpRate = Double(targetCount - displayedCount) / maximumLag
    }

    /// Moves the reveal to `time` and returns the text to show, or nil when
    /// nothing changed since the last frame.
    public mutating func advance(to time: TimeInterval) -> String? {
        defer { lastFrame = time }
        guard let lastFrame, displayedCount < targetCount else { return nil }
        let elapsed = min(0.1, max(0, time - lastFrame))
        let pace = max(rate, catchUpRate)
        carry += pace * elapsed
        let step = min(Int(carry), targetCount - displayedCount)
        guard step > 0 else { return nil }
        carry -= Double(step)
        // `displayed` is always a prefix of `target` ending on a character
        // boundary, so its UTF-8 length is the place to continue from.
        let start = target.utf8.index(target.utf8.startIndex, offsetBy: displayed.utf8.count)
        let end = target.index(start, offsetBy: step, limitedBy: target.endIndex) ?? target.endIndex
        displayed += target[start..<end]
        displayedCount += step
        return displayed
    }

    /// Shows everything that has arrived.
    public mutating func flush() -> String {
        displayed = target
        displayedCount = targetCount
        carry = 0
        return displayed
    }
}
