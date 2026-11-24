import Foundation

/// Reveals a streaming answer at a steady pace.
///
/// The engine sends text in batches about every 100 ms, and the window draws
/// on its own clock, so an answer arrived as runs of several words with pauses
/// between them. This holds what has arrived and gives back a little more of
/// it on every frame.
///
/// It works like a playout buffer. The reveal runs at the average rate text
/// arrived over the last few seconds and stays about one typical gap between
/// arrivals behind, so the next word is usually there before the current one
/// is finished. A slow model (a few tokens a second, unevenly spaced) then
/// writes at an even pace instead of rushing each word and waiting for the
/// next. When the backlog grows past that delay the pace rises to catch up;
/// when it shrinks the pace eases off instead of stopping. Text is revealed
/// one character (grapheme) at a time, so an emoji or accented letter is never
/// split. The caller shows everything at once when the answer ends, stops or
/// fails.
public struct StreamingTextPacer: Sendable {
    /// The most the reveal may deliberately run behind, in seconds.
    public var maximumLag: TimeInterval

    public private(set) var target = ""
    public private(set) var displayed = ""
    private var targetCount = 0
    private var displayedCount = 0
    /// Recent arrivals: time and total characters, oldest first.
    private var arrivals: [(time: TimeInterval, count: Int)] = []
    private var lastFrame: TimeInterval?
    /// Fractional characters owed from earlier frames.
    private var carry: Double = 0
    /// When each pending batch must be fully shown, oldest first. Kept apart
    /// from the rate history, which drops old arrivals; an entry leaves only
    /// once its text is on screen.
    private var deadlines: [(time: TimeInterval, count: Int)] = []

    /// Arrivals older than this stop counting toward the rate.
    static let window: TimeInterval = 3
    static let minimumDelay: TimeInterval = 0.1
    /// Nothing that has arrived stays hidden much longer than this.
    var catchUpWindow: TimeInterval { maximumLag * 1.5 }

    public init(maximumLag: TimeInterval = 1.0) {
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
        let wasCaughtUp = displayedCount >= targetCount
        targetCount = text.count
        displayedCount = displayed.count
        target = text
        deadlines.removeAll { $0.count <= displayedCount }
        let due = time + catchUpWindow
        if let last = deadlines.last, due - last.time < 0.1 {
            // Close together: hold the newer text to the earlier deadline.
            deadlines[deadlines.count - 1].count = targetCount
        } else if targetCount > displayedCount {
            deadlines.append((due, targetCount))
        }
        // Frames that passed while nothing was waiting (or while the window
        // was not drawing) cannot be spent on text that arrived after them.
        if wasCaughtUp, let lastFrame, lastFrame < time { self.lastFrame = time }
        arrivals.append((time, targetCount))
        // After a pause longer than the window (a tool call, a long think)
        // only the fresh arrival is left, so the old pace does not hold the
        // resumed text back.
        while arrivals.count > 1, let first = arrivals.first, time - first.time > Self.window {
            arrivals.removeFirst()
        }
        if arrivals.count > 32 { arrivals.removeFirst(arrivals.count - 32) }
    }

    /// Characters per second over the recent arrivals, or nil before there
    /// are two of them.
    var arrivalRate: Double? {
        guard let first = arrivals.first, let last = arrivals.last, arrivals.count >= 2 else { return nil }
        let span = max(1.0 / 60, last.time - first.time)
        return Double(last.count - first.count) / span
    }

    /// How far behind arrivals the reveal aims to run: a little more than
    /// most gaps between them, so the display rarely runs dry.
    var targetDelay: TimeInterval {
        var gaps: [TimeInterval] = []
        for index in arrivals.indices.dropFirst() {
            gaps.append(arrivals[index].time - arrivals[index - 1].time)
        }
        guard !gaps.isEmpty else { return Self.minimumDelay }
        gaps.sort()
        let typical = gaps[min(gaps.count - 1, Int(Double(gaps.count) * 0.8))]
        return min(max(Self.minimumDelay, maximumLag), max(Self.minimumDelay, typical * 1.1))
    }

    /// Moves the reveal to `time` and returns the text to show, or nil when
    /// nothing changed since the last frame.
    public mutating func advance(to time: TimeInterval) -> String? {
        defer { lastFrame = time }
        guard let lastFrame, displayedCount < targetCount else { return nil }
        let elapsed = min(0.1, max(0, time - lastFrame))
        // Each arrival must be fully shown by `catchUpWindow` after it came:
        // pace to the most urgent of those deadlines.
        var catchUpRate = 0.0
        for deadline in deadlines where deadline.count > displayedCount {
            let left = max(1.0 / 60, deadline.time - time)
            catchUpRate = max(catchUpRate, Double(deadline.count - displayedCount) / left)
        }
        let backlog = Double(targetCount - displayedCount)
        let delay = targetDelay
        let pace: Double
        if let rate = arrivalRate, rate > 0 {
            // Steer toward running `delay` behind: on target this is the
            // arrival rate; ahead of it, slower; behind it, faster.
            let behind = backlog / rate
            // Never leave arrived text hidden much past `catchUpWindow`,
            // whatever the recent rate says.
            pace = max(rate * min(3, max(0.5, 1 + 0.5 * (behind - delay) / delay)), catchUpRate)
        } else {
            // One arrival so far: no rate yet, show it over the delay.
            pace = max(backlog / delay, catchUpRate)
        }
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
        deadlines.removeAll { $0.count <= displayedCount }
        // Credit left over from catching up must not flash the next batch.
        if displayedCount >= targetCount { carry = 0 }
        return displayed
    }

    /// Shows everything that has arrived.
    public mutating func flush() -> String {
        displayed = target
        displayedCount = targetCount
        carry = 0
        deadlines.removeAll()
        return displayed
    }
}
