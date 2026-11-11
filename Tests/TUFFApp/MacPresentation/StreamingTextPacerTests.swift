import Foundation
import Testing
@testable import TUFFMacPresentation

@Suite struct StreamingTextPacerTests {
    /// Runs the pacer at 60 frames a second over batches that arrive like the
    /// engine's: `batch` characters every `interval` seconds.
    private func run(batches: [(time: Double, text: String)], until end: Double)
        -> (steps: [Int], pacer: StreamingTextPacer) {
        var pacer = StreamingTextPacer()
        var arrived = ""
        var remaining = batches[...]
        var steps: [Int] = []
        var shown = 0
        var time = 0.0
        while time <= end {
            while let next = remaining.first, next.time <= time {
                arrived += next.text
                pacer.receive(arrived, at: next.time)
                remaining = remaining.dropFirst()
            }
            if let text = pacer.advance(to: time) {
                steps.append(text.count - shown)
                shown = text.count
            }
            time += 1.0 / 60
        }
        return (steps, pacer)
    }

    @Test func steadyBatchesComeOutEvenly() {
        // 44 characters every 110 ms, about 400 characters a second.
        let batches = (0..<30).map { (time: Double($0) * 0.11,
                                      text: String(repeating: "a", count: 44)) }
        let (steps, pacer) = run(batches: batches, until: 3.6)
        // Past the first second the reveal is a few characters a frame, never
        // a whole batch at once.
        let settled = steps.dropFirst(60)
        #expect(!settled.isEmpty)
        #expect(settled.max()! <= 14, "\(settled.max()!)")
        #expect(pacer.isCaughtUp)
    }

    @Test func aPauseThenABurstStaysWithinTheLag() {
        // 10 characters, a one-second gap, then 300 at once.
        let (steps, pacer) = run(batches: [(0, String(repeating: "a", count: 10)),
                                           (1.0, String(repeating: "b", count: 300))],
                                 until: 1.6)
        #expect(pacer.isCaughtUp, "still \(pacer.target.count - pacer.displayed.count) behind")
        #expect(steps.count > 10, "the burst was spread over frames")
    }

    @Test func charactersAreNeverSplit() {
        var pacer = StreamingTextPacer()
        let text = "Café 👩‍👩‍👧 naïve"
        pacer.receive(text, at: 0)
        var time = 0.0
        var seen: [String] = []
        while !pacer.isCaughtUp {
            time += 1.0 / 120
            if let shown = pacer.advance(to: time) { seen.append(shown) }
        }
        #expect(seen.last == text)
        #expect(seen.allSatisfy { text.hasPrefix($0) })
    }

    @Test func flushShowsEverythingAndANewAnswerStartsOver() {
        var pacer = StreamingTextPacer()
        pacer.receive("first answer", at: 0)
        _ = pacer.advance(to: 0)
        _ = pacer.advance(to: 0.016)
        #expect(pacer.flush() == "first answer")
        pacer.receive("second", at: 1)
        #expect(pacer.displayed.isEmpty)
        #expect(pacer.target == "second")
    }

    /// Text that completes a character in the next batch (an accent, a flag's
    /// second half, an emoji joiner) extends the answer; it must not look
    /// like a new answer and replay from the start.
    @Test(arguments: [("Caf", "e", "\u{301} au lait"),
                      ("Go ", "\u{1F1EB}", "\u{1F1F7} now"),
                      ("Hi ", "\u{1F469}", "\u{200D}\u{1F4BB} there")])
    func charactersCompletedInTheNextBatchDoNotReplay(parts: (String, String, String)) {
        var pacer = StreamingTextPacer()
        var time = 0.0
        func drain() -> [String] {
            var seen: [String] = []
            for _ in 0..<60 {
                time += 1.0 / 60
                if let shown = pacer.advance(to: time) { seen.append(shown) }
            }
            return seen
        }
        pacer.receive(parts.0 + parts.1, at: time)
        _ = pacer.advance(to: time)
        let before = drain()
        #expect(before.last == parts.0 + parts.1)
        let full = parts.0 + parts.1 + parts.2
        pacer.receive(full, at: time)
        // At most the half-finished character is stepped back, never more.
        #expect(pacer.displayed.utf8.count >= parts.0.utf8.count)
        let after = drain()
        #expect(after.last == full)
        #expect(after.allSatisfy { full.utf8.starts(with: $0.utf8) })
    }
    @Test func longMarkdownFlushAndRestartPreserveExactBytes() {
        var pacer = StreamingTextPacer()
        let answer = String(repeating: "Reasoning: Cafe\u{301} 👩‍💻\n```swift\nlet value = 42\n```\n", count: 400)
        var arrived = ""
        var time = 0.0
        for character in answer {
            arrived.append(character)
            time += 0.0001
            pacer.receive(arrived, at: time)
            if let shown = pacer.advance(to: time) {
                #expect(answer.utf8.starts(with: shown.utf8))
            }
        }
        #expect(pacer.flush().utf8.elementsEqual(answer.utf8))
        // Completion, cancellation and disabling pacing share this flush.
        #expect(pacer.flush().utf8.elementsEqual(answer.utf8))
        pacer.receive("new reply", at: time + 1)
        #expect(pacer.displayed.isEmpty)
        #expect(pacer.flush() == "new reply")
    }

}
