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
                                 until: 3.0)
        #expect(pacer.isCaughtUp, "still \(pacer.target.count - pacer.displayed.count) behind")
        #expect(steps.count > 10, "the burst was spread over frames")
    }

    /// A slow, disk-bound model: one short word every 0.18 to 0.5 seconds,
    /// unevenly. The reveal should keep moving between words instead of
    /// finishing each one and waiting for the next.
    @Test func aSlowUnevenModelWritesWithoutStopping() {
        var seed: UInt64 = 42
        func next() -> Double {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Double(seed >> 11) / Double(1 << 53)
        }
        var batches: [(time: Double, text: String)] = []
        var time = 0.0
        for _ in 0..<60 {
            batches.append((time, "word "))
            time += 0.18 + 0.32 * next()
        }
        var pacer = StreamingTextPacer()
        var arrived = ""
        var remaining = batches[...]
        var frame = 0.0
        var idle = 0
        var longestIdle = 0
        var idleFrames = 0
        var frames = 0
        while frame <= time {
            while let batch = remaining.first, batch.time <= frame {
                arrived += batch.text
                pacer.receive(arrived, at: batch.time)
                remaining = remaining.dropFirst()
            }
            let moved = pacer.advance(to: frame) != nil
            if frame > 3 {
                frames += 1
                if moved { idle = 0 } else { idle += 1; idleFrames += 1 }
                longestIdle = max(longestIdle, idle)
            }
            frame += 1.0 / 60
        }
        // At about 12 characters a second a character is due every 5 frames.
        // The earlier pacer finished each word early and then sat still for
        // up to 15 frames waiting for the next one.
        #expect(longestIdle <= 8, "longest pause \(longestIdle) frames")
        #expect(frames > 0 && idleFrames < frames)
        #expect(pacer.target.count - pacer.displayed.count < 20)
    }

    /// A long pause (a tool call, a long think) must not leave the next
    /// words crawling out at the rate the pause implies.
    @Test func textAfterALongPauseIsNotHeldBack() {
        var pacer = StreamingTextPacer()
        pacer.receive("start", at: 0)
        var time = 0.0
        while time < 1 { time += 1.0 / 60; _ = pacer.advance(to: time) }
        #expect(pacer.isCaughtUp)
        // 60 seconds later, a few characters, then nothing more for a while.
        pacer.receive("start then", at: 60)
        time = 60
        while time < 61.5, !pacer.isCaughtUp { time += 1.0 / 60; _ = pacer.advance(to: time) }
        #expect(pacer.isCaughtUp, "\(pacer.target.count - pacer.displayed.count) still hidden")
        // Also when the pause is shorter than the window.
        pacer.receive("start then more", at: 62.9)
        time = 62.9
        while time < 64.5, !pacer.isCaughtUp { time += 1.0 / 60; _ = pacer.advance(to: time) }
        #expect(pacer.isCaughtUp, "\(pacer.target.count - pacer.displayed.count) still hidden")
    }

    /// A large batch followed by a slow trickle: the trickle's rate must not
    /// slow the reveal of the batch that is already waiting.
    @Test func aBigBatchIsShownWithinTheCatchUpWindow() {
        var pacer = StreamingTextPacer()
        let big = String(repeating: "a", count: 1_000)
        pacer.receive(big, at: 0)
        var time = 0.0
        _ = pacer.advance(to: time)
        var arrived = big
        var nextArrival = 0.1
        while time < 2.0 {
            time += 1.0 / 60
            if time >= nextArrival {
                arrived += "b"
                pacer.receive(arrived, at: time)
                nextArrival += 0.1
            }
            _ = pacer.advance(to: time)
        }
        #expect(pacer.target.count - pacer.displayed.count < 15,
                "\(pacer.target.count - pacer.displayed.count) still hidden")
    }

    /// Text that arrives after the window stopped drawing for a while is
    /// revealed from its arrival, not all at once on the first frame.
    @Test func textArrivingAfterIdleFramesIsNotFlashed() {
        var pacer = StreamingTextPacer()
        pacer.receive("ready", at: 0)
        var time = 0.0
        while time < 1 { time += 1.0 / 60; _ = pacer.advance(to: time) }
        #expect(pacer.isCaughtUp)
        // No frames for five seconds, then 200 characters arrive.
        let more = "ready" + String(repeating: "x", count: 200)
        pacer.receive(more, at: 6)
        let first = pacer.advance(to: 6 + 1.0 / 60)
        #expect((first?.count ?? 5) - 5 < 100, "first frame showed \((first?.count ?? 5) - 5)")
    }

    /// A big batch then one character on every frame: the trickle must not
    /// push the big batch's deadline back.
    @Test func aBatchKeepsItsDeadlineUnderAPerFrameTrickle() {
        var pacer = StreamingTextPacer()
        var arrived = String(repeating: "a", count: 1_000)
        pacer.receive(arrived, at: 0)
        var time = 0.0
        _ = pacer.advance(to: time)
        while time < 1.6 {
            time += 1.0 / 60
            arrived += "b"
            pacer.receive(arrived, at: time)
            _ = pacer.advance(to: time)
        }
        #expect(pacer.displayed.count >= 1_000, "only \(pacer.displayed.count) shown")
    }

    /// Catching up on an overdue batch must not leave credit that shows the
    /// next batch all at once.
    @Test func overdueCatchUpLeavesNoCredit() {
        var pacer = StreamingTextPacer()
        let first = String(repeating: "a", count: 1_000)
        pacer.receive(first, at: 0)
        _ = pacer.advance(to: 0)
        _ = pacer.advance(to: 2)
        #expect(pacer.isCaughtUp)
        pacer.receive(first + String(repeating: "b", count: 200), at: 2)
        let shown = pacer.advance(to: 2 + 1.0 / 60)
        #expect((shown?.count ?? 1_000) - 1_000 < 100)
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
            for _ in 0..<150 {
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
