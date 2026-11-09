import AppKit
import QuartzCore
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// The answer currently being generated.
///
/// Tokens arrive faster than SwiftUI wants to rebuild a view, so this one keeps
/// the streaming fast path: a mailbox drained on every display frame, the new
/// suffix appended to the text storage, and the whole response re-rendered as
/// Markdown exactly once, when the run ends. With smoothing on, a pacer reveals
/// the arrived text at a steady rate instead of in the engine's batches. It reports the height it needs, so it sits
/// in the transcript's scroll view as an ordinary message rather than bringing
/// a second scroll view along with it.
struct StreamingResponseView: NSViewRepresentable {
    /// The response as the model state knows it. The mailbox is the fast path;
    /// this is the fallback and the source of truth when there is no mailbox.
    var text: String
    var mailbox: GenerationTranscriptMailbox?
    var isTerminal: Bool
    var showsPrefillPlaceholder: Bool
    var smoothsText: Bool = true
    /// The zoom. The streamed answer is an attributed string built with baked
    /// point sizes, so the controller is made for one scale and the view is
    /// rebuilt (see the `id` at the call site) if that changes.
    var scale: CGFloat
    @Binding var height: CGFloat

    @MainActor
    final class Coordinator: NSObject {
        weak var textView: MessageTextView?
        var mailbox: GenerationTranscriptMailbox?
        var isTerminal = false
        var showsPrefillPlaceholder = false
        var smoothsText = true
        var pacer = StreamingTextPacer()
        /// The placeholder request last handed to the document controller.
        private var appliedPlaceholderRequest: Bool?
        private var appliedTerminal = false
        var displayLink: CADisplayLink?
        var prefillAnimationTimer: Timer?
        let documentController: StreamingResponseDocumentController

        init(scale: CGFloat) {
            documentController = StreamingResponseDocumentController(scale: scale)
        }

        func attach(_ textView: MessageTextView) {
            self.textView = textView
            guard displayLink == nil else { return }
            // Drawn in step with the screen rather than on a 100 ms timer,
            // which beat against the engine's own 100 ms batches.
            let link = textView.displayLink(target: self, selector: #selector(frame))
            link.add(to: .main, forMode: .common)
            displayLink = link
        }

        func synchronize(
            text: String,
            mailbox: GenerationTranscriptMailbox?,
            isTerminal: Bool,
            showsPrefillPlaceholder: Bool,
            smoothsText: Bool
        ) {
            self.mailbox = mailbox
            self.isTerminal = isTerminal
            self.showsPrefillPlaceholder = showsPrefillPlaceholder
            self.smoothsText = smoothsText
            render(complete: mailbox?.drain().completeText ?? text)
        }

        @objc private func frame() {
            render(complete: mailbox?.drain().completeText ?? pacer.target)
        }

        /// Shows `complete` (everything that has arrived), paced unless
        /// smoothing is off or the run is over, which show it all at once.
        private func render(complete: String) {
            let now = CACurrentMediaTime()
            pacer.receive(complete, at: now)
            if !smoothsText || isTerminal {
                let all = pacer.flush()
                if !all.utf8.elementsEqual(documentController.response.utf8)
                    || appliedPlaceholderRequest != showsPrefillPlaceholder
                    || isTerminal != appliedTerminal {
                    apply(response: all)
                }
                return
            }
            if let shown = pacer.advance(to: now) {
                apply(response: shown)
            } else if pacer.displayed.isEmpty, !documentController.response.isEmpty {
                // A new answer started: clear the old one.
                apply(response: "")
            } else if appliedPlaceholderRequest != showsPrefillPlaceholder {
                apply(response: pacer.displayed)
            }
        }

        @objc private func animatePrefillPlaceholderIfNeeded() {
            guard documentController.showsPrefillPlaceholder,
                  let textView,
                  let storage = textView.textStorage else { return }
            let selection = textView.selectedRanges.map(\.rangeValue)
            storage.beginEditing()
            let changed = documentController.advancePrefillAnimation(storage: storage)
            storage.endEditing()
            guard changed else { return }
            restoreSelection(selection, in: textView, storage: storage)
            textView.reportHeight()
        }

        private func apply(response: String) {
            guard let textView, let storage = textView.textStorage else { return }
            let selection = textView.selectedRanges.map(\.rangeValue)
            appliedPlaceholderRequest = showsPrefillPlaceholder
            appliedTerminal = isTerminal
            storage.beginEditing()
            let update = documentController.synchronize(
                storage: storage,
                response: response,
                isTerminal: isTerminal,
                showsPrefillPlaceholder: showsPrefillPlaceholder)
            storage.endEditing()
            updatePrefillAnimationTimer()
            guard update.mutation != .none else { return }
            restoreSelection(selection, in: textView, storage: storage)
            textView.reportHeight()
        }

        private func restoreSelection(
            _ selection: [NSRange],
            in textView: NSTextView,
            storage: NSTextStorage
        ) {
            let restored = StreamingResponseDocumentController.clampedRanges(
                selection, toLength: storage.length)
            if restored.isEmpty {
                textView.setSelectedRange(NSRange(location: storage.length, length: 0))
            } else {
                textView.selectedRanges = restored.map(NSValue.init(range:))
            }
        }

        private func updatePrefillAnimationTimer() {
            if documentController.showsPrefillPlaceholder {
                guard prefillAnimationTimer == nil else { return }
                let timer = Timer(
                    timeInterval: 0.25,
                    target: self,
                    selector: #selector(animatePrefillPlaceholderIfNeeded),
                    userInfo: nil,
                    repeats: true)
                timer.tolerance = 0.025
                RunLoop.main.add(timer, forMode: .common)
                prefillAnimationTimer = timer
            } else {
                prefillAnimationTimer?.invalidate()
                prefillAnimationTimer = nil
            }
        }

        func invalidate() {
            displayLink?.invalidate()
            displayLink = nil
            prefillAnimationTimer?.invalidate()
            prefillAnimationTimer = nil
            mailbox = nil
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(scale: scale) }

    func makeNSView(context: Context) -> MessageTextView {
        let textView = MessageTextView.make()
        textView.setAccessibilityLabel("Model answer")
        textView.onHeightChange = { measured in height = measured }
        return textView
    }

    func updateNSView(_ textView: MessageTextView, context: Context) {
        textView.onHeightChange = { measured in height = measured }
        context.coordinator.attach(textView)
        context.coordinator.synchronize(
            text: text,
            mailbox: mailbox,
            isTerminal: isTerminal,
            showsPrefillPlaceholder: showsPrefillPlaceholder,
            smoothsText: smoothsText)
    }

    static func dismantleNSView(_ nsView: MessageTextView, coordinator: Coordinator) {
        coordinator.invalidate()
    }
}
