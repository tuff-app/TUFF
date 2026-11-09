import AppKit
import UniformTypeIdentifiers
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// One exchange: what was asked, with the attachments it was asked with, and
/// what came back — from the model named on the answer.
///
/// The same view draws a finished exchange and the one being generated. Only
/// the answer differs: a finished one is laid out once from rendered Markdown,
/// while the live one streams through `StreamingResponseView`. Everything
/// around it — the images, the files, the reasoning — belongs to the message,
/// which is why it is drawn here rather than above the conversation.
struct TranscriptMessageView: View {
    let prompt: String
    let response: String
    let thinking: String
    let images: [AppImageAttachment]
    let documents: [AppDocumentAttachment]
    let modelName: String
    let renderer: ResponseMarkdownRenderer
    @Environment(\.appFontScale) private var fontScale
    /// Nil for a finished message. Present while this message is the live one.
    var live: LiveResponse?
    /// What this message can be rewound to. Absent while a run is in flight,
    /// and on the live message, which has nothing to rewind yet.
    var actions: MessageActions?
    /// Tool calls made for this answer, and the sources they found.
    var toolEntries: [ToolActivityEntry] = []
    var sources: [AppSource] = []

    struct MessageActions {
        let edit: () -> Void
        let regenerate: () -> Void
        /// Present only on the newest message, and only when the answer looks
        /// cut off rather than finished.
        let carryOn: (() -> Void)?
    }

    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var thinkingExpanded = false
    @State private var liveHeight: CGFloat = 0
    /// Which of the two copy buttons last fired, so only that one shows the
    /// checkmark. A single flag lit both.
    @State private var copied: AnswerCopyFormat?

    struct LiveResponse {
        let mailbox: GenerationTranscriptMailbox?
        let isTerminal: Bool
        let showsPrefillPlaceholder: Bool
        var smoothsText = true
    }

    /// False until a message that is being generated has come on screen.
    /// Finished messages start shown: they are made again every time they
    /// scroll into view, and must not fade in each time.
    @State private var hasArrived = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if hasUserMessage { userMessage }
            if hasAssistantMessage { assistantMessage }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        // A message just sent rises into place. Through the modifier, not
        // `withAnimation`: the window drops explicit animations while a
        // model is answering.
        .opacity(arrives && !hasArrived ? 0 : 1)
        .offset(y: arrives && !hasArrived && !reduceMotion ? 8 : 0)
        .animation(TUFFMotion.standard, value: hasArrived)
        .onAppear {
            if arrives { hasArrived = true }
        }
    }

    private var arrives: Bool { live?.isTerminal == false }

    private var hasUserMessage: Bool {
        !prompt.isEmpty || !images.isEmpty || !documents.isEmpty
    }

    private var hasAssistantMessage: Bool {
        live != nil || !response.isEmpty || !thinking.isEmpty
    }

    private var userMessage: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 4) {
                MessageRoleLabel(text: "You")
                if let actions {
                    MessageActionButton(
                        symbol: "pencil",
                        help: "Edit this message and send it again",
                        action: actions.edit)
                }
                Spacer(minLength: 0)
            }
            MessageAttachmentsView(images: images, documents: documents)
            if !prompt.isEmpty {
                // Rendered the same way an answer is: a prompt with a list, a
                // code block, or a formula in it was the one half of the
                // conversation still shown as raw source.
                AttributedTextView(
                    attributed: renderer.render(prompt).attributedString)
            }
        }
    }

    private var assistantMessage: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 4) {
                MessageRoleLabel(text: modelName)
                if showsActions { answerActions.transition(.opacity) }
                Spacer(minLength: 0)
            }
            .animation(TUFFMotion.standard, value: showsActions)
            if !thinking.isEmpty {
                ThinkingDisclosure(
                    text: thinking,
                    isRunning: live?.isTerminal == false && response.isEmpty
                        && !toolEntries.contains { $0.state == .running },
                    isExpanded: $thinkingExpanded,
                    reduceTransparency: reduceTransparency)
                    .transition(.opacity)
            }
            if !toolEntries.isEmpty {
                ToolActivityView(
                    entries: toolEntries, sources: sources,
                    isRunning: toolEntries.contains { $0.state == .running },
                    reduceTransparency: reduceTransparency)
                    .transition(.opacity)
            }
            responseBody
            if let note = citationNote {
                Label(note, systemImage: "exclamationmark.circle")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .transition(.opacity)
            }
        }
        // The reasoning and tool rows come in as the model starts on them,
        // rather than pushing the answer down in one step.
        .animation(TUFFMotion.standard, value: thinking.isEmpty)
        .animation(TUFFMotion.standard, value: toolEntries.isEmpty)
        .animation(TUFFMotion.standard, value: citationNote)
    }

    /// Said when the answer cites a number no retrieved source has. The
    /// number stays plain text; nothing is made to look like a source.
    private var citationNote: String? {
        guard live == nil, !toolEntries.isEmpty, !response.isEmpty else { return nil }
        let invalid = AppCitations.check(response, sources: sources).invalid
        guard !invalid.isEmpty else { return nil }
        let list = invalid.map { "[\($0)]" }.joined(separator: ", ")
        return invalid.count == 1
            ? "\(list) does not match a retrieved source."
            : "\(list) do not match retrieved sources."
    }

    private func openCitation(_ url: URL) {
        guard let source = AppCitations.source(for: url, in: sources),
              let target = source.openURL else { return }
        if target.isFileURL {
            NSWorkspace.shared.activateFileViewerSelecting([target])
        } else {
            NSWorkspace.shared.open(target)
        }
    }

    /// Copy, copy as Markdown, regenerate and continue, beside the name of
    /// whoever produced the answer. They used to sit under it and appear only
    /// on hover, which meant the reader had to know they were there before
    /// they would show.
    private var answerActions: some View {
        // Touching: each button's 28 pt target already includes its spacing.
        HStack(spacing: 0) {
            MessageActionButton(
                symbol: copied == .text ? "checkmark" : "doc.on.doc",
                help: copied == .text ? "Copied" : "Copy this answer",
                tint: copied == .text ? TUFFMacTheme.accentTextColor : nil,
                action: { copyResponse(.text) })
            MessageActionButton(
                symbol: copied == .markdown
                    ? "checkmark" : "chevron.left.forwardslash.chevron.right",
                help: copied == .markdown
                    ? "Copied" : "Copy this answer as Markdown",
                tint: copied == .markdown ? TUFFMacTheme.accentTextColor : nil,
                action: { copyResponse(.markdown) })
            MessageActionButton(
                symbol: "square.and.arrow.down",
                help: "Save this answer and its sources as a Markdown file",
                action: saveResponse)
            if let actions {
                MessageActionButton(
                    symbol: "arrow.clockwise",
                    help: "Send this message again for a different answer",
                    action: actions.regenerate)
                if let carryOn = actions.carryOn {
                    MessageActionButton(
                        symbol: "text.append",
                        help: "Carry on from where this answer stopped",
                        tint: TUFFMacTheme.accentTextColor,
                        action: carryOn)
                }
            }
        }
        .task(id: copied) {
            guard copied != nil else { return }
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            copied = nil
        }
    }

    private var showsActions: Bool {
        !response.isEmpty && live?.isTerminal != false
    }

    @ViewBuilder
    private var responseBody: some View {
        if let live {
            StreamingResponseView(
                text: response,
                mailbox: live.mailbox,
                isTerminal: live.isTerminal,
                showsPrefillPlaceholder: live.showsPrefillPlaceholder,
                smoothsText: live.smoothsText,
                scale: fontScale,
                height: $liveHeight)
                // The live answer's text is built at one scale, so changing
                // the zoom rebuilds it rather than leaving it at the old size.
                .id(fontScale)
                .frame(height: max(liveHeight, 1))
        } else if !response.isEmpty {
            AttributedTextView(
                attributed: sources.isEmpty
                    ? renderer.render(response).attributedString
                    : renderer.render(response) { text in
                        AppCitations.numberRanges(in: text, sources: sources).map {
                            ($0.range, AppCitations.url(for: $0.number))
                        }
                    },
                onLink: openCitation)
        } else if !thinking.isEmpty, live?.isTerminal != false {
            // Stopped while still thinking: say so rather than leave an
            // empty answer under a collapsed row.
            Text("Stopped before answering. The reasoning so far is under Thinking.")
                .appFont(.callout)
                .foregroundStyle(.secondary)
        }
    }

    /// Writes the answer only where the person chooses in the save panel.
    private func saveResponse() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = AppAnswerExport.suggestedFileName(prompt: prompt)
        panel.allowedContentTypes = [.init(filenameExtension: "md") ?? .plainText]
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let text = AppAnswerExport.markdown(prompt: prompt, response: response,
                                            modelName: modelName, sources: sources)
        try? text.write(to: url, atomically: true, encoding: .utf8)
    }

    private func copyResponse(_ format: AnswerCopyFormat) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(
            format.pasteboardText(for: response, renderer: renderer),
            forType: .string)
        copied = format
    }
}

/// One icon-only control on a message: copy, edit, regenerate, continue.
///
/// Small, quiet, and always present. Text labels beside the role name made the
/// line read as a sentence rather than a heading with tools on it.
struct MessageActionButton: View {
    let symbol: String
    let help: String
    var tint: Color?
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .appFont(.callout)
                .contentTransition(.symbolEffect(.replace))
                // The glyph stays small; the target is a comfortable 28 pt.
                .frame(width: 28, height: 28)
                .contentShape(Rectangle())
                .tuffHoverGlass()
        }
        .buttonStyle(.plain)
        .foregroundStyle(tint ?? .secondary)
        .help(help)
        .accessibilityLabel(help)
    }
}

/// Who is speaking. Both sides are the app's purple: the conversation is one
/// surface, and colouring only one half of it made the other read as chrome.
struct MessageRoleLabel: View {
    let text: String

    var body: some View {
        Text(text)
            .appFont(.caption.weight(.semibold))
            .foregroundStyle(TUFFMacTheme.accentTextColor)
            .lineLimit(1)
            .accessibilityAddTraits(.isHeader)
    }
}

/// Collapsed reasoning that belongs to one message.
///
/// Expanded, it flows into the message rather than into a box of its own that
/// scrolls: one conversation, one scroll.
struct ThinkingDisclosure: View {
    let text: String
    let isRunning: Bool
    @Binding var isExpanded: Bool
    let reduceTransparency: Bool

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            Text(text)
                .appFont(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 8)
        } label: {
            Label(isRunning ? "Thinking…" : "Thinking", systemImage: "brain")
                .appFont(.caption.weight(.medium))
                .foregroundStyle(.secondary)
                .contentTransition(.opacity)
                .animation(TUFFMotion.standard, value: isRunning)
        }
        .padding(10)
        .tuffPanel()
        .accessibilityLabel("Model thinking")
        .accessibilityHint("Shows or hides the reasoning for this message")
    }
}
