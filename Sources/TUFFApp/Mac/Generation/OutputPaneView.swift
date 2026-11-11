import AppKit
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// The conversation: one scrolling surface holding every message.
///
/// Each exchange is a view that sizes to its own content, including the one
/// being generated. Nothing inside a message scrolls on its own — the reasoning
/// disclosure and the attachment row used to, which put a scroller inside a
/// scroller and made the wheel do different things depending on what the
/// pointer was over.
struct OutputPaneView: View {
    let model: AppModel
    /// Height of the glass composer floating over the bottom of the
    /// transcript; the last message scrolls clear of it.
    var bottomInset: CGFloat = 0
    @Environment(\.appFontScale) private var fontScale
    /// True while the view should keep itself at the newest message. Set by the
    /// reader arriving at the bottom and cleared by the reader leaving it, so
    /// following the answer never fights someone reading back.
    @State private var isPinnedToBottom = true

    var body: some View {
        Group {
            if model.hasOutputTranscript {
                transcript
            } else {
                placeholder
            }
        }
        .contextMenu {
            Button("Copy conversation") {
                copy(model.conversationPlainText)
            }
            .disabled(model.conversationPlainText.isEmpty)

            Divider()

            Button("New Chat") { model.clearOutput() }
                .disabled(model.isSelectedConversationRunning || !model.hasOutputTranscript)
        }
    }

    private var placeholder: some View {
        EmptyConversationLayout(spacing: 8) {
            TUFFLoadingMark(stage: markStage)

            emptyPlaceholderContent
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private static let bottomID = "tuff.transcript.bottom"

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 26) {
                    ForEach(model.conversation) { turn in
                        recordedMessage(turn)
                    }
                    if model.hasLiveMessage {
                        liveMessage.id(model.runIdentity)
                    }
                    // The scroll target. Anchoring on the last message instead
                    // stopped short whenever that message was taller than the
                    // viewport, which is most of them.
                    Color.clear
                        .frame(height: bottomInset)
                        .accessibilityHidden(true)
                    Color.clear
                        .frame(height: 1)
                        .id(Self.bottomID)
                        .accessibilityHidden(true)
                }
                .padding(.horizontal, StatusHUDView.contentInset)
                .padding(.vertical, 22)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .onScrollPhaseChange { _, phase in
                // A programmatic scroll reports `.animating`, so this reads the
                // reader only.
                if phase == .interacting || phase == .tracking {
                    isPinnedToBottom = false
                }
            }
            .onScrollGeometryChange(for: TranscriptGeometry.self) { geometry in
                TranscriptGeometry(
                    offset: geometry.contentOffset.y,
                    contentHeight: geometry.contentSize.height,
                    viewportHeight: geometry.containerSize.height)
            } action: { previous, current in
                if current.isAtBottom {
                    isPinnedToBottom = true
                } else if isPinnedToBottom,
                          current.contentHeight > previous.contentHeight {
                    // The answer grew under a view that was at the bottom.
                    // Deferred to the next turn of the run loop: scrolling from
                    // inside this callback runs during layout, and the lazy
                    // stack measuring newly visible rows changed the height
                    // again, until AppKit stopped the window for looping.
                    DispatchQueue.main.async {
                        if isPinnedToBottom { scrollToBottom(proxy) }
                    }
                }
            }
            .onChange(of: model.runIdentity) { _, _ in
                // A new message is the thing that was just asked for: it goes
                // on screen whatever the reader was looking at.
                isPinnedToBottom = true
                scrollToBottom(proxy)
            }
        }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        var transaction = Transaction()
        transaction.disablesAnimations = true
        withTransaction(transaction) {
            proxy.scrollTo(Self.bottomID, anchor: .bottom)
        }
    }

    private func recordedMessage(_ turn: AppChatTurn) -> some View {
        TranscriptMessageView(
            prompt: turn.prompt,
            response: turn.response,
            thinking: Self.displayedThinking(turn),
            images: turn.images,
            documents: turn.documents,
            modelName: model.modelShortName(forProfileKey: turn.modelID),
            renderer: Self.renderer(scale: fontScale),
            actions: actions(for: turn),
            toolEntries: ToolActivityEntry.entries(from: turn.toolRounds),
            sources: turn.sources)
    }

    /// Every round's reasoning, then the answer's, in the order produced.
    private static func displayedThinking(_ turn: AppChatTurn) -> String {
        (turn.toolRounds.compactMap(\.thinking) + [turn.thinking ?? ""])
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }

    /// Rewind controls, offered only while the chat is idle enough to rewind.
    private func actions(for turn: AppChatTurn) -> TranscriptMessageView.MessageActions? {
        guard model.canRewind(to: turn) else { return nil }
        let isNewest = model.conversation.last?.id == turn.id
        return TranscriptMessageView.MessageActions(
            edit: { model.editMessage(turn) },
            regenerate: { model.regenerate(turn) },
            carryOn: isNewest && model.canContinueLastAnswer
                ? { model.continueLastAnswer() } : nil)
    }

    private var liveMessage: some View {
        TranscriptMessageView(
            prompt: model.outputPromptText,
            response: model.outputResponsePlainText,
            thinking: model.outputThinkingText,
            images: model.outputImageAttachments,
            documents: model.outputDocumentAttachments,
            modelName: model.selectedModelName,
            renderer: Self.renderer(scale: fontScale),
            live: TranscriptMessageView.LiveResponse(
                mailbox: model.hasLiveMessage
                    ? model.generationTranscriptMailbox : nil,
                isTerminal: !model.isRunning,
                // Only until something comes back: once the model is
                // thinking, the Thinking row says so.
                showsPrefillPlaceholder: model.isRunning
                    && model.outputResponsePlainText.isEmpty
                    && model.outputThinkingText.isEmpty
                    && model.phase != .tools,
                smoothsText: model.smoothStreaming),
            toolEntries: ToolActivityEntry.entries(from: model.outputToolActivities),
            sources: model.outputSources)
    }

    /// One renderer per zoom. The point sizes are baked into the attributed
    /// string it produces, so a renderer belongs to the scale it was made for;
    /// they are cached rather than rebuilt because every transcript update
    /// renders through one.
    @MainActor
    private static var renderersByScale: [CGFloat: ResponseMarkdownRenderer] = [:]

    @MainActor
    private static func renderer(scale: CGFloat) -> ResponseMarkdownRenderer {
        if let cached = renderersByScale[scale] { return cached }
        let renderer = ResponseMarkdownRenderer(scale: scale)
        renderersByScale[scale] = renderer
        return renderer
    }

    /// What the empty chat is saying, so a change from one to the next
    /// (load, loading, failed, ready) can fade instead of swapping.
    private var placeholderStage: Int {
        if isLoadingModel { return 1 }
        if model.loadState.isFailed { return 2 }
        if model.hasStaleLoadedRuntime { return 3 }
        return needsModelLoad ? 0 : 4
    }

    private var emptyPlaceholderContent: some View {
        VStack(spacing: 8) {
            if case .loading(let phase) = model.loadState {
                Text(phase.label)
                    .appFont(.callout)
                    .foregroundStyle(TUFFDesign.secondaryTextColor)
                    .contentTransition(.opacity)
                    .animation(TUFFMotion.standard, value: phase)
            } else if let placeholderHint {
                Text(placeholderHint)
                    .appFont(.callout)
                    .foregroundStyle(.tertiary)
            }
            if let detail = model.presentation.detail {
                Text(detail)
                    .appFont(.caption)
                    .foregroundStyle(model.presentation.severity == .error ? TUFFMacTheme.accentTextColor : .secondary)
                    .multilineTextAlignment(.center)
                    .transition(.opacity)
            }
            if model.canLoadModel {
                Button(model.loadState.isFailed ? "Retry Load" : "Load Model",
                       action: model.loadModel)
                    .buttonStyle(.tuffProminent)
                    .controlSize(.large)
                    .transition(.tuffArrival(scale: 0.96))
            } else if isLoadingModel {
                Button("Load Model", action: {})
                    .buttonStyle(.tuffProminent)
                    .controlSize(.large)
                    .hidden()
                    .accessibilityHidden(true)
            } else if model.canReloadModel {
                Button("Reload Model", action: model.reloadModel)
                    .buttonStyle(.tuffProminent)
                    .transition(.tuffArrival(scale: 0.96))
            }
        }
        .frame(maxWidth: .infinity)
        .animation(TUFFMotion.standard, value: placeholderStage)
        .animation(TUFFMotion.standard, value: model.presentation.detail)
    }

    private var needsModelLoad: Bool {
        !model.isAppleSelected && !model.loadState.isReady
    }

    private var isLoadingModel: Bool {
        if case .loading = model.loadState { return true }
        return false
    }

    private var markStage: TUFFLoadingMark.Stage {
        switch model.loadState {
        case .loading(let phase): .loading(phase.progressFraction)
        case .ready: .ready
        default: .idle
        }
    }

    private var placeholderHint: String? {
        if model.loadState.isFailed { return "The model could not be loaded" }
        if model.hasStaleLoadedRuntime { return "Reload the model to use changed settings" }
        return needsModelLoad ? "Load the model to begin" : nil
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// What the transcript needs from its scroll view to decide whether to follow.
private struct TranscriptGeometry: Equatable {
    var offset: CGFloat = 0
    var contentHeight: CGFloat = 0
    var viewportHeight: CGFloat = 0

    /// A band rather than an exact match: a streaming answer grows by a line at
    /// a time, and an exact test read every one of those as the reader leaving.
    var isAtBottom: Bool {
        offset + viewportHeight >= contentHeight - 32
    }
}

struct SubmittedImageThumbnail: View {
    let attachment: AppImageAttachment
    let maximumSize: CGSize
    @State private var image: NSImage?

    init(
        attachment: AppImageAttachment,
        maximumSize: CGSize = CGSize(width: 48, height: 48)
    ) {
        self.attachment = attachment
        self.maximumSize = maximumSize
    }

    var body: some View {
        Group {
            if let image {
                // Filled and cropped to a tile, not fitted inside one. Fitting
                // gave every attachment a different height — a screenshot came
                // out a third the height of a portrait photo — so a row of them
                // was ragged, and the remove badge, pinned to the tile, floated
                // clear of the short ones.
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: maximumSize.width, height: maximumSize.height)
                    .clipped()
            } else {
                Image(systemName: "photo")
                    .foregroundStyle(.tertiary)
                    .frame(width: maximumSize.width, height: maximumSize.height)
            }
        }
        .frame(width: maximumSize.width, height: maximumSize.height)
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .stroke(TUFFDesign.hairlineColor, lineWidth: 1)
        }
        .accessibilityLabel("Attached image \(attachment.displayName)")
        .help(attachment.displayName)
        .task(id: "\(attachment.id)-\(maximumSize.width)x\(maximumSize.height)") {
            let url = attachment.fileURL
            let key = attachment.sha256
            let pixels = Int(ceil(max(maximumSize.width, maximumSize.height) * 2))
            // This closure runs on the main actor, and a source near the decode
            // budget takes long enough that decoding here stalled the window.
            // The decode is done off the main thread and left in the cache; the
            // read below is then a lookup.
            await Task.detached(priority: .userInitiated) {
                _ = Self.loadThumbnail(
                    at: url, maximumPixelSize: pixels, cacheKey: key)
            }.value
            image = Self.loadThumbnail(
                at: url, maximumPixelSize: pixels, cacheKey: key)
        }
    }

    /// Every attached-image decode in the app goes through here, so the decode
    /// budget cannot be applied to one caller and forgotten on the next. Nil
    /// means refused or unreadable; both callers draw their placeholder for it.
    nonisolated static func loadThumbnail(
        at url: URL,
        maximumPixelSize: Int,
        cacheKey: String? = nil
    ) -> NSImage? {
        TranscriptImageLoader.thumbnail(
            at: url,
            maximumPixelSize: maximumPixelSize,
            budget: decodeBudget,
            cacheKey: cacheKey)
    }

    /// The single point where the app binds the runtime's limits, so the two
    /// cannot drift apart: see `VisionImageLimits` in
    /// Runtime/Vision/Preprocessing/ImageMetadataReader.swift.
    nonisolated private static let decodeBudget: TranscriptImageLoader.Budget = {
        let limits = VisionImageLimits()
        return TranscriptImageLoader.Budget(
            maximumSourcePixels: limits.maximumSourcePixels,
            maximumSourceDimension: limits.maximumSourceDimension,
            maximumDecodedBytes: limits.maximumDecodedBytes,
            allowedTypeIdentifiers: limits.allowedTypeIdentifiers)
    }()
}

private struct EmptyConversationLayout: Layout {
    let spacing: CGFloat

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        proposal.replacingUnspecifiedDimensions()
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        guard subviews.count == 2 else { return }

        let iconSize = subviews[0].sizeThatFits(.unspecified)
        let iconCenter = CGPoint(x: bounds.midX, y: bounds.midY)
        subviews[0].place(
            at: iconCenter,
            anchor: .center,
            proposal: ProposedViewSize(
                width: iconSize.width,
                height: iconSize.height))

        subviews[1].place(
            at: CGPoint(
                x: bounds.midX,
                y: iconCenter.y + iconSize.height / 2 + spacing),
            anchor: .top,
            proposal: ProposedViewSize(width: bounds.width, height: nil))
    }
}

#if DEBUG && !TUFF_NO_PREVIEWS
private struct MessagePreview: View {
    let prompt: String
    let response: String
    var thinking = ""

    var body: some View {
        TranscriptMessageView(
            prompt: prompt,
            response: response,
            thinking: thinking,
            images: [],
            documents: [],
            modelName: "Gemma 4 E4B",
            renderer: ResponseMarkdownRenderer())
            .padding(24)
            .frame(width: 720)
    }
}

#Preview("Empty") {
    VStack(spacing: 8) {
        Image(systemName: "cube.transparent")
            .appFont(.title2)
            .foregroundStyle(.quaternary)
    }
    .frame(width: 720, height: 420)
}

#Preview("Completed prose") {
    MessagePreview(
        prompt: "Explain this clearly.",
        response: "# A clear answer\n\nHere is a concise explanation with **useful emphasis**.\n\n- First point\n- Second point")
}

#Preview("Completed code") {
    MessagePreview(
        prompt: "Write `fibonacci`.",
        response: "Use `fibonacci(7)`:\n\n```python\ndef fibonacci(n: int) -> list[int]:\n    return []\n```")
}

#Preview("With reasoning") {
    MessagePreview(
        prompt: "What is 17 × 23?",
        response: "391.",
        thinking: "17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391.")
}
#endif
