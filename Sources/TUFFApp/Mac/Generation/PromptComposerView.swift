import AppKit
import TUFFInferenceTypes
import TUFFRuntimeDescriptions
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI
import UniformTypeIdentifiers

struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var isImageDropTargeted = false
    @State private var measuredEditorHeight = PromptComposerMetrics.baseMinimumEditorHeight
    @Environment(\.appFontScale) private var fontScale

    enum AttachmentPickerKind {
        case image
        case document

        var allowedContentTypes: [UTType] {
            switch self {
            case .image: VisionImageLimits().allowedContentTypes
            case .document: DocumentTextExtractor.allowedContentTypes
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if hasAttachments { attachmentStrip.transition(.opacity) }
            if let notice = attachmentNotice { attachmentNoticeText(notice).transition(.opacity) }
            editor
                // Navigate > Message Box (Command-L).
                .onReceive(NotificationCenter.default.publisher(for: .tuffFocusComposer)) { _ in
                    promptFocused = true
                }
            footer
            if let contextNotice { contextNoticeText(contextNotice).transition(.opacity) }
        }
        // The box grows and shrinks around what is added to it, rather than
        // jumping a row at a time.
        .animation(TUFFMotion.standard, value: hasAttachments)
        .animation(TUFFMotion.standard, value: attachmentNotice?.text)
        .animation(TUFFMotion.standard, value: contextNotice?.text)
        .padding(PromptComposerMetrics.padding)
        // Glass over the transcript, which scrolls on underneath. Not
        // interactive: it holds a text field, not a single action.
        .tuffChrome(RoundedRectangle(cornerRadius: PromptComposerMetrics.cornerRadius,
                                     style: .continuous))
    }

    /// Configure every selection mode explicitly: directory pickers and
    /// attachment pickers can reuse AppKit's underlying open panel.
    private func chooseAttachments(_ kind: AttachmentPickerKind) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.treatsFilePackagesAsDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = kind.allowedContentTypes
        panel.prompt = "Open"
        panel.message = ""
        guard panel.runModal() == .OK else { return }
        switch kind {
        case .image: model.addImages(panel.urls)
        case .document: model.attachDocuments(panel.urls)
        }
    }

    private var hasAttachments: Bool {
        !model.imageAttachments.isEmpty || !model.documentAttachments.isEmpty
    }

    private var editor: some View {
        PromptTextEditor(
            text: $model.promptText,
            // Bridge SwiftUI focus state to the binding the AppKit editor uses.
            isFocused: Binding(
                get: { promptFocused },
                set: { promptFocused = $0 }),
            contentHeight: $measuredEditorHeight,
            fontScale: fontScale,
            newlineShortcut: model.newlineShortcut,
            canRun: model.canSubmit,
            canAcceptImages: model.isImageInputAvailable
                && !model.isSelectedConversationRunning
                && !model.isAddingImages,
            // Every model reads text, so a file needs no companion pack — only
            // that a run is not already in flight.
            canAcceptDocuments: !model.isSelectedConversationRunning,
            onSubmit: model.submit,
            onImagesDropped: { model.addImages($0) },
            onDocumentsDropped: { model.attachDocuments($0) },
            onImageDataPasted: model.addImageData,
            onPromisedImagesReceived: { urls, directory in
                model.addImages(urls, discardingSourceDirectory: directory)
            },
            onPromisedImagesFailed: {
                model.reportImageAttachmentError(
                    "That drag did not deliver a file TUFF could read.")
            },
            onUnsupportedImagePaste: {
                model.reportImageAttachmentError(
                    "That clipboard content is not an image TUFF can read.")
            },
            onDropTargeted: { isImageDropTargeted = $0 })
            .accessibilityLabel("Prompt")
            .frame(height: editorHeight)
            .overlay(alignment: .topLeading) {
                if model.promptText.isEmpty {
                    // Matches the NSTextView text origin: 5pt line fragment
                    // padding, no vertical inset.
                    Text("Message \(model.selectedDescriptor.shortName)")
                        .appFont(PromptComposerMetrics.font)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
            .overlay {
                if isImageDropTargeted {
                    RoundedRectangle(cornerRadius: TUFFDesign.Radius.concentric(
                        outer: PromptComposerMetrics.cornerRadius, padding: PromptComposerMetrics.padding))
                        .stroke(.tint, lineWidth: 2)
                        .allowsHitTesting(false)
                        .transition(.opacity)
                }
            }
            .animation(TUFFMotion.hover, value: editorHeight)
            .animation(TUFFMotion.hover, value: isImageDropTargeted)
    }

    /// One line to start with, growing with what is typed and stopping at a
    /// height that still leaves the conversation the larger half of the window.
    /// A fixed box was two or three empty lines tall for every one-line question.
    private var editorHeight: CGFloat {
        min(max(measuredEditorHeight,
                PromptComposerMetrics.minimumEditorHeight(scale: fontScale)),
            PromptComposerMetrics.maximumEditorHeight(scale: fontScale))
    }

    /// What the context meter has to say, and nothing when it has nothing to
    /// say. The limit used to be invisible until it was hit: the renderer
    /// dropped the oldest turns silently, and a file large enough to fill the
    /// window on its own gave no warning before it was sent.
    private var contextNotice: (text: String, isError: Bool)? {
        let usage = model.contextUsage
        if usage.draftAloneOverflows {
            return ("This message alone is about \(MetricFormat.tokens(usage.draftTokens)) "
                + "of the \(MetricFormat.tokens(usage.maxTokens)) this model is "
                + "loaded with. Shorten it, or raise Context in Settings and "
                + "reload.", true)
        }
        if usage.willDropOldestTurns {
            return ("The context is full, so the oldest messages will be left "
                + "out of what the model reads.", false)
        }
        if let dropped = model.lastRunDroppedTurns, dropped > 0 {
            return ("The last answer left out the \(dropped) oldest "
                + "message\(dropped == 1 ? "" : "s") to fit the context.", false)
        }
        return nil
    }

    private func contextNoticeText(_ notice: (text: String, isError: Bool)) -> some View {
        Label(notice.text, systemImage: notice.isError
              ? "exclamationmark.triangle.fill" : "info.circle")
            .appFont(.caption)
            .foregroundStyle(notice.isError ? AnyShapeStyle(TUFFMacTheme.accentTextColor)
                             : AnyShapeStyle(.secondary))
            .fixedSize(horizontal: false, vertical: true)
    }

    private var footer: some View {
        HStack(spacing: 8) {
            attachMenu
            // Model and reasoning belong with the prompt they apply to, rather
            // than in a separate strip above the composer.
            ChatControlsView(model: model)
            ToolToggleControls(model: model)
            Spacer(minLength: 8)
            contextMeter
                .transition(.opacity)
            clearAction
                .transition(.tuffArrival(scale: 0.8))
            GenerateControl(model: model)
        }
        .animation(TUFFMotion.standard, value: model.contextUsage.isTight)
        .animation(TUFFMotion.standard, value: clearActionKind)
    }

    /// Which of the two buttons before Send is showing, if either.
    private var clearActionKind: Int {
        if model.isSelectedConversationRunning { return 0 }
        if model.hasComposedInput { return 1 }
        return model.hasOutputTranscript ? 2 : 0
    }

    /// One control for everything that can be added to a message.
    ///
    /// Two icon buttons sat side by side doing the same kind of thing — and a
    /// third appeared beside them whenever image support needed downloading, so
    /// the row's width depended on the model. A single round + reads as "add
    /// something", and the menu says what.
    private var attachMenu: some View {
        Menu {
            Button {
                chooseAttachments(.image)
            } label: {
                Label("Image…", systemImage: "photo")
            }
            .disabled(!canAddImages)

            Button {
                chooseAttachments(.document)
            } label: {
                Label("File…", systemImage: "doc.text")
            }
            .disabled(!canAddDocuments)

            // Where to get image support, rather than a button offering it.
            // The composer used to carry the whole download-activate-cancel
            // control set, which meant an icon appearing and disappearing in
            // the message box depending on which model was selected and how far
            // its optional pack had got. That belongs on Models, with the model
            // it is part of; this is the signpost.
            if !model.isImageInputAvailable,
               model.selectedDescriptor.supportsImageInput,
               model.isVisionRuntimeSupported {
                Divider()
                Text("\(model.selectedDescriptor.shortName) reads images once "
                    + "its image support is downloaded, in Settings > Models.")
            }
        } label: {
            ComposerIcon(systemImage: "plus", isOn: false)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(model.isSelectedConversationRunning || !(canAddImages || canAddDocuments))
        .help("Attach an image or a file")
        .accessibilityLabel("Add an attachment")
    }

    /// How full the window is, shown once it is worth knowing rather than all
    /// the time. Every figure is an estimate — the real tokenizer lives in the
    /// decode service — so the label says so on hover rather than implying a
    /// precision it does not have.
    @ViewBuilder
    private var contextMeter: some View {
        let usage = model.contextUsage
        if usage.isTight {
            HStack(spacing: 5) {
                Circle()
                    .fill(usage.willDropOldestTurns
                          ? AnyShapeStyle(TUFFMacTheme.accentTextColor)
                          : AnyShapeStyle(TUFFMacTheme.accentTextColor))
                    .frame(width: 6, height: 6)
                Text("\(MetricFormat.tokens(usage.estimatedTokens)) / "
                     + MetricFormat.tokens(usage.maxTokens))
                    .appFont(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .help("About \(usage.estimatedTokens) of \(usage.maxTokens) context "
                  + "tokens. An estimate: the exact count comes from the model's "
                  + "own tokenizer when the message is sent.")
            .accessibilityLabel("Context used")
            .accessibilityValue("about \(usage.estimatedTokens) of \(usage.maxTokens) tokens")
        }
    }

    private var canAddImages: Bool {
        model.isImageInputAvailable && !model.isSelectedConversationRunning && !model.isAddingImages
            && model.imageAttachments.count < model.maximumImageAttachments
    }

    private var canAddDocuments: Bool {
        !model.isSelectedConversationRunning
            && model.documentAttachments.count < AppModel.maximumDocumentAttachments
    }

    /// The same tiles the transcript draws, plus a way to take them off again.
    private var attachmentStrip: some View {
        MessageAttachmentsView(
            images: model.imageAttachments,
            documents: model.documentAttachments,
            imageTileSide: 54,
            onRemoveImage: { model.removeImage(id: $0) },
            onRemoveDocument: { model.removeDocument(id: $0) })
            .padding(.bottom, 2)
    }

    /// Whatever the composer currently has to say about the attachments: a
    /// refusal first, since it is the one the user has to act on.
    private var attachmentNotice: (text: String, isError: Bool)? {
        if let error = model.imageAttachmentError { return (error, true) }
        if let notice = model.documentAttachmentNotice, hasAttachments {
            return (notice, false)
        }
        return nil
    }

    private func attachmentNoticeText(_ notice: (text: String, isError: Bool)) -> some View {
        Text(notice.text)
            .appFont(.caption)
            .foregroundStyle(notice.isError ? AnyShapeStyle(TUFFMacTheme.accentTextColor)
                             : AnyShapeStyle(.secondary))
            .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder
    private var clearAction: some View {
        if !model.isSelectedConversationRunning && model.hasComposedInput {
            Button {
                model.promptText = ""
                model.clearImages()
                model.clearDocuments()
                promptFocused = true
            } label: {
                Label("Clear input", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 32, height: 32)
                    .contentShape(Circle())
                    .tuffHoverGlass()
            }
            .buttonStyle(.borderless)
            .help("Clear the message and its attachments")
        } else if !model.isSelectedConversationRunning && model.hasOutputTranscript {
            Button {
                model.clearOutput()
            } label: {
                Label("New chat", systemImage: "square.and.pencil")
                    .labelStyle(.iconOnly)
                    .frame(width: 32, height: 32)
                    .contentShape(Circle())
                    .tuffHoverGlass()
            }
            .buttonStyle(.borderless)
            .help("New chat")
        }
    }
}

/// Sizing and type for the message box. Kept in one place because the
/// placeholder overlay has to match the text view exactly, and it used to be
/// two independent font choices that happened to agree.
enum PromptComposerMetrics {
    static let cornerRadius: CGFloat = 22
    static let padding: CGFloat = 12
    /// A point larger than the system default. At 13 the message you are
    /// writing was set smaller than the answers around it.
    static let pointSize: CGFloat = 14
    static var font: AppFont { .system(size: pointSize) }
    /// The text view is AppKit, so it is handed a concrete size rather than
    /// reading the zoom out of the environment the way `font` does.
    static func nsFont(scale: CGFloat) -> NSFont {
        .systemFont(ofSize: pointSize * scale)
    }
    /// One line, and about eight. Both scale with the zoom, because they are
    /// heights for text whose size is changing. Past the maximum the box
    /// scrolls rather than eating the conversation.
    static let baseMinimumEditorHeight: CGFloat = 21
    static let baseMaximumEditorHeight: CGFloat = 168
    static func minimumEditorHeight(scale: CGFloat) -> CGFloat {
        baseMinimumEditorHeight * scale
    }
    static func maximumEditorHeight(scale: CGFloat) -> CGFloat {
        baseMaximumEditorHeight * scale
    }
}

private struct PromptTextEditor: NSViewRepresentable {
    @Binding var text: String
    let isFocused: Binding<Bool>
    /// The height the typed text needs. The view is given a clamped version of
    /// it, so the box grows with the message and stops.
    @Binding var contentHeight: CGFloat
    /// The zoom, passed in because an NSViewRepresentable's AppKit views do
    /// not read the environment themselves.
    let fontScale: CGFloat
    let newlineShortcut: AppNewlineShortcut
    let canRun: Bool
    let canAcceptImages: Bool
    let canAcceptDocuments: Bool
    let onSubmit: () -> Void
    let onImagesDropped: ([URL]) -> Void
    let onDocumentsDropped: ([URL]) -> Void
    let onImageDataPasted: (Data, String) -> Void
    let onPromisedImagesReceived: ([URL], URL) -> Void
    let onPromisedImagesFailed: () -> Void
    let onUnsupportedImagePaste: () -> Void
    let onDropTargeted: (Bool) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let textView = AttachmentDropTextView()
        textView.delegate = context.coordinator
        textView.string = text
        textView.font = PromptComposerMetrics.nsFont(scale: fontScale)
        textView.textColor = .labelColor
        textView.drawsBackground = false
        textView.isRichText = false
        textView.allowsUndo = true
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainerInset = .zero
        textView.textContainer?.lineFragmentPadding = 5
        textView.textContainer?.widthTracksTextView = true
        textView.setAccessibilityLabel("Prompt")
        // A promise-only drag — Photos, Mail, most browsers — never reaches
        // `draggingEntered` unless its types are registered here.
        textView.registerForDraggedTypes(
            textView.registeredDraggedTypes
                + NSFilePromiseReceiver.readableDraggedTypes.map {
                    NSPasteboard.PasteboardType(rawValue: $0)
                })

        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.documentView = textView
        context.coordinator.textView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? AttachmentDropTextView else { return }
        context.coordinator.parent = self
        // The view is not rebuilt when the zoom changes, so the font is
        // re-applied here or the message box keeps the size it was made with.
        let font = PromptComposerMetrics.nsFont(scale: fontScale)
        if textView.font != font { textView.font = font }
        textView.onContentHeightChange = { measured in contentHeight = measured }
        textView.canAcceptImages = canAcceptImages
        textView.canAcceptDocuments = canAcceptDocuments
        textView.onImagesDropped = onImagesDropped
        textView.onDocumentsDropped = onDocumentsDropped
        textView.onImageDataPasted = onImageDataPasted
        textView.onPromisedImagesReceived = onPromisedImagesReceived
        textView.onPromisedImagesFailed = onPromisedImagesFailed
        textView.onUnsupportedImagePaste = onUnsupportedImagePaste
        textView.onDropTargeted = onDropTargeted
        if textView.string != text {
            textView.string = text
        }
        textView.reportContentHeight()
        if isFocused.wrappedValue,
           textView.window?.firstResponder !== textView {
            DispatchQueue.main.async {
                textView.window?.makeFirstResponder(textView)
            }
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: PromptTextEditor
        weak var textView: NSTextView?

        init(_ parent: PromptTextEditor) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
            (textView as? AttachmentDropTextView)?.reportContentHeight()
        }

        func textDidBeginEditing(_ notification: Notification) {
            parent.isFocused.wrappedValue = true
        }

        func textDidEndEditing(_ notification: Notification) {
            parent.isFocused.wrappedValue = false
        }

        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            guard commandSelector == #selector(NSResponder.insertNewline(_:)) else {
                return false
            }
            let event = NSApp.currentEvent
            switch PromptSubmissionPolicy.decision(
                newlineShortcut: parent.newlineShortcut,
                modifiers: Self.modifiers(event?.modifierFlags ?? []),
                canRun: parent.canRun,
                hasMarkedText: textView.hasMarkedText(),
                isRepeat: event?.isARepeat == true) {
            case .submit:
                parent.onSubmit()
                return true
            case .consume:
                return true
            case .deferToEditor:
                return false
            }
        }

        private static func modifiers(_ flags: NSEvent.ModifierFlags) -> EventModifiers {
            var modifiers: EventModifiers = []
            if flags.contains(.command) { modifiers.insert(.command) }
            if flags.contains(.shift) { modifiers.insert(.shift) }
            if flags.contains(.option) { modifiers.insert(.option) }
            if flags.contains(.control) { modifiers.insert(.control) }
            return modifiers
        }
    }
}

/// The prompt editor's drop and paste target.
///
/// It handles two kinds of attachment, not one. It used to accept images only,
/// and refuse the drop outright whenever image input was unavailable — so
/// dragging a PDF onto the composer did nothing at all, and on a text-only
/// model so did dragging anything. Each dropped or pasted URL is now sorted:
/// images the vision pack can decode go one way, files the extractor can read
/// go the other, and the two capabilities are checked independently.
private final class AttachmentDropTextView: NSTextView {
    var canAcceptImages = false
    var canAcceptDocuments = false
    var onContentHeightChange: ((CGFloat) -> Void)?
    private var reportedHeight: CGFloat = -1
    private var measuredWidth: CGFloat = -1

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        guard abs(newSize.width - measuredWidth) > 0.5 else { return }
        measuredWidth = newSize.width
        reportContentHeight()
    }

    /// Measures the text and, when it changed, reports it. Deferred, because
    /// this runs from `updateNSView` and from AppKit's layout, and writing
    /// SwiftUI state directly from either is a "modifying state during view
    /// update" violation.
    func reportContentHeight() {
        guard let container = textContainer, let layoutManager else { return }
        layoutManager.ensureLayout(for: container)
        let measured = ceil(layoutManager.usedRect(for: container).height)
        guard measured > 0, abs(measured - reportedHeight) > 0.5 else { return }
        reportedHeight = measured
        let handler = onContentHeightChange
        DispatchQueue.main.async { handler?(measured) }
    }

    var onImagesDropped: (([URL]) -> Void)?
    var onDocumentsDropped: (([URL]) -> Void)?
    /// Promised files arrive in a directory we made and must not keep.
    var onPromisedImagesReceived: (([URL], URL) -> Void)?
    var onImageDataPasted: ((Data, String) -> Void)?
    var onUnsupportedImagePaste: (() -> Void)?
    var onPromisedImagesFailed: (() -> Void)?
    var onDropTargeted: ((Bool) -> Void)?

    override func paste(_ sender: Any?) {
        guard canAcceptImages || canAcceptDocuments else {
            super.paste(sender)
            return
        }
        let pasteboard = NSPasteboard.general
        let sorted = attachments(from: pasteboard)
        if sorted.deliver(images: onImagesDropped, documents: onDocumentsDropped,
                          canAcceptImages: canAcceptImages,
                          canAcceptDocuments: canAcceptDocuments) {
            return
        }
        // Text on the pasteboard wins over a picture that came with it.
        // Copying a range of cells from Numbers or Excel, or a text box from
        // Keynote, writes TIFF alongside the RTF and plain text; so does a
        // rich selection from many other apps. Taking the picture and
        // returning swallowed the text the user actually selected, in a text
        // editor, with nothing said about it. A screenshot or an image copied
        // from an image editor carries no text, so it still attaches.
        //
        // Only the byte-carrying branches need this. A file paste is handled
        // above, and Finder puts the path on the pasteboard as a string too.
        if pasteboard.string(forType: .string)?.isEmpty == false {
            super.paste(sender)
            return
        }
        // An image copied from another app arrives as bytes with no file
        // behind it, which used to be refused with an instruction to go and
        // find one. Bytes can only be an image, so this branch stays gated on
        // image support alone.
        if canAcceptImages {
            if let image = imageData(from: pasteboard) {
                onImageDataPasted?(image.data, image.name)
                return
            }
            if containsImage(in: pasteboard) {
                onUnsupportedImagePaste?()
                return
            }
        }
        super.paste(sender)
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        guard carriesAttachments(sender.draggingPasteboard) else {
            return super.draggingEntered(sender)
        }
        guard acceptsDrag(sender.draggingPasteboard) else { return [] }
        onDropTargeted?(true)
        return .copy
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        guard carriesAttachments(sender.draggingPasteboard) else {
            return super.draggingUpdated(sender)
        }
        return acceptsDrag(sender.draggingPasteboard) ? .copy : []
    }

    override func draggingExited(_ sender: NSDraggingInfo?) {
        onDropTargeted?(false)
        super.draggingExited(sender)
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let sorted = attachments(from: sender.draggingPasteboard)
        if !sorted.isEmpty {
            onDropTargeted?(false)
            return sorted.deliver(
                images: onImagesDropped, documents: onDocumentsDropped,
                canAcceptImages: canAcceptImages,
                canAcceptDocuments: canAcceptDocuments)
        }
        // A promise carries no URL to sort on, so it stays the image path it
        // was built for: Photos, Mail and browsers promise pictures.
        let promises = filePromises(from: sender.draggingPasteboard)
        guard !promises.isEmpty else { return super.performDragOperation(sender) }
        onDropTargeted?(false)
        guard canAcceptImages else { return false }
        receive(promises)
        return true
    }

    private func acceptsDrag(_ pasteboard: NSPasteboard) -> Bool {
        let sorted = attachments(from: pasteboard)
        if !sorted.images.isEmpty, canAcceptImages { return true }
        if !sorted.documents.isEmpty, canAcceptDocuments { return true }
        return sorted.isEmpty && canAcceptImages
            && !filePromises(from: pasteboard).isEmpty
    }

    /// Drags from Photos, Mail and most browsers carry a promise rather than a
    /// file: the source writes the bytes only once a destination asks for them.
    /// Without this the drop was simply refused.
    private func receive(_ promises: [NSFilePromiseReceiver]) {
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("TUFF-Promises", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: destination, withIntermediateDirectories: true)
        } catch {
            // Failing to make a staging directory is a disk or permission
            // failure, not a statement about what was dropped. Reporting it as
            // unsupported content sent someone off converting file formats
            // because the volume was full.
            onPromisedImagesFailed?()
            return
        }
        let queue = OperationQueue()
        // One completion arrives per promised FILE, not per receiver: a legacy
        // promiser puts several file names on a single pasteboard item. Counting
        // receivers finished the batch at the first file and handed the
        // directory straight to `addImages`, whose `defer` deleted it while
        // AppKit was still writing the rest — so a three-image drag attached one
        // and destroyed two, with nothing reporting the loss.
        let expected = promises.reduce(0) { $0 + max(1, $1.fileNames.count) }
        let received = ReceivedPromises(expected: expected)
        for promise in promises {
            promise.receivePromisedFiles(
                atDestination: destination,
                options: [:],
                operationQueue: queue
            ) { [weak self] url, error in
                // The staged copy is what the request uses, so the promised
                // file is only needed until then.
                let finished = received.record(url: error == nil ? url : nil)
                guard let finished else { return }
                DispatchQueue.main.async {
                    if finished.isEmpty {
                        // A promise that failed is not "unsupported content";
                        // saying so sends the user looking for the wrong thing.
                        self?.onPromisedImagesFailed?()
                        try? FileManager.default.removeItem(at: destination)
                    } else if let handler = self?.onPromisedImagesReceived {
                        handler(finished, destination)
                    } else {
                        // The text view went away before the promises resolved,
                        // so nothing downstream will ever own this directory:
                        // the optional chain was a no-op and the sweep covers
                        // only the staging root, so a closed window left the
                        // full-size copies behind for every future launch.
                        try? FileManager.default.removeItem(at: destination)
                    }
                }
            }
        }
    }

    /// Only files in a format the vision pack can actually decode. Without the
    /// conformance filter any file URL was staged: copying a document in Finder
    /// and pressing Cmd-V attached it, replacing the editor's own paste, and the
    /// failure surfaced only once the run reached image decoding.
    ///
    /// The list is the runtime's, not `public.image`: conforming to
    /// `public.image` admits camera RAW, EXR and WebP, which are copied at full
    /// size and decoded for a thumbnail before the run refuses them. The picker
    /// reads the same list, and paste, drop and the drag-accept highlight all
    /// read through here, so the four stay in agreement.
    private func fileURLs(from pasteboard: NSPasteboard) -> [URL] {
        let objects = pasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [
                .urlReadingFileURLsOnly: true,
                .urlReadingContentsConformToTypes:
                    Array(VisionImageLimits().allowedTypeIdentifiers),
            ]) as? [NSURL]
        return objects?.map { $0 as URL } ?? []
    }

    /// Every file URL on the pasteboard, unfiltered, so documents can be sorted
    /// out of it below. `fileURLs` above filters to the vision pack's formats
    /// and is what decides which of them is an image.
    private func allFileURLs(from pasteboard: NSPasteboard) -> [URL] {
        let objects = pasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [.urlReadingFileURLsOnly: true]) as? [NSURL]
        return objects?.map { $0 as URL } ?? []
    }

    /// One drop or paste, split by what each file is.
    ///
    /// A mixed selection — a screenshot and a PDF together — is a normal thing
    /// to drag, and taking only the first kind would silently drop the rest.
    struct SortedAttachments {
        var images: [URL] = []
        var documents: [URL] = []

        var isEmpty: Bool { images.isEmpty && documents.isEmpty }

        /// Hands each set to its callback, and reports whether anything was
        /// taken. False means the drop was not ours and AppKit should have it.
        func deliver(
            images imageHandler: (([URL]) -> Void)?,
            documents documentHandler: (([URL]) -> Void)?,
            canAcceptImages: Bool,
            canAcceptDocuments: Bool
        ) -> Bool {
            var handled = false
            if !images.isEmpty, canAcceptImages {
                imageHandler?(images)
                handled = true
            }
            if !documents.isEmpty, canAcceptDocuments {
                documentHandler?(documents)
                handled = true
            }
            return handled
        }
    }

    private func attachments(from pasteboard: NSPasteboard) -> SortedAttachments {
        let images = Set(fileURLs(from: pasteboard).map(\.standardizedFileURL))
        var sorted = SortedAttachments()
        for url in allFileURLs(from: pasteboard) {
            if images.contains(url.standardizedFileURL) {
                sorted.images.append(url)
            } else if DocumentTextExtractor.canExtract(from: url) {
                sorted.documents.append(url)
            }
        }
        return sorted
    }

    private func filePromises(from pasteboard: NSPasteboard) -> [NSFilePromiseReceiver] {
        pasteboard.readObjects(
            forClasses: [NSFilePromiseReceiver.self],
            options: nil) as? [NSFilePromiseReceiver] ?? []
    }

    private func carriesAttachments(_ pasteboard: NSPasteboard) -> Bool {
        !attachments(from: pasteboard).isEmpty
            || !filePromises(from: pasteboard).isEmpty
    }

    /// The first pasteboard representation the vision pack can actually decode.
    /// PNG and JPEG come through unchanged; anything else is re-encoded as PNG
    /// rather than handing the decoder a format it was never validated against.
    private func imageData(from pasteboard: NSPasteboard) -> (data: Data, name: String)? {
        if let data = pasteboard.data(forType: .png) {
            return (data, "Pasted image.png")
        }
        if let data = pasteboard.data(forType: NSPasteboard.PasteboardType("public.jpeg")) {
            return (data, "Pasted image.jpeg")
        }
        guard let tiff = pasteboard.data(forType: .tiff),
              let representation = NSBitmapImageRep(data: tiff),
              let png = representation.representation(using: .png, properties: [:])
        else { return nil }
        return (png, "Pasted image.png")
    }

    private func containsImage(in pasteboard: NSPasteboard) -> Bool {
        (pasteboard.types ?? []).contains { type in
            UTType(type.rawValue)?.conforms(to: .image) == true
        }
    }
}

/// Collects the results of a multi-file promise drop, which arrive one
/// completion at a time on an arbitrary queue.
private final class ReceivedPromises: @unchecked Sendable {
    private let lock = NSLock()
    private let expected: Int
    private var urls: [URL] = []
    private var completed = 0

    init(expected: Int) {
        self.expected = expected
    }

    /// Returns the collected URLs once every promise has reported, and nil
    /// until then.
    func record(url: URL?) -> [URL]? {
        lock.lock()
        defer { lock.unlock() }
        if let url { urls.append(url) }
        completed += 1
        return completed == expected ? urls : nil
    }
}
