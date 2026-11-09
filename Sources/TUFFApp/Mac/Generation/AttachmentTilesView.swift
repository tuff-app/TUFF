import AppKit
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// The attachments on one message: images as thumbnails, files as tiles.
///
/// The same row is used by the composer and by the transcript, so an attachment
/// looks the same before and after it is sent. `onRemove` is what separates the
/// two: the composer passes one, the transcript does not.
struct MessageAttachmentsView: View {
    var images: [AppImageAttachment] = []
    var documents: [AppDocumentAttachment] = []
    var imageTileSide: CGFloat = 76
    var onRemoveImage: ((UUID) -> Void)?
    var onRemoveDocument: ((UUID) -> Void)?

    var body: some View {
        if !images.isEmpty || !documents.isEmpty {
            AttachmentFlowLayout(spacing: 8, rowSpacing: 8) {
                ForEach(images, id: \.id) { attachment in
                    RemovableAttachment(
                        label: "Remove \(attachment.displayName)",
                        onRemove: onRemoveImage.map { remove in
                            { remove(attachment.id) }
                        }
                    ) {
                        SubmittedImageThumbnail(
                            attachment: attachment,
                            maximumSize: CGSize(
                                width: imageTileSide, height: imageTileSide))
                    }
                    .transition(.tuffArrival(scale: 0.85))
                }
                ForEach(documents) { document in
                    RemovableAttachment(
                        label: "Remove \(document.displayName)",
                        onRemove: onRemoveDocument.map { remove in
                            { remove(document.id) }
                        }
                    ) {
                        DocumentAttachmentTile(document: document)
                    }
                    .transition(.tuffArrival(scale: 0.85))
                }
            }
            // A tile added or taken off: the others slide to their places.
            .animation(TUFFMotion.standard, value: images.map(\.id) + documents.map(\.id))
        }
    }
}

/// A file, shown as a file: name, kind, and roughly how much of the context it
/// will use. The text still reaches the model — it just no longer reaches the
/// prompt box.
struct DocumentAttachmentTile: View {
    let document: AppDocumentAttachment
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        HStack(spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: TUFFDesign.Radius.concentric(
                    outer: TUFFDesign.Radius.panel, padding: 8), style: .continuous)
                    .fill(TUFFMacTheme.accentColor.opacity(0.16))
                Image(systemName: symbolName)
                    .appFont(.system(size: 13, weight: .medium))
                    .foregroundStyle(TUFFMacTheme.accentTextColor)
            }
            .frame(width: 30, height: 30)

            VStack(alignment: .leading, spacing: 1) {
                Text(document.displayName)
                    .appFont(.callout.weight(.medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("\(document.kindLabel) · \(document.estimatedTokensLabel) tokens")
                    .appFont(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(8)
        .frame(width: 232, alignment: .leading)
        .tuffPanel()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Attached file \(document.displayName)")
        .accessibilityValue(
            "\(document.kindLabel), about \(document.estimatedTokens) tokens")
        .help(document.displayName)
    }

    private var symbolName: String {
        switch document.fileExtension {
        case "pdf": "doc.richtext"
        case "csv", "tsv": "tablecells"
        case "json", "yaml", "yml": "curlybraces"
        case "md", "markdown": "text.alignleft"
        default: "doc.text"
        }
    }
}

/// Wraps an attachment tile and, in the composer, its remove badge.
private struct RemovableAttachment<Content: View>: View {
    let label: String
    let onRemove: (() -> Void)?
    @ViewBuilder let content: Content
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        content
            .overlay(alignment: .topTrailing) {
                if let onRemove {
                    Button(action: onRemove) {
                        Image(systemName: "xmark.circle.fill")
                            .symbolRenderingMode(.hierarchical)
                            .appFont(.body)
                    }
                    .buttonStyle(.plain)
                    .background(
                        TUFFMacTheme.surfaceStyle(
                            reduceTransparency: reduceTransparency),
                        in: Circle())
                    .offset(x: 6, y: -6)
                    .accessibilityLabel(label)
                }
            }
            .padding(.top, onRemove == nil ? 0 : 6)
            .padding(.trailing, onRemove == nil ? 0 : 6)
    }
}

/// Lays attachments out left to right and wraps.
///
/// A horizontal `ScrollView` was what this replaced: the transcript is one
/// scrollable surface now, and a row that scrolls sideways inside it swallows
/// trackpad gestures meant for the conversation.
struct AttachmentFlowLayout: Layout {
    var spacing: CGFloat = 8
    var rowSpacing: CGFloat = 8

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        let width = proposal.width ?? .infinity
        let rows = rows(for: subviews, width: width)
        let height = rows.reduce(CGFloat.zero) { $0 + $1.height }
            + rowSpacing * CGFloat(max(0, rows.count - 1))
        let widest = rows.map(\.width).max() ?? 0
        return CGSize(
            width: proposal.width ?? widest,
            height: height)
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        var y = bounds.minY
        for row in rows(for: subviews, width: bounds.width) {
            var x = bounds.minX
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(
                    at: CGPoint(x: x, y: y),
                    anchor: .topLeading,
                    proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + rowSpacing
        }
    }

    private struct Row {
        var indices: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    private func rows(for subviews: Subviews, width: CGFloat) -> [Row] {
        var rows: [Row] = []
        var current = Row()
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let advance = current.indices.isEmpty ? size.width : current.width + spacing + size.width
            if !current.indices.isEmpty, advance > width {
                rows.append(current)
                current = Row()
            }
            current.width = current.indices.isEmpty
                ? size.width
                : current.width + spacing + size.width
            current.height = max(current.height, size.height)
            current.indices.append(index)
        }
        if !current.indices.isEmpty { rows.append(current) }
        return rows
    }
}
