import SwiftUI

/// An `info.circle` that both hovers and clicks: the tooltip alone was
/// invisible, so the explanation it carried never reached the people who read
/// the figure next to it as a bug.
struct InfoPopoverButton: View {
    let subject: String
    let text: String
    var arrowEdge: Edge = .trailing
    @State private var isShowingPopover = false

    var body: some View {
        Button {
            isShowingPopover.toggle()
        } label: {
            Image(systemName: "info.circle")
                .appFont(.caption2)
                .foregroundStyle(.tertiary)
                .frame(width: 24, height: 24)
                .contentShape(Rectangle())
                .tuffHoverGlass()
        }
        .buttonStyle(.plain)
        .help(text)
        .popover(isPresented: $isShowingPopover, arrowEdge: arrowEdge) {
            Text(text)
                .appFont(.callout)
                .frame(width: 280, alignment: .leading)
                // Without this the popover kept one line's worth of height
                // and cut longer explanations off mid-sentence.
                .fixedSize(horizontal: false, vertical: true)
                .padding()
        }
        .accessibilityLabel("\(subject) information")
        .accessibilityHint(text)
    }
}
