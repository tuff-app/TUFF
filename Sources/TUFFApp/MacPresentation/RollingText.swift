import SwiftUI

/// Text whose changed characters turn like the drum of a mechanical counter,
/// as in Apple's Timer: the old digit tips back over the top of the drum as
/// the new one rolls up from underneath. Nothing is clipped, and nothing is
/// blurred, which the system's numeric transition always does.
public struct RollingText: View {
    let text: String
    var animated: Bool
    /// When the text last changed, and whether the gap before that change
    /// was long enough to roll. A count that changes faster than once a
    /// second just updates: rolling every few frames read as a blur.
    @State private var lastChange = Date.distantPast
    @State private var isSlow = true

    public init(_ text: String, animated: Bool = true) {
        self.text = text
        self.animated = animated
    }

    public var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(text.enumerated()), id: \.offset) { item in
                let character = String(item.element)
                ZStack {
                    Text(character)
                        .id(character)
                        .transition(.asymmetric(
                            insertion: .modifier(active: DrumTurn(progress: 1),
                                                 identity: DrumTurn(progress: 0)),
                            removal: .modifier(active: DrumTurn(progress: -1),
                                               identity: DrumTurn(progress: 0))))
                }
                // The drum's face: digits turning past the top or bottom fade
                // into its edge instead of spilling over the line.
                .padding(.vertical, 2)
                .mask {
                    LinearGradient(stops: [.init(color: .clear, location: 0),
                                           .init(color: .black, location: 0.28),
                                           .init(color: .black, location: 0.72),
                                           .init(color: .clear, location: 1)],
                                   startPoint: .top, endPoint: .bottom)
                }
                .padding(.vertical, -2)
            }
        }
        .monospacedDigit()
        // Set here so it survives the transcript turning animations off
        // while a run is in progress.
        // Quick enough that a fast count finishes each turn before the next.
        .animation(animated && isSlow ? .spring(response: 0.3, dampingFraction: 0.88) : nil, value: text)
        .onChange(of: text) { _, _ in
            let now = Date()
            isSlow = now.timeIntervalSince(lastChange) >= 1
            lastChange = now
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(text)
    }
}

/// One digit's place on the drum: 0 facing the reader, 1 below the front
/// (about to roll up), -1 over the top (rolled away).
private struct DrumTurn: ViewModifier {
    let progress: CGFloat

    func body(content: Content) -> some View {
        content
            // Shading as the face turns away from the reader, not a blur.
            .opacity(1 - abs(progress))
            .rotation3DEffect(.degrees(Double(progress) * -70), axis: (x: 1, y: 0, z: 0),
                              anchor: .center, perspective: 0.5)
            .offset(y: progress * 7)
    }
}
