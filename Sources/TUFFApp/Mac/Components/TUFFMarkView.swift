import SwiftUI
import TUFFAppCore
import TUFFMacPresentation

/// The TUFF mark: a bird built from one equilateral triangle and four quarter
/// circles of equal radius. Two quarters form the belly, one forms the
/// shoulder, one forms the raised wing, and the triangle is the beak.
struct TUFFMarkView: View {
    var body: some View {
        GeometryReader { proxy in
            let size = min(proxy.size.width, proxy.size.height)
            ZStack {
                RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
                    .fill(.black)
                TUFFWingAndBeak().fill(TUFFMacTheme.markWingColor)
                TUFFBirdBody().fill(TUFFMacTheme.accentColor)
            }
            .frame(width: size, height: size)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .aspectRatio(1, contentMode: .fit)
        .accessibilityHidden(true)
    }
}

/// The raised wing quarter and the beak triangle.
private struct TUFFWingAndBeak: Shape {
    func path(in rect: CGRect) -> Path {
        var path = TUFFWing().path(in: rect)
        path.addPath(TUFFBeak().path(in: rect))
        return path
    }
}

/// The raised wing quarter alone, so the loading mark can move it.
struct TUFFWing: Shape {
    /// Where the wing meets the shoulder: the point it turns around.
    static let hinge = UnitPoint(x: 420.19 / 1024, y: 435.24 / 1024)

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: point(113.14, 128.19, in: rect))
        path.addCurve(
            to: point(420.19, 435.24, in: rect),
            control1: point(282.72, 128.19, in: rect),
            control2: point(420.19, 265.66, in: rect))
        path.addLine(to: point(113.14, 435.24, in: rect))
        path.closeSubpath()
        return path
    }

    private func point(_ x: CGFloat, _ y: CGFloat, in rect: CGRect) -> CGPoint {
        CGPoint(
            x: rect.minX + (x / 1024) * rect.width,
            y: rect.minY + (y / 1024) * rect.height)
    }
}

/// The beak triangle alone.
struct TUFFBeak: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: point(683.09, 281.72, in: rect))
        path.addLine(to: point(952, 435.24, in: rect))
        path.addLine(to: point(683.09, 588.76, in: rect))
        path.closeSubpath()
        return path
    }

    private func point(_ x: CGFloat, _ y: CGFloat, in rect: CGRect) -> CGPoint {
        CGPoint(
            x: rect.minX + (x / 1024) * rect.width,
            y: rect.minY + (y / 1024) * rect.height)
    }
}

/// The belly and shoulder quarter as one outline.
struct TUFFBirdBody: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: point(72, 588.76, in: rect))
        path.addCurve(
            to: point(379.05, 895.81, in: rect),
            control1: point(72, 758.34, in: rect),
            control2: point(209.47, 895.81, in: rect))
        path.addCurve(
            to: point(686.1, 588.76, in: rect),
            control1: point(548.62, 895.81, in: rect),
            control2: point(686.1, 758.34, in: rect))
        path.addLine(to: point(686.09, 281.72, in: rect))
        path.addCurve(
            to: point(379.04, 588.76, in: rect),
            control1: point(516.51, 281.71, in: rect),
            control2: point(379.04, 419.18, in: rect))
        path.addLine(to: point(72, 588.76, in: rect))
        path.closeSubpath()
        return path
    }

    private func point(_ x: CGFloat, _ y: CGFloat, in rect: CGRect) -> CGPoint {
        CGPoint(
            x: rect.minX + (x / 1024) * rect.width,
            y: rect.minY + (y / 1024) * rect.height)
    }
}

/// The mark at the top of an empty chat, which also shows a model loading.
///
/// While loading, the wing beats slowly and a ring around the mark fills as
/// the load moves through its phases. Between phases the ring keeps easing
/// toward the next one, so a long phase such as verifying weights never looks
/// stuck. When the model is ready the ring closes and fades outward and the
/// mark gives one spring. With Reduce Motion the wing holds still, the ring
/// only fills, and the finish is a fade.
struct TUFFLoadingMark: View {
    enum Stage: Equatable {
        /// No model loaded: the mark rests, dimmed.
        case idle
        /// Loading, with how far the current phase puts it, 0...1.
        case loading(Double)
        case ready
    }

    let stage: Stage
    var size: CGFloat = 56

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phaseStart = Date()
    @State private var finishing = false
    @State private var pops = 0

    var body: some View {
        TimelineView(.animation(paused: !isLoading || reduceMotion)) { context in
            let elapsed = context.date.timeIntervalSince(phaseStart)
            ZStack {
                ring(elapsed: elapsed, time: context.date.timeIntervalSinceReferenceDate)
                mark(time: context.date.timeIntervalSinceReferenceDate)
                    .frame(width: size * 0.62, height: size * 0.62)
            }
        }
        .frame(width: size, height: size)
        .keyframeAnimator(initialValue: 1.0, trigger: pops) { content, scale in
            content.scaleEffect(scale)
        } keyframes: { _ in
            KeyframeTrack {
                SpringKeyframe(1.12, duration: 0.2, spring: .snappy)
                SpringKeyframe(1.0, duration: 0.6, spring: .bouncy)
            }
        }
        .onChange(of: stage) { old, new in
            switch new {
            case .loading:
                if case .loading = old {} else {
                    finishing = false
                }
                phaseStart = .now
            case .ready:
                if case .loading = old {
                    withAnimation(.smooth(duration: 0.55)) { finishing = true }
                    if !reduceMotion { pops += 1 }
                }
            case .idle:
                finishing = false
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityText)
    }

    private var isLoading: Bool {
        if case .loading = stage { return true }
        return false
    }

    private var target: Double {
        if case .loading(let fraction) = stage { return fraction }
        return stage == .ready ? 1 : 0
    }

    /// The ring's fill: the phase's own fraction, then easing a share of the
    /// way to the next one the longer the phase runs.
    private func displayedProgress(elapsed: TimeInterval) -> Double {
        guard isLoading else { return finishing ? 1 : 0 }
        let next = min(0.96, target + 0.16)
        let creep = reduceMotion ? 0 : 1 - exp(-max(0, elapsed) / 6)
        return target + (next - target) * creep
    }

    @ViewBuilder
    private func ring(elapsed: TimeInterval, time: TimeInterval) -> some View {
        let lineWidth = max(2, size * 0.05)
        let accent = TUFFMacTheme.accentColor
        ZStack {
            Circle()
                .stroke(Color.primary.opacity(0.08), lineWidth: lineWidth)
            Circle()
                .trim(from: 0, to: displayedProgress(elapsed: elapsed))
                .stroke(
                    AngularGradient(
                        colors: [accent.opacity(0.35), accent, TUFFMacTheme.markWingColor, accent],
                        center: .center,
                        angle: .degrees(reduceMotion ? 0 : time.truncatingRemainder(dividingBy: 3) * 120)),
                    style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.smooth(duration: 0.6), value: target)
        }
        .opacity(isLoading || finishing ? 1 : 0)
        .scaleEffect(finishing && !isLoading && !reduceMotion ? 1.22 : 1)
        .opacity(finishing && !isLoading ? 0 : 1)
        .animation(.smooth(duration: 0.55), value: finishing)
        .animation(TUFFMotion.standard, value: isLoading)
    }

    private func mark(time: TimeInterval) -> some View {
        // One slow beat every 1.4 s: up quickly, down gently.
        let beat = (time / 1.4).truncatingRemainder(dividingBy: 1)
        let lift = isLoading && !reduceMotion ? (1 - cos(beat * 2 * .pi)) / 2 : 0
        return GeometryReader { proxy in
            let side = min(proxy.size.width, proxy.size.height)
            ZStack {
                RoundedRectangle(cornerRadius: side * 0.22, style: .continuous)
                    .fill(.black)
                TUFFWing()
                    .fill(TUFFMacTheme.markWingColor)
                    .rotationEffect(.degrees(-14 * lift), anchor: TUFFWing.hinge)
                TUFFBeak().fill(TUFFMacTheme.markWingColor)
                TUFFBirdBody().fill(TUFFMacTheme.accentColor)
            }
            .frame(width: side, height: side)
            .clipShape(RoundedRectangle(cornerRadius: side * 0.22, style: .continuous))
            // The tile is black, which vanishes on the dark page without an edge.
            .overlay(RoundedRectangle(cornerRadius: side * 0.22, style: .continuous)
                .strokeBorder(TUFFDesign.hairlineColor, lineWidth: 1))
        }
        .aspectRatio(1, contentMode: .fit)
        .saturation(stage == .idle ? 0 : 1)
        .opacity(stage == .idle ? 0.6 : 1)
        .animation(.smooth(duration: 0.4), value: stage == .idle)
    }

    private var accessibilityText: String {
        switch stage {
        case .idle: "No model loaded"
        case .loading(let fraction): "Loading model, \(Int(fraction * 100)) percent"
        case .ready: "Model ready"
        }
    }
}

extension AppModelLoadPhase {
    /// How far along a load is when it enters this phase. Verifying weights
    /// is usually the longest step, so it starts the ring near the middle.
    var progressFraction: Double {
        switch self {
        case .calibratingAuto: 0.04
        case .preparingEngine: 0.1
        case .validatingDirectory: 0.2
        case .tokenizer: 0.28
        case .verifyingWeights: 0.36
        case .preparingRunner: 0.72
        case .mappingImageTower: 0.86
        }
    }
}
