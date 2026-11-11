import AppKit
import SwiftUI
import TUFFAppCore
import TUFFMacPresentation

/// Run TUFF's standard benchmark on installed models and share the result
/// in the Benchmarks discussions.
struct BenchmarksWorkspaceView: View {
    let model: AppModel
    @Bindable var controller: BenchmarkController

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                WorkspaceTitle(
                    title: "Benchmarks")
                if controller.installed.isEmpty {
                    emptyCard
                } else {
                    setupCard
                    if controller.isRunning || !controller.states.isEmpty {
                        progressCard
                            .transition(.opacity)
                    }
                    if let error = controller.error {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .appFont(.callout)
                            .foregroundStyle(TUFFDesign.secondaryTextColor)
                            .textSelection(.enabled)
                            .transition(.opacity)
                    }
                    if let result = controller.result {
                        resultCard(result)
                            .transition(.opacity)
                    }
                }
            }
            // A run starting, moving on to the next model and finishing:
            // the cards ease in and the rest moves down for them.
            .animation(TUFFMotion.standard, value: controller.isRunning)
            .animation(TUFFMotion.standard, value: controller.states.count)
            .animation(TUFFMotion.standard, value: controller.error)
            .animation(TUFFMotion.standard, value: controller.result != nil)
            .workspaceColumn()
        }
        .workspacePageChrome()
        .task { controller.refresh() }
    }

    // MARK: - Cards

    private var emptyCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("No models installed").appFont(.headline)
            Text("Install a model in the Models tab, then come back to benchmark it.")
                .appFont(.callout)
                .foregroundStyle(TUFFDesign.secondaryTextColor)
        }
        .tuffCard()
    }

    private var setupCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline) {
                Text("Models").appFont(.headline)
                Spacer()
                Button(allSelected ? "Select None" : "Select All") {
                    controller.selected = allSelected ? [] : Set(controller.installed.map(\.id))
                }
                .buttonStyle(.tuffSecondary)
                .controlSize(.small)
                .disabled(controller.isRunning)
            }
            VStack(spacing: 0) {
                ForEach(Array(controller.installed.enumerated()), id: \.element.id) { index, item in
                    if index > 0 { Divider() }
                    modelRow(item)
                }
            }
            Divider()
            HStack(alignment: .center, spacing: 14) {
                TUFFTabBar(tabs: AppBenchmarkMode.allCases, selection: $controller.mode,
                           title: \.title, compact: true)
                    .fixedSize()
                .disabled(controller.isRunning)
                .help("Quick runs each test once. Standard runs each three times, which the leaderboard uses.")
                Spacer(minLength: 8)
                if controller.isRunning {
                    Button("Stop", role: .cancel) { controller.stop() }
                        .controlSize(.large)
                } else {
                    Button {
                        Task {
                            await controller.start { await releaseChatModel() }
                        }
                    } label: {
                        Label("Run Benchmark", systemImage: "play.fill")
                    }
                    .help("Chat's model is unloaded first. Avoid chatting while it runs.")
                    .buttonStyle(.tuffProminent)
                    .controlSize(.large)
                    .disabled(!controller.canStart || model.isRunning)
                }
            }
        }
        .tuffCard()
    }

    private func modelRow(_ item: AppBenchmarkModel) -> some View {
        Toggle(isOn: Binding(
            get: { controller.selected.contains(item.id) },
            set: { isOn in
                if isOn { controller.selected.insert(item.id) } else { controller.selected.remove(item.id) }
            })) {
            HStack {
                Text(item.descriptor.displayName).appFont(.body)
                Spacer()
                if item.descriptor.source.installedBytes > 50_000_000_000 {
                    Text("Slow on most Macs")
                        .appFont(.caption)
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                }
                Text(Self.gigabytes(item.descriptor.source.installedBytes))
                    .appFont(.caption.monospacedDigit())
                    .foregroundStyle(TUFFDesign.secondaryTextColor)
                    .frame(minWidth: 64, alignment: .trailing)
            }
        }
        .toggleStyle(.checkbox)
        .disabled(controller.isRunning)
        .padding(.vertical, 7)
    }

    private var progressCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text(controller.isRunning ? "Running" : "Last run").appFont(.headline)
                Spacer()
                if controller.isRunning {
                    Text("\(Int(controller.progress * 100))%")
                        .appFont(.callout.monospacedDigit())
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                        .contentTransition(.numericText(value: controller.progress))
                        .animation(.smooth(duration: 0.4), value: controller.progress)
                }
            }
            if controller.isRunning {
                ProgressView(value: controller.progress)
                    .animation(.smooth(duration: 0.4), value: controller.progress)
            }
            ForEach(controller.installed.filter { controller.states[$0.id] != nil }) { item in
                HStack(spacing: 10) {
                    stateIcon(controller.states[item.id])
                        .frame(width: 18)
                    Text(item.descriptor.displayName).appFont(.body)
                    Spacer()
                    Text(stateText(controller.states[item.id]))
                        .appFont(.callout)
                        .foregroundStyle(TUFFDesign.secondaryTextColor)
                }
            }
        }
        .tuffCard()
    }

    private func resultCard(_ result: AppBenchmarkResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline) {
                Text("Results").appFont(.headline)
                Text(result.machine.shortDescription)
                    .appFont(.callout)
                    .foregroundStyle(TUFFDesign.secondaryTextColor)
                Spacer()
            }
            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 10) {
                GridRow {
                    header("Model")
                    header("Generation").help("Output tokens generated per second.").gridColumnAlignment(.trailing)
                    header("Prompt").help("Tokens processed per second for a 1,500-token prompt.").gridColumnAlignment(.trailing)
                    header("First token").help("Time until the first output token for that prompt.").gridColumnAlignment(.trailing)
                    header("Follow-up").help("Time until the first output token when reusing the earlier conversation.").gridColumnAlignment(.trailing)
                    header("Check").gridColumnAlignment(.center)
                }
                Divider().gridCellUnsizedAxes(.horizontal)
                ForEach(result.runs, id: \.model.id) { run in
                    GridRow {
                        Text(run.model.name)
                            .appFont(.body)
                            .frame(minWidth: 190, alignment: .leading)
                        if run.status == .completed {
                            value(AppBenchmarkShare.rate(run.summary?.decodeTokensPerSecond?.median))
                            value(AppBenchmarkShare.rate(run.summary?.prefillTokensPerSecond?.median))
                            value(run.summary?.longTimeToFirstTokenSeconds.map { AppBenchmarkShare.seconds($0.median) } ?? "n/a")
                            value(run.summary?.followUpTimeToFirstTokenSeconds.map(AppBenchmarkShare.seconds) ?? "n/a")
                            checkSymbol(run.check)
                        } else {
                            Text(run.error ?? run.status.rawValue.capitalized)
                                .appFont(.callout)
                                .foregroundStyle(TUFFDesign.secondaryTextColor)
                                .gridCellColumns(5)
                        }
                    }
                }
            }
            Divider()
            HStack(spacing: 10) {
                Button {
                    controller.share()
                } label: {
                    Label("Share on GitHub", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.tuffProminent)
                .disabled(result.completedRuns.isEmpty)
                Button("Show File") { controller.revealResult() }
                Button("Leaderboard") { NSWorkspace.shared.open(AppBenchmarkShare.leaderboardURL) }
                Spacer()
            }
            Text(controller.shareNote ?? "Sharing posts this table in TUFF's Benchmarks discussions from your GitHub account. It includes your chip, memory, macOS version and these timings, and nothing else.")
                .appFont(.caption)
                .foregroundStyle(TUFFDesign.secondaryTextColor)
        }
        .tuffCard()
    }

    // MARK: - Pieces

    private var allSelected: Bool {
        controller.selected.count == controller.installed.count
    }

    private func header(_ title: String) -> some View {
        Text(title)
            .appFont(.caption.weight(.semibold))
            .foregroundStyle(TUFFDesign.secondaryTextColor)
    }

    @ViewBuilder
    private func checkSymbol(_ check: AppBenchmarkResult.Check?) -> some View {
        switch check?.passed {
        case true?:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(TUFFMacTheme.accentTextColor)
                .help("Answered the check question correctly")
                .accessibilityLabel("Check passed")
        case false?:
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(TUFFMacTheme.accentTextColor)
                .help("Answered the check question wrongly: \(check?.answer ?? "")")
                .accessibilityLabel("Check failed")
        case nil:
            Text("n/a").foregroundStyle(TUFFDesign.secondaryTextColor)
        }
    }

    private func value(_ text: String) -> some View {
        Text(text).appFont(.body.monospacedDigit())
    }

    @ViewBuilder
    private func stateIcon(_ state: BenchmarkController.ModelState?) -> some View {
        switch state {
        case .running?:
            ProgressView().controlSize(.small)
        case .finished(.completed)?:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(TUFFMacTheme.accentTextColor)
        case .finished(.failed)?:
            Image(systemName: "xmark.circle.fill").foregroundStyle(TUFFMacTheme.accentTextColor)
        case .finished?:
            Image(systemName: "minus.circle").foregroundStyle(TUFFDesign.secondaryTextColor)
        default:
            Image(systemName: "circle").foregroundStyle(.tertiary)
        }
    }

    private func stateText(_ state: BenchmarkController.ModelState?) -> String {
        switch state {
        case .running(let step)?: step
        case .finished(let status)?: status.rawValue.capitalized
        default: "Waiting"
        }
    }

    private func releaseChatModel() async {
        guard model.canUnloadModel else { return }
        model.unloadModel()
        for _ in 0..<100 where model.loadState != .notLoaded {
            try? await Task.sleep(for: .milliseconds(100))
        }
    }


    static func gigabytes(_ bytes: UInt64) -> String {
        String(format: "%.1f GB", Double(bytes) / 1e9)
    }
}
