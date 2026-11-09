import AppKit
import SwiftUI
import TUFFModelCatalog
import TUFFAppCore
import TUFFMacPresentation

struct BugReportSheet: View {
    let model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var includeDiagnostics = false

    private var report: AppBugReport {
        AppBugReport(system: .current(), model: TUFFModelCatalog.all.first { $0.apiModelID == model.selectedDescriptor.apiModelID },
            contextTokens: model.effectiveMaxContextTokens, temperature: model.temperature,
            topK: model.topKEnabled ? model.topK : 0, topP: model.topPEnabled ? model.topP : 1,
            runtime: model.runtimeOptions, diagnostics: model.diagnostics)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Report a Bug").appFont(.title2.weight(.semibold))
            Text("Describe what happened in the GitHub bug form. You can include the diagnostic summary below after reviewing it.")
            Toggle("Include diagnostic summary", isOn: $includeDiagnostics)
            ScrollView {
                Text(report.summary).appFont(.callout.monospaced()).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }.frame(height: 240)
            Text("The summary contains system details, model identity, settings and timing. It contains no chat text, images, credentials or file paths.")
                .appFont(.caption).foregroundStyle(.secondary)
            HStack {
                Button("Cancel", role: .cancel) { dismiss() }
                Spacer()
                Button("Copy Summary") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(report.summary, forType: .string)
                }
                Button("Open Bug Form") {
                    NSWorkspace.shared.open(report.issueURL(includeDiagnostics: includeDiagnostics))
                    dismiss()
                }.buttonStyle(.tuffProminent)
            }
        }.padding(24).frame(width: 560)
    }
}
