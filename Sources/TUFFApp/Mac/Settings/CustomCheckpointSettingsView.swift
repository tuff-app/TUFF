import AppKit
import TUFFAppCore
import TUFFModelCatalog
import SwiftUI
import TUFFMacPresentation

struct CustomCheckpointSettingsView: View {
    @Bindable var model: AppModel
    @State private var name = ""
    @State private var architecture = TUFFModelCatalog.default.id
    @State private var path = ""

    var body: some View {
        Section("Custom checkpoints") {
            DisclosureGroup("Add a custom checkpoint") {
                // The same rows as the rest of Settings: label on the left,
                // control on the right, the action last.
                VStack(alignment: .leading, spacing: 12) {
                    LabeledContent("Name") {
                        TextField("My checkpoint", text: $name)
                            .tuffField()
                            .multilineTextAlignment(.trailing)
                            .frame(maxWidth: 240)
                    }
                    GlassMenuPicker("Architecture", selection: $architecture,
                                    options: TUFFModelCatalog.all.map(\.id),
                                    label: { id in TUFFModelCatalog.all.first { $0.id == id }?.shortName ?? "" })
                    LabeledContent("Folder") {
                        HStack(spacing: 10) {
                            Text(path.isEmpty ? "None chosen" : URL(fileURLWithPath: path).lastPathComponent)
                                .foregroundStyle(TUFFDesign.secondaryTextColor)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Button("Choose…") { chooseCheckpoint() }
                                .controlSize(.small)
                        }
                    }
                    HStack(alignment: .center, spacing: 12) {
                        Text("Text only. Needs a verified TUFF pack.")
                            .appFont(.caption)
                            .foregroundStyle(TUFFDesign.secondaryTextColor)
                            .help("Requires a verified TUFF pack with matching architecture, tokenizer and quantization. The selected engine is reused. Raw weights need repacking first.")
                        Spacer(minLength: 8)
                        Button(model.isImportingCustomCheckpoint ? "Verifying…" : "Verify and Add") {
                            Task {
                                await model.importCustomCheckpoint(directory: URL(fileURLWithPath: path),
                                    name: name, templateID: architecture)
                                if model.customCheckpointImportError == nil { name = ""; path = "" }
                            }
                        }
                        .buttonStyle(.tuffProminent)
                        .disabled(path.isEmpty || name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || model.isImportingCustomCheckpoint || model.isRunning || model.loadState.isLoading)
                    }
                }
                .padding(.top, 10)
            }
            forEachCheckpoint
            if let error = model.customCheckpointImportError {
                Text(error).font(.caption).foregroundStyle(TUFFMacTheme.accentTextColor).textSelection(.enabled)
            }
        }
    }

    private var forEachCheckpoint: some View {
        ForEach(model.customCheckpointInstalls) { checkpoint in
            HStack {
                Text(checkpoint.descriptor.displayName)
                Spacer()
                Button("Remove entry") { model.removeCustomCheckpoint(checkpoint) }
                    .disabled(model.isImportingCustomCheckpoint)
            }
        }
    }

    private func chooseCheckpoint() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = []
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.treatsFilePackagesAsDirectories = true
        panel.prompt = "Choose checkpoint"
        panel.message = ""
        if panel.runModal() == .OK, let selected = panel.url {
            path = selected.path
            if name.isEmpty { name = selected.deletingPathExtension().lastPathComponent }
        }
    }
}
