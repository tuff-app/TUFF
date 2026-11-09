import AppKit
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// The composer's Web and Files switches: two icons, tinted while on, so what
/// the next message may use is visible before it is sent. Files opens its
/// folder list when there is nothing to search yet, and from its context menu.
struct ToolToggleControls: View {
    @Bindable var model: AppModel
    @State private var showingFolders = false

    var body: some View {
        HStack(spacing: 6) {
            Button {
                model.webSearchEnabled.toggle()
            } label: {
                ComposerIcon(systemImage: "globe", isOn: webOn,
                             badge: model.webSearchEnabled && model.isWebSearchPaused ? TUFFMacTheme.accentTextColor : nil)
            }
            .buttonStyle(.plain)
            .help(webHelp)
            .accessibilityLabel("Web search")
            .accessibilityValue(model.isWebSearchPaused && model.webSearchEnabled
                                ? "Paused while offline" : (webOn ? "On" : "Off"))

            Button {
                if model.toolStore.folders.isEmpty {
                    showingFolders = true
                } else {
                    model.fileSearchEnabled.toggle()
                }
            } label: {
                ComposerIcon(systemImage: "folder", isOn: filesOn)
            }
            .buttonStyle(.plain)
            .help(filesHelp)
            .contextMenu {
                Button("Choose Folders…") { showingFolders = true }
            }
            .accessibilityLabel("File search")
            .accessibilityValue(filesOn ? "On, \(model.toolStore.folders.count) folders" : "Off")
            .accessibilityAction(named: "Choose folders") { showingFolders = true }
            .popover(isPresented: $showingFolders, arrowEdge: .bottom) {
                FolderAccessPanel(model: model)
                    .frame(width: 360)
                    .padding(14)
            }
        }
        .appFont(.callout)
        .disabled(!model.toolSupport.allowsTools || model.isRunning)
    }

    private var webOn: Bool { model.webSearchEnabled && !model.isWebSearchPaused && model.toolSupport.allowsTools }

    private var filesOn: Bool {
        model.fileSearchEnabled && model.toolSupport.allowsTools && !model.toolStore.folders.isEmpty
    }

    private var webHelp: String {
        if let note = unavailableNote { return note }
        if model.isWebSearchPaused && model.webSearchEnabled {
            return "Web: paused while offline. It resumes when your connection returns."
        }
        return (webOn ? "Web: on. " : "Web: off. ")
            + model.searchProvider.privacyNote
            + (model.toolSupport.note.map { " " + $0 } ?? "")
    }

    private var filesHelp: String {
        if let note = unavailableNote { return note }
        if model.toolStore.folders.isEmpty {
            return "Files: choose folders for the model to search. Their contents stay on this Mac."
        }
        let count = model.toolStore.folders.count
        return "Files: \(filesOn ? "on" : "off"), \(count) folder\(count == 1 ? "" : "s"). "
            + "Right-click to choose folders."
    }

    private var unavailableNote: String? {
        if case .unavailable(let reason) = model.toolSupport { return reason }
        return nil
    }
}

/// The folders file search may read, their index state, and the controls to
/// add, remove or stop using them. Shared by the composer and Settings.
struct FolderAccessPanel: View {
    @Bindable var model: AppModel
    var showsToggle = true

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if showsToggle {
                Toggle("Search these folders", isOn: $model.fileSearchEnabled)
                    .toggleStyle(TUFFRowToggleStyle())
                    .disabled(model.toolStore.folders.isEmpty)
            }
            if model.toolStore.folders.isEmpty {
                Text("Only folders you add are read. Nothing leaves this Mac.")
                    .appFont(.caption)
                    .foregroundStyle(TUFFDesign.secondaryTextColor)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(model.toolStore.folders) { folder in
                        folderRow(folder)
                    }
                }
            }
            if !model.toolStore.folders.isEmpty {
                Divider()
                GlassMenuPicker("Workspace", selection: $model.codingFolderID,
                                options: [""] + model.toolStore.folders.map(\.id.uuidString),
                                label: { id in model.toolStore.folders.first { $0.id.uuidString == id }?.displayName ?? "None" })
                Toggle("Write and edit files", isOn: $model.codingWriteEnabled)
                    .disabled(!model.fileSearchEnabled)
                Toggle("Run commands", isOn: $model.codingCommandEnabled)
                    .disabled(!model.fileSearchEnabled)
                Toggle("Allow command network access", isOn: $model.codingNetworkEnabled)
                    .disabled(!model.fileSearchEnabled || !model.codingCommandEnabled)
                Text("The model can read the workspace. Commands stay inside it.")
                    .appFont(.caption).foregroundStyle(TUFFDesign.secondaryTextColor)
                    .help("Choose a workspace to let the model list and read its files directly, including hidden files. Commands are confined to the workspace. Write access requires Write and edit files; network access is off unless enabled. Search can use all added folders.")
            }
            HStack {
                Button("Add Folder…", action: addFolders)
                Spacer()
                if model.toolStore.isIndexing {
                    Button("Stop Indexing") { model.toolStore.cancelIndexing() }
                } else if !model.toolStore.folders.isEmpty {
                    Button("Update Index") { model.toolStore.reindex() }
                }
            }
            if let error = model.toolStore.lastError {
                Text(error)
                    .appFont(.caption)
                    .foregroundStyle(TUFFMacTheme.accentTextColor)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .disabled(model.isRunning)
        .help(model.isRunning ? "Folder and coding permissions can be changed after the current answer finishes." : "")
        // A folder added or removed: the rows and the options under them
        // ease in and out.
        .animation(TUFFMotion.standard, value: model.toolStore.folders.map(\.id))
        .animation(TUFFMotion.standard, value: model.toolStore.isIndexing)
        .animation(TUFFMotion.standard, value: model.toolStore.lastError)
    }

    private func folderRow(_ folder: AppSearchFolder) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: "folder")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(folder.displayName)
                    .lineLimit(1)
                Text(status(folder))
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            .help((folder.path as NSString).abbreviatingWithTildeInPath)
            Spacer(minLength: 4)
            Button {
                model.toolStore.removeFolder(id: folder.id)
                if model.toolStore.folders.isEmpty { model.fileSearchEnabled = false }
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .symbolRenderingMode(.hierarchical)
            }
            .buttonStyle(.borderless)
            .help("Stop searching this folder and delete its index")
            .accessibilityLabel("Remove \(folder.displayName)")
        }
    }

    private func status(_ folder: AppSearchFolder) -> String {
        let indexing = model.toolStore.indexingFolderID == folder.id
        guard let status = model.toolStore.folderStatus[folder.id] else {
            return indexing ? "Indexing…" : "Not indexed yet"
        }
        if let error = status.lastError { return error }
        var parts: [String] = []
        if indexing, status.pendingFiles > 0 {
            parts.append("Indexing, \(status.pendingFiles) file\(status.pendingFiles == 1 ? "" : "s") left")
        }
        parts.append("\(status.indexedFiles) file\(status.indexedFiles == 1 ? "" : "s") indexed")
        if status.skippedFiles > 0 { parts.append("\(status.skippedFiles) skipped") }
        if status.reachedFileLimit { parts.append("file limit reached") }
        return parts.joined(separator: ", ")
    }

    private func addFolders() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = []
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        panel.treatsFilePackagesAsDirectories = false
        panel.prompt = "Add"
        panel.message = "Choose folders for TUFF to search. Only these folders are read. "
            + "With the file sandbox on, macOS may ask whether \u{201C}container-runtime-linux\u{201D} "
            + "may read the folder; allow it."
        guard panel.runModal() == .OK else { return }
        model.toolStore.addFolders(panel.urls)
        if !model.toolStore.folders.isEmpty { model.fileSearchEnabled = true }
    }
}
