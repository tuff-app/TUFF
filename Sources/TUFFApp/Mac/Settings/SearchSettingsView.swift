import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// Where web searches go, the keys for providers that need one, and the
/// folders file search may read.
struct SearchSettingsView: View {
    @Bindable var model: AppModel
    @State private var keyDrafts: [AppSearchProviderKind: String] = [:]
    @State private var confirmingRemoval = false

    var body: some View {
        Form {
            Section("Web search") {
                if model.isSandboxSupported || model.runWebInSandbox {
                    sandboxRows
                }
                Toggle("Automatically pause web search while offline", isOn: $model.pauseWebSearchWhenOffline)
                Text("When your connection returns, web search resumes only if you left Web on. A connection does not guarantee that the search provider is available.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Picker("Provider", selection: $model.searchProvider) {
                    ForEach(AppSearchProviderKind.allCases) { provider in
                        Text(provider.displayName).tag(provider)
                    }
                }
                Text(model.searchProvider.privacyNote)
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if model.searchProvider.requiresKey {
                    keyRow(model.searchProvider)
                    if model.runWebInSandbox {
                        Text("With the sandbox on, the key goes from TUFF to the sandbox and from there only to \(model.searchProvider.displayName).")
                            .appFont(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Text("Web search sends only the model's search query. The contents of your files are never sent to a search provider, and a query that copies text from them is refused.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Section("Files") {
                if model.isSandboxSupported || model.runFilesInSandbox {
                    filesSandboxRows
                }
                FolderAccessPanel(model: model)
                if !model.toolStore.folders.isEmpty {
                    Button("Remove All Folders and Delete Index…", role: .destructive) {
                        confirmingRemoval = true
                    }
                    .disabled(model.isRunning)
                }
            }
            Section("Limits") {
                let limits = AppToolLimits.standard
                Text("Each answer may use up to \(limits.maximumToolRounds) rounds of tools (12 with write or command access), \(limits.maximumWebRequests) web requests and \(limits.maximumFileSearches) file searches, within \(Int(limits.maximumToolSeconds)) seconds. Pages are read up to \(AppWebPageReader.maximumBytes / 1_048_576) MB. Results are shortened to fit the model's context.")
                    .appFont(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let note = model.toolSupport.note {
                    Label(note, systemImage: "info.circle")
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .onAppear { model.toolStore.refreshKeyPresence() }
        .confirmationDialog("Remove every folder and delete the search index?",
                            isPresented: $confirmingRemoval) {
            Button("Remove All", role: .destructive) {
                model.toolStore.removeAllFolders()
                model.fileSearchEnabled = false
            }
        } message: {
            Text("Your files are not touched. TUFF stops reading the folders and deletes its index of them.")
        }
    }

    @ViewBuilder private var sandboxRows: some View {
        Toggle("Run web search and page reads in a sandbox", isOn: $model.runWebInSandbox)
        if model.runWebInSandbox || sandboxFailed {
            let status = sandboxStatus(model.sandbox.state)
            HStack(spacing: 6) {
                if status.busy { ProgressView().controlSize(.small) }
                Text(status.text)
                    .appFont(.caption)
                    .foregroundStyle(status.isProblem ? Color.red : Color.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if status.needsContainer, let releases = URL(string: "https://github.com/apple/container/releases") {
                Link("Apple container releases", destination: releases)
                    .appFont(.caption)
            }
        }
        Text("The sandbox is a small virtual machine run by Apple's container tool. Web searches are sent from inside it, and the pages the model reads are fetched and turned into text inside it, so a harmful page or file never reaches TUFF directly. TUFF starts it when you turn this on or open the app, and stops it when it has been idle or when TUFF quits. If it is not ready, web search fails; it never falls back to a direct request. File search has its own switch under Files. Off by default; needs macOS 26 and Apple container.")
            .appFont(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder private var filesSandboxRows: some View {
        Toggle("Read my search folders in a sandbox too", isOn: $model.runFilesInSandbox)
        if model.runFilesInSandbox || filesSandboxFailed {
            let status = filesSandboxStatus(model.filesSandbox.state)
            HStack(spacing: 6) {
                if status.busy { ProgressView().controlSize(.small) }
                Text(status.text)
                    .appFont(.caption)
                    .foregroundStyle(status.isProblem ? Color.red : Color.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        Text("A second sandbox with no network at all reads the folders you add, read-only, and turns the files into text, so a harmful file or PDF is never opened by TUFF itself. It runs while TUFF indexes your folders and stops when it has been idle or when TUFF quits. If it cannot start, indexing stops with a message; it never falls back to reading the files in TUFF. macOS may ask whether \u{201C}container-runtime-linux\u{201D} may read a folder such as Documents; allow it, or file search cannot read that folder. Folder names with a comma cannot be used. Off by default; needs macOS 26 and Apple container.")
            .appFont(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var filesSandboxFailed: Bool {
        if case .failed = model.filesSandbox.state { return true }
        return false
    }

    private func filesSandboxStatus(_ state: SandboxController.State)
        -> (text: String, busy: Bool, isProblem: Bool) {
        switch state {
        case .off:
            ("Stopped. It starts when TUFF indexes your folders.", false, false)
        case .preparing(let step): ("\(step)…", true, false)
        case .starting: ("Starting the sandbox…", true, false)
        case .ready: ("Ready", false, false)
        case .stopping: ("Stopping…", true, false)
        case .failed(let message): (message, false, true)
        }
    }

    private var sandboxFailed: Bool {
        if case .failed = model.sandbox.state { return true }
        return false
    }

    private func sandboxStatus(_ state: SandboxController.State)
        -> (text: String, busy: Bool, isProblem: Bool, needsContainer: Bool) {
        switch state {
        case .off:
            ("Stopped. It starts again when you send a message that uses the web.", false, false, false)
        case .preparing(let step): ("\(step)…", true, false, false)
        case .starting: ("Starting the sandbox…", true, false, false)
        case .ready: ("Ready", false, false, false)
        case .stopping: ("Stopping…", true, false, false)
        case .failed(let message):
            (message, false, true, message.hasPrefix(SandboxError.containerNotInstalled.description)
                || message.hasPrefix(SandboxError.containerNotTrusted.description))
        }
    }

    private func keyRow(_ provider: AppSearchProviderKind) -> some View {
        let saved = model.toolStore.keyPresence[provider] == true
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                SecureField(saved ? "Replace the saved key" : "\(provider.displayName) API key",
                            text: Binding(get: { keyDrafts[provider] ?? "" },
                                          set: { keyDrafts[provider] = $0 }))
                    .textContentType(.password)
                Button("Save") {
                    model.toolStore.setKey(keyDrafts[provider] ?? "", for: provider)
                    keyDrafts[provider] = nil
                }
                .disabled((keyDrafts[provider] ?? "").trimmingCharacters(in: .whitespaces).isEmpty)
                if saved {
                    Button("Remove", role: .destructive) {
                        model.toolStore.removeKey(for: provider)
                    }
                }
            }
            Text(saved
                 ? "A key is saved in your Keychain. It is used only for searches, and never shown to the model, saved in chats or written to logs."
                 : "No key is saved. Searches with \(provider.displayName) will fail until one is added.")
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
