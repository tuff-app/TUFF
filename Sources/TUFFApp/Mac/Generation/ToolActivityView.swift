import AppKit
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// One tool call as the transcript shows it, live or after the fact.
struct ToolActivityEntry: Identifiable, Equatable {
    let id: String
    let name: String
    let detail: String
    let summary: String
    let resultText: String
    let state: AppToolActivity.State

    var title: String {
        let running = state == .running
        if state == .failed || state == .refused || state == .cancelled {
            let action: String
            switch AppToolName(rawValue: name) {
            case .webSearch?: action = "Web search"
            case .readWebpage?: action = "Page read"
            case .searchFiles?: action = "File search"
            case .listFiles?: action = "File listing"
            case .readFile?: action = "File read"
            case .writeFile?: action = "File write"
            case .editFile?: action = "File edit"
            case .runCommand?: action = "Command"
            case nil: action = "Tool call"
            }
            let suffix = state == .failed ? "failed" : (state == .refused ? "not run" : "stopped")
            return "\(action) \(suffix)"
        }
        switch AppToolName(rawValue: name) {
        case .webSearch?: return running ? "Searching the web" : "Searched the web"
        case .readWebpage?: return running ? "Reading a web page" : "Read a web page"
        case .searchFiles?: return running ? "Searching your files" : "Searched your files"
        case .listFiles?: return running ? "Listing files" : "Listed files"
        case .readFile?: return running ? "Reading a file" : "Read a file"
        case .writeFile?: return running ? "Writing a file" : "Wrote a file"
        case .editFile?: return running ? "Editing a file" : "Edited a file"
        case .runCommand?: return running ? "Running a command" : "Ran a command"
        case nil: return name.isEmpty ? "Retried a tool call" : "Tool call"
        }
    }

    var symbol: String {
        switch state {
        case .running: "ellipsis"
        case .succeeded: "checkmark.circle"
        case .failed: "exclamationmark.triangle"
        case .refused: "nosign"
        case .cancelled: "stop.circle"
        }
    }

    static func entries(from activities: [AppToolActivity]) -> [ToolActivityEntry] {
        activities.map {
            ToolActivityEntry(id: $0.id, name: $0.name, detail: $0.detail,
                              summary: $0.summary, resultText: $0.resultText, state: $0.state)
        }
    }

    static func entries(from rounds: [AppToolRound]) -> [ToolActivityEntry] {
        rounds.flatMap { round in
            zip(round.calls, round.results).map { call, result in
                let state: AppToolActivity.State = switch result.status {
                case .succeeded: .succeeded
                case .failed: .failed
                case .refused: .refused
                case .cancelled: .cancelled
                }
                return ToolActivityEntry(
                    id: round.id.uuidString + call.id, name: call.name,
                    detail: AppToolAnswerSession.detail(for: call), summary: result.summary,
                    resultText: result.modelText, state: state)
            }
        }
    }
}

/// What the tools did for an answer, folded to one line until opened: each
/// call with its query and outcome, then the numbered sources the answer can
/// cite, each of which opens the page or file it came from.
struct ToolActivityView: View {
    let entries: [ToolActivityEntry]
    let sources: [AppSource]
    let isRunning: Bool
    let reduceTransparency: Bool
    @State private var isExpanded: Bool

    init(entries: [ToolActivityEntry], sources: [AppSource], isRunning: Bool,
         reduceTransparency: Bool, initiallyExpanded: Bool = false) {
        self.entries = entries
        self.sources = sources
        self.isRunning = isRunning
        self.reduceTransparency = reduceTransparency
        _isExpanded = State(initialValue: initiallyExpanded)
    }

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(entries) { entry in
                    entryRow(entry)
                }
                if !sources.isEmpty {
                    Divider()
                    ForEach(sources) { source in
                        SourceRow(source: source)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 8)
        } label: {
            HStack(spacing: 6) {
                if isRunning {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: headlineSymbol)
                }
                Text(headline)
                if !sources.isEmpty {
                    Text("· \(sources.count) source\(sources.count == 1 ? "" : "s")")
                        .foregroundStyle(.tertiary)
                }
            }
            .appFont(.caption.weight(.medium))
            .foregroundStyle(.secondary)
        }
        .padding(10)
        .background(
            TUFFMacTheme.surfaceStyle(reduceTransparency: reduceTransparency, material: .thin),
            in: RoundedRectangle(cornerRadius: 10))
        .accessibilityLabel("Tool activity")
        .accessibilityValue(headline)
        .accessibilityHint("Shows tool calls, results, and any sources")
    }

    /// The newest running call, or a summary of what ran.
    var headline: String {
        if let running = entries.last(where: { $0.state == .running }) {
            return running.detail.isEmpty ? running.title + "…"
                : "\(running.title): \(running.detail)"
        }
        let kinds = Set(entries.compactMap { AppToolName(rawValue: $0.name) })
        let usedWeb = kinds.contains(.webSearch) || kinds.contains(.readWebpage)
        let usedFiles = kinds.contains(.searchFiles)
        let usedWorkspace = kinds.contains(.listFiles) || kinds.contains(.readFile)
            || kinds.contains(.writeFile) || kinds.contains(.editFile)
        let usedCommand = kinds.contains(.runCommand)
        let done: String
        if entries.count > 1 && (usedWorkspace || usedCommand) {
            if usedWeb || usedFiles { done = "Used search and workspace tools" }
            else if usedWorkspace && usedCommand { done = "Worked with files and ran commands" }
            else if usedCommand { done = "Ran commands" }
            else { done = "Worked with files" }
        } else { switch (usedWeb, usedFiles) {
        case (true, true): done = "Searched the web and your files"
        case (true, false): done = kinds.contains(.webSearch) ? "Searched the web" : "Read a web page"
        case (false, true): done = "Searched your files"
        case (false, false): done = entries.last?.title ?? "Used tools"
        } }
        if entries.contains(where: { $0.state == .failed }) { return done + ", with errors" }
        return done
    }

    private var headlineSymbol: String {
        entries.contains { $0.state == .failed } ? "exclamationmark.triangle"
            : entries.contains { $0.name == AppToolName.runCommand.rawValue } ? "terminal"
            : entries.contains { [AppToolName.listFiles.rawValue, AppToolName.readFile.rawValue,
                                   AppToolName.writeFile.rawValue, AppToolName.editFile.rawValue].contains($0.name) } ? "folder"
            : entries.contains { $0.name == AppToolName.searchFiles.rawValue }
                && !entries.contains { $0.name == AppToolName.webSearch.rawValue }
                ? "doc.text.magnifyingglass" : "globe"
    }

    private func entryRow(_ entry: ToolActivityEntry) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: entry.symbol)
                .foregroundStyle(entry.state == .failed ? AnyShapeStyle(.orange)
                                 : AnyShapeStyle(.secondary))
                .frame(width: 14)
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.detail.isEmpty ? entry.title : "\(entry.title): \(entry.detail)")
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                if !entry.summary.isEmpty && entry.summary != "Completed" {
                    Text(entry.summary)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if !entry.resultText.isEmpty {
                    ToolResultDisclosure(text: entry.resultText)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .appFont(.caption)
        .textSelection(.enabled)
    }
}

/// Each call owns its own disclosure so output is available without dumping
/// every result into the transcript by default.
private struct ToolResultDisclosure: View {
    let text: String
    @State private var expanded = false
    var body: some View {
        DisclosureGroup("Result", isExpanded: $expanded) {
            Text(text)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A numbered source: the title opens it; the excerpt is what the answer
/// was given.
struct SourceRow: View {
    let source: AppSource
    @State private var showsExcerpt = false

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("[\(source.id)]")
                    .appFont(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                Button {
                    if let url = source.openURL {
                        if url.isFileURL {
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        } else {
                            NSWorkspace.shared.open(url)
                        }
                    }
                } label: {
                    Text(source.title)
                        .lineLimit(1)
                        .foregroundStyle(TUFFMacTheme.accentColor)
                }
                .buttonStyle(.plain)
                .disabled(source.openURL == nil)
                .help(source.openURL.map { $0.isFileURL ? $0.path : $0.absoluteString } ?? "")
                Text(location)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 0)
                if !source.excerpt.isEmpty {
                    Button(showsExcerpt ? "Hide excerpt" : "Excerpt") { showsExcerpt.toggle() }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                }
            }
            if showsExcerpt {
                Text(source.excerpt)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, 26)
            }
        }
        .appFont(.caption)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Source \(source.id), \(source.title), \(location)")
    }

    private var location: String {
        switch source.kind {
        case .web:
            return source.url.flatMap(URL.init(string:))?.host ?? source.origin
        case .file:
            return [source.location, source.filePath.map { AppFileNameText.safe(($0 as NSString).abbreviatingWithTildeInPath) }]
                .compactMap { $0 }.joined(separator: " · ")
        }
    }
}
