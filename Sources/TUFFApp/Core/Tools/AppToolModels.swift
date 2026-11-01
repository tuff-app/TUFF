import Foundation
import TUFFInferenceTypes

/// What a chat message may use. Chosen in the composer per message and stored
/// with the turn, so a saved chat renders the same tool declarations it was
/// answered with.
public struct AppChatCapabilities: Codable, Equatable, Hashable, Sendable {
    public var web: Bool
    public var files: Bool
    /// Raw list_files/read_file over an explicitly chosen coding workspace.
    /// Separate from `files` (index search) so enabling Local Files never
    /// grants unfiltered reads.
    public var workspaceRead: Bool
    public var fileWrite: Bool
    public var command: Bool
    public var commandNetwork: Bool

    public init(web: Bool = false, files: Bool = false, workspaceRead: Bool = false, fileWrite: Bool = false, command: Bool = false, commandNetwork: Bool = false) {
        self.web = web
        self.files = files
        self.workspaceRead = workspaceRead
        self.fileWrite = fileWrite
        self.command = command
        self.commandNetwork = commandNetwork
    }

    public static let none = AppChatCapabilities()
    public var isEmpty: Bool { !web && !files && !workspaceRead && !fileWrite && !command }

    private enum CodingKeys: String, CodingKey { case web, files, workspaceRead, fileWrite, command, commandNetwork }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        web = try c.decodeIfPresent(Bool.self, forKey: .web) ?? false
        files = try c.decodeIfPresent(Bool.self, forKey: .files) ?? false
        workspaceRead = try c.decodeIfPresent(Bool.self, forKey: .workspaceRead) ?? false
        fileWrite = try c.decodeIfPresent(Bool.self, forKey: .fileWrite) ?? false
        command = try c.decodeIfPresent(Bool.self, forKey: .command) ?? false
        commandNetwork = try c.decodeIfPresent(Bool.self, forKey: .commandNetwork) ?? false
    }
}

/// A call the model made, exactly as the structured decoder parsed it. The
/// identifier is the decoder's, and it is kept so a later render pairs each
/// result with its call the same way the model saw it.
public struct AppToolCall: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let arguments: JSONValue

    public init(id: String, name: String, arguments: JSONValue) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    public var historical: InferenceToolCall {
        .init(id: id, name: name, arguments: arguments)
    }
}

public enum AppToolResultStatus: String, Codable, Equatable, Sendable {
    /// The tool ran and its output was given to the model.
    case succeeded
    /// The tool ran and failed; the model was told why.
    case failed
    /// The call was refused before running: an unknown tool, invalid
    /// arguments, a capability that was off, or a limit reached.
    case refused
    /// Stopped by the user before it finished.
    case cancelled
}

/// A command produced output but did not complete successfully. Keeping the
/// captured text separate from the reason lets the transcript mark the call
/// failed without hiding diagnostics from the model or the person.
public struct AppCommandFailure: Error, Equatable, Sendable, CustomStringConvertible {
    public enum Reason: Equatable, Sendable {
        case exit(Int32)
        case signal(Int32)
        case timedOut(Int)
        case statusUnavailable(Int32)

        public var summary: String {
            switch self {
            case .exit(let code): "Exit status \(code)"
            case .signal(let signal): "Terminated by signal \(signal)"
            case .timedOut(let seconds):
                "Command timed out after \(seconds) second\(seconds == 1 ? "" : "s")"
            case .statusUnavailable(let code):
                "Could not determine command status (waitpid error \(code))"
            }
        }
    }

    public let output: String
    public let reason: Reason

    public init(output: String, reason: Reason) {
        self.output = output
        self.reason = reason
    }

    public var summary: String { reason.summary }
    public var description: String { output.isEmpty ? "[\(summary)]" : output }
}

/// One tool result. `modelText` is exactly what the model read, kept so the
/// conversation re-renders identically; `summary` is the short line the
/// transcript shows.
public struct AppToolResult: Codable, Equatable, Sendable {
    public let callID: String
    public let name: String
    public let status: AppToolResultStatus
    public let modelText: String
    public let summary: String
    /// Sources this result introduced, by citation number.
    public let sourceIDs: [Int]

    public init(callID: String, name: String, status: AppToolResultStatus,
                modelText: String, summary: String, sourceIDs: [Int] = []) {
        self.callID = callID
        self.name = name
        self.status = status
        self.modelText = modelText
        self.summary = summary
        self.sourceIDs = sourceIDs
    }
}

/// One generation that ended in tool calls, and the results that answered
/// them. Reasoning, any text the model wrote before its calls, the calls and
/// their results stay separate fields: they render as separate template
/// parts and must never be flattened into a user message.
public struct AppToolRound: Codable, Equatable, Sendable, Identifiable {
    public let id: UUID
    public var thinking: String?
    public var content: String
    public var calls: [AppToolCall]
    public var results: [AppToolResult]

    public init(id: UUID = UUID(), thinking: String? = nil, content: String = "",
                calls: [AppToolCall], results: [AppToolResult] = []) {
        self.id = id
        self.thinking = thinking
        self.content = content
        self.calls = calls
        self.results = results
    }

    /// Every call has a result, in call order. A round is only sent back to
    /// the model in this state.
    public var isComplete: Bool {
        results.count == calls.count
            && zip(calls, results).allSatisfy { $0.id == $1.callID && $0.name == $1.name }
    }
}

public enum AppSourceKind: String, Codable, Equatable, Sendable {
    case web
    case file
}

/// Something retrieved for an answer. The application assigns the number;
/// the model only ever sees it, and a citation is valid only if it names one.
public struct AppSource: Codable, Equatable, Sendable, Identifiable {
    public let id: Int
    public let kind: AppSourceKind
    public let title: String
    /// An http or https URL for a web source.
    public let url: String?
    /// The file a local source came from.
    public let filePath: String?
    /// One-based PDF page, when there is one.
    public let page: Int?
    /// A human description of where in the file: "page 3", "lines 40-72".
    public let location: String?
    /// The text the model was given, so a saved answer keeps what it used.
    public let excerpt: String
    /// The search provider or "page" for a webpage the model read.
    public let origin: String

    public init(id: Int, kind: AppSourceKind, title: String, url: String? = nil,
                filePath: String? = nil, page: Int? = nil, location: String? = nil,
                excerpt: String, origin: String) {
        self.id = id
        self.kind = kind
        self.title = title
        self.url = url
        self.filePath = filePath
        self.page = page
        self.location = location
        self.excerpt = excerpt
        self.origin = origin
    }

    /// Where a click goes: the web page, or the local file.
    public var openURL: URL? {
        switch kind {
        case .web:
            guard let url, let parsed = URL(string: url),
                  let scheme = parsed.scheme?.lowercased(),
                  scheme == "http" || scheme == "https" else { return nil }
            return parsed
        case .file:
            return filePath.map { URL(fileURLWithPath: $0) }
        }
    }
}
