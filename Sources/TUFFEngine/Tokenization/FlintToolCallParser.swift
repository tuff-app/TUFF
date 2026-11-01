import Foundation
import TUFFInferenceTypes

/// Parses the publisher's <tool_call>function<arg_key>... XML body.
struct FlintToolCallParser {
    func parse(_ text: String, allowedTools: Set<String>, id: String) throws -> ParsedToolCall {
        guard text.utf8.count <= QwenToolCallParser.maximumBytes else { throw ToolCallParserError.oversized }
        var body = Substring(text)
        while body.first?.isWhitespace == true { body.removeFirst() }
        let nameEnd = body.firstIndex(of: "<") ?? body.endIndex
        let name = String(body[..<nameEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard name.range(of: "^[A-Za-z0-9_-]{1,64}$", options: .regularExpression) != nil else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else { throw ToolCallParserError.unknownTool(name) }
        body = body[nameEnd...]
        var args: [String: JSONValue] = [:]
        while !body.isEmpty {
            while body.first?.isWhitespace == true { body.removeFirst() }
            if body.isEmpty { break }
            guard body.hasPrefix("<arg_key>") else { throw ToolCallParserError.malformed }
            body.removeFirst("<arg_key>".count)
            guard let keyEnd = body.range(of: "</arg_key>") else { throw ToolCallParserError.malformed }
            let key = String(body[..<keyEnd.lowerBound])
            guard !key.isEmpty, !key.contains("<"), args[key] == nil else { throw ToolCallParserError.malformed }
            body = body[keyEnd.upperBound...]
            guard body.hasPrefix("<arg_value>") else { throw ToolCallParserError.malformed }
            body.removeFirst("<arg_value>".count)
            guard let end = body.range(of: "</arg_value>") else { throw ToolCallParserError.malformed }
            let raw = String(body[..<end.lowerBound])
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if let first = trimmed.first, "{[-0123456789tfn".contains(first),
               let parsed = try? JSONDecoder().decode(JSONValue.self, from: Data(trimmed.utf8)),
               case .string = parsed { args[key] = .string(raw) }
            else if let first = trimmed.first, "{[-0123456789tfn".contains(first),
                    let parsed = try? JSONDecoder().decode(JSONValue.self, from: Data(trimmed.utf8)) { args[key] = parsed }
            else { args[key] = .string(raw) }
            body = body[end.upperBound...]
        }
        let arguments = JSONValue.object(args)
        return ParsedToolCall(id: id, name: name, arguments: arguments, argumentsJSON: try arguments.encoded())
    }
}
