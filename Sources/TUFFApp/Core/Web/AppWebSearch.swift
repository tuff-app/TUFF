import Foundation

public enum AppSearchProviderKind: String, Codable, CaseIterable, Sendable, Identifiable {
    case duckDuckGo = "duckduckgo"
    case brave
    case tavily

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .duckDuckGo: "DuckDuckGo"
        case .brave: "Brave Search"
        case .tavily: "Tavily"
        }
    }

    public var requiresKey: Bool { self != .duckDuckGo }

    /// Where the query goes, in the words the composer and Settings use.
    public var privacyNote: String {
        switch self {
        case .duckDuckGo:
            "Searches send the query to DuckDuckGo. No key is needed. When DuckDuckGo challenges a search, TUFF tries its lite page once and otherwise reports the block."
        case .brave:
            "Searches send the query to Brave Search with your API key."
        case .tavily:
            "Searches send the query to Tavily with your API key."
        }
    }
}

public struct AppWebResult: Equatable, Sendable {
    public let title: String
    public let url: URL
    public let excerpt: String

    public init(title: String, url: URL, excerpt: String) {
        self.title = title
        self.url = url
        self.excerpt = excerpt
    }
}

public struct AppWebSearchResponse: Equatable, Sendable {
    public let provider: AppSearchProviderKind
    public let results: [AppWebResult]
}

/// Why a search returned nothing. Each case says what actually happened;
/// none of them is retried against another provider behind the user's back.
public enum AppWebSearchError: Error, Equatable, Sendable, CustomStringConvertible {
    case missingKey(AppSearchProviderKind)
    case invalidKey(AppSearchProviderKind)
    case rateLimited(AppSearchProviderKind)
    case blocked(AppSearchProviderKind, status: Int)
    case challenge(AppSearchProviderKind)
    case markupChanged(AppSearchProviderKind)
    case noResults(AppSearchProviderKind)
    case http(AppSearchProviderKind, status: Int)
    case transport(AppSearchProviderKind, AppHTTPError)

    public var description: String {
        switch self {
        case .missingKey(let provider):
            "\(provider.displayName) needs an API key. Add one in Settings, or choose DuckDuckGo."
        case .invalidKey(let provider):
            "\(provider.displayName) rejected the API key."
        case .rateLimited(let provider):
            "\(provider.displayName) is limiting requests or the plan's quota is used up. Try again later."
        case .blocked(let provider, let status):
            "\(provider.displayName) refused the search (HTTP \(status)). It may be blocking automated requests."
        case .challenge(let provider):
            "\(provider.displayName) asked for a verification step that TUFF does not complete. No results were retrieved."
        case .markupChanged(let provider):
            "\(provider.displayName) returned a page TUFF could not read. Its format may have changed."
        case .noResults(let provider):
            "\(provider.displayName) found no results."
        case .http(let provider, let status):
            "\(provider.displayName) returned HTTP \(status)."
        case .transport(let provider, .connectionFailed(code: 35)):
            "\(provider.displayName): \(AppHTTPError.connectionFailed(code: 35).description) You can also choose another search provider in Settings."
        case .transport(let provider, let error):
            "\(provider.displayName): \(error.description)"
        }
    }
}

public protocol AppWebSearchProvider: Sendable {
    var kind: AppSearchProviderKind { get }
    func search(_ query: String, count: Int) async throws -> AppWebSearchResponse
}

public enum AppWebSearchLimits {
    public static let maximumResponseBytes = 1_024 * 1_024
    public static let maximumExcerptCharacters = 500
    public static let maximumTitleCharacters = 200
}

func boundedSearchText(_ text: String, limit: Int) -> String {
    let collapsed = AppHTML.collapse(text)
    guard collapsed.count > limit else { return collapsed }
    return String(collapsed.prefix(limit)) + "..."
}

/// DuckDuckGo's HTML results page, which needs no key. The page is parsed
/// for its result anchors and snippets; ads are skipped. A verification page,
/// a refusal, a page without the expected markup and a genuine empty result
/// are told apart and reported as such.
///
/// DuckDuckGo sometimes answers the HTML page with a verification step or a
/// refusal while its lite page still serves the same search. The query then
/// goes once to the lite page, which is the same provider: it is never sent
/// anywhere else. If the lite page answers with an error, that error is
/// reported; if it cannot be reached, the original block is.
public struct DuckDuckGoSearchProvider: AppWebSearchProvider {
    public let kind = AppSearchProviderKind.duckDuckGo
    public static let endpoint = URL(string: "https://html.duckduckgo.com/html/")!
    public static let liteEndpoint = URL(string: "https://lite.duckduckgo.com/lite/")!
    private let transport: any AppHTTPTransport
    private let timeoutSeconds: Double

    public init(transport: any AppHTTPTransport, timeoutSeconds: Double = 15) {
        self.transport = transport
        self.timeoutSeconds = timeoutSeconds
    }

    public func search(_ query: String, count: Int) async throws -> AppWebSearchResponse {
        var form = URLComponents()
        form.queryItems = [URLQueryItem(name: "q", value: query)]
        let body = Data((form.percentEncodedQuery ?? "").utf8)
        do {
            let response = try await post(body, to: Self.endpoint)
            return AppWebSearchResponse(provider: kind,
                                        results: try Self.parse(response, count: count))
        } catch let block as AppWebSearchError {
            switch block {
            case .challenge, .blocked: break
            default: throw block
            }
            let response: AppHTTPResponse
            do {
                response = try await post(body, to: Self.liteEndpoint)
            } catch AppWebSearchError.transport {
                // The lite page could not be reached at all; the web sandbox,
                // for one, does not allow it. The block is what happened.
                throw block
            }
            return AppWebSearchResponse(
                provider: kind,
                results: try Self.parse(response, count: count, results: Self.parseLiteResults))
        }
    }

    private func post(_ body: Data, to url: URL) async throws -> AppHTTPResponse {
        do {
            return try await transport.perform(AppHTTPRequest(
                url: url, method: "POST",
                headers: ["Content-Type": "application/x-www-form-urlencoded",
                          "Accept": "text/html"],
                body: body, maximumBytes: AppWebSearchLimits.maximumResponseBytes,
                timeoutSeconds: timeoutSeconds))
        } catch let error as AppHTTPError {
            throw AppWebSearchError.transport(kind, error)
        }
    }

    static func parse(
        _ response: AppHTTPResponse, count: Int,
        results parseResults: (String, Int) -> [AppWebResult] = DuckDuckGoSearchProvider.parseResults
    ) throws -> [AppWebResult] {
        // 202 is DuckDuckGo's answer to traffic it treats as automated.
        switch response.statusCode {
        case 200: break
        case 202, 403: throw AppWebSearchError.blocked(.duckDuckGo, status: response.statusCode)
        case 429: throw AppWebSearchError.rateLimited(.duckDuckGo)
        default: throw AppWebSearchError.http(.duckDuckGo, status: response.statusCode)
        }
        let html = response.text
        let lowered = html.lowercased()
        if lowered.contains("anomaly-modal") || lowered.contains("challenge-form")
            || lowered.contains("bots use duckduckgo too") || lowered.contains("anomaly.js") {
            throw AppWebSearchError.challenge(.duckDuckGo)
        }
        let results = parseResults(html, count)
        if results.isEmpty {
            if lowered.contains("class=\"no-results\"") || lowered.contains("no results.") {
                throw AppWebSearchError.noResults(.duckDuckGo)
            }
            throw AppWebSearchError.markupChanged(.duckDuckGo)
        }
        return results
    }

    /// Reads results in document order. Each result is a `div.result`; an ad
    /// is one carrying `result--ad`, and everything until the next result is
    /// ignored. The title is the text of `a.result__a`, whose href is the
    /// link; the excerpt is the text of the `.result__snippet` element.
    static func parseResults(_ html: String, limit: Int) -> [AppWebResult] {
        struct Partial { var title = ""; var href: String?; var snippet = "" }
        var results: [AppWebResult] = []
        var current: Partial?
        var inAd = false
        var capturing: (kind: String, tag: String, nesting: Int)?
        func finish() {
            defer { current = nil }
            guard let partial = current, let href = partial.href,
                  let url = resolve(href) else { return }
            let title = boundedSearchText(partial.title, limit: AppWebSearchLimits.maximumTitleCharacters)
            guard !title.isEmpty, !results.contains(where: { $0.url == url }) else { return }
            results.append(AppWebResult(
                title: title, url: url,
                excerpt: boundedSearchText(partial.snippet,
                                           limit: AppWebSearchLimits.maximumExcerptCharacters)))
        }
        for token in AppHTML.tokens(html) {
            if results.count >= limit { break }
            switch token {
            case .start(let name, let attributes, let selfClosing):
                let classes = Set((attributes["class"] ?? "").split(separator: " ").map(String.init))
                if name == "div", classes.contains("result") {
                    finish()
                    capturing = nil
                    inAd = classes.contains("result--ad") || classes.contains("result--ad--small")
                    continue
                }
                if inAd || selfClosing { continue }
                if var open = capturing, open.tag == name {
                    open.nesting += 1
                    capturing = open
                }
                if name == "a", classes.contains("result__a") {
                    finish()
                    current = Partial(href: attributes["href"])
                    capturing = ("title", "a", 1)
                } else if classes.contains("result__snippet"), current != nil {
                    capturing = ("snippet", name, 1)
                }
            case .end(let name):
                if var open = capturing, open.tag == name {
                    open.nesting -= 1
                    capturing = open.nesting == 0 ? nil : open
                }
            case .text(let text):
                guard !inAd, let open = capturing else { continue }
                if open.kind == "title" { current?.title += text } else { current?.snippet += text }
            }
        }
        if results.count < limit { finish() }
        return Array(results.prefix(limit))
    }

    /// Reads the lite page's results in document order. Each result's link is
    /// an `a.result-link` and its excerpt the next `td.result-snippet`. Ads
    /// sit in rows marked `result-sponsored`, and those rows are ignored.
    static func parseLiteResults(_ html: String, limit: Int) -> [AppWebResult] {
        struct Partial { var title = ""; var href: String?; var snippet = "" }
        var results: [AppWebResult] = []
        var current: Partial?
        var inAd = false
        var capturing: (kind: String, tag: String, nesting: Int)?
        func finish() {
            defer { current = nil }
            guard let partial = current, let href = partial.href,
                  let url = resolve(href) else { return }
            let title = boundedSearchText(partial.title, limit: AppWebSearchLimits.maximumTitleCharacters)
            guard !title.isEmpty, !results.contains(where: { $0.url == url }) else { return }
            results.append(AppWebResult(
                title: title, url: url,
                excerpt: boundedSearchText(partial.snippet,
                                           limit: AppWebSearchLimits.maximumExcerptCharacters)))
        }
        for token in AppHTML.tokens(html) {
            if results.count >= limit { break }
            switch token {
            case .start(let name, let attributes, let selfClosing):
                let classes = Set((attributes["class"] ?? "").split(separator: " ").map(String.init))
                if name == "tr" {
                    capturing = nil
                    inAd = classes.contains("result-sponsored")
                    continue
                }
                if inAd || selfClosing { continue }
                if var open = capturing, open.tag == name {
                    open.nesting += 1
                    capturing = open
                }
                if name == "a", classes.contains("result-link") {
                    finish()
                    current = Partial(href: attributes["href"])
                    capturing = ("title", "a", 1)
                } else if name == "td", classes.contains("result-snippet"), current != nil {
                    capturing = ("snippet", "td", 1)
                }
            case .end(let name):
                if var open = capturing, open.tag == name {
                    open.nesting -= 1
                    capturing = open.nesting == 0 ? nil : open
                }
            case .text(let text):
                guard !inAd, let open = capturing else { continue }
                if open.kind == "title" { current?.title += text } else { current?.snippet += text }
            }
        }
        if results.count < limit { finish() }
        return Array(results.prefix(limit))
    }

    /// Result links are direct, or DuckDuckGo redirects whose target is the
    /// `uddg` parameter. Only http and https targets are kept.
    static func resolve(_ href: String) -> URL? {
        var text = href.trimmingCharacters(in: .whitespaces)
        if text.hasPrefix("//") { text = "https:" + text }
        guard var url = URL(string: text) else { return nil }
        // `URL.path` drops a trailing slash, so the redirect path reads "/l".
        if let host = url.host, isDuckDuckGoHost(host),
           url.path == "/l" || url.path.hasPrefix("/l/") {
            guard let target = URLComponents(url: url, resolvingAgainstBaseURL: false)?
                .queryItems?.first(where: { $0.name == "uddg" })?.value,
                  let resolved = URL(string: target) else { return nil }
            url = resolved
        }
        guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
              url.host != nil else { return nil }
        if let host = url.host, isDuckDuckGoHost(host) { return nil }
        return url
    }

    /// DuckDuckGo itself or one of its subdomains. A plain suffix match would
    /// also catch unrelated sites such as `notduckduckgo.com` and drop them.
    static func isDuckDuckGoHost(_ host: String) -> Bool {
        let host = host.lowercased()
        return host == "duckduckgo.com" || host.hasSuffix(".duckduckgo.com")
    }
}

/// Brave Search's web endpoint, with the user's subscription token.
public struct BraveSearchProvider: AppWebSearchProvider {
    public let kind = AppSearchProviderKind.brave
    public static let endpoint = URL(string: "https://api.search.brave.com/res/v1/web/search")!
    private let transport: any AppHTTPTransport
    private let key: String
    private let timeoutSeconds: Double

    public init(transport: any AppHTTPTransport, key: String, timeoutSeconds: Double = 15) {
        self.transport = transport
        self.key = key
        self.timeoutSeconds = timeoutSeconds
    }

    public func search(_ query: String, count: Int) async throws -> AppWebSearchResponse {
        guard !key.isEmpty else { throw AppWebSearchError.missingKey(kind) }
        var components = URLComponents(url: Self.endpoint, resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "q", value: query),
                                 URLQueryItem(name: "count", value: String(count))]
        let response: AppHTTPResponse
        do {
            response = try await transport.perform(AppHTTPRequest(
                url: components.url!,
                headers: ["Accept": "application/json", "X-Subscription-Token": key],
                maximumBytes: AppWebSearchLimits.maximumResponseBytes,
                timeoutSeconds: timeoutSeconds))
        } catch let error as AppHTTPError {
            throw AppWebSearchError.transport(kind, error)
        }
        return AppWebSearchResponse(provider: kind, results: try Self.parse(response, count: count))
    }

    static func parse(_ response: AppHTTPResponse, count: Int) throws -> [AppWebResult] {
        switch response.statusCode {
        case 200: break
        case 401, 403: throw AppWebSearchError.invalidKey(.brave)
        case 429: throw AppWebSearchError.rateLimited(.brave)
        default: throw AppWebSearchError.http(.brave, status: response.statusCode)
        }
        struct Body: Decodable {
            struct Web: Decodable { let results: [Result]? }
            struct Result: Decodable { let title: String?; let url: String?; let description: String? }
            let web: Web?
        }
        guard let body = try? JSONDecoder().decode(Body.self, from: response.body) else {
            throw AppWebSearchError.markupChanged(.brave)
        }
        let results = (body.web?.results ?? []).compactMap { item -> AppWebResult? in
            guard let text = item.url, let url = URL(string: text),
                  let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
                  let title = item.title.map({ boundedSearchText(
                      AppHTMLDocumentText.extract($0).text,
                      limit: AppWebSearchLimits.maximumTitleCharacters) }),
                  !title.isEmpty else { return nil }
            // Brave marks query terms with <strong>; strip markup.
            let excerpt = AppHTMLDocumentText.extract(item.description ?? "").text
            return AppWebResult(title: title, url: url,
                                excerpt: boundedSearchText(excerpt, limit: AppWebSearchLimits.maximumExcerptCharacters))
        }
        guard !results.isEmpty else { throw AppWebSearchError.noResults(.brave) }
        return Array(results.prefix(count))
    }
}

/// Tavily's search endpoint, with the user's API key.
public struct TavilySearchProvider: AppWebSearchProvider {
    public let kind = AppSearchProviderKind.tavily
    public static let endpoint = URL(string: "https://api.tavily.com/search")!
    private let transport: any AppHTTPTransport
    private let key: String
    private let timeoutSeconds: Double

    public init(transport: any AppHTTPTransport, key: String, timeoutSeconds: Double = 15) {
        self.transport = transport
        self.key = key
        self.timeoutSeconds = timeoutSeconds
    }

    public func search(_ query: String, count: Int) async throws -> AppWebSearchResponse {
        guard !key.isEmpty else { throw AppWebSearchError.missingKey(kind) }
        let body = try JSONSerialization.data(withJSONObject: [
            "query": query, "max_results": count, "search_depth": "basic",
            "include_answer": false, "include_raw_content": false,
        ])
        let response: AppHTTPResponse
        do {
            response = try await transport.perform(AppHTTPRequest(
                url: Self.endpoint, method: "POST",
                headers: ["Content-Type": "application/json", "Accept": "application/json",
                          "Authorization": "Bearer \(key)"],
                body: body, maximumBytes: AppWebSearchLimits.maximumResponseBytes,
                timeoutSeconds: timeoutSeconds))
        } catch let error as AppHTTPError {
            throw AppWebSearchError.transport(kind, error)
        }
        return AppWebSearchResponse(provider: kind, results: try Self.parse(response, count: count))
    }

    static func parse(_ response: AppHTTPResponse, count: Int) throws -> [AppWebResult] {
        switch response.statusCode {
        case 200: break
        case 401, 403: throw AppWebSearchError.invalidKey(.tavily)
        // Tavily reports exhausted plan credit with 432 and 433.
        case 429, 432, 433: throw AppWebSearchError.rateLimited(.tavily)
        default: throw AppWebSearchError.http(.tavily, status: response.statusCode)
        }
        struct Body: Decodable {
            struct Result: Decodable { let title: String?; let url: String?; let content: String? }
            let results: [Result]?
        }
        guard let body = try? JSONDecoder().decode(Body.self, from: response.body),
              let items = body.results else {
            throw AppWebSearchError.markupChanged(.tavily)
        }
        let results = items.compactMap { item -> AppWebResult? in
            guard let text = item.url, let url = URL(string: text),
                  let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
                  let title = item.title.map({ boundedSearchText($0, limit: AppWebSearchLimits.maximumTitleCharacters) }),
                  !title.isEmpty else { return nil }
            return AppWebResult(title: title, url: url,
                                excerpt: boundedSearchText(item.content ?? "",
                                                           limit: AppWebSearchLimits.maximumExcerptCharacters))
        }
        guard !results.isEmpty else { throw AppWebSearchError.noResults(.tavily) }
        return Array(results.prefix(count))
    }
}
