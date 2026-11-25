import Foundation
import Testing
@testable import TUFFAppCore

/// A transport that answers from fixtures and records what it was asked.
final class FixtureHTTPTransport: AppHTTPTransport, @unchecked Sendable {
    typealias Handler = @Sendable (AppHTTPRequest) async throws -> AppHTTPResponse
    private let lock = NSLock()
    private var recorded: [AppHTTPRequest] = []
    private let handler: Handler

    init(_ handler: @escaping Handler) { self.handler = handler }

    var requests: [AppHTTPRequest] { lock.withLock { recorded } }

    func perform(_ request: AppHTTPRequest) async throws -> AppHTTPResponse {
        lock.withLock { recorded.append(request) }
        return try await handler(request)
    }

}

extension AppHTTPResponse {
    static func html(_ body: String, status: Int = 200,
                     url: URL = URL(string: "https://html.duckduckgo.com/html/")!) -> AppHTTPResponse {
        AppHTTPResponse(finalURL: url, statusCode: status,
                        contentType: "text/html; charset=utf-8", body: Data(body.utf8))
    }

    static func json(_ object: Any, status: Int = 200) -> AppHTTPResponse {
        AppHTTPResponse(finalURL: URL(string: "https://api.example.test/")!, statusCode: status,
                        contentType: "application/json",
                        body: try! JSONSerialization.data(withJSONObject: object))
    }
}

/// Synthetic pages in the shape of DuckDuckGo's HTML endpoint. Titles,
/// links and snippets are invented; only the markup structure is copied.
enum DuckDuckGoFixtures {
    static func result(_ title: String, _ href: String, _ snippet: String, ad: Bool = false) -> String {
        """
        <div class="result results_links results_links_deep web-result\(ad ? " result--ad" : "")">
          <div class="links_main links_deep result__body">
            <h2 class="result__title"><a rel="nofollow" class="result__a" href="\(href)">\(title)</a></h2>
            <div class="result__extras"><div class="result__extras__url">
              <span class="result__icon"><a rel="nofollow" href="\(href)"><img class="result__icon__img" width="16" height="16" alt="" src="//external-content.example/icon.ico" /></a></span>
              <a class="result__url" href="\(href)">example</a>
            </div></div>
            <a class="result__snippet" href="\(href)">\(snippet)</a>
          </div>
        </div>
        """
    }

    static func page(_ results: [String]) -> String {
        """
        <!DOCTYPE html><html><head><title>query at DuckDuckGo</title>
        <script>var x = "<div class='result'>not a result</div>";</script></head>
        <body><div class="serp__results"><div id="links" class="results">
        \(results.joined(separator: "\n"))
        </div></div></body></html>
        """
    }

    static let ordinary = page([
        result("Sponsored thing", "https://ads.example/buy", "Buy now", ad: true),
        result("Swift actors &amp; isolation", "https://docs.example.org/actors", "Actors <b>isolate</b> mutable state."),
        result("Redirected result", "//duckduckgo.com/l/?uddg=https%3A%2F%2Fblog.example.net%2Fpost%3Fid%3D4&rut=abc", "A post about actors."),
        result("Second docs page", "https://docs.example.org/reentrancy", "Reentrancy &#x27;explained&#x27;."),
    ])
}

/// Synthetic pages in the shape of DuckDuckGo's lite endpoint. The content is
/// invented; only the markup structure is copied.
enum DuckDuckGoLiteFixtures {
    static func result(_ number: Int, _ title: String, _ href: String, _ snippet: String,
                       ad: Bool = false) -> String {
        let row = ad ? #"<tr class="result-sponsored">"# : "<tr>"
        return """
        \(row)<td valign="top">\(number).&nbsp;</td><td>
          <a rel="nofollow" href="\(href)" class='result-link'>\(title)</a>
          \(ad ? #"(Sponsored link - <a href="https://duckduckgo.com/duckduckgo-help-pages/ads/" class="result-link">more info</a>)"# : "")
        </td></tr>
        \(row)<td>&nbsp;&nbsp;&nbsp;</td><td class='result-snippet'>\(snippet)</td></tr>
        \(row)<td>&nbsp;</td><td><span class='link-text'>example</span></td></tr>
        """
    }

    static let ordinary = """
        <html><body><form action="/lite/" method="post"><input class='query' name="q" value="x"></form>
        <table border="0">
        \(result(1, "Sponsored thing", "https://duckduckgo.com/y.js?ad_domain=ads.example", "Buy now", ad: true))
        \(result(1, "Swift actors &amp; isolation", "https://docs.example.org/actors", "Actors <b>isolate</b> mutable state."))
        \(result(2, "Second docs page", "https://docs.example.org/reentrancy", "Reentrancy &#x27;explained&#x27;."))
        </table></body></html>
        """
}

@Suite struct WebSearchProviderTests {
    @Test func tlsFailureNamesTheProviderAndExplainsRecovery() async {
        let transport = FixtureHTTPTransport { _ in throw AppHTTPError.connectionFailed(code: 35) }
        await #expect(throws: AppWebSearchError.transport(.duckDuckGo, .connectionFailed(code: 35))) {
            _ = try await DuckDuckGoSearchProvider(transport: transport).search("example", count: 1)
        }
        let message = AppWebSearchError.transport(.duckDuckGo, .connectionFailed(code: 35)).description
        #expect(message.contains("DuckDuckGo"))
        #expect(message.contains("TLS handshake"))
        #expect(message.contains("another search provider in Settings"))
        #expect(transport.requests.count == 1)
    }

    @Test func duckDuckGoParsesResultsSkipsAdsAndResolvesRedirects() async throws {
        let transport = FixtureHTTPTransport { _ in .html(DuckDuckGoFixtures.ordinary) }
        let response = try await DuckDuckGoSearchProvider(transport: transport)
            .search("swift actors", count: 5)
        #expect(response.results.map(\.url.absoluteString) == [
            "https://docs.example.org/actors",
            "https://blog.example.net/post?id=4",
            "https://docs.example.org/reentrancy",
        ])
        #expect(response.results[0].title == "Swift actors & isolation")
        #expect(response.results[0].excerpt == "Actors isolate mutable state.")
        #expect(response.results[2].excerpt == "Reentrancy 'explained'.")
        let request = try #require(transport.requests.first)
        #expect(request.method == "POST")
        #expect(String(data: request.body ?? Data(), encoding: .utf8) == "q=swift%20actors")
        #expect(request.maximumBytes == AppWebSearchLimits.maximumResponseBytes)
    }

    @Test func duckDuckGoKeepsSitesWhoseNamesOnlyEndLikeItsOwn() {
        #expect(DuckDuckGoSearchProvider.resolve("https://notduckduckgo.com/page")?.absoluteString
            == "https://notduckduckgo.com/page")
        #expect(DuckDuckGoSearchProvider.resolve(
            "//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.example.org%2Factors")?.absoluteString
            == "https://docs.example.org/actors")
        #expect(DuckDuckGoSearchProvider.resolve("https://html.DuckDuckGo.com/y.js?ad=1") == nil)
        #expect(DuckDuckGoSearchProvider.resolve("https://duckduckgo.com/about") == nil)
    }

    @Test(arguments: [(2 * 1_048_576, "2 MB"), (1_048_576, "1 MB"),
                      (50_000, "48 KB"), (1_024, "1 KB")])
    func sizeLimitsReadTheWayTheyAreDocumented(limit: Int, text: String) {
        #expect(AppHTTPError.oversized(limit: limit).description
            == "The response was larger than \(text), so reading stopped.")
    }

    @Test func duckDuckGoHonoursTheResultCount() async throws {
        let transport = FixtureHTTPTransport { _ in .html(DuckDuckGoFixtures.ordinary) }
        let response = try await DuckDuckGoSearchProvider(transport: transport).search("x", count: 1)
        #expect(response.results.count == 1)
    }

    @Test(arguments: [
        (DuckDuckGoFixtures.page([]) + #"<div class="no-results">No results.</div>"#,
         200, AppWebSearchError.noResults(.duckDuckGo)),
        ("<html><body><div class=\"serp\">A redesigned page</div></body></html>",
         200, AppWebSearchError.markupChanged(.duckDuckGo)),
        ("<html><body><form id=\"challenge-form\">Unfortunately, bots use DuckDuckGo too.</form></body></html>",
         200, AppWebSearchError.challenge(.duckDuckGo)),
        ("<html><body><div class=\"anomaly-modal\"></div></body></html>",
         200, AppWebSearchError.challenge(.duckDuckGo)),
        ("", 202, AppWebSearchError.blocked(.duckDuckGo, status: 202)),
        ("", 403, AppWebSearchError.blocked(.duckDuckGo, status: 403)),
        ("", 429, AppWebSearchError.rateLimited(.duckDuckGo)),
        ("", 500, AppWebSearchError.http(.duckDuckGo, status: 500)),
    ])
    func duckDuckGoReportsWhatHappened(body: String, status: Int,
                                       expected: AppWebSearchError) async throws {
        let transport = FixtureHTTPTransport { _ in .html(body, status: status) }
        await #expect(throws: expected) {
            _ = try await DuckDuckGoSearchProvider(transport: transport).search("x", count: 5)
        }
    }

    @Test(arguments: [
        ("<html><body><div class=\"anomaly-modal\"></div></body></html>", 200),
        ("", 202),
        ("", 403),
    ])
    func duckDuckGoTriesItsLitePageWhenTheHTMLPageIsChallenged(body: String, status: Int) async throws {
        let transport = FixtureHTTPTransport { request in
            request.url == DuckDuckGoSearchProvider.liteEndpoint
                ? .html(DuckDuckGoLiteFixtures.ordinary, url: request.url)
                : .html(body, status: status)
        }
        let response = try await DuckDuckGoSearchProvider(transport: transport)
            .search("swift actors", count: 5)
        #expect(response.provider == .duckDuckGo)
        #expect(response.results.map(\.url.absoluteString) == [
            "https://docs.example.org/actors", "https://docs.example.org/reentrancy"])
        #expect(response.results.first?.title == "Swift actors & isolation")
        #expect(response.results.first?.excerpt == "Actors isolate mutable state.")
        #expect(response.results.last?.excerpt == "Reentrancy 'explained'.")
        // The same query, once to each DuckDuckGo page, and nowhere else.
        #expect(transport.requests.map(\.url) == [
            DuckDuckGoSearchProvider.endpoint, DuckDuckGoSearchProvider.liteEndpoint])
        #expect(Set(transport.requests.map(\.body)).count == 1)
    }

    @Test(arguments: [
        ("", 429),
        ("", 500),
        ("<html><body><div class=\"serp\">A redesigned page</div></body></html>", 200),
        (DuckDuckGoFixtures.page([]) + #"<div class="no-results">No results.</div>"#, 200),
    ])
    func duckDuckGoUsesTheLitePageOnlyForChallenges(body: String, status: Int) async {
        let transport = FixtureHTTPTransport { _ in .html(body, status: status) }
        await #expect(throws: AppWebSearchError.self) {
            _ = try await DuckDuckGoSearchProvider(transport: transport).search("x", count: 5)
        }
        #expect(transport.requests.map(\.url) == [DuckDuckGoSearchProvider.endpoint])
    }

    @Test func aLitePageThatCannotBeReachedReportsTheOriginalBlock() async {
        let transport = FixtureHTTPTransport { request in
            if request.url == DuckDuckGoSearchProvider.liteEndpoint {
                throw AppHTTPError.sandbox("The sandbox does not reach this site.")
            }
            return .html("<html><body><div class=\"anomaly-modal\"></div></body></html>")
        }
        await #expect(throws: AppWebSearchError.challenge(.duckDuckGo)) {
            _ = try await DuckDuckGoSearchProvider(transport: transport).search("x", count: 5)
        }
        #expect(transport.requests.count == 2)
    }

    @Test func duckDuckGoLitePageHonoursTheResultCount() {
        let results = DuckDuckGoSearchProvider.parseLiteResults(DuckDuckGoLiteFixtures.ordinary, limit: 1)
        #expect(results.map(\.url.absoluteString) == ["https://docs.example.org/actors"])
    }

    @Test(arguments: [AppHTTPError.timedOut, .oversized(limit: 1_048_576),
                      .network("offline"), .tooManyRedirects])
    func transportFailuresNameTheProvider(error: AppHTTPError) async throws {
        let transport = FixtureHTTPTransport { _ in throw error }
        await #expect(throws: AppWebSearchError.transport(.duckDuckGo, error)) {
            _ = try await DuckDuckGoSearchProvider(transport: transport).search("x", count: 5)
        }
    }

    @Test func braveSendsTheKeyInAHeaderAndParsesResults() async throws {
        let transport = FixtureHTTPTransport { _ in .json([
            "web": ["results": [
                ["title": "Swift <strong>actors</strong>", "url": "https://swift.example/actors",
                 "description": "About <strong>actors</strong>."],
                ["title": "Not web", "url": "ftp://files.example/x", "description": "skipped"],
            ]],
        ]) }
        let response = try await BraveSearchProvider(transport: transport, key: "brave-test-key")
            .search("swift actors", count: 3)
        #expect(response.results.count == 1)
        #expect(response.results[0].title == "Swift actors")
        #expect(response.results[0].excerpt == "About actors.")
        let request = try #require(transport.requests.first)
        #expect(request.headers["X-Subscription-Token"] == "brave-test-key")
        #expect(!request.url.absoluteString.contains("brave-test-key"))
        #expect(request.url.query?.contains("count=3") == true)
    }

    @Test(arguments: [(401, AppWebSearchError.invalidKey(.brave)),
                      (403, AppWebSearchError.invalidKey(.brave)),
                      (429, AppWebSearchError.rateLimited(.brave)),
                      (500, AppWebSearchError.http(.brave, status: 500))])
    func braveErrors(status: Int, expected: AppWebSearchError) async throws {
        let transport = FixtureHTTPTransport { _ in .json(["error": "x"], status: status) }
        await #expect(throws: expected) {
            _ = try await BraveSearchProvider(transport: transport, key: "k").search("x", count: 5)
        }
    }

    @Test func aKeyedProviderWithoutAKeyNeverSendsARequest() async throws {
        let transport = FixtureHTTPTransport { _ in .json([:]) }
        await #expect(throws: AppWebSearchError.missingKey(.brave)) {
            _ = try await BraveSearchProvider(transport: transport, key: "").search("x", count: 5)
        }
        await #expect(throws: AppWebSearchError.missingKey(.tavily)) {
            _ = try await TavilySearchProvider(transport: transport, key: "").search("x", count: 5)
        }
        #expect(transport.requests.isEmpty)
    }

    @Test func tavilyPostsJSONWithABearerKey() async throws {
        let transport = FixtureHTTPTransport { _ in .json([
            "results": [["title": "Actors", "url": "https://tavily.example/a", "content": "Isolation."]],
        ]) }
        let response = try await TavilySearchProvider(transport: transport, key: "tvly-test")
            .search("actors", count: 2)
        #expect(response.results == [AppWebResult(
            title: "Actors", url: URL(string: "https://tavily.example/a")!, excerpt: "Isolation.")])
        let request = try #require(transport.requests.first)
        #expect(request.method == "POST")
        #expect(request.headers["Authorization"] == "Bearer tvly-test")
        let body = try JSONSerialization.jsonObject(with: try #require(request.body)) as? [String: Any]
        #expect(body?["query"] as? String == "actors")
        #expect(body?["max_results"] as? Int == 2)
        #expect(body?["api_key"] == nil)
    }

    @Test(arguments: [(401, AppWebSearchError.invalidKey(.tavily)),
                      (429, AppWebSearchError.rateLimited(.tavily)),
                      (432, AppWebSearchError.rateLimited(.tavily))])
    func tavilyErrors(status: Int, expected: AppWebSearchError) async throws {
        let transport = FixtureHTTPTransport { _ in .json(["detail": "x"], status: status) }
        await #expect(throws: expected) {
            _ = try await TavilySearchProvider(transport: transport, key: "k").search("x", count: 5)
        }
    }

    @Test func malformedJSONIsAChangedFormatNotAnEmptyResult() async throws {
        let transport = FixtureHTTPTransport { _ in
            AppHTTPResponse(finalURL: URL(string: "https://x.test")!, statusCode: 200,
                            contentType: "application/json", body: Data("<html>".utf8))
        }
        await #expect(throws: AppWebSearchError.markupChanged(.brave)) {
            _ = try await BraveSearchProvider(transport: transport, key: "k").search("x", count: 5)
        }
        await #expect(throws: AppWebSearchError.markupChanged(.tavily)) {
            _ = try await TavilySearchProvider(transport: transport, key: "k").search("x", count: 5)
        }
    }
}

@Suite struct WebPageReaderTests {
    @Test func readsArticleTextWithoutChrome() async throws {
        let html = """
        <html><head><title>Actors &amp; you</title><style>p{color:red}</style></head>
        <body><nav>Home | About</nav><header>Site header</header>
        <article><h1>Actors</h1><p>Actors protect their state by allowing one task at a time.</p>
        <script>alert('x')</script><p>Calls from outside are asynchronous, so each one may suspend.
        That keeps data races out of ordinary Swift code without locks.</p>
        <aside>Related links</aside></article><footer>Copyright</footer></body></html>
        """
        let transport = FixtureHTTPTransport { request in
            AppHTTPResponse(finalURL: request.url, statusCode: 200,
                            contentType: "text/html", body: Data(html.utf8))
        }
        let page = try await AppWebPageReader(transport: transport)
            .read(URL(string: "https://docs.example.org/actors")!)
        #expect(page.title == "Actors & you")
        #expect(page.text.hasPrefix("Actors\nActors protect their state"))
        #expect(!page.text.contains("Home | About"))
        #expect(!page.text.contains("alert"))
        #expect(!page.text.contains("Related links"))
        #expect(!page.text.contains("Copyright"))
    }

    @Test func plainTextAndErrorsAreReported() throws {
        let url = URL(string: "https://x.test/a")!
        let text = try AppWebPageReader.page(from: AppHTTPResponse(
            finalURL: url, statusCode: 200, contentType: "text/plain", body: Data("hello".utf8)))
        #expect(text.text == "hello")
        #expect(throws: AppWebPageError.http(404)) {
            _ = try AppWebPageReader.page(from: AppHTTPResponse(
                finalURL: url, statusCode: 404, contentType: "text/html", body: Data()))
        }
        #expect(throws: AppWebPageError.unsupportedContent("image/png")) {
            _ = try AppWebPageReader.page(from: AppHTTPResponse(
                finalURL: url, statusCode: 200, contentType: "image/png", body: Data([1, 2])))
        }
        #expect(throws: AppWebPageError.empty) {
            _ = try AppWebPageReader.page(from: AppHTTPResponse(
                finalURL: url, statusCode: 200, contentType: "text/html",
                body: Data("<html><body><script>x</script></body></html>".utf8)))
        }
    }

    @Test func declaredCharsetsDecode() {
        let latin1 = AppHTTPResponse(
            finalURL: URL(string: "https://x.test")!, statusCode: 200,
            contentType: "text/html; charset=ISO-8859-1", body: Data([0x63, 0x61, 0x66, 0xE9]))
        #expect(latin1.text == "café")
    }
}

@Suite struct NetworkPolicyTests {
    @Test(arguments: ["localhost", "127.0.0.1", "10.1.2.3", "192.168.1.1", "172.20.0.5",
                      "169.254.10.1", "100.64.0.1", "::1", "[::1]", "fe80::1", "fd00::5",
                      "printer.local", "router", "0.0.0.0", "::ffff:192.168.0.1", "224.0.0.1",
                      "fec0::1", "64:ff9b::7f00:1", "2002:c0a8:1::1", "2001:db8::1"])
    func privateHostsAreRefused(host: String) {
        #expect(!AppNetworkPolicy.isPublicHost(host))
    }

    @Test(arguments: ["example.org", "93.184.216.34", "2606:2800:220:1::1", "docs.swift.org"])
    func publicHostsAreAllowed(host: String) {
        #expect(AppNetworkPolicy.isPublicHost(host))
    }

    @Test func theTransportRefusesSchemesAndPrivateHostsBeforeConnecting() async {
        let transport = URLSessionHTTPTransport()
        await #expect(throws: AppHTTPError.unsupportedScheme) {
            _ = try await transport.perform(AppHTTPRequest(
                url: URL(string: "file:///etc/passwd")!, maximumBytes: 10, timeoutSeconds: 1))
        }
        await #expect(throws: AppHTTPError.privateAddress("127.0.0.1")) {
            _ = try await transport.perform(AppHTTPRequest(
                url: URL(string: "http://127.0.0.1:8080/")!, maximumBytes: 10, timeoutSeconds: 1))
        }
    }
}
