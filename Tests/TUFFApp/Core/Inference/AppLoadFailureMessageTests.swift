import Foundation
import Testing
import TUFFRuntimeManagement
@testable import TUFFAppCore

@Suite struct AppLoadFailureMessageTests {
    @Test func engineDownloadFailuresAreSaidInWords() {
        let refused = AppLoadFailureMessage.text(for: InferenceEngineDownloadError.unexpectedHTTPStatus(404))
        #expect(refused.contains("could not download the engine"))
        #expect(refused.contains("404"))
        #expect(!refused.contains("unexpectedHTTPStatus"))

        let offline = AppLoadFailureMessage.text(for: URLError(.notConnectedToInternet))
        #expect(offline.contains("could not reach the internet"))
        #expect(!offline.contains("NSURLErrorDomain"))
    }

    @Test func otherErrorsKeepTheirOwnDescription() {
        struct Odd: Error, CustomStringConvertible { var description: String { "odd failure" } }
        #expect(AppLoadFailureMessage.text(for: Odd()) == "odd failure")
    }
}
