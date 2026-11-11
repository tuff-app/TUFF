import Foundation
import Testing
@testable import TUFFAppCore

@Suite struct AppleModelAvailabilityTests {
    @Test func onlyAvailablePermitsGeneration() {
        for state in AppleModelAvailability.allCases {
            #expect(state.canGenerate == (state == .available))
        }
        #expect(AppleModelAvailability.notSupported.label == "Not Supported on This Mac")
        #expect(AppleModelAvailability.updateRequired.label == "Update macOS to Use")
        #expect(AppleModelAvailability.intelligenceDisabled.label == "Set Up Apple Intelligence")
    }

    @Test func actionsDoNotPromiseHardwareCanBeUpdated() {
        #expect(AppleModelAvailability.available.action == .select)
        #expect(AppleModelAvailability.notSupported.action == .none)
        #expect(AppleModelAvailability.updateRequired.action == .openSystemSettings)
        #expect(AppleModelAvailability.intelligenceDisabled.action == .openSystemSettings)
        #expect(AppleModelAvailability.preparing.action == .refresh)
        #expect(AppleModelAvailability.unknown.action == .refresh)
    }

    @Test func realAvailableAdapter() async throws {
        guard ProcessInfo.processInfo.environment["TUFF_TEST_APPLE_MODEL"] == "1",
              AppleModelAvailability.current().canGenerate else { return }
        let client = AppleInferenceClient()
        var response = ""
        var finished = false
        let request = AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/unused"),
            prompt: "Reply with just hello.", maxNewTokens: 16, temperature: 0)
        for try await event in client.generate(request) {
            switch event {
            case .token(let token): response += token.textDelta
            case .finished: finished = true
            default: break
            }
        }
        #expect(!response.isEmpty)
        #expect(finished)
    }

    @Test func unavailableStatesRejectGenerationWithoutHostReadiness() async {
        for state in AppleModelAvailability.allCases where !state.canGenerate {
            let client = AppleInferenceClient(availability: { state })
            do {
                for try await _ in client.generate(.init(modelDirectory: URL(fileURLWithPath: "/unused"), prompt: "Hello")) {}
                Issue.record("Unavailable state accepted generation: \(state)")
            } catch let error as AppInferenceError {
                #expect(error == .invalidRequest(state.label))
            } catch { Issue.record("Unexpected error: \(error)") }
        }
    }

    @Test func rejectsUnsupportedContinuationBeforeSDKInvocation() async {
        let client = AppleInferenceClient(availability: { .available })
        do {
            let request = AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/unused"),
                prompt: "Hello", assistantPrefix: "Continue this")
            for try await _ in client.generate(request) {}
            Issue.record("Unsupported Continue accepted")
        } catch let error as AppInferenceError {
            #expect(error == .invalidRequest("Apple supports text chat here; images, tools and Continue are unavailable."))
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test func hostUnavailableRejectsGeneration() async {
        guard !AppleModelAvailability.current().canGenerate else { return }
        let client = AppleInferenceClient()
        do {
            for try await _ in client.generate(.init(modelDirectory: URL(fileURLWithPath: "/unused"), prompt: "Hello")) {}
            Issue.record("Unavailable Apple model accepted generation")
        } catch { }
    }
}
