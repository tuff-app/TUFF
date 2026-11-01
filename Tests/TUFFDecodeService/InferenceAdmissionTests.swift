import Testing
@testable import TUFFEngineAdapter

@Suite struct InferenceAdmissionTests {
    @Test func lifecycleAndLegacyExcludeSharedSessions() async {
        let admission = InferenceAdmission()
        #expect(admission.beginLegacy())
        #expect(!admission.beginShared())
        admission.endLegacy()
        #expect(admission.beginShared())
        #expect(admission.beginShared())
        #expect(!admission.beginLegacy())
        await admission.beginLifecycle()
        #expect(!admission.beginShared())
        #expect(!admission.beginLegacy())
        admission.endShared()
        admission.endShared()
        admission.endLifecycle()
        #expect(admission.beginLegacy())
        admission.endLegacy()
    }
}
