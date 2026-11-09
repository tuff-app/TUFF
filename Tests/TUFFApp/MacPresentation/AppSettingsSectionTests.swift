import Testing
@testable import TUFFMacPresentation

@Suite struct AppSettingsSectionTests {
    @Test func settingsSectionsStayCompactAndDistinct() {
        let sections = AppSettingsSection.allCases

        #expect(sections == [.general, .models, .search, .advanced])
        #expect(sections.map(\.title) == ["General", "Generation", "Tools", "Advanced"])
        #expect(Set(sections.map(\.systemImage)).count == sections.count)
    }
}
