import Foundation
import Testing
@testable import TUFFEngine

/// Opt-in: native chat rendering of the repacked Bonsai packs against the
/// publisher's Jinja template (rendered by `bonsai_template_fixture.py`).
/// `TUFF_BONSAI_TEMPLATE_FIXTURE` is that JSON; `TUFF_BONSAI_2B_PACK` and
/// `TUFF_BONSAI_27B_PACK` are the `.gturbo` directories.
@Suite struct BonsaiTemplateTests {
    private struct Case: Decodable {
        struct Message: Decodable { let role: String; let content: String }
        let messages: [Message]
        let thinking: Bool
        let rendered: String
        let ids: [Int]
    }

    private func check(size: String, pack: String, variant: ModelVariant, fixture: [String: Case]) async throws {
        let tokenizer = try await GFTokenizer.load(forModelDirectory: URL(fileURLWithPath: pack))
        var mismatches: [String] = []
        for (name, item) in fixture.sorted(by: { $0.key < $1.key }) {
            let messages = item.messages.map {
                GFTokenizer.Message(role: GFTokenizer.Role(rawValue: $0.role)!, content: $0.content)
            }
            let rendered = try tokenizer.applyChatTemplate(
                messages, modelVariant: variant, reasoning: item.thinking ? .on : .off)
            let ids = tokenizer.encode(rendered, addBOS: false).map(Int.init)
            if ids != item.ids {
                mismatches.append(name)
                print("Bonsai \(size) template MISMATCH \(name)\n native:    \(rendered.debugDescription)\n publisher: \(item.rendered.debugDescription)")
            }
        }
        print("Bonsai \(size) template cases: \(fixture.count), mismatches: \(mismatches)")
        #expect(mismatches.isEmpty, "\(size) renders differ from the publisher template: \(mismatches)")
    }

    @Test func nativeRenderingMatchesPublisherTemplate() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let fixturePath = env["TUFF_BONSAI_TEMPLATE_FIXTURE"] else { return }
        let all = try JSONDecoder().decode([String: [String: Case]].self,
            from: Data(contentsOf: URL(fileURLWithPath: fixturePath)))
        if let pack = env["TUFF_BONSAI_2B_PACK"], let fixture = all["2b"] {
            try await check(size: "2B", pack: pack, variant: .bonsai2_2B, fixture: fixture)
        }
        if let pack = env["TUFF_BONSAI_27B_PACK"], let fixture = all["27b"] {
            try await check(size: "27B", pack: pack, variant: .bonsai2_27B, fixture: fixture)
        }
    }
}
