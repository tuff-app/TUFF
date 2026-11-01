import Testing
import TUFFAppCore
@testable import TUFFMac

@Suite struct ToolActivityPresentationTests {
    private func entry(_ id: String, _ name: String, _ state: AppToolActivity.State = .succeeded,
                       result: String = "") -> ToolActivityEntry {
        ToolActivityEntry(id: id, name: name, detail: "", summary: "", resultText: result, state: state)
    }

    @MainActor
    @Test func multipleWorkspaceCallsAreSummarizedWithoutMislabeling() {
        let view = ToolActivityView(entries: [entry("1", "list_files"), entry("2", "edit_file"), entry("3", "read_file")],
                                    sources: [], isRunning: false, reduceTransparency: true)
        #expect(view.headline == "Worked with files")
        let mixed = ToolActivityView(entries: [entry("1", "edit_file"), entry("2", "run_command", .failed)],
                                     sources: [], isRunning: false, reduceTransparency: true)
        #expect(mixed.headline == "Worked with files and ran commands, with errors")
        #expect(entry("2", "run_command", .failed).title == "Command failed")
    }

    @MainActor
    @Test func anotherChatDoesNotHideAnAvailableSendButton() {
        #expect(GenerateControl.mode(selectedRunning: false, loading: false, anyRunning: true, canSubmit: true) == .send)
        #expect(GenerateControl.mode(selectedRunning: true, loading: false, anyRunning: true, canSubmit: true) == .running)
        #expect(GenerateControl.mode(selectedRunning: false, loading: true, anyRunning: true, canSubmit: false) == .loading)
        #expect(GenerateControl.mode(selectedRunning: false, loading: false, anyRunning: true, canSubmit: false) == .otherChatRunning)
    }

    @Test func completedCommandKeepsItsActualOutput() {
        let call = AppToolCall(id: "one", name: "run_command", arguments: .object(["command": .string("exit 7")]))
        let result = AppToolResult(callID: call.id, name: call.name, status: .failed,
                                   modelText: "marker\\n[Exit status 7.]", summary: "Exit status 7")
        let round = AppToolRound(calls: [call], results: [result])
        let shown = ToolActivityEntry.entries(from: [round])
        #expect(shown.count == 1)
        #expect(shown.first?.state == .failed)
        #expect(shown.first?.resultText == result.modelText)
    }
}
