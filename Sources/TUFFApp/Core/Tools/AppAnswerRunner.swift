import Foundation
import Synchronization

/// What the answer loop reports to the app as it runs.
public enum AppAnswerEvent: Equatable, Sendable {
    /// Progress, tokens and reasoning from the generation in progress.
    case inference(AppInferenceEvent)
    /// A generation is starting; `round` tool rounds precede it.
    case generationStarted(round: Int)
    case toolActivity(AppToolActivity)
    /// A round finished: its calls, results and every source so far.
    case toolRoundFinished(AppToolRound, sources: [AppSource])
    /// The last generation could not be read as a tool call and is being
    /// generated once more.
    case retryingMalformedToolCall(String)
    case finished(AppDiagnostics)
    case cancelled(AppDiagnostics?)
    case failed(AppInferenceError, partial: AppDiagnostics?)
}

/// Stops an answer wherever it is: the generation through the inference
/// client, and tool work through task cancellation, which ends network
/// requests and file searches.
public final class AppAnswerCancellation: Sendable {
    private struct State {
        var cancelled = false
        var toolTask: Task<AppToolRound, Never>?
    }
    private let state = Mutex(State())

    public init() {}

    public var isCancelled: Bool { state.withLock { $0.cancelled } }

    public func cancel() {
        let task = state.withLock { state -> Task<AppToolRound, Never>? in
            state.cancelled = true
            return state.toolTask
        }
        task?.cancel()
    }

    func register(_ task: Task<AppToolRound, Never>) -> Bool {
        state.withLock { state in
            guard !state.cancelled else { return false }
            state.toolTask = task
            return true
        }
    }

    func clear() { state.withLock { $0.toolTask = nil } }
}

/// One answer: generate, run any tool calls, give the model the results, and
/// generate again, until it answers or a limit is reached.
///
/// Bounds: `maximumToolRounds` rounds of calls, after which calls are
/// refused and the model is told to answer; at most two further generations
/// after that; one regeneration after a call the decoder cannot read.
public struct AppAnswerRunner: Sendable {
    public let client: any AppInferenceClient
    public let toolbox: AppToolbox
    public let limits: AppToolLimits

    public init(client: any AppInferenceClient, toolbox: AppToolbox,
                limits: AppToolLimits = .standard) {
        self.client = client
        self.toolbox = toolbox
        self.limits = limits
    }

    public func run(_ request: AppGenerationRequest,
                    capabilities: AppChatCapabilities,
                    firstSourceID: Int,
                    userText: String,
                    hasLocalContext: Bool = false,
                    cancellation: AppAnswerCancellation,
                    sink: @escaping @Sendable (AppAnswerEvent) async -> Void) async {
        let session = AppToolAnswerSession(capabilities: capabilities, limits: limits,
                                           toolbox: toolbox, firstSourceID: firstSourceID,
                                           userText: userText,
                                           hasLocalContext: hasLocalContext
                                            || !request.imageAttachments.isEmpty
                                            || request.history.contains(where: Self.turnContainsLocalContext))
        var current = request
        var malformedRetries = 0
        var generations = 0
        var toolSeconds = 0.0
        let maximumGenerations = limits.maximumToolRounds + 2 + limits.maximumMalformedRetries

        func finalize(_ diagnostics: AppDiagnostics?) -> AppDiagnostics? {
            guard var diagnostics else { return nil }
            diagnostics.toolRounds = session.rounds.count
            diagnostics.toolSeconds = session.rounds.isEmpty ? nil : toolSeconds
            return diagnostics
        }

        while true {
            if cancellation.isCancelled {
                await sink(.cancelled(nil))
                return
            }
            generations += 1
            await sink(.generationStarted(round: session.rounds.count))

            var calls: [AppToolCall] = []
            var visible = ""
            var thinking = ""
            var terminal: AppInferenceEvent?
            do {
                for try await event in client.generate(current) {
                    switch event {
                    case .token(let token):
                        visible += token.textDelta
                        await sink(.inference(event))
                    case .thinking(let token):
                        thinking += token.textDelta
                        await sink(.inference(event))
                    case .toolCalls(let parsed):
                        calls = parsed
                    case .finished, .cancelled, .failed:
                        terminal = event
                    case .prefillProgress, .memorySample:
                        await sink(.inference(event))
                    }
                }
            } catch let error as AppInferenceError {
                if terminal == nil { terminal = .failed(error, partial: nil) }
            } catch {
                if terminal == nil { terminal = .failed(.unknown("\(error)"), partial: nil) }
            }

            switch terminal {
            case .finished(let diagnostics)? where !calls.isEmpty:
                guard !cancellation.isCancelled else {
                    await sink(.cancelled(finalize(diagnostics)))
                    return
                }
                guard generations < maximumGenerations else {
                    await sink(.failed(
                        .invalidRequest("The model kept calling tools after its limit, so TUFF stopped. The results so far are kept."),
                        partial: finalize(diagnostics)))
                    return
                }
                let budget = characterBudget(request: current, diagnostics: diagnostics)
                let started = ContinuousClock.now
                let session = session
                let task = Task<AppToolRound, Never> {
                    await session.execute(
                        calls: calls, thinking: thinking, content: visible,
                        characterBudget: budget,
                        onActivity: { activity in await sink(.toolActivity(activity)) })
                }
                guard cancellation.register(task) else {
                    task.cancel()
                    _ = await task.value
                    await sink(.cancelled(finalize(diagnostics)))
                    return
                }
                let round = await task.value
                cancellation.clear()
                let elapsed = ContinuousClock.now - started
                toolSeconds += Double(elapsed.components.seconds)
                    + Double(elapsed.components.attoseconds) / 1e18
                await sink(.toolRoundFinished(round, sources: session.sources))
                if cancellation.isCancelled
                    || round.results.contains(where: { $0.status == .cancelled }) {
                    await sink(.cancelled(finalize(diagnostics)))
                    return
                }
                current.currentRounds = session.rounds
            case .finished(let diagnostics)?:
                await sink(.finished(finalize(diagnostics) ?? diagnostics))
                return
            case .failed(.malformedToolCall(let detail), let partial)?
                where malformedRetries < limits.maximumMalformedRetries
                    && generations < maximumGenerations && !cancellation.isCancelled:
                malformedRetries += 1
                _ = partial
                // Repeating the identical prompt with the same seed repeats
                // a malformed call. Give this one corrective retry actual
                // feedback, without echoing untrusted decoder text.
                current.systemPrompt += "\nYour previous response contained an invalid tool call. Use the declared tool format, tool names and required argument types, and finish every argument. If you cannot make a valid call, answer directly without calling tools."
                await sink(.retryingMalformedToolCall(detail))
            case .failed(let error, let partial)?:
                await sink(.failed(error, partial: finalize(partial)))
                return
            case .cancelled(let diagnostics)?:
                await sink(.cancelled(finalize(diagnostics)))
                return
            case nil, .token?, .thinking?, .toolCalls?, .prefillProgress?, .memorySample?:
                if cancellation.isCancelled {
                    await sink(.cancelled(nil))
                } else {
                    await sink(.failed(.unknown("The generation ended without a result."),
                                       partial: nil))
                }
                return
            }
        }
    }

    /// Result text the context can still take, from the token counts the
    /// last generation reported, at a conservative three characters a token,
    /// with room kept for the answer. The tool session applies the per-call
    /// minimum when dividing this total across results.
    func characterBudget(request: AppGenerationRequest, diagnostics: AppDiagnostics) -> Int {
        let used = (diagnostics.promptTokenCount ?? 0) + diagnostics.generatedTokens
        let free = request.maxContextTokens - used - limits.answerReserveTokens
        return max(0, free) * 3
    }

    private static func turnContainsLocalContext(_ turn: AppChatTurn) -> Bool {
        if !turn.documents.isEmpty || !turn.images.isEmpty
            || turn.sources.contains(where: { $0.kind == .file }) {
            return true
        }
        return turn.toolRounds.contains { round in
            round.results.contains { result in
                if result.name == AppToolName.searchFiles.rawValue {
                    return result.status == .succeeded
                }
                guard result.status != .refused,
                      let name = AppToolName(rawValue: result.name) else { return false }
                switch name {
                case .listFiles, .readFile, .writeFile, .editFile, .runCommand: return true
                case .webSearch, .readWebpage, .searchFiles: return false
                }
            }
        }
    }
}
