import Foundation
import Testing

@testable import TUFFAppCore

@Suite struct FakeInferenceClientTests {

  @MainActor
  @Test func installedModelSupportsLoadAndGenerationWithTestClient() async throws {
    let directory = try makeCompleteModelInstall("fake-runtime")
    defer { try? FileManager.default.removeItem(at: directory) }
    let model = makeAppModel(
      modelDirectory: directory,
      client: FakeInferenceClient(eventDelay: .zero),
      installer: MockModelInstallerClient())

    #expect(model.isModelInstalled)
    #expect(model.loadState == .notLoaded)
    model.loadModel()
    try await waitForFakeClient { model.loadState.isReady }
    model.promptText = "Hello from the test"
    model.run()
    try await waitForFakeClient { model.diagnostics != nil && !model.isRunning }

    #expect(model.isModelInstalled)
    #expect(model.outputText.contains("Simulated response"))
    #expect(model.outputText.contains("Hello from the test"))
    #expect(model.diagnostics?.generatedTokens ?? 0 > 0)
  }

  @Test func fakeInferenceCancellationEmitsCancelledEvent() async throws {
    let client = FakeInferenceClient(eventDelay: .milliseconds(5))
    let request = AppGenerationRequest(
      modelDirectory: URL(fileURLWithPath: "/missing/fake-model"),
      prompt: "Cancel this simulated response")
    try await client.ensureLoaded(
      modelDirectory: request.modelDirectory,
      maxContextTokens: request.maxContextTokens,
      options: request.runtimeOptions,
      forceLogitsHead: !request.isPureGreedy,
      onState: { _ in })
    var sawCancellation = false

    do {
      for try await event in client.generate(request) {
        if case .token = event {
          client.cancel()
        } else if case .cancelled = event {
          sawCancellation = true
        }
      }
    } catch AppInferenceError.cancelled {
    }

    #expect(sawCancellation)
  }

  @Test func fakeInferenceRejectsConcurrentGeneration() async throws {
    let client = FakeInferenceClient(eventDelay: .milliseconds(20))
    let request = AppGenerationRequest(
      modelDirectory: URL(fileURLWithPath: "/missing/fake-model"),
      prompt: "First")
    try await client.ensureLoaded(
      modelDirectory: request.modelDirectory,
      maxContextTokens: request.maxContextTokens,
      options: request.runtimeOptions,
      forceLogitsHead: !request.isPureGreedy,
      onState: { _ in })
    let first = client.generate(request)
    let second = client.generate(request)
    var failure: AppInferenceError?
    do {
      for try await event in second {
        if case .failed(let error, _) = event { failure = error }
      }
    } catch let error as AppInferenceError {
      failure = failure ?? error
    }
    #expect(failure == .generationInFlight)
    client.cancel()
    _ = first
  }

  @Test func scopedSessionAllowsImmediateNextTurnAfterTerminalStream() async throws {
    let session = ImmediateSessionInferenceClient().scopedSession(UUID())
    let request = AppGenerationRequest(
      modelDirectory: URL(fileURLWithPath: "/missing/fake-model"), prompt: "one turn")

    for _ in 0..<20 {
      var sawInFlight = false
      for try await event in session.generate(request) {
        if case .failed(.generationInFlight, _) = event { sawInFlight = true }
      }
      #expect(!sawInFlight)
    }
  }

  @Test func cancelledTurnKeepsTheSessionOccupiedUntilTheProducerHasDrained() async throws {
    let underlying = ControlledSessionInferenceClient()
    let session = underlying.scopedSession(UUID())
    let request = AppGenerationRequest(
      modelDirectory: URL(fileURLWithPath: "/missing/fake-model"), prompt: "turn")

    let consumer = Task { for try await _ in session.generate(request) {} }
    #expect(await underlying.waitForStart(count: 1))
    consumer.cancel()
    _ = await consumer.result
    #expect(await underlying.waitForCancel())

    // The helper has not reported its terminal event, so old GPU work may
    // still be running: the same chat must not start another request yet.
    var rejected = false
    do {
      for try await event in session.generate(request) {
        if case .failed(.generationInFlight, _) = event { rejected = true }
      }
    } catch AppInferenceError.generationInFlight {
      rejected = true
    }
    #expect(rejected)
    #expect(underlying.startCount == 1)

    // Once the producer drains, the very next turn is admitted.
    underlying.finishAll()
    var admitted = false
    for _ in 0..<200 {
      let before = underlying.startCount
      let attempt = Task {
        for try await _ in session.generate(request) {}
      }
      try await Task.sleep(for: .milliseconds(10))
      if underlying.startCount > before {
        admitted = true
        underlying.finishAll()
        _ = await attempt.result
        break
      }
      _ = await attempt.result
    }
    #expect(admitted)
  }

  @Test func failedTurnAlsoReleasesTheSessionBeforeTheStreamEnds() async throws {
    let underlying = ControlledSessionInferenceClient(failing: true)
    let session = underlying.scopedSession(UUID())
    let request = AppGenerationRequest(
      modelDirectory: URL(fileURLWithPath: "/missing/fake-model"), prompt: "turn")
    for _ in 0..<10 {
      var failed = false
      do { for try await _ in session.generate(request) {} } catch { failed = true }
      #expect(failed)
    }
    #expect(underlying.startCount == 10)
  }

  @MainActor
  private func waitForFakeClient(_ predicate: @escaping @MainActor () -> Bool) async throws {
    for _ in 0..<200 {
      if predicate() { return }
      try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("timed out waiting for fake client state")
  }

}

private final class ImmediateSessionInferenceClient: AppSessionInferenceClient, Sendable {
  func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
    stream()
  }

  func generate(_ request: AppGenerationRequest, sessionID: UUID,
                requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error> {
    stream()
  }

  func cancel() {}
  func cancel(requestID: UUID) {}
  func endSession(_ sessionID: UUID, requestID: UUID) async throws {}

  private func stream() -> AsyncThrowingStream<AppInferenceEvent, Error> {
    AsyncThrowingStream { continuation in
      continuation.yield(.token(AppTokenEvent(index: 0, textDelta: "ok", elapsedDecodeSeconds: 0)))
      continuation.finish()
    }
  }
}

private final class ControlledSessionInferenceClient: AppSessionInferenceClient, @unchecked Sendable {
  private let lock = NSLock()
  private let failing: Bool
  private var continuations: [AsyncThrowingStream<AppInferenceEvent, Error>.Continuation] = []
  private var cancels = 0
  private var starts = 0

  init(failing: Bool = false) { self.failing = failing }

  var startCount: Int { lock.withLock { starts } }

  func waitForStart(count: Int) async -> Bool {
    for _ in 0..<400 {
      if startCount >= count { return true }
      try? await Task.sleep(for: .milliseconds(5))
    }
    return false
  }

  func waitForCancel() async -> Bool {
    for _ in 0..<400 {
      if lock.withLock({ cancels }) > 0 { return true }
      try? await Task.sleep(for: .milliseconds(5))
    }
    return false
  }

  func finishAll() {
    let pending = lock.withLock { () -> [AsyncThrowingStream<AppInferenceEvent, Error>.Continuation] in
      defer { continuations.removeAll() }
      return continuations
    }
    for continuation in pending { continuation.finish() }
  }

  func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
    stream()
  }

  func generate(_ request: AppGenerationRequest, sessionID: UUID,
                requestID: UUID) -> AsyncThrowingStream<AppInferenceEvent, Error> {
    stream()
  }

  func cancel() {}
  func cancel(requestID: UUID) { lock.withLock { cancels += 1 } }
  func endSession(_ sessionID: UUID, requestID: UUID) async throws {}

  private func stream() -> AsyncThrowingStream<AppInferenceEvent, Error> {
    lock.withLock { starts += 1 }
    return AsyncThrowingStream { continuation in
      if failing {
        continuation.finish(throwing: AppInferenceError.unknown("scripted failure"))
      } else {
        lock.withLock { continuations.append(continuation) }
      }
    }
  }
}
