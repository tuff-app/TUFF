import Foundation
import Synchronization
import Testing
import TUFFAppCore
import TUFFDecodeProtocol
import TUFFServerCore
@testable import TUFFAppServer

@Suite(.serialized) struct ServingHostInferenceClientTests {
    private struct Record: Sendable {
        var resident: ServingModelPreparation?
        var preparations: [ServingModelPreparation] = []
        var starts = 0
        var holdPrepare = false
        var preparing = false
        var flood = false
        var cancelled: Set<UUID> = []
    }

    @Test func cancellationDuringReprepareNeverStartsAndResidentConfigurationIsAdopted() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let profile = try ServingHostProfile(applicationSupportURL: root, bundleIdentifier: "test.gui.client")
        let record = Mutex(Record())
        let server = PrivateServingIPCServer(profile: profile, token: try profile.token(), buildIdentity: "test", statusSnapshot: {
            record.withLock { value in
                let models = value.resident.map { [ServingLoadedModel(modelID: $0.modelID, maxContextTokens: $0.maxContextTokens, runtimeOptions: $0.runtimeOptions, forceLogitsHead: $0.forceLogitsHead)] } ?? []
                return .init(loadedModelIDs: models.map(\.modelID), loadedModels: models, activeRequests: 0, openSessions: 0)
            }
        }) { context, command, emit in
            let clientID = context.clientID
            _ = clientID
            switch command {
            case .prepare(let preparation):
                context.markAdmitted()
                record.withLock { $0.preparing = true; $0.preparations.append(preparation) }
                while record.withLock({ $0.holdPrepare }) { try? await Task.sleep(for: .milliseconds(1)) }
                record.withLock { $0.resident = preparation; $0.preparing = false }
                emit(.lifecycle(operationID: preparation.operationID, modelID: preparation.modelID, state: .ready(loadSeconds: 0)))
            case .start(let request):
                record.withLock { $0.starts += 1 }
                let identity = ServingRequestIdentity(clientID: clientID, request: request)
                emit(.queue(identity: identity, state: .admitted, position: nil))
                let flood = record.withLock { $0.flood }
                let count = flood ? PrivateServingIPCClient.streamCapacity * 3 : 1
                for index in 0..<count {
                    emit(.generation(identity: identity, event: DecodeServiceEvent(
                        kind: .snapshot, generationID: request.requestID,
                        sequence: UInt64(index + 1), textDelta: "text", thinkingDelta: "x")))
                    if flood && index % 32 == 0 { try? await Task.sleep(for: .milliseconds(1)) }
                }
                emit(.generation(identity: identity, event: DecodeServiceEvent(kind: .finished, generationID: request.requestID)))
            case .cancel(_, let requestID):
                _ = record.withLock { $0.cancelled.insert(requestID) }
            default: break
            }
        }
        try server.start()
        defer { server.shutdown() }
        let connections = Mutex(0)
        let client = ServingHostInferenceClient(modelsRoot: root) {
            connections.withLock { $0 += 1 }
            return try PrivateServingIPCConnection.connect(profile: profile, clientID: try profile.clientID(for: "gui"), role: .gui)
        }
        async let firstStatus = client.hostStatus()
        async let secondStatus = client.hostStatus()
        _ = try await (firstStatus, secondStatus)
        #expect(connections.withLock { $0 } == 1)
        let directory = root.appendingPathComponent("gemma4-e2b.gturbo")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let options = AppRuntimeOptions()
        try await client.ensureLoaded(modelDirectory: directory, maxContextTokens: 1024, options: options, forceLogitsHead: false) { _ in }
        let original = try #require(record.withLock { $0.resident })
        let adopted = try ServingModelPreparation(modelID: original.modelID, maxContextTokens: 2048, runtimeOptions: original.runtimeOptions, forceLogitsHead: original.forceLogitsHead)
        record.withLock { $0.resident = adopted; $0.holdPrepare = true }
        let request = AppGenerationRequest(modelDirectory: directory, prompt: "hello", maxNewTokens: 8, maxContextTokens: 1024, temperature: 0, topK: nil, topP: nil, runtimeOptions: options)
        let id = UUID()
        let stream = client.generate(request, sessionID: UUID(), requestID: id)
        let consumer = Task<String?, Never> {
            do {
                for try await _ in stream {}
                return nil
            } catch {
                return "\(error)"
            }
        }
        var waited = 0
        while !record.withLock({ $0.preparing }), waited < 5_000 {
            try await Task.sleep(for: .milliseconds(1))
            waited += 1
        }
        #expect(record.withLock { $0.preparing }, "the re-prepare never reached the host")
        client.cancel(requestID: id)
        record.withLock { $0.holdPrepare = false }
        let consumerFailure = await consumer.value
        #expect(consumerFailure == nil)
        #expect(record.withLock { $0.starts } == 0)
        #expect(record.withLock { $0.preparations.last?.maxContextTokens } == 2048)
        // Pooled requests deliver text without retaining it in the legacy mailbox.
        for try await _ in client.generate(request, sessionID: UUID(), requestID: UUID()) {}
        #expect(client.generationTranscriptMailbox.completeText.isEmpty)
        #expect(client.generationTranscriptMailbox.drain().pendingText.isEmpty)

        // Leave the GUI stream unread until it has cancelled the overflowing
        // request, then drain the buffered events and observe its failure.
        record.withLock { $0.flood = true }
        let floodedID = UUID()
        let flooded = client.generate(request, sessionID: UUID(), requestID: floodedID)
        for _ in 0..<10_000 {
            if record.withLock({ $0.cancelled.contains(floodedID) }) { break }
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(record.withLock { $0.cancelled.contains(floodedID) })
        var failure: (any Error)?
        do { for try await _ in flooded {} } catch { failure = error }
        #expect(failure is PrivateServingIPCClientError)
        #expect(client.generationTranscriptMailbox.completeText.isEmpty)
    }
}
