import Darwin
import Foundation
import TUFFAppCore
import TUFFAppServer
import TUFFDecodeProtocol
import TUFFServerCore

// A minimal serving client that runs as its own process, so a test can kill it
// the way quitting the GUI or stopping the HTTP adapter would, and observe that
// the shared host keeps the other surface's work running.
//
//   TUFFServingProbe <gui|api> --support <dir> --bundle <id> --host <TUFFServer>
//       --model <install directory name> --model-id <API model ID>
//       --prompt <text> --tokens <count>
//
// It prints `READY` once its model is resident, `FIRST_TOKEN` when the first
// token arrives, then `DONE <count>`; any failure prints `ERROR <reason>`.

setvbuf(stdout, nil, _IOLBF, 0)

func say(_ line: String) {
    print(line)
    fflush(stdout)
}

func fail(_ reason: String) -> Never {
    say("ERROR \(reason)")
    exit(1)
}

var rest = Array(CommandLine.arguments.dropFirst())
guard !rest.isEmpty else { fail("usage: TUFFServingProbe <gui|api> --key value ...") }
let role = rest.removeFirst()
let arguments: [String: String] = {
    var parsed: [String: String] = [:]
    while rest.count >= 2 {
        let key = rest.removeFirst()
        parsed[key] = rest.removeFirst()
    }
    return parsed
}()

func required(_ key: String) -> String {
    guard let value = arguments[key] else { fail("missing \(key)") }
    return value
}

do {
    let support = URL(fileURLWithPath: required("--support"), isDirectory: true)
    let profile = try ServingHostProfile(
        applicationSupportURL: support, bundleIdentifier: required("--bundle"))
    let host = URL(fileURLWithPath: required("--host"))
    let models = support.appendingPathComponent("TUFF/Models", isDirectory: true)
    let directory = models.appendingPathComponent(required("--model"), isDirectory: true)
    let prompt = required("--prompt")
    let tokens = Int(required("--tokens")) ?? 64
    var options = AppRuntimeOptions()
    options.modelVerification = .fullSha256
    let clientID = UUID()
    let generation = AppGenerationRequest(
        modelDirectory: directory, prompt: prompt, maxNewTokens: tokens, maxContextTokens: 1024,
        temperature: 0, topK: nil, topP: nil, runtimeOptions: options)

    switch role {
    case "gui":
        let client = ServingHostInferenceClient(modelsRoot: models) {
            try ServingHostRuntime.connectOrLaunch(
                profile: profile, clientID: clientID, role: .gui, hostExecutable: host).connection
        }
        try await client.ensureLoaded(
            modelDirectory: directory, maxContextTokens: 1024, options: options,
            forceLogitsHead: false) { _ in }
        say("READY")
        var count = 0
        for try await event in client.scopedSession(UUID()).generate(generation) {
            if case .token = event {
                count += 1
                if count == 1 { say("FIRST_TOKEN") }
            }
        }
        say("DONE \(count)")
    case "api":
        let attachment = try ServingHostRuntime.connectOrLaunch(
            profile: profile, clientID: clientID, role: .api, hostExecutable: host)
        let ipc = PrivateServingIPCClient(connection: attachment.connection)
        let preparation = try ServingModelPreparation(
            modelID: required("--model-id"), maxContextTokens: 1024,
            runtimeOptions: DecodeServiceInferenceClient.decodeRuntimeOptions(options),
            forceLogitsHead: false)
        for try await _ in ipc.events(for: .prepare(preparation)) {}
        say("READY")
        let requestID = UUID()
        let request = try ServingGenerationRequest(
            modelID: required("--model-id"), sessionID: UUID(), requestID: requestID,
            payload: DecodeGenerationTranslation.decodeRequest(generation, generationID: requestID))
        var count = 0
        for try await event in ipc.events(for: .start(request)) {
            if case .generation(_, let decoded) = event, !decoded.textDelta.isEmpty {
                count += 1
                if count == 1 { say("FIRST_TOKEN") }
            }
        }
        say("DONE \(count)")
    default:
        fail("unknown role \(role)")
    }
} catch {
    fail("\(error)")
}
