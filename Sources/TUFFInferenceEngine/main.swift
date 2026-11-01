import Darwin
import Foundation
import TUFFCLIArguments
import TUFFCLIEngine
import TUFFDecodeProtocol
import TUFFDecodeServiceCore
import TUFFInferenceTypes
import TUFFRuntimeManagement

func argument(after flag: String) -> String? {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else { return nil }
    return arguments[index + 1]
}

let lease: InferenceEngineProcessLease?
do {
    lease = try argument(after: "--engine-lock").map {
        try InferenceEngineProcessLease(lockFileURL: URL(fileURLWithPath: $0))
    } ?? Bundle.main.executableURL.flatMap { try InferenceEngineProcessLease.forInstalledExecutable($0) }
} catch {
    FileHandle.standardError.write(Data("could not hold engine package lease: \(error)\n".utf8))
    exit(1)
}

if CommandLine.arguments.contains("--cli-json") {
    do {
        let args = try DecodeFrameCodec.read(Args.self, from: .standardInput)
        guard (1...262_144).contains(args.maxContext) else {
            throw ArgsError.invalidValue(flag: "--max-context", value: String(args.maxContext))
        }
        _ = try args.resolvedRuntimeConfiguration(forceLogitsHead: true)
        try GenerationConfig(maxNewTokens: args.maxNew, temperature: args.temperature,
            topK: args.topK, topP: args.topP, repetitionPenalty: args.repetitionPenalty,
            seed: args.seed, stopStrings: args.stops,
            speculative: .init(mode: args.speculativeMode, draftTokens: args.speculativeDraftTokens)).validate()
        let status = CLIEngineCommand.drive(args)
        withExtendedLifetime(lease) {}
        exit(status)
    } catch {
        FileHandle.standardError.write(Data("error: \(error)\n".utf8))
        exit(1)
    }
}

Task {
    defer { withExtendedLifetime(lease) {} }
    await TUFFDecodeServiceMain.main()
    exit(0)
}
dispatchMain()
