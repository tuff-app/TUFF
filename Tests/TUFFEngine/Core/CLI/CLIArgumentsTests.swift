import Testing
import Foundation
import TUFFEngine
@testable import TUFFCLICore

@Suite struct CLIArgumentsTests {
    @Test func topKHelpExplainsHowToDisableTruncation() {
        #expect(Args.usage.contains("0 = off with --top-p 1 or --temperature 0"))
    }
    @Test func automaticMemoryFlagAndOlderJSONPayloadsRemainCompatible() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", "--memory-auto"])
        #expect(arguments.automaticMemory == true)
        var older = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(arguments)) as? [String: Any])
        older.removeValue(forKey: "automaticMemory")
        let decoded = try JSONDecoder().decode(Args.self, from: JSONSerialization.data(withJSONObject: older))
        #expect(decoded.automaticMemory == nil)
        #expect(decoded.model == arguments.model)
    }
    @Test func defaultsUseProductionGenerationValues() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.model == "m.gturbo")
        #expect(arguments.prompt == "hi")
        #expect(arguments.systemPrompt == nil)
        #expect(arguments.messagesFile == nil)
        #expect(arguments.maxNew == 1_024)
        // 8K as of 2026-08-17, so an image and its prompt fit without anyone
        // passing --max-context.
        #expect(arguments.maxContext == 8192)
        #expect(arguments.thinking == .off)
        #expect(arguments.reasoningEffort == nil)
        #expect(arguments.temperature == 0.2)
        #expect(arguments.topK == 64)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1)
        #expect(arguments.seed == nil)
        #expect(arguments.stops.isEmpty)
        #expect(!arguments.quiet)

        let runtime = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        #expect(runtime == RuntimeConfiguration.production)
    }

    @Test func generationOptionsParseAndStopsRepeat() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--max-new", "32", "--max-context", "512",
            "--temperature", "0", "--top-k", "40", "--top-p", "0.95",
            "--repetition-penalty", "1.1", "--seed", "42",
            "--stop", "A", "--stop", "B", "--quiet",
        ])
        #expect(arguments.maxNew == 32)
        #expect(arguments.maxContext == 512)
        #expect(arguments.temperature == 0)
        #expect(arguments.topK == 40)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1.1)
        #expect(arguments.seed == 42)
        #expect(arguments.stops == ["A", "B"])
        #expect(arguments.quiet)
    }

    @Test func topKZeroRequiresTopPToBeDisabled() throws {
        let disabled = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == nil)
        #expect(disabled.topP == 1)

        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "0",
            ])
        }
    }

    @Test func topKAboveKernelLimitRejected() {
        #expect(throws: ArgsError.invalidValue(flag: "--top-k", value: "257")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "257",
            ])
        }
    }

    @Test func helpListsExactlyThePublicOptions() {
        let expected: Set<String> = [
            "--model", "--prompt", "--messages-file", "--max-new", "--max-context", "--memory-auto",
            "--system-prompt",
            "--thinking", "--reasoning",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--seed", "--stop", "--quiet", "--expert-cache-slots",
            "--expert-cache-policy", "--prefill", "--prefill-chunk-tokens",
            "--prefill-chunk-max",
            "--rdadvise", "--help",
            "--chat-prompt", "--image", "--vision-pack", "--vision-residency",
            "--speculative", "--speculative-draft-tokens",
            "--dense-residency", "--dense-weight-budget-mb",
        ]
        let words = Args.usage.split { $0.isWhitespace || $0 == "(" || $0 == ")" }
        let options = Set(words.map(String.init).filter { $0.hasPrefix("--") })
        #expect(options == expected)
    }

    @Test func cacheSlotHelpMatchesSharedRuntimeChoices() throws {
        let choices = RuntimeConfiguration.allowedExpertCacheSlots.map(String.init).joined(separator: ", ")
        #expect(Args.usage.contains("Expert-cache slots: \(choices) (default 16)."))
        for count in RuntimeConfiguration.allowedExpertCacheSlots {
            let args = try Args.parse(["--model", "m.gturbo", "--prompt", "hi",
                "--prefill", "off", "--expert-cache-slots", String(count)])
            #expect(args.expertCacheSlots == count)
        }
    }

    @Test func autoChunkCapDefaultsToTheHistoricalSize() throws {
        let direct = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--prefill-chunk-tokens", "auto"])
        #expect(direct.prefillChunkTokensAuto)
        #expect(direct.prefillChunkMax == 256)

        let wrapped = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--prefill-chunk-tokens", "auto", "--prefill-chunk-max", "2048"])
        #expect(wrapped.prefillChunkMax == 2_048)

        let fixed = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--prefill-chunk-tokens", "1024"])
        #expect(fixed.prefillChunkTokens == 1_024)
        #expect(try fixed.resolvedRuntimeConfiguration(forceLogitsHead: false)
            .prefillChunkTokens == 1_024)
    }

    @Test func speculativeOptionsAreOptInAndBounded() throws {
        let defaults = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(defaults.speculativeMode == .off)
        #expect(defaults.speculativeDraftTokens == 4)

        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--speculative", "greedy", "--speculative-draft-tokens", "8",
        ])
        #expect(arguments.speculativeMode == .greedy)
        #expect(arguments.speculativeDraftTokens == 8)

        let automatic = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--speculative", "auto",
        ])
        #expect(automatic.speculativeMode == .auto)

        for value in ["0", "9"] {
            #expect(throws: ArgsError.invalidValue(
                flag: "--speculative-draft-tokens", value: value)) {
                _ = try Args.parse([
                    "--model", "m.gturbo", "--prompt", "hi",
                    "--speculative-draft-tokens", value,
                ])
            }
        }
    }

    @Test func systemPromptParsesOnlyWithAChatPrompt() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--chat-prompt", "hi",
            "--system-prompt", "Be helpful.",
        ])
        #expect(arguments.systemPrompt == "Be helpful.")

        #expect(throws: ArgsError.requiredMissing("--chat-prompt")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--system-prompt", "Be helpful.",
            ])
        }
    }

    @Test func thinkingParsesForChatAndIsRejectedForRawCompletion() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--chat-prompt", "hi",
            "--thinking", "on",
        ])
        #expect(arguments.thinking == .on)

        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--thinking")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--thinking", "on",
            ])
        }
        #expect(throws: ArgsError.invalidValue(flag: "--thinking", value: "medium")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--chat-prompt", "hi",
                "--thinking", "medium",
            ])
        }
    }

    @Test func gptOssReasoningParsesForChatAndRejectsConflicts() throws {
        for effort in GPTOSSReasoningEffort.allCases {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--chat-prompt", "hi",
                "--reasoning", effort.rawValue,
            ])
            #expect(arguments.reasoningEffort == effort)
        }
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--reasoning")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--reasoning", "low",
            ])
        }
        #expect(throws: ArgsError.mutuallyExclusive("--thinking", "--reasoning")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--chat-prompt", "hi",
                "--thinking", "on", "--reasoning", "high",
            ])
        }
        #expect(throws: ArgsError.invalidValue(flag: "--reasoning", value: "on")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--chat-prompt", "hi",
                "--reasoning", "on",
            ])
        }
    }

    @Test func runtimeOptionsReachTypedConfiguration() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--expert-cache-slots", "24",
            "--expert-cache-policy", "lru",
            "--prefill", "off",
            "--prefill-chunk-tokens", "64",
            "--rdadvise", "adaptive",
        ])

        #expect(arguments.expertCacheSlots == 24)
        #expect(arguments.expertCachePolicy == .lru)
        #expect(arguments.prefillPolicy == .off)
        #expect(arguments.prefillChunkTokens == 64)
        #expect(arguments.rdadvisePolicy == .adaptive)

        let runtime = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: true)
        #expect(runtime.expertCacheSlots == 24)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.prefillPolicy == .off)
        #expect(runtime.prefillChunkTokens == 64)
        #expect(runtime.rdadvisePolicy == .adaptive)
        #expect(runtime.headPath == .logits)
    }

    @Test func everySupportedRuntimeOptionParses() throws {
        for value in RuntimeConfiguration.allowedExpertCacheSlots {
            let prefill = value < RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
                ? "off" : "on"
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-cache-slots", "\(value)",
                "--prefill", prefill,
            ])
            #expect(arguments.expertCacheSlots == value)
        }
        for value in RuntimeConfiguration.allowedPrefillChunkTokens {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--prefill-chunk-tokens", "\(value)",
            ])
            #expect(arguments.prefillChunkTokens == value)
        }
        for value in ["lfu", "lru"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-cache-policy", value,
            ])
            #expect(arguments.expertCachePolicy.rawValue == value)
        }
        for value in ["off", "default", "bounded", "adaptive"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--rdadvise", value,
            ])
            #expect(arguments.rdadvisePolicy.rawValue == value)
        }
        #expect(try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--prefill", "on",
        ]).prefillPolicy == .chunked)
        #expect(try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--prefill", "off",
        ]).prefillPolicy == .off)
    }

    @Test func unsupportedRuntimeOptionValuesAreRejected() throws {
        let invalidValues = [
            ("--expert-cache-slots", "7"),
            ("--expert-cache-policy", "fifo"),
            ("--prefill", "yes"),
            ("--prefill-chunk-tokens", "4096"),
            ("--prefill-chunk-max", "4096"),
            ("--rdadvise", "automatic"),
        ]
        for (flag, value) in invalidValues {
            #expect(throws: ArgsError.invalidValue(flag: flag, value: value)) {
                _ = try Args.parse([
                    "--model", "m.gturbo", "--prompt", "hi", flag, value,
                ])
            }
        }

        // Parsing doesn't know the model, so a small cache with chunked
        // prefill passes it; the run rejects it unless the model is GPT-OSS,
        // which prefills in chunks with any cache size. `tuff prompt` used to
        // fail on GPT-OSS's own defaults here.
        let small = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--expert-cache-slots", "4", "--prefill", "on",
        ])
        #expect(throws: ArgsError.invalidValue(
            flag: "--expert-cache-slots", value: "4 requires --prefill off")) {
            _ = try small.resolvedRuntimeConfiguration(forceLogitsHead: false)
        }
        let gptOSS = try small.resolvedRuntimeConfiguration(
            forceLogitsHead: false, chunkedPrefillNeedsSlotFloor: false)
        #expect(gptOSS.prefillPolicy == .chunked && gptOSS.expertCacheSlots == 4)
    }

    @Test func programmaticArgumentsCannotReachRuntimePreconditions() {
        var arguments = Args(model: "m.gturbo", prompt: "hi")
        arguments.expertCacheSlots = 7
        #expect(throws: ArgsError.invalidValue(
            flag: "--expert-cache-slots", value: "7")) {
            _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        }

        arguments.expertCacheSlots = RuntimeConfiguration.production.expertCacheSlots
        arguments.prefillChunkTokens = 4_096
        #expect(throws: ArgsError.invalidValue(
            flag: "--prefill-chunk-tokens", value: "4096")) {
            _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        }

        arguments.prefillChunkTokens = RuntimeConfiguration.production.prefillChunkTokens
        arguments.expertCacheSlots = 8
        #expect(throws: ArgsError.invalidValue(
            flag: "--expert-cache-slots", value: "8 requires --prefill off")) {
            _ = try arguments.resolvedRuntimeConfiguration(forceLogitsHead: false)
        }
    }

    @Test func unsupportedSelectorsAreRejected() {
        for flag in ["--runtime-profile", "--experiment-id", "-h"] {
            #expect(throws: ArgsError.unknownFlag(flag)) {
                _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", flag])
            }
        }
    }

    @Test func modelAndPromptAreRequired() {
        #expect(throws: ArgsError.requiredMissing("--model")) {
            _ = try Args.parse(["--prompt", "hi"])
        }
        #expect(throws: ArgsError.modeMissing) {
            _ = try Args.parse(["--model", "m.gturbo"])
        }
    }

    @Test func messagesFileSelectsChatMode() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--messages-file", "chat.json",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "chat.json")
    }

    @Test func promptAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--messages-file", "chat.json",
            ])
        }
    }
    @Test func imageChatOptionsPreserveOrderAndResidency() throws {
        let a = try Args.parse([
            "--model", "m.gturbo",
            "--image", "first.png",
            "--chat-prompt", "compare",
            "--image", "second.jpg",
            "--vision-pack", "vision.gturbo",
            "--vision-residency", "keep-ready",
        ])
        #expect(a.chatPrompt == "compare")
        #expect(a.images == ["first.png", "second.jpg"])
        #expect(a.visionPack == "vision.gturbo")
        #expect(a.visionResidency == .keepReady)
    }

    @Test func imagesRejectRawAndMessagesModesButNotACount() {
        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "raw", "--image", "x.png",
            ])
        }
        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--messages-file", "m.json",
                "--image", "x.png",
            ])
        }
        // No fixed image count: what bounds a request is the context, checked
        // against the rendered prompt once token costs are known. The server
        // moved to a context-derived budget and the CLI follows.
        let many = try? Args.parse([
            "--model", "m.gturbo", "--chat-prompt", "x",
            "--image", "1", "--image", "2", "--image", "3",
            "--image", "4", "--image", "5", "--image", "6",
        ])
        #expect(many?.images.count == 6)
    }

    @Test func mutual_exclusion_chatPromptVsMessagesFile() {
        do {
            _ = try Args.parse([
                "--model", "m.gturbo",
                "--chat-prompt", "X",
                "--messages-file", "Y.json",
            ])
            Issue.record("expected ArgsError.mutuallyExclusive")
        } catch let e as ArgsError {
            #expect(e == .mutuallyExclusive("--chat-prompt", "--messages-file"))
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    @Test func mutual_exclusion_promptVsChatPrompt() {
        do {
            _ = try Args.parse([
                "--model", "m.gturbo",
                "--prompt", "X",
                "--chat-prompt", "Y",
            ])
            Issue.record("expected ArgsError.mutuallyExclusive")
        } catch let e as ArgsError {
            #expect(e == .mutuallyExclusive("--prompt", "--chat-prompt"))
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

}

extension CLIArgumentsTests {
    @Test func appleRejectsEngineAndUnsupportedSamplingOptions() throws {
        let base = ["--model", "apple.on-device", "--chat-prompt", "Hello"]
        #expect(try Args.parse(base).model == "apple.on-device")
        #expect(try Args.parse(["--model", "apple", "--prompt", "--top-k"]).prompt == "--top-k")
        for options in [["--max-context", "8192"], ["--top-k", "64"], ["--top-p", "0.9"], ["--seed", "2"],
                        ["--prefill-chunk-tokens", "auto"], ["--memory-auto"],
                        ["--speculative", "off"]] {
            #expect(throws: ArgsError.self) { try Args.parse(base + options) }
        }
    }
}
