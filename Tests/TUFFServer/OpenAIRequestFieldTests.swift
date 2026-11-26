import Foundation
import Testing
@testable import TUFFEngine
@testable import TUFFServerCore

/// Top-level request keys are swept before the typed decode, so a key the
/// server does not declare is refused instead of silently ignored.
@Suite("OpenAI request fields")
struct OpenAIRequestFieldTests {
    private static func decode(_ extra: String) throws -> OpenAIChatRequest {
        let separator = extra.isEmpty ? "" : ","
        return try JSONDecoder().decode(OpenAIChatRequest.self, from: Data("""
        {"model":"m","messages":[{"role":"user","content":"x"}]\(separator)\(extra)}
        """.utf8))
    }

    private static func rejection(_ extra: String) throws
        -> (message: String, param: String?, code: String) {
        do {
            _ = try decode(extra)
        } catch ServerRequestError.invalid(let message, let param, let code) {
            return (message, param, code)
        }
        Issue.record("expected \(extra) to be refused")
        return ("", nil, "")
    }

    @Test func aMisspelledOptionIsRefusedWithTheFieldItMeant() throws {
        let refused = try Self.rejection(#""max_token":64"#)
        #expect(refused.code == "unknown_parameter")
        #expect(refused.param == "max_token")
        #expect(refused.message == #"unrecognized request field "max_token"; did you mean max_tokens?"#)
    }

    @Test func aKeyNearTwoDeclaredFieldsGetsNoGuess() {
        // "top_x" is one edit from both top_p and top_k.
        #expect(OpenAIChatRequest.closestDeclaredKey(to: "top_x") == nil)
        #expect(OpenAIChatRequest.closestDeclaredKey(to: "temprature") == "temperature")
        #expect(OpenAIChatRequest.closestDeclaredKey(to: "completely_unrelated") == nil)
    }

    @Test func otherLocalServersFieldsPointAtTheTUFFEquivalent() throws {
        let predict = try Self.rejection(#""num_predict":32"#)
        #expect(predict.message.hasSuffix("; use max_tokens"))
    }

    @Test func severalUnknownKeysAreSortedAndBounded() throws {
        let keys = (0..<11).map { #""zz\#($0)":1"# }.reversed().joined(separator: ",")
        let refused = try Self.rejection(keys)
        #expect(refused.param == "zz0")
        #expect(refused.message.hasPrefix(#"unrecognized request fields "zz0", "zz1", "zz10""#))
        #expect(refused.message.hasSuffix(", and 3 more"))
        #expect(!refused.message.contains("did you mean"))
    }

    @Test func aHostileKeyIsEchoedOnlyInBoundedForm() throws {
        let long = String(repeating: "k", count: 10_000)
        let refused = try Self.rejection(#""\#(long)":1"#)
        #expect((refused.param?.utf8.count ?? .max) <= 67)
        #expect(refused.message.utf8.count < 200)

        // One Character carrying a thousand combining marks.
        let marks = "a" + String(repeating: "\u{0301}", count: 1_000)
        let combining = try Self.rejection(#""\#(marks)":1"#)
        #expect((combining.param?.utf8.count ?? .max) <= 67)
    }

    @Test func aNullUnknownKeyAsksForNothingAndIsAccepted() throws {
        _ = try Self.decode(#""logit_bias":null,"something_new":null"#)
    }

    @Test func bookkeepingKeysAreTolerated() throws {
        _ = try Self.decode(#"""
        "user":"u","store":false,"metadata":{"a":"b"},"service_tier":"auto",
        "prompt_cache_key":"k","safety_identifier":"s"
        """#)
    }

    @Test func realOpenAIParametersAreUnsupportedNotUnknown() throws {
        let bias = try Self.rejection(#""logit_bias":{"50256":-100}"#)
        #expect(bias.code == "unsupported_value")
        #expect(bias.param == "logit_bias")

        let functions = try Self.rejection(#""functions":[]"#)
        #expect(functions.code == "unsupported_value")
        #expect(functions.message == "legacy functions are not supported; use tools")
    }

    /// llama.cpp and LM Studio clients send -1 for "no seed" and "no limit".
    /// Both were refused, the seed as "malformed JSON request".
    @Test func minusOneSeedAndLimitMeanTheSameAsLeavingThemOut() throws {
        let request = try Self.decode(#""seed":-1,"max_tokens":-1"#)
        #expect(request.seed == nil)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        #expect(validated.generationConfig.seed == nil)
        #expect(validated.maximumCompletionTokens == 4096)
        #expect(validated.generationConfig.maxNewTokens == 4096)
        let completion = try OpenAIRequestValidator.validate(
            try Self.decode(#""max_completion_tokens":-1"#), modelID: "m")
        #expect(completion.maximumCompletionTokens == 4096)

        // Every other limit below one is still refused.
        for limit in ["0", "-2"] {
            #expect(throws: ServerRequestError.self) {
                try OpenAIRequestValidator.validate(
                    try Self.decode(#""max_tokens":\#(limit)"#), modelID: "m")
            }
        }
    }

    /// OpenAI seeds are signed 64-bit integers, so a negative seed other than
    /// -1 is a fixed seed and the whole unsigned range still decodes.
    @Test func negativeAndLargeSeedsStayFixedSeeds() throws {
        #expect(try Self.decode(#""seed":-7"#).seed == UInt64(bitPattern: -7))
        #expect(try Self.decode(#""seed":-9223372036854775808"#).seed == UInt64(1) << 63)
        #expect(try Self.decode(#""seed":0"#).seed == 0)
        #expect(try Self.decode(#""seed":18446744073709551615"#).seed == .max)
        #expect(try Self.decode(#""seed":null"#).seed == nil)

        // A request the server re-encodes keeps the same seed.
        let request = try Self.decode(#""seed":-7"#)
        let again = try JSONDecoder().decode(
            OpenAIChatRequest.self, from: JSONEncoder().encode(request))
        #expect(again.seed == request.seed)
    }

    /// The Apple model takes no seed. -1 asks for none, so it is accepted
    /// there; a fixed seed, negative or not, is still refused.
    @Test func appleAcceptsMinusOneSeedAndRefusesFixedSeeds() throws {
        func validate(_ extra: String) throws -> ValidatedChatRequest {
            try OpenAIRequestValidator.validate(
                try Self.decode(extra), modelID: "m", dialect: .chatml,
                preStagedImages: [:], attachmentLease: nil, appleModel: true)
        }
        #expect(try validate(#""seed":-1"#).generationConfig.seed == nil)
        for seed in ["-7", "7"] {
            #expect(throws: ServerRequestError.self) { try validate(#""seed":\#(seed)"#) }
        }
    }

    /// Foundation reports these as corrupt data with no coding path, which the
    /// server answered with "malformed JSON request".
    @Test func aFractionOrOverflowInAWholeNumberFieldNamesTheField() throws {
        let cases = [
            ("max_tokens", "100.5"), ("max_completion_tokens", "1e30"),
            ("top_k", "2.5"), ("n", "1.5"),
            ("seed", "1.5"), ("seed", "-1.5"), ("seed", "1e40"), ("seed", "-1e40"),
        ]
        for (field, value) in cases {
            let refused = try Self.rejection(#""\#(field)":\#(value)"#)
            #expect(refused.param == field, "\(field)=\(value)")
            #expect(refused.code == "invalid_value")
            #expect(refused.message == "\(field) must be a whole number")
        }
        // A whole number written with a fraction part is still that number.
        #expect(try Self.decode(#""max_tokens":64.0,"seed":42.0"#).maxTokens == 64)
    }

    /// The sweep wins over a type error elsewhere, so the answer names the
    /// real problem whatever order the decoder would have met them in.
    @Test func theSweepRunsBeforeTypedDecoding() throws {
        let refused = try Self.rejection(#""temperature":"hot","logit_bias":{}"#)
        #expect(refused.param == "logit_bias")
        let unknown = try Self.rejection(#""temperature":"hot","max_token":3"#)
        #expect(unknown.param == "max_token")
    }

    /// What oh-my-pi sends with every request to a Qwen model.
    @Test func qwenClientThinkingSwitchesAreAccepted() throws {
        let omp = try Self.decode(
            #""enable_thinking":true,"preserve_thinking":true,"chat_template_kwargs":{"preserve_thinking":true}"#)
        #expect(omp.enableThinking == true)
        #expect(omp.preserveThinking == true)
        let validated = try OpenAIRequestValidator.validate(omp, modelID: "m", dialect: .chatml)
        #expect(validated.reasoning == .on)
        #expect(validated.preserveThinking)
        #expect(validated.conversationTranscript.preserveThinking)
        #expect(!validated.allowsTextBridge)

        // The template argument alone drives TUFF's own control.
        let nested = try Self.decode(#""chat_template_kwargs":{"enable_thinking":false}"#)
        #expect(nested.enableThinking == false)
        #expect(try Self.decode(#""chat_template_kwargs":{}"#).enableThinking == nil)
    }

    @Test func assistantReasoningSurvivesValidationAndPreserveThinkingChangesCacheIdentity() throws {
        let data = Data(#"""
        {"model":"m","preserve_thinking":true,"messages":[
          {"role":"user","content":"first"},
          {"role":"assistant","content":"answer","reasoning_content":"checked carefully"},
          {"role":"user","content":"next"}]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m", dialect: .chatml)
        #expect(validated.messages[1].thinking == "checked carefully")
        #expect(validated.conversationTranscript.messages[1].thinking == "checked carefully")
        #expect(validated.preserveThinking)
        #expect(!validated.allowsTextBridge)
        let ordinary = try OpenAIRequestValidator.validate(try Self.decode(""), modelID: "m")
        #expect(!ordinary.preserveThinking)
        #expect(ordinary.allowsTextBridge)
    }

    @Test func reasoningOnNonAssistantMessagesIsRefused() throws {
        let data = Data(#"{"model":"m","messages":[{"role":"user","content":"x","reasoning_content":"y"}]}"#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(request, modelID: "m")
        }
    }

    @Test func chatTemplateKwargsAcceptOnlyTheThinkingSwitches() throws {
        let unknown = try Self.rejection(#""chat_template_kwargs":{"add_vision_id":true}"#)
        #expect(unknown.code == "unknown_parameter")
        #expect(unknown.param == "chat_template_kwargs")
        #expect(unknown.message.contains("enable_thinking and preserve_thinking"))

        let conflict = try Self.rejection(
            #""enable_thinking":true,"chat_template_kwargs":{"enable_thinking":false}"#)
        #expect(conflict.code == "invalid_value")

        let notBool = try Self.rejection(#""chat_template_kwargs":{"preserve_thinking":"yes"}"#)
        #expect(notBool.code == "invalid_value")
    }

    @Test func tuffsOwnReasoningControlsStillDecode() throws {
        let harmony = try Self.decode(#""reasoning_effort":"low""#)
        #expect(harmony.reasoningEffort == .low)
        let validated = try OpenAIRequestValidator.validate(
            harmony, modelID: "m", dialect: .harmony)
        #expect(validated.reasoningEffort == .low)

        let gemma = try OpenAIRequestValidator.validate(harmony, modelID: "m", dialect: .gemma)
        #expect(gemma.reasoning == .on)
        #expect(gemma.reasoningEffort == nil)

        let thinking = try Self.decode(#""enable_thinking":true"#)
        #expect(thinking.enableThinking == true)
    }

    /// gBench sends `reasoning_effort: none` next to
    /// `chat_template_kwargs.enable_thinking: false` on every no-thinking
    /// request, whatever the model.
    @Test func reasoningEffortNoneTurnsThinkingOffOnEveryFamily() throws {
        let both = try Self.decode(
            #""reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}"#)
        let alone = try Self.decode(#""reasoning_effort":"none""#)
        #expect(alone.reasoningEffort == OpenAIReasoningEffort.none)
        for request in [both, alone] {
            // Gemma, Qwen (ChatML), Flint and Inkling: thinking off.
            for dialect in [ChatDialect.gemma, .chatml, .flint, .inkling] {
                let validated = try OpenAIRequestValidator.validate(
                    request, modelID: "m", dialect: dialect)
                #expect(validated.reasoning == .off)
                #expect(validated.reasoningEffort == nil)
            }
            // GPT-OSS cannot skip reasoning; none is its least, Low.
            let harmony = try OpenAIRequestValidator.validate(
                request, modelID: "m", dialect: .harmony)
            #expect(harmony.reasoning == .off)
            #expect(harmony.reasoningEffort == .low)
            // MiniMax always reasons, as it does with enable_thinking false.
            let minimax = try OpenAIRequestValidator.validate(
                request, modelID: "m", dialect: .minimax)
            #expect(minimax.reasoning == .on)
            // Apple has no reasoning to turn off.
            _ = try OpenAIRequestValidator.validate(
                request, modelID: "m", dialect: .gemma, preStagedImages: [:],
                attachmentLease: nil, appleModel: true)
        }

        // Thinking on and off at once is refused, not guessed.
        let conflict = try Self.decode(#""reasoning_effort":"none","enable_thinking":true"#)
        for dialect in [ChatDialect.gemma, .chatml, .harmony] {
            #expect(throws: ServerRequestError.invalid(
                message: "reasoning_effort none and enable_thinking true disagree",
                param: "reasoning_effort", code: "invalid_value")) {
                try OpenAIRequestValidator.validate(conflict, modelID: "m", dialect: dialect)
            }
        }
        #expect(throws: ServerRequestError.invalid(
            message: "reasoning_effort none and enable_thinking true disagree",
            param: "reasoning_effort", code: "invalid_value")) {
            try OpenAIRequestValidator.validate(
                conflict, modelID: "m", dialect: .gemma, preStagedImages: [:],
                attachmentLease: nil, appleModel: true)
        }
        // enable_thinking alone is still GPT-OSS's wrong switch.
        #expect(throws: ServerRequestError.self) {
            try OpenAIRequestValidator.validate(
                try Self.decode(#""enable_thinking":false"#), modelID: "m", dialect: .harmony)
        }
        let contradictoryEffort = try Self.rejection(
            #""reasoning_effort":"high","enable_thinking":false"#)
        #expect(contradictoryEffort.param == "reasoning_effort")
        #expect(contradictoryEffort.code == "invalid_value")
        for value in [#""minimal""#, "1", "true"] {
            let refused = try Self.rejection(#""reasoning_effort":"# + value)
            #expect(refused.code == "invalid_value")
            #expect(refused.param == "reasoning_effort")
        }
        #expect(try Self.decode(#""reasoning_effort":null"#).reasoningEffort == nil)
    }

    @Test func ignoreEOSIsForwardedToGeneration() throws {
        let plain = try OpenAIRequestValidator.validate(try Self.decode(""), modelID: "m")
        for value in ["true", "false"] {
            let request = try Self.decode(#""ignore_eos":"# + value)
            #expect(request.ignoreEOS == (value == "true"))
            let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
            var expected = plain.generationConfig
            if value == "true" { expected.ignoreEOS = true }
            #expect(validated.generationConfig == expected)
            #expect(validated.maximumCompletionTokens == plain.maximumCompletionTokens)
        }
        #expect(throws: (any Error).self) { try Self.decode(#""ignore_eos":"yes""#) }
    }

    @Test func responseFormatTextIsAcceptedAndStructuredOutputIsRefused() throws {
        let text = try Self.decode(#""response_format":{"type":"text"}"#)
        _ = try OpenAIRequestValidator.validate(text, modelID: "m")

        let cases: [(String, String)] = [
            (#"{"type":"json_object"}"#, "unsupported_value"),
            (#"{"type":"json_schema","json_schema":{"name":"x"}}"#, "unsupported_value"),
            (#"{"type":"yaml"}"#, "invalid_value"),
            (#"{"type":3}"#, "invalid_value"),
            (#"{}"#, "invalid_value"),
            (#""text""#, "invalid_value"),
        ]
        for (format, code) in cases {
            let request = try Self.decode(#""response_format":\#(format)"#)
            do {
                _ = try OpenAIRequestValidator.validate(request, modelID: "m")
                Issue.record("expected response_format \(format) to be refused")
            } catch ServerRequestError.invalid(_, let param, let actual) {
                #expect(param == "response_format")
                #expect(actual == code, "\(format)")
            }
        }
    }

    @Test func capturedClientRequestsStillDecode() throws {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasSuffix(".json") }
        #expect(!names.isEmpty)
        for name in names {
            let data = try Data(contentsOf: directory.appendingPathComponent(name))
            _ = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        }
    }
}
