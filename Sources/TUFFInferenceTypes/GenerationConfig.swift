public struct GenerationConfig: Codable, Sendable, Equatable {
    public var maxNewTokens: Int = 256
    public var temperature: Float = 1.0
    public var topK: Int? = nil            // nil = no truncation
    public var topP: Float? = nil          // nil = no nucleus truncation
    public var repetitionPenalty: Float = 1.0
    public var seed: UInt64? = nil         // nil = nondeterministic
    public var stopStrings: [String] = []
    public var extraStopTokens: Set<Int32> = []
    /// Plain-text benchmark mode keeps sampling after a model stop marker.
    /// Optional for compatibility with previously encoded configurations.
    public var ignoreEOS: Bool?
    public var speculative: SpeculativeDecodeConfig = .off

    public init(maxNewTokens: Int = 256,
                temperature: Float = 1.0,
                topK: Int? = nil,
                topP: Float? = nil,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stopStrings: [String] = [],
                extraStopTokens: Set<Int32> = [],
                speculative: SpeculativeDecodeConfig = .off, ignoreEOS: Bool? = nil) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stopStrings = stopStrings
        self.extraStopTokens = extraStopTokens
        self.speculative = speculative
        self.ignoreEOS = ignoreEOS
    }

    /// Compatibility overload retained for clients built against the
    /// pre-speculation initializer.
    public init(maxNewTokens: Int,
                temperature: Float,
                topK: Int?,
                topP: Float?,
                repetitionPenalty: Float,
                seed: UInt64?,
                stopStrings: [String],
                extraStopTokens: Set<Int32>) {
        self.init(maxNewTokens: maxNewTokens,
                  temperature: temperature,
                  topK: topK,
                  topP: topP,
                  repetitionPenalty: repetitionPenalty,
                  seed: seed,
                  stopStrings: stopStrings,
                  extraStopTokens: extraStopTokens,
                  speculative: .off)
    }

    public func validate() throws {
        guard maxNewTokens > 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "maxNewTokens must be greater than zero")
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "temperature must be finite and nonnegative")
        }
        // Its sibling above has been checked since this method existed; this one
        // never was, so an infinite penalty reached the kernel and quietly
        // degraded the output instead of refusing. Every CLI float guard without
        // an upper bound admits infinity, and the result was a count that broke
        // into another language and then apologised for itself, where the
        // baseline counted correctly to eight.
        guard repetitionPenalty.isFinite, repetitionPenalty > 0 else {
            throw GeneratorError.invalidGenerationConfig(
                "repetitionPenalty must be finite and greater than zero")
        }
        if let topK, !(1...256).contains(topK) {
            throw GeneratorError.invalidGenerationConfig(
                "topK must be between 1 and 256")
        }
        if let topP, (!topP.isFinite || topP <= 0 || topP > 1) {
            throw GeneratorError.invalidGenerationConfig(
                "topP must be greater than zero and at most one")
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw GeneratorError.invalidGenerationConfig(
                "topP below one requires topK; full-vocabulary nucleus sampling is not implemented")
        }
        guard !speculative.isEnabled
                || speculative.mode == .greedy
                || speculative.mode == .auto else {
            throw GeneratorError.invalidGenerationConfig(
                "only greedy speculative decoding is currently supported")
        }
        guard !speculative.isEnabled || (1...8).contains(speculative.draftTokens) else {
            throw GeneratorError.invalidGenerationConfig(
                "speculative draftTokens must be between 1 and 8")
        }
    }

}
