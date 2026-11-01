import Foundation

/// Family-owned vision architecture and processor contract. Gemma remains the
/// default so existing callers and fixtures keep their v1.0 behavior.
public struct VisionConfig: Sendable, Equatable {
    public enum Architecture: String, Sendable, Equatable {
        case gemma4Legacy
        case gemma4E4B
        case gemma4E2B
        case gemma4Unified12B
        case qwen36
    }

    public let family: ModelFamily
    public let architecture: Architecture
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let headDimension: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let patchDimension: Int
    public let maximumPatches: Int
    public let poolingKernel: Int
    public let textHiddenSize: Int
    public let positionEmbeddingSize: Int
    public let positionGridSide: Int
    public let rmsEpsilon: Float
    public let ropeTheta: Float
    public let attentionScale: Float
    public let minimumPixels: Int
    public let maximumPixels: Int

    public var maximumPooledTokens: Int {
        maximumPatches / (poolingKernel * poolingKernel)
    }

    package var usesESeriesVisionTower: Bool {
        architecture == .gemma4E2B || architecture == .gemma4E4B
    }

    public init(family: ModelFamily = .gemma4) {
        self.init(family: family, modelVariant: nil)
    }

    public init(family: ModelFamily,
                modelVariant: ModelVariant?) {
        self.family = family
        rmsEpsilon = 1e-6
        switch family {
        case .gemma4:
            if modelVariant == .gemma4_E2B {
                architecture = .gemma4E2B
                hiddenSize = 768
                intermediateSize = 3_072
                numLayers = 16
                numHeads = 12
                headDimension = 64
                patchSize = 16
                temporalPatchSize = 1
                patchDimension = 768
                maximumPatches = 2_520
                poolingKernel = 3
                textHiddenSize = 1_536
                positionEmbeddingSize = 10_240
                positionGridSide = 10_240
                ropeTheta = 100
                attentionScale = 1
                minimumPixels = 0
                maximumPixels = maximumPatches * patchSize * patchSize
            } else if modelVariant == .gemma4_E4B {
                architecture = .gemma4E4B
                hiddenSize = 768
                intermediateSize = 3_072
                numLayers = 16
                numHeads = 12
                headDimension = 64
                patchSize = 16
                temporalPatchSize = 1
                patchDimension = 768
                maximumPatches = 2_520
                poolingKernel = 3
                textHiddenSize = 2_560
                positionEmbeddingSize = 10_240
                positionGridSide = 10_240
                ropeTheta = 100
                attentionScale = 1
                minimumPixels = 0
                maximumPixels = maximumPatches * patchSize * patchSize
            } else if modelVariant == .gemma4_12B_QAT {
                architecture = .gemma4Unified12B
                hiddenSize = 3_840
                intermediateSize = 0
                numLayers = 0
                numHeads = 0
                headDimension = 0
                // Unified Gemma consumes already-merged 48x48 RGB patches.
                patchSize = 48
                temporalPatchSize = 1
                patchDimension = 6_912
                maximumPatches = 280
                poolingKernel = 1
                textHiddenSize = 3_840
                positionEmbeddingSize = 1_120
                positionGridSide = 1_120
                ropeTheta = 0
                attentionScale = 0
                minimumPixels = 0
                maximumPixels = maximumPatches * patchSize * patchSize
            } else {
                architecture = .gemma4Legacy
                hiddenSize = 1_152
                intermediateSize = 4_304
                numLayers = 27
                numHeads = 16
                headDimension = 72
                patchSize = 16
                temporalPatchSize = 1
                patchDimension = 768
                maximumPatches = 2_520
                poolingKernel = 3
                textHiddenSize = 2_816
                positionEmbeddingSize = 10_240
                positionGridSide = 10_240
                ropeTheta = 100
                attentionScale = 1
                minimumPixels = 0
                maximumPixels = maximumPatches * patchSize * patchSize
            }
        case .qwen36:
            architecture = .qwen36
            hiddenSize = 1_152
            intermediateSize = 4_304
            numLayers = 27
            numHeads = 16
            headDimension = 72
            patchSize = 16
            temporalPatchSize = 2
            patchDimension = 1_536
            maximumPatches = 65_536
            poolingKernel = 2
            // Every Qwen 3.5-contract checkpoint carries this tower; the
            // merger projects into whatever residual the text model has:
            // 2,048 for Qwen 3.6 and Ornith 35B, 3,072 for Ornith 122B,
            // 4,096 for Ornith 9B and 397B, 5,120 for Qwen 3.8 27B.
            textHiddenSize = modelVariant
                .flatMap { ArchConfig.registeredArchitectures[$0] }
                .map(\.hiddenSize) ?? 2_048
            positionEmbeddingSize = 2_304
            positionGridSide = 48
            ropeTheta = 10_000
            // Qwen's vision attention is standard scaled dot product
            // attention, with head dimension 1152 / 16 = 72.
            attentionScale = 1 / sqrt(Float(headDimension))
            minimumPixels = 65_536
            maximumPixels = 16_777_216
        case .qwen4Exp:
            // Qwen3.8 Flash Next carries the same vision tower as Qwen 3.6:
            // 27 blocks at hidden 1152, 16 heads, patch 16, spatial merge 2,
            // 2,304 learned position embeddings, and the same 333 tensors.
            // Only the merger's output width differs, because it has to land
            // in a 2,560-wide residual rather than a 2,048-wide one.
            architecture = .qwen36
            hiddenSize = 1_152
            intermediateSize = 4_304
            numLayers = 27
            numHeads = 16
            headDimension = 72
            patchSize = 16
            temporalPatchSize = 2
            patchDimension = 1_536
            maximumPatches = 65_536
            poolingKernel = 2
            textHiddenSize = 2_560
            positionEmbeddingSize = 2_304
            positionGridSide = 48
            ropeTheta = 10_000
            attentionScale = 1 / sqrt(Float(headDimension))
            minimumPixels = 65_536
            maximumPixels = 16_777_216
        case .gptOss, .minimaxM2, .flint:
            architecture = .gemma4Legacy
            hiddenSize = 0
            intermediateSize = 0
            numLayers = 0
            numHeads = 0
            headDimension = 0
            patchSize = 0
            // GPT-OSS v2 checkpoints are text-only. Keep this initializer
            // fail-closed if a caller tries to construct a vision runtime.
            temporalPatchSize = 0
            patchDimension = 0
            maximumPatches = 0
            poolingKernel = 0
            textHiddenSize = 0
            positionEmbeddingSize = 0
            positionGridSide = 0
            ropeTheta = 0
            attentionScale = 0
            minimumPixels = 0
            maximumPixels = 0
            preconditionFailure("This model family has no vision companion")
        }
    }
}
