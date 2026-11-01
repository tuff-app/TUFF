import Foundation
import Metal
@_exported import TUFFRuntimeDescriptions
@_exported import TUFFInferenceTypes

public typealias ModelFamily = TUFFRuntimeDescriptions.ModelFamily
public typealias ModelVariant = TUFFRuntimeDescriptions.ModelVariant
public typealias FeedForwardKind = TUFFRuntimeDescriptions.FeedForwardKind
public typealias YaRNRopeConfig = TUFFRuntimeDescriptions.YaRNRopeConfig
public typealias LinearAttentionConfig = TUFFRuntimeDescriptions.LinearAttentionConfig
public typealias HyperConnectionConfig = TUFFRuntimeDescriptions.HyperConnectionConfig
public typealias NgramEmbeddingConfig = TUFFRuntimeDescriptions.NgramEmbeddingConfig
public typealias AttentionIndexerConfig = TUFFRuntimeDescriptions.AttentionIndexerConfig
public typealias ArchConfig = TUFFRuntimeDescriptions.ArchConfig

/// Failure modes for the validation gates in `Model.load`.
package typealias ModelError = TUFFModelMetadata.ModelError

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    /// Resident index name, when the view came from a named resident tensor.
    /// Used to resolve per-projection metadata such as Prism sign vectors.
    public let name: String?
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8, name: String? = nil) {
        self.name = name
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
