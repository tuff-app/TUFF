import Foundation

/// Publisher sidecar for a Prism rotated-weight snapshot. No model is marked
/// supported until the repacker carries this contract into the runtime.
struct PrismHadamard: Decodable {
    let version: Int
    let blockSize: Int
    let transform: String
    let axis: String
    let signMode: String
    let weightNames: [String]
    let inverseWeightNames: [String]
    let signWidths: [Int]
    let signValues: [Int]
    let gdnValuesGrouped: Bool
    /// Older untied packs omit this field; absence means an explicit head.
    let tiedOutput: Bool?

    enum CodingKeys: String, CodingKey {
        case version = "prism.hadamard.version"
        case blockSize = "prism.hadamard.block_size"
        case transform = "prism.hadamard.transform"
        case axis = "prism.hadamard.axis"
        case signMode = "prism.hadamard.sign_mode"
        case weightNames = "prism.hadamard.weight_names"
        case inverseWeightNames = "prism.hadamard.inverse_weight_names"
        case signWidths = "prism.hadamard.sign_widths"
        case signValues = "prism.hadamard.sign_values"
        case gdnValuesGrouped = "prism.hadamard.gdn_v_grouped"
        case tiedOutput = "prism.hadamard.tied_output"
    }

    static func load(path: String) throws -> PrismHadamard {
        let value: PrismHadamard
        do {
            value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        } catch {
            throw RepackError.configJsonInvalid(path: path, detail: "invalid Prism Hadamard sidecar: \(error)")
        }
        let names = value.weightNames + value.inverseWeightNames
        var total = 0
        for width in value.signWidths {
            let (next, overflow) = total.addingReportingOverflow(width)
            guard width > 0, width % max(1, value.blockSize) == 0, !overflow else {
                throw RepackError.configJsonInvalid(path: path, detail: "invalid Prism sign width")
            }
            total = next
        }
        guard value.version == 1,
              [512, 1024, 2048, 4096].contains(value.blockSize),
              value.transform == "normalized-sylvester-walsh-hadamard",
              value.axis == "input-last-dimension", value.signMode == "explicit",
              !value.signWidths.isEmpty, Set(value.signWidths).count == value.signWidths.count,
              total == value.signValues.count,
              value.signValues.allSatisfy({ $0 == -1 || $0 == 1 }),
              !names.isEmpty, names.allSatisfy({ !$0.isEmpty }), Set(names).count == names.count else {
            throw RepackError.configJsonInvalid(path: path, detail: "unsupported or inconsistent Prism Hadamard contract")
        }
        return value
    }

    /// Check each embedded sign tensor against the explicit sidecar vectors.
    /// Names use the publisher's safetensors namespace; the runtime may only
    /// deduplicate these vectors after this byte-level comparison succeeds.
    func validateTensors(_ tensors: [String: SourceTensor], sidecarPath: String) throws {
        var signsByWidth: [Int: ArraySlice<Int>] = [:]
        var offset = 0
        for width in signWidths {
            signsByWidth[width] = signValues[offset..<(offset + width)]
            offset += width
        }
        for name in weightNames + inverseWeightNames {
            guard name.hasSuffix(".weight"), let weight = tensors[name],
                  weight.dtype == .u32, weight.shape.count == 2,
                  weight.shape[1] <= UInt64(Int.max / 16) else {
                throw RepackError.configJsonInvalid(path: sidecarPath, detail: "invalid rotated weight \(name)")
            }
            let width = Int(weight.shape[1]) * 16
            let signName = String(name.dropLast(".weight".count)) + ".signs"
            guard let signs = tensors[signName], signs.dtype == .fp32,
                  signs.shape == [UInt64(width)], signs.sizeBytes == UInt64(width) * 4,
                  let expected = signsByWidth[width] else {
                throw RepackError.configJsonInvalid(path: sidecarPath, detail: "invalid sign tensor for \(name)")
            }
            let data: Data
            do {
                let file = try FileHandle(forReadingFrom: URL(fileURLWithPath: signs.shardPath))
                defer { try? file.close() }
                try file.seek(toOffset: signs.absoluteOffset)
                data = try file.read(upToCount: width * 4) ?? Data()
            } catch {
                throw RepackError.configJsonInvalid(path: sidecarPath, detail: "cannot read signs for \(name): \(error)")
            }
            guard data.count == width * 4 else {
                throw RepackError.configJsonInvalid(path: sidecarPath, detail: "truncated signs for \(name)")
            }
            let matches = data.withUnsafeBytes { bytes in
                zip(0..<width, expected).allSatisfy { index, value in
                    UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self))
                        == Float(value).bitPattern
                }
            }
            guard matches else {
                throw RepackError.configJsonInvalid(path: sidecarPath, detail: "sidecar signs do not match \(name)")
            }
        }
    }
}
