import Foundation
import TUFFFormat

/// Plans a text-only resident image from a validated Prism source. The plan
/// retains packed Q2 weights, their FP16 affine companions and F32 signs;
/// executing it still requires a Prism-aware manifest and forward runner.
enum PrismResidentPlanner {
    static func planModel(snapshot: PrismSnapshot, outputDirectory: String) throws -> RepackPlan {
        let arch = try ArchInfo.loadValidatedPrism(snapshot)
        let resident = try plan(snapshot: snapshot,
                                outputPath: (outputDirectory as NSString).appendingPathComponent("model_weights.bin"))
        let excluded = snapshot.tensors.keys.filter { !$0.hasPrefix("language_model.") }.sorted()
        return RepackPlan(arch: arch, baseMode: "affine", baseGroupSize: 128,
                          bitsOverrideCount: 0, resident: resident, layers: [], ngramTable: nil,
                          matchedModelID: nil, excludedMultimodalTensorNames: excluded)
    }

    static func plan(snapshot: PrismSnapshot, outputPath: String) throws -> ResidentFilePlan {
        let names = snapshot.tensors.keys.filter {
            $0.hasPrefix("language_model.") && !$0.hasSuffix(".scales") && !$0.hasSuffix(".biases")
        }.sorted()
        guard !names.isEmpty else {
            throw RepackError.configurationInvalid(detail: "Prism snapshot has no text tensors")
        }
        var strings: [UInt8] = []
        var stringOffsets: [UInt32] = []
        for name in names {
            guard strings.count <= Int(UInt32.max) else {
                throw RepackError.configurationInvalid(detail: "Prism tensor names exceed index limits")
            }
            stringOffsets.append(UInt32(strings.count))
            strings.append(contentsOf: name.utf8)
        }
        let count = UInt64(names.count)
        let fixed = UInt64(GTurboBinary.indexHeaderBytes)
        let (entriesBytes, entryOverflow) = count.multipliedReportingOverflow(by: UInt64(GTurboBinary.indexEntryBytes))
        let (prefix, prefixOverflow) = fixed.addingReportingOverflow(entriesBytes)
        let (rawIndex, stringOverflow) = prefix.addingReportingOverflow(UInt64(strings.count))
        let (rounded, roundOverflow) = rawIndex.addingReportingOverflow(Layout.pageBytes - 1)
        guard !entryOverflow, !prefixOverflow, !stringOverflow, !roundOverflow else {
            throw RepackError.configurationInvalid(detail: "Prism resident index size overflows")
        }
        let indexSize = rounded / Layout.pageBytes * Layout.pageBytes
        var cursor = indexSize
        var result: [ResidentEntry] = []
        result.reserveCapacity(names.count)
        let rotatedWeights = Set(snapshot.hadamard.weightNames + snapshot.hadamard.inverseWeightNames)
        func reserve(_ size: UInt64) throws -> UInt64 {
            let start = cursor
            let (end, overflow) = cursor.addingReportingOverflow(size)
            guard !overflow else { throw RepackError.configurationInvalid(detail: "Prism resident payload overflows") }
            cursor = end
            return start
        }
        func shape(_ dimensions: [UInt64], name: String) throws -> [UInt32] {
            guard dimensions.count <= 4, dimensions.allSatisfy({ $0 > 0 && $0 <= UInt64(UInt32.max) }) else {
                throw RepackError.configurationInvalid(detail: "invalid Prism tensor shape for \(name)")
            }
            return dimensions.map(UInt32.init) + Array(repeating: 0, count: 4 - dimensions.count)
        }
        for name in names {
            guard let weight = snapshot.tensors[name] else { throw RepackError.missingTensor(name: name) }
            let quantized = weight.dtype == .u32 && name.hasSuffix(".weight")
            if quantized {
                guard rotatedWeights.contains(name) else {
                    throw RepackError.configurationInvalid(detail: "unlisted Prism packed weight \(name)")
                }
                let base = String(name.dropLast(".weight".count))
                guard let scales = snapshot.tensors[base + ".scales"],
                      let biases = snapshot.tensors[base + ".biases"], weight.shape.count == 2,
                      weight.shape[1] <= UInt64(UInt32.max) / 16 else {
                    throw RepackError.configurationInvalid(detail: "invalid Prism companions for \(name)")
                }
                let w = try reserve(weight.sizeBytes)
                let s = try reserve(scales.sizeBytes)
                let b = try reserve(biases.sizeBytes)
                result.append(ResidentEntry(name: name, dtype: SourceTensor.Dtype.u32.rawValue,
                    logicalShape4: try shape([weight.shape[0], weight.shape[1] * 16], name: name),
                    fileOffset: w, sizeBytes: weight.sizeBytes,
                    scaleOffset: s, scaleSize: scales.sizeBytes, biasOffset: b, biasSize: biases.sizeBytes,
                    quantSpec: QuantSpec(bits: 2, groupSize: 128), sourceWeight: weight,
                    sourceScales: scales, sourceBiases: biases))
            } else {
                guard weight.dtype == .fp32 || weight.dtype == .fp16 else {
                    throw RepackError.configurationInvalid(detail: "unsupported Prism tensor dtype for \(name)")
                }
                let off = try reserve(weight.sizeBytes)
                result.append(ResidentEntry(name: name, dtype: weight.dtype.rawValue,
                    logicalShape4: try shape(weight.shape, name: name), fileOffset: off, sizeBytes: weight.sizeBytes,
                    scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0,
                    quantSpec: nil, sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }
        return ResidentFilePlan(path: outputPath, entries: result, stringTable: strings,
                                stringTableOffsets: stringOffsets, indexSize: indexSize,
                                residentSize: cursor - indexSize)
    }
}
