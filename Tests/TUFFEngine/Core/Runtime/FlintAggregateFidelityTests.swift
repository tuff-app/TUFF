import Foundation
import Testing

/// Opt-in aggregate comparison of complete logits and router sets against the
/// same-weight FP32 activation reference and the publisher's BF16 execution.
@Suite(.serialized) struct FlintAggregateFidelityTests {
    private struct Metrics {
        var klSum = 0.0
        var klMax = 0.0
        var absoluteSum = 0.0
        var absoluteMax = 0.0
        var flips = 0
        var routerMismatches = 0
    }

    private func npyPayload(_ data: Data) throws -> Int {
        try #require(data.count >= 10 && data.prefix(6) == Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]))
        let major = data[6]
        let headerBytes: Int
        let prefix: Int
        if major == 1 {
            prefix = 10
            headerBytes = Int(data[8]) | (Int(data[9]) << 8)
        } else {
            try #require(data.count >= 12 && (major == 2 || major == 3))
            prefix = 12
            headerBytes = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
        }
        try #require(prefix + headerBytes <= data.count)
        return prefix + headerBytes
    }

    @Test func aggregateMetricsBeatPublisherBF16() throws {
        guard let root = ProcessInfo.processInfo.environment["TUFF_FLINT_FIDELITY_DIR"] else { return }
        let nativeVariant = ProcessInfo.processInfo.environment["TUFF_FLINT_FIDELITY_VARIANT"] ?? "native"
        let directory = URL(fileURLWithPath: root, isDirectory: true)
        let vocabulary = 150_144, layers = 47, topK = 6
        func data(_ name: String) throws -> Data {
            try Data(contentsOf: directory.appendingPathComponent(name), options: .mappedIfSafe)
        }
        for prompt in ["tool", "code", "chat", "long"] {
            let promptFile = prompt == "tool" ? "flint-reference-tool-bf16-margins.json" : "flint-fidelity-\(prompt)-prompt.json"
            let object = try #require(JSONSerialization.jsonObject(with: data(promptFile)) as? [String: Any])
            let positions = try #require((object["input_ids"] as? [Int])?.count)
            let base = "flint-fidelity-\(prompt)-"
            let native = try data(base + nativeVariant + ".logits.f16")
            let fp32 = try data(base + "fp32.logits.npy")
            let bf16 = try data(base + "bf16.logits.npy")
            let nativeRoutes = try data(base + nativeVariant + ".routes.u32")
            let fp32Routes = try data(base + "fp32.routes.npy")
            let bf16Routes = try data(base + "bf16.routes.npy")
            let fpStart = try npyPayload(fp32), bfStart = try npyPayload(bf16)
            let fpRouteStart = try npyPayload(fp32Routes), bfRouteStart = try npyPayload(bf16Routes)
            try #require(native.count == positions * vocabulary * 2)
            try #require(fp32.count - fpStart == positions * vocabulary * 4 && bf16.count - bfStart == positions * vocabulary * 4)
            try #require(nativeRoutes.count == positions * layers * topK * 4)
            try #require(fp32Routes.count - fpRouteStart == nativeRoutes.count && bf16Routes.count - bfRouteStart == nativeRoutes.count)
            var n = Metrics(), b = Metrics()
            try native.withUnsafeBytes { nr in
                try fp32.withUnsafeBytes { fr in
                    try bf16.withUnsafeBytes { br in
                        for position in 0..<positions {
                            let row = position * vocabulary
                            func reference(_ i: Int) -> Double { Double(fr.loadUnaligned(fromByteOffset: fpStart + (row + i) * 4, as: Float.self)) }
                            func observed(_ i: Int, publisher: Bool) -> Double {
                                publisher ? Double(br.loadUnaligned(fromByteOffset: bfStart + (row + i) * 4, as: Float.self))
                                    : Double(nr.loadUnaligned(fromByteOffset: (row + i) * 2, as: Float16.self))
                            }
                            var maxRef = -Double.infinity, maxNative = -Double.infinity, maxBF = -Double.infinity
                            var refID = 0, nativeID = 0, bfID = 0
                            for i in 0..<vocabulary {
                                let r = reference(i), x = observed(i, publisher: false), y = observed(i, publisher: true)
                                guard r.isFinite && x.isFinite && y.isFinite else {
                                    throw NSError(domain: "FlintAggregateFidelity", code: 1,
                                        userInfo: [NSLocalizedDescriptionKey: "\(prompt): non-finite logit at position \(position), index \(i)"])
                                }
                                if r > maxRef { maxRef = r; refID = i }
                                if x > maxNative { maxNative = x; nativeID = i }
                                if y > maxBF { maxBF = y; bfID = i }
                                let ne = abs(x - r), be = abs(y - r)
                                n.absoluteSum += ne; b.absoluteSum += be
                                n.absoluteMax = max(n.absoluteMax, ne); b.absoluteMax = max(b.absoluteMax, be)
                            }
                            if nativeID != refID { n.flips += 1 }
                            if bfID != refID { b.flips += 1 }
                            var zr = 0.0, zn = 0.0, zb = 0.0, weightedN = 0.0, weightedB = 0.0
                            for i in 0..<vocabulary {
                                let r = reference(i), x = observed(i, publisher: false), y = observed(i, publisher: true)
                                let en = exp(x - maxNative), eb = exp(y - maxBF)
                                zr += exp(r - maxRef); zn += en; zb += eb
                                weightedN += en * (x - r); weightedB += eb * (y - r)
                            }
                            let logR = maxRef + log(zr)
                            let kn = weightedN / zn + logR - (maxNative + log(zn))
                            let kb = weightedB / zb + logR - (maxBF + log(zb))
                            n.klSum += kn; b.klSum += kb
                            n.klMax = max(n.klMax, kn); b.klMax = max(b.klMax, kb)
                        }
                    }
                }
            }
            nativeRoutes.withUnsafeBytes { nr in
                fp32Routes.withUnsafeBytes { fr in
                    bf16Routes.withUnsafeBytes { br in
                        for layer in 0..<layers {
                            for position in 0..<positions {
                                let nativeBase = (position * layers + layer) * topK
                                let refBase = (layer * positions + position) * topK
                                let nativeSet = Set((0..<topK).map { nr.loadUnaligned(fromByteOffset: (nativeBase + $0) * 4, as: UInt32.self) })
                                let referenceSet = Set((0..<topK).map { fr.loadUnaligned(fromByteOffset: fpRouteStart + (refBase + $0) * 4, as: UInt32.self) })
                                let publisherSet = Set((0..<topK).map { br.loadUnaligned(fromByteOffset: bfRouteStart + (refBase + $0) * 4, as: UInt32.self) })
                                if nativeSet != referenceSet { n.routerMismatches += 1 }
                                if publisherSet != referenceSet { b.routerMismatches += 1 }
                            }
                        }
                    }
                }
            }
            print("Flint \(prompt): KL mean native=\(n.klSum / Double(positions)) BF16=\(b.klSum / Double(positions)); max native=\(n.klMax) BF16=\(b.klMax); absolute mean native=\(n.absoluteSum / Double(positions * vocabulary)) BF16=\(b.absoluteSum / Double(positions * vocabulary)); max native=\(n.absoluteMax) BF16=\(b.absoluteMax); flips native=\(n.flips) BF16=\(b.flips); route mismatches native=\(n.routerMismatches) BF16=\(b.routerMismatches)")
            #expect(n.klSum <= b.klSum && n.klMax <= b.klMax, "\(prompt): KL exceeds BF16")
            #expect(n.absoluteSum <= b.absoluteSum && n.absoluteMax <= b.absoluteMax, "\(prompt): logit error exceeds BF16")
            #expect(n.flips <= b.flips && n.routerMismatches <= b.routerMismatches, "\(prompt): flips or router mismatches exceed BF16")
        }
    }
}
