import Foundation
import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

@Suite struct MoEFusedFFNTests {
    private static let dimension = 128
    private static let intermediate = 64
    private static let topK = 8

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    @Test func flintBF16IntermediatePreservesLargeSwiGLUProducts() throws {
        let d = Self.dimension, f = Self.intermediate, top = Self.topK
        let gate = (0..<f).map { _ in (0..<d).map { $0.isMultiple(of: 2) ? Float(0.24) : Float(0.26) } }
        let down = (0..<d).map { _ in (0..<f).map { $0.isMultiple(of: 2) ? Float(0.000009) : Float(0.000011) } }
        let blob = Self.makeBlob(gate: gate, up: gate, down: down)
        let context = try MetalContext()
        let kernel = try MoE(context: context, siluActivation: true,
            specializedD: UInt32(d), specializedF: UInt32(f), specializedNumExperts: 128,
            topKExperts: top, wideIntermediate: true)
        let buffers = try (0..<top).map { _ in try #require(context.device.makeBuffer(
            bytes: blob.bytes, length: blob.bytes.count, options: .storageModeShared)) }
        let arg = try #require(kernel.makeRoutedArgumentBuffer(routedBlobs: buffers, topK: UInt32(top)))
        let input = try #require(Fp16Buffer.make(context.device, values: Array(repeating: 16, count: d)))
        let residual = try #require(Fp16Buffer.make(context.device, values: Array(repeating: 0, count: d)))
        let weights = try #require(Fp16Buffer.make(context.device, values: Array(repeating: 1.0 / Float(top), count: top)))
        let acts = try #require(Fp16Buffer.make(context.device, count: top * f))
        let output = try #require(Fp16Buffer.make(context.device, count: d))
        let cb = try #require(context.queue.makeCommandBuffer())
        kernel.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb, routedArgBuffer: arg,
            routedBlobs: buffers, routedOffsets: blob.offsets, x: input, acts: acts,
            d: UInt32(d), f: UInt32(f), topK: UInt32(top))
        kernel.encodeRoutedPersistentPhase2Reduce(commandBuffer: cb, routedArgBuffer: arg,
            routedBlobs: buffers, routedOffsets: blob.offsets, acts: acts, routingWeights: weights,
            residual: residual, y: output, d: UInt32(d), f: UInt32(f), topK: UInt32(top))
        cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb)
        let firstBits = acts.contents().load(as: UInt16.self)
        #expect(Quantization.bf16ToFloat(firstBits) > 65_504)
        for value in Fp16Buffer.read(output, count: d) {
            #expect(value.isFinite)
            #expect(abs(value - 168) < 10)
        }
    }

    @Test func flintFloatIntermediatePreservesLargeSwiGLUProducts() throws {
        let d = Self.dimension, f = Self.intermediate, top = Self.topK
        let gate = (0..<f).map { _ in (0..<d).map { $0.isMultiple(of: 2) ? Float(0.24) : Float(0.26) } }
        let down = (0..<d).map { _ in (0..<f).map { $0.isMultiple(of: 2) ? Float(0.000009) : Float(0.000011) } }
        let blob = Self.makeBlob(gate: gate, up: gate, down: down)
        let context = try MetalContext()
        let kernel = try MoE(context: context, siluActivation: true,
            specializedD: UInt32(d), specializedF: UInt32(f), specializedNumExperts: 128,
            topKExperts: top, wideIntermediate: true, floatIntermediate: true)
        let buffers = try (0..<top).map { _ in try #require(context.device.makeBuffer(
            bytes: blob.bytes, length: blob.bytes.count, options: .storageModeShared)) }
        let arg = try #require(kernel.makeRoutedArgumentBuffer(routedBlobs: buffers, topK: UInt32(top)))
        let input = try #require(Fp16Buffer.make(context.device, values: Array(repeating: 16, count: d)))
        let residual = try #require(Fp16Buffer.make(context.device, values: Array(repeating: 0, count: d)))
        let weights = try #require(Fp16Buffer.make(context.device, values: Array(repeating: 1.0 / Float(top), count: top)))
        let acts = try #require(context.device.makeBuffer(length: top * f * MemoryLayout<Float>.stride, options: .storageModeShared))
        let output = try #require(Fp16Buffer.make(context.device, count: d))
        let cb = try #require(context.queue.makeCommandBuffer())
        kernel.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb, routedArgBuffer: arg,
            routedBlobs: buffers, routedOffsets: blob.offsets, x: input, acts: acts,
            d: UInt32(d), f: UInt32(f), topK: UInt32(top))
        kernel.encodeRoutedPersistentPhase2Reduce(commandBuffer: cb, routedArgBuffer: arg,
            routedBlobs: buffers, routedOffsets: blob.offsets, acts: acts, routingWeights: weights,
            residual: residual, y: output, d: UInt32(d), f: UInt32(f), topK: UInt32(top))
        cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb)
        #expect(acts.contents().load(as: Float.self) > 65_504)
        for value in Fp16Buffer.read(output, count: d) {
            #expect(value.isFinite)
            #expect(abs(value - 168) < 10)
        }
    }

    @Test func mixedAffineWidthsFullAndSubsetDecode() throws {
        let d = 128, f = 64, top = 2
        let context = try MetalContext()
        let inputValues = (0..<d).map { Float(Float16(Float(($0 % 7) - 3) / 8)) }
        let input = try #require(Fp16Buffer.make(context.device, values: inputValues))
        let residual = try #require(Fp16Buffer.make(context.device, values: Array(repeating: 0.25, count: d)))
        let routing = try #require(Fp16Buffer.make(context.device, values: [0.25, 0.75]))
        let slots: [UInt32] = [1, 0]
        let slotBuffer = try #require(context.device.makeBuffer(
            bytes: slots, length: slots.count * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let scale: Float = 1 / 512
        let bias: Float = -1 / 32
        func q(_ row: Int, _ column: Int, _ slot: Int, _ bits: UInt32) -> Int {
            (row * 3 + column * 5 + slot * 11) % (bits == 4 ? 16 : 256)
        }
        for wide in [false, true] {
            let kernel = try MoE(context: context, siluActivation: true,
                specializedD: UInt32(d), specializedF: UInt32(f),
                topKExperts: top, wideIntermediate: wide)
            for gateBits in [UInt32(4), 8] {
                for upBits in [UInt32(4), 8] {
                    for downBits in [UInt32(4), 8] {
                        let widths = MoEExpertBitWidths(gate: gateBits, up: upBits, down: downBits)
                        var blobs = [MTLBuffer]()
                        var offsets: MoEExpertOffsets!
                        for slot in 0..<top {
                            var bytes = [UInt8]()
                            func appendU16(_ value: UInt16) {
                                bytes.append(UInt8(truncatingIfNeeded: value))
                                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
                            }
                            func projection(rows: Int, columns: Int, bits: UInt32) -> (UInt32, UInt32, UInt32) {
                                let w = UInt32(bytes.count)
                                for row in 0..<rows {
                                    if bits == 8 {
                                        for col in 0..<columns { bytes.append(UInt8(q(row, col, slot, bits))) }
                                    } else {
                                        for col in stride(from: 0, to: columns, by: 2) {
                                            bytes.append(UInt8(q(row, col, slot, bits) | (q(row, col + 1, slot, bits) << 4)))
                                        }
                                    }
                                }
                                let s = UInt32(bytes.count)
                                for _ in 0..<(rows * columns / 64) { appendU16(Quantization.bf16Bits(scale)) }
                                let b = UInt32(bytes.count)
                                for _ in 0..<(rows * columns / 64) { appendU16(Quantization.bf16Bits(bias)) }
                                return (w, s, b)
                            }
                            let g = projection(rows: f, columns: d, bits: gateBits)
                            let u = projection(rows: f, columns: d, bits: upBits)
                            let down = projection(rows: d, columns: f, bits: downBits)
                            offsets = MoEExpertOffsets(gateWOff: g.0, gateSOff: g.1, gateBOff: g.2,
                                upWOff: u.0, upSOff: u.1, upBOff: u.2,
                                downWOff: down.0, downSOff: down.1, downBOff: down.2)
                            blobs.append(try #require(context.device.makeBuffer(
                                bytes: bytes, length: bytes.count, options: .storageModeShared)))
                        }
                        let arg = try #require(kernel.makeRoutedArgumentBuffer(routedBlobs: blobs, topK: UInt32(top)))
                        let fullActs = try #require(Fp16Buffer.make(context.device, count: top * f))
                        let subsetActs = try #require(Fp16Buffer.make(context.device, count: top * f))
                        let full = try #require(Fp16Buffer.make(context.device, count: d))
                        let subset = try #require(Fp16Buffer.make(context.device, count: d))
                        let cb = try #require(context.queue.makeCommandBuffer())
                        kernel.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb, routedArgBuffer: arg,
                            routedBlobs: blobs, routedOffsets: offsets, x: input, acts: fullActs,
                            d: UInt32(d), f: UInt32(f), topK: UInt32(top), bitWidths: widths)
                        kernel.encodeRoutedPersistentPhase1SubsetU16Load(commandBuffer: cb, routedArgBuffer: arg,
                            routedBlobs: blobs, routedOffsets: offsets, x: input, acts: subsetActs,
                            activeSlots: slotBuffer, activeSlotIndices: slots, activeCount: UInt32(top),
                            d: UInt32(d), f: UInt32(f), topK: UInt32(top), bitWidths: widths)
                        for (acts, output) in [(fullActs, full), (subsetActs, subset)] {
                            kernel.encodeRoutedPersistentPhase2Reduce(commandBuffer: cb, routedArgBuffer: arg,
                                routedBlobs: blobs, routedOffsets: offsets, acts: acts, routingWeights: routing,
                                residual: residual, y: output, d: UInt32(d), f: UInt32(f),
                                topK: UInt32(top), bitWidths: widths)
                        }
                        cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb)
                        var reference = Array(repeating: Float(0.25), count: d)
                        for slot in 0..<top {
                            let acts = (0..<f).map { row -> Float in
                                func dot(_ bits: UInt32) -> Float {
                                    (0..<d).reduce(Float(0)) {
                                        $0 + (Float(q(row, $1, slot, bits)) * scale + bias) * inputValues[$1]
                                    }
                                }
                                let gate = dot(gateBits)
                                let value = gate / (1 + exp(-gate)) * dot(upBits)
                                return wide ? Quantization.bf16ToFloat(Quantization.bf16Bits(value)) : Float(Float16(value))
                            }
                            for row in 0..<d {
                                let value = (0..<f).reduce(Float(0)) {
                                    $0 + (Float(q(row, $1, slot, downBits)) * scale + bias) * acts[$1]
                                }
                                reference[row] += (slot == 0 ? 0.25 : 0.75) * value
                            }
                        }
                        let result = Fp16Buffer.read(full, count: d)
                        #expect(result == Fp16Buffer.read(subset, count: d))
                        for row in 0..<d {
                            #expect(abs(result[row] - reference[row]) < 0.01)
                        }
                    }
                }
            }
        }
    }

    @Test func productionRoutedPipelineAndHitSplitMatchReference() throws {
        var rng = SeedTree(0x2D3).key("production-routed-moe")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }

        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<Self.topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let residual = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let routingWeights = (0..<Self.topK).map {
            Float(Float16(0.04 + Float($0) * 0.015))
        }
        let expected = MoeRef.applyStreamedRouted(
            x: x,
            residual: residual,
            routedGate: gates.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedUp: ups.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedDown: downs.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            indices: Array(0..<Self.topK),
            routingWeights: routingWeights,
            d: Self.dimension,
            f: Self.intermediate)
        let blobs = (0..<Self.topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == Self.topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let fullActs = Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate),
              let splitActs = Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate),
              let fullOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let splitOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let lowSlots = context.device.makeBuffer(
                bytes: [UInt32](0...3),
                length: 4 * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let highSlots = context.device.makeBuffer(
                bytes: [UInt32](4...7),
                length: 4 * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(Self.topK)) else {
            Issue.record("buffer allocation failed")
            return
        }

        let fullCommand = context.queue.makeCommandBuffer()!
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: fullActs,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: fullActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: fullOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        fullCommand.commit()
        fullCommand.waitUntilCompleted()
        #expect(fullCommand.error == nil)

        let splitCommand = context.queue.makeCommandBuffer()!
        for (slots, activeSlots) in [([UInt32](0...3), lowSlots),
                                     ([UInt32](4...7), highSlots)] {
            kernel.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: splitCommand,
                routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers,
                routedOffsets: blobs[0].offsets,
                x: xBuffer,
                acts: splitActs,
                activeSlots: activeSlots,
                activeSlotIndices: slots,
                activeCount: UInt32(slots.count),
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(Self.topK))
        }
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: splitCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: splitActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: splitOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        splitCommand.commit()
        splitCommand.waitUntilCompleted()
        #expect(splitCommand.error == nil)

        let full = Fp16Buffer.read(fullOutput, count: Self.dimension)
        let split = Fp16Buffer.read(splitOutput, count: Self.dimension)
        #expect(full == split)
        #expect(RelError.compute(actual: full, reference: expected)
            < Tolerance.fp16ChainedReduction)
    }

    private static func makeBlob(gate: [[Float]],
                                 up: [[Float]],
                                 down: [[Float]]) -> RoutedBlob {
        func packed(_ rows: [[Float]])
            -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
            return (quantized.flatMap(\.packed),
                    quantized.flatMap(\.scales),
                    quantized.flatMap(\.biases))
        }
        var bytes = [UInt8]()
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func append(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let gateValues = packed(gate)
        let upValues = packed(up)
        let downValues = packed(down)
        let gateW = UInt32(bytes.count); append(gateValues.weights)
        let gateS = UInt32(bytes.count); append(gateValues.scales)
        let gateB = UInt32(bytes.count); append(gateValues.biases)
        let upW = UInt32(bytes.count); append(upValues.weights)
        let upS = UInt32(bytes.count); append(upValues.scales)
        let upB = UInt32(bytes.count); append(upValues.biases)
        let downW = UInt32(bytes.count); append(downValues.weights)
        let downS = UInt32(bytes.count); append(downValues.scales)
        let downB = UInt32(bytes.count); append(downValues.biases)
        return RoutedBlob(
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB))
    }
}
