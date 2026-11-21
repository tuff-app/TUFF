import Foundation
import Metal
import Testing
@testable import TUFFEngine

/// The three ways an expert product is computed, against each other on
/// random weights: one expert and row at a time, a decoded token's eight
/// experts fused, and a prompt block's batched matmul. Needs no model.
@Suite struct InklingExpertKernelTests {
    private struct Generator {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state >> 33
        }
        mutating func byte(_ range: ClosedRange<UInt8>) -> UInt8 {
            range.lowerBound + UInt8(next() % UInt64(range.count))
        }
        mutating func unit() -> Float { Float(next() % 20_001) / 10_000 - 1 }
    }

    /// A decoded token's two fused dispatches against one expert at a time:
    /// gate/up product, activation and weighted down product for each of eight.
    @Test(arguments: [true, false])
    func fusedDecodeMatchesOneExpertAtATime(nvfp4: Bool) throws {
        let context = try MetalContext()
        try context.prepareKernelGroups([.core, .inkling])
        let kernels = try InklingKernels(context: context)
        let width = 192, intermediate = 96, experts = 8
        var generator = Generator()
        func buffer(_ bytes: Int) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(length: bytes, options: .storageModeShared))
        }
        // One blob per expert, laid out as a layer file lays them out: gate/up
        // codes, gate/up scales, down codes, down scales.
        let gateUpCodeBytes = 2 * intermediate * (nvfp4 ? width / 2 : width)
        let gateUpScaleBytes = 2 * intermediate * (nvfp4 ? width / 16 : width / 32)
        let downCodeBytes = width * (nvfp4 ? intermediate / 2 : intermediate)
        let downScaleBytes = width * (nvfp4 ? intermediate / 16 : intermediate / 32)
        let offsets = [0, gateUpCodeBytes, gateUpCodeBytes + gateUpScaleBytes,
                       gateUpCodeBytes + gateUpScaleBytes + downCodeBytes]
        let blobBytes = offsets[3] + downScaleBytes
        let padding = 64  // the blobs do not start at the front of their buffers
        let blobs = try (0..<experts).map { _ in try buffer(padding + blobBytes) }
        for blob in blobs {
            let bytes = blob.contents().assumingMemoryBound(to: UInt8.self) + padding
            for (index, (start, end)) in zip(offsets, offsets.dropFirst() + [blobBytes]).enumerated() {
                for position in start..<end {
                    if index % 2 == 1 {
                        bytes[position] = nvfp4 ? generator.byte(0x30...0x4F) : generator.byte(121...125)
                    } else {
                        var value = generator.byte(0...255)
                        if !nvfp4, value & 0x7F == 0x7F { value &= 0xF7 }
                        bytes[position] = value
                    }
                }
            }
        }
        let x = try buffer(width * 4)
        for index in 0..<width { x.contents().assumingMemoryBound(to: Float.self)[index] = generator.unit() }
        let gateFactors = (0..<experts).map { _ in nvfp4 ? (0.5 + generator.unit() * 0.3) / 200 : Float(1) }
        let downFactors = (0..<experts).map { _ in nvfp4 ? (0.5 + generator.unit() * 0.3) / 200 : Float(1) }
        let weights = (0..<experts).map { _ in 0.5 + generator.unit() * 0.4 }
        let zero = try buffer(4), one = try buffer(4), weight = try buffer(experts * 4)
        zero.contents().assumingMemoryBound(to: UInt32.self)[0] = 0
        one.contents().assumingMemoryBound(to: Float.self)[0] = 1
        for (index, value) in weights.enumerated() {
            weight.contents().assumingMemoryBound(to: Float.self)[index] = value
        }
        let gateUp = try buffer(2 * intermediate * 4), hidden = try buffer(intermediate * 4)
        let expected = try buffer(width * 4), fusedHidden = try buffer(experts * intermediate * 4)
        let actual = try buffer(width * 4)
        memset(expected.contents(), 0, expected.length)

        let pass = try InklingPass(context.queue)
        let gemv = nvfp4 ? "ink_expert_gemv_nvfp4" : "ink_expert_gemv_mxfp8"
        for (index, blob) in blobs.enumerated() {
            kernels.groups(pass, gemv,
                           [.buffer(blob, padding + offsets[0]), .buffer(blob, padding + offsets[1]), .buffer(x),
                            .buffer(gateUp), .buffer(zero), .buffer(one), .u32(width), .u32(2 * intermediate),
                            .f32(gateFactors[index]), .u32(0)], grid: (2 * intermediate, 1), threads: 32)
            kernels.threads(pass, "ink_expert_activation",
                            [.buffer(gateUp), .buffer(hidden), .u32(intermediate), .f32(4), .f32(25)],
                            grid: (intermediate, 1, 1))
            kernels.groups(pass, gemv,
                           [.buffer(blob, padding + offsets[2]), .buffer(blob, padding + offsets[3]),
                            .buffer(hidden), .buffer(expected), .buffer(zero), .buffer(weight, index * 4),
                            .u32(intermediate), .u32(width), .f32(downFactors[index]), .u32(1)],
                           grid: (width, 1), threads: 32)
        }
        let bound = blobs.map { InklingArgument.buffer($0, padding) }
        let format = nvfp4 ? "nvfp4" : "mxfp8"
        kernels.groups(pass, "ink_expert_phase1_" + format,
                       bound + [.buffer(x), .buffer(fusedHidden), .floats(gateFactors), .u32(offsets[0]),
                                .u32(offsets[1]), .u32(width), .u32(intermediate), .f32(4), .f32(25)],
                       grid: (intermediate, experts), threads: 32)
        kernels.groups(pass, "ink_expert_phase2_" + format,
                       bound + [.buffer(fusedHidden), .buffer(actual),
                                .floats(zip(downFactors, weights).map { $0 * $1 }), .u32(offsets[2]),
                                .u32(offsets[3]), .u32(intermediate)],
                       grid: (width, 1), threads: 32)
        pass.endEncoding()
        pass.commandBuffer.commit()
        pass.commandBuffer.waitUntilCompleted()
        #expect(pass.commandBuffer.error == nil)
        let want = expected.contents().assumingMemoryBound(to: Float.self)
        let got = actual.contents().assumingMemoryBound(to: Float.self)
        var worst: Float = 0, scale: Float = 0
        for index in 0..<width {
            worst = max(worst, abs(want[index] - got[index]))
            scale = max(scale, abs(want[index]))
        }
        // The MXFP8 case drives gates into the hundreds, where an unguarded
        // tanh overflows; every value must still be a number.
        #expect((0..<width).allSatisfy { want[$0].isFinite && got[$0].isFinite })
        #expect(scale > 0 && worst <= 2e-5 * scale, "worst \(worst) against values up to \(scale)")
    }

    /// Every block size up to past the largest tile, and a column count that
    /// ends inside a tile.
    @Test(arguments: [true, false])
    func expertProductsMatchThePerRowKernels(nvfp4: Bool) throws {
        let context = try MetalContext()
        try context.prepareKernelGroups([.core, .inkling])
        // Systems without the tensor pipeline read blocks with the per-row kernels.
        guard let batched = InklingBatchedProducts(context: context) else { return }
        let kernels = try InklingKernels(context: context)
        let columns = 256, outRows = 200, maxRows = 70
        var generator = Generator()
        func buffer(_ bytes: Int) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(length: bytes, options: .storageModeShared))
        }
        let codes = try buffer(outRows * (nvfp4 ? columns / 2 : columns))
        let scales = try buffer(outRows * (nvfp4 ? columns / 16 : columns / 32))
        let codeBytes = codes.contents().assumingMemoryBound(to: UInt8.self)
        let scaleBytes = scales.contents().assumingMemoryBound(to: UInt8.self)
        for index in 0..<codes.length {
            // Any E2M1 pair, or any finite E4M3 value of either sign.
            var value = generator.byte(0...255)
            if !nvfp4, value & 0x7F == 0x7F { value &= 0xF7 }
            codeBytes[index] = value
        }
        for index in 0..<scales.length {
            // Block scales: E4M3 from subnormal to large, or powers of two near one.
            scaleBytes[index] = nvfp4 ? generator.byte(0...0x7E) : generator.byte(119...131)
        }
        let x = try buffer(maxRows * columns * 4), xHalf = try buffer(maxRows * columns * 2)
        let floats = x.contents().assumingMemoryBound(to: Float.self)
        let halves = xHalf.contents().assumingMemoryBound(to: Float16.self)
        for index in 0..<(maxRows * columns) {
            // Values a half holds exactly, so the two paths see the same input.
            let value = Float(Float16(generator.unit()))
            floats[index] = value
            halves[index] = Float16(value)
        }
        let identity = try buffer(maxRows * 4), ones = try buffer(maxRows * 4)
        for row in 0..<maxRows {
            identity.contents().assumingMemoryBound(to: UInt32.self)[row] = UInt32(row)
            ones.contents().assumingMemoryBound(to: Float.self)[row] = 1
        }
        let expected = try buffer(maxRows * outRows * 4), actual = try buffer(maxRows * outRows * 4)
        let factor: Float = nvfp4 ? 0.37 / 2_688 : 1

        for rows in 1...maxRows {
            memset(actual.contents(), 0xFF, actual.length)
            let pass = try InklingPass(context.queue)
            kernels.groups(pass, nvfp4 ? "ink_expert_gemv_nvfp4" : "ink_expert_gemv_mxfp8",
                           [.buffer(codes), .buffer(scales), .buffer(x), .buffer(expected), .buffer(identity),
                            .buffer(ones), .u32(columns), .u32(outRows), .f32(factor), .u32(0)],
                           grid: (outRows, rows), threads: 32)
            batched.expert(pass, nvfp4: nvfp4, codes: (codes, 0), scales: (scales, 0),
                           x: (xHalf, 0), y: (actual, 0), m: rows, n: outRows, k: columns, factor: factor)
            pass.endEncoding()
            pass.commandBuffer.commit()
            pass.commandBuffer.waitUntilCompleted()
            #expect(pass.commandBuffer.error == nil)
            let want = expected.contents().assumingMemoryBound(to: Float.self)
            let got = actual.contents().assumingMemoryBound(to: Float.self)
            var worst: Float = 0, scale: Float = 0
            for index in 0..<(rows * outRows) {
                worst = max(worst, abs(want[index] - got[index]))
                scale = max(scale, abs(want[index]))
            }
            #expect(scale > 0 && worst.isFinite && worst <= 2e-3 * scale,
                    "rows \(rows): worst \(worst) against values up to \(scale)")
            // Nothing is written past the block.
            #expect(got[rows * outRows].isNaN || rows == maxRows)
        }
    }
}
