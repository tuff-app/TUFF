import Foundation
import Testing
@testable import TUFFRepackCore

@Suite struct PrismHadamardTests {
    @Test func validatesPublisherContractAndRejectsCorruptSigns() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("tuff-prism-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: path) }
        var root: [String: Any] = [
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 512,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["layer.weight"],
            "prism.hadamard.inverse_weight_names": ["embed.weight"],
            "prism.hadamard.sign_widths": [512],
            "prism.hadamard.sign_values": Array(repeating: 1, count: 512),
            "prism.hadamard.gdn_v_grouped": true,
            "prism.hadamard.tied_output": true,
        ]
        try JSONSerialization.data(withJSONObject: root).write(to: path)
        let parsed = try PrismHadamard.load(path: path.path)
        #expect(parsed.blockSize == 512 && parsed.signValues.count == 512)
        let shard = FileManager.default.temporaryDirectory.appendingPathComponent("tuff-prism-signs-\(UUID()).bin")
        defer { try? FileManager.default.removeItem(at: shard) }
        let bits = Array(repeating: Float(1).bitPattern.littleEndian, count: 512)
        let signBytes = bits.withUnsafeBufferPointer {
            Data(bytes: $0.baseAddress!, count: $0.count * MemoryLayout<UInt32>.stride)
        }
        try signBytes.write(to: shard)
        let weight = SourceTensor(name: "layer.weight", shardPath: shard.path, dtype: .u32,
                                  shape: [1, 32], absoluteOffset: 0, sizeBytes: 128)
        let signs = SourceTensor(name: "layer.signs", shardPath: shard.path, dtype: .fp32,
                                 shape: [512], absoluteOffset: 0, sizeBytes: 2048)
        let embedding = SourceTensor(name: "embed.weight", shardPath: shard.path, dtype: .u32,
                                     shape: [1, 32], absoluteOffset: 0, sizeBytes: 128)
        let embeddingSigns = SourceTensor(name: "embed.signs", shardPath: shard.path, dtype: .fp32,
                                          shape: [512], absoluteOffset: 0, sizeBytes: 2048)
        let tensors = [weight, signs, embedding, embeddingSigns].reduce(into: [String: SourceTensor]()) { $0[$1.name] = $1 }
        try parsed.validateTensors(tensors, sidecarPath: path.path)
        try Data(repeating: 0, count: 2048).write(to: shard)
        #expect(throws: RepackError.self) { try parsed.validateTensors(tensors, sidecarPath: path.path) }
        root["prism.hadamard.sign_values"] = Array(repeating: 0, count: 512)
        try JSONSerialization.data(withJSONObject: root).write(to: path)
        #expect(throws: RepackError.self) { try PrismHadamard.load(path: path.path) }
    }

    @Test func inspectsSingleShardAndRefusesMissingCompanions() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tuff-prism-snapshot-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let prefix = "language_model.model.embed_tokens"
        let textConfig: [String: Any] = ["hidden_size": 2048, "intermediate_size": 6144,
            "num_hidden_layers": 24, "num_attention_heads": 8, "num_key_value_heads": 2,
            "head_dim": 256, "vocab_size": 248320, "tie_word_embeddings": true, "attn_output_gate": true,
            "linear_num_key_heads": 16, "linear_num_value_heads": 16, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "hidden_act": "silu",
            "rope_parameters": ["rope_theta": 10000000, "partial_rotary_factor": 0.25],
            "layer_types": (0..<24).map { $0 % 4 == 3 ? "full_attention" : "linear_attention" }]
        let config: [String: Any] = ["model_type": "prism_hadamard_qwen35", "base_model_type": "qwen3_5",
            "text_config": textConfig,
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "modules": [["path": "model.embed_tokens", "block": 512, "dtype": "float16", "embedding": true]]]
        let sidecar: [String: Any] = ["prism.hadamard.version": 1, "prism.hadamard.block_size": 512,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard", "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit", "prism.hadamard.weight_names": [],
            "prism.hadamard.inverse_weight_names": [prefix + ".weight"], "prism.hadamard.sign_widths": [512],
            "prism.hadamard.sign_values": Array(repeating: 1, count: 512),
            "prism.hadamard.gdn_v_grouped": true, "prism.hadamard.tied_output": true]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: sidecar).write(to: directory.appendingPathComponent("hadamard.json"))
        var header: [String: Any] = [:]
        var payload = Data()
        func add(_ suffix: String, dtype: String, shape: [Int], data: Data) {
            let start = payload.count
            payload.append(data)
            header[prefix + suffix] = ["dtype": dtype, "shape": shape, "data_offsets": [start, payload.count]]
        }
        add(".weight", dtype: "U32", shape: [1, 32], data: Data(repeating: 0, count: 128))
        add(".scales", dtype: "F16", shape: [1, 4], data: Data(repeating: 0, count: 8))
        add(".biases", dtype: "F16", shape: [1, 4], data: Data(repeating: 0, count: 8))
        let signs = Array(repeating: Float(1).bitPattern.littleEndian, count: 512)
        add(".signs", dtype: "F32", shape: [512], data: signs.withUnsafeBufferPointer {
            Data(bytes: $0.baseAddress!, count: $0.count * MemoryLayout<UInt32>.stride)
        })
        func writeShard() throws {
            let bytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            var length = UInt64(bytes.count).littleEndian
            var file = withUnsafeBytes(of: &length) { Data($0) }
            file.append(bytes); file.append(payload)
            try file.write(to: directory.appendingPathComponent("model.safetensors"))
        }
        try writeShard()
        let snapshot = try PrismSnapshot.inspect(directory: directory)
        #expect(snapshot.tensors.count == 4)
        let arch = try ArchInfo.loadValidatedPrism(snapshot)
        #expect(arch.variant == .bonsai2_2B && arch.tieWordEmbeddings && arch.hiddenSize == 2048)
        let plan = try PrismResidentPlanner.plan(snapshot: snapshot,
            outputPath: directory.appendingPathComponent("model_weights.bin").path)
        #expect(plan.entries.count == 2)
        let embedding = try #require(plan.entries.first { $0.name == prefix + ".weight" })
        #expect(embedding.logicalShape4 == [1, 512, 0, 0])
        #expect(embedding.quantSpec == QuantSpec(bits: 2, groupSize: 128))
        #expect(embedding.sizeBytes == 128 && embedding.scaleSize == 8 && embedding.biasSize == 8)
        let signEntry = try #require(plan.entries.first { $0.name == prefix + ".signs" })
        #expect(signEntry.dtype == SourceTensor.Dtype.fp32.rawValue && signEntry.sizeBytes == 2048)
        var shards: [String: MmapHandle] = [:]
        let file = try ResidentWriter.write(plan: plan, shardsByPath: &shards, audit: RepackAudit())
        #expect(file.size == plan.totalSize)
        let written = try Data(contentsOf: URL(fileURLWithPath: plan.path))
        let expectedSigns = signs.withUnsafeBufferPointer {
            Data(bytes: $0.baseAddress!, count: $0.count * MemoryLayout<UInt32>.stride)
        }
        #expect(written.subdata(in: Int(signEntry.fileOffset)..<Int(signEntry.fileOffset + signEntry.sizeBytes)) == expectedSigns)
        let modelPlan = try PrismResidentPlanner.planModel(snapshot: snapshot, outputDirectory: directory.path)
        let manifest = try GTurboJSON.encodeManifest(plan: modelPlan, modelID: "fixture-bonsai",
            sourceSnapshotHash: String(repeating: "a", count: 64),
            files: [("model_weights.bin", .init(size: file.size, sha256: file.sha256))],
            expertsPerLayer: 0, numLayers: 24, expertStride: 0,
            bitWidths: .init(embedding: 2, attention: 2, router: 8, sharedExpert: 2, routedExpert: 4))
        let root = try #require(JSONSerialization.jsonObject(with: manifest) as? [String: Any])
        let quant = try #require(root["quant"] as? [String: Any])
        let attention = try #require(quant["attention"] as? [String: Any])
        #expect(attention["weightBits"] as? Int == 2 && attention["groupSize"] as? Int == 128)
        #expect(attention["scaleType"] as? String == "FP16")
        header.removeValue(forKey: prefix + ".biases")
        try writeShard()
        #expect(throws: RepackError.self) { try PrismSnapshot.inspect(directory: directory) }
    }

    @Test func localRepackerRejectsUnpinnedSourceWithoutCreatingOutput() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tuff-prism-unpinned-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("not a checkpoint".utf8).write(to: directory.appendingPathComponent("model.safetensors"))
        let output = directory.appendingPathComponent("model.gturbo")
        #expect(throws: RepackError.self) {
            try PrismLocalRepacker.write(snapshotDirectory: directory, outputDirectory: output)
        }
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    @Test func pinnedLocalSnapshotWritesWhenExplicitlyRequested() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let source = environment["TUFF_PRISM_2B_SNAPSHOT"],
              let output = environment["TUFF_PRISM_2B_OUTPUT"] else { return }
        try PrismLocalRepacker.write(snapshotDirectory: URL(fileURLWithPath: source, isDirectory: true),
                                     outputDirectory: URL(fileURLWithPath: output, isDirectory: true))
        #expect(FileManager.default.fileExists(atPath: (output as NSString).appendingPathComponent("manifest.json")))
    }

    @Test func pinned27BSnapshotWritesWhenExplicitlyRequested() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let source = environment["TUFF_PRISM_27B_SNAPSHOT"],
              let output = environment["TUFF_PRISM_27B_OUTPUT"] else { return }
        try PrismLocalRepacker.write(snapshotDirectory: URL(fileURLWithPath: source, isDirectory: true),
                                     outputDirectory: URL(fileURLWithPath: output, isDirectory: true))
        #expect(FileManager.default.fileExists(atPath: (output as NSString).appendingPathComponent("manifest.json")))
    }
}
