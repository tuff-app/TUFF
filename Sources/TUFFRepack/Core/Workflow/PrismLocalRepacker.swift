import Foundation

/// Builds a local experimental text pack from a pinned publisher snapshot.
/// It is deliberately not exposed through the installer until the
/// Prism runtime and schema are qualified end to end.
enum PrismLocalRepacker {
    static let pins2B: [(String, String)] = [
        ("model.safetensors", "da5095b9f34d7f0d5094090afeeadf124eef821c75136dc28957ce586ae7ac0c"),
        ("config.json", "f8d80cb7fd148b8841e63051c337d2758b4bc16243c577d35eaab6779534b591"),
        ("hadamard.json", "9e7c5f6e5f073f912dc536fc0e70e0789bd05e81c94e95721a7d4f370293f2de"),
        ("tokenizer.json", "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
        ("tokenizer_config.json", "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"),
        ("chat_template.jinja", "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"),
    ]
    static let pins27B: [(String, String)] = [
        ("model.safetensors", "130de5925082c168b7866b2e91b52e44abbafc99017e3ca352b77b5b55a269ed"),
        ("config.json", "238de7c512cc56a733421e3fd011d88f8260739e3d00e32c5d65b7943cc9f837"),
        ("hadamard.json", "7132a3ec364f0bdac1f08f905f24f0ad2f14245060f592637a0396826d3b5fe6"),
        ("tokenizer.json", "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
        ("tokenizer_config.json", "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
        ("chat_template.jinja", "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    ]

    static func write(snapshotDirectory: URL, outputDirectory: URL) throws {
        let fm = FileManager.default
        guard !fm.fileExists(atPath: outputDirectory.path) else {
            throw RepackError.configurationInvalid(detail: "Prism output already exists")
        }
        let configPath = snapshotDirectory.appendingPathComponent("config.json").path
        let configHash = try Sha256Stream.hashFile(path: configPath, noFollow: true)
        let pins: [(String, String)]
        let modelID: String
        if configHash == pins2B[1].1 {
            pins = pins2B
            modelID = "prism-ml/Ternary-Bonsai-2-2B-mlx-2bit"
        } else if configHash == pins27B[1].1 {
            pins = pins27B
            modelID = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
        } else {
            throw RepackError.sourceFingerprintRejected(path: configPath, sha256: configHash)
        }
        for (name, expected) in pins {
            try Task.checkCancellation()
            let path = snapshotDirectory.appendingPathComponent(name).path
            let actual = try Sha256Stream.hashFile(path: path, noFollow: true)
            guard actual == expected else { throw RepackError.sourceFingerprintRejected(path: path, sha256: actual) }
        }
        let snapshot = try PrismSnapshot.inspect(directory: snapshotDirectory)
        let parent = outputDirectory.deletingLastPathComponent()
        let partial = parent.appendingPathComponent(".tuff-prism-\(UUID().uuidString).partial", isDirectory: true)
        try fm.createDirectory(at: partial, withIntermediateDirectories: false)
        var committed = false
        defer { if !committed { try? fm.removeItem(at: partial) } }
        try Task.checkCancellation()
        let plan = try PrismResidentPlanner.planModel(snapshot: snapshot, outputDirectory: partial.path)
        let audit = RepackAudit()
        var shards: [String: MmapHandle] = [:]
        let weights = try ResidentWriter.write(plan: plan.resident, shardsByPath: &shards, audit: audit)
        let weightPath = snapshotDirectory.appendingPathComponent("model.safetensors").path
        guard try Sha256Stream.hashFile(path: weightPath, noFollow: true) == pins[0].1 else {
            throw RepackError.configurationInvalid(detail: "Prism weights changed during copy")
        }
        try Task.checkCancellation()
        let tokenizerDir = partial.appendingPathComponent("tokenizer", isDirectory: true)
        try fm.createDirectory(at: tokenizerDir, withIntermediateDirectories: false)
        var files: [(String, GTurboJSON.FileEntry)] = [
            ("model_weights.bin", .init(size: weights.size, sha256: weights.sha256))
        ]
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
            let destination = tokenizerDir.appendingPathComponent(name)
            try fm.copyItem(at: snapshotDirectory.appendingPathComponent(name), to: destination)
            let sha = try Sha256Stream.hashFile(path: destination.path, noFollow: true)
            guard sha == pins.first(where: { $0.0 == name })?.1 else {
                throw RepackError.configurationInvalid(detail: "Prism sidecar changed during copy: \(name)")
            }
            let size = try fm.attributesOfItem(atPath: destination.path)[.size] as? NSNumber
            guard let size else { throw RepackError.configurationInvalid(detail: "missing copied file size") }
            files.append(("tokenizer/" + name, .init(size: size.uint64Value, sha256: sha)))
        }
        let manifest = try GTurboJSON.encodeManifest(plan: plan, modelID: modelID,
            sourceSnapshotHash: "sha256:" + pins[0].1, files: files, expertsPerLayer: 0,
            numLayers: plan.arch.numLayers, expertStride: 0,
            bitWidths: .init(embedding: 2, attention: 2, router: 8, sharedExpert: 2, routedExpert: 4))
        let manifestURL = partial.appendingPathComponent("manifest.json")
        try manifest.write(to: manifestURL, options: .atomic)
        // The app, CLI and server only treat a pack with a receipt bound to its
        // final directory as installed.
        let receipt = try VerifiedInstallReceiptWriter.encode(outputDir: outputDirectory.path,
            manifestSha256: Sha256Stream.hashFile(path: manifestURL.path, noFollow: true),
            manifestSize: UInt64(manifest.count), sourceRepoID: modelID,
            sourceRevision: "sha256:" + pins[0].1,
            files: files.map { RepackAudit.OutputFile(relativePath: $0.0, size: $0.1.size, sha256: $0.1.sha256) })
        try receipt.write(to: partial.appendingPathComponent(VerifiedInstallReceiptWriter.fileName), options: .atomic)
        try Task.checkCancellation()
        try fm.moveItem(at: partial, to: outputDirectory)
        committed = true
    }
}
