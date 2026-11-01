import Foundation
import SwiftUI
import TUFFRuntimeManagement
import TUFFMacPresentation

struct InferenceEngineSettingsView: View {
    @State private var packages: [InferenceEnginePackageStatus] = []
    @State private var busy = false
    @State private var message: String?

    var body: some View {
        Section("Inference engines") {
            Text("Engines install with models and are shared by compatible models. Required or running engines are kept during cleanup.")
                .appFont(.caption).foregroundStyle(.secondary)
            ForEach(packages) { package in
                HStack {
                    VStack(alignment: .leading) {
                        Text(name(package.reference.packageID) + " " + package.reference.version.directoryName)
                        Text(summary(package)).appFont(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if !package.verified {
                        Button("Repair") { Task { await repair(package) } }
                            .disabled(busy || package.inUse)
                    }
                }
            }
            if packages.isEmpty {
                Text(busy ? "Checking installed engines…" : "No engines installed.")
                    .foregroundStyle(.secondary)
            }
            HStack {
                Button("Refresh") { Task { await refresh() } }
                Button("Remove unused engines") { Task { await cleanup() } }
            }.disabled(busy)
            if let message { Text(message).appFont(.caption).foregroundStyle(.secondary) }
        }
        .onAppear { Task { await refresh() } }
    }

    private func name(_ id: String) -> String {
        switch id {
        case "gemma-affine": "Gemma"
        case "qwen-hybrid-affine": "Qwen / Ornith / Bonsai"
        case "flint-affine": "Flint"
        case "gpt-oss-mxfp4": "GPT-OSS (Legacy)"
        case "minimax-affine": "MiniMax (Legacy)"
        default: id
        }
    }
    private func summary(_ package: InferenceEnginePackageStatus) -> String {
        let bytes = package.installedBytes.map { " · " + MetricFormat.storage($0) } ?? ""
        return "\(package.verified ? "Verified" : "Needs repair") · \(package.requiredModelCount) model dependencies\(package.inUse ? " · In use" : "")\(bytes)"
    }
    @MainActor private func refresh() async {
        guard !busy else { return }
        busy = true
        defer { busy = false }
        message = nil
        do {
            packages = try await Task.detached(priority: .utility) {
                try await InferenceEngineEnvironment.controller().statuses()
            }.value
        }
        catch { message = String(describing: error) }
    }
    @MainActor private func cleanup() async {
        busy = true
        defer { busy = false }
        do {
            let controller = try InferenceEngineEnvironment.controller()
            let count = try await controller.removeUnusedPackages()
            packages = try await controller.statuses()
            message = count == 1 ? "Removed 1 unused engine." : "Removed \(count) unused engines."
        } catch { message = String(describing: error) }
    }
    @MainActor private func repair(_ package: InferenceEnginePackageStatus) async {
        busy = true
        defer { busy = false }
        do {
            let controller = try InferenceEngineEnvironment.controller()
            guard let release = controller.catalog.releases.filter({ $0.packageID == package.reference.packageID })
                .sorted(by: { $0.version > $1.version }).first,
                  let architecture = release.architectureIDs.sorted().first else { return }
            let lease = try await controller.acquire(architectureID: architecture,
                quantizationLayout: package.reference.packageID == "gpt-oss-mxfp4" ? "mxfp4" : "affine",
                protocolVersion: 1, macOSMajorVersion: ProcessInfo.processInfo.operatingSystemVersion.majorVersion)
            try await controller.release(lease)
            packages = try await controller.statuses()
            message = "Compatible engine verified."
        } catch { message = String(describing: error) }
    }
}
