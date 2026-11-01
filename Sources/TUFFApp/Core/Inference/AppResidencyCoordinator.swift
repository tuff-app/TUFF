import Foundation
import TUFFDecodeProtocol
import TUFFModelCatalog
import TUFFRuntimeDescriptions
import TUFFModelMetadata

/// Reserves the decode service's working set and asks idle API models to yield.
public struct AppResidencyCoordinator: Sendable {
    public let registry: TUFFResidencyRegistry
    public let budgetBytes: UInt64
    public let reclaim: @Sendable (Int) async throws -> Bool

    public init(registry: TUFFResidencyRegistry, budgetBytes: UInt64,
                reclaim: @escaping @Sendable (Int) async throws -> Bool) {
        self.registry = registry
        self.budgetBytes = budgetBytes
        self.reclaim = reclaim
    }

    public static func current() -> Self {
        let support = AppSupportMigration.applicationSupportURL()
        return Self(registry: TUFFResidencyRegistry(
            directory: TUFFResidencyRegistry.directory(applicationSupport: support)),
            budgetBytes: TUFFDeviceCapabilities.current().safeAppMemoryBudgetBytes,
            reclaim: { port in
                let tokenURL = support.appendingPathComponent("TUFF/Server/control-token")
                guard let token = try? String(contentsOf: tokenURL, encoding: .utf8),
                      (1...65_535).contains(port), token.trimmingCharacters(in: .whitespacesAndNewlines).count == 64
                else { return false }
                var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/tuff/v1/unload")!)
                request.httpMethod = "POST"
                request.timeoutInterval = 5
                request.setValue("Bearer \(token.trimmingCharacters(in: .whitespacesAndNewlines))",
                                 forHTTPHeaderField: "Authorization")
                let (_, response) = try await URLSession.shared.data(for: request)
                return (response as? HTTPURLResponse)?.statusCode == 200
            })
    }

    public func reserve(model: TUFFModelDescriptor, context: Int, slots: Int,
                        chunk: Int = 128) async throws -> TUFFResidencyLease {
        let record = TUFFResidencyRecord(owner: .app, modelID: model.apiModelID,
            estimatedBytes: model.estimatedInferenceWorkingSetBytes(contextTokens: context,
                expertCacheSlots: slots, prefillChunkTokens: chunk))
        return try await reserve(record: record)
    }

    public func reserve(directory: URL, context: Int, slots: Int,
                        chunk: Int = 128, options: AppRuntimeOptions? = nil) async throws -> TUFFResidencyLease {
        struct ManifestIdentity: Decodable { let modelID: String }
        let identity = try JSONDecoder().decode(ManifestIdentity.self,
            from: GTurboModelDirectory(rootURL: directory).readMetadata("manifest.json", maxBytes: ManifestReader.defaultMaxBytes))
        if let model = TUFFModelCatalog.model(manifestModelID: identity.modelID) {
            if let options, let estimate = AppAutomaticMemoryPlanner.estimatedAllocation(
                directory: directory, context: context, options: options) {
                return try await reserve(record: .init(owner: .app, modelID: model.apiModelID,
                    estimatedBytes: estimate.peakAllocatedBytes))
            }
            return try await reserve(model: model, context: context, slots: slots, chunk: chunk)
        }
        // A custom pack still participates in admission, without publishing its path.
        return try await reserve(record: .init(owner: .app, modelID: "custom", estimatedBytes: budgetBytes))
    }

    /// Reserve the private state and runtime allocation for an additional
    /// session. Weight pages are covered by the owning model's existing lease.
    /// When a layout cannot be estimated, do not guess that another session fits.
    public func reserveSession(directory: URL, context: Int,
                               options: AppRuntimeOptions) async throws -> TUFFResidencyLease {
        guard let estimate = AppAutomaticMemoryPlanner.estimatedAllocation(
            directory: directory, context: context, options: options) else {
            throw AppInferenceError.modelLoadFailed(
                "This model's additional session memory cannot be estimated safely.")
        }
        let total = estimate.stateBytes.addingReportingOverflow(estimate.runtimeBytes)
        guard !total.overflow else {
            throw AppInferenceError.modelLoadFailed("Session memory estimate overflowed.")
        }
        return try await reserve(record: .init(owner: .app, modelID: "session",
                                               estimatedBytes: total.partialValue))
    }

    /// Image tower peaks are not yet qualified alongside other models. Reserve
    /// the whole budget exclusively while image work or retained tower mappings
    /// are active; ordinary text loads do not reserve an absent image workload.
    public func reserveVision(allowing own: [TUFFResidencyLease]) throws -> TUFFResidencyLease {
        do {
            return try registry.reserveExclusive(
                .init(owner: .app, modelID: "vision-workspace", estimatedBytes: budgetBytes),
                allowing: own)
        } catch is TUFFResidencyRegistry.MemoryBusy {
            throw AppInferenceError.modelLoadFailed(
                "Image processing \(ServingMemoryPressure.exclusivePhrase). Finish or unload other sessions and models first.")
        }
    }

    private func reserve(record: TUFFResidencyRecord) async throws -> TUFFResidencyLease {
        do {
            return try registry.reserve(record, budgetBytes: budgetBytes)
        } catch let busy as TUFFResidencyRegistry.MemoryBusy {
            for holder in busy.holders where holder.owner == .backgroundServer {
                if let port = holder.controlPort {
                    guard (try? await reclaim(port)) == true else {
                        throw AppInferenceError.modelLoadFailed(
                            "The Background API is busy or unavailable. Wait for its requests to finish, then load the model again.")
                    }
                }
            }
            do {
                return try registry.reserve(record, budgetBytes: budgetBytes)
            } catch is TUFFResidencyRegistry.MemoryBusy {
                throw AppInferenceError.modelLoadFailed(
                    "Another TUFF process \(ServingMemoryPressure.phrase). Unload its model before loading this one.")
            }
        }
    }
}
