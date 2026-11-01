import Foundation
import TUFFFormat

enum GTurboLayoutValidator {
    static func validate(path: String,
                                plan: RepackPlan,
                                audit: RepackAudit? = nil) throws {
        // The shared cap rather than a literal: layout.json scales with layers
        // x experts, so a second copy of this bound silently caps a larger
        // model at the older value (Qwen 3.6's is ~22 MB).
        let data = try Posix.readBoundedData(
            path, maximumBytes: VerifiedInstallTool.layoutMaxBytes)
        let layout: GTurboPackedExpertsLayoutV1
        do { layout = try GTurboPackedExpertsLayoutCodec.decode(data) }
        catch {
            throw RepackError.configurationInvalid(
                detail: "layout.json validation failed: \(error)")
        }
        var validatedLogicalExperts = 0
        for layer in layout.layers {
            guard let planLayer = plan.layers.first(where: { $0.layerIndex == layer.layer }) else {
                throw RepackError.configurationInvalid(detail: "layout.json validation failed: malformed layer")
            }
            guard layer.experts.count == planLayer.expertsPerLayer,
                  (planLayer.expertsPerLayer == 0 ||
                   (planLayer.expertStride <= layout.expertStride &&
                    layer.experts.allSatisfy { $0.size == planLayer.expertStride })) else {
                throw RepackError.configurationInvalid(detail:
                    "layout.json validation failed: plan mismatch in layer \(layer.layer)")
            }
            validatedLogicalExperts += layer.experts.count
        }
        guard layout.layers.count == plan.layers.count else {
            throw RepackError.configurationInvalid(
                detail: "layout.json validation failed: layer count mismatch")
        }
        audit?.packedExpertLayoutAuditLogicalIDCount = validatedLogicalExperts
        audit?.packedExpertLayoutOffsetValidationPassed = true
    }
}
