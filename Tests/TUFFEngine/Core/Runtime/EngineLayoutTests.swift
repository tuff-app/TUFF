import Testing
import TUFFModelCatalog
import TUFFRuntimeDescriptions

/// The host asks for an engine by architecture and weight layout. A family
/// that names the wrong layout finds no engine: Inkling's first packaged
/// build asked for "affine" and was refused its own "nvfp4" engine.
@Suite struct EngineLayoutTests {
    @Test func everyCatalogModelAsksForTheLayoutItsEngineLists() throws {
        for descriptor in TUFFModelCatalog.all {
            let variant = try #require(ModelVariant(rawValue: descriptor.architecture.id.rawValue))
            let config = try #require(ArchConfig.registeredArchitectures[variant])
            let expected = switch descriptor.family {
            case .gptOss: "mxfp4"
            case .inkling2: "nvfp4"
            default: "affine"
            }
            #expect(config.family.engineQuantizationLayout == expected, "\(descriptor.selector)")
        }
        #expect(TUFFModelCatalog.inkling2Mini.architecture.weightLayout.rawValue
            == ModelFamily.inkling2.engineQuantizationLayout)
    }
}
