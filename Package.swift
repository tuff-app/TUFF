// swift-tools-version: 6.2
import PackageDescription
import Foundation

let engineFamily = ProcessInfo.processInfo.environment["TUFF_ENGINE_FAMILY"]
let familyDefine: String? = ["gemma-affine": "TUFF_ENGINE_GEMMA", "qwen-hybrid-affine": "TUFF_ENGINE_QWEN",
    "gpt-oss-mxfp4": "TUFF_ENGINE_GPT_OSS", "minimax-affine": "TUFF_ENGINE_MINIMAX",
    "flint-affine": "TUFF_ENGINE_FLINT"][engineFamily ?? ""]
let familySettings: [SwiftSetting] = familyDefine.map { [.define($0)] } ?? []
let engineSettings = familySettings + ((engineFamily != nil && engineFamily != "gpt-oss-mxfp4")
    ? [.define("TUFF_ENGINE_AFFINE_ONLY")] : [])

let package = Package(
    name: "TUFF",
    platforms: [
        .macOS(.v15),
        .iOS(.v26),
    ],
    products: [
        .executable(name: "TUFFQualification", targets: ["TUFFQualification"]),
        .executable(name: "TUFFModelPlan", targets: ["TUFFModelPlan"]),
        .executable(name: "TUFFEnginePackage", targets: ["TUFFEnginePackage"]),
        .library(name: "TUFFEngine", targets: ["TUFFEngine"]),
        .executable(name: "TUFFRepack", targets: ["TUFFRepack"]),
        .executable(name: "TUFFCLI", targets: ["TUFFCLI"]),
        .executable(name: "TUFFCommand", targets: ["TUFFCommand"]),
        .executable(name: "TUFF", targets: ["TUFFMac"]),
        .executable(name: "TUFFDecodeService", targets: ["TUFFDecodeService"]),
        .executable(name: "TUFFInferenceEngine", targets: ["TUFFInferenceEngine"]),
        .executable(name: "TUFFServer", targets: ["TUFFServer"]),
    ],
    dependencies: [
        // swift-transformers 1.3.3, patched to load tokenizers faster.
        // See Vendor/swift-transformers/README.md.
        .package(path: "Vendor/swift-transformers"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.3"),
        .package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.9.2"),
        // SwiftMath 1.7.3, patched to find its fonts inside a packaged app.
        // See Vendor/SwiftMath/README.md.
        .package(path: "Vendor/SwiftMath"),
    ],
    targets: [
        .executableTarget(name: "TUFFQualification", dependencies: ["TUFFAppCore", "TUFFModelMetadata", "TUFFModelCatalog", "TUFFInferenceTypes"], path: "Sources/TUFFQualification"),
        .executableTarget(name: "TUFFModelPlan", dependencies: ["TUFFAppCore", "TUFFModelMetadata", "TUFFModelCatalog"], path: "Sources/TUFFModelPlan"),
        .executableTarget(
            name: "TUFFEnginePackage",
            dependencies: ["TUFFRuntimeManagement", "TUFFModelMetadata"],
            path: "Sources/TUFFEnginePackage"
        ),
        .target(
            name: "TUFFModelCatalog",
            path: "Sources/TUFFModelCatalog"
        ),
        .target(
            name: "TUFFFormat",
            path: "Sources/TUFFFormat"
        ),
        .target(
            name: "TUFFInferenceTypes",
            path: "Sources/TUFFInferenceTypes"
        ),
        .target(
            name: "TUFFRuntimeDescriptions",
            dependencies: ["TUFFModelCatalog"],
            path: "Sources/TUFFRuntimeDescriptions",
            swiftSettings: familySettings
        ),
        .target(
            name: "TUFFModelMetadata",
            dependencies: ["TUFFModelCatalog", "TUFFRuntimeDescriptions", "TUFFInferenceTypes", "TUFFFormat"],
            path: "Sources/TUFFModelMetadata"
        ),
        .target(
            name: "TUFFRuntimeManagement",
            dependencies: ["TUFFInferenceTypes", "TUFFModelCatalog", "TUFFModelMetadata", "TUFFRuntimeDescriptions"],
            path: "Sources/TUFFRuntimeManagement"
        ),
        .target(
            name: "TUFFEngine",
            dependencies: [
                "TUFFModelCatalog",
                "TUFFFormat",
                "TUFFInferenceTypes",
                "TUFFRuntimeDescriptions",
                "TUFFModelMetadata",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/TUFFEngine",
            exclude: engineFamily == "gpt-oss-mxfp4" ? ["Runtime/Inference/RealForwardRunner.swift"] : [],
            resources: [
                .copy("Metal"),
            ],
            swiftSettings: engineSettings
        ),
        .target(
            name: "TUFFRepackCore",
            dependencies: ["TUFFModelCatalog", "TUFFFormat", "TUFFRuntimeDescriptions", "TUFFModelMetadata", "TUFFInferenceTypes"],
            path: "Sources/TUFFRepack/Core"
        ),
        .executableTarget(
            name: "TUFFRepack",
            dependencies: ["TUFFRepackCore", "TUFFRuntimeManagement", "TUFFModelCatalog"],
            path: "Sources/TUFFRepack/Command"
        ),
        .target(
            name: "TUFFCLIArguments",
            dependencies: ["TUFFInferenceTypes", "TUFFRuntimeDescriptions"],
            path: "Sources/TUFFCLIArguments"
        ),
        .target(
            name: "TUFFCLIEngine",
            dependencies: ["TUFFCLIArguments", "TUFFCLICore", "TUFFEngine", "TUFFAppCore"],
            path: "Sources/TUFFCLIEngine"
        ),
        .target(
            name: "TUFFCLICore",
            dependencies: ["TUFFModelCatalog", "TUFFCLIArguments", "TUFFDecodeProtocol", "TUFFRuntimeManagement", "TUFFAppCore", "TUFFModelMetadata"],
            path: "Sources/TUFFCLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "TUFFCLI",
            dependencies: ["TUFFCLICore"],
            path: "Sources/TUFFCLI/Command"
        ),
        .target(
            name: "TUFFCommandCore",
            dependencies: ["TUFFModelCatalog"],
            path: "Sources/TUFFCommand/Core"
        ),
        .executableTarget(
            name: "TUFFCommand",
            dependencies: ["TUFFCommandCore"],
            path: "Sources/TUFFCommand/Command"
        ),
        .target(
            name: "TUFFAppCore",
            dependencies: ["TUFFModelCatalog", "TUFFRepackCore", "TUFFDecodeProtocol", "TUFFInferenceTypes", "TUFFRuntimeDescriptions", "TUFFModelMetadata", "TUFFRuntimeManagement"],
            path: "Sources/TUFFApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
                .copy("Resources/model-layouts.json"),
            ]
        ),
        .target(
            name: "TUFFEngineAdapter",
            dependencies: ["TUFFAppCore", "TUFFEngine", "TUFFRuntimeManagement"],
            path: "Sources/TUFFEngineAdapter"
        ),
        .target(
            name: "TUFFMacPresentation",
            dependencies: [
                "TUFFModelCatalog",
                "TUFFAppCore",
                .product(name: "SwiftMath", package: "SwiftMath"),
            ],
            path: "Sources/TUFFApp/MacPresentation"
        ),
        .target(
            name: "TUFFDecodeProtocol",
            dependencies: ["TUFFInferenceTypes"],
            path: "Sources/TUFFDecodeProtocol"
        ),
        .target(
            name: "TUFFDecodeServiceCore",
            dependencies: ["TUFFAppCore", "TUFFDecodeProtocol", "TUFFEngineAdapter", "TUFFRuntimeManagement"],
            path: "Sources/TUFFDecodeService",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "TUFFDecodeService",
            dependencies: ["TUFFDecodeServiceCore"],
            path: "Sources/TUFFDecodeService/Command"
        ),
        .executableTarget(
            name: "TUFFInferenceEngine",
            dependencies: ["TUFFCLIArguments", "TUFFCLIEngine", "TUFFDecodeProtocol",
                           "TUFFDecodeServiceCore", "TUFFInferenceTypes", "TUFFRuntimeManagement"],
            path: "Sources/TUFFInferenceEngine"
        ),
        .target(
            name: "TUFFServerCore",
            dependencies: [
                "TUFFModelCatalog",
                "TUFFAppCore",
                "TUFFDecodeProtocol",
                "TUFFInferenceTypes",
                "TUFFRuntimeDescriptions",
                "TUFFModelMetadata",
                "TUFFRuntimeManagement",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/TUFFServer/Core"
        ),
        .target(
            name: "TUFFServerEngine",
            dependencies: ["TUFFServerCore", "TUFFEngine", "TUFFDecodeProtocol", "TUFFRuntimeManagement"],
            path: "Sources/TUFFServerEngine"
        ),
        .target(
            name: "TUFFAppServer",
            dependencies: [
                "TUFFAppCore",
                "TUFFDecodeProtocol",
                "TUFFModelCatalog",
                "TUFFServerCore",
            ],
            path: "Sources/TUFFApp/Server"
        ),
        .target(
            name: "TUFFAppUpdater",
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/TUFFApp/Updater"
        ),
        .executableTarget(
            name: "TUFFServer",
            dependencies: ["TUFFServerCore"],
            path: "Sources/TUFFServer/Command"
        ),
        // A minimal GUI- or API-role serving client run as its own process by
        // the opt-in real-host tests, so process death can be observed. It is
        // not a product and is never packaged.
        .executableTarget(
            name: "TUFFServingProbe",
            dependencies: ["TUFFAppCore", "TUFFAppServer", "TUFFDecodeProtocol", "TUFFServerCore"],
            path: "Sources/TUFFServingProbe"
        ),
        .executableTarget(
            name: "TUFFMac",
            dependencies: [
                "TUFFAppCore",
                "TUFFAppServer",
                "TUFFAppUpdater",
                "TUFFMacPresentation",
                "TUFFInferenceTypes",
                "TUFFRuntimeDescriptions",
                "TUFFRuntimeManagement",
                "TUFFCLICore",
                "TUFFServerCore",
            ],
            path: "Sources/TUFFApp/Mac",
            resources: [
                .copy("Resources/tuff-app-icon.png"),
            ]
        ),
        .target(
            name: "TUFFValidationSupport",
            dependencies: ["TUFFEngine"],
            path: "Sources/TUFFValidation/Support"
        ),
        .testTarget(
            name: "TUFFInferenceTypesTests",
            dependencies: ["TUFFInferenceTypes", "TUFFDecodeProtocol", "TUFFRuntimeManagement", "TUFFRuntimeDescriptions"],
            path: "Tests/TUFFInferenceTypes"
        ),
        .testTarget(
            name: "TUFFModelCatalogTests",
            dependencies: ["TUFFModelCatalog"],
            path: "Tests/TUFFModelCatalog"
        ),
        .testTarget(
            name: "TUFFCommandTests",
            dependencies: ["TUFFCommandCore", "TUFFModelCatalog"],
            path: "Tests/TUFFCommand"
        ),
        .testTarget(
            name: "TUFFFormatTests",
            dependencies: ["TUFFFormat"],
            path: "Tests/TUFFFormat"
        ),
        .testTarget(
            name: "TUFFFormatCompatibilityTests",
            dependencies: ["TUFFFormat", "TUFFEngine", "TUFFRepackCore"],
            path: "Tests/TUFFFormatCompatibility",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TUFFTestsCore",
            dependencies: [
                "TUFFEngine",
                "TUFFValidationSupport",
                "TUFFRepackCore",
                "TUFFCLICore",
                "TUFFCLIEngine",
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Tests/TUFFEngine/Core",
            resources: [
                .copy("Tokenization/Fixtures"),
                .copy("Runtime/Vision/Fixtures/images"),
                .copy("Kernels/Fixtures/qwen4exp"),
            ]
        ),
        .testTarget(
            name: "TUFFRepackTests",
            dependencies: ["TUFFFormat", "TUFFRepackCore"],
            path: "Tests/TUFFRepack/Core",
            resources: [.copy("Fixtures/NewModels")]
        ),
        .testTarget(
            name: "TUFFAppCoreTests",
            dependencies: [
                "TUFFModelCatalog",
                "TUFFAppCore",
                "TUFFEngine",
                "TUFFEngineAdapter",
                "TUFFRepackCore",
                "TUFFDecodeProtocol",
            ],
            path: "Tests/TUFFApp/Core"
        ),
        .testTarget(
            name: "TUFFDecodeServiceTests",
            dependencies: ["TUFFDecodeServiceCore", "TUFFAppCore", "TUFFDecodeProtocol", "TUFFEngineAdapter"],
            path: "Tests/TUFFDecodeService"
        ),
        .testTarget(
            name: "TUFFMacPresentationTests",
            dependencies: [
                "TUFFModelCatalog",
                "TUFFAppCore",
                "TUFFAppServer",
                "TUFFAppUpdater",
                "TUFFMac",
                "TUFFMacPresentation",
                .product(name: "SwiftMath", package: "SwiftMath"),
            ],
            path: "Tests/TUFFApp/MacPresentation"
        ),
        .testTarget(
            name: "TUFFServerTests",
            dependencies: [
                "TUFFServerCore",
                "TUFFServerEngine",
                "TUFFEngine",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "Tests/TUFFServer",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TUFFAppServerTests",
            dependencies: [
                "TUFFAppServer",
                "TUFFAppCore",
                "TUFFDecodeProtocol",
                "TUFFServerCore",
                "TUFFModelCatalog",
            ],
            path: "Tests/TUFFAppServer"
        ),
        .testTarget(
            name: "TUFFAppUpdaterTests",
            dependencies: ["TUFFAppUpdater"],
            path: "Tests/TUFFAppUpdater"
        ),
    ]
)
