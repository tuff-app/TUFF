import Foundation
import Metal

enum MetalError: Error, CustomStringConvertible {
    case noDevice
    case noQueue
    case missingShaderResource(String)
    case missingFunction(String)
    case libraryCompileFailed(String)
    case commandBufferFailed(String)

    public var description: String {
        switch self {
        case .noDevice:                   return "No Metal device"
        case .noQueue:                    return "Failed to create Metal command queue"
        case .missingShaderResource(let n): return "Shader resource missing: \(n)"
        case .missingFunction(let n):     return "Metal function missing in library: \(n)"
        case .libraryCompileFailed(let s):return "Metal library compile failed: \(s)"
        case .commandBufferFailed(let s): return "Metal command buffer failed: \(s)"
        }
    }
}

func metalCommandBufferStatusName(_ status: MTLCommandBufferStatus) -> String {
    switch status {
    case .notEnqueued: return "notEnqueued"
    case .enqueued:    return "enqueued"
    case .committed:   return "committed"
    case .scheduled:   return "scheduled"
    case .completed:   return "completed"
    case .error:       return "error"
    @unknown default:  return "unknown(\(status.rawValue))"
    }
}

/// The diagnostic for a command buffer that did not complete, or nil when it
/// did. Status is checked on its own because a failed buffer is not
/// guaranteed to carry an error object. `userInfo` values are not included:
/// only their keys, so a driver payload cannot leak into a chat error.
func metalCommandBufferFailureDetail(label: String?,
                                     status: MTLCommandBufferStatus,
                                     error: (any Error)?) -> String? {
    if status == .completed && error == nil { return nil }

    var parts = ["label=\(label.map { $0.isEmpty ? "<empty>" : $0 } ?? "<none>")"]
    parts.append("status=\(metalCommandBufferStatusName(status))")
    if let error {
        let nsError = error as NSError
        parts.append("domain=\(nsError.domain)")
        parts.append("code=\(nsError.code)")
        parts.append("description=\(nsError.localizedDescription)")
        if !nsError.userInfo.isEmpty {
            parts.append("userInfoKeys=\(nsError.userInfo.keys.sorted().joined(separator: ","))")
        }
    } else {
        parts.append("error=<none>")
    }
    return parts.joined(separator: " ")
}

/// Call after `waitUntilCompleted()`. Throws when the buffer failed, naming
/// its label so a long prefill that the GPU watchdog killed says which layer
/// and phase it was in.
func checkCommandBufferError(_ commandBuffer: MTLCommandBuffer) throws {
    guard let detail = metalCommandBufferFailureDetail(label: commandBuffer.label,
                                                       status: commandBuffer.status,
                                                       error: commandBuffer.error) else {
        return
    }
    throw MetalError.commandBufferFailed(detail)
}

public struct MetalFunctionConstant: Hashable, Sendable {
    public enum Value: Hashable, Sendable {
        case bool(Bool)
        case uint32(UInt32)
        case float(Float)
    }

    public let index: Int
    public let value: Value

    public init(index: Int, value: Value) {
        self.index = index
        self.value = value
    }
}

/// A set of shader modules compiled into one library. A model loads only the
/// groups its architecture uses: every model needs `core`, a dense Gemma does
/// not need the mixture-of-experts, GPT-OSS, Gated-DeltaNet or vision kernels.
///
/// Every group is compiled after the shared prelude (`quant_group`, which
/// declares the function constants, and `activation`), so each group is a
/// complete translation unit on its own. `supportModules` carries helper source
/// a group's kernels call; their kernels still belong to their own group.
public enum MetalKernelGroup: String, CaseIterable, Sendable, Comparable {
    case core
    case mixtureOfExperts = "moe"
    case int8
    case mxfp4
    case linearAttention = "gdn"
    case sparseAttention = "qsa"
    case residualStreams = "streams"
    case prismHadamard = "prism"
    case flint
    case vision

    /// Modules whose kernels this group provides, in compile order.
    public var modules: [String] {
        switch self {
        case .core:
            // per_layer_embedding also holds the generic scale and residual
            // kernels every family uses, so it is core rather than Gemma-only.
            ["dequant_int4", "rmsnorm", "rope", "attention", "logit", "utility",
             "per_layer_embedding", "fused", "prefill"]
        case .mixtureOfExperts: ["moe"]
        case .int8: ["dequant_int8"]
        case .mxfp4: ["mxfp4"]
        case .linearAttention: ["gdn"]
        case .sparseAttention: ["qsa"]
        case .residualStreams: ["hyper_connection", "ngram_ple"]
        case .prismHadamard: ["prism_hadamard", "prism_affine2_gemm"]
        case .flint: ["flint"]
        case .vision: ["vision"]
        }
    }

    /// Modules compiled ahead of `modules` because their helpers are called.
    var supportModules: [String] {
        switch self {
        // gdn's input projection reuses dequant_int4_gemv_simd_body.
        case .linearAttention: ["dequant_int4"]
        // Flint's batched expert kernels reuse the decode expert row helpers.
        case .flint: ["moe"]
        default: []
        }
    }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        allCases.firstIndex(of: lhs)! < allCases.firstIndex(of: rhs)!
    }

    /// The groups a model of this architecture dispatches. Selection follows
    /// the configuration's features rather than model names, so a new
    /// checkpoint of an existing architecture needs no change here.
    public static func required(for config: ArchConfig,
                                sharedExpertWeightBits: Int = 4,
                                maxContext: Int = .max) -> Set<MetalKernelGroup> {
        var groups: Set<MetalKernelGroup> = [.core]
        if config.family == .gptOss {
            groups.formUnion([.mixtureOfExperts, .mxfp4])
            return groups
        }
        if config.numExperts > 0 { groups.insert(.mixtureOfExperts) }
        if config.family == .flint { groups.insert(.flint) }
        if config.variant == .bonsai2_2B || config.variant == .bonsai2_27B {
            groups.insert(.prismHadamard)
        }
        if sharedExpertWeightBits == 8 || config.sharedExpertGated { groups.insert(.int8) }
        if config.hasLinearAttentionLayers { groups.insert(.linearAttention) }
        if config.attentionIndexer.isEnabled,
           maxContext >= config.attentionIndexer.budget + config.attentionIndexer.compressRatio {
            groups.insert(.sparseAttention)
        }
        if config.hyperConnection.isEnabled || config.ngramEmbedding.isEnabled {
            groups.insert(.residualStreams)
        }
        return groups
    }
}

/// Single owner of the `MTLDevice`, queue, and the runtime-compiled shader
/// libraries. We ship `.metal` source files as bundle resources and compile
/// them at runtime, which keeps the dev loop fast: edit a shader, rebuild the
/// Swift target, no Xcode metallib step.
///
/// Libraries are compiled per `MetalKernelGroup`. A runner prepares the groups
/// its architecture needs when it is created; a pipeline whose group is not
/// compiled yet compiles that group on demand and records it as a late load,
/// so a missing prediction costs time, never correctness. Setting
/// `TUFF_KERNEL_GROUPS=combined` compiles every module into one library, as
/// releases before 8.0 did.
///
/// `@unchecked Sendable`: device and queue are immutable, Metal objects are
/// thread-safe for encoding, and the library and pipeline caches are
/// lock-guarded.
public final class MetalContext: @unchecked Sendable {
    public let device:  MTLDevice
    public let queue:   MTLCommandQueue

    /// Group libraries compiled so far. In combined mode every group maps to
    /// the same library.
    private var libraries: [MetalKernelGroup: MTLLibrary] = [:]
    private var lateGroups: Set<MetalKernelGroup> = []
    private var compileSeconds: [MetalKernelGroup: Double] = [:]
    private let libraryLock = NSLock()
    public let combinesAllGroups: Bool
    private let logsKernelGroups: Bool

    public var compiledKernelGroups: Set<MetalKernelGroup> {
        libraryLock.withLock { Set(libraries.keys) }
    }

    /// Groups compiled because a pipeline asked for them rather than because
    /// a runner prepared them. Empty when predictions are complete.
    public var lateKernelGroups: Set<MetalKernelGroup> {
        libraryLock.withLock { lateGroups }
    }

    public var kernelGroupCompileSeconds: [MetalKernelGroup: Double] {
        libraryLock.withLock { compileSeconds }
    }

    /// Pipelines created and the time spent creating them, for load
    /// diagnostics and the kernel-loading benchmark.
    public var pipelineStatistics: (count: Int, seconds: Double) {
        pipelineCacheLock.withLock { (pipelineCache.count, pipelineSeconds) }
    }
    private var pipelineSeconds = 0.0

    private struct PipelineCacheKey: Hashable {
        var name: String
        var constants: [MetalFunctionConstant]
        var maxTotalThreadsPerThreadgroup: Int?
    }

    private var pipelineCache: [PipelineCacheKey: MTLComputePipelineState] = [:]
    private let pipelineCacheLock = NSLock()

    /// Environment variable the AGX driver reads once, at the first device
    /// creation in a process.
    static let interactivityWatchdogVariable = "AGX_RELAX_CDM_CTXSTORE_TIMEOUT"

    /// Long prefill dispatches (full attention over tens of thousands of
    /// keys) can otherwise be killed with `ImpactingInteractivity` while a
    /// display is active. This relaxes the deadline; it does not guarantee a
    /// dispatch survives. An operator who sets the variable, including to 0
    /// for stock behavior, is not overridden.
    public static func relaxInteractivityWatchdog() {
        #if os(macOS)
        setenv(interactivityWatchdogVariable, "1", 0)
        #endif
    }

    /// Every production device creation goes through here so the watchdog
    /// mitigation is in place before the driver reads it.
    public static func makeSystemDefaultDevice() -> MTLDevice? {
        relaxInteractivityWatchdog()
        return MTLCreateSystemDefaultDevice()
    }

    /// `kernelGroups` are compiled now; others compile when first used.
    public init(kernelGroups: Set<MetalKernelGroup> = [.core],
                environment: [String: String] = ProcessInfo.processInfo.environment) throws {
        guard let dev = Self.makeSystemDefaultDevice() else { throw MetalError.noDevice }
        guard let q   = dev.makeCommandQueue()           else { throw MetalError.noQueue }
        self.device  = dev
        self.queue   = q
        self.combinesAllGroups = environment["TUFF_KERNEL_GROUPS"] == "combined"
        self.logsKernelGroups = environment["TUFF_LOG_KERNELS"] != nil
        try prepareKernelGroups(kernelGroups)
    }

    /// Compiles any of `groups` not compiled yet.
    public func prepareKernelGroups(_ groups: Set<MetalKernelGroup>) throws {
        try libraryLock.withLock {
            for group in groups.sorted() where libraries[group] == nil {
                try compileLocked(group)
            }
        }
    }

    /// Caller holds `libraryLock`.
    private func compileLocked(_ group: MetalKernelGroup) throws {
        let start = ContinuousClock.now
        if combinesAllGroups {
            let library = try Self.compileShaderLibrary(device: device, modules: Self.shaderModules)
            for every in MetalKernelGroup.allCases { libraries[every] = library }
        } else {
            libraries[group] = try Self.compileShaderLibrary(
                device: device,
                modules: Self.preludeModules + group.supportModules + group.modules)
        }
        let elapsed = ContinuousClock.now - start
        compileSeconds[group] = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        if logsKernelGroups {
            FileHandle.standardError.write(Data(String(
                format: "[kernels] compiled %@%@ in %.3fs\n", group.rawValue,
                combinesAllGroups ? " (combined)" : "", compileSeconds[group]!).utf8))
        }
    }

    /// The library that defines `name`, compiling its group if needed.
    private func library(containing name: String) throws -> MTLLibrary {
        try libraryLock.withLock {
            if let group = Self.functionGroups[name] {
                if let library = libraries[group] { return library }
                try compileLocked(group)
                lateGroups.insert(group)
                if logsKernelGroups {
                    FileHandle.standardError.write(Data(
                        "[kernels] \(group.rawValue) was not prepared; compiled for \(name)\n".utf8))
                }
                return libraries[group]!
            }
            // Not in the index: look in what is compiled, then compile the
            // rest before giving up.
            if let library = libraries.values.first(where: { $0.functionNames.contains(name) }) {
                return library
            }
            for group in MetalKernelGroup.allCases where libraries[group] == nil {
                try compileLocked(group)
                if libraries[group]!.functionNames.contains(name) {
                    lateGroups.insert(group)
                    return libraries[group]!
                }
            }
            throw MetalError.missingFunction(name)
        }
    }

    /// Writes the groups compiled and the pipelines created so far when
    /// `TUFF_LOG_KERNELS` is set, after a model has loaded.
    public func logLoadStatistics(_ label: String) {
        guard logsKernelGroups else { return }
        let pipelines = pipelineStatistics
        let groups = compiledKernelGroups.sorted().map(\.rawValue).joined(separator: ",")
        FileHandle.standardError.write(Data(String(
            format: "[kernels] %@ loaded: groups=%@ pipelines=%d in %.3fs\n",
            label, groups, pipelines.count, pipelines.seconds).utf8))
    }

    /// An unspecialized function, for argument encoders.
    public func function(named name: String) throws -> MTLFunction {
        guard let function = try library(containing: name).makeFunction(name: name) else {
            throw MetalError.missingFunction(name)
        }
        return function
    }

    /// Which group's library defines each kernel, read from the shipped
    /// sources. A kernel is listed under the group that owns its module, not a
    /// group that compiles the module only for its helpers.
    static let functionGroups: [String: MetalKernelGroup] = {
        var index: [String: MetalKernelGroup] = [:]
        for group in MetalKernelGroup.allCases {
            for module in group.modules {
                guard let url = shaderURL(module: module),
                      let source = try? String(contentsOf: url, encoding: .utf8) else { continue }
                for name in kernelNames(in: source) { index[name] = group }
            }
        }
        return index
    }()

    static func kernelNames(in source: String) -> Set<String> {
        let patterns = [#"\bkernel\s+void\s+(\w+)"#,
                        #"\[\[\s*kernel[^\]]*\]\]\s*void\s+(\w+)"#,
                        #"host_name\(\s*"(\w+)"\s*\)"#]
        var names = Set<String>()
        for pattern in patterns {
            let regex = try! NSRegularExpression(pattern: pattern)
            for match in regex.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
                if let range = Range(match.range(at: 1), in: source) { names.insert(String(source[range])) }
            }
        }
        return names
    }

    /// Declares the function constants the quantized kernels share. Compiled
    /// first into the shared library, and prepended to any private library
    /// whose module reads them.
    private static let quantConstantsModule = "quant_group"

    /// Compiled ahead of every group: the shared function constants and the
    /// activation math more than one group calls.
    static let preludeModules = ["quant_group", "activation"]

    /// Production shader modules compiled into the shared runtime library.
    private static let shaderModules: [String] = [
        // First: it declares the function constants the quantized kernels
        // share, and the library is one concatenated translation unit.
        "quant_group",
        "activation",
        "dequant_int4",
        "dequant_int8",
        "mxfp4",
        "rmsnorm",
        "rope",
        "attention",
        "qsa",
        "moe",
        "logit",
        "utility",
        "per_layer_embedding",
        "hyper_connection",
        "ngram_ple",
        "prism_hadamard",
        "prism_affine2_gemm",
        "flint",
        "fused",
        "prefill",
        "gdn",
        "vision",
    ]

    /// Bundle locations for runtime shader modules.
    private static let shaderSubdirectories: [String: String] = [
        "attention": "Metal/Attention",
        "qsa": "Metal/Attention",
        "dequant_int4": "Metal/Quant",
        "dequant_int8": "Metal/Quant",
        "mxfp4": "Metal/Quant",
        "quant_group": "Metal/Quant",
        "activation": "Metal/Primitives",
        "fused": "Metal/Fusions",
        "gdn": "Metal/GDN",
        "logit": "Metal/Sampling",
        "moe": "Metal/MoE",
        "prefill": "Metal/Prefill",
        "hyper_connection": "Metal/Primitives",
        "ngram_ple": "Metal/Primitives",
        "per_layer_embedding": "Metal/Primitives",
        "prism_hadamard": "Metal/Primitives",
        "prism_affine2_gemm": "Metal/Primitives",
        "flint": "Metal/Flint",
        "rmsnorm": "Metal/Primitives",
        "rope": "Metal/Primitives",
        "tensorops": "Metal/TensorCore",
        "utility": "Metal/Primitives",
        "vision": "Metal/Vision",
        "vision_register_gemm": "Metal/Vision",
        "vision_resize": "Metal/Vision",
    ]

    private static let shaderBundle: Bundle = {
        for resources in packagedResourceDirectories(
            mainResourceURL: Bundle.main.resourceURL,
            executableURL: Bundle.main.executableURL) {
            if let packaged = Bundle(
                url: resources.appendingPathComponent(
                    "TUFF_TUFFEngine.bundle",
                    isDirectory: true
                )
            ) {
                return packaged
            }
        }
        // SwiftPM's accessor falls back to the absolute build directory and
        // traps when that is absent, so it is reached only from a clone.
        return Bundle.module
    }()

    /// Where a packaged engine bundle can sit. The app finds it in its own
    /// `Contents/Resources`; the command-line tools shipped in
    /// `Contents/Resources/bin` are bare executables whose main bundle is
    /// `bin` itself, so their resources are one directory up.
    static func packagedResourceDirectories(mainResourceURL: URL?,
                                            executableURL: URL?) -> [URL] {
        var directories: [URL] = []
        if let mainResourceURL { directories.append(mainResourceURL) }
        if let executableURL {
            let executableDirectory = executableURL.deletingLastPathComponent()
            if executableDirectory.lastPathComponent == "bin" {
                directories.append(executableDirectory.deletingLastPathComponent())
            }
        }
        return directories
    }

    private static func shaderURL(module: String) -> URL? {
        guard let subdirectory = shaderSubdirectories[module] else { return nil }
        return shaderBundle.url(forResource: module, withExtension: "metal",
                                subdirectory: subdirectory)
    }

    private static func compileShaderLibrary(device: MTLDevice,
                                             modules: [String]) throws -> MTLLibrary {
        var combined = ""
        for name in modules {
            guard let url = shaderURL(module: name) else {
                throw MetalError.missingShaderResource(name)
            }
            let src = try String(contentsOf: url, encoding: .utf8)
            combined += "\n// ==== \(name).metal ====\n" + src + "\n"
        }
        do {
            let opts = makeCompileOptions()
            return try device.makeLibrary(source: combined, options: opts)
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
    }

    /// Compile one shader module into its own library, leaving the shared
    /// runtime library untouched.
    ///
    /// Cached per device, module, math mode and source variant: libraries are
    /// immutable, and
    /// one `VisionRuntime` init otherwise compiles the identical tensorops
    /// source three times (linear, attention, projector) on every load.
    public static func privateLibrary(device: MTLDevice, module: String,
                                      mathMode: MTLMathMode? = nil,
                                      includeVisionTensorOps: Bool = false) throws
        -> MTLLibrary {
        let key = "\(ObjectIdentifier(device).hashValue)#\(module)"
            + "#\(mathMode?.rawValue ?? -1)#\(includeVisionTensorOps)"
        privateLibraryLock.lock()
        defer { privateLibraryLock.unlock() }
        if let cached = privateLibraryCache[key] {
            return cached
        }
        guard let url = shaderURL(module: module) else {
            throw MetalError.missingShaderResource(module)
        }
        // A private library is one module compiled on its own, so it does not
        // get the shared library's ordering. Modules that read the quantized
        // function constants need the prelude that declares them prepended.
        var src = try String(contentsOf: url, encoding: .utf8)
        if src.contains("quant_group_size()") {
            guard let preludeURL = shaderURL(module: quantConstantsModule) else {
                throw MetalError.missingShaderResource(quantConstantsModule)
            }
            src = try String(contentsOf: preludeURL, encoding: .utf8) + "\n" + src
        }
        let opts = makeCompileOptions(
            mathMode: mathMode,
            includeVisionTensorOps: includeVisionTensorOps)
        do {
            let library = try device.makeLibrary(source: src, options: opts)
            privateLibraryCache[key] = library
            return library
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
    }

    private static func makeCompileOptions(
        mathMode: MTLMathMode? = nil,
        includeVisionTensorOps: Bool = false
    ) -> MTLCompileOptions {
        let options = MTLCompileOptions()
        if #available(macOS 26.0, iOS 26.0, *) {
            // MSL 4 exposes the optional MPP tensor paths. The same sources
            // guard those kernels with __HAVE_TENSOR__ so older systems compile
            // the production Metal 3.2 fallbacks instead.
            options.languageVersion = .version4_0
        } else {
            options.languageVersion = .version3_2
        }
        if let mathMode {
            options.mathMode = mathMode
        }
        if includeVisionTensorOps {
            options.preprocessorMacros = [
                "TUFF_VISION_TENSOROPS": NSNumber(value: true)
            ]
        }
        return options
    }

    private static let privateLibraryLock = NSLock()
    private static nonisolated(unsafe) var privateLibraryCache: [String: MTLLibrary] = [:]

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        try pipeline(name, constants: [])
    }

    public func pipeline(_ name: String,
                         constants: [MetalFunctionConstant]) throws -> MTLComputePipelineState {
        try pipeline(name, constants: constants, maxTotalThreadsPerThreadgroup: nil)
    }

    public func pipeline(_ name: String,
                         constants: [MetalFunctionConstant],
                         maxTotalThreadsPerThreadgroup hint: Int?) throws -> MTLComputePipelineState {
        if let hint {
            precondition(hint > 0, "maxTotalThreadsPerThreadgroup must be positive")
        }
        let sortedConstants = constants.sorted {
            if $0.index != $1.index { return $0.index < $1.index }
            return Self.constantSortKey($0.value) < Self.constantSortKey($1.value)
        }
        let key = PipelineCacheKey(name: name,
                                   constants: sortedConstants,
                                   maxTotalThreadsPerThreadgroup: hint)
        pipelineCacheLock.lock()
        let cached = pipelineCache[key]
        pipelineCacheLock.unlock()
        if let cached { return cached }

        let library = try library(containing: name)
        guard library.functionNames.contains(name) else {
            throw MetalError.missingFunction(name)
        }
        let started = ContinuousClock.now

        let values = MTLFunctionConstantValues()
        for constant in sortedConstants {
            switch constant.value {
            case .bool(let value):
                var v = value
                values.setConstantValue(&v, type: .bool, index: constant.index)
            case .uint32(let value):
                var v = value
                values.setConstantValue(&v, type: .uint, index: constant.index)
            case .float(let value):
                var v = value
                values.setConstantValue(&v, type: .float, index: constant.index)
            }
        }

        let fn = try library.makeFunction(name: name, constantValues: values)
        let p: MTLComputePipelineState
        if let hint {
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = fn
            descriptor.maxTotalThreadsPerThreadgroup = hint
            var reflection: MTLAutoreleasedComputePipelineReflection?
            p = try device.makeComputePipelineState(descriptor: descriptor,
                                                    options: [],
                                                    reflection: &reflection)
        } else {
            p = try device.makeComputePipelineState(function: fn)
        }
        let elapsed = ContinuousClock.now - started
        pipelineCacheLock.lock()
        pipelineCache[key] = p
        GPUOpProfiler.registerPipeline(p, name: name)
        pipelineSeconds += Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        pipelineCacheLock.unlock()
        return p
    }

    private static func constantSortKey(_ value: MetalFunctionConstant.Value) -> String {
        switch value {
        case .bool(let v):   return "b:\(v ? 1 : 0)"
        case .uint32(let v): return "u:\(v)"
        case .float(let v):  return "f:\(v.bitPattern)"
        }
    }
}
