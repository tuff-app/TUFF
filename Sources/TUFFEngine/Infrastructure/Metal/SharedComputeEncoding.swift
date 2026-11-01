import Foundation
import Metal
import ObjectiveC

/// Opt-in per command buffer: every primitive in a marked command buffer that
/// asks for a compute encoder receives one serial encoder that stays open for
/// the whole buffer, instead of a fresh encoder per dispatch.
///
/// A serial compute encoder runs its dispatches in order with their writes
/// visible to later dispatches, so primitives written as one-encoder-per-op
/// keep their semantics. What changes is the GPU-side cost of the several
/// hundred encoder boundaries in a decode token: each boundary otherwise
/// drains and restarts the compute pipeline.
///
/// Mechanism: the concrete command-buffer and encoder classes' `computeCommandEncoder`,
/// `endEncoding`, `blitCommandEncoder` and `commit` implementations are wrapped
/// once. Unmarked command buffers take the original code path unchanged. The
/// shared encoder is ended exactly once, by `commit` (or a blit request).
public enum SharedComputeEncoding {
    /// `TUFF_SHARED_ENCODER=0` restores one encoder per primitive.
    public static let enabled: Bool = ProcessInfo.processInfo.environment["TUFF_SHARED_ENCODER"] != "0"
        && ProcessInfo.processInfo.environment["TUFF_GPU_OP_PROFILE"] != "1"

    /// Mark `commandBuffer` before any primitive encodes into it.
    public static func share(_ commandBuffer: MTLCommandBuffer) {
        guard enabled else { return }
        state.install(sampleBuffer: commandBuffer)
        state.mark(commandBuffer)
    }

    private static let state = State()

    private final class State: @unchecked Sendable {
        private typealias MakeCompute = @convention(c) (AnyObject, Selector) -> AnyObject?
        private typealias Void0 = @convention(c) (AnyObject, Selector) -> Void
        private typealias MakeBlit = @convention(c) (AnyObject, Selector) -> AnyObject?

        private let lock = NSLock()
        private var installed = false
        private var encoderClassSwizzled = false
        private var marked: [ObjectIdentifier: AnyObject] = [:]
        private var open: [ObjectIdentifier: AnyObject] = [:]
        private var openEncoders = Set<ObjectIdentifier>()
        private var originalEnd: Void0?
        private var originalMakeCompute: MakeCompute?
        private var originalCommit: Void0?
        private var originalMakeBlit: MakeBlit?
        private let computeSelector = NSSelectorFromString("computeCommandEncoder")
        private let blitSelector = NSSelectorFromString("blitCommandEncoder")
        private let commitSelector = NSSelectorFromString("commit")
        private let endSelector = NSSelectorFromString("endEncoding")

        func mark(_ commandBuffer: MTLCommandBuffer) {
            lock.lock()
            marked[ObjectIdentifier(commandBuffer)] = commandBuffer
            lock.unlock()
        }

        func install(sampleBuffer commandBuffer: MTLCommandBuffer) {
            lock.lock()
            defer { lock.unlock() }
            guard !installed, let cls = object_getClass(commandBuffer) else { return }
            guard let compute = class_getInstanceMethod(cls, computeSelector),
                  let blit = class_getInstanceMethod(cls, blitSelector),
                  let commit = class_getInstanceMethod(cls, commitSelector) else {
                fatalError("SharedComputeEncoding: unexpected command buffer class \(cls)")
            }
            let makeCompute = unsafeBitCast(method_getImplementation(compute), to: MakeCompute.self)
            let makeBlit = unsafeBitCast(method_getImplementation(blit), to: MakeBlit.self)
            let realCommit = unsafeBitCast(method_getImplementation(commit), to: Void0.self)
            originalMakeCompute = makeCompute
            originalMakeBlit = makeBlit
            originalCommit = realCommit
            let computeSelector = self.computeSelector, blitSelector = self.blitSelector
            let commitSelector = self.commitSelector

            let computeBlock: @convention(block) (AnyObject) -> AnyObject? = { [unowned self] buffer in
                self.lock.lock()
                let id = ObjectIdentifier(buffer)
                if self.marked[id] != nil {
                    if let encoder = self.open[id] {
                        self.lock.unlock()
                        return encoder
                    }
                    self.lock.unlock()
                    guard let encoder = makeCompute(buffer, computeSelector) else { return nil }
                    self.registerOpenEncoder(encoder, for: id)
                    return encoder
                }
                self.lock.unlock()
                return makeCompute(buffer, computeSelector)
            }
            method_setImplementation(compute, imp_implementationWithBlock(computeBlock))

            let blitBlock: @convention(block) (AnyObject) -> AnyObject? = { [unowned self] buffer in
                self.closeOpenEncoder(of: ObjectIdentifier(buffer))
                return makeBlit(buffer, blitSelector)
            }
            method_setImplementation(blit, imp_implementationWithBlock(blitBlock))

            let commitBlock: @convention(block) (AnyObject) -> Void = { [unowned self] buffer in
                let id = ObjectIdentifier(buffer)
                self.closeOpenEncoder(of: id)
                self.lock.lock()
                self.marked[id] = nil
                self.lock.unlock()
                realCommit(buffer, commitSelector)
            }
            method_setImplementation(commit, imp_implementationWithBlock(commitBlock))
            installed = true
        }

        private func registerOpenEncoder(_ encoder: AnyObject, for id: ObjectIdentifier) {
            lock.lock()
            open[id] = encoder
            openEncoders.insert(ObjectIdentifier(encoder))
            let needsSwizzle = !encoderClassSwizzled
            encoderClassSwizzled = true
            lock.unlock()
            guard needsSwizzle, let cls = object_getClass(encoder),
                  let end = class_getInstanceMethod(cls, endSelector) else { return }
            let realEnd = unsafeBitCast(method_getImplementation(end), to: Void0.self)
            lock.lock()
            originalEnd = realEnd
            lock.unlock()
            let endSelector = self.endSelector
            let endBlock: @convention(block) (AnyObject) -> Void = { [unowned self] encoder in
                self.lock.lock()
                let shared = self.openEncoders.contains(ObjectIdentifier(encoder))
                self.lock.unlock()
                if !shared { realEnd(encoder, endSelector) }
            }
            method_setImplementation(end, imp_implementationWithBlock(endBlock))
        }

        private func closeOpenEncoder(of id: ObjectIdentifier) {
            lock.lock()
            guard let encoder = open.removeValue(forKey: id) else {
                lock.unlock()
                return
            }
            openEncoders.remove(ObjectIdentifier(encoder))
            let realEnd = originalEnd
            lock.unlock()
            realEnd?(encoder, endSelector)
        }
    }
}
