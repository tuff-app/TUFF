#if os(macOS)
import Foundation
import Testing
@testable import TUFFAppCore

private let registryName = "registry.example/tuff-sandbox"

private func registryPin(digest: String = FakeContainerWorld.digest) -> SandboxImagePin {
    SandboxImagePin(reference: registryName, digest: digest, source: .registry)
}

private func localPin(digest: String = FakeContainerWorld.digest) -> SandboxImagePin {
    SandboxImagePin(reference: "tuff-sandbox:dev", digest: digest, source: .local)
}

private func makeProvisioner(_ world: FakeContainerWorld) -> SandboxImageProvisioner {
    SandboxImageProvisioner(tool: ContainerTool(
        runner: world, path: FakeContainerWorld.containerPath, isExecutable: { _ in true }))
}

/// The `SandboxError` a call throws, or nil when it does not throw.
private func thrownError(_ body: () async throws -> Void) async -> SandboxError? {
    do {
        try await body()
    } catch let error as SandboxError {
        return error
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
    return nil
}

private func commandNames(_ world: FakeContainerWorld) -> [String] {
    world.calls.map { FakeContainerWorld.name(of: $0) }
}

private final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    func add(_ message: String) { lock.withLock { items.append(message) } }
    var messages: [String] { lock.withLock { items } }
}

@Suite struct SandboxImagePinTests {
    @Test func aDigestIsSha256AndSixtyFourLowercaseHexDigits() {
        #expect(SandboxImagePin.isWellFormed(FakeContainerWorld.digest))
        #expect(SandboxImagePin.isWellFormed(FakeContainerWorld.otherDigest))
        #expect(SandboxImagePin.isWellFormed("sha256:" + String(repeating: "0123456789abcdef", count: 4)))
        #expect(SandboxImagePin.isWellFormed(SandboxImagePin.placeholderDigest))
    }

    @Test func malformedDigestsAreRejected() {
        let hex = String(repeating: "a", count: 64)
        #expect(!SandboxImagePin.isWellFormed(""))
        #expect(!SandboxImagePin.isWellFormed("sha256:"))
        #expect(!SandboxImagePin.isWellFormed(hex))
        #expect(!SandboxImagePin.isWellFormed("sha512:" + hex))
        #expect(!SandboxImagePin.isWellFormed("SHA256:" + hex))
        #expect(!SandboxImagePin.isWellFormed("sha256:" + String(repeating: "a", count: 63)))
        #expect(!SandboxImagePin.isWellFormed("sha256:" + String(repeating: "a", count: 65)))
        #expect(!SandboxImagePin.isWellFormed("sha256:" + String(repeating: "A", count: 64)))
        #expect(!SandboxImagePin.isWellFormed("sha256:" + String(repeating: "g", count: 64)))
        #expect(!SandboxImagePin.isWellFormed("sha256:" + hex + "\n"))
        #expect(!SandboxImagePin.isWellFormed(" sha256:" + hex))
        #expect(!SandboxImagePin.isWellFormed("sha256:" + hex + " "))
        #expect(!SandboxImagePin.isWellFormed("x" + "sha256:" + hex))
    }

    @Test func thePlaceholderIsSha256AndSixtyFourZeros() {
        #expect(SandboxImagePin.placeholderDigest == "sha256:" + String(repeating: "0", count: 64))
    }

    @Test func placeholderDetectionIsExact() {
        #expect(registryPin(digest: SandboxImagePin.placeholderDigest).isPlaceholder)
        #expect(localPin(digest: SandboxImagePin.placeholderDigest).isPlaceholder)
        #expect(!registryPin().isPlaceholder)
        #expect(!registryPin(digest: "sha256:" + String(repeating: "0", count: 63) + "1").isPlaceholder)
    }

    @Test func theReleasePinIsThePublishedImage() {
        #expect(!SandboxImagePin.release.isPlaceholder)
        #expect(SandboxImagePin.release.digest == "sha256:34e950fac80834e569c3781b233b3f0ea21e2cd0be646fcdd04225db48b99528")
        #expect(SandboxImagePin.release.source == .registry)
        #expect(SandboxImagePin.release.reference == "ghcr.io/tuff-app/tuff-sandbox")
    }

    @Test func theReleasePinIsWellFormedSoPublishingIsOnlyADigestChange() {
        #expect(SandboxImagePin.isWellFormed(SandboxImagePin.release.digest))
    }

    @Test func aRegistryImageIsRunByReferenceAndDigest() {
        let pin = registryPin()
        #expect(pin.runReference == "\(registryName)@\(FakeContainerWorld.digest)")
    }

    @Test func aLocalImageIsRunByReferenceAlone() {
        let pin = localPin()
        #expect(pin.runReference == "tuff-sandbox:dev")
        #expect(!pin.runReference.contains("@"))
    }

    @Test func pinsCompareByAllFields() {
        #expect(registryPin() == registryPin())
        #expect(registryPin() != registryPin(digest: FakeContainerWorld.otherDigest))
        #expect(registryPin() != SandboxImagePin(
            reference: registryName, digest: FakeContainerWorld.digest, source: .local))
        #expect(SandboxImagePin.Source.registry != SandboxImagePin.Source.local)
    }
}

@Suite struct SandboxImageDigestParsingTests {
    @Test func findsEveryDigestInInspectOutput() {
        let output = FakeContainerWorld.inspectOutput([FakeContainerWorld.digest, FakeContainerWorld.otherDigest])
        #expect(SandboxImageProvisioner.digests(in: output)
                == [FakeContainerWorld.digest, FakeContainerWorld.otherDigest])
    }

    @Test func repeatedDigestsCountOnce() {
        let text = "\(FakeContainerWorld.digest) and again \(FakeContainerWorld.digest)"
        #expect(SandboxImageProvisioner.digests(in: text) == [FakeContainerWorld.digest])
    }

    @Test func ignoresThingsThatAreNotDigests() {
        let short = "sha256:" + String(repeating: "a", count: 63)
        let upper = "sha256:" + String(repeating: "A", count: 64)
        let other = "sha512:" + String(repeating: "a", count: 64)
        let noPrefix = String(repeating: "a", count: 64)
        #expect(SandboxImageProvisioner.digests(in: [short, upper, other, noPrefix].joined(separator: "\n")).isEmpty)
        #expect(SandboxImageProvisioner.digests(in: "").isEmpty)
        #expect(SandboxImageProvisioner.digests(in: "[]").isEmpty)
    }

    @Test func findsADigestInsideJSONAndInsideAReference() {
        let json = "[{\"index\": {\"digest\": \"\(FakeContainerWorld.digest)\"}}]"
        #expect(SandboxImageProvisioner.digests(in: json) == [FakeContainerWorld.digest])
        let reference = "registry.example/img@\(FakeContainerWorld.otherDigest)"
        #expect(SandboxImageProvisioner.digests(in: reference) == [FakeContainerWorld.otherDigest])
    }

    @Test func theFakeInspectOutputWithNoDigestsHasNone() {
        #expect(SandboxImageProvisioner.digests(in: FakeContainerWorld.inspectOutput([])).isEmpty)
    }
}

@Suite struct SandboxImageProvisionerTests {
    // MARK: registry images

    @Test func anImageAlreadyPresentWithTheRightDigestIsNotPulled() async throws {
        let world = FakeContainerWorld()
        world.imageDigests = [FakeContainerWorld.digest]
        let progress = ProgressLog()
        try await makeProvisioner(world).ensure(registryPin(), progress: { progress.add($0) })
        #expect(world.count("image pull") == 0)
        #expect(progress.messages.isEmpty)
        #expect(world.containerCalls == [["image", "inspect", registryPin().runReference]])
    }

    @Test func theDigestOfAPlatformOrLayerIsNotTheImageDigest() async throws {
        // Only the index digest counts; the pinned digest as a variant does not.
        let world = FakeContainerWorld()
        world.imageDigests = [FakeContainerWorld.otherDigest, FakeContainerWorld.digest]
        try await makeProvisioner(world).ensure(registryPin(), progress: { _ in })
        #expect(world.count("image pull") == 1)
    }

    @Test func theIndexDigestIsReadFromTheConfiguration() {
        let output = FakeContainerWorld.inspectOutput([FakeContainerWorld.digest, FakeContainerWorld.otherDigest])
        #expect(SandboxImageProvisioner.indexDigest(in: output) == FakeContainerWorld.digest)
        #expect(SandboxImageProvisioner.indexDigest(in: "") == nil)
        #expect(SandboxImageProvisioner.indexDigest(in: "[]") == nil)
        #expect(SandboxImageProvisioner.indexDigest(in: "{}") == nil)
        #expect(SandboxImageProvisioner.indexDigest(in: "[{\"configuration\": {}}]") == nil)
        #expect(SandboxImageProvisioner.indexDigest(
            in: "[{\"configuration\": {\"descriptor\": {\"digest\": \"sha256:abc\"}}}]") == nil)
    }

    @Test func aMissingImageIsPulledAndThenAccepted() async throws {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        world.digestsAfterPull = [FakeContainerWorld.digest]
        let progress = ProgressLog()
        try await makeProvisioner(world).ensure(registryPin(), progress: { progress.add($0) })
        #expect(world.count("image pull") == 1)
        #expect(progress.messages == ["Downloading the sandbox image"])
        #expect(commandNames(world) == ["image inspect", "image pull", "image inspect"])
        let reference = registryPin().runReference
        #expect(world.containerCalls == [
            ["image", "inspect", reference],
            ["image", "pull", reference],
            ["image", "inspect", reference],
        ])
    }

    @Test func anImageWithAnotherDigestIsPulledAgain() async throws {
        let world = FakeContainerWorld()
        world.imageDigests = [FakeContainerWorld.otherDigest]
        world.digestsAfterPull = [FakeContainerWorld.digest]
        try await makeProvisioner(world).ensure(registryPin(), progress: { _ in })
        #expect(world.count("image pull") == 1)
        #expect(world.imageDigests == [FakeContainerWorld.digest])
    }

    @Test func aDifferentDigestAfterThePullIsAMismatch() async {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        world.digestsAfterPull = [FakeContainerWorld.otherDigest]
        let provisioner = makeProvisioner(world)
        let error = await thrownError { try await provisioner.ensure(registryPin(), progress: { _ in }) }
        #expect(error == SandboxError.digestMismatch(
            expected: FakeContainerWorld.digest, found: FakeContainerWorld.otherDigest))
    }

    @Test func theMismatchNamesTheIndexDigestNotAPlatformDigest() async {
        let world = FakeContainerWorld()
        let low = "sha256:" + String(repeating: "11", count: 32)
        world.imageDigests = nil
        world.digestsAfterPull = [FakeContainerWorld.otherDigest, low]
        let provisioner = makeProvisioner(world)
        let error = await thrownError { try await provisioner.ensure(registryPin(), progress: { _ in }) }
        #expect(error == SandboxError.digestMismatch(expected: FakeContainerWorld.digest, found: FakeContainerWorld.otherDigest))
    }

    @Test func noImageAfterThePullIsAMismatchWithNothingFound() async {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        world.digestsAfterPull = nil
        let provisioner = makeProvisioner(world)
        let error = await thrownError { try await provisioner.ensure(registryPin(), progress: { _ in }) }
        #expect(error == SandboxError.digestMismatch(expected: FakeContainerWorld.digest, found: nil))
    }

    @Test func anImageWithNoDigestsAfterThePullIsAMismatchWithNothingFound() async {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        world.digestsAfterPull = []
        let provisioner = makeProvisioner(world)
        let error = await thrownError { try await provisioner.ensure(registryPin(), progress: { _ in }) }
        #expect(error == SandboxError.digestMismatch(expected: FakeContainerWorld.digest, found: nil))
    }

    @Test func aFailedPullPropagatesAsCommandFailed() async {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        world.failures["image pull"] = SandboxProcessResult(status: 1, output: "denied\n")
        let provisioner = makeProvisioner(world)
        let error = await thrownError { try await provisioner.ensure(registryPin(), progress: { _ in }) }
        #expect(error == SandboxError.containerCommandFailed(
            action: "download the sandbox image", detail: "denied"))
    }

    @Test func aTimedOutPullPropagatesAsTimedOut() async {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        world.failures["image pull"] = SandboxProcessResult(status: 0, output: "", timedOut: true)
        let provisioner = makeProvisioner(world)
        let error = await thrownError { try await provisioner.ensure(registryPin(), progress: { _ in }) }
        #expect(error == SandboxError.timedOut(action: "download the sandbox image"))
    }

    // MARK: unpublished pins

    @Test func thePlaceholderPinMeansNoImageIsPublishedAndNothingRuns() async {
        let world = FakeContainerWorld()
        let provisioner = makeProvisioner(world)
        let pin = registryPin(digest: SandboxImagePin.placeholderDigest)
        let error = await thrownError { try await provisioner.ensure(pin, progress: { _ in }) }
        #expect(error == SandboxError.imageNotPublished)
        #expect(world.calls.isEmpty)
    }

    @Test func aLocalPlaceholderPinIsAlsoNotPublished() async {
        let world = FakeContainerWorld()
        let pin = localPin(digest: SandboxImagePin.placeholderDigest)
        let provisioner = makeProvisioner(world)
        let error = await thrownError { try await provisioner.ensure(pin, progress: { _ in }) }
        #expect(error == SandboxError.imageNotPublished)
        #expect(world.calls.isEmpty)
    }

    @Test func aMalformedDigestIsTreatedAsNotPublished() async {
        let world = FakeContainerWorld()
        let pin = registryPin(digest: "sha256:abc")
        let provisioner = makeProvisioner(world)
        let error = await thrownError { try await provisioner.ensure(pin, progress: { _ in }) }
        #expect(error == SandboxError.imageNotPublished)
        #expect(world.calls.isEmpty)
    }

    // MARK: local images

    @Test func aLocalImageWithTheRightDigestIsAcceptedWithoutAPull() async throws {
        let world = FakeContainerWorld()
        world.imageDigests = [FakeContainerWorld.digest]
        let progress = ProgressLog()
        try await makeProvisioner(world).ensure(localPin(), progress: { progress.add($0) })
        #expect(world.count("image pull") == 0)
        #expect(progress.messages.isEmpty)
        #expect(world.containerCalls == [["image", "inspect", "tuff-sandbox:dev"]])
    }

    @Test func aMissingLocalImageIsNeverPulled() async {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        let progress = ProgressLog()
        let provisioner = makeProvisioner(world)
        let error = await thrownError { try await provisioner.ensure(localPin(), progress: { progress.add($0) }) }
        #expect(error == SandboxError.digestMismatch(expected: FakeContainerWorld.digest, found: nil))
        #expect(world.count("image pull") == 0)
        #expect(progress.messages.isEmpty)
    }

    @Test func aLocalImageWithAnotherDigestIsNeverPulled() async {
        let world = FakeContainerWorld()
        world.imageDigests = [FakeContainerWorld.otherDigest]
        let provisioner = makeProvisioner(world)
        let error = await thrownError { try await provisioner.ensure(localPin(), progress: { _ in }) }
        guard case .digestMismatch(let expected, _)? = error else {
            Issue.record("Expected digestMismatch, got \(String(describing: error))")
            return
        }
        #expect(expected == FakeContainerWorld.digest)
        #expect(world.count("image pull") == 0)
    }
}
#endif
