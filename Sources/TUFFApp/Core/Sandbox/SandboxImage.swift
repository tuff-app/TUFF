#if os(macOS)
import Foundation

/// The sandbox image TUFF will run, named by the digest of its content. The
/// digest is part of the app, so a changed image on the registry is never
/// run: the pull asks for exactly this content, and the result is checked
/// again after the pull.
struct SandboxImagePin: Equatable, Sendable {
    enum Source: Equatable, Sendable {
        /// Pulled from a registry as `reference@digest`.
        case registry
        /// Already in the local image store under `reference`, such as the
        /// image a developer built from tuff-app/sandbox. It is never pulled.
        case local
    }

    let reference: String
    let digest: String
    let source: Source

    /// What `container run` is given.
    var runReference: String {
        source == .registry ? "\(reference)@\(digest)" : reference
    }

    /// The digest is `sha256:` and 64 lowercase hexadecimal digits.
    static func isWellFormed(_ digest: String) -> Bool {
        digest.wholeMatch(of: #/sha256:[0-9a-f]{64}/#) != nil
    }

    static let placeholderDigest = "sha256:" + String(repeating: "0", count: 64)

    /// True until the first image has been published and its digest put here.
    var isPlaceholder: Bool { digest == Self.placeholderDigest }

    /// The image this version of TUFF runs: the index the maintainer's
    /// workflow (`.github/workflows/image.yml` in tuff-app/sandbox) published as
    /// `ghcr.io/tuff-app/tuff-sandbox:3` from the `sandbox-v3` tag. A new
    /// image means a new digest here (PUBLISHING.md in tuff-app/sandbox).
    static let release = SandboxImagePin(
        reference: "ghcr.io/tuff-app/tuff-sandbox",
        digest: "sha256:34e950fac80834e569c3781b233b3f0ea21e2cd0be646fcdd04225db48b99528", source: .registry)

    /// For developers and tests before an image is published: when both
    /// `TUFF_SANDBOX_DEV_IMAGE` (a name in the local image store, such as the
    /// `tuff-sandbox:dev` that `scripts/selftest.sh build` in tuff-app/sandbox makes) and
    /// `TUFF_SANDBOX_DEV_DIGEST` (its `sha256:` index digest) are set, that
    /// local image is run instead. The image must still have exactly that
    /// digest, and it is never pulled. A person who can set these can already
    /// run anything as this user, so they widen nothing; they are not in the
    /// settings and the app never sets them.
    static func fromEnvironment(_ environment: [String: String] = ProcessInfo.processInfo.environment) -> SandboxImagePin? {
        guard let reference = environment["TUFF_SANDBOX_DEV_IMAGE"], !reference.isEmpty,
              let digest = environment["TUFF_SANDBOX_DEV_DIGEST"], isWellFormed(digest),
              !reference.contains("@") else { return nil }
        return SandboxImagePin(reference: reference, digest: digest, source: .local)
    }
}

/// Makes sure the pinned image is in the local image store.
struct SandboxImageProvisioner: Sendable {
    let tool: ContainerTool

    /// Pulls the image unless it is already there with the pinned digest,
    /// then checks the digest. `progress` gets a short sentence before a
    /// download, which can take a while on a first start.
    func ensure(_ pin: SandboxImagePin, progress: @Sendable (String) -> Void) async throws {
        guard SandboxImagePin.isWellFormed(pin.digest), !pin.isPlaceholder else {
            throw SandboxError.imageNotPublished
        }
        let present = await tool.inspectImage(pin.runReference).flatMap(Self.indexDigest(in:))
        if present == pin.digest { return }
        guard pin.source == .registry else {
            throw SandboxError.digestMismatch(expected: pin.digest, found: present)
        }
        progress("Downloading the sandbox image")
        try await tool.pullImage(pin.runReference)
        guard let pulled = await tool.inspectImage(pin.runReference) else {
            throw SandboxError.digestMismatch(expected: pin.digest, found: nil)
        }
        let found = Self.indexDigest(in: pulled)
        guard found == pin.digest else {
            throw SandboxError.digestMismatch(expected: pin.digest, found: found)
        }
    }

    /// The image's own digest in the output of `container image inspect`:
    /// `[0].configuration.descriptor.digest`, the digest of the image index
    /// that `ref@digest` names. The digests of its platforms (`variants`) and
    /// layers do not count.
    static func indexDigest(in output: String) -> String? {
        guard let data = output.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]],
              let first = root.first,
              let configuration = first["configuration"] as? [String: Any],
              let descriptor = configuration["descriptor"] as? [String: Any],
              let digest = descriptor["digest"] as? String,
              SandboxImagePin.isWellFormed(digest) else { return nil }
        return digest
    }

    /// Every `sha256:` digest anywhere in the output of `container image
    /// inspect`. `ensure` compares only `indexDigest(in:)`; the guarantee
    /// behind it is that `container image pull ref@digest` fetches exactly
    /// that content.
    static func digests(in output: String) -> Set<String> {
        Set(output.matches(of: #/sha256:[0-9a-f]{64}/#).map { String($0.output) })
    }
}
#endif
