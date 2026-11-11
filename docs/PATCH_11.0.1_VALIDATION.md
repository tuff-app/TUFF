# 11.0.1 macOS 27 compatibility preview

## Changes

- Restores the sidebar's 8.5-point outer inset on macOS 27, retains its 196-point content width at 100% zoom, rounded panel and persistent outline. macOS 26 retains its native inset.
- Checks the app's actual Private Cloud Compute signing entitlement before constructing the cloud model. Unentitled builds no longer offer an unusable provider. Existing cloud selections report that the build is not configured.
- On-device Apple Foundation Model identity and routing remain unchanged.

## Verification

- Built and packaged `/private/tmp/tuff-11.0.1-sidebar-final/TUFF.app`, using the published 11.0.0 engine catalog. No engine changes.
- Compared the running macOS 27 preview against the user's approved macOS 26 light and dark screenshots. Verified inset panel, outline, light/dark appearance, and collapse/expand returning to the same width.
- Real UI Apple chat returned `4` for `2 + 2`. Model list offered Apple Foundation Model and did not offer unentitled PCC.
- Fifteen focused Apple availability/routing tests passed; includes proving that the cloud SDK is not queried without the entitlement.
- Full canonical source gate, package checks and updater fixtures passed (exit 0). App core: 1,203 tests passed. Verification logs: `/private/tmp/tuff-11.0.1-source-gate.log`, `/private/tmp/tuff-11.0.1-packaging-tests.log`, `/private/tmp/tuff-11.0.1-updater-tests.log`. See final results in these logs.
- `git diff --check` passed.

## Limits and state

- macOS 26 comparison uses supplied screenshots; this host runs macOS 27. No live macOS 26 validation claimed.
- PCC needs Apple's managed entitlement and approved distribution; this patch does not obtain that approval or claim working cloud inference.
- Preview uses isolated `/private/tmp/tuff-11.0.1-qa` data and links to existing model folders. Production app, chats and model files are preserved.
- Preview left open. No patch commit, push or publication.
