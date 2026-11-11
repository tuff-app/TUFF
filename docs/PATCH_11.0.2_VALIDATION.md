# 11.0.2 validation

## Scope

Fixed-width sidebar shell, stable small toggle, shared full-window background and saved 10% transparency steps. Native AppKit window controls and the native outer frame remain intact. No inference-engine changes.

## Verified locally

- 64 focused tests passed: native window controls dispatch once and respect a close veto; old transparency settings migrate; nondefault values including 0% and 100% persist across AppModel recreation; window layout constraints pass.
- Compared the macOS 27 preview against supplied light/dark references. Checked chat and Settings, corners without protruding glass, and a continuous background at 0% and 100% transparency.
- Tested pointer clicks in both sidebar-toggle positions, rapid shortcut toggles with a populated transcript, and dragging the former divider area. The sidebar returned to the same edge and did not resize.
- Verified a snapped slider value and a legible 100% label at 150% app zoom.
- Loaded installed Gemma E2B using the published engine catalog and generated a Python code response. The transcript remained intact after rapid toggles.
- Preview: `/private/tmp/tuff-11.0.2-unified-preview/TUFF.app`, isolated data under `/private/tmp/tuff-sidebar-fixed-qa` with model-directory symlinks.

## Release gates

Canonical source-gate log: `/private/tmp/tuff-11.0.2-source-gate.log`. Production packaging, signed update feed, updater fixtures, CI and public-asset verification are required before publication.

## Limits

macOS 26 comparison uses supplied screenshots; live UI validation was on macOS 27.2.1. A temporary preview's replacement window buttons caused a recursive zoom crash; that approach was removed entirely, and the final implementation retains the system buttons. The extra corner mask was also removed.
