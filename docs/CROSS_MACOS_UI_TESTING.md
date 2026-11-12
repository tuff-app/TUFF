# Compare macOS 26 and 27

Use real macOS 26 and 27 installations (two Macs, or an Apple Silicon macOS VM).
Changing an availability flag or SDK does not simulate another OS's SwiftUI or
window compositor. No macOS 26 environment is provisioned by these scripts.

Use the same source commit and Xcode/SDK version on both installations. The
render matrix is opt-in, uses an inference stub and isolated support directory,
and never loads weights or writes production settings. Its Apple chat fixture forces
availability to Ready so VM hardware eligibility cannot change the layout; native
model availability is checked separately in the real app. Run from the checkout:

```sh
python3 Scripts/visual_matrix.py /tmp/tuff-ui-26
# On macOS 27, at the same commit:
python3 Scripts/visual_matrix.py /tmp/tuff-ui-27
# Copy both output directories onto one Mac, then:
python3 Scripts/compare_visual_runs.py /tmp/tuff-ui-26 /tmp/tuff-ui-27 /tmp/tuff-ui-review.html
open /tmp/tuff-ui-review.html
```

`--smoke` captures one case to check setup. The full matrix has 48 cases:
light/dark, Chat/Settings, expanded/collapsed sidebar, 0/70/100% transparency,
and 1080/640-point widths at 720 points tall. Each case records pixel dimensions
and OS metadata. The comparison rejects mismatched commits, dirty product source,
missing cases and different pixel scales. Use identical display scaling. Keep the
artifacts out of Git; retain approved captures as external release evidence.
The report embeds its images and works offline. Review side-by-side and with its
opacity overlay; OS font rasterization changes are not automatically failures.

## Native-window acceptance (required on both OS versions)

Offscreen snapshots cannot qualify desktop transparency, the native frame or
animations. Test the exact same packaged app on both OS installations, with
identical wallpaper, accent, Reduce Transparency/Motion settings and window size.
Use an isolated account/profile. Do not compare a debug render against a release
app and call it native parity.

- Light and dark, 0/70/100% transparency: title-bar and Chat surfaces must match.
- Expanded/collapsed sidebar: fixed width; toggle lands at the intended position.
- Toggle rapidly ten times, then pause: no stale offsets, jumps or clipped content.
- Drag the old divider area: sidebar width must stay fixed.
- Check outer/inner corners, traffic lights and shadows at all four window corners.
- Check Settings matches its title-bar background; repeat with the window inactive.
- Resize to 640 × 720 and 1080 × 720; inspect clipping and all button hit targets.
- Repeat at 125% app zoom and with Reduce Transparency enabled.

Capture native-window screenshots on each OS using macOS's window screenshot
shortcut (Shift-Command-4, then Space). Avoid screen recording and its indicator.
These captures remain a human-reviewed gate, separate from the offscreen matrix.
Do not claim cross-version parity until both OS runs and native checks pass.

A dedicated GUI runner can execute the same commands, but a logged-in desktop
and matching display configuration must be provisioned first. Do not register an
unrestricted self-hosted runner for automatic untrusted PR jobs in this public
repository. The scripts are local/manual until suitable hosts exist.

## macOS 26 VM on the macOS 27 Mac

The setup helper follows [Tart's official quick start](https://tart.run/quick-start/).
The base image is approximately a 25 GB download. The helper conservatively
requires 80 GiB free for the download, cache and guest growth; this is our storage
budget, not a claim about a guaranteed image size. Check available space before provisioning; use a larger volume when needed.
VM provisioning and interactive validation are separate from the render matrix.

```sh
python3 Scripts/setup_macos26_ui_vm.py --store /Volumes/YourSSD/TUFF-UI-26 --app dist/v11.0.3/TUFF.app
# After the preflight passes, use the same command with --create.
```

Install Tart from its official release archive (the Homebrew tap currently lacks its formula). The helper uses Tart from PATH or `~/Downloads/TUFF UI Testing/tools/tart.app/Contents/MacOS/tart`, creates a separate VM and shares only a copy
of the packaged app read-only. It does not share the checkout, production models,
chats or credentials. Use a fresh storage directory. Inside the VM, first verify
`sw_vers` reports macOS 26; record its build and the image used for your baseline.
Copy `/Volumes/My Shared Files/tuff-app/TUFF.app` to the guest's Applications
folder and perform the native checklist above. Use the guest login instructions
from Tart's documentation; no guest password is stored in the repository.

The base image has no matching Xcode toolchain provisioned. To run source render
snapshots in the guest, install a toolchain that can run on macOS 26 and use that
same toolchain on the host. Native release-app comparison can run without Xcode.
VM graphics can differ from physical hardware; investigate compositor-only
mismatches on a physical macOS 26 Mac before changing product geometry.
