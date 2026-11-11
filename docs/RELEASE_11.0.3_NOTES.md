# TUFF 11.0.3

## Window and sidebar

- Restored the rounder window and sidebar curves shown in TUFF 11's screenshots on macOS 27. The sidebar follows the outer frame's inset and keeps its fixed width.
- Restored the large collapsed Chat sidebar toggle, matching the model pill's height and surface color. Expanded and Settings controls retain their compact size and placement.
- Retained native window controls, resizing, shadows and glass clipping, without extra frame masks or replacement traffic-light buttons. The title bar and chat retain a shared transparency surface.

## Apple Foundation Model in the CLI and API

- Use `tuff prompt 'Hello' --model apple` or `TUFFCLI --model apple.on-device --chat-prompt 'Hello'` with Apple's on-device model. `tuff load apple` selects it in the app.
- The local server lists `apple.on-device` when Apple Intelligence reports the model ready. It can be selected as the server default in Settings or with `tuff serve --default-model apple.on-device`.
- Supports streamed and non-streamed text, system/developer instructions, conversation history, temperature, output limits and cancellation. Apple requests do not acquire a TUFF engine or require downloaded model weights.
- Rejects unsupported image, tool, reasoning, sampling-filter and engine options instead of silently ignoring them.
- Reports Apple token usage as unavailable rather than inventing counts. Cancelling a stream no longer reports an empty or partial response as a successful completion.

## Website

- Redesigned the homepage around the supplied app screenshots, concise product details and installation requirements.
- Removed repetitive promotional sections and the dated competitor checklist. The expert-streaming diagram now has explicit play/pause controls.
- Updated the benchmark page's styling while preserving its reports, filters and sorting, with responsive layouts checked from 320 to 1280 pixels.

## Compatibility

Apple Foundation Model requires an eligible Mac with macOS 26 or later and Apple Intelligence enabled and ready. Private Cloud Compute is not exposed through this CLI/API integration; it requires separate Apple entitlement approval. The rounder native frame uses an app-scoped, undocumented AppKit preference, which may need adjustment in future macOS versions; it does not change system-wide preferences.

This patch reuses the published 11.0.0 inference engines. No DFlash components or new engine-performance claims are included. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
