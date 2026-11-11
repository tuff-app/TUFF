# TUFF 11.0.0

## A redesigned Mac app

- A shared design system brings consistent light and dark appearances to Chat, Models, Benchmarks, Server and Settings.
- Settings are organized into a hub, with a refined chat sidebar, hover controls, clearer spacing, larger hit targets and consistent inactive-window styling.
- Code blocks, accent colors and filled controls have improved contrast. Shorter labels and fewer repeated explanations keep the interface focused. Image-support guidance opens Models directly.

## Smoother responses

- Adaptive smooth streaming is enabled by default and can be changed in Settings.
- Streaming preserves Unicode graphemes, combining marks, flags and emoji sequences. Completion flushes the exact response, and stopping or restarting a response preserves the appropriate partial text.

## Apple Foundation Model

- Select Apple's on-device model on eligible Macs with macOS 26 or later, Apple Intelligence enabled and the model ready.
- The collapsible Apple inventory reflects device availability and provides setup or refresh actions. Macs without an available Apple model show “No Apple Models Found on This Mac”.
- Apple chats have an independent system prompt, restore their provider per conversation, and support streaming, Stop and subsequent generation. Cancellation no longer surfaces a raw cancellation error.
- Capability checks keep unsupported requests from silently switching to a downloaded engine. Private Cloud Compute is not offered in this release.

## Local engines and reliability

- Bonsai prefill, GDN and projection improvements reduce work in supported paths. Performance depends on the model, prompt and hardware; these changes do not imply a universal speed multiplier.
- Mapped expert file-cache binding, slot-cache opt-out and exclusions for retired engine families improve cache handling. RDAdvice requests are bounded and deduplicated; no speculative performance gain is claimed.
- Fresh Gemma, Qwen and Flint packages use authenticated acquisition with the existing release signing key. Legacy GPT-OSS and MiniMax payloads are retained.
- Engine packaging handles canonical temporary paths correctly. Downloaded engine paths and updater settings have been corrected.
- API cancellation and sampled-to-greedy reload now preserve model ownership, allowing a subsequent unload to release the model while protecting models owned by the app.

## Validation and compatibility

The full model-free source gate passed, including 1,018 engine tests, 271 next-suite tests and 1,201 app tests. Live Apple routing tests passed. Packaged Gemma E2B, Gemma 26B, Qwen 3.6, Flash Next and Bonsai 2B probes generated the expected response and unloaded. Disconnect/reload/unload, signatures, archive checksums, fresh engine acquisition and updater fixtures were checked.

TUFF requires Apple Silicon and macOS 15 or later. Apple model availability additionally depends on Apple's OS, hardware, language and Apple Intelligence requirements. Model weights are separate downloads. The Mac app is ad-hoc signed and is not notarized.

Known cosmetic limitation: selected rows in native macOS context menus use the system accent color.
