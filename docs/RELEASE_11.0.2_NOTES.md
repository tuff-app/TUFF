# TUFF 11.0.2

- Fixed sidebar collapse and expansion: the toggle stays small, returns to the same position, and moves with the content. Rapid toggles no longer leave an outdated transcript-layout reset queued.
- Removed sidebar resizing and the draggable divider. Its width stays fixed at each app zoom level.
- Unified the title bar and chat background so transparency affects both consistently. Settings keeps a continuous solid background.
- Replaced the transparency switch with a slider from 0% to 100%, snapping in 10% steps. The value is saved, and the old off/on setting migrates to 0%/70%. macOS Reduce Transparency remains respected.
- Kept native window controls and the native outer frame. Sidebar corners follow the window inset without an extra corner mask or protruding glass.

This patch uses the existing 11.0.0 inference engines. Models, chats and settings are preserved.
