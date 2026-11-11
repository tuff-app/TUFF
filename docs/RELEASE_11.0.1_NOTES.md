# TUFF 11.0.1

- Restored the inset, rounded sidebar on macOS 27, with consistent panel spacing, content width and outline. The sidebar keeps the same width after hiding and showing it.
- Fixed Private Cloud Compute availability detection. TUFF now checks its signing entitlement before offering the cloud model, avoiding a selection that cannot run in this distribution. Apple Foundation Model continues to run on-device on supported Macs.

This patch reuses the 11.0.0 inference engines. Existing models, chats and settings are preserved.

Private Cloud Compute is not available in this distribution; it requires Apple's managed entitlement and approved distribution.
