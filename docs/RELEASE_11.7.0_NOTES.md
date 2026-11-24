# TUFF 11.7.0

## Ternary Bonsai 2.5 27B

- PrismML's Ternary Bonsai 2.5 27B is in the model list. It is an upgrade of Bonsai 2 27B for coding, multi-step tool use and longer tasks, with the same 8.6 GB download and the same 16 GB memory requirement. PrismML reports SWE-bench Verified going from 60.8 to 75.6.
- It uses the same architecture and ternary format as Bonsai 2 27B, so it reads and writes at the same speed. Like Bonsai 2, it is text only in TUFF.
- Bonsai 2 27B moves to the Legacy section. It still installs and runs, and an installed copy is kept. Bonsai 2 2B is unchanged; there is no 2.5 version of it.

## Memory checks for models an engine does not know

- An inference engine sizes the models it was built with. A model it did not know took the whole memory budget, so it could not share memory with a second chat, an API request or another model. It is now sized from its own files when the engine knows a model of the same architecture.

## Compatibility

Qwen-family models, including both Bonsai models, download the 11.7.0 inference engine the first time they load. Other engines are unchanged. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
