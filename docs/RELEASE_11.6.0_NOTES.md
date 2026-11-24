# TUFF 11.6.0

## Faster Inkling 2 Mini

- Writing is about 18 percent faster, and reading a long prompt about 1.36 times faster. Measured on a 16 GB M2 MacBook Air against the 11.5.0 engine, from a cold file cache, alternating the two engines: four prompts went from 5.0, 5.0, 5.0 and 5.5 tokens per second to 5.7, 5.7, 6.2 and 6.7, with the same text in every pair. A 1,062-token prompt took 33.5 s instead of 45.4 s.
- The experts' math on the GPU does more work per group of threads: each group now computes several rows of an expert instead of one. The 4-bit expert layers take 39 percent less GPU time per token. The four 8-bit layers improve only a little, because speeding them up further would change the order numbers are added in.
- While writing, TUFF guesses which experts the next layer will pick and asks the disk to start loading them early. The guess is right about three times in four, and time spent waiting for the disk dropped by 35 to 45 percent.
- While reading a prompt, TUFF loads the experts the next layer needs in the background while the GPU works on the current one.
- Answers are the same. The new kernels give the same bytes as the old ones in every test, and the engine compares the two on the Mac's own GPU when the model loads. If a single byte differs, it keeps the old kernels. The comparison adds about 70 ms to loading.

## Benchmark clients can use the local API

- The local API accepted neither `ignore_eos` nor `reasoning_effort: "none"`. Benchmark tools such as gBench send both with every request, so every request failed with HTTP 400.
- `ignore_eos` is now accepted and ignored: replies still end where the model ends its turn, and the usage block counts the tokens really written. Honoring it would mean writing past the end of a turn, which the chat formats, tool parsing and conversation reuse are not built for.
- `reasoning_effort: "none"` now means thinking off on every model. GPT-OSS can't skip reasoning, so there it means `low`. MiniMax always thinks, as it already did with `enable_thinking: false`. Sending `"none"` together with `enable_thinking: true` is refused instead of guessed. The other effort levels still work only with GPT-OSS.

## Compatibility

Inkling 2 Mini downloads the 11.6.0 inference engine the first time it loads. Every other model keeps the engine it has. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
