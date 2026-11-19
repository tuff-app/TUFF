# TUFF 11.4.0

## Faster replies on mixture-of-experts models

- Gemma 4 26B, Qwen 3.6 and the other routed models pick a few experts in every layer for every token. The scores are computed on the GPU, and so is the choice of the best ones. That choice ran on a single GPU thread, once per layer and once more for the lookahead: 59 times a token on Gemma 4 26B, each time just before TUFF reads the answer back to start loading experts. It now runs on a whole group of threads, each working out where its experts rank.
- Measured on a 16 GB M2 MacBook Air against the 11.3.0 engines, greedy, engines alternated with pauses: Gemma 4 26B 16.2 to 16.8 tokens per second (4.1 percent), Qwen 3.6 35B 16.9 to 17.5 (3.7 percent). With sampling at temperature 0.7, Gemma 4 26B went from 16.0 to 16.2 over three rounds, which is inside the run-to-run spread of a model that reads experts from disk. The GPU time saved was about 2.6 ms a token on Gemma 4 26B.
- Models without routed experts (Gemma E2B, E4B and 12B, Ornith 9B, Bonsai) are unchanged. So is prompt reading, which picks experts a different way.
- Answers are the same. The new kernel picked the same experts, in the same order, with the same weights byte for byte in every test (5 to 512 experts, 4 to 16 chosen, ties, and scores it must never pick), and the same prompts gave the same text on both models, all on an M2. Other Apple GPUs were not tested, so the engine compares the two on the Mac's own GPU when a model loads and keeps the previous one if a single byte differs.

## Compatibility

Gemma and Qwen-family models (Qwen, Ornith, Bonsai) download the 11.4.0 inference engine the first time they load. Flint, GPT-OSS and MiniMax stay on 11.0.0. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
