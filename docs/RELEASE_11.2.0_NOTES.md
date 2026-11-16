# TUFF 11.2.0

## Faster long prompts on Gemma and Qwen-family models

- Gemma, Ornith, Qwen 3.6, Qwen 3.8 27B and Flash Next read long prompts faster. Most of prefill is matrix multiplies against the quantized weights. Whole blocks of 128 prompt tokens now use a wider tile that reads each weight tile once for twice as many tokens.
- Measured on a 16 GB M2 MacBook Air, same prompts, greedy, engines alternated with cooldowns: Ornith 1.5 9B, 1,040-token prompt, 6.44 s to 5.85 s (162 to 178 tokens per second). Gemma 4 E4B, 1,032-token prompt, 3.79 s to 3.38 s (272 to 305 tokens per second). Qwen 3.6 gained about 3 percent, close to run-to-run noise, because its prefill mostly waits on expert reads.
- Prompts under 128 tokens and decode speed are unchanged. A layer whose input width is not a multiple of 256, such as one projection in Gemma 4 26B, keeps the previous tile.
- Answers were the same in every case tested. The wider tile matched the previous one bit for bit across the layer shapes these models use, and greedy outputs matched on Ornith 9B, Gemma 4 E4B and Qwen 3.6, all on an M2. Other Apple GPUs were not tested, so each model load runs one small comparison on the Mac's own GPU and keeps the previous tile if the bytes differ or the wider one does not build.
- Bonsai, Flint, GPT-OSS and MiniMax are unchanged.

## Compatibility

Gemma and Qwen-family models download the 11.2.0 inference engine the first time they load. Other engines stay on 11.0.0. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
