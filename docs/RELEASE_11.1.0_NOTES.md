# TUFF 11.1.0

## Faster prompts on Qwen-family models

- Ornith, Qwen 3.6, Qwen 3.8 27B and Flash Next read long prompts faster. The gated DeltaNet recurrence in prefill used about a tenth of the GPU time on Ornith 9B; a new kernel for these models' head size does the same arithmetic in the same order about five times faster.
- Measured on a 16 GB M2 MacBook Air in Low Power Mode, same prompts, greedy, engines alternated with cooldowns: Ornith 1.5 9B, 1,040-token prompt, 14.0 s to 12.8 s (74 to 81 tokens per second). Qwen 3.6 and Flash Next gained 3 to 5 percent, close to run-to-run noise, because their prefill mostly waits on expert reads. Short prompts are unchanged.
- Answers are the same. The new kernel is checked bit for bit against the previous one, and greedy outputs matched on Ornith 9B, Qwen 3.6 and Flash Next. A Mac that cannot build the new kernel keeps the previous one.
- Decode speed, Bonsai, Gemma, Flint, GPT-OSS and MiniMax are unchanged.

## Compatibility

Qwen-family models download the 11.1.0 inference engine the first time they load. Other engines stay on 11.0.0. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
