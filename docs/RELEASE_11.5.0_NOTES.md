# TUFF 11.5.0

## Inkling 2 Mini

- TUFF can now run Inkling 2 Mini, the 26B mixture-of-experts model Thinking Machines released this week. It installs from the 4-bit `mlx-community/Inkling-2-Mini-4bit` checkpoint: a 16.7 GB download, for Macs with 16 GB of memory or more. It is in the MoE section of the model list as Inkling 2 Mini Q4, and in the CLI and API as `inkling-2-mini`.
- Chat, the Thinking switch, Web and Files search, and tool calls through the local API all work with it. It is text only for now. The checkpoint also has an image encoder, which TUFF does not install yet.
- With Thinking off the model still thinks briefly before it answers, usually 50 to 150 tokens. That is how it was trained: off is the lowest of its effort levels, not a different mode. Thinking on uses the publisher's "high" level.
- The model is built differently from the others TUFF runs, so it has its own inference engine. Twelve of its 24 layers run three times for every token, passing notes from one pass to the next. Attention uses a learned position bias instead of rotary positions, most layers only look back 1,024 tokens, and the experts are stored in two 4-bit and 8-bit float formats (NVFP4 and MXFP8) that TUFF reads as they are.
- Speed on a 16 GB M2 MacBook Air: about 4 to 6 tokens per second writing and about 22 tokens per second reading a long prompt. Short answers right after a load are at the low end (3.6 to 4.8 in my runs) and longer ones reach 6.4, because the experts are 15.7 GB and do not fit in memory beside everything else on a 16 GB Mac. Both numbers are mostly the SSD. A Mac with more memory should do better. I could not measure one.
- Installing took 20 minutes here on a connection that averaged about 13 MB a second.
- Reading a prompt works in blocks of up to 512 tokens, because a block of any size ends up needing nearly every expert in every layer. Stop still works in the middle of a block.
- Memory planning knows the model keeps a separate attention cache for each pass of a repeated layer. Auto picks 16K context on a 16 GB Mac.

## How it was checked

- I wrote a plain NumPy version of the model first and compared it with the publisher's own implementation on their small test checkpoint, including with randomized weights. The two agree to rounding error.
- TUFF's engine is checked against that reference on the real weights: the same twelve greedy tokens with the ten largest scores within half a unit at every step, and the first layer's attention output within a part in a thousand at every position of a prompt that runs past the attention window. Tests also cover reading in blocks against reading a token at a time, saving and restoring a conversation, rewinding, and stopping in the middle of a block.
- One thing to know: the batched prompt path works in half precision. Inkling picks its eight experts by score, and now and then the eighth and ninth are close enough that rounding decides. The answer is equally valid either way, but it means two runs through different paths are not identical number for number.
- The installer checks the checkpoint against pinned hashes, as for every other model, and the tokenizer matches Hugging Face's on 185 test strings.

## Also fixed

- The bundled tokenizer library now honors a tokenizer's `ignore_merges` setting. Inkling's tokenizer needs it. No other model TUFF installs sets it, so they encode exactly as before.

## Compatibility

Inkling 2 Mini downloads its own inference engine, 11.5.0, the first time it loads. Every other model keeps the engine it has. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
