# TUFF 11.6.1

## The local API reads prompts much faster

- Since 10.0, the local API read every prompt one token at a time, as fast as the model writes. Chat sessions routed through the same shared host were affected too. Standalone `tuff prompt` used batched prefill. The shared host now reads prompts in blocks. Through `tuff serve` on a 16 GB M2 MacBook Air, a 930-token prompt went from 47 to 51 tokens per second to 438 to 494 on Gemma 4 E2B, and from 8 to 14 to 107 to 168 on Ornith 9B, with the same answers. Flint read at 28 and Inkling 2 Mini at 36.
- The API and Chat can share one loaded model, and a lock keeps them taking turns. Reading a token at a time was how a long prompt avoided holding the lock for its whole length. The lock is now taken once per block, so another request on the same model still gets a turn between blocks. On Inkling 2 Mini a block can take several seconds, so a second request waits that long at most.

## Repeated conversations are reused again

- A request without `prompt_cache_key` ran in a fresh session that was cleared when the request ended. A client that resent its conversation every turn had it read from the start each time, and `cached_tokens` stayed at 0. Requests without a key now share one session per model, as before 10.0, and continue from where the last one left off. A follow-up turn on a 930-token conversation reused 970 of its 991 tokens on Gemma 4 E2B. Sending exactly the same request twice still reads it again: reuse works when a request extends a conversation the server holds.
- That shared session is only used while no client has a `prompt_cache_key` session open on the same model, so it never adds memory. A request that arrives while its session is busy runs in a fresh one instead of failing with `model_busy`. That applies to keyed sessions too.

## Compatibility

Every model downloads the 11.6.1 inference engine the first time it loads. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
