# TUFF 11.2.1

## Local server fixes

- `"seed": -1` is accepted and means a random seed, the same as leaving it out. Clients written for llama.cpp and similar servers send it that way. TUFF used to answer it with `400 malformed JSON request`, although the JSON was fine. Other negative seeds are fixed seeds, as in the OpenAI API.
- `"max_tokens": -1` and `"max_completion_tokens": -1` are accepted and mean the default limit of 4,096 tokens. LM Studio and llama.cpp clients send -1 for "no limit of my own". Zero and other negative limits are still refused.
- A fraction or an out-of-range number in `max_tokens`, `max_completion_tokens`, `top_k`, `n` or `seed` now gets an error that names the field, for example `max_tokens must be a whole number`, instead of `malformed JSON request`.

The Messages and Responses routes keep their own rule that an output limit must be positive.

## Compatibility

This patch uses the existing inference engines. Models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
