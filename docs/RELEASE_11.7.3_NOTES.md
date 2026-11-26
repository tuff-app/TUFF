# TUFF 11.7.3

## Shared inference

Chat and the local API take turns at the model's own prefill block boundaries.
Only the final block computes the output head. Greedy and sampled requests can
reuse a loaded model and its conversation state. Where supported, GPU
sampling runs in the same command buffer as the token forward pass.

The Inspector calculates prefill throughput from the uncached tokens it
actually computed.

## Sampling

All supported Top-K values from 1 through 256 use tiled selection. Unfiltered
temperature sampling keeps the random draw strictly inside its valid interval.
A failed Qwen GDN input projection now reports a Metal error instead of
continuing with stale scratch data. The Bonsai prefill test requires an
actual tiled dispatch on supported hardware.

## Local API

`ignore_eos: true` runs a plain-text benchmark in a fresh session.
Generation continues through model stop markers until the requested token
limit, an explicit stop string, or the context limit. This mode excludes
tool requests and Apple on-device inference.

`reasoning_effort` accepts `low`, `medium`, and `high` on models with a
thinking toggle, which treats all three as thinking on. GPT-OSS continues
to distinguish the three levels. Responses report the effective mode in
`X-TUFF-Reasoning-Mode`.

## Compatibility

This release updates every TUFF inference engine family. The app is ad-hoc
signed.
