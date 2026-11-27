# TUFF 11.7.4

## Local API

- Streaming sends text as soon as it can rule out a stop-string match. Long
  stop strings that never match no longer hold back the reply.
- Messages streams include the prepared input-token count in `message_start`.
- Auto uses the inference engine’s prefill, first-token, and decode timings.
  It tracks sampled and image requests under their own workload settings and
  starts fresh with measurements collected using the corrected timings.
- Inkling 2 Mini accepts distinct `low`, `medium`, and `high` reasoning efforts.
  Its native `none` setting requests zero effort. The model can still produce
  reasoning, which counts toward the output token limit.

## Engine checks

- Unsupported routed expert counts produce a load error. The regression test
  exercises the rejection path.
- Inkling’s batched kernel test requires a real dispatch on supported hardware.

This release updates all six downloadable inference engines. The app is ad-hoc
signed.
