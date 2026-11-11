# Apple Foundation Model from the CLI and API

The on-device Apple Foundation Model uses the same adapter as TUFF chat. Apple
Intelligence must be enabled and its model ready on this Mac. No model download
or TUFF inference engine is needed. Private Cloud Compute is not offered by this
CLI/API integration; it requires separate Apple entitlement approval.

```sh
tuff prompt 'What is 2 + 2?' --model apple --temperature 0 --max-new 32
tuff prompt 'Say hello.' --model apple --system-prompt 'Answer briefly.'
tuff serve --default-model apple.on-device
```

`TUFFCLI --model apple.on-device --chat-prompt 'Hello'` also works directly.
`--messages-file` accepts a JSON array of text messages with `role` and `content`.
Leading system/developer instructions and alternating user/assistant history are
preserved; the last message must be a user message.

`GET /v1/models` includes `apple.on-device` only while the device reports it ready.
Use that ID (or `apple`) in the API's model field. `default` also works when the
server default is Apple. Both streaming and non-streaming text requests use the
existing API routes, queue and cancellation behavior.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"apple.on-device","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0,"stream":true}'
```

Supported controls are instructions, text history, temperature and output-token
limit. Images, tools, assistant-prefix continuation, reasoning controls, explicit
sampling filters, seeds, stop strings and repetition penalties are rejected.
Apple controls the context window and model residency; TUFF memory allocation,
prefill and speculative-decoding options do not apply.

The Apple SDK does not expose token usage. Chat-completion usage counts are
`null`, with `token_counts_available: false`; Responses usage is `null` and
Messages counts are `null`. TUFF does not estimate counts or throughput from
streamed text fragments.
