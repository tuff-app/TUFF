# Local server

TUFF can serve your installed models to other apps, scripts and coding
agents through an OpenAI-compatible API. By default it only listens on
`127.0.0.1` and needs no key. It also refuses requests whose `Host` or
`Origin` isn't `127.0.0.1`, `localhost` or `[::1]`, so a web page can't reach
it through DNS rebinding. A proxy in front of it has to pass the original
`Host`, or send the API key described below.

## Start it

**In the app:** turn on **Background API** on the Server screen and allow the
login item when macOS asks. It then runs whether TUFF is open or not. Pick a
default model, a port, and how long an idle model stays loaded.

![The Server screen](assets/tuff-server.png)

**In a terminal:**

```sh
tuff serve --default-model gemma4-e2b --unload-after 300 --port 8080
```

From a clone: `swift run -c release TUFFServer --models-root scratch`.

## Other devices on your network

To use TUFF from another machine, such as Open WebUI on a Docker host, turn on
**Allow other devices on your network** on the Server screen, or start the
server with `--host`:

```sh
tuff serve --host 0.0.0.0 --port 8080
```

`0.0.0.0` listens on every interface, this Mac's loopback included. A specific
address, such as `192.168.1.20`, listens only there, and then this Mac has to
use that address too.

Requests from other devices need the API key, sent as
`Authorization: Bearer <key>` (OpenAI clients call this the API key) or
`x-api-key: <key>` (Anthropic clients). Without it they get 401
`invalid_api_key`. TUFF creates the key the first time it listens beyond
loopback and keeps it in `~/Library/Application Support/TUFF/Server/api-key`,
readable only by you. The Server screen shows and copies it. Delete the file
to get a new one. Requests from this Mac still need no key.

Anyone with the key can use your models, and requests are plain HTTP, so only
do this on a network you trust. macOS may ask whether TUFF can accept
incoming connections the first time; allow it.

## Using it

Point your client at `http://127.0.0.1:8080/v1`. `GET /v1/models` lists the
installed models and their context and output limits. A request for
`default` uses the default model; a model ID loads that one. The short names
`tuff prompt` takes, such as `gemma4` or `qwen36`, work too. Custom
checkpoints you imported in Settings are listed under a name made from their
own, so "My Tune" is `my-tune`. Requests run one at a time, and a full queue
answers 429 with `Retry-After`.

Each model is served with the memory settings it has in Settings > Models.
With Auto on, the server uses the Auto profile you picked. With Auto off, it
uses the Context you chose, as long as that fits this Mac's memory budget;
otherwise it falls back to Balanced. A model you haven't set up gets
Balanced. Changes apply the next time the server loads the model.

The API and Chat share one inference host per user, and with it one loaded
copy of each model. If Chat already has a model loaded, an API request uses it
as it is: the request runs with Chat's context and memory settings, and
answers `context_length_exceeded` if it does not fit. The one difference that
forces a reload is greedy (`temperature` 0) versus sampled generation; the API
then reloads the model with Chat's context kept, but only while the host is
idle, and answers 503 `model_busy` otherwise. A model the API loaded itself is
unloaded again after the idle time you chose.
Image processing needs exclusive model memory. The host releases idle sessions
and idle models when possible, without interrupting active turns. If memory is
still held, the API answers 503 `model_memory_busy`; retry after the other work
finishes or its model is unloaded.

The conversation must alternate the way a chat does: system or developer
messages first (merged into one system prompt), then user and assistant
messages, with each assistant tool call followed by a result for every call.
Consecutive user messages, a user message straight after tool results, and an
assistant message with tool calls and no results are refused with
`unsupported_value`. A final assistant message is continued rather than
answered. `stop` strings are applied by the server and end the reply.

## Endpoints

| Endpoint | |
| --- | --- |
| `GET /health` | Is it up, and which model is loaded |
| `GET /v1/models` | Installed models and limits |
| `POST /v1/chat/completions` | The main route, with everything below |
| `POST /v1/messages` | Messages format, text and function tools |
| `POST /v1/responses` | Responses format, text and function tools |

Chat Completions supports streaming, reasoning, function tools, images (with
an image pack installed) and prompt reuse. Your client runs tool calls; the
server never does.

Unknown fields return `unknown_parameter`, and unsupported values return
`unsupported_value`, so you find out instead of getting silently different
behavior. `chat_template_kwargs` accepts only `enable_thinking` and
`preserve_thinking`.

### Messages and Responses

These are smaller subsets, enough for clients that speak those formats.
Send the whole conversation every time; nothing is stored by ID. Reasoning is
left out of their output, so use Chat Completions if you want it.

They refuse things they can't do faithfully, rather than guessing: text after
a tool call, signed thinking, image blocks, server tools, cache-control,
forced tool choice, `stop_sequences`, stored responses and hosted tools,
among others. Gemma needs tool-only assistant turns, and GPT-OSS allows one
tool call per turn.

They are not drop-in replacements for the default Claude Code or Codex
setups, which send options these subsets refuse.

## Reasoning

Models that reason return it separately from the answer, as
`reasoning_content` (or `delta.reasoning_content` when streaming). The usage
block does not split out reasoning tokens.

## Prompt reuse

The server keeps the last conversation's state, plus up to four others,
within a memory budget of at most 1 GB. Sending a conversation it has seen
continues from where it left off instead of reading everything again.
`usage.prompt_tokens_details.cached_tokens` says how many tokens were reused,
and `prompt_cache_key` helps it find the right conversation first. Under
memory pressure the extra states are dropped.

A new conversation that starts with the same system prompt and tools as an
earlier one (256 tokens or more) skips reading them again. The server also
saves that state to disk the first time it reads a system prompt, so it
survives restarts. The files live in `~/Library/Caches/TUFF/PrefixSnapshots`,
use at most 2 GB, and are only used for the same model, settings and TUFF
version.

`TUFF_CONVERSATION_CACHE_MB` lowers the memory budget (`0` keeps only the
current conversation), `TUFF_PREFIX_DISK_CACHE_MB` lowers the disk budget
(`0` turns it off), and `TUFF_LOG_CACHE=1` logs each reuse decision. The legacy `TFF_LOG_CACHE` name still works.

## Health

`GET /health` answers without loading anything:

```json
{"status": "ok", "mode": "router", "vision": "ready",
 "loaded_model": "gpt-oss-20b", "active_requests": 1, "queued_requests": 0}
```

`loaded_model` is the ID of the model in memory, the same one you'd send as
`model`, or `null` when none is loaded. Check it before sending a request if
you'd rather not trigger a load, or start a second large model, by accident.

## Timings

Every response includes `tuff_timings_seconds`. All values are seconds,
measured inside TUFF, so they don't include the network or how your client
displays the reply. Some spans contain others, so don't add them all up.

The request, in order:

| Field | What it covers |
| --- | --- |
| `body_and_validation` | Reading and checking the request |
| `queue_and_model_load` | Waiting for earlier requests, plus loading the model if it wasn't in memory |
| `preparation` | Preparing the request for the model |
| `stream_setup` | Starting the streamed response |
| `generation_to_first_event` | From the start of generation to the first visible output: reasoning, text or a tool call |
| `time_to_first_event` | From the request arriving to the first visible output. This is the wait a user sees |

Inside generation:

| Field | What it covers |
| --- | --- |
| `prompt_render_and_tokenization` | Turning the messages into tokens |
| `cache_plan` | Deciding how much saved state to reuse, including the three below |
| `cache_lookup_and_bridge` | Comparing the request with saved conversations |
| `cache_capture` | Copying another conversation's state out of the way |
| `cache_restore` | Restoring a saved conversation |
| `prefix_snapshot_load` | Loading a saved system prompt from disk |
| `prefill` | Reading the prompt tokens that weren't reused |
| `decode` | Writing every output token, after the prompt is read |

Where reading the prompt went, when any of it had to be read:

| Field | What it covers |
| --- | --- |
| `prefill_expert_read_wait` | Waiting for routed experts to be read from the SSD |
| `prefill_attention_gpu` | GPU time for attention and routing (the whole layer for a dense model) |
| `prefill_routed_expert_gpu` | GPU time for the routed experts |
| `prefill_shared_expert_gpu` | GPU time for the shared expert, on models that have one |
| `prefill_other_gpu` | GPU time for everything else: combining experts, embeddings and the output head |
| `prefill_gpu_wait` | Time spent waiting for the GPU to finish |

These overlap: experts are read while earlier ones compute. On a 16 GB Mac a
large share of `prefill` is usually `prefill_expert_read_wait`.
`TUFF_PHASES=1 tuff prompt ...` prints the same breakdown in the terminal.

Image requests also report `image_admission` and
`multimodal_render_and_encode`. The cache fields are 0 when that step didn't
run, except `prefix_snapshot_load`, which is left out unless a saved system
prompt was loaded from disk.
`usage.prompt_tokens_details.cached_tokens` says how many prompt tokens were
reused, and `usage.completion_tokens` is the count `decode` produced, so
tokens per second is `completion_tokens / decode`.

## Agents

[OMP](OMP.md) has a tested setup. Big models can take minutes to read an
agent's instructions before they answer, so give your client a long timeout.
