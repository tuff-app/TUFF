# How TUFF works

You don't need any of this to use TUFF. It's here for anyone who wants to
change the engine or just wonders how a 111 GB model fits on a 16 GB Mac.

## Expert streaming

Most big open models are mixture-of-experts (MoE) models. Each layer has many
small networks, the experts, and a router picks a few for each token.
GPT-OSS 120B has 128 experts per layer and uses 4, so a token only touches a
small part of the model.

When you install a model, TUFF repacks it into its own `.gturbo` layout:

- **Shared weights** (attention, embeddings, routers) are memory-mapped from
  disk.
- **Experts** are stored so any one can be read with a single `pread`. Each
  layer gets a fixed number of cache slots. When the router asks for an
  expert that isn't cached, TUFF reads it from the SSD into a slot, evicting
  the least frequently used one.

So memory use depends on the number of slots, not the model's size, and
speed depends mostly on how often an expert has to come off the SSD. Two
things help:

- **Lookahead.** While one layer is still computing, TUFF guesses the next
  layer's experts and starts reading them early. A wrong guess only wastes a
  read; the layer still routes on its real input. It switches itself off if
  its guesses are mostly wrong.
- **Batched prefill.** Prompts are processed in chunks, so each expert is
  read once per chunk instead of once per token.

Dense models have no routed experts. Auto keeps every projection resident when
it safely fits. Oversized dense packs can retain selected whole layers and stage
the remaining layers through one bounded buffer after prior GPU work completes.
Reading most weights for every token can be very slow. Affine MoE models can also
stage their attention/shared projections; lookahead is disabled while staging.
GPT-OSS row-access projections currently remain resident.

## Memory planning

Before loading, TUFF estimates everything it will allocate: expert slots, the
KV cache for the chosen context, prefill scratch and a safety margin.
**Auto** settings must fit in 75% of the Mac's memory. The estimate is
deliberately cautious and includes resident mappings, all expert layers, row
tables, state, scratch, staging, retained snapshots and transition overlap.
Sustained pressure reduces the budget. Repeated comparable local timings can
favor a smaller cache; one fast warm run does not establish SSD speed.
See [`AppAutomaticLayoutPlanner.swift`](../Sources/TUFFApp/Core/Configuration/AppAutomaticLayoutPlanner.swift).

## Processes

The app ships a lightweight host executable whose role depends on its invoked
name ([`ProcessRole.swift`](../Sources/TUFFApp/Mac/App/ProcessRole.swift)):

| Started as | Does |
| --- | --- |
| `TUFF` | The app (or the benchmark, with `--benchmark`) |
| `TUFFServer` | The Background API |
| `TUFFCLI` | Launches the installed helper for command-line inference |

Family inference helpers and shaders are authenticated external packages,
installed with their first dependent model and shared by compatible models.
The host, CLI and HTTP server contain no family engine. The installed
`TUFFInferenceEngine` handles chat sockets, CLI requests and API RPC. Required
model bindings and live helper leases protect engines during cleanup. At
launch the app removes engine versions that an update replaced, once no model
is bound to them.

Chat talks to the decode service over a Unix socket, so a GPU problem can't
take your chats down with it.

## Code map

| Directory | What's there |
| --- | --- |
| [`Sources/TUFFModelCatalog`](../Sources/TUFFModelCatalog) | Sources, hashes, memory floors, defaults and Legacy maintenance status. |
| [`Sources/TUFFRuntimeManagement`](../Sources/TUFFRuntimeManagement) | Signed engine manifests, downloads, installation, repair, bindings and leases. |
| [`Sources/TUFFRuntimeDescriptions`](../Sources/TUFFRuntimeDescriptions) | Architecture and runtime contracts without Metal. |
| [`Sources/TUFFModelMetadata`](../Sources/TUFFModelMetadata) | Pack metadata, receipts and calculated weight layouts. |
| [`Sources/TUFFFormat`](../Sources/TUFFFormat) | The `.gturbo` format. |
| [`Sources/TUFFRepack`](../Sources/TUFFRepack) | Download, verify and repack, with resume. |
| [`Sources/TUFFEngine`](../Sources/TUFFEngine) | Metal setup, expert streaming, forward passes, prefill, KV cache, sampling, tokenizers, and the shaders. |
| [`Sources/TUFFServer`](../Sources/TUFFServer) | The HTTP server and API adapters. |
| [`Sources/TUFFApp/Core`](../Sources/TUFFApp/Core) | App state, chats, installs, search tools and benchmarks. No UI, well tested. |
| [`Sources/TUFFApp/Mac`](../Sources/TUFFApp/Mac) | SwiftUI views. |
| [`Sources/TUFFValidation`](../Sources/TUFFValidation) | CPU references the Metal kernels are tested against. |
| [`Scripts`](../Scripts) | Tests, packaging, release checks and measurement tools. |

Tests mirror `Sources` under `Tests` and run without a model: kernels are
checked against CPU references, and each model family against toy models and
tokenizer goldens.

## Switches for measurements

These aren't app settings; they exist so changes can be compared.

| Variable | Effect |
| --- | --- |
| `TUFF_PHASES=1` | Print expert-cache and timing counters. |
| `TUFF_EXPERT_LOOKAHEAD=off` | Turn off lookahead. |
| `TUFF_SMALL_BLOCK_PREFILL=on` | Experimental small-block prefill (Gemma 26B and Flash Next). |
| `TUFF_SHARED_EXPERT_OVERLAP=on` | Overlap the shared expert with routed reads (same two models). |
| `TUFF_KERNEL_GROUPS=combined` | Compile every shader at load, as before 8.0. |
| `TUFF_LOG_KERNELS=1` | Log which shader groups were compiled. |
| `TUFF_TOKENIZATION_CACHE=off` | Turn off the tokenization cache. |
| `TUFF_CONVERSATION_CACHE_MB=N` | Lower the saved-conversation budget. |
| `TUFF_LOG_CACHE=1` | Log conversation reuse decisions; `TFF_LOG_CACHE` is also accepted. |
| `TUFF_SAVED_CHATS=off` | Keep chat state in memory only, never on disk. |
