# TUFF 9.0 local builds

The host app, CLI and HTTP server do not link model-family inference engines.
Generate authenticated family packages before trying inference from a checkout:

```sh
bash Scripts/package_engines.sh dist/local-engines dist/local-private/signing-key 1.0.0
export TUFF_ENGINE_CATALOG_PATH="$PWD/dist/local-engines/engine-catalog.json"
swift build -c release
```

The private key stays outside the package output and must never be published.
The local catalogue references the generated archives. A model install or load
acquires only its family engine; compatible models share it. Installed engines
work offline. Advanced model settings offer verification, repair and unused-engine
cleanup. Dependencies and running helper leases prevent removal.

For an isolated app build and test data:

```sh
export TUFF_APPLICATION_SUPPORT_ROOT="$PWD/dist/local-data"
TUFF_LOCAL_BUILD=1 TUFF_BUNDLE_IDENTIFIER=com.rexmhall09.TUFF.local9 \
  bash Scripts/package_app.sh 9.0.0 dist/local-app
```

This preserves the installed app and normal TUFF data. The local build disables
automatic Sparkle updates. Launch its executable with the same environment so
the GUI and child helpers use the isolated data directory. Do not enable its
Background API login item during isolated qualification.

Release maintainers can add a fourth argument to `package_engines.sh`: the
GitHub HTTPS download directory for that release. This produces a public catalog
with no local file root. Upload the five `.tar` archives and their five signed
`.json` manifests beside the app archive; the app embeds the catalog and public
verification key. Private keys stay outside every distribution directory.
A release build must pass the existing release signing key as the second
argument. The packager refuses to create a new key or use any other key for a
release, because changing keys makes installed engines fail verification.

Each engine package version is immutable. After changing helper code, use a new
version and a new output directory. A host build must run with
`TUFF_ENGINE_FAMILY` unset; the packaging script handles its family builds.

Auto guidance reports calculated allocation and execution mode, not promised
speed. All-layer expert caches, projection groups, state, scratch and overlapping
allocations are charged. Oversized dense models can stream layers but decode may
be very slow. Ordinary request timings refine comparable choices; **Tune Auto on
this Mac** runs up to three neutral probe candidates for 15 seconds, then restores
the selected session. It saves aggregate timings, never probe chat contents.
Advanced settings offer manual resident, partial and streamed projection modes;
Auto preserves those controls while active. Reusable conversation state is capped
by the remaining safe allocation budget rather than forcing weights out of RAM.
The host negotiates these features before loading a helper.

Weight streaming reads existing model files; it does not rewrite weights during
decode. SSD endurance is primarily associated with writes and erase cycles.
Memory pressure can nevertheless cause macOS to write swap pages. Auto rejects
observations with pressure or paging, using cumulative VM swap-outs as well as
net swap occupancy so freed swap pages cannot hide ongoing writes. These
counters are system-wide and cannot attribute writes exclusively to TUFF.
When the minimum all-layer expert cache alone exceeds a quarter of physical
RAM, the calculated starting budget falls to half of RAM, with the minimum
useful expert count and a small prefill allocation. This leaves more headroom
for streaming and other applications. The same model naturally uses the normal
budget on a roomier Mac. This conservative heuristic is not a hardware guarantee.
The local performance sweep stops a trial after 256 MiB of additional system
swap-outs; that is a conservative test limit, not an SSD damage threshold.
Flash Next defaults to 16 expert slots after repeated 16 GB tests showed no
swap-outs, while 32-slot trials exceeded that limit. Larger caches remain
available for measured tuning and manual control.

An isolated 16 GB M2 Air Gemma 4 E2B Q4 text run with the current local
10.0.0 CLI, a 36-token prompt and greedy decoding measured 47.06 decode
tokens/s over 116 generated tokens and 36-token prefill in 0.36 s. The
matching MLX checkpoint measured 44.14 decode tokens/s over 124 generated
tokens and 24.10 prefill tokens/s; the generated text differed, so these are
performance measurements, not an equivalence test. Swap in use before the
TUFF run was approximately 2.49 GiB. These are single observations, not
throughput guarantees. Logs: `scratch/v10-local-review/speed-existing-e2b-current.log`
and `speed-existing-e2b-mlx.log`.

The API's disk prefix cache is bounded and saves a previously unseen instruction
prefix once, rather than writing generated-token state continuously. Set
`TUFF_PREFIX_DISK_CACHE_MB=0` to disable it. Isolated builds keep this cache inside
their scoped data root. SSD health or lifetime cannot be guaranteed by these
software checks, and the operating system controls swapping and thermal limits.

Chats use the same cache so a chat reopened after a restart doesn't read its
whole history again. A chat of 1,024 tokens or more is saved where its newest
message begins, then saved again only once it has grown by a quarter. Its
state is deleted with the chat, or when you edit an earlier message. Set
`TUFF_SAVED_CHATS=off` to keep chat state in memory only.

Raising macOS's GPU wired-memory limit is not a default TUFF setup step.
TUFF runs its own Metal helpers, rather than MLX's allocator. A larger system
allowance does not add physical RAM or make disk-bound streaming faster.
Apple distinguishes the recommended GPU working set from total allocations;
the app should manage its active resources and leave system headroom.
MLX documents a manual system override, but explicitly requires a wired limit
below total memory. Consider such an override only for a demonstrated residency
bottleneck on a machine with adequate headroom, with before/after measurements.
See [Apple's Metal guidance](https://developer.apple.com/videos/play/tech-talks/10580/)
and [MLX wired-limit documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html).

Catalog `tuff prompt` commands use calculated Auto by default. The packaged
`TUFFCLI` supports `--memory-auto`; explicit cache/residency controls remain
manual. CLI and API Auto use the same engine-and-volume-specific history as the
app, and record their own completed runs into it, so what one learns the others
use. SIGINT, SIGTERM and
SIGHUP cancel the CLI request and interrupt its child helper.

The canonical serial gate is `bash Scripts/test.sh`. `TUFF_LOCAL_BUILD=1 Scripts/check.sh --source-only`
also runs the model-free script checks. Package validation uses
`python3 Scripts/test_packaging.py --app PATH` and the updater fixture checks.
Real model qualification is separate; passing model-free tests is not a claim
that every model or Mac has been tested.

Local mode validates the real existing Homebrew pin while allowing its published
version to precede the unreleased source version. It does not change the cask,
release feed, public checksums or GitHub assets.

## Model catalog

MoE, Dense and Legacy sections can be collapsed. Cards show download size,
minimum RAM and actions. Engine and residency details stay here rather than
filling every model card. Calculated allocations are estimates, not measured
process memory or guaranteed speed.

Gemma 4 26B-A4B Heretic Q4 uses the same Gemma engine as the ordinary model, with its own
checkpoint, settings, install and image pack. The pinned mlx-community conversion
comes from coder3101's abliterated model. Routed experts and attention are Q4;
shared MLPs and routers are 8-bit. Auto accounts for those larger shared weights.
Gemma 12B keeps its existing QAT weights and QAT catalog label.

For a large mandatory expert cache on a small Mac, calculated Auto starts with
streamed shared projections and a short context, with conversation snapshots
disabled. When minimum expert slots are selected, the streamed path releases
prior-layer expert buffers after GPU completion instead of keeping all layers
resident. Roomier devices retain their caches. More
residency needs comparable healthy measurements. These defaults reduce paging
risk; they do not guarantee that the model will fit beside every other app.

## Advanced custom checkpoints

In Advanced settings, expand Add a custom checkpoint, enter a name, choose the
matching supported architecture, and select a complete verified `.gturbo` folder.
TUFF checks the engine's tensor/quantization contract, declared file hashes,
receipt binding, tokenizer format and token ID bounds before adding the entry.
The same signed family engine is acquired and shared.

This first version imports text checkpoints only. It does not convert loose
safetensors/GGUF files, add new engines or import custom image packs. The local
API offers each custom checkpoint under a name made from its own, such as
`my-tune` for "My Tune". Repack the source using the CLI first and run
`TUFFRepack verify-install` to produce its receipt. Keep the chosen folder in
place. A changed manifest requires removing and importing the entry again.
Removing the entry removes TUFF's link and registration, leaving the source
weights and saved chats on disk. Custom entries have their own settings.

For constrained large-model streaming, filesystem caching is selectively
bypassed alongside the bounded TUFF caches, following
[Apple's file-cache guidance](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/FileSystem/Articles/FilePerformance.html).
It remains enabled for routed-expert working sets that fit comfortably on
roomier devices. This is a memory policy, not a guaranteed speed or SSD-write limit.
