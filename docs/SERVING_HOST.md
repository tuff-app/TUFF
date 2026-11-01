# Shared serving host (10.0.0)

One serving pool per profile serves the app window, the foreground server and
the optional Background API. Model weights, helper processes and GPU
scheduling live in a dedicated `TUFFServer --private-host` process, so
quitting the app does not stop API work and stopping the API does not stop the
app's chats.

## Lifetime
- Started on demand by the first client (`ServingHostRuntime.connectOrLaunch`)
  in its own session (`POSIX_SPAWN_SETSID`). It is **not** a login item or
  launchd registration.
- Racing launches are safe: the profile lease (`serving-host.lock`) admits one
  host; losers exit immediately and every caller retries its connection
  (re-launching if the host it started has gone).
- Exits by itself 20 s after the last client disconnects with no request,
  load, unload or session cleanup in flight. Active work is cancelled only
  when its own client disconnects.
- A host from another build is never attached to (`incompatibleBuild`).

## Profile and isolation
The profile is keyed by user, bundle identifier and Application Support root
(`TUFF_APPLICATION_SUPPORT_ROOT` honoured). Socket, lock, token and client IDs
live under that root (`TUFF/Server`, socket in a private temp directory). A
second bundle identifier pointed at the same root fails closed
(`profileIdentityMismatch`). A local preview therefore cannot attach to a
production host or process. Local builds may set
`TUFF_SERVING_HELPER_EXECUTABLE` (with `TUFF_LOCAL_BUILD=1`) to launch a
decode helper directly.
The private socket uses `serving-host-token`, separate from the HTTP
`control-token`. Each API adapter instance has its own client identity, and
concurrent operations on an adapter share its in-flight connection attempt.


## Protocol
Authenticated same-UID unix socket, 0600, token + profile identity + protocol
version. Clients send model identifiers (never paths). Typed commands
(`prepare`, `start`, `cancel`, `endSession`, `unloadModel`, `hostStatus`) and
routed events share one multiplexed connection (`PrivateServingIPCClient`):
per-stream buffers are bounded (a stalled consumer is failed and its
generation cancelled), stale cancels are ignored, session cleanup failures are
reported (`sessionEndFailed`) and leave the session owned for retry.
`hostStatus` is answered live (loaded models + configuration, active
requests, whether an `api` client is attached); there is no static
"public API enabled" flag.

## Memory
The coordinator evicts idle models (never ones with requests, loads or session
cleanup) and retries once when a load fails because another resident model
holds memory. Text and image requests reclaim idle sessions and idle other
models when their memory reservation fails, then retry once; session history is
replayed from the transcript on the next turn. The app ends sessions when chats
are deleted or rebound to another model, and releases idle sessions under
system memory pressure. Failed cleanup retains the session identity for retry.
Concurrent preparations of the same model wait for the current load and then
check configuration compatibility. A lost helper is removed with its sessions,
so the next preparation starts a fresh helper.
