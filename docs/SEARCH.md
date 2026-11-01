# Web and Files search

The globe and folder buttons under the message box let the model search the
web or folders you choose. Both are off until you click them on, and they
light up while on. Right-click the folder button to choose folders. Answers
show what was searched and cite their sources with numbers you can click.

Search results can be wrong, or written to mislead a model. They can't turn
on tools or reach other folders, but check the sources on anything that
matters.

## Web

DuckDuckGo works with no key. Brave Search and Tavily work with your own key,
which is kept in your Keychain and never shown to the model or saved in
chats.

What leaves your Mac: the search query, sent to the provider you picked, and
requests for the pages the model reads. Pages are limited to 2 MB and five
redirects, and local or private network addresses are refused.

On macOS 26 and newer with Apple's `container` tool installed, **Settings >
Search** can run web search and page reads in a sandbox, a small virtual
machine, so a page never reaches TUFF as anything but finished text. It is off
until you turn it on. See [Web sandbox](WEB_SANDBOX.md).

Once a chat includes your own files, images or folder results, web searches
can only use words you typed in your current message. That keeps private
material out of search queries.

DuckDuckGo sometimes blocks automated searches. TUFF tells you when that
happens instead of working around it. **Settings > Tools** can pause web
search automatically while you're offline.

## Files

Files only reads folders you add. TUFF keeps a small text index in
`~/Library/Caches/TUFF/LocalSearch` (no embedding model) covering text,
Markdown, code, CSV, JSON, YAML and PDF. It skips hidden and build folders,
links, and big files, and pauses while the model is answering. Removing a
folder deletes its part of the index. On macOS 26 and newer, **Settings >
Search** can also read the folders in a second sandbox with no network, so a
harmful file never reaches TUFF directly; it is off until you turn it on. See
[Web sandbox](WEB_SANDBOX.md).

## Limits

Per answer: 4 rounds of tools, 3 calls per round, 8 web requests, 6 file
searches and 120 seconds of tool time. Stop cancels everything.

Search has been checked with real tool rounds on every model except GPT-OSS
120B and MiniMax M2.7. Those support tools but haven't been tried yet.

### Workspace coding tools

Local Files alone never grants raw file access. Choose a workspace from the
folders in the Files panel (the default is None); only then can the model list
directories and read UTF-8 files there, hidden files included, alongside the
existing search across all selected folders. Write and edit, command execution,
and command network access have separate switches; all three default off.
The workspace and switches apply to the next answer, and their controls are
unavailable while it runs. A saved tool round records what happened; it does
not grant a later answer permission. File reads and writes use relative paths,
reject symlinks and hard links, and are limited to 128 KiB. Writes replace files
atomically and refuse any `.git` path component (checked case-insensitively).
Edits require exactly one match for the old text.

Enabling write or command access has a residual risk: the model can write
files that other programs run later, outside any sandbox, such as `.envrc`,
`Makefile`, `package.json` scripts, `.vscode/tasks.json`, shell scripts or
build and editor configuration. TUFF blocks `.git` (hooks, `core.fsmonitor`
and other configuration) in both tools and commands, but cannot block these,
so review changes to such files before running a build, opening the folder in
an editor that trusts it, or letting a direnv-style tool load it.

Answers with write or command access can use up to 12 tool rounds, subject to
the existing total tool-time and context limits.

Commands run under a default-deny macOS sandbox with the selected workspace as
its working directory. The write switch controls whether commands can modify
it, except `.git` (any case, at any depth), which commands can never write.
Networking is denied unless the network switch is enabled, in which case
commands may make outbound TCP and UDP connections, except to the sandbox's
`localhost`. In tests that covers `127.0.0.1` and this Mac's own LAN address,
so TUFF's own server and local development services are not reachable, even
ones listening on all interfaces. Other hosts on the local network and the
internet are reachable, so a command with network access can still reach LAN
services such as routers, NAS devices and other computers. The sandbox allows reads
of macOS runtime files and Apple developer toolchains, so the installed Apple
Git and Python tools can run. Other installed tools may be unavailable until
explicit support is added. The command process receives a minimal environment,
not the user's shell environment. TUFF refuses commands when it cannot inspect
the workspace or finds a hard-linked regular file, because a hard link can
refer to an inode outside the workspace. Symbolic links do not grant access to
outside files. Like other local file operations, this assumes an unsandboxed
process is not concurrently changing the workspace to defeat the preflight.
The Local Files indexer opens each file without following links, relative to
the folder, and requires the same regular, singly-linked file its scan saw.

A command is limited to 30 seconds and 64 KiB of output. Child processes also
inherit CPU, open-file, file-size and user-process limits. Since macOS counts
processes across the account, an unusually busy account can prevent a command
from starting children. Cancellation, timeout and normal exit end the command's
process group. The sandbox denies the `setsid` and `setpgid` system calls, but
`posix_spawn` can ask the kernel for a new session or group directly, so a
descendant can leave the group. Each run's sandbox profile therefore carries a
unique marker, and when the command ends, is cancelled or times out, TUFF
sweeps every process of the same user whose sandbox carries that marker and
kills it, wherever it sits in the process tree. This uses the private
`sandbox_check` call; if it is unavailable, commands do not run. A process
that reaches uninterruptible kernel state, or one started by another user,
cannot be killed by the sweep. A macOS sandbox startup failure fails the
command instead of running it without protection. The system `sandbox-exec`
interface is deprecated, so compatibility must be checked on future macOS
releases. Commands run only after the corresponding switch is enabled and do
not ask for per-command confirmation.

After local files, images or workspace output enter an answer, web searches are
limited to text you wrote: the whole message, one of its lines, or a phrase in
double quotes, at least four characters long, each once. A failed workspace
tool also counts as local content.
