# Web sandbox

By default TUFF fetches search results and web pages itself and reads them
(HTML, PDF) in its own process. The web sandbox moves that work into a small
Linux virtual machine, so a hostile page or PDF meets the VM's limits and not
your Mac's files. It is **off by default**.

## Turn it on

**Settings > Tools > Run web search and page reads in a sandbox.** The row is
shown on macOS 26 and newer. It needs Apple's
[`container`](https://github.com/apple/container/releases) tool (version 1.5 or
newer), installed at `/usr/local/bin/container`. If it is missing, the status
under the switch says so and links to Apple's releases page.

TUFF starts the sandbox when you turn the switch on and when the app opens with
it on. TUFF stops it after 10 minutes without web use; the next message that
uses the web starts it again, and a web request that finds it stopped waits
for that start. A sandbox that failed to start or stopped answering stays
failed, with the reason shown under the switch, and is tried again at your
next message. It stops when you turn the switch off and when TUFF quits. The
first start can download the image.

The switch applies from the next message. An answer that is already running
keeps using the sandbox; if you turn the switch off meanwhile, its remaining
web requests fail with a message instead of going out directly.

## What changes

With the switch on, every web search and every page the model reads goes
through the sandbox:

- Searches (DuckDuckGo, Brave, Tavily) are sent by the sandbox, which reaches
  only those three sites. TUFF still reads the result list.
- Pages and PDFs are fetched and turned into text inside the sandbox. TUFF
  receives only the text and the title, shortened to 200,000 characters, with
  control and invisible characters removed again.
- There is **no fallback**. If the sandbox is not ready, web search and page
  reads fail with a message and nothing is fetched directly. Turn the switch
  off to use the normal path.

Every request is signed with a secret made at each start, and an answer without
the matching signature is thrown away, so another program on the Mac cannot
pass for the sandbox. Local file search has its own switch and its own
sandbox, described below.

## What it protects, and what it does not

It protects against a page or PDF that tries to exploit the parser, to fill
memory or disk, or to reach other machines: the VM has a read-only filesystem,
runs without privileges, and its firewall refuses private ranges and this
Mac's own addresses (TUFF passes them in at each start).

It does not hide your searches or which sites you read: they still leave
through your connection. **DNS lookups go through your Mac**, so your Mac sees
which names are looked up. **API keys pass through the sandbox**: with Brave
or Tavily, the key goes from TUFF to the sandbox and from there only to that
provider; TUFF asks the sandbox to prove it is still the running one before it
sends a key. For every other request it does the same when the sandbox has not
given a signed answer for 5 seconds, so after a quiet time, or after the
sandbox has died and another program has taken its port, even a search query
or a folder id is not sent before that check. Other machines on your network and addresses your Mac gains after
the start are not covered by the VM's block list. See
the [sandbox README](https://github.com/tuff-app/sandbox#readme) for the full list.

## File search in a sandbox

**Settings > Tools > Files > Read my search folders in a sandbox too** is a
separate switch, off by default, with the same macOS and `container`
requirements. With it on, the folders you add for file search are read by a
second sandbox, a VM with **no network at all**, instead of by TUFF:

- The VM has each folder mounted **read-only**. It lists the files and turns
  text files and PDFs into passages; TUFF gets only paths, sizes and plain
  text, removes control and invisible characters again, and keeps the index on
  your Mac as before. A harmful PDF or text file meets the VM's limits and not
  your Mac's files.
- TUFF still decides what is indexed, as far as paths go. Hidden files,
  build and dependency folders, links, packages, deep folders and big files
  are skipped by TUFF's own rules whatever the VM lists, and each file is
  looked at on the Mac again before and after it is read; a file that
  changed meanwhile waits for the next update. TUFF cannot check the *text*
  the VM returns, though: a VM taken over by a harmful PDF could return other
  files from the same folder as the text of a file TUFF accepted.
- The VM runs while TUFF indexes. It starts when indexing starts, is started
  again with the new folders when you add or remove a folder, stops after 10
  minutes without use and when TUFF quits, and stops when you turn the switch
  off. Searching your index does not need it.
- There is **no fallback**. If the VM cannot start, indexing stops with the
  reason shown under the folder, and TUFF does not read the files itself.
  Turning the switch on or off re-reads the folders with the other reader;
  unchanged files are not read again. As without the sandbox, TUFF looks for
  new and changed files when you press **Update Index** and when the folders
  or the setting change, not while you edit files. If the VM stopped
  answering (for example after it crashed), the next update notices and
  cleans up, and the one after starts it again.
- macOS asks whether `container-runtime-linux` may read folders such as
  Documents or Desktop. Allow it, or file search cannot read that folder; the
  message under the folder says where to change it (**System Settings > Privacy
  & Security > Files and Folders**). If macOS refuses one folder, the other
  folders are still read (TUFF finds the folder by starting the VM with each
  one alone, up to 8 folders; with more, none is read and the message names
  the folders in Documents, Desktop, Downloads and similar places). A folder whose path contains a comma or a
  line break cannot be mounted and is reported as such. Your home folder, or a
  folder that contains it, is refused (it would hand the VM your keys and app
  data); add the folders inside it. Folders inside your home folder are not
  refused, also sensitive ones such as `~/Library`: choose with care. The one
  exception is your SSH keys folder, `~/.ssh`, and anything inside it: it is
  never given to the sandbox.
  At most 64 folders are mounted. In file names, invisible and
  direction-changing characters (such as U+202E) are shown written out as
  `\u{202E}` in file-search results and their sources (on screen, to the model and
  in exports), so a name cannot pass for another file type. Names of documents
  you attach to a chat are written out in the prompt sent to the model only.

What it does not do: the VM can read **everything inside the folders you add**,
including files TUFF would skip, and the Mac's folder permissions (such as 0700)
do not apply inside the VM. A hard link to a file outside the folder looks like
an ordinary file and is readable; links to files and folders are not followed.
Documents you attach to a chat are still read by TUFF itself.

## Developers

Before an image is published, run a locally built one: build it with
`scripts/selftest.sh build` in [tuff-app/sandbox](https://github.com/tuff-app/sandbox), then start TUFF with
`TUFF_SANDBOX_DEV_IMAGE=tuff-sandbox:dev` and `TUFF_SANDBOX_DEV_DIGEST` set to
the image's index digest (`container image inspect tuff-sandbox:dev`, the
`configuration.descriptor.digest` field). TUFF checks that the local image has
exactly that digest and never pulls it.

How to publish a new image and change the digest the app pins:
[PUBLISHING.md](https://github.com/tuff-app/sandbox/blob/main/PUBLISHING.md) in the sandbox repository.

When a `container` command fails, its whole output goes to standard error
(visible when TUFF is started from Terminal) and to the unified log
(subsystem `TUFF`, folder paths private); the message in the settings keeps
only the last lines.
