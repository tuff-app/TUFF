# TUFF 11.2.2

## Old inference engines are removed on their own

- Each update that ships a new inference engine used to leave the previous one on disk, about 15 MB each, until you pressed Remove Unused Engines in Settings > Advanced.
- TUFF now removes them at launch, after it has checked for the engines your installed models need. A version goes only when it is older than the one this release of TUFF installs, no installed model is bound to it, nothing is running it, and nothing in that engine family's folder has changed for at least a day.
- A version kept as a model's fallback stays until that model has answered on the newer engine and been loaded again. Engines for a model whose folder is missing, for example on a drive that is not plugged in, are left alone, and so is anything a newer TUFF build installed. Remove Unused Engines still does the fuller cleanup.

## Loads no longer fail when two TUFF processes start at once

- The app, `tuff prompt`, `tuff serve` and the Background API share one record of which engine each model uses. A process that found another one using it gave up at once, and its model load failed with `packageBusy`. It now waits for the record, up to about two seconds.
- The record is also held for less time: engine files are checked before it is locked, and it is released before old engine files are deleted.
- Measured with four `tuff prompt` runs started together, four times over, on a 16 GB M2 with Gemma 4 E2B: on 11.2.1, 12 of the 16 failed with `error: packageBusy`. On 11.2.2 all 16 answered.

## Compatibility

This patch uses the existing inference engines. Models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
