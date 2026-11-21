# swift-transformers (vendored)

This is [swift-transformers](https://github.com/huggingface/swift-transformers)
1.3.3, revision `2fa33e1f5e7131a7fc64c28e6d161dcec0d24820`, under its
[Apache 2.0 license](LICENSE). Only the `Hub` and `Tokenizers` libraries are
kept, which are the two TUFF uses. `Package.swift` is upstream's
`Package@swift-6.1.swift` with the other targets and the tests removed.

Three changes make loading a tokenizer about twice as fast. On a 16 GB M2,
Ornith 1.5 9B's tokenizer went from about 950 ms to 400 ms and Gemma 4's from
about 1,450 ms to 620 ms. A model load spends most of its time here once its
weights are verified.

**Merge pairs compare their exact bytes.** `BytePair` in
`Sources/Tokenizers/BPETokenizer.swift` hashes and compares the UTF-8 bytes of
its two strings instead of using `String`'s Unicode-equivalence comparison,
which normalized every non-ASCII token while the merge table was built. This
is how the reference tokenizers compare merges. Every Qwen and Ornith
tokenizer encodes exactly as before. Gemma 4's changes for 9 of its 262,144
tokens, in Arabic and Bengali, which upstream merged wrongly because two
different merges were canonically equivalent; all 9 now match Hugging Face's
`tokenizers`. The merge table is also sized up front.

**Strings convert without a grapheme walk.** In
`Sources/Hub/BinaryDistinct.swift`, `BinaryDistinctString.init(_ str: String)`
takes the string's UTF-16 code units directly. Upstream walked every string
by `Character`, which segments it into grapheme clusters, while parsing
`tokenizer.json`. The `NSString` initializer is unchanged.

**`nsString` builds the NSString directly** from the UTF-16 code units, with
no Swift `String` in between.

A fourth change is for correctness. **`ignore_merges` is honored.**
`BPETokenizer` reads `model.ignore_merges` from `tokenizer.json`, and when it
is set, a pre-tokenized word that is already in the vocabulary becomes that
token without running the merges, as Hugging Face's `tokenizers` does.
Upstream ignored the setting. Inkling 2's tokenizer sets it; with it ignored,
about one word in a hundred encoded differently from the reference. No other
tokenizer TUFF installs sets it, so they encode as before.

To update, copy the new release's `Sources/Hub`, `Sources/Tokenizers` and
`Package@swift-6.1.swift` over this directory, trim the manifest the same way,
then reapply the four changes. Check them by encoding every vocabulary entry
of each installed tokenizer with both versions and comparing.
