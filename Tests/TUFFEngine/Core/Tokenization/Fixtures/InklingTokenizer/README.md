# Inkling tokenizer fixture

`tokenizer.json` is a 258-entry byte-level vocabulary with Inkling 2's
control tokens added, so prompt and decoder tests run without the 26 MB
tokenizer. Its pre-tokenizer, decoder and `ignore_merges` setting are the
real ones.

`template_cases.json` holds conversations rendered by the publisher's
`chat_template.jinja` (through `transformers`), and `tokenizer_cases.json`
holds strings encoded by Hugging Face `tokenizers` with the real
`tokenizer.json`. The second is only compared when the real tokenizer is
available: set `TUFF_INKLING_TOKENIZER_DIR` to its folder.

`greedy_reference.json` and `window_reference.json` come from
`Scripts/inkling_reference.py` run on the real 4-bit checkpoint: twelve greedy
steps of a short chat, and the first layer's attention output for a prompt
read with a 16-position window. `InklingReferenceTests` compares the engine
with them when the installed model is available.
