# TUFF 11.4.1

## Turning Web on costs fewer prompt tokens

- With Web or Files on, TUFF adds a few instructions and the tool descriptions to the start of the chat, and the model reads them before your first message. That text was longer than it needed to be. The instructions are now five short sentences, and each search tool is declared with a one-line description and its one required field.
- Counted with each model's own tokenizer, turning Web on used to add 284 tokens on Gemma models, 318 on Flint and 463 on Qwen, Ornith and Bonsai. It now adds 155, 204 and 347. Most of what is left on the Qwen-family models is the tool format their own chat template spells out, which TUFF does not change.
- A "hi" with Web on went from 294 prompt tokens to 165 on Gemma E2B, and from 476 to 360 on Ornith 9B.
- Search still works the same way. With the same canned results and seeds, Gemma E2B, Gemma 4 26B, Ornith 9B and Bonsai 2B searched, answered from the results, ignored a result that tried to give them instructions, and did not search for "hi", with both the old and the new wording. Gemma E2B cited its source with the new wording and had not with the old. Flint 40B was checked on one question and one "hi".
- How short the descriptions could go was decided by testing, not by eye. A first draft described the page reader as "Read a web page from the results or the user." With five realistic results to work through, Ornith 9B then called a tool named `web_read_webpage`, which does not exist, in three runs of five. Naming `web_search` in that one description, as the old text did, fixed it: ten runs of ten were clean on Ornith 9B, twenty of twenty on Gemma E2B, ten of ten on Bonsai 2B and six of six on Gemma 4 26B.
- The rules that keep your files, images and folder results out of search queries are unchanged. They were always enforced when a search runs, not by asking the model. The model is still asked to keep private details out of search queries, in one sentence instead of two. A model can still ask for fewer results; the option is just no longer described to it.
- The first message you send in an existing chat with Web or Files on reads that chat again once, because its opening text changed.

## An answer no longer fails right after its tools ran

- When a tool finished instantly, the next step of the answer could start before the previous one had let go of the model, and the answer failed with "A generation is already running." It showed up once in about 150 answers in a test harness with instant search results; with real searches the gap is far too short to hit. The answer now waits a moment and tries again, up to three times.

## Compatibility

This patch uses the existing inference engines. Models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
