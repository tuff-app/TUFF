# TUFF 11.3.0

## Faster replies when sampling

- Every reply written with a temperature above zero, which is what Chat does by default, picks each token from a probability for every entry in the vocabulary (262,144 on Gemma). Turning the model's scores into those probabilities ran on 256 GPU threads, twice over the whole vocabulary, about a millisecond per token. The two simple passes now run one thread per entry. The sum that normalizes them still runs on the same 256 threads in the same order, because its result depends on that order.
- Measured on a 16 GB M2 MacBook Air, temperature 0.7, the same seed, engines alternated with pauses so the Mac stayed cool: Gemma 4 E2B 48.8 to 51.2 tokens per second (4.8 percent), Gemma 4 E4B 27.9 to 28.4 (2 percent), Bonsai 2 2B 76.1 to 78.2 (2.7 percent). Ornith 1.5 9B measured the same before and after. The saving is a fixed part of a millisecond per token, so it shows most on small models, and more on Gemma, whose logit cap made the old pass costlier.
- Greedy replies (temperature 0) and prompt reading are unchanged.
- Answers are the same. The new path produced the previous one's probabilities byte for byte in every test (vocabularies from 1 to 262,144 entries, with and without Gemma's logit cap), and the same seed gave the same text on the models measured, all on an M2. Other Apple GPUs were not tested, so the engine compares the two paths on the Mac's own GPU before it first samples, on four sets of scores, and keeps the previous path if a single byte differs.

## Compatibility

Gemma and Qwen-family models (Qwen, Ornith, Bonsai) download the 11.3.0 inference engine the first time they load. Flint, GPT-OSS and MiniMax stay on 11.0.0. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
