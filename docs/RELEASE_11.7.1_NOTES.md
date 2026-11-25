# TUFF 11.7.1

## Bonsai writes and reads faster

- Bonsai models write faster. Their 2-bit weights are now read with about half the arithmetic per weight, with exactly the same results. On a 16 GB M2 MacBook Air, Bonsai 2.5 27B went from 7.6 to 9.8 tokens per second and Bonsai 2 2B from 89 to 101, with identical answers.
- Bonsai models read prompts faster. Their prompt projections now run on the same tiled matrix multiply as TUFF's 4-bit models. A 1,549-token prompt on Bonsai 2.5 27B went from 64 to 94 seconds down to 40 to 46, with the same answer.

## Smoother streaming on slow models

- With smooth output on, a slow model such as Inkling 2 Mini, at a few uneven tokens a second, showed each word smoothly and then paused until the next one arrived. The text now runs slightly behind the model, about one gap between tokens, at the pace the model has recently been writing, so it keeps moving between words. Text that has arrived is never more than about 1.5 seconds behind, and fast models look the same as before.

## Compatibility

Qwen-family models, including both Bonsai models, download the 11.7.1 inference engine the first time they load. Other engines are unchanged. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
