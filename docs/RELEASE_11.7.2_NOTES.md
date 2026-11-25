# TUFF 11.7.2

## Tool calls through the local API

- When a model wrote a tool call TUFF could not read, the local API answered with a bare HTTP 500 ("generation failed"). It now regenerates once, with a short note telling the model to use the declared tool format or answer without tools, the same way the app already does. If the second attempt is also unreadable, the API returns HTTP 502 with the code `malformed_tool_call` and a message saying what happened. A streamed reply that has already sent text is not regenerated, so nothing is repeated; it ends with the same error. In a 16-question web-search test with Gemma 4 E2B, 4 answers used to fail this way; now 3 of them recover and the fourth reports the clear error.

## Web search with DuckDuckGo

- DuckDuckGo sometimes answers TUFF's searches with a verification step or a refusal, and the search then failed. TUFF now sends the same query once to DuckDuckGo's lite page, which usually still answers, and reports the block only if that fails too. The query never goes to any other service. With the web sandbox on, the sandbox reaches only DuckDuckGo's main page, so a block is still reported straight away.

## Compatibility

No engine changes; inference engines stay at their current versions. Existing models, chats and settings are preserved. The app is ad-hoc signed and not notarized.
