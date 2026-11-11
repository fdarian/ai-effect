---
"ai-effect": minor
---

`streamText` now returns Effect-typed streams and fields instead of the raw SDK result: `stream`, `fullStream`, `textStream`, `partialOutputStream` and `elementStream` are `Stream<_, AiError>`, and `text`, `usage`, `output` and the other promise fields are `Effect<_, AiError>`. `textStream`, `partialOutputStream` and `elementStream` fail on error or abort parts the SDK would drop. Interrupting a fiber that consumes any of them aborts the call. The SDK result is available under `raw`. This is breaking for code that used the old fields directly. `ToolLoopAgent.stream` returns the same Effect-typed result.
