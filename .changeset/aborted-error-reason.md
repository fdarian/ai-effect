---
"ai-effect": minor
---

Add an `AbortedError` reason to `AiError`. Aborts from a user `abortSignal` (SDK abort and timeout errors, and stream `abort` parts) now fail with `AbortedError` instead of `UnknownError`. Breaking for code that matched `UnknownError` to detect aborts. Fiber interruption is unchanged.
