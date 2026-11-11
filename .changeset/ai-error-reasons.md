---
"ai-effect": minor
---

`AiError` now carries a `reason` union instead of a flat message. Its shape changed: `message` and `cause` are replaced by `method` (`'generateText' | 'streamText'`) and `reason`, one of `RateLimitError`, `AuthenticationError`, `InvalidRequestError`, `ProviderError`, `InvalidOutputError` or `UnknownError`. Each reason's `_tag` is `ai-effect/`-prefixed (for example `ai-effect/RateLimitError`) so it doesn't collide with `effect/ai` reason tags. The raw SDK error is on `reason.cause`. `AiError` keeps a derived `message` and gains an `isRetryable` getter, and `Effect.catchReason('ai-effect/AiError', 'ai-effect/RateLimitError', ...)` handles a single reason.
