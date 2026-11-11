---
"ai-effect": minor
---

`AiError` now carries a `reason` union instead of a flat message. Its shape changed: `message` and `cause` are replaced by `method` (`'generateText' | 'streamText'`) and `reason`, one of `RateLimitError`, `AuthenticationError`, `InvalidRequestError`, `ProviderError`, `InvalidOutputError` or `UnknownError`. The raw SDK error is on `reason.cause`. `AiError` keeps a derived `message` and gains an `isRetryable` getter, and `Effect.catchReason('ai-effect/AiError', 'RateLimitError', ...)` handles a single reason.
