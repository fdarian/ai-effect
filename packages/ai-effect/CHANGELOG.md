# ai-effect

## 7.1.0

### Minor Changes

- 9bd212d: Add an `AbortedError` reason to `AiError`. Aborts from a user `abortSignal` (SDK abort and timeout errors, and stream `abort` parts) now fail with `AbortedError` instead of `UnknownError`. Breaking for code that matched `UnknownError` to detect aborts. Fiber interruption is unchanged.
- 6450113: Add Effect wrappers for `embed`, `embedMany` and `rerank`.
- ed35866: `AiError` now carries a `reason` union instead of a flat message. Its shape changed: `message` and `cause` are replaced by `method` (`'generateText' | 'streamText'`) and `reason`, one of `RateLimitError`, `AuthenticationError`, `InvalidRequestError`, `ProviderError`, `InvalidOutputError` or `UnknownError`. Each reason's `_tag` is `ai-effect/`-prefixed (for example `ai-effect/RateLimitError`) so it doesn't collide with `effect/ai` reason tags. The raw SDK error is on `reason.cause`. `AiError` keeps a derived `message` and gains an `isRetryable` getter, and `Effect.catchReason('ai-effect/AiError', 'ai-effect/RateLimitError', ...)` handles a single reason.
- 3395298: `streamText` now returns Effect-typed streams and fields instead of the raw SDK result: `stream`, `fullStream`, `textStream`, `partialOutputStream` and `elementStream` are `Stream<_, AiError>`, and `text`, `usage`, `output` and the other promise fields are `Effect<_, AiError>`. `textStream`, `partialOutputStream` and `elementStream` fail on error or abort parts the SDK would drop. Interrupting a fiber that consumes any of them aborts the call. The SDK result is available under `raw`. This is breaking for code that used the old fields directly. `ToolLoopAgent.stream` returns the same Effect-typed result.
- 390e416: Every wrapper now emits Effect spans with OpenTelemetry GenAI attributes (`generateText`, `streamText`, `ToolLoopAgent`, `createAgentUIStream`, `createUIMessageStream`, `embed`, `embedMany`, `rerank`, plus an `execute_tool` span per tool), so calls appear in any Effect tracer. They also parent `@ai-sdk/otel` spans under the wrapper span.
- e5a3f61: Add `ToolLoopAgent.make`, an Effect wrapper for AI SDK 7's `ToolLoopAgent`: constructor and per-call callbacks accept Effects, and `generate`/`stream` return Effects that require `Ai`.
- 742c3b4: Add `createUIMessageStream`, an Effect wrapper for the SDK's function of the same name: `execute` is an Effect and `writer.merge` takes an Effect `Stream`. The Effect-typed `streamText` result gains `toUIMessageStream()`, and `createAgentUIStream` builds a UI message stream from a `ToolLoopAgent` and client `UIMessage`s, failing the outer Effect before streaming when the messages are invalid. The SDK agent is available as `ToolLoopAgent.raw`. Response helpers aren't wrapped; convert with `Stream.toReadableStreamEffect`.

### Patch Changes

- 8d19f85: Interrupt tool `execute` effects when the call is aborted or interrupted, instead of leaving them running until the `Ai` layer closes.
- cb9d855: Free-form records like `Schema.Record(Schema.String, Schema.Unknown)` stay open in the tool and output JSON Schema instead of being closed with `additionalProperties: false`.
- 5ffeacd: `generateText` and `streamText` accept an Effect for `onStepError`.

## 7.0.0

### Major Changes

- 8411f5a: Initial release: Effect wrappers for AI SDK 7's `generateText`, `streamText` and `tool`, backed by a single `Ai` service whose layer owns callback fibers and stream lifetimes.
