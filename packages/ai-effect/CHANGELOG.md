# ai-effect

## 7.1.1

### Patch Changes

- c7a7a16: `AiError.cause` is now the raw SDK error (the same value as `reason.cause`), so code that walks `.cause` chains finds `statusCode` and `responseBody` again.

## 7.1.0

This release has breaking changes to `AiError` and the `streamText` result, even though it's a minor version. Read **Breaking changes** before you upgrade.

### Breaking changes

- **`AiError` now has a typed `reason`.** The `message` and `cause` fields are replaced by:
  - `method`: `'generateText' | 'streamText'`
  - `reason`: one of `RateLimitError`, `AuthenticationError`, `InvalidRequestError`, `ProviderError`, `InvalidOutputError`, `AbortedError` or `UnknownError`. Each tag has an `ai-effect/` prefix (for example `ai-effect/RateLimitError`), so it doesn't collide with `effect/ai` reason tags. The raw SDK error is on `reason.cause`.

  `message` is still there, now derived from the reason. A new `isRetryable` getter is available. To handle a single reason, use `Effect.catchReason('ai-effect/AiError', 'ai-effect/RateLimitError', ...)`.

- **Aborts now fail with `AbortedError`.** Before, an abort from your `abortSignal` failed with `UnknownError`. That covers SDK abort errors, timeout errors and stream `abort` parts. If you matched `UnknownError` to detect aborts, match `AbortedError` instead. Fiber interruption works the same as before.

- **`streamText` returns Effect types instead of the raw SDK result.**
  - `stream`, `fullStream`, `textStream`, `partialOutputStream` and `elementStream` are `Stream<_, AiError>`.
  - `text`, `usage`, `output` and the other promise fields are `Effect<_, AiError>`.
  - `textStream`, `partialOutputStream` and `elementStream` now fail on error and abort parts. The SDK drops those parts silently.
  - Interrupting a fiber that consumes any of these streams or fields aborts the call.
  - The SDK result is still available as `raw`.

  `ToolLoopAgent.stream` returns the same Effect-typed result.

### New features

- **`ToolLoopAgent.make`** wraps AI SDK 7's `ToolLoopAgent`. The constructor and per-call callbacks accept Effects. `generate` and `stream` return Effects that require `Ai`. The SDK agent is available as `ToolLoopAgent.raw`.
- **UI message streams:**
  - `createUIMessageStream` wraps the SDK function of the same name. `execute` is an Effect, and `writer.merge` takes an Effect `Stream`.
  - `createAgentUIStream` builds a UI message stream from a `ToolLoopAgent` and client `UIMessage`s. If the messages are invalid, the outer Effect fails before streaming starts.
  - The `streamText` result has a new `toUIMessageStream()` method.
  - Response helpers aren't wrapped. Use `Stream.toReadableStreamEffect` to convert.
- **`embed`, `embedMany` and `rerank`** now have Effect wrappers.
- **Tracing:** every wrapper emits Effect spans with OpenTelemetry GenAI attributes, so calls show up in any Effect tracer. This covers `generateText`, `streamText`, `ToolLoopAgent`, `createAgentUIStream`, `createUIMessageStream`, `embed`, `embedMany` and `rerank`, plus an `execute_tool` span for each tool call. `@ai-sdk/otel` spans are nested under the wrapper's span.
- **`onStepError`** in `generateText` and `streamText` accepts an Effect.

### Fixes

- When a call is aborted or interrupted, its tool `execute` effects are now interrupted too. Before, they kept running until the `Ai` layer closed.
- Free-form records like `Schema.Record(Schema.String, Schema.Unknown)` now stay open in the generated tool and output JSON Schema. They used to get `additionalProperties: false`.

## 7.0.0

### Major Changes

- 8411f5a: Initial release: Effect wrappers for AI SDK 7's `generateText`, `streamText` and `tool`, backed by a single `Ai` service whose layer owns callback fibers and stream lifetimes.
