---
"ai-effect": patch
---

Stream errors from `streamText` and `ToolLoopAgent.stream` without a user `onError` are now logged with `Effect.logError`, so the caller's `Logger` and span apply, instead of `console.error`. Pass `onError` to replace the log.
