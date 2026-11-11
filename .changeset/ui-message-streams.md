---
"ai-effect": minor
---

Add `createUIMessageStream`, an Effect wrapper for the SDK's function of the same name: `execute` is an Effect and `writer.merge` takes an Effect `Stream`. The Effect-typed `streamText` result gains `toUIMessageStream()`, and `createAgentUIStream` builds a UI message stream from a `ToolLoopAgent` and client `UIMessage`s, failing the outer Effect before streaming when the messages are invalid. The SDK agent is available as `ToolLoopAgent.raw`. Response helpers aren't wrapped; convert with `Stream.toReadableStreamEffect`.
