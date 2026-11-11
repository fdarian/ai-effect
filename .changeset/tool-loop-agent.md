---
"ai-effect": minor
---

Add `toolLoopAgent`, an Effect wrapper for AI SDK 7's `ToolLoopAgent`: constructor and per-call callbacks accept Effects, and `generate`/`stream` return Effects that require `Ai`.
