---
"ai-effect": minor
---

Every wrapper now emits Effect spans with OpenTelemetry GenAI attributes (`generateText`, `streamText`, `ToolLoopAgent`, `createAgentUIStream`, `createUIMessageStream`, `embed`, `embedMany`, `rerank`, plus an `execute_tool` span per tool), so calls appear in any Effect tracer. They also parent `@ai-sdk/otel` spans under the wrapper span.
