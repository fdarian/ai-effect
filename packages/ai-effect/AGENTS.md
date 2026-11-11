Effect wrappers for AI SDK 7's `generateText`, `streamText` and `tool`, all backed by one `Ai` service. Published as `ai-effect`; the version tracks the AI SDK major.

## Stack

ESM TypeScript built with tsdown, Effect v4 and `ai` as peer dependencies (pinned in devDependencies), vitest for tests. Internal imports use `#src/*`; filenames are kebab-case.

## Dev

- `pnpm check:type`, `pnpm check:lint`, `pnpm test`, `pnpm build`

## Architecture

The design rule: every exported function requires `Ai`, never `Scope`. `Ai` owns a `FiberSet` and an `AbortSignal` created in `Ai.layer`; callbacks and tool `execute` functions are forked into the set with the call-site's context, so they outlive the call and are interrupted only when the layer closes. The layer's finalizer also aborts the signal, which `streamText` passes to the SDK so the stream stops too (`generateText` is already aborted by fiber interruption). Don't reintroduce a per-call `FiberSet` or `Effect.scoped`.

One file per wrapped SDK function, each with its `.test.ts` beside it:

- `ai.ts` — the `Ai` service and `Ai.layer`
- `ai-error.ts` — `AiError`, tag `ai-effect/AiError`
- `generate-text.ts`, `stream-text.ts`, `tool.ts` — the wrappers; the long type comments explain inference workarounds, keep them in sync with `ai`'s types
- `schema.ts` — `effectSchema`, Effect `Schema` to AI SDK schema
- `internal/callbacks.ts` — Effect-to-Promise callback bridging shared by the wrappers
- `index.ts` — the public API; export new wrappers here

When the AI SDK ships a new major, bump `ai` and the `peerDependencies` range together, and add a `major` changeset.
