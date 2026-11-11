Effect-native wrappers for the Vercel AI SDK, published to npm as `ai-effect`. The package version tracks the AI SDK major (`ai-effect@7` wraps `ai@7`).

## Stack

pnpm workspaces, Bun for running `.ts` scripts, Turborepo for checks only, Biome (formatting + lint), Changesets for releases, Effect v4 (`effect/Schema`, never Zod, in library code).

## Workspace map

- `packages/ai-effect/` — the published library ([AGENTS.md](./packages/ai-effect/AGENTS.md))
- `apps/` — empty for now; a docs site will live here later. Every app and any non-published workspace must be `"private": true`, or `changeset publish` will publish it.
- `.changeset/` — pending release notes. Add one with `pnpm changeset` for any user-visible change.
- `.github/` — `ci.yml` (auto-format commit, then `turbo run check test --affected`) and `release.yml` (Changesets action, npm via OIDC trusted publishing, no token)

## Commands

- `pnpm format` — `biome check --write`; the only root check script
- `pnpm turbo run check:type check:lint test` — everything CI runs
- `pnpm --filter ai-effect build` — tsdown build
- `pnpm changeset status` — preview the version bump. Never run `changeset version` by hand; the Version Packages PR does it.

## Gotchas

- Biome's `markdown.parser.frontmatter` must stay on, or `biome check --write` mangles changeset headers.
- No `devEngines` in any `package.json`: npm refuses to run `changeset publish` when it names a non-node runtime.
- No LICENSE file is committed yet (`package.json` says Apache-2.0).
