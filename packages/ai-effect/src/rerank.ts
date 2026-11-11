import * as AiSdk from 'ai';
import { Effect } from 'effect';
import type { Ai } from '#src/ai.ts';
import { AiError } from '#src/ai-error.ts';
import { combineAbortSignals } from '#src/internal/abort-signal.ts';
import { reasonFromSdkError } from '#src/internal/ai-error-from-sdk.ts';
import {
	type EffectifyCallbacks,
	makeRunPromise,
	wrapCallbacks,
} from '#src/internal/callbacks.ts';

/**
 * Both the deprecated `experimental_*` names and their v7 replacements are
 * listed so EffectifyCallbacks rewrites whichever one a caller uses.
 */
export const RERANK_CALLBACK_KEYS = [
	'onStart',
	'experimental_onStart',
	'onEnd',
	'experimental_onEnd',
] as const;
type RerankCallbackKeys = (typeof RERANK_CALLBACK_KEYS)[number];

/**
 * `ai` doesn't export `JSONObject` (it lives in `@ai-sdk/provider`, which isn't
 * a dependency here), so the constraint is read off `rerank`'s own signature.
 */
type RerankValue = Parameters<typeof AiSdk.rerank>[0]['documents'][number];

type OriginalRerankDef<
	VALUE extends RerankValue,
	RUNTIME_CONTEXT extends Record<string, unknown>,
> = Parameters<typeof AiSdk.rerank<VALUE, RUNTIME_CONTEXT>>[0];

type RerankResult<
	VALUE extends RerankValue,
	RUNTIME_CONTEXT extends Record<string, unknown>,
> = Awaited<ReturnType<typeof AiSdk.rerank<VALUE, RUNTIME_CONTEXT>>>;

type EffectRerankDef<
	VALUE extends RerankValue,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	R,
> = EffectifyCallbacks<
	OriginalRerankDef<VALUE, RUNTIME_CONTEXT>,
	RerankCallbackKeys,
	R
> & { abortSignal?: AbortSignal };

export function rerank<
	VALUE extends RerankValue,
	RUNTIME_CONTEXT extends Record<string, unknown> = Record<string, unknown>,
	R = never,
>(
	params: EffectRerankDef<VALUE, RUNTIME_CONTEXT, R>,
): Effect.Effect<RerankResult<VALUE, RUNTIME_CONTEXT>, AiError, Ai | R> {
	return Effect.gen(function* () {
		const runPromise = yield* makeRunPromise<R>();

		return yield* Effect.tryPromise({
			try: (signal) =>
				AiSdk.rerank({
					...params,
					...wrapCallbacks(runPromise, params, RERANK_CALLBACK_KEYS),
					abortSignal: combineAbortSignals(signal, params.abortSignal),
				} as OriginalRerankDef<VALUE, RUNTIME_CONTEXT>),
			catch: (cause) =>
				new AiError({
					method: 'rerank',
					reason: reasonFromSdkError(cause),
				}),
		});
	});
}
