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
export const EMBED_MANY_CALLBACK_KEYS = [
	'onStart',
	'experimental_onStart',
	'onEnd',
	'experimental_onEnd',
] as const;
type EmbedManyCallbackKeys = (typeof EMBED_MANY_CALLBACK_KEYS)[number];

type OriginalEmbedManyDef<RUNTIME_CONTEXT extends Record<string, unknown>> =
	Parameters<typeof AiSdk.embedMany<RUNTIME_CONTEXT>>[0];

type EmbedManyResult<RUNTIME_CONTEXT extends Record<string, unknown>> = Awaited<
	ReturnType<typeof AiSdk.embedMany<RUNTIME_CONTEXT>>
>;

type EffectEmbedManyDef<
	RUNTIME_CONTEXT extends Record<string, unknown>,
	R,
> = EffectifyCallbacks<
	OriginalEmbedManyDef<RUNTIME_CONTEXT>,
	EmbedManyCallbackKeys,
	R
> & { abortSignal?: AbortSignal };

export function embedMany<
	RUNTIME_CONTEXT extends Record<string, unknown> = Record<string, unknown>,
	R = never,
>(
	params: EffectEmbedManyDef<RUNTIME_CONTEXT, R>,
): Effect.Effect<EmbedManyResult<RUNTIME_CONTEXT>, AiError, Ai | R> {
	return Effect.gen(function* () {
		const runPromise = yield* makeRunPromise<R>();

		return yield* Effect.tryPromise({
			try: (signal) =>
				AiSdk.embedMany({
					...params,
					...wrapCallbacks(runPromise, params, EMBED_MANY_CALLBACK_KEYS),
					abortSignal: combineAbortSignals(signal, params.abortSignal),
				} as OriginalEmbedManyDef<RUNTIME_CONTEXT>),
			catch: (cause) =>
				new AiError({
					method: 'embedMany',
					reason: reasonFromSdkError(cause),
				}),
		});
	});
}
