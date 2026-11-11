import * as AiSdk from 'ai';
import { Effect } from 'effect';
import type { Ai } from '#src/ai.ts';
import { AiError } from '#src/ai-error.ts';
import { combineAbortSignals } from '#src/internal/abort-signal.ts';
import { reasonFromSdkError } from '#src/internal/ai-error-from-sdk.ts';
import {
	type EffectifyCallbacks,
	makeRunPromise,
	START_END_CALLBACK_KEYS,
	type StartEndCallbackKeys,
	wrapCallbacks,
} from '#src/internal/callbacks.ts';

type OriginalEmbedDef<RUNTIME_CONTEXT extends Record<string, unknown>> =
	Parameters<typeof AiSdk.embed<RUNTIME_CONTEXT>>[0];

type EmbedResult<RUNTIME_CONTEXT extends Record<string, unknown>> = Awaited<
	ReturnType<typeof AiSdk.embed<RUNTIME_CONTEXT>>
>;

type EffectEmbedDef<
	RUNTIME_CONTEXT extends Record<string, unknown>,
	R,
> = EffectifyCallbacks<
	OriginalEmbedDef<RUNTIME_CONTEXT>,
	StartEndCallbackKeys,
	R
> & { abortSignal?: AbortSignal };

export function embed<
	RUNTIME_CONTEXT extends Record<string, unknown> = Record<string, unknown>,
	R = never,
>(
	params: EffectEmbedDef<RUNTIME_CONTEXT, R>,
): Effect.Effect<EmbedResult<RUNTIME_CONTEXT>, AiError, Ai | R> {
	return Effect.gen(function* () {
		const runPromise = yield* makeRunPromise<R>();

		return yield* Effect.tryPromise({
			try: (signal) =>
				AiSdk.embed({
					...params,
					...wrapCallbacks(runPromise, params, START_END_CALLBACK_KEYS),
					abortSignal: combineAbortSignals(signal, params.abortSignal),
				} as OriginalEmbedDef<RUNTIME_CONTEXT>),
			catch: (cause) =>
				new AiError({
					method: 'embed',
					reason: reasonFromSdkError(cause),
				}),
		});
	});
}
