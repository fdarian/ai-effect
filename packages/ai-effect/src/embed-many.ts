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
import {
	composeCallbacks,
	embedManyCallbacks,
	modelIdOf,
	START_END_CALLBACK_ALIASES,
	withCallSpan,
} from '#src/internal/tracing.ts';

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
	StartEndCallbackKeys,
	R
> & { abortSignal?: AbortSignal };

export function embedMany<
	RUNTIME_CONTEXT extends Record<string, unknown> = Record<string, unknown>,
	R = never,
>(
	params: EffectEmbedManyDef<RUNTIME_CONTEXT, R>,
): Effect.Effect<EmbedManyResult<RUNTIME_CONTEXT>, AiError, Ai | R> {
	return withCallSpan(
		`embeddings ${modelIdOf(params.model)}`,
		{
			kind: 'client',
			attributes: { 'gen_ai.operation.name': 'embeddings' },
		},
		(call) =>
			Effect.gen(function* () {
				const runPromise = yield* makeRunPromise<R>();
				const wrapped = wrapCallbacks(
					runPromise,
					params,
					START_END_CALLBACK_KEYS,
				);

				return yield* Effect.tryPromise({
					try: (signal) =>
						AiSdk.embedMany({
							...params,
							...wrapped,
							...composeCallbacks(
								wrapped,
								embedManyCallbacks(call),
								START_END_CALLBACK_ALIASES,
							),
							abortSignal: combineAbortSignals(signal, params.abortSignal),
						} as OriginalEmbedManyDef<RUNTIME_CONTEXT>),
					catch: (cause) =>
						new AiError({
							method: 'embedMany',
							reason: reasonFromSdkError(cause),
						}),
				});
			}),
	);
}
