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
	CurrentCallSpan,
	composeCallbacks,
	modelIdOf,
	rerankCallbacks,
	START_END_CALLBACK_ALIASES,
	withCallSpan,
} from '#src/internal/tracing.ts';

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
	StartEndCallbackKeys,
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
		const call = yield* CurrentCallSpan;
		const runPromise = yield* makeRunPromise<R>();
		const wrapped = wrapCallbacks(runPromise, params, START_END_CALLBACK_KEYS);

		return yield* Effect.tryPromise({
			try: (signal) =>
				AiSdk.rerank({
					...params,
					...wrapped,
					...composeCallbacks(
						wrapped,
						rerankCallbacks(call),
						START_END_CALLBACK_ALIASES,
					),
					abortSignal: combineAbortSignals(signal, params.abortSignal),
				} as OriginalRerankDef<VALUE, RUNTIME_CONTEXT>),
			catch: (cause) =>
				new AiError({
					method: 'rerank',
					reason: reasonFromSdkError(cause),
				}),
		});
	}).pipe(
		withCallSpan(`rerank ${modelIdOf(params.model)}`, {
			kind: 'client',
			attributes: { 'gen_ai.operation.name': 'rerank' },
		}),
	);
}
