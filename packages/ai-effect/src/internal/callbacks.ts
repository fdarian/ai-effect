import { Cause, Effect, FiberSet } from 'effect';
import { Ai } from '#src/ai.ts';

/**
 * `runtimePromise` captures the call-site context (so callbacks see the
 * caller's services `R`) but forks into the `Ai` layer's FiberSet, so callbacks
 * that outlive the call are only interrupted when the layer's scope closes.
 */
export const makeRunPromise = <R>() =>
	Effect.gen(function* () {
		const ai = yield* Ai;
		return yield* FiberSet.runtimePromise(ai.fiberSet)<R>();
	});

// biome-ignore lint/suspicious/noExplicitAny: internal bridging helper, type safety enforced at public API boundary
export function wrapCallback(runPromise: any, callback: any) {
	if (callback == null) return undefined;
	// biome-ignore lint/suspicious/noExplicitAny: internal bridging helper
	return (...args: any[]) => runPromise(callback(...args));
}

export function wrapCallbacks(
	// biome-ignore lint/suspicious/noExplicitAny: internal bridging helper, type safety enforced at public API boundary
	runPromise: any,
	// biome-ignore lint/suspicious/noExplicitAny: internal bridging helper, type safety enforced at public API boundary
	params: any,
	keys: readonly string[],
) {
	const wrapped: Record<string, unknown> = {};
	for (const key of keys) {
		wrapped[key] = wrapCallback(runPromise, params[key]);
	}
	return wrapped;
}

/**
 * Fallback `onError` for the streaming wrappers. Setting any `onError` replaces
 * the SDK's default `console.error`, and tracing always sets one, so without
 * this a stream error would only reach the span. Pass the same `runPromise`
 * the user callbacks run with so the caller's `Logger` and span apply.
 */
export const logStreamError =
	(
		runPromise: (effect: Effect.Effect<void>) => Promise<unknown>,
		method: string,
	) =>
	(event: { readonly error: unknown }) =>
		runPromise(
			Effect.logError(`${method}: stream error`, Cause.fail(event.error)),
		);

export type EffectifyCallbacks<T, Keys extends string, R> = Omit<
	T,
	Keys & keyof T
> & {
	[K in Keys & keyof T]?: NonNullable<T[K]> extends (
		...args: infer A
	) => unknown
		? (...args: A) => Effect.Effect<void, never, R>
		: T[K];
};

/**
 * Both the deprecated `experimental_*` names and their v7 replacements are
 * listed so EffectifyCallbacks rewrites whichever one a caller uses.
 */
export const START_END_CALLBACK_KEYS = [
	'onStart',
	'experimental_onStart',
	'onEnd',
	'experimental_onEnd',
] as const;
export type StartEndCallbackKeys = (typeof START_END_CALLBACK_KEYS)[number];
