import { Effect, FiberSet } from 'effect';
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
