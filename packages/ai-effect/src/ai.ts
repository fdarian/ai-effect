import { Context, Effect, FiberSet, Layer } from 'effect';

/**
 * Owns the FiberSet that callback bridging forks into and the AbortSignal
 * handed to streams. Providing `Ai.layer` once at app level ties callback
 * fibers and SDK streams to the layer's scope instead of to each individual
 * call.
 */
export class Ai extends Context.Service<Ai>()('ai-effect/Ai', {
	make: Effect.gen(function* () {
		const fiberSet = yield* FiberSet.make();
		const controller = new AbortController();
		// Explicit annotation: without it declaration emit fails (TS4020) because
		// @types/node's AbortSignal can't be named from the inferred service type.
		const abortSignal: AbortSignal = controller.signal;
		yield* Effect.addFinalizer(() => Effect.sync(() => controller.abort()));
		return { fiberSet, abortSignal };
	}),
}) {
	static readonly layer = Layer.effect(Ai, Ai.make);
}
