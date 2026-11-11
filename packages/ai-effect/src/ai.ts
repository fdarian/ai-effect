import { Context, Effect, FiberSet, Layer } from 'effect';

/**
 * Owns the FiberSet that callback bridging forks into. Providing `Ai.layer`
 * once at app level ties callback fibers to the layer's scope instead of to
 * each individual call.
 */
export class Ai extends Context.Service<Ai>()('ai-effect/Ai', {
	make: Effect.gen(function* () {
		const fiberSet = yield* FiberSet.make();
		return { fiberSet };
	}),
}) {
	static readonly layer = Layer.effect(Ai, Ai.make);
}
