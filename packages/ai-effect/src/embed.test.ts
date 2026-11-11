import type * as AiSdk from 'ai';
import { MockEmbeddingModelV4 } from 'ai/test';
import { Context, Effect, Fiber } from 'effect';
import { describe, expect, expectTypeOf, test, vi } from 'vitest';
import { Ai } from '#src/ai.ts';
import { AiError, UnknownError } from '#src/ai-error.ts';
import { embed } from '#src/embed.ts';

const embedding = [0.1, 0.2, 0.3];

const makeModel = () =>
	new MockEmbeddingModelV4({
		doEmbed: {
			embeddings: [embedding],
			usage: { tokens: 3 },
			warnings: [],
		},
	});

describe('embed', () => {
	test('returns the embedding', () =>
		Effect.gen(function* () {
			const result = yield* embed({ model: makeModel(), value: 'hello' });

			expect(result.value).toBe('hello');
			expect(result.embedding).toEqual(embedding);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('runs Effect callbacks with services from the call-site context', () => {
		class Dep extends Context.Service<Dep, { value: string }>()('Dep') {}
		const seen: string[] = [];

		return Effect.gen(function* () {
			yield* embed({
				model: makeModel(),
				value: 'hello',
				onStart: () =>
					Effect.gen(function* () {
						const dep = yield* Dep;
						seen.push(`start:${dep.value}`);
					}),
				onEnd: () =>
					Effect.gen(function* () {
						const dep = yield* Dep;
						seen.push(`end:${dep.value}`);
					}),
			});

			expect(seen).toEqual(['start:dep', 'end:dep']);
		}).pipe(
			Effect.provideService(Dep, { value: 'dep' }),
			Effect.provide(Ai.layer),
			Effect.runPromise,
		);
	});

	test('wraps the deprecated experimental_ callbacks', () => {
		const onStart = vi.fn(() => Effect.void);
		const onEnd = vi.fn(() => Effect.void);

		return Effect.gen(function* () {
			yield* embed({
				model: makeModel(),
				value: 'hello',
				experimental_onStart: onStart,
				experimental_onEnd: onEnd,
			});

			expect(onStart).toHaveBeenCalledTimes(1);
			expect(onEnd).toHaveBeenCalledTimes(1);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('surfaces errors as AiError', () =>
		Effect.gen(function* () {
			const model = new MockEmbeddingModelV4({
				doEmbed: async () => {
					throw new Error('API error');
				},
			});

			const result = yield* embed({
				model,
				value: 'hello',
				maxRetries: 0,
			}).pipe(Effect.flip);

			expect(result).toBeInstanceOf(AiError);
			expect(result.method).toBe('embed');
			expect(result.message).toBe('embed: API error');
			expect(result.reason).toBeInstanceOf(UnknownError);
			expect(result.reason.cause).toBeInstanceOf(Error);
			expect((result.reason.cause as Error).message).toBe('API error');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));
});

describe('embed interruption', () => {
	const makeHangingModel = (received: { signal?: AbortSignal }) =>
		new MockEmbeddingModelV4({
			doEmbed: (options) => {
				received.signal = options.abortSignal;
				return new Promise(() => {});
			},
		});

	test('interrupting aborts the signal passed to the model', () => {
		const received: { signal?: AbortSignal } = {};

		return Effect.gen(function* () {
			const fiber = yield* embed({
				model: makeHangingModel(received),
				value: 'hello',
			}).pipe(Effect.forkChild);
			yield* Effect.sleep('10 millis');

			expect(received.signal?.aborted).toBe(false);

			yield* Fiber.interrupt(fiber);
			expect(received.signal?.aborted).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('aborts when the user signal aborts', () => {
		const received: { signal?: AbortSignal } = {};
		const userController = new AbortController();

		return Effect.gen(function* () {
			const fiber = yield* embed({
				model: makeHangingModel(received),
				value: 'hello',
				abortSignal: userController.signal,
			}).pipe(Effect.forkChild);
			yield* Effect.sleep('10 millis');

			expect(received.signal?.aborted).toBe(false);
			userController.abort();
			expect(received.signal?.aborted).toBe(true);

			yield* Fiber.interrupt(fiber);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});
});

describe('type-level regressions', () => {
	test('infers RUNTIME_CONTEXT from `runtimeContext`', () => {
		const program = embed({
			model: {} as AiSdk.EmbeddingModel,
			value: 'hello',
			runtimeContext: { userId: 'x' },
			onStart: (event) => {
				expectTypeOf(event.runtimeContext).toEqualTypeOf<{
					userId: string;
				}>();
				return Effect.void;
			},
		});
		expectTypeOf<Effect.Services<typeof program>>().toEqualTypeOf<Ai>();
	});

	test('infers R from a callback requiring a service', () => {
		class Dep extends Context.Service<Dep, { value: string }>()('Dep') {}

		const program = embed({
			model: {} as AiSdk.EmbeddingModel,
			value: 'hello',
			onEnd: () =>
				Effect.gen(function* () {
					yield* Dep;
				}),
		});
		expectTypeOf<Effect.Services<typeof program>>().toEqualTypeOf<Ai | Dep>();
	});
});
