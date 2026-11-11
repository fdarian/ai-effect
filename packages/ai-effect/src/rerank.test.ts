import type * as AiSdk from 'ai';
import { MockRerankingModelV4 } from 'ai/test';
import { Context, Effect, Fiber } from 'effect';
import { describe, expect, expectTypeOf, test, vi } from 'vitest';
import { Ai } from '#src/ai.ts';
import { AiError, UnknownError } from '#src/ai-error.ts';
import { rerank } from '#src/rerank.ts';

const makeModel = () =>
	new MockRerankingModelV4({
		doRerank: async () => ({
			ranking: [
				{ index: 1, relevanceScore: 0.9 },
				{ index: 0, relevanceScore: 0.1 },
			],
			warnings: [],
		}),
	});

describe('rerank', () => {
	test('returns the reranked documents', () =>
		Effect.gen(function* () {
			const result = yield* rerank({
				model: makeModel(),
				query: 'greeting',
				documents: ['bye', 'hello'],
			});

			expect(result.rerankedDocuments).toEqual(['hello', 'bye']);
			expect(result.ranking.map((item) => item.originalIndex)).toEqual([1, 0]);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('runs Effect callbacks with services from the call-site context', () => {
		class Dep extends Context.Service<Dep, { value: string }>()('Dep') {}
		const seen: string[] = [];

		return Effect.gen(function* () {
			yield* rerank({
				model: makeModel(),
				query: 'greeting',
				documents: ['bye', 'hello'],
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
			yield* rerank({
				model: makeModel(),
				query: 'greeting',
				documents: ['bye', 'hello'],
				experimental_onStart: onStart,
				experimental_onEnd: onEnd,
			});

			expect(onStart).toHaveBeenCalledTimes(1);
			expect(onEnd).toHaveBeenCalledTimes(1);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('surfaces errors as AiError', () =>
		Effect.gen(function* () {
			const model = new MockRerankingModelV4({
				doRerank: async () => {
					throw new Error('API error');
				},
			});

			const result = yield* rerank({
				model,
				query: 'greeting',
				documents: ['bye', 'hello'],
				maxRetries: 0,
			}).pipe(Effect.flip);

			expect(result).toBeInstanceOf(AiError);
			expect(result.method).toBe('rerank');
			expect(result.message).toBe('rerank: API error');
			expect(result.reason).toBeInstanceOf(UnknownError);
			expect(result.reason.cause).toBeInstanceOf(Error);
			expect((result.reason.cause as Error).message).toBe('API error');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));
});

describe('rerank interruption', () => {
	const makeHangingModel = (received: { signal?: AbortSignal }) =>
		new MockRerankingModelV4({
			doRerank: (options) => {
				received.signal = options.abortSignal;
				return new Promise(() => {});
			},
		});

	test('interrupting aborts the signal passed to the model', () => {
		const received: { signal?: AbortSignal } = {};

		return Effect.gen(function* () {
			const fiber = yield* rerank({
				model: makeHangingModel(received),
				query: 'greeting',
				documents: ['bye', 'hello'],
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
			const fiber = yield* rerank({
				model: makeHangingModel(received),
				query: 'greeting',
				documents: ['bye', 'hello'],
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
	test('infers VALUE from string documents', () => {
		const documents: string[] = ['bye', 'hello'];
		const program = rerank({
			model: {} as AiSdk.RerankingModel,
			query: 'greeting',
			documents,
		});
		type Result = Effect.Success<typeof program>;
		expectTypeOf<
			Result['ranking'][number]['document']
		>().toEqualTypeOf<string>();
	});

	test('infers VALUE from object documents', () => {
		const program = rerank({
			model: {} as AiSdk.RerankingModel,
			query: 'greeting',
			documents: [{ id: 'a' }, { id: 'b' }],
		});
		type Result = Effect.Success<typeof program>;
		expectTypeOf<Result['ranking'][number]['document']>().toEqualTypeOf<{
			id: string;
		}>();
	});

	test('infers RUNTIME_CONTEXT from `runtimeContext`', () => {
		const program = rerank({
			model: {} as AiSdk.RerankingModel,
			query: 'greeting',
			documents: ['bye', 'hello'],
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

		const program = rerank({
			model: {} as AiSdk.RerankingModel,
			query: 'greeting',
			documents: ['bye', 'hello'],
			onEnd: () =>
				Effect.gen(function* () {
					yield* Dep;
				}),
		});
		expectTypeOf<Effect.Services<typeof program>>().toEqualTypeOf<Ai | Dep>();
	});
});
