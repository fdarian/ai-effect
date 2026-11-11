import * as AiSdk from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import { Context, Duration, Effect, Fiber, Schema } from 'effect';
import { describe, expect, expectTypeOf, test, vi } from 'vitest';
import { Ai } from '#src/ai.ts';
import { AiError, UnknownError } from '#src/ai-error.ts';
import { generateText } from '#src/generate-text.ts';
import { effectSchema } from '#src/schema.ts';
import { streamText } from '#src/stream-text.ts';
import { tool } from '#src/tool.ts';

vi.mock('ai', async (importOriginal) => {
	const actual = await importOriginal<typeof AiSdk>();
	return {
		...actual,
		generateText: vi.fn(),
	};
});

const actual = await vi.importActual<typeof AiSdk>('ai');

describe('generateText', () => {
	test('wraps promise and returns result', () => {
		const mockResult = { text: 'hello', finishReason: 'stop' };
		vi.mocked(AiSdk.generateText).mockResolvedValue(mockResult as never);

		return Effect.gen(function* () {
			const result = yield* generateText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
			});

			expect(result).toBe(mockResult);
			expect(AiSdk.generateText).toHaveBeenCalledWith(
				expect.objectContaining({ prompt: 'test' }),
			);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('surfaces errors as AiError', () =>
		Effect.gen(function* () {
			vi.mocked(AiSdk.generateText).mockRejectedValue(new Error('API error'));

			const result = yield* generateText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
			}).pipe(Effect.flip);

			expect(result).toBeInstanceOf(AiError);
			expect(result.method).toBe('generateText');
			expect(result.message).toBe('generateText: API error');
			expect(result.reason).toBeInstanceOf(UnknownError);
			expect(result.reason.cause).toBeInstanceOf(Error);
			expect((result.reason.cause as Error).message).toBe('API error');
			expect(result.isRetryable).toBe(false);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('lets callers catch a reason with Effect.catchReason', () =>
		Effect.gen(function* () {
			vi.mocked(AiSdk.generateText).mockRejectedValue(
				new AiSdk.APICallError({
					message: 'slow down',
					url: 'https://example.com',
					requestBodyValues: {},
					statusCode: 429,
					responseHeaders: { 'retry-after': '5' },
				}),
			);

			const retryAfter = yield* generateText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
			}).pipe(
				Effect.catchReason(
					'ai-effect/AiError',
					'ai-effect/RateLimitError',
					(reason) =>
						Effect.succeed(
							Duration.toMillis(reason.retryAfter as Duration.Duration),
						),
				),
			);

			expect(retryAfter).toBe(5000);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('wraps onStepFinish callback', () => {
		const stepResult = { text: 'step1' };
		vi.mocked(AiSdk.generateText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			(async (params: any) => {
				if (params.onStepFinish) await params.onStepFinish(stepResult);
				return { text: 'done' };
			}) as never,
		);

		const onStepFinishSpy = vi.fn(() => Effect.void);

		return Effect.gen(function* () {
			yield* generateText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
				onStepFinish: onStepFinishSpy,
			});

			expect(onStepFinishSpy).toHaveBeenCalledWith(stepResult);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('runs an Effect passed as onStepError when the model call throws', () => {
		vi.mocked(AiSdk.generateText).mockImplementation(actual.generateText);
		const boom = new Error('model failed');
		const errors: unknown[] = [];

		return Effect.gen(function* () {
			const error = yield* generateText({
				model: new MockLanguageModelV4({
					doGenerate: async () => {
						throw boom;
					},
				}),
				prompt: 'test',
				maxRetries: 0,
				onStepError: (event) => Effect.sync(() => errors.push(event.error)),
			}).pipe(Effect.flip);

			expect(error).toBeInstanceOf(AiError);
			expect(errors).toEqual([boom]);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('wraps onFinish callback', () => {
		const finishEvent = { text: 'done', steps: [], totalUsage: {} };
		vi.mocked(AiSdk.generateText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			(async (params: any) => {
				if (params.onFinish) await params.onFinish(finishEvent);
				return { text: 'done' };
			}) as never,
		);

		const onFinishSpy = vi.fn(() => Effect.void);

		return Effect.gen(function* () {
			yield* generateText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
				onFinish: onFinishSpy,
			});

			expect(onFinishSpy).toHaveBeenCalledWith(finishEvent);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	// v7 renamed onFinish -> onEnd and onStepFinish -> onStepEnd; both names
	// must be wrapped so callers on either name get Effect ergonomics.
	test('wraps onEnd callback (v7 rename of onFinish)', () => {
		const endEvent = { text: 'done', steps: [], totalUsage: {} };
		vi.mocked(AiSdk.generateText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			(async (params: any) => {
				if (params.onEnd) await params.onEnd(endEvent);
				return { text: 'done' };
			}) as never,
		);

		const onEndSpy = vi.fn(() => Effect.void);

		return Effect.gen(function* () {
			yield* generateText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
				onEnd: onEndSpy,
			});

			expect(onEndSpy).toHaveBeenCalledWith(endEvent);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('wraps onStepEnd callback (v7 rename of onStepFinish)', () => {
		const stepResult = { text: 'step1' };
		vi.mocked(AiSdk.generateText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			(async (params: any) => {
				if (params.onStepEnd) await params.onStepEnd(stepResult);
				return { text: 'done' };
			}) as never,
		);

		const onStepEndSpy = vi.fn(() => Effect.void);

		return Effect.gen(function* () {
			yield* generateText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
				onStepEnd: onStepEndSpy,
			});

			expect(onStepEndSpy).toHaveBeenCalledWith(stepResult);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});
});

describe('generateText interruption', () => {
	test('interrupting aborts the signal passed to ai.generateText', () => {
		let received: AbortSignal | undefined;
		vi.mocked(AiSdk.generateText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			((params: any) => {
				received = params.abortSignal;
				return new Promise(() => {});
			}) as never,
		);

		return Effect.gen(function* () {
			const fiber = yield* generateText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
			}).pipe(Effect.forkChild);
			yield* Effect.yieldNow;

			expect(received).toBeDefined();
			expect(received?.aborted).toBe(false);

			yield* Fiber.interrupt(fiber);
			expect(received?.aborted).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('aborts when either the effect is interrupted or the user signal aborts', () => {
		let received: AbortSignal | undefined;
		vi.mocked(AiSdk.generateText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			((params: any) => {
				received = params.abortSignal;
				return new Promise(() => {});
			}) as never,
		);
		const userController = new AbortController();

		return Effect.gen(function* () {
			const fiber = yield* generateText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
				abortSignal: userController.signal,
			}).pipe(Effect.forkChild);
			yield* Effect.yieldNow;

			expect(received?.aborted).toBe(false);
			userController.abort();
			expect(received?.aborted).toBe(true);

			yield* Fiber.interrupt(fiber);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});
});

// These are compile-time-only checks: a tsc-only regression (like OUTPUT
// silently falling back to its default, or a swapped generic position) never
// throws at runtime, so it can't be caught by `expect()` assertions above —
// `check:type` is what would fail if any of these mismatched.
describe('type-level regressions', () => {
	const TestSchema = Schema.Struct({ foo: Schema.String });

	test('generateText/streamText/tool require Ai | R and never Scope', () => {
		class Dep extends Context.Service<Dep, { value: string }>()('Dep') {}

		const generate = generateText<
			AiSdk.ToolSet,
			Record<string, unknown>,
			AiSdk.Output.Output<string, string>,
			Dep
		>({
			model: {} as AiSdk.LanguageModel,
			prompt: 'hi',
			onFinish: () =>
				Effect.gen(function* () {
					yield* Dep;
				}),
		});
		const stream = streamText<
			AiSdk.ToolSet,
			Record<string, unknown>,
			AiSdk.Output.Output<string, string, never>,
			Dep
		>({
			model: {} as AiSdk.LanguageModel,
			prompt: 'hi',
			onChunk: () =>
				Effect.gen(function* () {
					yield* Dep;
				}),
		});
		const aTool = tool<{ x: number }, string, Dep>({
			inputSchema: {} as AiSdk.FlexibleSchema<{ x: number }>,
			execute: () =>
				Effect.gen(function* () {
					const dep = yield* Dep;
					return dep.value;
				}),
		});

		expectTypeOf<Effect.Services<typeof generate>>().toEqualTypeOf<Ai | Dep>();
		expectTypeOf<Effect.Services<typeof stream>>().toEqualTypeOf<Ai | Dep>();
		expectTypeOf<Effect.Services<typeof aTool>>().toEqualTypeOf<Ai | Dep>();

		const bare = generateText({
			model: {} as AiSdk.LanguageModel,
			prompt: 'hi',
		});
		expectTypeOf<Effect.Services<typeof bare>>().toEqualTypeOf<Ai>();
	});

	test('generateText infers OUTPUT from the `output` param instead of defaulting', () => {
		const program = generateText({
			model: {} as AiSdk.LanguageModel,
			prompt: 'hi',
			output: AiSdk.Output.object({ schema: effectSchema(TestSchema) }),
		});
		type Output = Effect.Success<typeof program>['output'];
		expectTypeOf<Output>().toEqualTypeOf<{ readonly foo: string }>();
	});
});
