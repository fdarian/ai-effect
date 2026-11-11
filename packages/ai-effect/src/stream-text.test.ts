import * as AiSdk from 'ai';
import { Deferred, Effect, Exit, Layer, Schema, Scope } from 'effect';
import { describe, expect, expectTypeOf, test, vi } from 'vitest';
import { Ai } from '#src/ai.ts';
import { AiError } from '#src/ai-error.ts';
import { effectSchema } from '#src/schema.ts';
import { streamText } from '#src/stream-text.ts';

vi.mock('ai', async (importOriginal) => {
	const actual = await importOriginal<typeof AiSdk>();
	return {
		...actual,
		streamText: vi.fn(),
	};
});

describe('streamText', () => {
	test('wraps synchronous return', () => {
		const mockResult = { textStream: 'mock-stream' };
		vi.mocked(AiSdk.streamText).mockReturnValue(mockResult as never);

		return Effect.gen(function* () {
			const result = yield* streamText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
			});

			expect(result).toBe(mockResult);
			expect(AiSdk.streamText).toHaveBeenCalledWith(
				expect.objectContaining({ prompt: 'test' }),
			);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('surfaces thrown errors as AiError', () => {
		vi.mocked(AiSdk.streamText).mockImplementation(() => {
			throw new Error('stream error');
		});

		return Effect.gen(function* () {
			const result = yield* streamText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
			}).pipe(Effect.flip);

			expect(result).toBeInstanceOf(AiError);
			expect(result.message).toBe('streamText failed');
			expect(result.cause).toBeInstanceOf(Error);
			expect((result.cause as Error).message).toBe('stream error');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('wraps onFinish callback', () => {
		const finishEvent = { text: 'done', steps: [], totalUsage: {} };
		vi.mocked(AiSdk.streamText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			((params: any) => {
				if (params.onFinish) params.onFinish(finishEvent);
				return { textStream: 'mock' };
			}) as never,
		);

		const onFinishSpy = vi.fn(() => Effect.void);

		return Effect.gen(function* () {
			yield* streamText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
				onFinish: onFinishSpy,
			});

			expect(onFinishSpy).toHaveBeenCalledWith(finishEvent);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	// v7 renamed onFinish -> onEnd; both names must be wrapped so callers on
	// either name get Effect ergonomics (same event, one calling convention).
	test('wraps onEnd callback (v7 rename of onFinish)', () => {
		const endEvent = { text: 'done', steps: [], totalUsage: {} };
		vi.mocked(AiSdk.streamText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			((params: any) => {
				if (params.onEnd) params.onEnd(endEvent);
				return { textStream: 'mock' };
			}) as never,
		);

		const onEndSpy = vi.fn(() => Effect.void);

		return Effect.gen(function* () {
			yield* streamText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
				onEnd: onEndSpy,
			});

			expect(onEndSpy).toHaveBeenCalledWith(endEvent);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});
	test('callbacks outlive the call and are interrupted when the Ai layer scope closes', () => {
		let onChunk: ((event: unknown) => Promise<void>) | undefined;
		vi.mocked(AiSdk.streamText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			((params: any) => {
				onChunk = params.onChunk;
				return { textStream: 'mock' };
			}) as never,
		);

		return Effect.gen(function* () {
			const layerScope = yield* Scope.make();
			const context = yield* Layer.buildWithScope(Ai.layer, layerScope);
			const started = yield* Deferred.make<void>();
			const interrupted = yield* Deferred.make<void>();

			yield* streamText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
				onChunk: () =>
					Deferred.succeed(started, undefined).pipe(
						Effect.andThen(Effect.never),
						Effect.onInterrupt(() => Deferred.succeed(interrupted, undefined)),
					),
			}).pipe(Effect.provideContext(context));

			// The streamText effect has returned; the callback fires afterwards,
			// the way the SDK invokes it while the stream is consumed.
			const callbackPromise = onChunk?.({});
			callbackPromise?.catch(() => {});
			yield* Deferred.await(started);
			expect(yield* Deferred.isDone(interrupted)).toBe(false);

			yield* Scope.close(layerScope, Exit.void);
			yield* Deferred.await(interrupted);
		}).pipe(Effect.runPromise);
	});
});

// These are compile-time-only checks: a tsc-only regression (like OUTPUT
// silently falling back to its default, or a swapped generic position) never
// throws at runtime, so it can't be caught by `expect()` assertions above —
// `check:type` is what would fail if any of these mismatched.
describe('type-level regressions', () => {
	const TestSchema = Schema.Struct({ foo: Schema.String });

	test('streamText infers OUTPUT from the `output` param instead of defaulting', () => {
		const program = streamText({
			model: {} as AiSdk.LanguageModel,
			prompt: 'hi',
			output: AiSdk.Output.object({ schema: effectSchema(TestSchema) }),
		});
		type StreamResult = Effect.Success<typeof program>;
		expectTypeOf<StreamResult['output']>().toEqualTypeOf<
			PromiseLike<{ readonly foo: string }>
		>();
	});
});
