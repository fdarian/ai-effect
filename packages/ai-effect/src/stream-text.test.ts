import * as AiSdk from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import {
	Context,
	Deferred,
	Effect,
	Exit,
	Fiber,
	Layer,
	Schema,
	Scope,
	Stream,
} from 'effect';
import { beforeEach, describe, expect, expectTypeOf, test, vi } from 'vitest';
import { Ai } from '#src/ai.ts';
import { AbortedError, AiError, UnknownError } from '#src/ai-error.ts';
import { effectSchema } from '#src/schema.ts';
import { streamText } from '#src/stream-text.ts';

vi.mock('ai', async (importOriginal) => {
	const actual = await importOriginal<typeof AiSdk>();
	return {
		...actual,
		streamText: vi.fn(),
	};
});

const actual = await vi.importActual<typeof AiSdk>('ai');

const usage = {
	inputTokens: {
		total: 3,
		noCache: 3,
		cacheRead: undefined,
		cacheWrite: undefined,
	},
	outputTokens: { total: 10, text: 10, reasoning: undefined },
};

const textChunks = (...deltas: string[]) => [
	{ type: 'stream-start' as const, warnings: [] },
	{ type: 'text-start' as const, id: '1' },
	...deltas.map((delta) => ({
		type: 'text-delta' as const,
		id: '1',
		delta,
	})),
	{ type: 'text-end' as const, id: '1' },
	{
		type: 'finish' as const,
		usage,
		finishReason: { unified: 'stop' as const, raw: 'stop' },
	},
];

type ModelChunk = ReturnType<typeof textChunks>[number];

const modelFromStream = (makeStream: () => ReadableStream<unknown>) =>
	new MockLanguageModelV4({
		doStream: async () => ({ stream: makeStream() as never }),
	});

const modelFromChunks = (chunks: unknown[]) =>
	modelFromStream(
		() =>
			new ReadableStream({
				start(controller) {
					for (const chunk of chunks) controller.enqueue(chunk);
					controller.close();
				},
			}),
	);

/** Emits the chunks, then stays open until the SDK aborts it. */
const hangingModel = (chunks: ModelChunk[]) =>
	modelFromStream(
		() =>
			new ReadableStream({
				start(controller) {
					for (const chunk of chunks) controller.enqueue(chunk);
				},
			}),
	);

const sdkSignal = () => {
	const params = vi.mocked(AiSdk.streamText).mock.calls[0]?.[0];
	if (params?.abortSignal === undefined) throw new Error('no signal');
	return params.abortSignal;
};

const useRealSdk = () => {
	vi.mocked(AiSdk.streamText).mockImplementation(actual.streamText);
};

beforeEach(() => {
	vi.mocked(AiSdk.streamText).mockReset();
});

/** The slice of the SDK's end event that tracing reads. */
const completedEvent = {
	finishReason: 'stop',
	totalUsage: {
		inputTokens: 1,
		outputTokens: 2,
		inputTokenDetails: {
			cacheReadTokens: undefined,
			cacheWriteTokens: undefined,
		},
		outputTokenDetails: { reasoningTokens: undefined },
	},
	finalStep: { response: { id: 'resp-1', modelId: 'model-1' } },
	steps: [],
};

describe('streamText', () => {
	test('wraps synchronous return', () => {
		const mockResult = { textStream: 'mock-stream' };
		vi.mocked(AiSdk.streamText).mockReturnValue(mockResult as never);

		return Effect.gen(function* () {
			const result = yield* streamText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
			});

			expect(result.raw).toBe(mockResult);
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
			expect(result.method).toBe('streamText');
			expect(result.message).toBe('streamText: stream error');
			expect(result.reason).toBeInstanceOf(UnknownError);
			expect(result.reason.cause).toBeInstanceOf(Error);
			expect((result.reason.cause as Error).message).toBe('stream error');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('wraps onFinish callback', () => {
		const finishEvent = { ...completedEvent, text: 'done' };
		vi.mocked(AiSdk.streamText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			((params: any) => {
				if (params.onEnd) params.onEnd(finishEvent);
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
		const endEvent = { ...completedEvent, text: 'done' };
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
	test('aborts the SDK stream when the Ai layer scope closes', () => {
		let signal: AbortSignal | undefined;
		vi.mocked(AiSdk.streamText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			((params: any) => {
				signal = params.abortSignal;
				return { textStream: 'mock' };
			}) as never,
		);

		return Effect.gen(function* () {
			const layerScope = yield* Scope.make();
			const context = yield* Layer.buildWithScope(Ai.layer, layerScope);

			yield* streamText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
			}).pipe(Effect.provideContext(context));

			expect(signal?.aborted).toBe(false);
			yield* Scope.close(layerScope, Exit.void);
			expect(signal?.aborted).toBe(true);
		}).pipe(Effect.runPromise);
	});

	test('still honors a user abortSignal alongside the layer signal', () => {
		let signal: AbortSignal | undefined;
		vi.mocked(AiSdk.streamText).mockImplementation(
			// biome-ignore lint/suspicious/noExplicitAny: test mock
			((params: any) => {
				signal = params.abortSignal;
				return { textStream: 'mock' };
			}) as never,
		);
		const userController = new AbortController();

		return Effect.gen(function* () {
			yield* streamText({
				model: {} as AiSdk.LanguageModel,
				prompt: 'test',
				abortSignal: userController.signal,
			});

			expect(signal?.aborted).toBe(false);
			userController.abort();
			expect(signal?.aborted).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});
});

describe('streamText result', () => {
	beforeEach(useRealSdk);

	test('textStream collects the deltas and effects succeed', () =>
		Effect.gen(function* () {
			const result = yield* streamText({
				model: modelFromChunks(textChunks('Hel', 'lo')),
				prompt: 'hi',
			});

			expect(yield* Stream.runCollect(result.textStream)).toEqual([
				'Hel',
				'lo',
			]);
			expect(yield* result.text).toBe('Hello');
			expect(yield* result.finishReason).toBe('stop');
			expect((yield* result.usage).outputTokens).toBe(10);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('stream and fullStream emit the parts', () =>
		Effect.gen(function* () {
			const result = yield* streamText({
				model: modelFromChunks(textChunks('Hi')),
				prompt: 'hi',
			});

			const parts = yield* Stream.runCollect(result.stream);
			expect(parts.map((part) => part.type)).toContain('text-delta');
			expect(yield* Stream.runCollect(result.fullStream)).toEqual(parts);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('a thrown model stream error fails textStream and text with AiError', () =>
		Effect.gen(function* () {
			const boom = new Error('boom');
			const result = yield* streamText({
				model: modelFromStream(
					() =>
						new ReadableStream({
							start(controller) {
								controller.error(boom);
							},
						}),
				),
				prompt: 'hi',
			});

			const streamError = yield* Stream.runCollect(result.textStream).pipe(
				Effect.flip,
			);
			expect(streamError).toBeInstanceOf(AiError);
			expect(streamError.method).toBe('streamText');
			expect(streamError.reason._tag).toBe('ai-effect/UnknownError');
			expect(streamError.reason.cause).toBe(boom);

			const textError = yield* result.text.pipe(Effect.flip);
			expect(textError).toBeInstanceOf(AiError);
			expect(textError.method).toBe('streamText');
			expect(textError.reason._tag).toBe('ai-effect/UnknownError');
			expect(textError.reason.cause).toBe(boom);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('runs an Effect passed as onStepError when the model call throws', () => {
		useRealSdk();
		const boom = new Error('model failed');
		const errors: unknown[] = [];

		return Effect.gen(function* () {
			const result = yield* streamText({
				model: new MockLanguageModelV4({
					doStream: async () => {
						throw boom;
					},
				}),
				prompt: 'test',
				maxRetries: 0,
				onError: () => Effect.void,
				onStepError: (event) => Effect.sync(() => errors.push(event.error)),
			});
			yield* Effect.promise(() => Promise.resolve(result.raw.consumeStream()));

			expect(errors).toEqual([boom]);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('an error chunk fails textStream but stream emits it as a part', () =>
		Effect.gen(function* () {
			const chunkError = new Error('chunk error');
			const result = yield* streamText({
				model: modelFromChunks([
					...textChunks('Hi').slice(0, 3),
					{ type: 'error', error: chunkError },
				]),
				prompt: 'hi',
				onError: () => Effect.void,
			});

			const error = yield* Stream.runCollect(result.textStream).pipe(
				Effect.flip,
			);
			expect(error).toBeInstanceOf(AiError);
			expect(error.method).toBe('streamText');
			expect(error.reason._tag).toBe('ai-effect/UnknownError');
			expect(error.reason.cause).toBe(chunkError);

			const parts = yield* Stream.runCollect(result.stream);
			const errorPart = parts.find((part) => part.type === 'error');
			expect(errorPart?.error).toBe(chunkError);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('a user abortSignal aborting mid-stream fails textStream as aborted', () =>
		Effect.gen(function* () {
			const userController = new AbortController();
			const result = yield* streamText({
				model: hangingModel(textChunks('Hi').slice(0, 3)),
				prompt: 'hi',
				abortSignal: userController.signal,
			});

			const error = yield* Stream.runCollect(
				result.textStream.pipe(
					Stream.tap(() => Effect.sync(() => userController.abort())),
				),
			).pipe(Effect.flip);
			expect(error).toBeInstanceOf(AiError);
			expect(error.method).toBe('streamText');
			expect(error.reason).toBeInstanceOf(AbortedError);
			expect(error.message).toBe('streamText: aborted');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('a user abortSignal aborting mid-stream fails text as aborted', () =>
		Effect.gen(function* () {
			const userController = new AbortController();
			const result = yield* streamText({
				model: hangingModel(textChunks('Hi').slice(0, 3)),
				prompt: 'hi',
				abortSignal: userController.signal,
			});

			const fiber = yield* result.text.pipe(Effect.flip, Effect.forkChild);
			yield* Effect.sleep('10 millis');
			userController.abort();
			const error = yield* Fiber.join(fiber);

			expect(error).toBeInstanceOf(AiError);
			expect(error.reason).toBeInstanceOf(AbortedError);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('interrupting a textStream consumer aborts the call', () =>
		Effect.gen(function* () {
			const result = yield* streamText({
				model: hangingModel(textChunks('Hi').slice(0, 3)),
				prompt: 'hi',
			});
			const gotChunk = yield* Deferred.make<void>();

			const fiber = yield* Stream.runDrain(
				result.textStream.pipe(
					Stream.tap(() => Deferred.succeed(gotChunk, undefined)),
				),
			).pipe(Effect.forkChild);
			yield* Deferred.await(gotChunk);
			expect(sdkSignal().aborted).toBe(false);

			yield* Fiber.interrupt(fiber);
			expect(sdkSignal().aborted).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('interrupting a fiber awaiting text aborts the call', () =>
		Effect.gen(function* () {
			const result = yield* streamText({
				model: hangingModel(textChunks('Hi').slice(0, 3)),
				prompt: 'hi',
			});

			const fiber = yield* Effect.forkChild(result.text);
			yield* Effect.sleep(20);
			expect(sdkSignal().aborted).toBe(false);

			yield* Fiber.interrupt(fiber);
			expect(sdkSignal().aborted).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('stopping textStream early with Stream.take does not abort the call', () =>
		Effect.gen(function* () {
			const result = yield* streamText({
				model: hangingModel(textChunks('a', 'b', 'c').slice(0, 5)),
				prompt: 'hi',
			});

			const taken = yield* Stream.runCollect(
				result.textStream.pipe(Stream.take(1)),
			);
			expect(taken).toEqual(['a']);
			expect(sdkSignal().aborted).toBe(false);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('completing or failing does not abort the call', () =>
		Effect.gen(function* () {
			const result = yield* streamText({
				model: modelFromChunks(textChunks('Hi')),
				prompt: 'hi',
			});

			yield* Stream.runDrain(result.textStream);
			yield* result.text;
			expect(sdkSignal().aborted).toBe(false);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));
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
			Effect.Effect<{ readonly foo: string }, AiError>
		>();
		expectTypeOf<StreamResult['partialOutputStream']>().toEqualTypeOf<
			Stream.Stream<
				Effect.Success<StreamResult['output']> extends infer O
					? Partial<O>
					: never,
				AiError
			>
		>();
	});

	test('streamText exposes Effect-typed streams and fields', () => {
		const program = streamText({
			model: {} as AiSdk.LanguageModel,
			prompt: 'hi',
		});
		type StreamResult = Effect.Success<typeof program>;
		expectTypeOf<StreamResult['textStream']>().toEqualTypeOf<
			Stream.Stream<string, AiError>
		>();
		expectTypeOf<StreamResult['text']>().toEqualTypeOf<
			Effect.Effect<string, AiError>
		>();
	});
});

describe('streamText toUIMessageStream', () => {
	test('emits UI message chunks', () =>
		Effect.gen(function* () {
			useRealSdk();
			const result = yield* streamText({
				model: modelFromChunks(textChunks('Hi')),
				prompt: 'hi',
			});

			const chunks = yield* Stream.runCollect(result.toUIMessageStream());
			expect(chunks.map((chunk) => chunk.type)).toEqual([
				'start',
				'start-step',
				'text-start',
				'text-delta',
				'text-end',
				'finish-step',
				'finish',
			]);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('runs an Effect onEnd with the call-site services', () => {
		class CallDep extends Context.Service<CallDep, { value: string }>()(
			'CallDep',
		) {}
		return Effect.gen(function* () {
			useRealSdk();
			const seen = yield* Deferred.make<string>();
			const result = yield* streamText({
				model: modelFromChunks(textChunks('Hi')),
				prompt: 'hi',
			});

			yield* Stream.runDrain(
				result.toUIMessageStream({
					onEnd: () =>
						Effect.gen(function* () {
							const dep = yield* CallDep;
							yield* Deferred.succeed(seen, dep.value);
						}),
				}),
			);

			expect(yield* Deferred.await(seen)).toBe('from-call-site');
		}).pipe(
			Effect.provideService(CallDep, { value: 'from-call-site' }),
			Effect.provide(Ai.layer),
			Effect.runPromise,
		);
	});

	test('can be run twice', () =>
		Effect.gen(function* () {
			useRealSdk();
			const result = yield* streamText({
				model: modelFromChunks(textChunks('Hi')),
				prompt: 'hi',
			});
			const view = result.toUIMessageStream();

			const first = yield* Stream.runCollect(view);
			const second = yield* Stream.runCollect(view);
			expect(first.length).toBeGreaterThan(0);
			expect(second).toEqual(first);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('interrupting the consumer aborts the call', () =>
		Effect.gen(function* () {
			useRealSdk();
			const result = yield* streamText({
				model: hangingModel(textChunks('Hi').slice(0, 3)),
				prompt: 'hi',
			});
			const gotChunk = yield* Deferred.make<void>();

			const fiber = yield* Stream.runDrain(
				result
					.toUIMessageStream()
					.pipe(Stream.tap(() => Deferred.succeed(gotChunk, undefined))),
			).pipe(Effect.forkChild);
			yield* Deferred.await(gotChunk);
			expect(sdkSignal().aborted).toBe(false);

			yield* Fiber.interrupt(fiber);
			expect(sdkSignal().aborted).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));
});
