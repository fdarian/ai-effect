import * as AiSdk from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import {
	Context,
	Deferred,
	Effect,
	Exit,
	Fiber,
	Layer,
	Scope,
	Stream,
} from 'effect';
import { describe, expect, expectTypeOf, test } from 'vitest';
import { Ai } from '#src/ai.ts';
import { AiError, InvalidOutputError } from '#src/ai-error.ts';
import { createAgentUIStream } from '#src/create-agent-ui-stream.ts';
import { ToolLoopAgent } from '#src/tool-loop-agent.ts';

const usage = {
	inputTokens: {
		total: 1,
		noCache: 1,
		cacheRead: undefined,
		cacheWrite: undefined,
	},
	outputTokens: { total: 1, text: 1, reasoning: undefined },
};

const textStream = (text: string) => ({
	stream: AiSdk.simulateReadableStream({
		chunks: [
			{ type: 'stream-start' as const, warnings: [] },
			{ type: 'text-start' as const, id: 't' },
			{ type: 'text-delta' as const, id: 't', delta: text },
			{ type: 'text-end' as const, id: 't' },
			{
				type: 'finish' as const,
				finishReason: { unified: 'stop' as const, raw: undefined },
				usage,
			},
		],
	}),
});

const textModel = (text: string) =>
	new MockLanguageModelV4({ doStream: textStream(text) });

/** Emits one chunk, then hangs until the call's abort signal fires. */
const hangingModel = (
	onStarted: () => void,
	onAbort: (reason: unknown) => void,
) =>
	new MockLanguageModelV4({
		doStream: async (options) => {
			options.abortSignal?.addEventListener('abort', () =>
				onAbort(options.abortSignal?.reason),
			);
			return {
				stream: new ReadableStream({
					start(controller) {
						controller.enqueue({ type: 'stream-start', warnings: [] });
						controller.enqueue({ type: 'text-start', id: 't' });
						onStarted();
					},
				}),
			};
		},
	});

const uiMessages = [
	{ id: 'm1', role: 'user', parts: [{ type: 'text', text: 'hi' }] },
];

class CallDep extends Context.Service<CallDep, { value: string }>()('CallDep') {
	static readonly layer = Layer.succeed(CallDep, { value: 'call-dep' });
}

describe('createAgentUIStream', () => {
	test('streams UI message chunks for valid uiMessages', () =>
		Effect.gen(function* () {
			const model = textModel('hello');
			const agent = yield* ToolLoopAgent.make({ model });

			const stream = yield* createAgentUIStream({ agent, uiMessages });
			const chunks = yield* Stream.runCollect(stream);

			expect(chunks.map((chunk) => chunk.type)).toContain('text-delta');
			expect(chunks.at(-1)?.type).toBe('finish');
			expect(JSON.stringify(model.doStreamCalls[0]?.prompt)).toContain('hi');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('invalid uiMessages fail the outer Effect with an AiError', () =>
		Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({ model: textModel('hello') });

			const error = yield* createAgentUIStream({
				agent,
				uiMessages: [{ not: 'a message' }],
			}).pipe(Effect.flip);

			expect(error).toBeInstanceOf(AiError);
			expect(error.method).toBe('createAgentUIStream');
			expect(error.reason).toBeInstanceOf(InvalidOutputError);
			expect(AiSdk.TypeValidationError.isInstance(error.reason.cause)).toBe(
				true,
			);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('model-start failures fail the outer Effect', () =>
		Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({
				model: textModel('hello'),
				prepareCall: () => {
					throw new Error('prepare failed');
				},
			});

			const error = yield* createAgentUIStream({ agent, uiMessages }).pipe(
				Effect.flip,
			);

			expect(error).toBeInstanceOf(AiError);
			expect(error.method).toBe('createAgentUIStream');
			expect((error.reason.cause as Error).message).toBe('prepare failed');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('runs Effect callbacks with the call-site services', () => {
		const seen: string[] = [];

		return Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({ model: textModel('hello') });

			const stream = yield* createAgentUIStream({
				agent,
				uiMessages,
				onUIMessageStepEnd: () =>
					Effect.gen(function* () {
						const dep = yield* CallDep;
						seen.push(`uiStep:${dep.value}`);
					}),
				onStepEnd: () =>
					Effect.gen(function* () {
						const dep = yield* CallDep;
						seen.push(`step:${dep.value}`);
					}),
				onEnd: () =>
					Effect.gen(function* () {
						const dep = yield* CallDep;
						seen.push(`end:${dep.value}`);
					}),
			}).pipe(Effect.provide(CallDep.layer));
			yield* Stream.runDrain(stream);

			expect(seen.sort()).toEqual([
				'end:call-dep',
				'step:call-dep',
				'uiStep:call-dep',
			]);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('still runs settings callbacks from ToolLoopAgent.make', () => {
		const seen: string[] = [];

		return Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({
				model: textModel('hello'),
				onStepEnd: () => Effect.sync(() => seen.push('settings')),
			});

			const stream = yield* createAgentUIStream({ agent, uiMessages });
			yield* Stream.runDrain(stream);

			expect(seen).toEqual(['settings']);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('interrupting the stream consumer aborts the model call', () =>
		Effect.gen(function* () {
			const gotChunk = yield* Deferred.make<void>();
			const aborted = yield* Deferred.make<void>();
			const agent = yield* ToolLoopAgent.make({
				model: hangingModel(
					() => {},
					() => Deferred.doneUnsafe(aborted, Exit.void),
				),
			});

			const stream = yield* createAgentUIStream({ agent, uiMessages });
			const fiber = yield* stream.pipe(
				Stream.tap(() => Deferred.succeed(gotChunk, undefined)),
				Stream.runDrain,
				Effect.forkChild,
			);
			yield* Deferred.await(gotChunk);

			yield* Fiber.interrupt(fiber);
			yield* Deferred.await(aborted);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('early stops like Stream.take do not abort the model call', () =>
		Effect.gen(function* () {
			let abortedCount = 0;
			const agent = yield* ToolLoopAgent.make({
				model: new MockLanguageModelV4({
					doStream: async (options) => {
						options.abortSignal?.addEventListener('abort', () => {
							abortedCount++;
						});
						return textStream('hello');
					},
				}),
			});

			const stream = yield* createAgentUIStream({ agent, uiMessages });
			yield* Stream.runCollect(Stream.take(stream, 1));

			expect(abortedCount).toBe(0);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('closing the Ai layer aborts the model call', () =>
		Effect.gen(function* () {
			const started = yield* Deferred.make<void>();
			const aborted = yield* Deferred.make<void>();
			const scope = yield* Scope.make();
			const context = yield* Layer.buildWithScope(Ai.layer, scope);

			yield* Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					model: hangingModel(
						() => Deferred.doneUnsafe(started, Exit.void),
						() => Deferred.doneUnsafe(aborted, Exit.void),
					),
				});
				const stream = yield* createAgentUIStream({ agent, uiMessages });
				yield* Stream.runDrain(stream);
			}).pipe(Effect.provideContext(context), Effect.forkChild);
			yield* Deferred.await(started);

			yield* Scope.close(scope, Exit.void);
			yield* Deferred.await(aborted);
		}).pipe(Effect.runPromise));

	test('the user abortSignal aborts the model call', () =>
		Effect.gen(function* () {
			const started = yield* Deferred.make<void>();
			const aborted = yield* Deferred.make<void>();
			const userController = new AbortController();
			const agent = yield* ToolLoopAgent.make({
				model: hangingModel(
					() => Deferred.doneUnsafe(started, Exit.void),
					() => Deferred.doneUnsafe(aborted, Exit.void),
				),
			});

			const stream = yield* createAgentUIStream({
				agent,
				uiMessages,
				abortSignal: userController.signal,
			});
			yield* Stream.runDrain(stream).pipe(Effect.forkChild);
			yield* Deferred.await(started);

			userController.abort();
			yield* Deferred.await(aborted);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('types', () =>
		Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({ model: textModel('hello') });
			const effect = createAgentUIStream({
				agent,
				uiMessages,
				onEnd: () => Effect.provide(Effect.void, CallDep.layer),
			});

			expectTypeOf(effect).toEqualTypeOf<
				Effect.Effect<
					Stream.Stream<
						AiSdk.InferUIMessageChunk<
							AiSdk.UIMessage<
								unknown,
								AiSdk.UIDataTypes,
								AiSdk.InferUITools<{}>
							>
						>,
						AiError
					>,
					AiError,
					Ai
				>
			>();
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));
});
