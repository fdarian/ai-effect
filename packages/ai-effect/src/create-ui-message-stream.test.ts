import type * as AiSdk from 'ai';
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
import { AiError } from '#src/ai-error.ts';
import { createUIMessageStream } from '#src/create-ui-message-stream.ts';
import { streamText } from '#src/stream-text.ts';

const usage = {
	inputTokens: {
		total: 3,
		noCache: 3,
		cacheRead: undefined,
		cacheWrite: undefined,
	},
	outputTokens: { total: 10, text: 10, reasoning: undefined },
};

const textModel = (text: string) =>
	new MockLanguageModelV4({
		doStream: async () => ({
			stream: new ReadableStream({
				start(controller) {
					for (const chunk of [
						{ type: 'stream-start', warnings: [] },
						{ type: 'text-start', id: '1' },
						{ type: 'text-delta', id: '1', delta: text },
						{ type: 'text-end', id: '1' },
						{
							type: 'finish',
							usage,
							finishReason: { unified: 'stop', raw: 'stop' },
						},
					]) {
						controller.enqueue(chunk);
					}
					controller.close();
				},
			}) as never,
		}),
	});

const textChunk = (id: string, text: string) =>
	[
		{ type: 'text-start', id },
		{ type: 'text-delta', id, delta: text },
		{ type: 'text-end', id },
	] as const;

describe('createUIMessageStream', () => {
	test('emits written and merged chunks in order', () =>
		Effect.gen(function* () {
			const result = yield* streamText({
				model: textModel('from model'),
				prompt: 'hi',
			});
			const chunks = yield* Stream.runCollect(
				createUIMessageStream({
					execute: ({ writer }) =>
						Effect.gen(function* () {
							for (const chunk of textChunk('a', 'written')) {
								yield* writer.write(chunk);
							}
							yield* writer.merge(
								Stream.fromIterable(textChunk('b', 'merged')),
							);
							yield* writer.merge(result.toUIMessageStream());
						}),
				}),
			);

			const deltas = chunks.flatMap((chunk) =>
				chunk.type === 'text-delta' ? [chunk.delta] : [],
			);
			expect(deltas).toEqual(['written', 'merged', 'from model']);
			expect(chunks.map((chunk) => chunk.type)).toContain('finish');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('execute sees the call-site services', () => {
		class CallDep extends Context.Service<CallDep, { value: string }>()(
			'CallDep',
		) {}
		return Effect.gen(function* () {
			const chunks = yield* Stream.runCollect(
				createUIMessageStream<never, CallDep>({
					execute: ({ writer }) =>
						Effect.gen(function* () {
							const dep = yield* CallDep;
							yield* writer.write({ type: 'text-start', id: dep.value });
						}),
				}),
			);

			expect(chunks).toContainEqual({ type: 'text-start', id: 'from-call' });
		}).pipe(
			Effect.provideService(CallDep, { value: 'from-call' }),
			Effect.provide(Ai.layer),
			Effect.runPromise,
		);
	});

	test('a failing execute becomes an error chunk and the stream succeeds', () =>
		Effect.gen(function* () {
			const chunks = yield* Stream.runCollect(
				createUIMessageStream({
					execute: () => Effect.fail('boom'),
					onError: () => 'handled',
				}),
			);

			expect(chunks).toContainEqual({ type: 'error', errorText: 'handled' });
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('runs an Effect onEnd with the call-site services', () => {
		class CallDep extends Context.Service<CallDep, { value: string }>()(
			'CallDep',
		) {}
		return Effect.gen(function* () {
			const seen = yield* Deferred.make<{
				value: string;
				isAborted: boolean;
				messages: number;
			}>();

			yield* Stream.runDrain(
				createUIMessageStream<never, CallDep>({
					execute: ({ writer }) =>
						writer.write({ type: 'text-start', id: '1' }),
					onEnd: (event) =>
						Effect.gen(function* () {
							const dep = yield* CallDep;
							yield* Deferred.succeed(seen, {
								value: dep.value,
								isAborted: event.isAborted,
								messages: event.messages.length,
							});
						}),
				}),
			);

			expect(yield* Deferred.await(seen)).toEqual({
				value: 'from-call-site',
				isAborted: false,
				messages: 1,
			});
		}).pipe(
			Effect.provideService(CallDep, { value: 'from-call-site' }),
			Effect.provide(Ai.layer),
			Effect.runPromise,
		);
	});

	test('maps a throwing onEnd to AiError', () =>
		Effect.gen(function* () {
			const error = yield* Stream.runDrain(
				createUIMessageStream({
					execute: ({ writer }) =>
						writer.write({ type: 'text-start', id: '1' }),
					onEnd: () => Effect.die(new Error('end failed')),
				}),
			).pipe(Effect.flip);

			expect(error).toBeInstanceOf(AiError);
			expect(error.method).toBe('createUIMessageStream');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('can be run twice, running execute each time', () =>
		Effect.gen(function* () {
			let runs = 0;
			const stream = createUIMessageStream({
				execute: ({ writer }) =>
					Effect.suspend(() => {
						runs += 1;
						return writer.write({ type: 'text-start', id: '1' });
					}),
			});

			const first = yield* Stream.runCollect(stream);
			const second = yield* Stream.runCollect(stream);
			expect(runs).toBe(2);
			expect(second.map((chunk) => chunk.type)).toEqual(
				first.map((chunk) => chunk.type),
			);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('interrupting the consumer interrupts execute', () =>
		Effect.gen(function* () {
			const gotChunk = yield* Deferred.make<void>();
			const interrupted = yield* Deferred.make<void>();

			const fiber = yield* Stream.runDrain(
				createUIMessageStream({
					execute: ({ writer }) =>
						writer.write({ type: 'text-start', id: '1' }).pipe(
							Effect.andThen(Effect.never),
							Effect.onInterrupt(() =>
								Deferred.succeed(interrupted, undefined),
							),
						),
				}).pipe(Stream.tap(() => Deferred.succeed(gotChunk, undefined))),
			).pipe(Effect.forkChild);
			yield* Deferred.await(gotChunk);

			yield* Fiber.interrupt(fiber);
			yield* Deferred.await(interrupted);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('interrupting the consumer interrupts a merged stream', () =>
		Effect.gen(function* () {
			const gotChunk = yield* Deferred.make<void>();
			const interrupted = yield* Deferred.make<void>();

			const fiber = yield* Stream.runDrain(
				createUIMessageStream({
					execute: ({ writer }) =>
						writer.merge(
							Stream.make({ type: 'text-start', id: '1' } as const).pipe(
								Stream.concat(Stream.never),
								Stream.onExit((exit) =>
									exit._tag === 'Failure'
										? Deferred.succeed(interrupted, undefined)
										: Effect.void,
								),
							),
						),
				}).pipe(Stream.tap(() => Deferred.succeed(gotChunk, undefined))),
			).pipe(Effect.forkChild);
			yield* Deferred.await(gotChunk);

			yield* Fiber.interrupt(fiber);
			yield* Deferred.await(interrupted);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('closing the Ai layer stops a merged stream', () =>
		Effect.gen(function* () {
			const started = yield* Deferred.make<void>();
			const interrupted = yield* Deferred.make<void>();
			const scope = yield* Scope.make();
			const context = yield* Layer.buildWithScope(Ai.layer, scope);

			yield* Stream.runDrain(
				createUIMessageStream({
					execute: ({ writer }) =>
						writer.merge(
							Stream.fromEffect(Deferred.succeed(started, undefined)).pipe(
								Stream.drain,
								Stream.concat(Stream.never),
								Stream.onExit(() => Deferred.succeed(interrupted, undefined)),
							),
						),
				}),
			).pipe(Effect.provideContext(context), Effect.forkChild);
			yield* Deferred.await(started);

			yield* Scope.close(scope, Exit.void);
			yield* Deferred.await(interrupted);
		}).pipe(Effect.runPromise));

	test('types', () => {
		expectTypeOf(
			createUIMessageStream({ execute: () => Effect.void }),
		).toEqualTypeOf<
			Stream.Stream<AiSdk.InferUIMessageChunk<AiSdk.UIMessage>, AiError, Ai>
		>();
	});
});
