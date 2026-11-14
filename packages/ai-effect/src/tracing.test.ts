import * as AiSdk from 'ai';
import {
	MockEmbeddingModelV4,
	MockLanguageModelV4,
	MockRerankingModelV4,
} from 'ai/test';
import {
	Cause,
	Effect,
	Exit,
	Fiber,
	Layer,
	Option,
	Scope,
	Stream,
} from 'effect';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import { Ai } from '#src/ai.ts';
import { AbortedError, AiError, AuthenticationError } from '#src/ai-error.ts';
import { embed } from '#src/embed.ts';
import { embedMany } from '#src/embed-many.ts';
import { generateText } from '#src/generate-text.ts';
import {
	attributesOf,
	endedExit,
	makeTestTracer,
	parentSpanId,
	spanNamed,
} from '#src/internal/test-tracer.ts';
import {
	finishChunk,
	generateModel,
	hangingModel,
	makeTestLogger,
	run,
	streamModel,
	streamOf,
	textChunks,
	toolCallChunks,
	weatherInput,
} from '#src/internal/tracing-test-helpers.ts';
import { rerank } from '#src/rerank.ts';
import { streamText } from '#src/stream-text.ts';

vi.mock('ai', async (importOriginal) => {
	const actual = await importOriginal<typeof AiSdk>();
	return { ...actual, streamText: vi.fn(actual.streamText) };
});

const aiErrorOf = (span: Parameters<typeof endedExit>[0]) => {
	const error = Option.getOrUndefined(Exit.findErrorOption(endedExit(span)));
	if (!(error instanceof AiError)) throw new Error('expected an AiError');
	return error;
};

const settled = (check: () => void) => vi.waitFor(check, { timeout: 2000 });

beforeEach(() => {
	vi.mocked(AiSdk.streamText).mockClear();
});

describe('generateText spans', () => {
	test('records one internal span with request, model and usage attributes', async () => {
		const tracer = makeTestTracer();
		await run(
			generateText({
				model: generateModel(),
				prompt: 'hi',
				temperature: 0.5,
				maxOutputTokens: 100,
				seed: 7,
			}),
			tracer,
		);

		const span = spanNamed(tracer.spans, 'ai-effect.generateText');
		expect(span.kind).toBe('internal');
		expect(Exit.isSuccess(endedExit(span))).toBe(true);
		expect(attributesOf(span)).toEqual({
			'gen_ai.operation.name': 'invoke_agent',
			'gen_ai.request.temperature': 0.5,
			'gen_ai.request.max_tokens': 100,
			'gen_ai.request.seed': 7,
			'gen_ai.provider.name': 'openai',
			'gen_ai.request.model': 'gpt-test',
			'gen_ai.response.id': 'resp-1',
			'gen_ai.response.model': 'resp-model',
			'gen_ai.response.finish_reasons': ['stop'],
			'gen_ai.usage.input_tokens': 3,
			'gen_ai.usage.output_tokens': 10,
			'gen_ai.usage.cache_read.input_tokens': 2,
			'gen_ai.usage.reasoning.output_tokens': 4,
		});
	});

	test('parents under the caller span', async () => {
		const tracer = makeTestTracer();
		await run(
			generateText({ model: generateModel(), prompt: 'hi' }).pipe(
				Effect.withSpan('root'),
			),
			tracer,
		);

		expect(
			parentSpanId(spanNamed(tracer.spans, 'ai-effect.generateText')),
		).toBe(spanNamed(tracer.spans, 'root').spanId);
	});

	test('ends with the AiError and error.type on failure', async () => {
		const tracer = makeTestTracer();
		const model = new MockLanguageModelV4({
			doGenerate: async () => {
				throw new AiSdk.APICallError({
					message: 'unauthorized',
					url: 'https://example.test',
					requestBodyValues: {},
					statusCode: 401,
				});
			},
		});
		const error = await run(
			generateText({ model, prompt: 'hi', maxRetries: 0 }).pipe(Effect.flip),
			tracer,
		);

		const span = spanNamed(tracer.spans, 'ai-effect.generateText');
		const exit = endedExit(span);
		expect(Exit.findErrorOption(exit)).toEqual(Option.some(error));
		expect(error.reason).toBeInstanceOf(AuthenticationError);
		expect(attributesOf(span)['error.type']).toBe(
			'ai-effect/AuthenticationError',
		);
	});

	test('ends as an interrupt when the fiber is interrupted', async () => {
		const tracer = makeTestTracer();
		const model = new MockLanguageModelV4({
			doGenerate: ({ abortSignal }) =>
				new Promise((_resolve, reject) => {
					abortSignal?.addEventListener('abort', () =>
						reject(abortSignal.reason),
					);
				}),
		});

		await run(
			Effect.gen(function* () {
				const fiber = yield* Effect.forkChild(
					generateText({ model, prompt: 'hi' }),
				);
				yield* Effect.sleep('20 millis');
				yield* Fiber.interrupt(fiber);
			}),
			tracer,
		);

		const span = spanNamed(tracer.spans, 'ai-effect.generateText');
		expect(Exit.hasInterrupts(endedExit(span))).toBe(true);
	});

	test('maps tool-calls to the tool_call finish reason', async () => {
		const tracer = makeTestTracer();
		const weather = AiSdk.tool({
			inputSchema: weatherInput,
			execute: async () => 'sunny',
		});
		await run(
			generateText({
				model: generateModel(
					[
						{
							type: 'tool-call',
							toolCallId: 'call-1',
							toolName: 'weather',
							input: '{"city":"Paris"}',
						},
					],
					'tool-calls',
				),
				prompt: 'hi',
				tools: { weather },
			}),
			tracer,
		);

		expect(
			attributesOf(spanNamed(tracer.spans, 'ai-effect.generateText'))[
				'gen_ai.response.finish_reasons'
			],
		).toEqual(['tool_call']);
	});

	test.each(['onFinish', 'onEnd', 'experimental_onStart', 'onStart'] as const)(
		'still runs a user %s callback and still traces',
		async (key) => {
			const tracer = makeTestTracer();
			const callback = vi.fn(() => Effect.void);
			await run(
				generateText({
					model: generateModel(),
					prompt: 'hi',
					[key]: callback,
				}),
				tracer,
			);

			expect(callback).toHaveBeenCalledTimes(1);
			const attributes = attributesOf(
				spanNamed(tracer.spans, 'ai-effect.generateText'),
			);
			expect(attributes['gen_ai.request.model']).toBe('gpt-test');
			expect(attributes['gen_ai.usage.input_tokens']).toBe(3);
		},
	);
});

describe('streamText spans', () => {
	test('fully consumed: one internal span ending in success with attributes', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const result = yield* streamText({
					model: streamModel(textChunks('Hel', 'lo')),
					prompt: 'hi',
					temperature: 0.1,
				});
				const span = spanNamed(tracer.spans, 'ai-effect.streamText');
				expect(span.status._tag).toBe('Started');

				yield* Stream.runDrain(result.stream);
				yield* Effect.promise(() => settled(() => endedExit(span)));

				expect(span.kind).toBe('internal');
				expect(Exit.isSuccess(endedExit(span))).toBe(true);
				expect(attributesOf(span)).toMatchObject({
					'gen_ai.operation.name': 'invoke_agent',
					'gen_ai.request.stream': true,
					'gen_ai.request.temperature': 0.1,
					'gen_ai.provider.name': 'mock-provider',
					'gen_ai.request.model': 'mock-model-id',
					'gen_ai.response.id': 'resp-1',
					'gen_ai.response.finish_reasons': ['stop'],
					'gen_ai.usage.input_tokens': 3,
					'gen_ai.usage.output_tokens': 10,
				});
			}),
			tracer,
		);
	});

	test('parents under the caller span', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const result = yield* streamText({
					model: streamModel(textChunks('Hi')),
					prompt: 'hi',
				});
				yield* Stream.runDrain(result.stream);
			}).pipe(Effect.withSpan('root')),
			tracer,
		);

		expect(parentSpanId(spanNamed(tracer.spans, 'ai-effect.streamText'))).toBe(
			spanNamed(tracer.spans, 'root').spanId,
		);
	});

	test('consumer interrupted: the span ends as an interrupt', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const result = yield* streamText({
					model: hangingModel(textChunks('Hi').slice(0, 4)),
					prompt: 'hi',
				});
				const fiber = yield* Effect.forkChild(Stream.runDrain(result.stream));
				yield* Effect.sleep('20 millis');
				yield* Fiber.interrupt(fiber);

				const span = spanNamed(tracer.spans, 'ai-effect.streamText');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				expect(Exit.hasInterrupts(endedExit(span))).toBe(true);
				expect(attributesOf(span)['error.type']).toBeUndefined();
			}),
			tracer,
		);
	});

	test('user abortSignal: the span fails with an AbortedError', async () => {
		const tracer = makeTestTracer();
		const controller = new AbortController();
		await run(
			Effect.gen(function* () {
				const result = yield* streamText({
					model: hangingModel(textChunks('Hi').slice(0, 4)),
					prompt: 'hi',
					abortSignal: controller.signal,
				});
				const fiber = yield* Effect.forkChild(Stream.runDrain(result.stream));
				yield* Effect.sleep('20 millis');
				controller.abort();
				yield* Fiber.await(fiber);

				const span = spanNamed(tracer.spans, 'ai-effect.streamText');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				const error = aiErrorOf(span);
				expect(error.reason).toBeInstanceOf(AbortedError);
				expect(attributesOf(span)['error.type']).toBe('ai-effect/AbortedError');
			}),
			tracer,
		);
	});

	test('an SDK timeout fails the span with an AbortedError, not an interrupt', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const result = yield* streamText({
					model: hangingModel(textChunks('Hi').slice(0, 4)),
					prompt: 'hi',
					timeout: 30,
				});
				yield* Effect.exit(Stream.runDrain(result.stream));

				const span = spanNamed(tracer.spans, 'ai-effect.streamText');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				expect(Exit.hasInterrupts(endedExit(span))).toBe(false);
				const error = aiErrorOf(span);
				expect(error.reason).toBeInstanceOf(AbortedError);
				expect(error.reason.cause).toBeInstanceOf(DOMException);
				expect(attributesOf(span)['error.type']).toBe('ai-effect/AbortedError');
			}),
			tracer,
		);
	});

	test('an already-aborted user signal fails the span with an AbortedError', async () => {
		const tracer = makeTestTracer();
		const controller = new AbortController();
		controller.abort();
		await run(
			Effect.gen(function* () {
				yield* streamText({
					model: hangingModel(textChunks('Hi').slice(0, 4)),
					prompt: 'hi',
					abortSignal: controller.signal,
				});

				const span = spanNamed(tracer.spans, 'ai-effect.streamText');
				const error = aiErrorOf(span);
				expect(error.reason).toBeInstanceOf(AbortedError);
			}),
			tracer,
		);
	});

	test('the Ai layer closing ends an open span as an interrupt', async () => {
		const tracer = makeTestTracer();
		await Effect.gen(function* () {
			const scope = yield* Scope.make();
			const context = yield* Layer.buildWithScope(
				Layer.merge(Ai.layer, tracer.layer),
				scope,
			);
			yield* streamText({
				model: hangingModel(textChunks('Hi').slice(0, 4)),
				prompt: 'hi',
			}).pipe(Effect.provideContext(context));

			const span = spanNamed(tracer.spans, 'ai-effect.streamText');
			expect(span.status._tag).toBe('Started');
			yield* Scope.close(scope, Exit.void);
			expect(Exit.hasInterrupts(endedExit(span))).toBe(true);
		}).pipe(Effect.runPromise);
	});

	test('a setup throw ends the span with the AiError', async () => {
		const tracer = makeTestTracer();
		vi.mocked(AiSdk.streamText).mockImplementationOnce(() => {
			throw new Error('bad settings');
		});
		await run(
			Effect.gen(function* () {
				const error = yield* streamText({
					model: streamModel(textChunks('Hi')),
					prompt: 'hi',
				}).pipe(Effect.flip);

				const span = spanNamed(tracer.spans, 'ai-effect.streamText');
				expect(Exit.findErrorOption(endedExit(span))).toEqual(
					Option.some(error),
				);
				expect(attributesOf(span)['error.type']).toBe('ai-effect/UnknownError');
			}),
			tracer,
		);
	});

	test('a stream error part records the real error type, not NoOutputGenerated', async () => {
		const tracer = makeTestTracer();
		const unauthorized = new AiSdk.APICallError({
			message: 'unauthorized',
			url: 'https://example.test',
			requestBodyValues: {},
			statusCode: 401,
		});
		await run(
			Effect.gen(function* () {
				const result = yield* streamText({
					model: streamModel([
						{ type: 'stream-start', warnings: [] },
						{ type: 'error', error: unauthorized },
					]),
					prompt: 'hi',
					maxRetries: 0,
				});
				yield* Stream.runDrain(result.stream);

				const span = spanNamed(tracer.spans, 'ai-effect.streamText');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				const error = aiErrorOf(span);
				expect(error.reason).toBeInstanceOf(AuthenticationError);
				expect(attributesOf(span)['error.type']).toBe(
					'ai-effect/AuthenticationError',
				);
			}).pipe(Effect.provide(makeTestLogger().layer)),
			tracer,
		);
	});

	describe('stream error logging', () => {
		const failingModel = (error: unknown) =>
			streamModel([
				{ type: 'stream-start', warnings: [] },
				{ type: 'error', error },
			]);

		test('without a user onError the error is logged through the Effect Logger, not console.error', async () => {
			const tracer = makeTestTracer();
			const logger = makeTestLogger();
			const consoleError = vi
				.spyOn(console, 'error')
				.mockImplementation(() => undefined);
			const boom = new Error('boom');
			try {
				await run(
					Effect.gen(function* () {
						const result = yield* streamText({
							model: failingModel(boom),
							prompt: 'hi',
						});
						yield* Stream.runDrain(result.stream);
						yield* Effect.promise(() =>
							settled(() => expect(logger.entries).toHaveLength(1)),
						);
					}).pipe(Effect.provide(logger.layer)),
					tracer,
				);

				const entry = logger.entries[0];
				expect(entry?.level).toBe('Error');
				expect(entry?.message).toEqual(['streamText: stream error']);
				expect(entry && Cause.squash(entry.cause)).toBe(boom);
				expect(consoleError).not.toHaveBeenCalled();
			} finally {
				consoleError.mockRestore();
			}
		});

		test('a user onError replaces the log', async () => {
			const tracer = makeTestTracer();
			const logger = makeTestLogger();
			const onError = vi.fn(() => Effect.void);
			await run(
				Effect.gen(function* () {
					const result = yield* streamText({
						model: failingModel(new Error('boom')),
						prompt: 'hi',
						onError,
					});
					yield* Stream.runDrain(result.stream);
					yield* Effect.promise(() =>
						settled(() => expect(onError).toHaveBeenCalledTimes(1)),
					);
				}).pipe(Effect.provide(logger.layer)),
				tracer,
			);

			expect(logger.entries).toEqual([]);
		});
	});

	describe('stream retries', () => {
		const unauthorized = () =>
			new AiSdk.APICallError({
				message: 'unauthorized',
				url: 'https://example.test',
				requestBodyValues: {},
				statusCode: 401,
			});
		const failingChunks = (error: unknown) => [
			{ type: 'stream-start' as const, warnings: [] },
			{ type: 'error' as const, error },
		];
		const modelFailing = (times: number) => {
			let calls = 0;
			return new MockLanguageModelV4({
				doStream: async () => {
					calls++;
					return {
						stream: streamOf(
							calls <= times ? failingChunks(unauthorized()) : textChunks('Hi'),
						) as never,
					};
				},
			});
		};

		test('a retry that recovers ends the span as a success', async () => {
			const tracer = makeTestTracer();
			const consoleError = vi
				.spyOn(console, 'error')
				.mockImplementation(() => undefined);
			await run(
				Effect.gen(function* () {
					const result = yield* streamText({
						model: modelFailing(1),
						prompt: 'hi',
						maxRetries: 0,
						streamRetries: 1,
					});
					yield* Stream.runDrain(result.stream);

					const span = spanNamed(tracer.spans, 'ai-effect.streamText');
					yield* Effect.promise(() => settled(() => endedExit(span)));
					expect(Exit.isSuccess(endedExit(span))).toBe(true);
					expect(attributesOf(span)['error.type']).toBeUndefined();
				}),
				tracer,
			);
			consoleError.mockRestore();
		});

		test('exhausted retries end the span as a failure with the real error type', async () => {
			const tracer = makeTestTracer();
			const consoleError = vi
				.spyOn(console, 'error')
				.mockImplementation(() => undefined);
			await run(
				Effect.gen(function* () {
					const result = yield* streamText({
						model: modelFailing(5),
						prompt: 'hi',
						maxRetries: 0,
						streamRetries: 1,
					});
					yield* Stream.runDrain(result.stream);

					const span = spanNamed(tracer.spans, 'ai-effect.streamText');
					yield* Effect.promise(() => settled(() => endedExit(span)));
					expect(aiErrorOf(span).reason).toBeInstanceOf(AuthenticationError);
					expect(attributesOf(span)['error.type']).toBe(
						'ai-effect/AuthenticationError',
					);
				}),
				tracer,
			);
			consoleError.mockRestore();
		});

		test('a callback-requested retry that recovers ends the span as a success', async () => {
			const tracer = makeTestTracer();
			await run(
				Effect.gen(function* () {
					const result = yield* streamText({
						model: modelFailing(1),
						prompt: 'hi',
						maxRetries: 0,
						streamRetries: 0,
						onError: () => Effect.succeed({ retry: true }),
					});
					yield* Stream.runDrain(result.stream);

					const span = spanNamed(tracer.spans, 'ai-effect.streamText');
					yield* Effect.promise(() => settled(() => endedExit(span)));
					expect(Exit.isSuccess(endedExit(span))).toBe(true);
				}),
				tracer,
			);
		});
	});

	describe('terminal errors', () => {
		const unauthorized = () =>
			new AiSdk.APICallError({
				message: 'unauthorized',
				url: 'https://example.test',
				requestBodyValues: {},
				statusCode: 401,
			});
		const quiet = () =>
			vi.spyOn(console, 'error').mockImplementation(() => undefined);

		test.each([
			{ name: 'with stream retries', streamRetries: 1 },
			{ name: 'without stream retries', streamRetries: undefined },
		])(
			'prepareStep throwing on step 1 fails the span $name',
			async (options) => {
				const tracer = makeTestTracer();
				const consoleError = quiet();
				await run(
					Effect.gen(function* () {
						const result = yield* streamText({
							model: streamModel(textChunks('Hi')),
							prompt: 'hi',
							maxRetries: 0,
							streamRetries: options.streamRetries,
							prepareStep: () => {
								throw unauthorized();
							},
						});
						yield* Stream.runDrain(result.stream);

						const span = spanNamed(tracer.spans, 'ai-effect.streamText');
						yield* Effect.promise(() => settled(() => endedExit(span)));
						expect(aiErrorOf(span).reason).toBeInstanceOf(AuthenticationError);
						expect(attributesOf(span)['error.type']).toBe(
							'ai-effect/AuthenticationError',
						);
					}),
					tracer,
				);
				consoleError.mockRestore();
			},
		);

		test.each([
			{ name: 'with stream retries', streamRetries: 1 },
			{ name: 'without stream retries', streamRetries: undefined },
		])(
			'prepareStep throwing on step 2 after a tool step fails the span $name',
			async (options) => {
				const tracer = makeTestTracer();
				const consoleError = quiet();
				let calls = 0;
				const model = new MockLanguageModelV4({
					doStream: async () => {
						calls++;
						return {
							stream: streamOf(
								calls === 1 ? toolCallChunks('weather') : textChunks('Hi'),
							) as never,
						};
					},
				});
				await run(
					Effect.gen(function* () {
						const result = yield* streamText({
							model,
							prompt: 'hi',
							maxRetries: 0,
							streamRetries: options.streamRetries,
							stopWhen: AiSdk.isStepCount(3),
							tools: {
								weather: AiSdk.tool({
									inputSchema: weatherInput,
									execute: async () => 'sunny',
								}),
							},
							prepareStep: ({ stepNumber }) => {
								if (stepNumber === 1) throw unauthorized();
								return undefined;
							},
						});
						yield* Stream.runDrain(result.stream);

						const span = spanNamed(tracer.spans, 'ai-effect.streamText');
						yield* Effect.promise(() => settled(() => endedExit(span)));
						expect(aiErrorOf(span).reason).toBeInstanceOf(AuthenticationError);
					}),
					tracer,
				);
				consoleError.mockRestore();
			},
		);

		test('an error chunk without retries fails the span and keeps the end attributes', async () => {
			const tracer = makeTestTracer();
			const consoleError = quiet();
			await run(
				Effect.gen(function* () {
					const result = yield* streamText({
						model: streamModel([
							{ type: 'stream-start', warnings: [] },
							{ type: 'error', error: unauthorized() },
						]),
						prompt: 'hi',
						maxRetries: 0,
					});
					yield* Stream.runDrain(result.stream);

					const span = spanNamed(tracer.spans, 'ai-effect.streamText');
					yield* Effect.promise(() => settled(() => endedExit(span)));
					expect(aiErrorOf(span).reason).toBeInstanceOf(AuthenticationError);
					expect(attributesOf(span)['gen_ai.response.finish_reasons']).toEqual([
						'error',
					]);
				}),
				tracer,
			);
			consoleError.mockRestore();
		});

		test('draining only a text view still terminates the span', async () => {
			const tracer = makeTestTracer();
			const consoleError = quiet();
			await run(
				Effect.gen(function* () {
					const result = yield* streamText({
						model: streamModel(textChunks('Hi')),
						prompt: 'hi',
						maxRetries: 0,
						prepareStep: () => {
							throw unauthorized();
						},
					});
					yield* Effect.exit(Stream.runDrain(result.textStream));

					const span = spanNamed(tracer.spans, 'ai-effect.streamText');
					yield* Effect.promise(() => settled(() => endedExit(span)));
					expect(aiErrorOf(span).reason).toBeInstanceOf(AuthenticationError);
				}),
				tracer,
			);
			consoleError.mockRestore();
		});
	});

	test('does not drain the stream when the consumer stops early with Stream.take', async () => {
		const tracer = makeTestTracer();
		const total = 400;
		let pulled = 0;
		const model = new MockLanguageModelV4({
			doStream: async () => ({
				stream: new ReadableStream({
					start(controller) {
						controller.enqueue({ type: 'stream-start', warnings: [] });
						controller.enqueue({ type: 'text-start', id: '1' });
					},
					pull(controller) {
						pulled++;
						if (pulled > total) {
							controller.enqueue({ type: 'text-end', id: '1' });
							controller.enqueue(finishChunk());
							controller.close();
							return;
						}
						controller.enqueue({ type: 'text-delta', id: '1', delta: 'x' });
					},
				}) as never,
			}),
		});
		const userController = new AbortController();

		await run(
			Effect.gen(function* () {
				const result = yield* streamText({
					model,
					prompt: 'hi',
					abortSignal: userController.signal,
				});
				yield* Stream.runDrain(Stream.take(result.stream, 3));
				yield* Effect.sleep('50 millis');

				expect(pulled).toBeLessThan(total / 2);
				expect(
					spanNamed(tracer.spans, 'ai-effect.streamText').status._tag,
				).toBe('Started');
				userController.abort();
			}),
			tracer,
		);
	});

	test.each(['onFinish', 'onEnd'] as const)(
		'still runs a user %s callback and still traces',
		async (key) => {
			const tracer = makeTestTracer();
			const callback = vi.fn(() => Effect.void);
			await run(
				Effect.gen(function* () {
					const result = yield* streamText({
						model: streamModel(textChunks('Hi')),
						prompt: 'hi',
						[key]: callback,
					});
					yield* Stream.runDrain(result.stream);

					const span = spanNamed(tracer.spans, 'ai-effect.streamText');
					yield* Effect.promise(() => settled(() => endedExit(span)));
					expect(callback).toHaveBeenCalledTimes(1);
					expect(Exit.isSuccess(endedExit(span))).toBe(true);
					expect(attributesOf(span)['gen_ai.usage.input_tokens']).toBe(3);
				}),
				tracer,
			);
		},
	);

	test('still runs a user onError callback', async () => {
		const tracer = makeTestTracer();
		const onError = vi.fn(() => Effect.void);
		const unauthorized = new Error('boom');
		await run(
			Effect.gen(function* () {
				const result = yield* streamText({
					model: streamModel([
						{ type: 'stream-start', warnings: [] },
						{ type: 'error', error: unauthorized },
					]),
					prompt: 'hi',
					maxRetries: 0,
					onError,
				});
				yield* Stream.runDrain(result.stream);
				yield* Effect.promise(() =>
					settled(() => expect(onError).toHaveBeenCalledTimes(1)),
				);
			}),
			tracer,
		);
	});

	test('still runs a user onAbort callback', async () => {
		const tracer = makeTestTracer();
		const onAbort = vi.fn(() => Effect.void);
		const controller = new AbortController();
		await run(
			Effect.gen(function* () {
				const result = yield* streamText({
					model: hangingModel(textChunks('Hi').slice(0, 4)),
					prompt: 'hi',
					abortSignal: controller.signal,
					onAbort,
				});
				const fiber = yield* Effect.forkChild(Stream.runDrain(result.stream));
				yield* Effect.sleep('20 millis');
				controller.abort();
				yield* Fiber.await(fiber);

				yield* Effect.promise(() =>
					settled(() => expect(onAbort).toHaveBeenCalledTimes(1)),
				);
				const span = spanNamed(tracer.spans, 'ai-effect.streamText');
				expect(aiErrorOf(span).reason).toBeInstanceOf(AbortedError);
			}),
			tracer,
		);
	});
});

describe('embedding and rerank spans', () => {
	test('embed: a client span named after the model', async () => {
		const tracer = makeTestTracer();
		await run(
			embed({
				model: new MockEmbeddingModelV4({
					provider: 'openai.embedding',
					modelId: 'embed-test',
					doEmbed: {
						embeddings: [[0.1, 0.2, 0.3]],
						usage: { tokens: 5 },
						warnings: [],
					},
				}),
				value: 'hello',
			}),
			tracer,
		);

		const span = spanNamed(tracer.spans, 'embeddings embed-test');
		expect(span.kind).toBe('client');
		expect(Exit.isSuccess(endedExit(span))).toBe(true);
		expect(attributesOf(span)).toEqual({
			'gen_ai.operation.name': 'embeddings',
			'gen_ai.provider.name': 'openai',
			'gen_ai.request.model': 'embed-test',
			'gen_ai.usage.input_tokens': 5,
			'gen_ai.embeddings.dimension.count': 3,
		});
	});

	test('embed: failure ends the span with error.type', async () => {
		const tracer = makeTestTracer();
		const model = new MockEmbeddingModelV4({
			modelId: 'embed-test',
			doEmbed: async () => {
				throw new Error('down');
			},
		});
		await run(
			embed({ model, value: 'hello', maxRetries: 0 }).pipe(Effect.flip),
			tracer,
		);

		const span = spanNamed(tracer.spans, 'embeddings embed-test');
		expect(Exit.isFailure(endedExit(span))).toBe(true);
		expect(attributesOf(span)['error.type']).toBe('ai-effect/UnknownError');
	});

	test('embedMany: dimension count comes from the first embedding', async () => {
		const tracer = makeTestTracer();
		await run(
			embedMany({
				model: new MockEmbeddingModelV4({
					modelId: 'embed-test',
					maxEmbeddingsPerCall: 2,
					doEmbed: {
						embeddings: [
							[1, 2],
							[3, 4],
						],
						usage: { tokens: 8 },
						warnings: [],
					},
				}),
				values: ['a', 'b'],
			}),
			tracer,
		);

		const span = spanNamed(tracer.spans, 'embeddings embed-test');
		expect(span.kind).toBe('client');
		expect(attributesOf(span)).toMatchObject({
			'gen_ai.usage.input_tokens': 8,
			'gen_ai.embeddings.dimension.count': 2,
		});
	});

	test('embed with a string model id names the span after the string', async () => {
		const tracer = makeTestTracer();
		await run(
			embed({ model: 'openai/text-embedding-3-small', value: 'hello' }).pipe(
				Effect.flip,
			),
			tracer,
		);

		expect(
			spanNamed(tracer.spans, 'embeddings openai/text-embedding-3-small').kind,
		).toBe('client');
	});

	test('rerank: a client span with response attributes', async () => {
		const tracer = makeTestTracer();
		await run(
			rerank({
				model: new MockRerankingModelV4({
					provider: 'cohere.reranking',
					modelId: 'rerank-test',
					doRerank: async () => ({
						ranking: [
							{ index: 1, relevanceScore: 0.9 },
							{ index: 0, relevanceScore: 0.1 },
						],
						warnings: [],
						response: { id: 'rr-1', modelId: 'rerank-resp' },
					}),
				}),
				query: 'greeting',
				documents: ['bye', 'hello'],
			}),
			tracer,
		);

		const span = spanNamed(tracer.spans, 'rerank rerank-test');
		expect(span.kind).toBe('client');
		expect(attributesOf(span)).toEqual({
			'gen_ai.operation.name': 'rerank',
			'gen_ai.provider.name': 'cohere',
			'gen_ai.request.model': 'rerank-test',
			'gen_ai.response.id': 'rr-1',
			'gen_ai.response.model': 'rerank-resp',
		});
	});

	test.each([
		['embed', 'experimental_onEnd'],
		['embed', 'onEnd'],
	] as const)('%s still runs a user %s callback', async (_name, key) => {
		const tracer = makeTestTracer();
		const callback = vi.fn(() => Effect.void);
		await run(
			embed({
				model: new MockEmbeddingModelV4({
					modelId: 'embed-test',
					doEmbed: { embeddings: [[1]], usage: { tokens: 1 }, warnings: [] },
				}),
				value: 'hello',
				[key]: callback,
			}),
			tracer,
		);

		expect(callback).toHaveBeenCalledTimes(1);
		expect(
			attributesOf(spanNamed(tracer.spans, 'embeddings embed-test'))[
				'gen_ai.usage.input_tokens'
			],
		).toBe(1);
	});
});
