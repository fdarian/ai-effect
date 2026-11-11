import * as AiSdk from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import {
	Effect,
	Exit,
	Fiber,
	Layer,
	Option,
	Schema,
	Scope,
	Stream,
} from 'effect';
import { describe, expect, test, vi } from 'vitest';
import { Ai } from '#src/ai.ts';
import {
	AbortedError,
	AiError,
	AuthenticationError,
	InvalidRequestError,
} from '#src/ai-error.ts';
import { createAgentUIStream } from '#src/create-agent-ui-stream.ts';
import { createUIMessageStream } from '#src/create-ui-message-stream.ts';
import {
	attributesOf,
	endedExit,
	makeTestTracer,
	parentSpanId,
	spanNamed,
	spansNamed,
} from '#src/internal/test-tracer.ts';
import {
	generateModel,
	hangingModel,
	run,
	streamModel,
	textChunks,
	toolCallChunks,
	usage,
	weatherInput,
} from '#src/internal/tracing-test-helpers.ts';
import { streamText } from '#src/stream-text.ts';
import { tool } from '#src/tool.ts';
import { ToolLoopAgent } from '#src/tool-loop-agent.ts';

const settled = (check: () => void) => vi.waitFor(check, { timeout: 2000 });

const aiErrorOf = (span: Parameters<typeof endedExit>[0]) => {
	const error = Option.getOrUndefined(Exit.findErrorOption(endedExit(span)));
	if (!(error instanceof AiError)) throw new Error('expected an AiError');
	return error;
};

const uiMessages = [
	{ id: 'm1', role: 'user', parts: [{ type: 'text', text: 'hi' }] },
];

const partialChunks = textChunks('Hi').slice(0, 4);

const toolThenTextGenerateModel = () => {
	let calls = 0;
	return new MockLanguageModelV4({
		provider: 'openai.chat',
		modelId: 'gpt-test',
		doGenerate: async () => {
			calls += 1;
			const first = calls === 1;
			return {
				content: first
					? [
							{
								type: 'tool-call' as const,
								toolCallId: 'call-1',
								toolName: 'weather',
								input: '{"city":"Paris"}',
							},
						]
					: [{ type: 'text' as const, text: 'sunny' }],
				finishReason: first
					? { unified: 'tool-calls' as const, raw: undefined }
					: { unified: 'stop' as const, raw: undefined },
				usage,
				warnings: [],
			};
		},
	});
};

const toolThenTextStreamModel = () => {
	let calls = 0;
	return new MockLanguageModelV4({
		doStream: async () => {
			calls += 1;
			const chunks =
				calls === 1 ? toolCallChunks('weather') : textChunks('sunny');
			return {
				stream: new ReadableStream({
					start(controller) {
						for (const chunk of chunks) controller.enqueue(chunk);
						controller.close();
					},
				}) as never,
			};
		},
	});
};

const unauthorized = () =>
	new AiSdk.APICallError({
		message: 'unauthorized',
		url: 'https://example.test',
		requestBodyValues: {},
		statusCode: 401,
	});

describe('ToolLoopAgent.generate spans', () => {
	test('names the span after the agent id and records attributes', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'weather-agent',
					model: generateModel(),
					temperature: 0.2,
				});
				yield* agent.generate({ prompt: 'hi' }).pipe(Effect.withSpan('root'));
			}),
			tracer,
		);

		const span = spanNamed(tracer.spans, 'invoke_agent weather-agent');
		expect(span.kind).toBe('internal');
		expect(parentSpanId(span)).toBe(spanNamed(tracer.spans, 'root').spanId);
		expect(Exit.isSuccess(endedExit(span))).toBe(true);
		expect(attributesOf(span)).toMatchObject({
			'gen_ai.operation.name': 'invoke_agent',
			'gen_ai.agent.name': 'weather-agent',
			'gen_ai.request.temperature': 0.2,
			'gen_ai.provider.name': 'openai',
			'gen_ai.request.model': 'gpt-test',
			'gen_ai.response.id': 'resp-1',
			'gen_ai.response.finish_reasons': ['stop'],
			'gen_ai.usage.input_tokens': 3,
			'gen_ai.usage.output_tokens': 10,
		});
	});

	test('an agent without an id gets a method-named span and no agent name', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({ model: generateModel() });
				yield* agent.generate({ prompt: 'hi' });
			}),
			tracer,
		);

		const span = spanNamed(tracer.spans, 'ai-effect.ToolLoopAgent.generate');
		expect(attributesOf(span)['gen_ai.agent.name']).toBeUndefined();
	});

	test('ends with the AiError and error.type on failure', async () => {
		const tracer = makeTestTracer();
		const model = new MockLanguageModelV4({
			doGenerate: async () => {
				throw unauthorized();
			},
		});
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model,
					maxRetries: 0,
				});
				const error = yield* agent.generate({ prompt: 'hi' }).pipe(Effect.flip);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				expect(Exit.findErrorOption(endedExit(span))).toEqual(
					Option.some(error),
				);
				expect(attributesOf(span)['error.type']).toBe(
					'ai-effect/AuthenticationError',
				);
			}),
			tracer,
		);
	});

	test('still runs settings-level and call-level user callbacks', async () => {
		const tracer = makeTestTracer();
		const seen: string[] = [];
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: generateModel(),
					onStart: () => Effect.sync(() => void seen.push('settings:start')),
					onFinish: () => Effect.sync(() => void seen.push('settings:finish')),
				});
				yield* agent.generate({
					prompt: 'hi',
					onStart: () => Effect.sync(() => void seen.push('call:start')),
					onEnd: () => Effect.sync(() => void seen.push('call:end')),
				});
			}),
			tracer,
		);

		await settled(() =>
			expect([...seen].sort()).toEqual([
				'call:end',
				'call:start',
				'settings:finish',
				'settings:start',
			]),
		);
		expect(
			attributesOf(spanNamed(tracer.spans, 'invoke_agent a')),
		).toMatchObject({ 'gen_ai.request.model': 'gpt-test' });
	});

	test('settings-level callbacks run under the agent span, not the span `make` ran in', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: generateModel(),
					onStart: () => Effect.void.pipe(Effect.withSpan('settings-callback')),
				}).pipe(Effect.withSpan('construction'));
				yield* agent.generate({ prompt: 'hi' });
			}),
			tracer,
		);

		await settled(() => spanNamed(tracer.spans, 'settings-callback'));
		expect(parentSpanId(spanNamed(tracer.spans, 'settings-callback'))).toBe(
			spanNamed(tracer.spans, 'invoke_agent a').spanId,
		);
	});

	test('raw produces no spans', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: generateModel(),
				});
				yield* Effect.promise(() => agent.raw.generate({ prompt: 'hi' }));
			}),
			tracer,
		);

		expect(tracer.spans).toEqual([]);
	});
});

describe('ToolLoopAgent tool spans', () => {
	test('generate: ai-effect and foreign tools are parented under the agent span with the agent name', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const weather = yield* tool({
					description: 'Get the weather',
					inputSchema: Schema.Struct({ city: Schema.String }),
					execute: () => Effect.succeed('sunny').pipe(Effect.withSpan('inner')),
				});
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: toolThenTextGenerateModel(),
					tools: { weather },
				});
				yield* agent.generate({ prompt: 'hi' });
			}),
			tracer,
		);

		const agentSpan = spanNamed(tracer.spans, 'invoke_agent a');
		const toolSpan = spanNamed(tracer.spans, 'execute_tool weather');
		expect(parentSpanId(toolSpan)).toBe(agentSpan.spanId);
		expect(parentSpanId(spanNamed(tracer.spans, 'inner'))).toBe(
			toolSpan.spanId,
		);
		expect(attributesOf(toolSpan)['gen_ai.agent.name']).toBe('a');
	});

	test('stream: a foreign tool gets a span under the agent span', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: toolThenTextStreamModel(),
					tools: {
						weather: AiSdk.tool({
							inputSchema: weatherInput,
							execute: async (input) => `sunny in ${input.city}`,
						}),
					},
				});
				const result = yield* agent.stream({ prompt: 'hi' });
				yield* Stream.runDrain(result.stream);
			}),
			tracer,
		);

		const toolSpan = spanNamed(tracer.spans, 'execute_tool weather');
		expect(parentSpanId(toolSpan)).toBe(
			spanNamed(tracer.spans, 'invoke_agent a').spanId,
		);
		expect(attributesOf(toolSpan)['gen_ai.agent.name']).toBe('a');
		expect(Exit.isSuccess(endedExit(toolSpan))).toBe(true);
	});

	test('tools returned by prepareCall get spans', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: toolThenTextGenerateModel(),
					prepareCall: (options) => ({
						...options,
						tools: {
							weather: AiSdk.tool({
								inputSchema: weatherInput,
								execute: async () => 'sunny',
							}),
						},
					}),
				});
				yield* agent.generate({ prompt: 'hi' });
			}),
			tracer,
		);

		expect(parentSpanId(spanNamed(tracer.spans, 'execute_tool weather'))).toBe(
			spanNamed(tracer.spans, 'invoke_agent a').spanId,
		);
	});

	test('prepareCall handing the settings tools back does not add a second span', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: toolThenTextGenerateModel(),
					tools: {
						weather: AiSdk.tool({
							inputSchema: weatherInput,
							execute: async () => 'sunny',
						}),
					},
					prepareCall: (options) => ({ ...options }),
				});
				yield* agent.generate({ prompt: 'hi' });
			}),
			tracer,
		);

		expect(spansNamed(tracer.spans, 'execute_tool weather')).toHaveLength(1);
	});
});

describe('ToolLoopAgent.stream spans', () => {
	test('fully consumed: one internal span ending in success with attributes', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel(textChunks('Hel', 'lo')),
				});
				const result = yield* agent
					.stream({ prompt: 'hi' })
					.pipe(Effect.withSpan('root'));
				const span = spanNamed(tracer.spans, 'invoke_agent a');
				expect(span.status._tag).toBe('Started');

				yield* Stream.runDrain(result.stream);
				yield* Effect.promise(() => settled(() => endedExit(span)));

				expect(span.kind).toBe('internal');
				expect(parentSpanId(span)).toBe(spanNamed(tracer.spans, 'root').spanId);
				expect(Exit.isSuccess(endedExit(span))).toBe(true);
				expect(attributesOf(span)).toMatchObject({
					'gen_ai.operation.name': 'invoke_agent',
					'gen_ai.agent.name': 'a',
					'gen_ai.request.stream': true,
					'gen_ai.response.id': 'resp-1',
					'gen_ai.response.finish_reasons': ['stop'],
					'gen_ai.usage.input_tokens': 3,
					'gen_ai.usage.output_tokens': 10,
				});
			}),
			tracer,
		);
	});

	test('an agent without an id gets a method-named span', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					model: streamModel(textChunks('Hi')),
				});
				const result = yield* agent.stream({ prompt: 'hi' });
				yield* Stream.runDrain(result.stream);
			}),
			tracer,
		);

		spanNamed(tracer.spans, 'ai-effect.ToolLoopAgent.stream');
	});

	test('consumer interrupted: the span ends as an interrupt', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: hangingModel(partialChunks),
				});
				const result = yield* agent.stream({ prompt: 'hi' });
				const fiber = yield* Effect.forkChild(Stream.runDrain(result.stream));
				yield* Effect.sleep('20 millis');
				yield* Fiber.interrupt(fiber);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
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
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: hangingModel(partialChunks),
				});
				const result = yield* agent.stream({
					prompt: 'hi',
					abortSignal: controller.signal,
				});
				const fiber = yield* Effect.forkChild(Stream.runDrain(result.stream));
				yield* Effect.sleep('20 millis');
				controller.abort();
				yield* Fiber.await(fiber);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				expect(aiErrorOf(span).reason).toBeInstanceOf(AbortedError);
				expect(aiErrorOf(span).method).toBe('ToolLoopAgent.stream');
				expect(attributesOf(span)['error.type']).toBe('ai-effect/AbortedError');
			}),
			tracer,
		);
	});

	test('an SDK timeout fails the span with an AbortedError, not an interrupt', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: hangingModel(partialChunks),
				});
				const result = yield* agent.stream({ prompt: 'hi', timeout: 30 });
				yield* Effect.exit(Stream.runDrain(result.stream));

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				expect(Exit.hasInterrupts(endedExit(span))).toBe(false);
				expect(aiErrorOf(span).reason).toBeInstanceOf(AbortedError);
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
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: hangingModel(partialChunks),
				});
				yield* agent
					.stream({ prompt: 'hi', abortSignal: controller.signal })
					.pipe(Effect.exit);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				expect(aiErrorOf(span).reason).toBeInstanceOf(AbortedError);
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
			yield* Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: hangingModel(partialChunks),
				});
				yield* agent.stream({ prompt: 'hi' });
			}).pipe(Effect.provideContext(context));

			const span = spanNamed(tracer.spans, 'invoke_agent a');
			expect(span.status._tag).toBe('Started');
			yield* Scope.close(scope, Exit.void);
			expect(Exit.hasInterrupts(endedExit(span))).toBe(true);
		}).pipe(Effect.runPromise);
	});

	test('a setup failure ends the span with the AiError', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel(textChunks('Hi')),
					prepareCall: () => {
						throw new Error('bad settings');
					},
				});
				const error = yield* agent.stream({ prompt: 'hi' }).pipe(Effect.flip);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				expect(Exit.findErrorOption(endedExit(span))).toEqual(
					Option.some(error),
				);
				expect(attributesOf(span)['error.type']).toBe('ai-effect/UnknownError');
			}),
			tracer,
		);
	});

	test('a stream error records the real error type and still runs the user onError', async () => {
		const tracer = makeTestTracer();
		const onError = vi.fn(() => Effect.void);
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel([
						{ type: 'stream-start', warnings: [] },
						{ type: 'error', error: unauthorized() },
					]),
					maxRetries: 0,
					onError,
				});
				const result = yield* agent.stream({ prompt: 'hi' });
				yield* Stream.runDrain(result.stream);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				expect(attributesOf(span)['error.type']).toBe(
					'ai-effect/AuthenticationError',
				);
				yield* Effect.promise(() =>
					settled(() => expect(onError).toHaveBeenCalledTimes(1)),
				);
			}),
			tracer,
		);
	});

	test('a stream error keeps the end attributes on the failed span', async () => {
		const tracer = makeTestTracer();
		const consoleError = vi
			.spyOn(console, 'error')
			.mockImplementation(() => undefined);
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel([
						{ type: 'stream-start', warnings: [] },
						{ type: 'error', error: unauthorized() },
					]),
					maxRetries: 0,
				});
				const result = yield* agent.stream({ prompt: 'hi' });
				yield* Stream.runDrain(result.stream);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
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

	test.each([
		{ step: 0, name: 'step 1' },
		{ step: 1, name: 'step 2 after a tool step' },
	])('prepareStep throwing on $name fails the span', async (options) => {
		const tracer = makeTestTracer();
		const consoleError = vi
			.spyOn(console, 'error')
			.mockImplementation(() => undefined);
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: toolThenTextStreamModel(),
					maxRetries: 0,
					stopWhen: AiSdk.isStepCount(3),
					tools: {
						weather: AiSdk.tool({
							inputSchema: weatherInput,
							execute: async () => 'sunny',
						}),
					},
					prepareStep: ({ stepNumber }) => {
						if (stepNumber === options.step) throw unauthorized();
						return undefined;
					},
				});
				const result = yield* agent.stream({ prompt: 'hi' });
				yield* Stream.runDrain(result.stream);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				expect(aiErrorOf(span).reason).toBeInstanceOf(AuthenticationError);
			}),
			tracer,
		);
		consoleError.mockRestore();
	});

	test('without a user onError the SDK default of logging the error is kept', async () => {
		const tracer = makeTestTracer();
		const consoleError = vi
			.spyOn(console, 'error')
			.mockImplementation(() => undefined);
		try {
			await run(
				Effect.gen(function* () {
					const agent = yield* ToolLoopAgent.make({
						id: 'a',
						model: streamModel([
							{ type: 'stream-start', warnings: [] },
							{ type: 'error', error: new Error('boom') },
						]),
					});
					const result = yield* agent.stream({ prompt: 'hi' });
					yield* Stream.runDrain(result.stream);
				}),
				tracer,
			);
			await settled(() => expect(consoleError).toHaveBeenCalledTimes(1));
		} finally {
			consoleError.mockRestore();
		}
	});

	test('still runs settings-level and call-level user callbacks', async () => {
		const tracer = makeTestTracer();
		const seen: string[] = [];
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel(textChunks('Hi')),
					onStart: () => Effect.sync(() => void seen.push('settings:start')),
					onFinish: () => Effect.sync(() => void seen.push('settings:finish')),
				});
				const result = yield* agent.stream({
					prompt: 'hi',
					onEnd: () => Effect.sync(() => void seen.push('call:end')),
				});
				yield* Stream.runDrain(result.stream);
			}),
			tracer,
		);

		await settled(() =>
			expect([...seen].sort()).toEqual([
				'call:end',
				'settings:finish',
				'settings:start',
			]),
		);
		const span = spanNamed(tracer.spans, 'invoke_agent a');
		await settled(() => endedExit(span));
		expect(attributesOf(span)['gen_ai.request.model']).toBe('mock-model-id');
	});

	test('settings-level callbacks run under the agent span, not the span `make` ran in', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel(textChunks('Hi')),
					onStart: () => Effect.void.pipe(Effect.withSpan('settings-callback')),
				}).pipe(Effect.withSpan('construction'));
				const result = yield* agent.stream({ prompt: 'hi' });
				yield* Stream.runDrain(result.stream);
			}),
			tracer,
		);

		await settled(() => spanNamed(tracer.spans, 'settings-callback'));
		expect(parentSpanId(spanNamed(tracer.spans, 'settings-callback'))).toBe(
			spanNamed(tracer.spans, 'invoke_agent a').spanId,
		);
	});

	test('raw produces no spans', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel(textChunks('Hi')),
				});
				const result = yield* Effect.promise(() =>
					agent.raw.stream({ prompt: 'hi' }),
				);
				yield* Effect.promise(() => result.consumeStream());
			}),
			tracer,
		);

		expect(tracer.spans).toEqual([]);
	});
});

describe('createAgentUIStream spans', () => {
	const makeAgent = (model: MockLanguageModelV4) =>
		ToolLoopAgent.make({ id: 'a', model });

	test('fully consumed: one internal span ending in success with attributes', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* makeAgent(streamModel(textChunks('Hi')));
				const stream = yield* createAgentUIStream({ agent, uiMessages }).pipe(
					Effect.withSpan('root'),
				);
				const span = spanNamed(tracer.spans, 'invoke_agent a');
				expect(span.status._tag).toBe('Started');

				yield* Stream.runDrain(stream);
				yield* Effect.promise(() => settled(() => endedExit(span)));

				expect(span.kind).toBe('internal');
				expect(parentSpanId(span)).toBe(spanNamed(tracer.spans, 'root').spanId);
				expect(Exit.isSuccess(endedExit(span))).toBe(true);
				expect(attributesOf(span)).toMatchObject({
					'gen_ai.operation.name': 'invoke_agent',
					'gen_ai.agent.name': 'a',
					'gen_ai.request.stream': true,
					'gen_ai.usage.output_tokens': 10,
				});
				// The agent's own stream() is not traced a second time.
				expect(spansNamed(tracer.spans, 'invoke_agent a')).toHaveLength(1);
			}),
			tracer,
		);
	});

	test('an agent without an id gets a method-named span', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					model: streamModel(textChunks('Hi')),
				});
				const stream = yield* createAgentUIStream({ agent, uiMessages });
				yield* Stream.runDrain(stream);
			}),
			tracer,
		);

		spanNamed(tracer.spans, 'ai-effect.createAgentUIStream');
	});

	test('tools run under the agent span', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: toolThenTextStreamModel(),
					tools: {
						weather: AiSdk.tool({
							inputSchema: weatherInput,
							execute: async () => 'sunny',
						}),
					},
				});
				const stream = yield* createAgentUIStream({ agent, uiMessages });
				yield* Stream.runDrain(stream);
			}),
			tracer,
		);

		expect(parentSpanId(spanNamed(tracer.spans, 'execute_tool weather'))).toBe(
			spanNamed(tracer.spans, 'invoke_agent a').spanId,
		);
	});

	test('consumer interrupted: the span ends as an interrupt', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* makeAgent(hangingModel(partialChunks));
				const stream = yield* createAgentUIStream({ agent, uiMessages });
				const fiber = yield* Effect.forkChild(Stream.runDrain(stream));
				yield* Effect.sleep('20 millis');
				yield* Fiber.interrupt(fiber);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				expect(Exit.hasInterrupts(endedExit(span))).toBe(true);
			}),
			tracer,
		);
	});

	test('user abortSignal: the span fails with an AbortedError', async () => {
		const tracer = makeTestTracer();
		const controller = new AbortController();
		await run(
			Effect.gen(function* () {
				const agent = yield* makeAgent(hangingModel(partialChunks));
				const stream = yield* createAgentUIStream({
					agent,
					uiMessages,
					abortSignal: controller.signal,
				});
				const fiber = yield* Effect.forkChild(Stream.runDrain(stream));
				yield* Effect.sleep('20 millis');
				controller.abort();
				yield* Fiber.await(fiber);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				const error = aiErrorOf(span);
				expect(error.reason).toBeInstanceOf(AbortedError);
				expect(error.method).toBe('createAgentUIStream');
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
				const agent = yield* makeAgent(hangingModel(partialChunks));
				yield* createAgentUIStream({
					agent,
					uiMessages,
					abortSignal: controller.signal,
				}).pipe(Effect.exit);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				expect(aiErrorOf(span).reason).toBeInstanceOf(AbortedError);
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
			yield* Effect.gen(function* () {
				const agent = yield* makeAgent(hangingModel(partialChunks));
				yield* createAgentUIStream({ agent, uiMessages });
			}).pipe(Effect.provideContext(context));

			const span = spanNamed(tracer.spans, 'invoke_agent a');
			expect(span.status._tag).toBe('Started');
			yield* Scope.close(scope, Exit.void);
			expect(Exit.hasInterrupts(endedExit(span))).toBe(true);
		}).pipe(Effect.runPromise);
	});

	test('a stream error keeps the end attributes on the failed span', async () => {
		const tracer = makeTestTracer();
		const consoleError = vi
			.spyOn(console, 'error')
			.mockImplementation(() => undefined);
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel([
						{ type: 'stream-start', warnings: [] },
						{ type: 'error', error: unauthorized() },
					]),
					maxRetries: 0,
				});
				const stream = yield* createAgentUIStream({ agent, uiMessages });
				yield* Stream.runDrain(stream);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
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

	test('prepareStep throwing on step 1 fails the span', async () => {
		const tracer = makeTestTracer();
		const consoleError = vi
			.spyOn(console, 'error')
			.mockImplementation(() => undefined);
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel(textChunks('Hi')),
					maxRetries: 0,
					prepareStep: () => {
						throw unauthorized();
					},
				});
				const stream = yield* createAgentUIStream({ agent, uiMessages });
				yield* Stream.runDrain(stream);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				yield* Effect.promise(() => settled(() => endedExit(span)));
				expect(aiErrorOf(span).reason).toBeInstanceOf(AuthenticationError);
			}),
			tracer,
		);
		consoleError.mockRestore();
	});

	test('an agent exposes its hooks under a global symbol so another ai-effect copy interoperates', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* makeAgent(streamModel(textChunks('Hi')));
				const foreign = { [Symbol.for('ai-effect/agent-tracing')]: undefined };
				expect(Object.getOwnPropertySymbols(agent)).toEqual(
					Object.getOwnPropertySymbols(foreign),
				);
			}),
			tracer,
		);
	});

	test('invalid uiMessages end the span with the InvalidRequestError', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* makeAgent(streamModel(textChunks('Hi')));
				const error = yield* createAgentUIStream({
					agent,
					uiMessages: [{ not: 'a message' }],
				}).pipe(Effect.flip);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				expect(error.reason).toBeInstanceOf(InvalidRequestError);
				expect(Exit.findErrorOption(endedExit(span))).toEqual(
					Option.some(error),
				);
				expect(attributesOf(span)['error.type']).toBe(
					'ai-effect/InvalidRequestError',
				);
			}),
			tracer,
		);
	});

	test('a failed setup leaves no abort listener behind on the signals', async () => {
		const tracer = makeTestTracer();
		const open = new Set<unknown>();
		const add = AbortSignal.prototype.addEventListener;
		const remove = AbortSignal.prototype.removeEventListener;
		vi.spyOn(AbortSignal.prototype, 'addEventListener').mockImplementation(
			function (this: AbortSignal, type, listener, options) {
				if (type === 'abort') open.add(listener);
				return add.call(this, type, listener, options);
			},
		);
		vi.spyOn(AbortSignal.prototype, 'removeEventListener').mockImplementation(
			function (this: AbortSignal, type, listener, options) {
				if (type === 'abort') open.delete(listener);
				return remove.call(this, type, listener, options);
			},
		);
		try {
			await run(
				Effect.gen(function* () {
					const agent = yield* makeAgent(streamModel(textChunks('Hi')));
					yield* createAgentUIStream({
						agent,
						uiMessages: [{ not: 'a message' }],
					}).pipe(Effect.flip);

					// Checked before the `Ai` layer closes: closing aborts its signal,
					// which would run and remove a leaked listener.
					expect(open.size).toBe(0);
				}),
				tracer,
			);
		} finally {
			vi.restoreAllMocks();
		}
	});

	test('a model-start failure ends the span with the AiError', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel(textChunks('Hi')),
					prepareCall: () => {
						throw new Error('bad settings');
					},
				});
				const error = yield* createAgentUIStream({ agent, uiMessages }).pipe(
					Effect.flip,
				);

				const span = spanNamed(tracer.spans, 'invoke_agent a');
				expect(Exit.findErrorOption(endedExit(span))).toEqual(
					Option.some(error),
				);
			}),
			tracer,
		);
	});

	test('still runs settings-level and call-level user callbacks', async () => {
		const tracer = makeTestTracer();
		const seen: string[] = [];
		await run(
			Effect.gen(function* () {
				const agent = yield* ToolLoopAgent.make({
					id: 'a',
					model: streamModel(textChunks('Hi')),
					onFinish: () => Effect.sync(() => void seen.push('settings:finish')),
				});
				const stream = yield* createAgentUIStream({
					agent,
					uiMessages,
					onStepEnd: () => Effect.sync(() => void seen.push('call:stepEnd')),
				});
				yield* Stream.runDrain(stream);
			}),
			tracer,
		);

		await settled(() =>
			expect([...seen].sort()).toEqual(['call:stepEnd', 'settings:finish']),
		);
	});
});

describe('createUIMessageStream spans', () => {
	test('one internal span covers the run, and a nested streamText parents under it', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const chunks = yield* Stream.runCollect(
					createUIMessageStream({
						execute: ({ writer }) =>
							Effect.gen(function* () {
								const result = yield* streamText({
									model: streamModel(textChunks('Hi')),
									prompt: 'hi',
								});
								yield* writer.merge(result.toUIMessageStream());
							}).pipe(Effect.withSpan('in-execute')),
					}).pipe(Stream.withSpan('root')),
				);
				expect(chunks.map((chunk) => chunk.type)).toContain('finish');

				const span = spanNamed(tracer.spans, 'ai-effect.createUIMessageStream');
				expect(span.kind).toBe('internal');
				expect(parentSpanId(span)).toBe(spanNamed(tracer.spans, 'root').spanId);
				expect(Exit.isSuccess(endedExit(span))).toBe(true);
				expect(attributesOf(span)['gen_ai.operation.name']).toBeUndefined();
				expect(parentSpanId(spanNamed(tracer.spans, 'in-execute'))).toBe(
					span.spanId,
				);
				expect(
					parentSpanId(spanNamed(tracer.spans, 'ai-effect.streamText')),
				).toBe(spanNamed(tracer.spans, 'in-execute').spanId);
			}),
			tracer,
		);
	});
});
