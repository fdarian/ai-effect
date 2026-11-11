import * as AiSdk from 'ai';
import { Effect, Exit, Schema, Stream } from 'effect';
import { describe, expect, test } from 'vitest';
import { generateText } from '#src/generate-text.ts';
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
	run,
	streamModel,
	toolCallChunks,
	weatherInput,
} from '#src/internal/tracing-test-helpers.ts';
import { streamText } from '#src/stream-text.ts';
import { tool } from '#src/tool.ts';

describe('tool spans', () => {
	const toolCallModel = () =>
		generateModel(
			[
				{
					type: 'tool-call',
					toolCallId: 'call-1',
					toolName: 'weather',
					input: '{"city":"Paris"}',
				},
			],
			'tool-calls',
		);

	test('an ai-effect tool is parented under the call span, not where tool() ran', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const weather = yield* tool({
					description: 'Get the weather',
					inputSchema: Schema.Struct({ city: Schema.String }),
					execute: () => Effect.succeed('sunny').pipe(Effect.withSpan('inner')),
				}).pipe(Effect.withSpan('built-here'));

				yield* generateText({
					model: toolCallModel(),
					prompt: 'hi',
					tools: { weather },
				}).pipe(Effect.withSpan('called-here'));
			}),
			tracer,
		);

		const callSpan = spanNamed(tracer.spans, 'ai-effect.generateText');
		const toolSpan = spanNamed(tracer.spans, 'execute_tool weather');
		expect(parentSpanId(callSpan)).toBe(
			spanNamed(tracer.spans, 'called-here').spanId,
		);
		expect(parentSpanId(toolSpan)).toBe(callSpan.spanId);
		expect(parentSpanId(spanNamed(tracer.spans, 'inner'))).toBe(
			toolSpan.spanId,
		);
		expect(toolSpan.kind).toBe('internal');
		expect(Exit.isSuccess(endedExit(toolSpan))).toBe(true);
		expect(attributesOf(toolSpan)).toEqual({
			'gen_ai.operation.name': 'execute_tool',
			'gen_ai.tool.name': 'weather',
			'gen_ai.tool.call.id': 'call-1',
			'gen_ai.tool.type': 'function',
			'gen_ai.tool.description': 'Get the weather',
		});
	});

	test('a foreign promise tool gets a span', async () => {
		const tracer = makeTestTracer();
		const weather = AiSdk.tool({
			inputSchema: weatherInput,
			execute: async (input) => `sunny in ${input.city}`,
		});
		const result = await run(
			generateText({
				model: toolCallModel(),
				prompt: 'hi',
				tools: { weather },
			}),
			tracer,
		);

		const toolSpan = spanNamed(tracer.spans, 'execute_tool weather');
		expect(parentSpanId(toolSpan)).toBe(
			spanNamed(tracer.spans, 'ai-effect.generateText').spanId,
		);
		expect(Exit.isSuccess(endedExit(toolSpan))).toBe(true);
		expect(result.toolResults.at(0)?.output).toBe('sunny in Paris');
	});

	test('a foreign synchronous tool gets a span', async () => {
		const tracer = makeTestTracer();
		const weather = AiSdk.tool({
			inputSchema: weatherInput,
			execute: () => 'sunny',
		});
		await run(
			generateText({
				model: toolCallModel(),
				prompt: 'hi',
				tools: { weather },
			}),
			tracer,
		);

		expect(
			Exit.isSuccess(
				endedExit(spanNamed(tracer.spans, 'execute_tool weather')),
			),
		).toBe(true);
	});

	test('a foreign tool still runs as a method of the tool', async () => {
		const tracer = makeTestTracer();
		const weather = {
			inputSchema: weatherInput,
			forecast: 'sunny',
			execute(input: { city: string }) {
				return `${this.forecast} in ${input.city}`;
			},
		};
		const result = await run(
			generateText({
				model: toolCallModel(),
				prompt: 'hi',
				tools: { weather },
			}),
			tracer,
		);

		expect(result.toolResults.at(0)?.output).toBe('sunny in Paris');
		expect(
			Exit.isSuccess(
				endedExit(spanNamed(tracer.spans, 'execute_tool weather')),
			),
		).toBe(true);
	});

	test('a class-instance tool keeps its prototype members', async () => {
		const tracer = makeTestTracer();
		class WeatherTool {
			inputSchema = weatherInput;
			unit = 'C';
			execute = async () => ({ temperature: 20 });
			toModelOutput(options: { output: { temperature: number } }) {
				return {
					type: 'text' as const,
					value: `${options.output.temperature}${this.unit}`,
				};
			}
		}
		const result = await run(
			generateText({
				model: toolCallModel(),
				prompt: 'hi',
				tools: { weather: new WeatherTool() },
			}),
			tracer,
		);

		expect(result.toolResults.at(0)?.output).toEqual({ temperature: 20 });
		expect(JSON.stringify(result.response.messages)).toContain('20C');
	});

	test('records a string tool description but not a dynamic one', async () => {
		const tracer = makeTestTracer();
		const described = AiSdk.tool({
			description: 'Weather',
			inputSchema: weatherInput,
			execute: async () => 'sunny',
		});
		const dynamic = AiSdk.tool({
			description: () => 'Weather',
			inputSchema: weatherInput,
			execute: async () => 'sunny',
		});
		for (const weather of [described, dynamic]) {
			await run(
				generateText({
					model: toolCallModel(),
					prompt: 'hi',
					tools: { weather },
				}),
				tracer,
			);
		}

		expect(
			spansNamed(tracer.spans, 'execute_tool weather').map(
				(span) => attributesOf(span)['gen_ai.tool.description'],
			),
		).toEqual(['Weather', undefined]);
	});

	test('a foreign AsyncIterable tool keeps its preliminary results and ends on done', async () => {
		const tracer = makeTestTracer();
		const weather = AiSdk.tool({
			inputSchema: weatherInput,
			async *execute() {
				yield 'partial';
				yield 'sunny';
			},
		});
		const result = await run(
			generateText({
				model: toolCallModel(),
				prompt: 'hi',
				tools: { weather },
			}),
			tracer,
		);

		const toolSpan = spanNamed(tracer.spans, 'execute_tool weather');
		expect(Exit.isSuccess(endedExit(toolSpan))).toBe(true);
		expect(result.toolResults.at(0)?.output).toBe('sunny');
	});

	test('a tool without execute passes through untouched', async () => {
		const tracer = makeTestTracer();
		const weather = AiSdk.tool({ inputSchema: weatherInput });
		await run(
			generateText({
				model: toolCallModel(),
				prompt: 'hi',
				tools: { weather },
			}),
			tracer,
		);

		expect(spansNamed(tracer.spans, 'execute_tool weather')).toEqual([]);
	});

	test('an ai-effect tool failure records its _tag as error.type', async () => {
		class WeatherDown extends Schema.TaggedError<WeatherDown>()(
			'WeatherDown',
			{},
		) {}
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const weather = yield* tool({
					inputSchema: Schema.Struct({ city: Schema.String }),
					execute: () => new WeatherDown(),
				});
				yield* generateText({
					model: toolCallModel(),
					prompt: 'hi',
					tools: { weather },
				});
			}),
			tracer,
		);

		const toolSpan = spanNamed(tracer.spans, 'execute_tool weather');
		expect(Exit.isFailure(endedExit(toolSpan))).toBe(true);
		expect(attributesOf(toolSpan)['error.type']).toBe('WeatherDown');
	});

	test('a foreign tool failure records the Error name and rethrows it unchanged', async () => {
		const tracer = makeTestTracer();
		const boom = new TypeError('boom');
		const weather = AiSdk.tool({
			inputSchema: weatherInput,
			execute: async (): Promise<string> => {
				throw boom;
			},
		});
		const errors: unknown[] = [];
		await run(
			generateText({
				model: toolCallModel(),
				prompt: 'hi',
				tools: { weather },
				onToolExecutionEnd: (event) =>
					Effect.sync(() => {
						if (event.toolOutput.type === 'tool-error') {
							errors.push(event.toolOutput.error);
						}
					}),
			}),
			tracer,
		);

		const toolSpan = spanNamed(tracer.spans, 'execute_tool weather');
		expect(Exit.isFailure(endedExit(toolSpan))).toBe(true);
		expect(attributesOf(toolSpan)['error.type']).toBe('TypeError');
		expect(errors).toEqual([boom]);
	});

	test('a synchronously throwing foreign tool ends its span and rethrows', async () => {
		const tracer = makeTestTracer();
		const boom = new RangeError('sync boom');
		const weather = AiSdk.tool({
			inputSchema: weatherInput,
			execute: (): string => {
				throw boom;
			},
		});
		await run(
			generateText({
				model: toolCallModel(),
				prompt: 'hi',
				tools: { weather },
			}),
			tracer,
		);

		const toolSpan = spanNamed(tracer.spans, 'execute_tool weather');
		expect(attributesOf(toolSpan)['error.type']).toBe('RangeError');
	});

	test('a tool still works outside any wrapper, without a span', async () => {
		const tracer = makeTestTracer();
		const output = await run(
			Effect.gen(function* () {
				const weather = yield* tool({
					inputSchema: Schema.Struct({ city: Schema.String }),
					execute: (input) => Effect.succeed(`sunny in ${input.city}`),
				});
				const execute = weather.execute;
				if (execute === undefined) throw new Error('no execute');
				return yield* Effect.promise(async () =>
					execute(
						{ city: 'Paris' },
						{ toolCallId: 'c', messages: [], context: {} },
					),
				);
			}),
			tracer,
		);

		expect(output).toBe('sunny in Paris');
		expect(spansNamed(tracer.spans, 'execute_tool weather')).toEqual([]);
	});

	test('streamText tool spans parent under the streamText span', async () => {
		const tracer = makeTestTracer();
		await run(
			Effect.gen(function* () {
				const weather = yield* tool({
					inputSchema: Schema.Struct({ city: Schema.String }),
					execute: () => Effect.succeed('sunny'),
				});
				const result = yield* streamText({
					model: streamModel(toolCallChunks('weather')),
					prompt: 'hi',
					tools: { weather },
				});
				yield* Stream.runDrain(result.stream);
			}),
			tracer,
		);

		expect(parentSpanId(spanNamed(tracer.spans, 'execute_tool weather'))).toBe(
			spanNamed(tracer.spans, 'ai-effect.streamText').spanId,
		);
	});
});
