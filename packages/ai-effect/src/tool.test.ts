import * as AiSdk from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import { Context, Deferred, Effect, Fiber, Layer, Schema } from 'effect';
import { describe, expect, expectTypeOf, test } from 'vitest';
import z from 'zod/v4';
import { Ai } from '#src/ai.ts';
import { generateText } from '#src/generate-text.ts';
import { effectSchema } from '#src/schema.ts';
import { tool } from '#src/tool.ts';

const settle = <A>(value: A | PromiseLike<A>) =>
	Effect.promise(async () => value);

describe('tool', () => {
	test('wraps execute with Effect -> Promise bridging', () =>
		Effect.gen(function* () {
			const myTool = yield* tool({
				description: 'test tool',
				inputSchema: Schema.Struct({ city: Schema.String }),
				execute: (input) => Effect.succeed(`Weather in ${input.city}: sunny`),
			});

			expect(myTool.description).toBe('test tool');
			expect(myTool.execute).toBeDefined();

			// biome-ignore lint/style/noNonNullAssertion: asserted above
			const execute = myTool.execute!;
			const result = yield* Effect.promise(() =>
				Promise.resolve(
					execute(
						{ city: 'London' },
						{
							toolCallId: 'test-id',
							messages: [],
							context: {},
						},
					),
				),
			);
			expect(result).toBe('Weather in London: sunny');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('passes through non-callback properties', () =>
		Effect.gen(function* () {
			const myTool = yield* tool({
				description: 'my tool',
				title: 'My Tool',
				inputSchema: { type: 'object' } as unknown as AiSdk.FlexibleSchema<{
					x: number;
				}>,
				strict: true,
			});

			expect(myTool.description).toBe('my tool');
			expect(myTool.title).toBe('My Tool');
			expect(myTool.strict).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('adapts raw Effect schemas and preserves wrapped schemas', () =>
		Effect.gen(function* () {
			const wrappedInput = effectSchema(
				Schema.Struct({ wrapped: Schema.String }),
			);
			const rawOutput = Schema.Struct({ count: Schema.Number });
			const myTool = yield* tool({
				inputSchema: Schema.Struct({ count: Schema.NumberFromString }),
				outputSchema: rawOutput,
			});

			const inputSchema = AiSdk.asSchema(myTool.inputSchema);
			expect(yield* settle(inputSchema.jsonSchema)).toMatchObject({
				type: 'object',
				properties: { count: { type: 'string' } },
				required: ['count'],
			});
			expect(yield* settle(inputSchema.validate?.({ count: '1' }))).toEqual({
				success: true,
				value: { count: 1 },
			});
			const invalidInput = yield* settle(inputSchema.validate?.({ count: 1 }));
			expect(invalidInput?.success).toBe(false);
			if (invalidInput?.success === false)
				expect(AiSdk.TypeValidationError.isInstance(invalidInput.error)).toBe(
					true,
				);

			if (myTool.outputSchema === undefined)
				throw new Error('missing output schema');
			const outputSchema = AiSdk.asSchema(myTool.outputSchema);
			expect(yield* settle(outputSchema.validate?.({ count: 1 }))).toEqual({
				success: true,
				value: { count: 1 },
			});
			expect(
				yield* settle(outputSchema.validate?.({ count: 'invalid' })),
			).toMatchObject({ success: false });

			const wrappedOutput = effectSchema(
				Schema.Struct({ wrapped: Schema.String }),
			);
			const wrappedTool = yield* tool({
				inputSchema: wrappedInput,
				outputSchema: wrappedOutput,
			});
			expect(wrappedTool.inputSchema).toBe(wrappedInput);
			expect(wrappedTool.outputSchema).toBe(wrappedOutput);

			const zodInput = z.object({ label: z.string() });
			const zodOutput = z.object({ ok: z.boolean() });
			const zodTool = yield* tool({
				inputSchema: zodInput,
				outputSchema: zodOutput,
			});
			expect(zodTool.inputSchema).toBe(zodInput);
			expect(zodTool.outputSchema).toBe(zodOutput);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('execute handler can access Effect services', () => {
		class WeatherService extends Context.Service<WeatherService>()(
			'WeatherService',
			{
				make: Effect.succeed({
					getWeather: (city: string) => `${city}: rainy`,
				}),
			},
		) {
			static readonly layer = Layer.effect(this, this.make);
		}

		return Effect.gen(function* () {
			const myTool = yield* tool({
				description: 'weather',
				inputSchema: { type: 'object' } as unknown as AiSdk.FlexibleSchema<{
					city: string;
				}>,
				execute: (input) =>
					Effect.gen(function* () {
						const weather = yield* WeatherService;
						return weather.getWeather(input.city);
					}),
			});

			// biome-ignore lint/style/noNonNullAssertion: test assertion
			const execute = myTool.execute!;
			const result = yield* Effect.promise(() =>
				Promise.resolve(
					execute(
						{ city: 'Paris' },
						{
							toolCallId: 'test-id',
							messages: [],
							context: {},
						},
					),
				),
			);
			expect(result).toBe('Paris: rainy');
		}).pipe(
			Effect.provide(WeatherService.layer),
			Effect.provide(Ai.layer),
			Effect.runPromise,
		);
	});

	test('interrupts execute when the generateText fiber is interrupted, keeping the layer open', () =>
		Effect.gen(function* () {
			const started = yield* Deferred.make<void>();
			const interrupted = yield* Deferred.make<void>();
			const myTool = yield* tool({
				inputSchema: Schema.Struct({}),
				execute: () =>
					Deferred.succeed(started, undefined).pipe(
						Effect.andThen(Effect.never),
						Effect.onInterrupt(() => Deferred.succeed(interrupted, undefined)),
					),
			});
			const model = new MockLanguageModelV4({
				doGenerate: {
					content: [
						{
							type: 'tool-call',
							toolCallId: 'call-1',
							toolName: 'slow',
							input: '{}',
						},
					],
					finishReason: { unified: 'tool-calls', raw: undefined },
					usage: {
						inputTokens: {
							total: 1,
							noCache: 1,
							cacheRead: undefined,
							cacheWrite: undefined,
						},
						outputTokens: { total: 1, text: 1, reasoning: undefined },
					},
					warnings: [],
				},
			});

			const fiber = yield* generateText({
				model,
				prompt: 'test',
				tools: { slow: myTool },
			}).pipe(Effect.forkChild);
			yield* Deferred.await(started);
			expect(yield* Deferred.isDone(interrupted)).toBe(false);

			yield* Fiber.interrupt(fiber);
			yield* Deferred.await(interrupted).pipe(Effect.timeout('1 second'));
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));
});

// These are compile-time-only checks: a tsc-only regression (like OUTPUT
// silently falling back to its default, or a swapped generic position) never
// throws at runtime, so it can't be caught by `expect()` assertions above —
// `check:type` is what would fail if any of these mismatched.
describe('type-level regressions', () => {
	test('tool<INPUT, OUTPUT, R> binds R to Effect requirements, not CONTEXT (0.0.14 call shape)', () => {
		class CommandExecutor extends Context.Service<
			CommandExecutor,
			{ run: (cmd: string) => string }
		>()('CommandExecutor') {}

		const program = tool<{ command: string }, string, CommandExecutor>({
			description: 'runs a command',
			inputSchema: {} as AiSdk.FlexibleSchema<{ command: string }>,
			execute: (input) =>
				Effect.gen(function* () {
					const executor = yield* CommandExecutor;
					return executor.run(input.command);
				}),
		});
		expectTypeOf<Effect.Services<typeof program>>().toEqualTypeOf<
			CommandExecutor | Ai
		>();
	});

	test('tool contextSchema infers CONTEXT instead of defaulting to Record<string, unknown>', () => {
		const program = tool({
			description: 'ctx',
			inputSchema: {} as AiSdk.FlexibleSchema<{ x: number }>,
			contextSchema: {} as AiSdk.FlexibleSchema<{ userId: string }>,
			execute: (_input, options) => Effect.succeed(options.context.userId),
		});
		type ToolType = Effect.Success<typeof program>;
		expectTypeOf<ToolType>().toEqualTypeOf<
			AiSdk.Tool<{ x: number }, string, { userId: string }>
		>();
	});

	test('tool infers OUTPUT from outputSchema instead of defaulting', () => {
		const program = tool({
			description: 'count',
			inputSchema: {} as AiSdk.FlexibleSchema<{ id: string }>,
			outputSchema: {} as AiSdk.FlexibleSchema<{ count: number }>,
		});
		expectTypeOf<
			AiSdk.InferToolOutput<Effect.Success<typeof program>>
		>().toEqualTypeOf<{ count: number }>();
	});

	test('tool infers INPUT and OUTPUT from raw Effect schemas without execute', () => {
		const program = tool({
			inputSchema: Schema.Struct({ query: Schema.String }),
			outputSchema: Schema.Struct({ count: Schema.Number }),
		});
		type ToolType = Effect.Success<typeof program>;
		expectTypeOf<ToolType>().toEqualTypeOf<
			AiSdk.Tool<
				{ readonly query: string },
				{ readonly count: number },
				Record<string, unknown>
			>
		>();
	});

	test('tool preserves explicit OUTPUT inference without an output schema', () => {
		const program = tool<{ readonly query: string }, string>({
			inputSchema: AiSdk.jsonSchema({
				type: 'object',
				properties: { query: { type: 'string' } },
				required: ['query'],
			}),
		});
		type ToolType = Effect.Success<typeof program>;
		expectTypeOf<ToolType>().toEqualTypeOf<
			AiSdk.Tool<{ readonly query: string }, string, Record<string, unknown>>
		>();
	});

	test('tool infers mixed raw Effect and AI schemas', () => {
		const rawInput = tool({
			inputSchema: Schema.Struct({ query: Schema.String }),
			outputSchema: z.object({ count: z.number() }),
		});
		const zodInput = tool({
			inputSchema: z.object({ query: z.string() }),
			outputSchema: Schema.Struct({ count: Schema.Number }),
		});

		expectTypeOf<Effect.Success<typeof rawInput>>().toEqualTypeOf<
			AiSdk.Tool<
				{ readonly query: string },
				{ count: number },
				Record<string, unknown>
			>
		>();
		expectTypeOf<Effect.Success<typeof zodInput>>().toEqualTypeOf<
			AiSdk.Tool<
				{ query: string },
				{ readonly count: number },
				Record<string, unknown>
			>
		>();
	});
});
