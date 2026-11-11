import * as AiSdk from 'ai';
import {
	Context,
	Deferred,
	Effect,
	Exit,
	Fiber,
	Layer,
	Schema,
	Scope,
} from 'effect';
import { describe, expect, expectTypeOf, test, vi } from 'vitest';
import z from 'zod/v4';
import {
	Ai,
	AiError,
	effectSchema,
	generateText,
	streamText,
	tool,
} from '#src/index.ts';

vi.mock('ai', async (importOriginal) => {
	const actual = await importOriginal<typeof AiSdk>();
	return {
		...actual,
		generateText: vi.fn(),
		streamText: vi.fn(),
	};
});

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
			expect(result.message).toBe('generateText failed');
			expect(result.cause).toBeInstanceOf(Error);
			expect((result.cause as Error).message).toBe('API error');
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

			const inputSchema = myTool.inputSchema as {
				validate: (value: unknown) => unknown;
			};
			expect(inputSchema.validate({ count: '1' })).toEqual({
				success: true,
				value: { count: 1 },
			});
			expect(inputSchema.validate({ count: 1 })).toMatchObject({
				success: false,
			});
			const outputValidator = (
				myTool.outputSchema as {
					validate?: (value: unknown) => unknown;
				}
			).validate;
			if (outputValidator === undefined)
				throw new Error('missing output validator');
			expect(outputValidator({ count: 1 })).toEqual({
				success: true,
				value: { count: 1 },
			});
			expect(outputValidator({ count: 'invalid' })).toMatchObject({
				success: false,
			});

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
