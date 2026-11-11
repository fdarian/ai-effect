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
import { describe, expect, expectTypeOf, test } from 'vitest';
import z from 'zod/v4';
import { Ai } from '#src/ai.ts';
import { AiError, UnknownError } from '#src/ai-error.ts';
import { effectSchema } from '#src/schema.ts';
import { tool } from '#src/tool.ts';
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

const textResult = (text: string) => ({
	content: [{ type: 'text' as const, text }],
	finishReason: { unified: 'stop' as const, raw: undefined },
	usage,
	warnings: [],
});

const toolCallResult = (toolName: string) => ({
	content: [
		{
			type: 'tool-call' as const,
			toolCallId: 'call-1',
			toolName,
			input: '{}',
		},
	],
	finishReason: { unified: 'tool-calls' as const, raw: undefined },
	usage,
	warnings: [],
});

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
	new MockLanguageModelV4({
		doGenerate: textResult(text),
		doStream: textStream(text),
	});

/** Model whose first call requests `toolName` and whose second one answers. */
const toolLoopModel = (toolName: string) => {
	let calls = 0;
	return new MockLanguageModelV4({
		doGenerate: async () =>
			calls++ === 0 ? toolCallResult(toolName) : textResult('done'),
	});
};

class Dep extends Context.Service<Dep, { value: string }>()('Dep') {
	static readonly layer = Layer.succeed(Dep, { value: 'dep' });
}
class CallDep extends Context.Service<CallDep, { value: string }>()('CallDep') {
	static readonly layer = Layer.succeed(CallDep, { value: 'call-dep' });
}

describe('ToolLoopAgent.make', () => {
	test('exposes id and tools', () =>
		Effect.gen(function* () {
			const slow = yield* tool({ inputSchema: Schema.Struct({}) });
			const agent = yield* ToolLoopAgent.make({
				id: 'my-agent',
				model: textModel('hi'),
				tools: { slow },
			});

			expect(agent.id).toBe('my-agent');
			expect(agent.tools.slow).toBe(slow);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('generate returns the result', () =>
		Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({ model: textModel('hello') });
			const result = yield* agent.generate({ prompt: 'test' });

			expect(result.text).toBe('hello');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('stream returns the result', () =>
		Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({ model: textModel('hello') });
			const result = yield* agent.stream({ prompt: 'test' });

			expect(yield* result.text).toBe('hello');
			expect(
				yield* Effect.promise(() => Promise.resolve(result.raw.text)),
			).toBe('hello');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('stream textStream collects the text through an Effect Stream', () =>
		Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({ model: textModel('hello') });
			const result = yield* agent.stream({ prompt: 'test' });

			expect(yield* Stream.runCollect(result.textStream)).toEqual(['hello']);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('stream toUIMessageStream emits UI message chunks', () =>
		Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({ model: textModel('hello') });
			const result = yield* agent.stream({ prompt: 'test' });

			const chunks = yield* Stream.runCollect(result.toUIMessageStream());
			expect(chunks.map((chunk) => chunk.type)).toContain('text-delta');
			expect(chunks.at(-1)?.type).toBe('finish');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('accepts messages instead of prompt', () =>
		Effect.gen(function* () {
			const model = textModel('hello');
			const agent = yield* ToolLoopAgent.make({ model });
			yield* agent.generate({
				messages: [{ role: 'user', content: 'from messages' }],
			});

			expect(JSON.stringify(model.doGenerateCalls[0]?.prompt)).toContain(
				'from messages',
			);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('runs constructor-level and call-level callbacks with their own services', () => {
		const seen: string[] = [];

		return Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({
				model: textModel('hello'),
				onEnd: () =>
					Effect.gen(function* () {
						const dep = yield* Dep;
						seen.push(`settings:${dep.value}`);
					}),
			}).pipe(Effect.provide(Dep.layer));

			yield* agent
				.generate({
					prompt: 'test',
					onEnd: () =>
						Effect.gen(function* () {
							const dep = yield* CallDep;
							seen.push(`call:${dep.value}`);
						}),
				})
				.pipe(Effect.provide(CallDep.layer));

			expect(seen.sort()).toEqual(['call:call-dep', 'settings:dep']);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('runs constructor-level and call-level callbacks for stream', () => {
		const seen: string[] = [];

		return Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({
				model: textModel('hello'),
				onStepEnd: () => Effect.sync(() => seen.push('settings')),
			});

			const result = yield* agent.stream({
				prompt: 'test',
				onStepEnd: () => Effect.sync(() => seen.push('call')),
			});
			yield* result.text;

			expect(seen.sort()).toEqual(['call', 'settings']);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('settings-only onError callback receives stream errors', () => {
		const errors: unknown[] = [];

		return Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({
				model: new MockLanguageModelV4({
					doStream: async () => {
						throw new Error('stream failed');
					},
				}),
				onError: (event) => Effect.sync(() => errors.push(event.error)),
			});

			const result = yield* agent.stream({ prompt: 'test' });
			yield* Effect.promise(() => Promise.resolve(result.raw.consumeStream()));

			expect(errors).toHaveLength(1);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise);
	});

	test('executes a tool built with this library inside the loop', () =>
		Effect.gen(function* () {
			const executed: unknown[] = [];
			const myTool = yield* tool({
				inputSchema: Schema.Struct({}),
				execute: () =>
					Effect.gen(function* () {
						const dep = yield* Dep;
						executed.push(dep.value);
						return 'tool-output';
					}),
			}).pipe(Effect.provide(Dep.layer));
			const agent = yield* ToolLoopAgent.make({
				model: toolLoopModel('lookup'),
				tools: { lookup: myTool },
			});

			const result = yield* agent.generate({ prompt: 'test' });

			expect(executed).toEqual(['dep']);
			expect(result.steps).toHaveLength(2);
			expect(result.steps[0]?.toolResults[0]?.output).toBe('tool-output');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));
});

describe('ToolLoopAgent.make as a service', () => {
	const seen: string[] = [];

	class WeatherAgent extends Context.Service<WeatherAgent>()(
		'test/WeatherAgent',
		{
			make: Effect.gen(function* () {
				const lookup = yield* tool({
					inputSchema: Schema.Struct({}),
					execute: () =>
						Effect.gen(function* () {
							const callDep = yield* CallDep;
							return callDep.value;
						}),
				});

				return yield* ToolLoopAgent.make({
					model: toolLoopModel('lookup'),
					tools: { lookup },
					onStart: () =>
						Effect.gen(function* () {
							const dep = yield* Dep;
							seen.push(dep.value);
						}),
				});
			}),
		},
	) {
		static readonly layer = Layer.effect(WeatherAgent, WeatherAgent.make);
	}

	test('layer requires Ai, the callback services and the tool services', () => {
		expectTypeOf<Layer.Services<typeof WeatherAgent.layer>>().toEqualTypeOf<
			Ai | Dep | CallDep
		>();
	});

	test('yielding the service gives a working agent', () =>
		Effect.gen(function* () {
			const agent = yield* WeatherAgent;
			const result = yield* agent.generate({ prompt: 'test' });

			expect(seen).toEqual(['dep']);
			expect(result.steps[0]?.toolResults[0]?.output).toBe('call-dep');
		}).pipe(
			Effect.provide(WeatherAgent.layer),
			Effect.provide(Layer.mergeAll(Ai.layer, Dep.layer, CallDep.layer)),
			Effect.runPromise,
		));
});

describe('ToolLoopAgent.make errors', () => {
	test('generate surfaces SDK failures as AiError', () =>
		Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({
				model: new MockLanguageModelV4({
					doGenerate: async () => {
						throw new Error('API error');
					},
				}),
				maxRetries: 0,
			});

			const error = yield* agent.generate({ prompt: 'test' }).pipe(Effect.flip);

			expect(error).toBeInstanceOf(AiError);
			expect(error.method).toBe('ToolLoopAgent.generate');
			expect(error.reason).toBeInstanceOf(UnknownError);
			expect((error.reason.cause as Error).message).toBe('API error');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('stream surfaces SDK failures as AiError', () =>
		Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({
				model: textModel('hello'),
				prepareCall: () => {
					throw new Error('prepare failed');
				},
			});

			const error = yield* agent.stream({ prompt: 'test' }).pipe(Effect.flip);

			expect(error).toBeInstanceOf(AiError);
			expect(error.method).toBe('ToolLoopAgent.stream');
			expect(error.reason).toBeInstanceOf(UnknownError);
			expect((error.reason.cause as Error).message).toBe('prepare failed');
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('a stream error fails textStream and text with AiError', () =>
		Effect.gen(function* () {
			const boom = new Error('boom');
			const agent = yield* ToolLoopAgent.make({
				model: new MockLanguageModelV4({
					doStream: async () => ({
						stream: new ReadableStream({
							start(controller) {
								controller.error(boom);
							},
						}),
					}),
				}),
				maxRetries: 0,
			});
			const result = yield* agent.stream({ prompt: 'test' });

			const streamError = yield* Stream.runCollect(result.textStream).pipe(
				Effect.flip,
			);
			expect(streamError).toBeInstanceOf(AiError);
			expect(streamError.method).toBe('ToolLoopAgent.stream');
			expect(streamError.reason.cause).toBe(boom);

			const textError = yield* result.text.pipe(Effect.flip);
			expect(textError).toBeInstanceOf(AiError);
			expect(textError.method).toBe('ToolLoopAgent.stream');
			expect(textError.reason.cause).toBe(boom);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('invalid call options fail with AiError', () =>
		Effect.gen(function* () {
			const agent = yield* ToolLoopAgent.make({
				model: textModel('hello'),
				callOptionsSchema: effectSchema(
					Schema.Struct({ userId: Schema.String }),
				),
			});

			const error = yield* agent
				.generate({ prompt: 'test', options: { userId: 1 } as never })
				.pipe(Effect.flip);

			expect(error).toBeInstanceOf(AiError);
			expect(error.method).toBe('ToolLoopAgent.generate');
			expect(AiSdk.TypeValidationError.isInstance(error.reason.cause)).toBe(
				true,
			);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));
});

describe('ToolLoopAgent.make interruption', () => {
	const hangingTool = Effect.gen(function* () {
		const started = yield* Deferred.make<void>();
		const interrupted = yield* Deferred.make<void>();
		const slow = yield* tool({
			inputSchema: Schema.Struct({}),
			execute: () =>
				Deferred.succeed(started, undefined).pipe(
					Effect.andThen(Effect.never),
					Effect.onInterrupt(() => Deferred.succeed(interrupted, undefined)),
				),
		});
		return { slow, started, interrupted };
	});

	test('interrupting generate aborts the call and interrupts a running tool execute', () =>
		Effect.gen(function* () {
			const toolSetup = yield* hangingTool;
			const model = toolLoopModel('slow');
			const agent = yield* ToolLoopAgent.make({
				model,
				tools: { slow: toolSetup.slow },
			});

			const fiber = yield* agent
				.generate({ prompt: 'test' })
				.pipe(Effect.forkChild);
			yield* Deferred.await(toolSetup.started);
			expect(yield* Deferred.isDone(toolSetup.interrupted)).toBe(false);

			yield* Fiber.interrupt(fiber);
			yield* Deferred.await(toolSetup.interrupted).pipe(
				Effect.timeout('1 second'),
			);
			expect(model.doGenerateCalls[0]?.abortSignal?.aborted).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('honours a call-level abortSignal', () =>
		Effect.gen(function* () {
			const toolSetup = yield* hangingTool;
			const agent = yield* ToolLoopAgent.make({
				model: toolLoopModel('slow'),
				tools: { slow: toolSetup.slow },
			});
			const controller = new AbortController();

			const fiber = yield* agent
				.generate({ prompt: 'test', abortSignal: controller.signal })
				.pipe(Effect.forkChild);
			yield* Deferred.await(toolSetup.started);

			controller.abort();
			yield* Deferred.await(toolSetup.interrupted).pipe(
				Effect.timeout('1 second'),
			);
			const exit = yield* Fiber.await(fiber);
			expect(Exit.isFailure(exit)).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('closing the Ai layer aborts an in-flight stream', () =>
		Effect.gen(function* () {
			const reached = yield* Deferred.make<AbortSignal>();
			const model = new MockLanguageModelV4({
				doStream: async (options) => {
					Effect.runSync(
						Deferred.succeed(reached, options.abortSignal as AbortSignal),
					);
					return { stream: new ReadableStream() };
				},
			});
			const layerScope = yield* Scope.make();
			const context = yield* Layer.buildWithScope(Ai.layer, layerScope);

			const agent = yield* ToolLoopAgent.make({ model }).pipe(
				Effect.provideContext(context),
			);
			yield* agent
				.stream({ prompt: 'test' })
				.pipe(Effect.provideContext(context));
			const signal = yield* Deferred.await(reached).pipe(
				Effect.timeout('1 second'),
			);

			expect(signal.aborted).toBe(false);
			yield* Scope.close(layerScope, Exit.void);
			expect(signal.aborted).toBe(true);
		}).pipe(Effect.runPromise));

	test('interrupting a textStream consumer aborts the call', () =>
		Effect.gen(function* () {
			const reached = yield* Deferred.make<AbortSignal>();
			const gotChunk = yield* Deferred.make<void>();
			const model = new MockLanguageModelV4({
				doStream: async (options) => {
					Effect.runSync(
						Deferred.succeed(reached, options.abortSignal as AbortSignal),
					);
					return {
						stream: new ReadableStream({
							start(controller) {
								controller.enqueue({ type: 'stream-start', warnings: [] });
								controller.enqueue({ type: 'text-start', id: 't' });
								controller.enqueue({
									type: 'text-delta',
									id: 't',
									delta: 'hi',
								});
							},
						}),
					};
				},
			});
			const agent = yield* ToolLoopAgent.make({ model });
			const result = yield* agent.stream({ prompt: 'test' });

			const fiber = yield* Stream.runDrain(
				result.textStream.pipe(
					Stream.tap(() => Deferred.succeed(gotChunk, undefined)),
				),
			).pipe(Effect.forkChild);
			yield* Deferred.await(gotChunk).pipe(Effect.timeout('1 second'));
			const signal = yield* Deferred.await(reached);
			expect(signal.aborted).toBe(false);

			yield* Fiber.interrupt(fiber);
			expect(signal.aborted).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('interrupting stream before the SDK call returns aborts the call', () =>
		Effect.gen(function* () {
			const preparing = yield* Deferred.make<void>();
			const release = yield* Deferred.make<void>();
			const reached = yield* Deferred.make<AbortSignal>();
			const model = new MockLanguageModelV4({
				doStream: async (options) => {
					Effect.runSync(
						Deferred.succeed(reached, options.abortSignal as AbortSignal),
					);
					return { stream: new ReadableStream() };
				},
			});
			const agent = yield* ToolLoopAgent.make({
				model,
				prepareCall: async (options) => {
					Effect.runSync(Deferred.succeed(preparing, undefined));
					await Effect.runPromise(Deferred.await(release));
					return options;
				},
			});

			const fiber = yield* agent
				.stream({ prompt: 'test' })
				.pipe(Effect.forkChild);
			yield* Deferred.await(preparing).pipe(Effect.timeout('1 second'));

			yield* Fiber.interrupt(fiber);
			yield* Deferred.succeed(release, undefined);
			const signal = yield* Deferred.await(reached).pipe(
				Effect.timeout('1 second'),
			);

			expect(signal.aborted).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));

	test('honours a call-level abortSignal for stream', () =>
		Effect.gen(function* () {
			const reached = yield* Deferred.make<AbortSignal>();
			const model = new MockLanguageModelV4({
				doStream: async (options) => {
					Effect.runSync(
						Deferred.succeed(reached, options.abortSignal as AbortSignal),
					);
					return { stream: new ReadableStream() };
				},
			});
			const agent = yield* ToolLoopAgent.make({ model });
			const controller = new AbortController();

			yield* agent.stream({ prompt: 'test', abortSignal: controller.signal });
			const signal = yield* Deferred.await(reached).pipe(
				Effect.timeout('1 second'),
			);

			expect(signal.aborted).toBe(false);
			controller.abort();
			expect(signal.aborted).toBe(true);
		}).pipe(Effect.provide(Ai.layer), Effect.runPromise));
});

// Compile-time-only checks, see the same section in generate-text.test.ts.
// The agents here are fakes, so each body is wrapped in `typeOnly` and never
// runs.
const typeOnly = (_check: () => void) => {};

describe('type-level regressions', () => {
	const model = {} as AiSdk.LanguageModel;
	const UserSchema = Schema.Struct({ userId: Schema.String });

	test('construction requires Ai | its callbacks R, calls require Ai | their callbacks R', () => {
		const construction = ToolLoopAgent.make({
			model,
			onEnd: () =>
				Effect.gen(function* () {
					yield* Dep;
				}),
		});
		expectTypeOf<Effect.Services<typeof construction>>().toEqualTypeOf<
			Ai | Dep
		>();

		const bare = ToolLoopAgent.make({ model });
		expectTypeOf<Effect.Services<typeof bare>>().toEqualTypeOf<Ai>();

		typeOnly(() => {
			const agent = {} as ToolLoopAgent;
			const generate = agent.generate({
				prompt: 'hi',
				onEnd: () =>
					Effect.gen(function* () {
						yield* CallDep;
					}),
			});
			const stream = agent.stream({
				prompt: 'hi',
				onStepEnd: () =>
					Effect.gen(function* () {
						yield* CallDep;
					}),
			});
			expectTypeOf<Effect.Services<typeof generate>>().toEqualTypeOf<
				Ai | CallDep
			>();
			expectTypeOf<Effect.Services<typeof stream>>().toEqualTypeOf<
				Ai | CallDep
			>();

			const bareGenerate = agent.generate({ prompt: 'hi' });
			const bareStream = agent.stream({ prompt: 'hi' });
			expectTypeOf<Effect.Services<typeof bareGenerate>>().toEqualTypeOf<Ai>();
			expectTypeOf<Effect.Services<typeof bareStream>>().toEqualTypeOf<Ai>();
			expectTypeOf<
				Effect.Error<typeof bareGenerate>
			>().toEqualTypeOf<AiError>();
		});
	});

	test('infers OUTPUT from `output` into the result', () => {
		const program = ToolLoopAgent.make({
			model,
			output: AiSdk.Output.object({ schema: effectSchema(UserSchema) }),
		});
		type Agent = Effect.Success<typeof program>;
		type Result = Effect.Success<ReturnType<Agent['generate']>>;
		expectTypeOf<Result['output']>().toEqualTypeOf<{
			readonly userId: string;
		}>();
	});

	test('stream returns the Effect-typed result', () => {
		typeOnly(() => {
			const agent = {} as ToolLoopAgent;
			const program = agent.stream({ prompt: 'hi' });
			type Result = Effect.Success<typeof program>;
			expectTypeOf<Result['text']>().toEqualTypeOf<
				Effect.Effect<string, AiError>
			>();
			expectTypeOf<Result['textStream']>().toEqualTypeOf<
				Stream.Stream<string, AiError>
			>();
		});
	});

	test('infers TOOLS into `tools` and into the result', () => {
		const aTool = AiSdk.tool({
			inputSchema: z.object({ x: z.number() }),
			execute: async () => 'out',
		});
		const program = ToolLoopAgent.make({ model, tools: { aTool } });
		type Agent = Effect.Success<typeof program>;
		expectTypeOf<Agent['tools']>().toEqualTypeOf<{ aTool: typeof aTool }>();
		type Result = Effect.Success<ReturnType<Agent['generate']>>;
		expectTypeOf<
			Result['staticToolResults'][number]['toolName']
		>().toEqualTypeOf<'aTool'>();
	});

	test('infers RUNTIME_CONTEXT from `runtimeContext` into callbacks', () => {
		ToolLoopAgent.make({
			model,
			runtimeContext: { tenant: 'acme' },
			onEnd: (event) => {
				expectTypeOf(event.runtimeContext).toEqualTypeOf<{ tenant: string }>();
				return Effect.void;
			},
		});
	});

	test('callOptionsSchema makes `options` required and typed', () => {
		const program = ToolLoopAgent.make({
			model,
			callOptionsSchema: effectSchema(UserSchema),
		});
		type Agent = Effect.Success<typeof program>;

		typeOnly(() => {
			const agent = {} as Agent;

			agent.generate({ prompt: 'hi', options: { userId: 'u1' } });
			agent.stream({ prompt: 'hi', options: { userId: 'u1' } });
			// @ts-expect-error `options` is required once `callOptionsSchema` is given
			agent.generate({ prompt: 'hi' });
			// @ts-expect-error `options` is required once `callOptionsSchema` is given
			agent.stream({ prompt: 'hi' });
			// @ts-expect-error `options` must match the schema
			agent.generate({ prompt: 'hi', options: { userId: 1 } });
		});
	});

	test('`options` is rejected without a callOptionsSchema', () => {
		typeOnly(() => {
			const agent = {} as ToolLoopAgent;

			// @ts-expect-error no `callOptionsSchema`, so no `options`
			agent.generate({ prompt: 'hi', options: { userId: 'u1' } });
			// @ts-expect-error no `callOptionsSchema`, so no `options`
			agent.stream({ prompt: 'hi', options: { userId: 'u1' } });
		});
	});

	test('keeps prompt/messages exclusivity', () => {
		typeOnly(() => {
			const agent = {} as ToolLoopAgent;

			agent.generate({ prompt: 'hi' });
			agent.generate({ messages: [{ role: 'user', content: 'hi' }] });
			agent.stream({ messages: [{ role: 'user', content: 'hi' }] });
			// @ts-expect-error `prompt` and `messages` are mutually exclusive
			agent.generate({ prompt: 'hi', messages: [] });
			// @ts-expect-error `prompt` and `messages` are mutually exclusive
			agent.stream({ prompt: 'hi', messages: [] });
			// @ts-expect-error one of `prompt` / `messages` is required
			agent.generate({});
		});
	});

	test('settings-only callbacks are not accepted per call', () => {
		typeOnly(() => {
			const agent = {} as ToolLoopAgent;

			// @ts-expect-error `onError` is only available on the settings
			agent.stream({ prompt: 'hi', onError: () => Effect.void });
		});
	});
});
