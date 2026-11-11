import type * as AiSdk from 'ai';
import { Effect, Exit, type Tracer } from 'effect';
import { type CallSpan, makeCallSpan } from '#src/internal/tracing.ts';

type ToolExecute = (
	input: never,
	options: AiSdk.ToolExecutionOptions<never>,
) => unknown;

// `Symbol.for` so the marker survives duplicate copies of this module.
const TOOL_INSTRUMENT = Symbol.for('ai-effect/tool-instrument');

export type ToolInstrument = <A, E, R>(
	effect: Effect.Effect<A, E, R>,
	options: { readonly toolCallId: string },
) => Effect.Effect<A, E, R>;

type InstrumentableExecute = ToolExecute & {
	readonly [TOOL_INSTRUMENT]: (instrument: ToolInstrument) => ToolExecute;
};

/**
 * `tool()` marks its `execute` so call-time wrappers can run it through their
 * own `instrument` using the tool's own `runPromise`, which carries the
 * tool's `R` services.
 */
export const markInstrumentable = <T extends ToolExecute>(
	execute: T,
	bindInstrument: (instrument: ToolInstrument) => ToolExecute,
): T => Object.assign(execute, { [TOOL_INSTRUMENT]: bindInstrument });

/**
 * An agent rebinds `settings.tools` and `prepareCall` may hand them back
 * unchanged (`...rest`), so tools it already rebound must not get a second span.
 */
const reboundExecutes = new WeakSet<ToolExecute>();

const isInstrumentable = (
	execute: ToolExecute,
): execute is InstrumentableExecute => TOOL_INSTRUMENT in execute;

const isAsyncIterable = (value: unknown): value is AsyncIterable<unknown> =>
	typeof value === 'object' && value !== null && Symbol.asyncIterator in value;

const isPromiseLike = (value: unknown): value is PromiseLike<unknown> =>
	typeof value === 'object' &&
	value !== null &&
	'then' in value &&
	typeof value.then === 'function';

async function* endWhenDone(
	iterable: AsyncIterable<unknown>,
	call: CallSpan,
): AsyncGenerator<unknown> {
	try {
		yield* iterable;
	} catch (error) {
		call.finish(Exit.fail(error));
		throw error;
	} finally {
		call.finish(Exit.void);
	}
}

function settleSpan(result: unknown, call: CallSpan): unknown {
	if (isAsyncIterable(result)) return endWhenDone(result, call);
	if (isPromiseLike(result)) {
		return result.then(
			(value) => {
				call.finish(Exit.succeed(value));
				return value;
			},
			(error) => {
				call.finish(Exit.fail(error));
				throw error;
			},
		);
	}
	call.finish(Exit.succeed(result));
	return result;
}

/**
 * Keeps the tool's prototype and every own property (including accessors) so
 * class-instance tools still have their methods, with `this` being the copy.
 */
const withExecute = (tool: AiSdk.Tool, execute: ToolExecute): AiSdk.Tool =>
	Object.create(Object.getPrototypeOf(tool), {
		...Object.getOwnPropertyDescriptors(tool),
		execute: {
			value: execute,
			writable: true,
			enumerable: true,
			configurable: true,
		},
	});

/**
 * Rewrites every executable tool so each call gets an `execute_tool` span
 * parented under `parent`, whether the tool came from `tool()` or from the
 * SDK. The span is created from the services at rebind time, so it uses the
 * same tracer as the wrapper's own span.
 */
export const rebindTools = <TOOLS extends AiSdk.ToolSet>(
	tools: TOOLS | undefined,
	options: {
		readonly parent: Tracer.AnySpan;
		readonly agentName?: string;
	},
): Effect.Effect<TOOLS | undefined> =>
	Effect.gen(function* () {
		if (tools === undefined) return undefined;
		const services = yield* Effect.context<never>();

		const startSpan = (
			name: string,
			tool: AiSdk.Tool,
			toolCallId: string,
		): CallSpan =>
			Effect.runSyncWith(services)(
				makeCallSpan(`execute_tool ${name}`, {
					kind: 'internal',
					parent: options.parent,
					attributes: {
						'gen_ai.operation.name': 'execute_tool',
						'gen_ai.tool.name': name,
						'gen_ai.tool.call.id': toolCallId,
						'gen_ai.tool.type': 'function',
						'gen_ai.tool.description': tool.description,
						'gen_ai.agent.name': options.agentName,
					},
				}),
			);

		const rebind = (name: string, tool: AiSdk.Tool): AiSdk.Tool => {
			const execute = tool.execute as ToolExecute | undefined;
			if (execute === undefined || reboundExecutes.has(execute)) return tool;

			if (isInstrumentable(execute)) {
				const instrumented = execute[TOOL_INSTRUMENT]((effect, callOptions) =>
					Effect.suspend(() => {
						const call = startSpan(name, tool, callOptions.toolCallId);
						return effect.pipe(
							Effect.withParentSpan(call.span),
							Effect.onExit((exit) => Effect.sync(() => call.finish(exit))),
						);
					}),
				);
				reboundExecutes.add(instrumented);
				return withExecute(tool, instrumented);
			}

			const traced: ToolExecute = (input, callOptions) => {
				const call = startSpan(name, tool, callOptions.toolCallId);
				try {
					return settleSpan(execute.call(tool, input, callOptions), call);
				} catch (error) {
					call.finish(Exit.fail(error));
					throw error;
				}
			};
			reboundExecutes.add(traced);
			return withExecute(tool, traced);
		};

		return Object.fromEntries(
			Object.entries(tools).map(([name, tool]) => [name, rebind(name, tool)]),
		) as TOOLS;
	});
