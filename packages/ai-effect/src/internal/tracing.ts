import type * as AiSdk from 'ai';
import {
	Cause,
	Clock,
	Effect,
	Exit,
	Option,
	References,
	Result,
	type Tracer,
} from 'effect';
import { AbortedError, AiError, type AiErrorReason } from '#src/ai-error.ts';
import { reasonFromSdkError } from '#src/internal/ai-error-from-sdk.ts';

type Method = AiError['method'];

// Providers resolve strings like `openai.chat` or `google.vertex.chat`. Order
// matters: specific prefixes must precede general ones.
const PROVIDER_PREFIXES: ReadonlyArray<readonly [string, string]> = [
	['google.vertex', 'gcp.vertex_ai'],
	['google.generative-ai', 'gcp.gemini'],
	['google-vertex', 'gcp.vertex_ai'],
	['amazon-bedrock', 'aws.bedrock'],
	['azure-openai', 'azure.ai.openai'],
	['anthropic', 'anthropic'],
	['openai', 'openai'],
	['azure', 'azure.ai.inference'],
	['google', 'gcp.gemini'],
	['mistral', 'mistral'],
	['cohere', 'cohere'],
	['bedrock', 'aws.bedrock'],
	['groq', 'groq'],
	['deepseek', 'deepseek'],
	['perplexity', 'perplexity'],
	['xai', 'x_ai'],
];

export function mapProviderName(provider: string): string {
	const lower = provider.toLowerCase();
	for (const [prefix, mapped] of PROVIDER_PREFIXES) {
		if (
			lower === prefix ||
			lower.startsWith(`${prefix}.`) ||
			lower.startsWith(`${prefix}-`)
		) {
			return mapped;
		}
	}
	return provider;
}

const FINISH_REASONS: Record<string, string> = {
	'tool-calls': 'tool_call',
	'content-filter': 'content_filter',
};

export const mapFinishReason = (finishReason: string): string =>
	FINISH_REASONS[finishReason] ?? finishReason;

export const modelIdOf = (model: string | { readonly modelId: string }) =>
	typeof model === 'string' ? model : model.modelId;

export function errorTypeOf(error: unknown): string | undefined {
	if (error instanceof AiError) return error.reason._tag;
	if (
		typeof error === 'object' &&
		error !== null &&
		'_tag' in error &&
		typeof error._tag === 'string'
	) {
		return error._tag;
	}
	if (error instanceof Error) return error.name;
	return undefined;
}

function failureOf(exit: Exit.Exit<unknown, unknown>): unknown {
	if (!Exit.isFailure(exit)) return undefined;
	const typed = Exit.findErrorOption(exit);
	if (Option.isSome(typed)) return typed.value;
	return Result.getOrUndefined(Cause.findDefect(exit.cause));
}

const definedEntries = (attributes: Record<string, unknown>) =>
	Object.fromEntries(
		Object.entries(attributes).filter(([, value]) => value !== undefined),
	);

export type CallSpan = {
	readonly span: Tracer.Span;
	readonly annotate: (attributes: Record<string, unknown>) => void;
	/** Synchronous and idempotent: the first exit wins. */
	readonly finish: (exit: Exit.Exit<unknown, unknown>) => void;
};

/**
 * Not tied to a scope or fiber, because stream spans outlive the Effect that
 * creates them and end from SDK callbacks.
 */
export const makeCallSpan = (
	name: string,
	options: Tracer.SpanOptionsNoTrace,
): Effect.Effect<CallSpan> =>
	Effect.gen(function* () {
		const span = yield* Effect.makeSpan(name, {
			...options,
			attributes: definedEntries(options.attributes ?? {}),
		});
		const clock = yield* Clock.Clock;
		const timingEnabled = yield* References.TracerTimingEnabled;
		let finished = false;
		return {
			span,
			annotate: (attributes) => {
				for (const [key, value] of Object.entries(definedEntries(attributes))) {
					span.attribute(key, value);
				}
			},
			finish: (exit) => {
				if (finished) return;
				finished = true;
				const errorType = errorTypeOf(failureOf(exit));
				if (errorType !== undefined) span.attribute('error.type', errorType);
				span.end(timingEnabled ? clock.currentTimeNanosUnsafe() : 0n, exit);
			},
		};
	});

/** Runs `use` with the span as the current parent and ends it on the effect's exit. */
export const withCallSpan = <A, E, R>(
	name: string,
	options: Tracer.SpanOptionsNoTrace,
	use: (call: CallSpan) => Effect.Effect<A, E, R>,
): Effect.Effect<A, E, Exclude<R, Tracer.ParentSpan>> =>
	Effect.flatMap(makeCallSpan(name, options), (call) =>
		use(call).pipe(
			Effect.withParentSpan(call.span),
			Effect.onExit((exit) => Effect.sync(() => call.finish(exit))),
		),
	);

/**
 * Like {@link withCallSpan}, but only ends the span when setup fails. On
 * success the span stays open for the stream lifecycle callbacks to end.
 */
export const withStreamSpan = <A, E, R>(
	name: string,
	options: Tracer.SpanOptionsNoTrace,
	use: (call: CallSpan) => Effect.Effect<A, E, R>,
): Effect.Effect<A, E, Exclude<R, Tracer.ParentSpan>> =>
	Effect.flatMap(makeCallSpan(name, options), (call) =>
		use(call).pipe(
			Effect.withParentSpan(call.span),
			Effect.onExit((exit) =>
				Effect.sync(() => {
					if (Exit.isFailure(exit)) call.finish(exit);
				}),
			),
		),
	);

/** Takes the call settings untyped: the wrappers' params are deferred generics TS can't narrow to the known keys. */
export const textRequestAttributes = (
	settings: Readonly<Record<string, unknown>>,
	options: { readonly stream: boolean; readonly agentName?: string },
) =>
	definedEntries({
		'gen_ai.operation.name': 'invoke_agent',
		'gen_ai.agent.name': options.agentName,
		'gen_ai.request.temperature': settings.temperature,
		'gen_ai.request.top_p': settings.topP,
		'gen_ai.request.top_k': settings.topK,
		'gen_ai.request.max_tokens': settings.maxOutputTokens,
		'gen_ai.request.frequency_penalty': settings.frequencyPenalty,
		'gen_ai.request.presence_penalty': settings.presencePenalty,
		'gen_ai.request.stop_sequences': settings.stopSequences,
		'gen_ai.request.seed': settings.seed,
		'gen_ai.request.stream': options.stream ? true : undefined,
	});

type StartEvent = { readonly provider: string; readonly modelId: string };

const startAttributes = (event: StartEvent) => ({
	'gen_ai.provider.name': mapProviderName(event.provider),
	'gen_ai.request.model': event.modelId,
});

type UsageCount = number | undefined;
type TextEndEvent = {
	readonly finishReason: string;
	readonly totalUsage: {
		readonly inputTokens: UsageCount;
		readonly outputTokens: UsageCount;
		readonly inputTokenDetails: {
			readonly cacheReadTokens: UsageCount;
			readonly cacheWriteTokens: UsageCount;
		};
		readonly outputTokenDetails: { readonly reasoningTokens: UsageCount };
	};
	readonly finalStep: {
		readonly response: { readonly id: string; readonly modelId: string };
	};
};

const positive = (count: UsageCount) =>
	count !== undefined && count > 0 ? count : undefined;

const textEndAttributes = (event: TextEndEvent) => ({
	'gen_ai.response.id': event.finalStep.response.id,
	'gen_ai.response.model': event.finalStep.response.modelId,
	'gen_ai.response.finish_reasons': [mapFinishReason(event.finishReason)],
	'gen_ai.usage.input_tokens': event.totalUsage.inputTokens,
	'gen_ai.usage.output_tokens': event.totalUsage.outputTokens,
	'gen_ai.usage.cache_read.input_tokens': positive(
		event.totalUsage.inputTokenDetails.cacheReadTokens,
	),
	'gen_ai.usage.cache_write.input_tokens': positive(
		event.totalUsage.inputTokenDetails.cacheWriteTokens,
	),
	'gen_ai.usage.reasoning.output_tokens': positive(
		event.totalUsage.outputTokenDetails.reasoningTokens,
	),
});

type EmbeddingUsage = { readonly tokens: number };

// The SDK reports `NaN` when the provider doesn't return usage.
const finiteTokens = (usage: EmbeddingUsage) =>
	Number.isNaN(usage.tokens) ? undefined : usage.tokens;

export type TracingCallbacks = {
	readonly onStart?: (event: never) => void;
	readonly onEnd?: (event: never) => void;
	readonly onError?: (event: never) => void;
	readonly onStepError?: (event: never) => void;
	readonly onAbort?: (event: never) => void;
};

export const textCallbacks = (call: CallSpan) => ({
	onStart: (event: StartEvent) => call.annotate(startAttributes(event)),
	onEnd: (event: TextEndEvent) => call.annotate(textEndAttributes(event)),
});

export const embedCallbacks = (call: CallSpan) => ({
	onStart: (event: StartEvent) => call.annotate(startAttributes(event)),
	onEnd: (event: {
		readonly usage: EmbeddingUsage;
		readonly embedding: ReadonlyArray<number>;
	}) =>
		call.annotate({
			'gen_ai.usage.input_tokens': finiteTokens(event.usage),
			'gen_ai.embeddings.dimension.count': event.embedding.length,
		}),
});

export const embedManyCallbacks = (call: CallSpan) => ({
	onStart: (event: StartEvent) => call.annotate(startAttributes(event)),
	onEnd: (event: {
		readonly usage: EmbeddingUsage;
		readonly embedding: ReadonlyArray<ReadonlyArray<number>>;
	}) =>
		call.annotate({
			'gen_ai.usage.input_tokens': finiteTokens(event.usage),
			'gen_ai.embeddings.dimension.count': event.embedding[0]?.length,
		}),
});

export const rerankCallbacks = (call: CallSpan) => ({
	onStart: (event: StartEvent) => call.annotate(startAttributes(event)),
	onEnd: (event: {
		readonly response: { readonly id?: string; readonly modelId: string };
	}) =>
		call.annotate({
			'gen_ai.response.id': event.response.id,
			'gen_ai.response.model': event.response.modelId,
		}),
});

/**
 * The SDK resolves aliased callbacks as `canonical ?? alias`, so the composed
 * object carries the tracing wrapper under the canonical key and clears the
 * alias; otherwise a user's alias-only callback would be shadowed.
 *
 * Tracing callbacks run first, synchronously and outside the `Ai` fiber set,
 * so they still fire after the layer closes.
 */
export function composeCallbacks(
	wrapped: Record<string, unknown>,
	tracing: TracingCallbacks,
	aliases: Readonly<Record<string, string>>,
	fallbacks: Readonly<Record<string, (event: never) => unknown>> = {},
): Record<string, unknown> {
	const composed: Record<string, unknown> = {};
	for (const [key, trace] of Object.entries(tracing)) {
		const alias = aliases[key];
		const user =
			wrapped[key] ??
			(alias === undefined ? undefined : wrapped[alias]) ??
			fallbacks[key];
		composed[key] = (event: never) => {
			trace(event);
			return typeof user === 'function' ? user(event) : undefined;
		};
		if (alias !== undefined) composed[alias] = undefined;
	}
	return composed;
}

export const TEXT_CALLBACK_ALIASES = {
	onStart: 'experimental_onStart',
	onEnd: 'onFinish',
} as const;

export const START_END_CALLBACK_ALIASES = {
	onStart: 'experimental_onStart',
	onEnd: 'experimental_onEnd',
} as const;

/**
 * Ends the span from the SDK's lifecycle callbacks and the abort signal, never
 * from a result getter: reading `finishReason` drains the stream and hides
 * the real error behind `NoOutputGeneratedError`.
 */
export function streamLifecycleCallbacks(
	call: CallSpan,
	options: {
		readonly method: Method;
		readonly signal: AbortSignal;
		readonly userSignal: AbortSignal | undefined;
	},
) {
	const aiError = (reason: AiErrorReason) =>
		Exit.fail(new AiError({ method: options.method, reason }));

	const finish = (exit: Exit.Exit<unknown, unknown>) => {
		options.signal.removeEventListener('abort', finishAborted);
		call.finish(exit);
	};

	function finishAborted() {
		const userSignal = options.userSignal;
		finish(
			userSignal?.aborted
				? aiError(
						new AbortedError({ message: 'aborted', cause: userSignal.reason }),
					)
				: Exit.interrupt(),
		);
	}

	options.signal.addEventListener('abort', finishAborted, { once: true });
	if (options.signal.aborted) finishAborted();

	const failWith = (event: { readonly error: unknown }) =>
		finish(aiError(reasonFromSdkError(event.error)));

	const text = textCallbacks(call);

	return {
		onStart: text.onStart,
		onEnd: (event: TextEndEvent) => {
			text.onEnd(event);
			finish(Exit.void);
		},
		onError: failWith,
		onStepError: failWith,
		onAbort: finishAborted,
	};
}

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
			if (execute === undefined) return tool;

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
				return { ...tool, execute: instrumented } as AiSdk.Tool;
			}

			const traced: ToolExecute = (input, callOptions) => {
				const call = startSpan(name, tool, callOptions.toolCallId);
				try {
					return settleSpan(execute(input, callOptions), call);
				} catch (error) {
					call.finish(Exit.fail(error));
					throw error;
				}
			};
			return { ...tool, execute: traced } as AiSdk.Tool;
		};

		return Object.fromEntries(
			Object.entries(tools).map(([name, tool]) => [name, rebind(name, tool)]),
		) as TOOLS;
	});
