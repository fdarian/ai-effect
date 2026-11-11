import { NoOutputGeneratedError } from 'ai';
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

export const agentSpanName = (id: string | undefined, method: Method) =>
	id === undefined ? `ai-effect.${method}` : `invoke_agent ${id}`;

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
	readonly onStepEnd?: (event: never) => void;
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
	onStepEnd: 'onStepFinish',
	onEnd: 'onFinish',
} as const;

export const START_END_CALLBACK_ALIASES = {
	onStart: 'experimental_onStart',
	onEnd: 'experimental_onEnd',
} as const;

export type StreamEnd = {
	/** The SDK's underlying stream completed on its own. */
	readonly closed: () => void;
	readonly failed: (error: unknown) => void;
};

/**
 * Ends the span from the SDK's lifecycle callbacks and the abort signal, never
 * from a result getter: reading `finishReason` drains the stream and hides
 * the real error behind `NoOutputGeneratedError`.
 *
 * `onError` only records the error. The SDK calls it before deciding to retry
 * and keeps running steps and tools afterwards, so the span ends at `onEnd`,
 * where the error still pending means no retry recovered it. The SDK has paths
 * that end without `onEnd` or `onAbort` (no step recorded); there `streamEnd`,
 * driven by the consumer-facing streams, is the backstop.
 */
export function streamLifecycleCallbacks(
	call: CallSpan,
	options: {
		readonly method: Method;
		/** Combined signal of everything that can abort the call. */
		readonly signal: AbortSignal;
		/** Aborts from these are interrupts; any other abort (user signal, SDK timeout) is the call's failure. */
		readonly interruptSignals: ReadonlyArray<AbortSignal>;
	},
) {
	const aiError = (reason: AiErrorReason) =>
		Exit.fail(new AiError({ method: options.method, reason }));

	const finish = (exit: Exit.Exit<unknown, unknown>) => {
		options.signal.removeEventListener('abort', finishAbortedBySignal);
		call.finish(exit);
	};

	/** The SDK's own timeouts never reach `options.signal`, so `onAbort` passes the reason it saw. */
	function finishAborted(reason: unknown) {
		finish(
			options.interruptSignals.some((signal) => signal.aborted)
				? Exit.interrupt()
				: aiError(new AbortedError({ message: 'aborted', cause: reason })),
		);
	}

	function finishAbortedBySignal() {
		finishAborted(options.signal.reason);
	}

	options.signal.addEventListener('abort', finishAbortedBySignal, {
		once: true,
	});
	if (options.signal.aborted) finishAbortedBySignal();

	const failWith = (error: unknown) =>
		finish(aiError(reasonFromSdkError(error)));

	const text = textCallbacks(call);
	let pendingError: { readonly error: unknown } | undefined;

	const callbacks = {
		onStart: text.onStart,
		onStepEnd: (event: { readonly finishReason: string }) => {
			if (event.finishReason !== 'error') pendingError = undefined;
		},
		onEnd: (event: TextEndEvent) => {
			text.onEnd(event);
			if (pendingError === undefined) finish(Exit.void);
			else failWith(pendingError.error);
		},
		onError: (event: { readonly error: unknown }) => {
			pendingError = event;
		},
		onStepError: (event: { readonly error: unknown }) => failWith(event.error),
		onAbort: (event: { readonly reason?: unknown }) =>
			finishAborted(event.reason),
	};

	const streamEnd: StreamEnd = {
		closed: () =>
			failWith(
				pendingError === undefined
					? new NoOutputGeneratedError({
							message: 'The stream ended without output.',
						})
					: pendingError.error,
			),
		failed: (error) =>
			failWith(pendingError === undefined ? error : pendingError.error),
	};

	return { callbacks, streamEnd };
}
