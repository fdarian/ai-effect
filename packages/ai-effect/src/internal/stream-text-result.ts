import type * as AiSdk from 'ai';
import { Effect, Exit, Stream } from 'effect';
import { AiError, type AiErrorReason, UnknownError } from '#src/ai-error.ts';
import { reasonFromSdkError } from '#src/internal/ai-error-from-sdk.ts';

type RawStreamTextResult<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = ReturnType<typeof AiSdk.streamText<TOOLS, RUNTIME_CONTEXT, OUTPUT>>;

const STREAM_TEXT_EFFECT_KEYS = [
	'content',
	'text',
	'files',
	'sources',
	'toolCalls',
	'staticToolCalls',
	'dynamicToolCalls',
	'staticToolResults',
	'dynamicToolResults',
	'toolResults',
	'finishReason',
	'rawFinishReason',
	'usage',
	'warnings',
	'steps',
	'finalStep',
	'responseMessages',
	'output',
] as const;
type StreamTextEffectKey = (typeof STREAM_TEXT_EFFECT_KEYS)[number];

type ElementOf<S> = S extends AiSdk.AsyncIterableStream<infer T> ? T : never;

type StreamTextEffects<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = {
	readonly [K in StreamTextEffectKey]: Effect.Effect<
		Awaited<RawStreamTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT>[K]>,
		AiError
	>;
};

/**
 * Effect-typed view of the SDK's `StreamTextResult`. Interrupting a fiber that
 * consumes any stream or effect here aborts the underlying call; finishing,
 * failing or stopping early (e.g. `Stream.take`) does not.
 */
export interface StreamTextResult<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> extends StreamTextEffects<TOOLS, RUNTIME_CONTEXT, OUTPUT> {
	/** The SDK result, for everything that isn't wrapped (response helpers, `consumeStream`, deprecated fields). */
	readonly raw: RawStreamTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT>;
	/** Error and abort parts are emitted as data, not failures. */
	readonly stream: Stream.Stream<AiSdk.TextStreamPart<TOOLS>, AiError>;
	/** @deprecated Use `stream` instead. */
	readonly fullStream: Stream.Stream<AiSdk.TextStreamPart<TOOLS>, AiError>;
	/** Fails on an error or abort part, which the SDK drops from this view. */
	readonly textStream: Stream.Stream<string, AiError>;
	/** Fails on an error or abort part, which the SDK drops from this view. */
	readonly partialOutputStream: Stream.Stream<
		ElementOf<
			RawStreamTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT>['partialOutputStream']
		>,
		AiError
	>;
	/** Fails on an error or abort part, which the SDK drops from this view. */
	readonly elementStream: Stream.Stream<
		ElementOf<
			RawStreamTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT>['elementStream']
		>,
		AiError
	>;
}

type Method = Extract<AiError['method'], 'streamText' | 'ToolLoopAgent.stream'>;

const abortReason = (cause: unknown) =>
	new UnknownError({ message: 'aborted', cause });

// The getters tee the base stream (and the output ones can throw), so read
// them once per run, not when the result is built.
const streamFromGetter = <A>(
	getter: () => AsyncIterable<A>,
	failed: (cause: unknown) => AiError,
) =>
	Stream.unwrap(
		Effect.try({ try: getter, catch: failed }).pipe(
			Effect.map((iterable) => Stream.fromAsyncIterable(iterable, failed)),
		),
	);

export function makeStreamTextResult<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
>(
	method: Method,
	raw: RawStreamTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT>,
	controller: AbortController,
): StreamTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT> {
	const toAiError = (reason: AiErrorReason) => new AiError({ method, reason });
	const failed = (cause: unknown) => toAiError(reasonFromSdkError(cause));
	const abort = Effect.sync(() => controller.abort());

	// `Stream.take` and friends close the upstream with a success Exit, so only
	// a real interruption reaches the abort.
	const abortOnInterrupt = <A>(stream: Stream.Stream<A, AiError>) =>
		Stream.onExit(stream, (exit) =>
			Exit.hasInterrupts(exit) ? abort : Effect.void,
		);

	const partStream = () =>
		abortOnInterrupt(streamFromGetter(() => raw.stream, failed));

	// textStream, partialOutputStream and elementStream silently drop these
	// parts, so a second pass over a fresh tee surfaces them as failures.
	const failOnErrorOrAbortPart = streamFromGetter(
		() => raw.stream,
		failed,
	).pipe(
		Stream.mapEffect((part) => {
			if (part.type === 'error') {
				return Effect.fail(failed(part.error));
			}
			if (part.type === 'abort') {
				return Effect.fail(toAiError(abortReason(part.reason)));
			}
			return Effect.void;
		}),
		Stream.drain,
	);

	const view = <A>(getter: () => AsyncIterable<A>) =>
		abortOnInterrupt(
			streamFromGetter(getter, failed).pipe(
				Stream.concat(failOnErrorOrAbortPart),
			),
		);

	const effects: Record<string, Effect.Effect<unknown, AiError>> = {};
	for (const key of STREAM_TEXT_EFFECT_KEYS) {
		effects[key] = Effect.tryPromise({
			try: async () => raw[key],
			catch: failed,
		}).pipe(Effect.onInterrupt(() => abort));
	}

	return {
		...effects,
		raw,
		stream: partStream(),
		fullStream: partStream(),
		textStream: view(() => raw.textStream),
		partialOutputStream: view(() => raw.partialOutputStream),
		elementStream: view(() => raw.elementStream),
	} as StreamTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT>;
}
