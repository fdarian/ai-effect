import * as AiSdk from 'ai';
import { Effect, Stream } from 'effect';
import { Ai } from '#src/ai.ts';
import { AiError } from '#src/ai-error.ts';
import {
	abortOnInterrupt,
	combineAbortSignals,
} from '#src/internal/abort-signal.ts';
import { reasonFromSdkError } from '#src/internal/ai-error-from-sdk.ts';
import {
	type EffectifyCallbacks,
	makeRunPromise,
	wrapCallbacks,
} from '#src/internal/callbacks.ts';

const CREATE_UI_MESSAGE_STREAM_CALLBACK_KEYS = [
	'onStepEnd',
	'onStepFinish',
	'onEnd',
	'onFinish',
] as const;
type CreateUIMessageStreamCallbackKeys =
	(typeof CREATE_UI_MESSAGE_STREAM_CALLBACK_KEYS)[number];

type OriginalCreateUIMessageStreamDef<UI_MESSAGE extends AiSdk.UIMessage> =
	Parameters<typeof AiSdk.createUIMessageStream<UI_MESSAGE>>[0];

type EffectCreateUIMessageStreamDef<
	UI_MESSAGE extends AiSdk.UIMessage,
	R,
> = Omit<
	EffectifyCallbacks<
		OriginalCreateUIMessageStreamDef<UI_MESSAGE>,
		CreateUIMessageStreamCallbackKeys,
		R
	>,
	'execute'
> & {
	/**
	 * A failure here reaches the SDK, which calls `onError` and emits an `error`
	 * chunk; the stream itself still ends normally.
	 */
	readonly execute: (options: {
		writer: UIMessageStreamWriter<UI_MESSAGE>;
	}) => Effect.Effect<void, unknown, R>;
};

/** Effect-typed view of the SDK's `UIMessageStreamWriterWithOutcome`. */
export interface UIMessageStreamWriter<UI_MESSAGE extends AiSdk.UIMessage> {
	readonly write: (
		chunk: AiSdk.InferUIMessageChunk<UI_MESSAGE>,
	) => Effect.Effect<void>;
	/**
	 * Forwards the stream with the current services. It stops when the run is
	 * interrupted or the `Ai` layer closes.
	 */
	readonly merge: <E, R2>(
		stream: Stream.Stream<AiSdk.InferUIMessageChunk<UI_MESSAGE>, E, R2>,
	) => Effect.Effect<void, never, R2>;
	readonly setOutcome: (
		outcome: AiSdk.UIMessageStreamOutcome,
	) => Effect.Effect<void>;
	/** The SDK writer, for `onError` and anything that isn't wrapped. */
	readonly raw: AiSdk.UIMessageStreamWriterWithOutcome<UI_MESSAGE>;
}

const whenAborted = (signal: AbortSignal) =>
	Effect.callback<void>((resume) => {
		if (signal.aborted) {
			resume(Effect.void);
			return;
		}
		const onAbort = () => resume(Effect.void);
		signal.addEventListener('abort', onAbort, { once: true });
		return Effect.sync(() => signal.removeEventListener('abort', onAbort));
	});

const makeWriter = <UI_MESSAGE extends AiSdk.UIMessage>(
	raw: AiSdk.UIMessageStreamWriterWithOutcome<UI_MESSAGE>,
	signal: AbortSignal,
): UIMessageStreamWriter<UI_MESSAGE> => ({
	write: (chunk) => Effect.sync(() => raw.write(chunk)),
	merge: (stream) =>
		Stream.toReadableStreamEffect(
			Stream.interruptWhen(stream, whenAborted(signal)),
		).pipe(Effect.map((readable) => raw.merge(readable))),
	setOutcome: (outcome) => Effect.sync(() => raw.setOutcome(outcome)),
	raw,
});

/**
 * Effect version of the SDK's `createUIMessageStream`. Each run of the stream
 * calls the SDK once; interrupting the consumer interrupts `execute` and any
 * merged stream.
 */
export function createUIMessageStream<
	UI_MESSAGE extends AiSdk.UIMessage = AiSdk.UIMessage,
	R = never,
>(
	options: EffectCreateUIMessageStreamDef<UI_MESSAGE, R>,
): Stream.Stream<AiSdk.InferUIMessageChunk<UI_MESSAGE>, AiError, Ai | R> {
	const failed = (cause: unknown) =>
		new AiError({
			method: 'createUIMessageStream',
			reason: reasonFromSdkError(cause),
		});

	return Stream.unwrap(
		Effect.gen(function* () {
			const ai = yield* Ai;
			const runPromise = yield* makeRunPromise<R>();
			const controller = new AbortController();
			const signal = combineAbortSignals(controller.signal, ai.abortSignal);

			const uiStream = yield* Effect.try({
				try: () =>
					AiSdk.createUIMessageStream<UI_MESSAGE>({
						...options,
						...wrapCallbacks(
							runPromise,
							options,
							CREATE_UI_MESSAGE_STREAM_CALLBACK_KEYS,
						),
						execute: (sdkOptions) =>
							runPromise(
								options.execute({
									writer: makeWriter(sdkOptions.writer, signal),
								}),
								{ signal: controller.signal },
							),
					} as OriginalCreateUIMessageStreamDef<UI_MESSAGE>),
				catch: failed,
			});

			return Stream.fromReadableStream({
				evaluate: () => uiStream,
				onError: failed,
			}).pipe(abortOnInterrupt(controller));
		}),
	);
}
