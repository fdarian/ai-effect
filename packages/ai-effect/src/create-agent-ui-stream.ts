import * as AiSdk from 'ai';
import { Effect, Stream } from 'effect';
import { Ai } from '#src/ai.ts';
import { AiError, InvalidRequestError } from '#src/ai-error.ts';
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
import {
	agentSpanName,
	CurrentCallSpan,
	streamLifecycleCallbacks,
	withStreamSpan,
} from '#src/internal/tracing.ts';
import {
	AGENT_TRACING_HOOKS,
	type ToolLoopAgent,
} from '#src/tool-loop-agent.ts';

const CREATE_AGENT_UI_STREAM_CALLBACK_KEYS = [
	'onStepEnd',
	'onStepFinish',
	'onUIMessageStepEnd',
	'onEnd',
	'onFinish',
] as const;
type CreateAgentUIStreamCallbackKeys =
	(typeof CREATE_AGENT_UI_STREAM_CALLBACK_KEYS)[number];

type OriginalParams<
	CALL_OPTIONS,
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
	MESSAGE_METADATA,
	UI_MESSAGE extends AiSdk.UIMessage<
		MESSAGE_METADATA,
		AiSdk.UIDataTypes,
		AiSdk.InferUITools<TOOLS>
	>,
> = Parameters<
	typeof AiSdk.createAgentUIStream<
		CALL_OPTIONS,
		TOOLS,
		RUNTIME_CONTEXT,
		OUTPUT,
		MESSAGE_METADATA,
		UI_MESSAGE
	>
>[0];

/**
 * `agent` is omitted off the SDK params and re-added in a plain intersected
 * object so TS infers the agent generics from ai-effect's `ToolLoopAgent`
 * instead of through the deferred `Parameters` type.
 */
type EffectParams<
	CALL_OPTIONS,
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
	MESSAGE_METADATA,
	UI_MESSAGE extends AiSdk.UIMessage<
		MESSAGE_METADATA,
		AiSdk.UIDataTypes,
		AiSdk.InferUITools<TOOLS>
	>,
	R,
> = EffectifyCallbacks<
	Omit<
		OriginalParams<
			CALL_OPTIONS,
			TOOLS,
			RUNTIME_CONTEXT,
			OUTPUT,
			MESSAGE_METADATA,
			UI_MESSAGE
		>,
		'agent'
	>,
	CreateAgentUIStreamCallbackKeys,
	R
> & {
	agent: ToolLoopAgent<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>;
};

/**
 * Effect version of the SDK's `createAgentUIStream`. The outer Effect fails on
 * validation, conversion and model-start errors, before any chunk exists, so a
 * server can still answer with an error status. The returned stream wraps one
 * SDK stream, so it can be run only once; interrupting its consumer aborts the
 * model call.
 */
export function createAgentUIStream<
	CALL_OPTIONS = never,
	// biome-ignore lint/complexity/noBannedTypes: same default as the SDK's createAgentUIStream
	TOOLS extends AiSdk.ToolSet = {},
	RUNTIME_CONTEXT extends Record<string, unknown> = Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output = never,
	MESSAGE_METADATA = unknown,
	UI_MESSAGE extends AiSdk.UIMessage<
		MESSAGE_METADATA,
		AiSdk.UIDataTypes,
		AiSdk.InferUITools<TOOLS>
	> = AiSdk.UIMessage<
		MESSAGE_METADATA,
		AiSdk.UIDataTypes,
		AiSdk.InferUITools<TOOLS>
	>,
	R = never,
>(
	params: EffectParams<
		CALL_OPTIONS,
		TOOLS,
		RUNTIME_CONTEXT,
		OUTPUT,
		MESSAGE_METADATA,
		UI_MESSAGE,
		R
	>,
): Effect.Effect<
	Stream.Stream<AiSdk.InferUIMessageChunk<UI_MESSAGE>, AiError>,
	AiError,
	Ai | R
> {
	const failed = (cause: unknown) =>
		new AiError({
			method: 'createAgentUIStream',
			reason: reasonFromSdkError(cause),
		});

	// Before any chunk, a TypeValidationError can only come from validating the
	// caller's `uiMessages`, not model output, so it isn't retryable.
	const failedStart = (cause: unknown) =>
		AiSdk.TypeValidationError.isInstance(cause)
			? new AiError({
					method: 'createAgentUIStream',
					reason: new InvalidRequestError({ message: cause.message, cause }),
				})
			: failed(cause);

	const hooks = params.agent[AGENT_TRACING_HOOKS];

	return Effect.gen(function* () {
		const ai = yield* Ai;
		const call = yield* CurrentCallSpan;
		const runPromise = yield* makeRunPromise<R>();
		const callController = new AbortController();
		const abortSignal = combineAbortSignals(
			ai.abortSignal,
			callController.signal,
			params.abortSignal,
		);
		const lifecycle = streamLifecycleCallbacks(call, {
			method: 'createAgentUIStream',
			signal: abortSignal,
			interruptSignals: [ai.abortSignal, callController.signal],
		});
		const agent = yield* hooks.buildSdkAgent({
			call,
			callbacks: lifecycle.callbacks,
		});

		// Not the fiber's signal: the stream outlives this Effect.
		const uiStream = yield* Effect.tryPromise({
			try: () =>
				AiSdk.createAgentUIStream({
					...params,
					...wrapCallbacks(
						runPromise,
						params,
						CREATE_AGENT_UI_STREAM_CALLBACK_KEYS,
					),
					agent,
					abortSignal,
				} as OriginalParams<
					CALL_OPTIONS,
					TOOLS,
					RUNTIME_CONTEXT,
					OUTPUT,
					MESSAGE_METADATA,
					UI_MESSAGE
				>),
			catch: failedStart,
		}).pipe(
			Effect.onInterrupt(() => Effect.sync(() => callController.abort())),
		);

		return Stream.fromReadableStream({
			evaluate: () => uiStream,
			onError: (cause) => {
				lifecycle.streamEnd.failed(cause);
				return failed(cause);
			},
		}).pipe(
			Stream.onEnd(Effect.sync(lifecycle.streamEnd.closed)),
			abortOnInterrupt(callController),
		);
	}).pipe(
		withStreamSpan(agentSpanName(params.agent.id, 'createAgentUIStream'), {
			kind: 'internal',
			attributes: hooks.requestAttributes(true),
		}),
	);
}
