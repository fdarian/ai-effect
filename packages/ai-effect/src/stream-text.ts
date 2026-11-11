import * as AiSdk from 'ai';
import { Effect } from 'effect';
import { Ai } from '#src/ai.ts';
import { AiError } from '#src/ai-error.ts';
import { GENERATE_TEXT_CALLBACK_KEYS } from '#src/generate-text.ts';
import { combineAbortSignals } from '#src/internal/abort-signal.ts';
import { reasonFromSdkError } from '#src/internal/ai-error-from-sdk.ts';
import {
	type EffectifyCallbacks,
	makeRunPromise,
	wrapCallbacks,
} from '#src/internal/callbacks.ts';
import {
	makeStreamTextResult,
	type StreamTextResult,
} from '#src/internal/stream-text-result.ts';
import {
	composeCallbacks,
	rebindTools,
	streamLifecycleCallbacks,
	TEXT_CALLBACK_ALIASES,
	textRequestAttributes,
	withStreamSpan,
} from '#src/internal/tracing.ts';

const STREAM_TEXT_CALLBACK_KEYS = [
	...GENERATE_TEXT_CALLBACK_KEYS,
	'onChunk',
	'onError',
	'onAbort',
] as const;
type StreamTextCallbackKeys = (typeof STREAM_TEXT_CALLBACK_KEYS)[number];

type OriginalStreamTextDef<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = Parameters<typeof AiSdk.streamText<TOOLS, RUNTIME_CONTEXT, OUTPUT>>[0];

/** Same deferred-`Parameters` inference problem as {@link EffectGenerateTextDef}. */
type EffectStreamTextDef<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
	R,
> = EffectifyCallbacks<
	Omit<
		OriginalStreamTextDef<TOOLS, RUNTIME_CONTEXT, OUTPUT>,
		'tools' | 'runtimeContext' | 'output'
	>,
	StreamTextCallbackKeys,
	R
> & {
	tools?: TOOLS;
	runtimeContext?: RUNTIME_CONTEXT;
	output?: OUTPUT;
	abortSignal?: AbortSignal;
};

/** Takes the settings untyped for the same deferred-generics reason as `textRequestAttributes`. */
function errorMayBeRetried(settings: Readonly<Record<string, unknown>>) {
	const streamRetries = settings.streamRetries;
	if (typeof streamRetries !== 'number') return false;
	return streamRetries > 0 || settings.onError !== undefined;
}

export function streamText<
	TOOLS extends AiSdk.ToolSet = AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown> = Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output = AiSdk.Output.Output<
		string,
		string,
		never
	>,
	R = never,
>(
	params: EffectStreamTextDef<TOOLS, RUNTIME_CONTEXT, OUTPUT, R>,
): Effect.Effect<
	StreamTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT>,
	AiError,
	Ai | R
> {
	return Effect.gen(function* () {
		const ai = yield* Ai;

		return yield* withStreamSpan(
			'ai-effect.streamText',
			{
				kind: 'internal',
				attributes: textRequestAttributes(params, { stream: true }),
			},
			(call) =>
				Effect.gen(function* () {
					const runPromise = yield* makeRunPromise<R>();
					const wrapped = wrapCallbacks(
						runPromise,
						params,
						STREAM_TEXT_CALLBACK_KEYS,
					);

					const callController = new AbortController();
					const abortSignal = combineAbortSignals(
						ai.abortSignal,
						callController.signal,
						params.abortSignal,
					);
					const tools = yield* rebindTools<TOOLS>(params.tools, {
						parent: call.span,
					});
					const lifecycle = streamLifecycleCallbacks(call, {
						method: 'streamText',
						signal: abortSignal,
						userSignal: params.abortSignal,
						errorMayBeRetried: errorMayBeRetried(params),
					});

					const raw = yield* Effect.try({
						try: () =>
							AiSdk.streamText({
								...params,
								...wrapped,
								...composeCallbacks(wrapped, lifecycle, TEXT_CALLBACK_ALIASES, {
									// Setting `onError` replaces the SDK's default logger.
									onError: ({ error }: { error: unknown }) =>
										console.error(error),
								}),
								...(tools === undefined ? {} : { tools }),
								abortSignal,
							} as OriginalStreamTextDef<TOOLS, RUNTIME_CONTEXT, OUTPUT>),
						catch: (cause) =>
							new AiError({
								method: 'streamText',
								reason: reasonFromSdkError(cause),
							}),
					});

					return makeStreamTextResult(
						'streamText',
						raw,
						callController,
						params.tools,
					);
				}),
		);
	});
}
