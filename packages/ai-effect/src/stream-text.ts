import * as AiSdk from 'ai';
import { Effect } from 'effect';
import { Ai } from '#src/ai.ts';
import { AiError } from '#src/ai-error.ts';
import { GENERATE_TEXT_CALLBACK_KEYS } from '#src/generate-text.ts';
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
		const runPromise = yield* makeRunPromise<R>();

		const callController = new AbortController();
		const signals = [ai.abortSignal, callController.signal];
		if (params.abortSignal !== undefined) signals.push(params.abortSignal);

		const raw = yield* Effect.try({
			try: () =>
				AiSdk.streamText({
					...params,
					...wrapCallbacks(runPromise, params, STREAM_TEXT_CALLBACK_KEYS),
					abortSignal: AbortSignal.any(signals),
				} as OriginalStreamTextDef<TOOLS, RUNTIME_CONTEXT, OUTPUT>),
			catch: (cause) =>
				new AiError({
					method: 'streamText',
					reason: reasonFromSdkError(cause),
				}),
		});

		return makeStreamTextResult('streamText', raw, callController);
	});
}
