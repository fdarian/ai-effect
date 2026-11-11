import * as AiSdk from 'ai';
import { Effect } from 'effect';
import { Ai } from '#src/ai.ts';
import { AiError } from '#src/ai-error.ts';
import { GENERATE_TEXT_CALLBACK_KEYS } from '#src/generate-text.ts';
import {
	type EffectifyCallbacks,
	makeRunPromise,
	wrapCallbacks,
} from '#src/internal/callbacks.ts';

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

type StreamTextResult<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = ReturnType<typeof AiSdk.streamText<TOOLS, RUNTIME_CONTEXT, OUTPUT>>;

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

		const originalParams = {
			...params,
			...wrapCallbacks(runPromise, params, STREAM_TEXT_CALLBACK_KEYS),
		} as OriginalStreamTextDef<TOOLS, RUNTIME_CONTEXT, OUTPUT>;

		try {
			// Object.assign keeps `originalParams`' prompt/messages union intact,
			// which spreading it would collapse.
			return AiSdk.streamText(
				Object.assign({}, originalParams, {
					abortSignal:
						originalParams.abortSignal === undefined
							? ai.abortSignal
							: AbortSignal.any([ai.abortSignal, originalParams.abortSignal]),
				}),
			);
		} catch (cause) {
			return yield* Effect.fail(
				new AiError({ message: 'streamText failed', cause }),
			);
		}
	});
}
