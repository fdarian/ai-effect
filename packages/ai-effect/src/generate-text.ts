import * as AiSdk from 'ai';
import { Effect } from 'effect';
import type { Ai } from '#src/ai.ts';
import { AiError } from '#src/ai-error.ts';
import { combineAbortSignals } from '#src/internal/abort-signal.ts';
import { reasonFromSdkError } from '#src/internal/ai-error-from-sdk.ts';
import {
	type EffectifyCallbacks,
	makeRunPromise,
	wrapCallbacks,
} from '#src/internal/callbacks.ts';
import { rebindTools } from '#src/internal/tool-tracing.ts';
import {
	composeCallbacks,
	TEXT_CALLBACK_ALIASES,
	textCallbacks,
	textRequestAttributes,
	withCallSpan,
} from '#src/internal/tracing.ts';

/**
 * v7 renamed several callbacks (e.g. onStepFinish -> onStepEnd, onFinish ->
 * onEnd, experimental_onStart -> onStart) and added onLanguageModelCallStart /
 * onLanguageModelCallEnd. Both the old and new names are listed here so
 * EffectifyCallbacks rewrites whichever one a caller uses — the SDK resolves
 * old/new pairs internally with `newName ?? oldName`, and an explicit
 * `undefined` for an unused key doesn't disturb that resolution.
 */
export const GENERATE_TEXT_CALLBACK_KEYS = [
	'onStepFinish',
	'onStepEnd',
	'onStepError',
	'onFinish',
	'onEnd',
	'onStart',
	'experimental_onStart',
	'onStepStart',
	'experimental_onStepStart',
	'onToolExecutionStart',
	'experimental_onToolCallStart',
	'onToolExecutionEnd',
	'experimental_onToolCallFinish',
	'onLanguageModelCallStart',
	'experimental_onLanguageModelCallStart',
	'onLanguageModelCallEnd',
	'experimental_onLanguageModelCallEnd',
] as const;
type GenerateTextCallbackKeys = (typeof GENERATE_TEXT_CALLBACK_KEYS)[number];

type OriginalGenerateTextDef<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = Parameters<typeof AiSdk.generateText<TOOLS, RUNTIME_CONTEXT, OUTPUT>>[0];

type GenerateTextResult<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = Awaited<
	ReturnType<typeof AiSdk.generateText<TOOLS, RUNTIME_CONTEXT, OUTPUT>>
>;

/**
 * `OriginalGenerateTextDef` stays deferred while TOOLS/RUNTIME_CONTEXT/OUTPUT
 * are still being inferred, so TS never walks into `output?: OUTPUT` to infer
 * it — of `ai`'s OUTPUT positions, that's the one bare (non-NoInfer) site.
 * Omit those three inference sites off it and intersect a plain, non-deferred
 * object type carrying them: TS merges inference candidates across
 * intersection constituents independently, so the params type callers see
 * restores inference. `OriginalGenerateTextDef` itself is left untouched (and
 * used as-is for the internal cast back into `ai.generateText`) because
 * flattening it here would collapse its `prompt`/`messages` discriminated
 * union and break that call.
 */
type EffectGenerateTextDef<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
	R,
> = EffectifyCallbacks<
	Omit<
		OriginalGenerateTextDef<TOOLS, RUNTIME_CONTEXT, OUTPUT>,
		'tools' | 'runtimeContext' | 'output'
	>,
	GenerateTextCallbackKeys,
	R
> & {
	tools?: TOOLS;
	runtimeContext?: RUNTIME_CONTEXT;
	output?: OUTPUT;
	abortSignal?: AbortSignal;
};

export function generateText<
	TOOLS extends AiSdk.ToolSet = AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown> = Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output = AiSdk.Output.Output<string, string>,
	R = never,
>(
	params: EffectGenerateTextDef<TOOLS, RUNTIME_CONTEXT, OUTPUT, R>,
): Effect.Effect<
	GenerateTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT>,
	AiError,
	Ai | R
> {
	return withCallSpan(
		'ai-effect.generateText',
		{
			kind: 'internal',
			attributes: textRequestAttributes(params, { stream: false }),
		},
		(call) =>
			Effect.gen(function* () {
				const runPromise = yield* makeRunPromise<R>();
				const wrapped = wrapCallbacks(
					runPromise,
					params,
					GENERATE_TEXT_CALLBACK_KEYS,
				);
				const tools = yield* rebindTools<TOOLS>(params.tools, {
					parent: call.span,
				});

				return yield* Effect.tryPromise({
					try: (signal) =>
						AiSdk.generateText({
							...params,
							...wrapped,
							...composeCallbacks(
								wrapped,
								textCallbacks(call),
								TEXT_CALLBACK_ALIASES,
							),
							...(tools === undefined ? {} : { tools }),
							abortSignal: combineAbortSignals(signal, params.abortSignal),
						} as OriginalGenerateTextDef<TOOLS, RUNTIME_CONTEXT, OUTPUT>),
					catch: (cause) =>
						new AiError({
							method: 'generateText',
							reason: reasonFromSdkError(cause),
						}),
				});
			}),
	);
}
