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

// `onStepError`, `onError` and `onAbort` can only be set on the settings: the
// SDK doesn't accept them per call.
const SETTINGS_CALLBACK_KEYS = [
	...GENERATE_TEXT_CALLBACK_KEYS,
	'onStepError',
	'onError',
	'onAbort',
] as const;
type SettingsCallbackKeys = (typeof SETTINGS_CALLBACK_KEYS)[number];

const CALL_CALLBACK_KEYS = [
	'onStart',
	'experimental_onStart',
	'onStepStart',
	'experimental_onStepStart',
	'onToolExecutionStart',
	'experimental_onToolCallStart',
	'onToolExecutionEnd',
	'experimental_onToolCallFinish',
	'onStepEnd',
	'onStepFinish',
	'onEnd',
	'onFinish',
] as const;
type CallCallbackKeys = (typeof CALL_CALLBACK_KEYS)[number];

type OriginalSettings<
	CALL_OPTIONS,
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = AiSdk.ToolLoopAgentSettings<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>;

type SdkAgent<
	CALL_OPTIONS,
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = AiSdk.ToolLoopAgent<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>;

type OriginalGenerateParams<
	CALL_OPTIONS,
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = Parameters<
	SdkAgent<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>['generate']
>[0];

type OriginalStreamParams<
	CALL_OPTIONS,
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = Parameters<
	SdkAgent<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>['stream']
>[0];

type GenerateTextResult<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = Awaited<
	ReturnType<SdkAgent<never, TOOLS, RUNTIME_CONTEXT, OUTPUT>['generate']>
>;

type StreamTextResult<
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = Awaited<
	ReturnType<SdkAgent<never, TOOLS, RUNTIME_CONTEXT, OUTPUT>['stream']>
>;

/**
 * Same deferred-`Parameters` inference problem as `EffectGenerateTextDef`: the
 * bare inference sites (`tools`, `runtimeContext`, `output`,
 * `callOptionsSchema`) are omitted off the SDK settings and re-added in a plain
 * intersected object so TS infers them.
 */
type EffectToolLoopAgentSettings<
	CALL_OPTIONS,
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
	R,
> = EffectifyCallbacks<
	Omit<
		OriginalSettings<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>,
		'tools' | 'runtimeContext' | 'output' | 'callOptionsSchema'
	>,
	SettingsCallbackKeys,
	R
> & {
	tools?: TOOLS;
	runtimeContext?: RUNTIME_CONTEXT;
	output?: OUTPUT;
	callOptionsSchema?: AiSdk.FlexibleSchema<CALL_OPTIONS>;
};

/**
 * The SDK's call parameters are a `prompt` | `messages` union, which
 * `EffectifyCallbacks` (it uses `Omit`) would collapse, so it's applied to each
 * member separately.
 */
type EffectifyCallParams<P, R> = P extends unknown
	? EffectifyCallbacks<P, CallCallbackKeys, R>
	: never;

export interface ToolLoopAgent<
	CALL_OPTIONS = never,
	// biome-ignore lint/complexity/noBannedTypes: same default as the SDK's ToolLoopAgent
	TOOLS extends AiSdk.ToolSet = {},
	RUNTIME_CONTEXT extends Record<string, unknown> = Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output = never,
> {
	readonly id: string | undefined;
	readonly tools: TOOLS;
	readonly generate: <R = never>(
		params: EffectifyCallParams<
			OriginalGenerateParams<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>,
			R
		>,
	) => Effect.Effect<
		GenerateTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT>,
		AiError,
		Ai | R
	>;
	readonly stream: <R = never>(
		params: EffectifyCallParams<
			OriginalStreamParams<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>,
			R
		>,
	) => Effect.Effect<
		StreamTextResult<TOOLS, RUNTIME_CONTEXT, OUTPUT>,
		AiError,
		Ai | R
	>;
}

export function toolLoopAgent<
	CALL_OPTIONS = never,
	// biome-ignore lint/complexity/noBannedTypes: same default as the SDK's ToolLoopAgent
	TOOLS extends AiSdk.ToolSet = {},
	RUNTIME_CONTEXT extends Record<string, unknown> = Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output = never,
	R = never,
>(
	settings: EffectToolLoopAgentSettings<
		CALL_OPTIONS,
		TOOLS,
		RUNTIME_CONTEXT,
		OUTPUT,
		R
	>,
): Effect.Effect<
	ToolLoopAgent<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>,
	never,
	Ai | R
> {
	return Effect.gen(function* () {
		const runPromise = yield* makeRunPromise<R>();

		const agent = new AiSdk.ToolLoopAgent({
			...settings,
			...wrapCallbacks(runPromise, settings, SETTINGS_CALLBACK_KEYS),
		} as OriginalSettings<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>);

		return {
			id: agent.id,
			tools: agent.tools,
			generate: <CALL_R = never>(
				params: EffectifyCallParams<
					OriginalGenerateParams<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>,
					CALL_R
				>,
			) =>
				Effect.gen(function* () {
					const callRunPromise = yield* makeRunPromise<CALL_R>();

					return yield* Effect.tryPromise({
						try: (signal) =>
							agent.generate({
								...params,
								...wrapCallbacks(callRunPromise, params, CALL_CALLBACK_KEYS),
								abortSignal:
									params.abortSignal === undefined
										? signal
										: AbortSignal.any([signal, params.abortSignal]),
							} as OriginalGenerateParams<
								CALL_OPTIONS,
								TOOLS,
								RUNTIME_CONTEXT,
								OUTPUT
							>),
						catch: (cause) =>
							new AiError({ message: 'ToolLoopAgent.generate failed', cause }),
					});
				}),
			stream: <CALL_R = never>(
				params: EffectifyCallParams<
					OriginalStreamParams<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>,
					CALL_R
				>,
			) =>
				Effect.gen(function* () {
					const ai = yield* Ai;
					const callRunPromise = yield* makeRunPromise<CALL_R>();

					// The layer signal, not the fiber's: the stream outlives this Effect.
					return yield* Effect.tryPromise({
						try: () =>
							agent.stream({
								...params,
								...wrapCallbacks(callRunPromise, params, CALL_CALLBACK_KEYS),
								abortSignal:
									params.abortSignal === undefined
										? ai.abortSignal
										: AbortSignal.any([ai.abortSignal, params.abortSignal]),
							} as OriginalStreamParams<
								CALL_OPTIONS,
								TOOLS,
								RUNTIME_CONTEXT,
								OUTPUT
							>),
						catch: (cause) =>
							new AiError({ message: 'ToolLoopAgent.stream failed', cause }),
					});
				}),
		};
	});
}
