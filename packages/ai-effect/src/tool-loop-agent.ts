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
import { rebindTools } from '#src/internal/tool-tracing.ts';
import {
	agentSpanName,
	type CallSpan,
	CurrentCallSpan,
	composeCallbacks,
	streamLifecycleCallbacks,
	TEXT_CALLBACK_ALIASES,
	type TracingCallbacks,
	textCallbacks,
	textRequestAttributes,
	withCallSpan,
	withStreamSpan,
} from '#src/internal/tracing.ts';

// `onStepError` arrives via GENERATE_TEXT_CALLBACK_KEYS; it, `onError` and
// `onAbort` can only be set on the settings: the SDK doesn't accept them per
// call.
const SETTINGS_CALLBACK_KEYS = [
	...GENERATE_TEXT_CALLBACK_KEYS,
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

/**
 * Lets `createAgentUIStream` build the per-call SDK agent. A symbol key keeps
 * it off the public surface; it is not re-exported from the package entry.
 * `Symbol.for` so an agent made by another copy of ai-effect still works with
 * this copy's `createAgentUIStream`.
 */
export const AGENT_TRACING_HOOKS: unique symbol = Symbol.for(
	'ai-effect/agent-tracing',
);

export type AgentCallTracing = {
	readonly call: CallSpan;
	readonly callbacks: TracingCallbacks;
};

export type AgentTracingHooks<
	CALL_OPTIONS,
	TOOLS extends AiSdk.ToolSet,
	RUNTIME_CONTEXT extends Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output,
> = {
	readonly requestAttributes: (stream: boolean) => Record<string, unknown>;
	readonly buildSdkAgent: (
		tracing: AgentCallTracing,
	) => Effect.Effect<SdkAgent<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>>;
};

export interface ToolLoopAgent<
	CALL_OPTIONS = never,
	// biome-ignore lint/complexity/noBannedTypes: same default as the SDK's ToolLoopAgent
	TOOLS extends AiSdk.ToolSet = {},
	RUNTIME_CONTEXT extends Record<string, unknown> = Record<string, unknown>,
	OUTPUT extends AiSdk.Output.Output = never,
> {
	readonly id: string | undefined;
	readonly tools: TOOLS;
	/**
	 * The SDK agent, for everything that isn't wrapped. Calling `raw.stream` or
	 * `raw.generate` directly bypasses the `Ai` layer's abort signal, the Effect
	 * callback bridging and Effect tracing.
	 */
	readonly raw: SdkAgent<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>;
	readonly [AGENT_TRACING_HOOKS]: AgentTracingHooks<
		CALL_OPTIONS,
		TOOLS,
		RUNTIME_CONTEXT,
		OUTPUT
	>;
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

function make<
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
		const wrapped = wrapCallbacks(runPromise, settings, SETTINGS_CALLBACK_KEYS);

		const agent = new AiSdk.ToolLoopAgent({
			...settings,
			...wrapped,
		} as OriginalSettings<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>);

		const buildSdkAgent = (tracing: AgentCallTracing) =>
			Effect.gen(function* () {
				const services = yield* Effect.context<never>();
				// `runPromise` carries `make`'s parent span, so each settings callback
				// is re-parented under this call's span.
				const callWrapped = wrapCallbacks(
					(effect: Effect.Effect<unknown, never, R>) =>
						runPromise(effect.pipe(Effect.withParentSpan(tracing.call.span))),
					settings,
					SETTINGS_CALLBACK_KEYS,
				);
				const rebind = (tools: TOOLS | undefined) =>
					Effect.runSyncWith(services)(
						rebindTools<TOOLS>(tools, {
							parent: tracing.call.span,
							agentName: agent.id,
						}),
					);
				const tools = rebind(settings.tools);
				const userPrepareCall = (
					settings as OriginalSettings<
						CALL_OPTIONS,
						TOOLS,
						RUNTIME_CONTEXT,
						OUTPUT
					>
				).prepareCall;

				return new AiSdk.ToolLoopAgent({
					...settings,
					...callWrapped,
					...composeCallbacks(
						callWrapped,
						tracing.callbacks,
						TEXT_CALLBACK_ALIASES,
						{
							// Setting `onError` replaces the SDK's default logger.
							onError: ({ error }: { error: unknown }) => console.error(error),
						},
					),
					...(tools === undefined ? {} : { tools }),
					...(userPrepareCall === undefined
						? {}
						: {
								// `prepareCall` may return its own `tools`, which would escape the rebinding above.
								prepareCall: async (options: never) => {
									const prepared = await userPrepareCall(options);
									if (prepared?.tools === undefined) return prepared;
									return {
										...prepared,
										tools: rebind(prepared.tools as TOOLS),
									};
								},
							}),
				} as OriginalSettings<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>);
			});

		return {
			id: agent.id,
			tools: agent.tools,
			raw: agent,
			[AGENT_TRACING_HOOKS]: {
				requestAttributes: (stream) =>
					textRequestAttributes(settings, { stream, agentName: agent.id }),
				buildSdkAgent,
			},
			generate: <CALL_R = never>(
				params: EffectifyCallParams<
					OriginalGenerateParams<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>,
					CALL_R
				>,
			) =>
				Effect.gen(function* () {
					const call = yield* CurrentCallSpan;
					const callRunPromise = yield* makeRunPromise<CALL_R>();
					const sdkAgent = yield* buildSdkAgent({
						call,
						callbacks: textCallbacks(call),
					});

					return yield* Effect.tryPromise({
						try: (signal) =>
							sdkAgent.generate({
								...params,
								...wrapCallbacks(callRunPromise, params, CALL_CALLBACK_KEYS),
								abortSignal: combineAbortSignals(signal, params.abortSignal),
							} as OriginalGenerateParams<
								CALL_OPTIONS,
								TOOLS,
								RUNTIME_CONTEXT,
								OUTPUT
							>),
						catch: (cause) =>
							new AiError({
								method: 'ToolLoopAgent.generate',
								reason: reasonFromSdkError(cause),
							}),
					});
				}).pipe(
					withCallSpan(agentSpanName(agent.id, 'ToolLoopAgent.generate'), {
						kind: 'internal',
						attributes: textRequestAttributes(settings, {
							stream: false,
							agentName: agent.id,
						}),
					}),
				),
			stream: <CALL_R = never>(
				params: EffectifyCallParams<
					OriginalStreamParams<CALL_OPTIONS, TOOLS, RUNTIME_CONTEXT, OUTPUT>,
					CALL_R
				>,
			) =>
				Effect.gen(function* () {
					const ai = yield* Ai;
					const call = yield* CurrentCallSpan;
					const callRunPromise = yield* makeRunPromise<CALL_R>();
					const callController = new AbortController();
					const abortSignal = combineAbortSignals(
						ai.abortSignal,
						callController.signal,
						params.abortSignal,
					);
					const lifecycle = streamLifecycleCallbacks(call, {
						method: 'ToolLoopAgent.stream',
						signal: abortSignal,
						interruptSignals: [ai.abortSignal, callController.signal],
					});
					const sdkAgent = yield* buildSdkAgent({
						call,
						callbacks: lifecycle.callbacks,
					});

					// Not the fiber's signal: the stream outlives this Effect.
					const raw = yield* Effect.tryPromise({
						try: () =>
							sdkAgent.stream({
								...params,
								...wrapCallbacks(callRunPromise, params, CALL_CALLBACK_KEYS),
								abortSignal,
							} as OriginalStreamParams<
								CALL_OPTIONS,
								TOOLS,
								RUNTIME_CONTEXT,
								OUTPUT
							>),
						catch: (cause) =>
							new AiError({
								method: 'ToolLoopAgent.stream',
								reason: reasonFromSdkError(cause),
							}),
					}).pipe(
						Effect.onInterrupt(() => Effect.sync(() => callController.abort())),
					);

					return makeStreamTextResult(
						'ToolLoopAgent.stream',
						raw,
						callController,
						agent.tools,
						lifecycle.streamEnd,
					);
				}).pipe(
					withStreamSpan(agentSpanName(agent.id, 'ToolLoopAgent.stream'), {
						kind: 'internal',
						attributes: textRequestAttributes(settings, {
							stream: true,
							agentName: agent.id,
						}),
					}),
				),
		};
	});
}

export const ToolLoopAgent = { make };
