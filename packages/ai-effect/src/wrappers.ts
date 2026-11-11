import * as AiSdk from 'ai';
import { Data, Effect, FiberSet, Schema } from 'effect';
import { Ai } from '#src/ai.ts';
import { effectSchema } from '#src/schema.ts';

export class AiError extends Data.TaggedError('ai-effect/AiError')<{
	message: string;
	cause?: unknown;
}> {}

/**
 * `runtimePromise` captures the call-site context (so callbacks see the
 * caller's services `R`) but forks into the `Ai` layer's FiberSet, so callbacks
 * that outlive the call are only interrupted when the layer's scope closes.
 */
const makeRunPromise = <R>() =>
	Effect.gen(function* () {
		const ai = yield* Ai;
		return yield* FiberSet.runtimePromise(ai.fiberSet)<R>();
	});

// biome-ignore lint/suspicious/noExplicitAny: internal bridging helper, type safety enforced at public API boundary
function wrapCallback(runPromise: any, callback: any) {
	if (callback == null) return undefined;
	// biome-ignore lint/suspicious/noExplicitAny: internal bridging helper
	return (...args: any[]) => runPromise(callback(...args));
}

// biome-ignore lint/suspicious/noExplicitAny: internal bridging helper, type safety enforced at public API boundary
function wrapCallbacks(runPromise: any, params: any, keys: readonly string[]) {
	const wrapped: Record<string, unknown> = {};
	for (const key of keys) {
		wrapped[key] = wrapCallback(runPromise, params[key]);
	}
	return wrapped;
}

/**
 * v7 renamed several callbacks (e.g. onStepFinish -> onStepEnd, onFinish ->
 * onEnd, experimental_onStart -> onStart) and added onLanguageModelCallStart /
 * onLanguageModelCallEnd. Both the old and new names are listed here so
 * EffectifyCallbacks rewrites whichever one a caller uses — the SDK resolves
 * old/new pairs internally with `newName ?? oldName`, and an explicit
 * `undefined` for an unused key doesn't disturb that resolution.
 */
const GENERATE_TEXT_CALLBACK_KEYS = [
	'onStepFinish',
	'onStepEnd',
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

const STREAM_TEXT_CALLBACK_KEYS = [
	...GENERATE_TEXT_CALLBACK_KEYS,
	'onChunk',
	'onError',
	'onAbort',
] as const;
type StreamTextCallbackKeys = (typeof STREAM_TEXT_CALLBACK_KEYS)[number];

type EffectifyCallbacks<T, Keys extends string, R> = Omit<T, Keys & keyof T> & {
	[K in Keys & keyof T]?: NonNullable<T[K]> extends (
		...args: infer A
	) => unknown
		? (...args: A) => Effect.Effect<void, never, R>
		: T[K];
};

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
	return Effect.gen(function* () {
		const runPromise = yield* makeRunPromise<R>();

		const originalParams = {
			...params,
			...wrapCallbacks(runPromise, params, GENERATE_TEXT_CALLBACK_KEYS),
		} as OriginalGenerateTextDef<TOOLS, RUNTIME_CONTEXT, OUTPUT>;

		return yield* Effect.tryPromise({
			try: (signal) =>
				// Object.assign keeps `originalParams`' prompt/messages union intact,
				// which spreading it would collapse.
				AiSdk.generateText(
					Object.assign({}, originalParams, {
						abortSignal:
							originalParams.abortSignal === undefined
								? signal
								: AbortSignal.any([signal, originalParams.abortSignal]),
					}),
				),
			catch: (cause) => new AiError({ message: 'generateText failed', cause }),
		});
	});
}

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
		const runPromise = yield* makeRunPromise<R>();

		const originalParams = {
			...params,
			...wrapCallbacks(runPromise, params, STREAM_TEXT_CALLBACK_KEYS),
		} as OriginalStreamTextDef<TOOLS, RUNTIME_CONTEXT, OUTPUT>;

		try {
			return AiSdk.streamText(originalParams);
		} catch (cause) {
			return yield* Effect.fail(
				new AiError({ message: 'streamText failed', cause }),
			);
		}
	});
}

type OriginalToolDef<
	INPUT,
	OUTPUT,
	CONTEXT extends Record<string, unknown>,
> = Parameters<typeof AiSdk.tool<INPUT, OUTPUT, CONTEXT>>[0];

type ToolModelOutput = Awaited<
	ReturnType<NonNullable<AiSdk.Tool<unknown, unknown>['toModelOutput']>>
>;

type EffectToolSchema<T> = AiSdk.FlexibleSchema<T> | Schema.Codec<T, unknown>;

function normalizeToolSchema<T>(schema: EffectToolSchema<T>) {
	return Schema.isSchema(schema)
		? effectSchema(schema as Schema.Codec<T, unknown>)
		: schema;
}

/**
 * Same deferred-`Parameters` inference problem as {@link EffectGenerateTextDef}
 * for the bare (non-`NoInfer`) `contextSchema` and `outputSchema` inference
 * sites. `OriginalToolDef` stays untouched for the cast back into `ai.tool`,
 * because flattening it would break overload resolution across the `Tool`
 * union.
 */
type EffectToolDef<
	INPUT,
	OUTPUT,
	CONTEXT extends Record<string, unknown>,
	R,
> = Omit<
	OriginalToolDef<INPUT, OUTPUT, CONTEXT>,
	| 'execute'
	| 'onInputStart'
	| 'onInputDelta'
	| 'onInputAvailable'
	| 'toModelOutput'
	| 'contextSchema'
	| 'inputSchema'
	| 'outputSchema'
> & {
	inputSchema: EffectToolSchema<INPUT>;
	execute?: (
		input: INPUT,
		options: AiSdk.ToolExecutionOptions<CONTEXT>,
	) => Effect.Effect<OUTPUT, unknown, R>;
	outputSchema?: EffectToolSchema<OUTPUT>;
	onInputStart?: (
		options: AiSdk.ToolExecutionOptions<CONTEXT>,
	) => Effect.Effect<void, never, R>;
	onInputDelta?: (
		options: { inputTextDelta: string } & AiSdk.ToolExecutionOptions<CONTEXT>,
	) => Effect.Effect<void, never, R>;
	onInputAvailable?: (
		options: { input: INPUT } & AiSdk.ToolExecutionOptions<CONTEXT>,
	) => Effect.Effect<void, never, R>;
	toModelOutput?: (options: {
		toolCallId: string;
		input: INPUT;
		output: OUTPUT;
	}) => Effect.Effect<ToolModelOutput, never, R>;
	contextSchema?: AiSdk.FlexibleSchema<CONTEXT>;
};

export function tool<
	INPUT,
	OUTPUT,
	R = never,
	CONTEXT extends Record<string, unknown> = Record<string, unknown>,
>(
	params: EffectToolDef<INPUT, OUTPUT, CONTEXT, R>,
): Effect.Effect<AiSdk.Tool<INPUT, OUTPUT, CONTEXT>, never, Ai | R> {
	return Effect.gen(function* () {
		const runPromise = yield* makeRunPromise<R>();

		const originalParams = {
			...params,
			inputSchema: normalizeToolSchema(params.inputSchema),
			...(params.outputSchema === undefined
				? {}
				: { outputSchema: normalizeToolSchema(params.outputSchema) }),
			...(params.execute && {
				execute: (input: INPUT, options: AiSdk.ToolExecutionOptions<CONTEXT>) =>
					// biome-ignore lint/style/noNonNullAssertion: guarded by truthiness check
					runPromise(params.execute!(input, options)),
			}),
			onInputStart: wrapCallback(runPromise, params.onInputStart),
			onInputDelta: wrapCallback(runPromise, params.onInputDelta),
			onInputAvailable: wrapCallback(runPromise, params.onInputAvailable),
			...(params.toModelOutput && {
				toModelOutput: (options: {
					toolCallId: string;
					input: INPUT;
					output: OUTPUT;
				}) =>
					// biome-ignore lint/style/noNonNullAssertion: guarded by truthiness check
					runPromise(params.toModelOutput!(options)),
			}),
		} as OriginalToolDef<INPUT, OUTPUT, CONTEXT>;

		return AiSdk.tool(originalParams);
	});
}
