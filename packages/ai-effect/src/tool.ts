import * as AiSdk from 'ai';
import { Effect, Schema } from 'effect';
import type { Ai } from '#src/ai.ts';
import { makeRunPromise, wrapCallback } from '#src/internal/callbacks.ts';
import { effectSchema } from '#src/schema.ts';

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
		const execute = params.execute;

		const originalParams = {
			...params,
			inputSchema: normalizeToolSchema(params.inputSchema),
			...(params.outputSchema === undefined
				? {}
				: { outputSchema: normalizeToolSchema(params.outputSchema) }),
			// Unlike the other callbacks, `execute` is bound to the call: the SDK's
			// abort signal interrupts the fiber instead of leaving it to run until
			// the layer closes.
			execute:
				execute &&
				((input: INPUT, options: AiSdk.ToolExecutionOptions<CONTEXT>) =>
					runPromise(execute(input, options), {
						signal: options.abortSignal,
					})),
			onInputStart: wrapCallback(runPromise, params.onInputStart),
			onInputDelta: wrapCallback(runPromise, params.onInputDelta),
			onInputAvailable: wrapCallback(runPromise, params.onInputAvailable),
			toModelOutput: wrapCallback(runPromise, params.toModelOutput),
		} as OriginalToolDef<INPUT, OUTPUT, CONTEXT>;

		return AiSdk.tool(originalParams);
	});
}
