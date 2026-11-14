import { MockLanguageModelV4 } from 'ai/test';
import { type Cause, Effect, Layer, Logger } from 'effect';
import z from 'zod/v4';
import { Ai } from '#src/ai.ts';
import type { makeTestTracer } from '#src/internal/test-tracer.ts';

export const usage = {
	inputTokens: { total: 3, noCache: 1, cacheRead: 2, cacheWrite: undefined },
	outputTokens: { total: 10, text: 6, reasoning: 4 },
};

export const finishChunk = (unified: 'stop' | 'tool-calls' = 'stop') => ({
	type: 'finish' as const,
	usage,
	finishReason: { unified, raw: unified },
});

export const textChunks = (...deltas: string[]) => [
	{ type: 'stream-start' as const, warnings: [] },
	{
		type: 'response-metadata' as const,
		id: 'resp-1',
		modelId: 'resp-model',
	},
	{ type: 'text-start' as const, id: '1' },
	...deltas.map((delta) => ({ type: 'text-delta' as const, id: '1', delta })),
	{ type: 'text-end' as const, id: '1' },
	finishChunk(),
];

export const toolCallChunks = (toolName: string) => [
	{ type: 'stream-start' as const, warnings: [] },
	{
		type: 'tool-call' as const,
		toolCallId: 'call-1',
		toolName,
		input: '{"city":"Paris"}',
	},
	finishChunk('tool-calls'),
];

export const streamOf = (
	chunks: ReadonlyArray<unknown>,
	options?: { open: boolean },
) =>
	new ReadableStream({
		start(controller) {
			for (const chunk of chunks) controller.enqueue(chunk);
			if (options?.open !== true) controller.close();
		},
	});

export const streamModel = (chunks: ReadonlyArray<unknown>) =>
	new MockLanguageModelV4({
		doStream: async () => ({ stream: streamOf(chunks) as never }),
	});

/** Emits the chunks, then stays open until the SDK aborts it. */
export const hangingModel = (chunks: ReadonlyArray<unknown>) =>
	new MockLanguageModelV4({
		doStream: async () => ({
			stream: streamOf(chunks, { open: true }) as never,
		}),
	});

export const generateModel = (
	content: unknown[] = [{ type: 'text', text: 'hello' }],
	unified: 'stop' | 'tool-calls' = 'stop',
) =>
	new MockLanguageModelV4({
		provider: 'openai.chat',
		modelId: 'gpt-test',
		doGenerate: async () => ({
			content: content as never,
			finishReason: { unified, raw: unified },
			usage,
			warnings: [],
			response: { id: 'resp-1', modelId: 'resp-model', timestamp: new Date() },
		}),
	});

export const weatherInput = z.object({ city: z.string() });

export const run = <A, E>(
	effect: Effect.Effect<A, E, Ai>,
	tracer: ReturnType<typeof makeTestTracer>,
) =>
	effect.pipe(
		Effect.provide(Layer.merge(Ai.layer, tracer.layer)),
		Effect.runPromise,
	);

type LogEntry = {
	readonly level: string;
	readonly message: ReadonlyArray<unknown>;
	readonly cause: Cause.Cause<unknown>;
};

/** Replaces the logger set, so entries are captured and nothing is printed. */
export const makeTestLogger = () => {
	const entries: LogEntry[] = [];
	const layer = Logger.layer(
		[
			Logger.make((options) => {
				entries.push({
					level: options.logLevel,
					message: options.message as ReadonlyArray<unknown>,
					cause: options.cause,
				});
			}),
		],
		{ mergeWithExisting: false },
	);
	return { entries, layer };
};
