import { OpenTelemetry } from '@ai-sdk/otel';
import * as OtelTracer from '@effect/opentelemetry/OtelTracer';
import { context, trace } from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import {
	BasicTracerProvider,
	InMemorySpanExporter,
	type ReadableSpan,
	SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import { isStepCount, registerTelemetry } from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import { Effect, Layer, Stream } from 'effect';
import {
	afterAll,
	beforeAll,
	beforeEach,
	describe,
	expect,
	test,
	vi,
} from 'vitest';
import { Ai } from '#src/ai.ts';
import { generateText } from '#src/generate-text.ts';
import {
	textChunks,
	toolCallChunks,
	usage,
	weatherInput,
} from '#src/internal/tracing-test-helpers.ts';
import { streamText } from '#src/stream-text.ts';
import { tool } from '#src/tool.ts';

const exporter = new InMemorySpanExporter();
const provider = new BasicTracerProvider({
	spanProcessors: [new SimpleSpanProcessor(exporter)],
});
const contextManager = new AsyncLocalStorageContextManager();

beforeAll(() => {
	trace.setGlobalTracerProvider(provider);
	context.setGlobalContextManager(contextManager.enable());
	registerTelemetry(new OpenTelemetry());
});

afterAll(async () => {
	context.disable();
	trace.disable();
	await provider.shutdown();
});

beforeEach(() => {
	exporter.reset();
});

const runTraced = <A, E>(
	effect: Effect.Effect<A, E, Ai | OtelTracer.OtelTracer>,
) =>
	effect.pipe(
		Effect.provide(Layer.merge(Ai.layer, OtelTracer.layerGlobal)),
		Effect.runPromise,
	);

const findSpans = (predicate: (span: ReadableSpan) => boolean) =>
	exporter.getFinishedSpans().filter(predicate);

// The Effect tracer and @ai-sdk/otel both emit invoke_agent and execute_tool
// spans; their instrumentation scope tells them apart.
const byOperation = (operation: string) =>
	findSpans(
		(span) =>
			span.instrumentationScope.name === 'gen_ai' &&
			span.attributes['gen_ai.operation.name'] === operation,
	);

const wrapperSpan = (name: string) => {
	const found = findSpans((span) => span.name === name);
	if (found.length !== 1) {
		throw new Error(`expected one "${name}" span, got ${found.length}`);
	}
	return found[0] as ReadableSpan;
};

const isDescendantOf = (span: ReadableSpan, ancestor: ReadableSpan) => {
	const byId = new Map(
		exporter
			.getFinishedSpans()
			.map((candidate) => [candidate.spanContext().spanId, candidate]),
	);
	let parentId = span.parentSpanContext?.spanId;
	while (parentId !== undefined) {
		if (parentId === ancestor.spanContext().spanId) return true;
		parentId = byId.get(parentId)?.parentSpanContext?.spanId;
	}
	return false;
};

const toolThenTextStreamModel = () => {
	let calls = 0;
	return new MockLanguageModelV4({
		doStream: async () => {
			calls += 1;
			const chunks =
				calls === 1 ? toolCallChunks('weather') : textChunks('sunny');
			return {
				stream: new ReadableStream({
					start(controller) {
						for (const chunk of chunks) controller.enqueue(chunk);
						controller.close();
					},
				}) as never,
			};
		},
	});
};

const toolThenTextGenerateModel = () => {
	let calls = 0;
	return new MockLanguageModelV4({
		doGenerate: async () => {
			calls += 1;
			const first = calls === 1;
			return {
				content: first
					? [
							{
								type: 'tool-call' as const,
								toolCallId: 'call-1',
								toolName: 'weather',
								input: '{"city":"Paris"}',
							},
						]
					: [{ type: 'text' as const, text: 'sunny' }],
				finishReason: first
					? { unified: 'tool-calls' as const, raw: undefined }
					: { unified: 'stop' as const, raw: undefined },
				usage,
				warnings: [],
			};
		},
	});
};

const weatherTool = tool({
	inputSchema: weatherInput,
	execute: () => Effect.succeed('sunny'),
});

const sdkSpans = () => [
	...byOperation('invoke_agent'),
	...byOperation('chat'),
	...byOperation('execute_tool'),
];

const expectSdkSpansUnder = (wrapper: ReadableSpan) => {
	expect(byOperation('invoke_agent')).toHaveLength(1);
	expect(byOperation('chat')).toHaveLength(2);
	expect(byOperation('execute_tool')).toHaveLength(1);
	for (const span of sdkSpans()) {
		expect(span.spanContext().traceId).toBe(wrapper.spanContext().traceId);
		expect(isDescendantOf(span, wrapper)).toBe(true);
	}
};

describe('@ai-sdk/otel spans nest under the ai-effect wrapper span', () => {
	test('generateText with a tool', async () => {
		await runTraced(
			Effect.gen(function* () {
				const weather = yield* weatherTool;
				yield* generateText({
					model: toolThenTextGenerateModel(),
					prompt: 'weather?',
					tools: { weather },
					stopWhen: isStepCount(2),
				});
			}),
		);

		expectSdkSpansUnder(wrapperSpan('ai-effect.generateText'));
	});

	test('streamText with a tool, consumed after the call returned', async () => {
		await runTraced(
			Effect.gen(function* () {
				const weather = yield* weatherTool;
				const result = yield* streamText({
					model: toolThenTextStreamModel(),
					prompt: 'weather?',
					tools: { weather },
					stopWhen: isStepCount(2),
				});
				yield* Effect.sleep('20 millis');
				yield* Stream.runDrain(result.stream);
			}),
		);

		await vi.waitFor(() => wrapperSpan('ai-effect.streamText'));
		expectSdkSpansUnder(wrapperSpan('ai-effect.streamText'));
	});

	test('without a context manager the SDK spans start their own trace', async () => {
		context.disable();
		try {
			await runTraced(
				Effect.gen(function* () {
					const weather = yield* weatherTool;
					yield* generateText({
						model: toolThenTextGenerateModel(),
						prompt: 'weather?',
						tools: { weather },
						stopWhen: isStepCount(2),
					});
				}),
			);
		} finally {
			context.setGlobalContextManager(contextManager.enable());
		}

		const wrapper = wrapperSpan('ai-effect.generateText');
		const root = byOperation('invoke_agent')[0] as ReadableSpan;
		expect(root.spanContext().traceId).not.toBe(wrapper.spanContext().traceId);
	});
});
