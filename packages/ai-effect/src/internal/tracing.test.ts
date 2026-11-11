import { Effect, Exit, Option } from 'effect';
import { describe, expect, test, vi } from 'vitest';
import { AbortedError, AiError, UnknownError } from '#src/ai-error.ts';
import {
	attributesOf,
	endedExit,
	makeTestTracer,
	spanNamed,
} from '#src/internal/test-tracer.ts';
import {
	composeCallbacks,
	errorTypeOf,
	makeCallSpan,
	mapFinishReason,
	mapProviderName,
	streamLifecycleCallbacks,
	textRequestAttributes,
} from '#src/internal/tracing.ts';

describe('mapProviderName', () => {
	test.each([
		['openai.chat', 'openai'],
		['openai', 'openai'],
		['anthropic.messages', 'anthropic'],
		['google.vertex.chat', 'gcp.vertex_ai'],
		['google-vertex', 'gcp.vertex_ai'],
		['google.generative-ai', 'gcp.gemini'],
		['google', 'gcp.gemini'],
		['amazon-bedrock', 'aws.bedrock'],
		['azure-openai.chat', 'azure.ai.openai'],
		['azure.chat', 'azure.ai.inference'],
		['mistral.chat', 'mistral'],
		['xai.responses', 'x_ai'],
		['Anthropic.Messages', 'anthropic'],
	])('%s -> %s', (provider, expected) => {
		expect(mapProviderName(provider)).toBe(expected);
	});

	test('passes unknown providers through verbatim', () => {
		expect(mapProviderName('gateway')).toBe('gateway');
		expect(mapProviderName('openaiish')).toBe('openaiish');
	});
});

test('mapFinishReason maps the SDK reasons that differ from OTel', () => {
	expect(mapFinishReason('tool-calls')).toBe('tool_call');
	expect(mapFinishReason('content-filter')).toBe('content_filter');
	expect(mapFinishReason('stop')).toBe('stop');
	expect(mapFinishReason('length')).toBe('length');
});

describe('errorTypeOf', () => {
	test('uses the AiError reason tag', () => {
		const error = new AiError({
			method: 'generateText',
			reason: new UnknownError({ message: 'x', cause: undefined }),
		});
		expect(errorTypeOf(error)).toBe('ai-effect/UnknownError');
	});

	test('prefers a string _tag over the Error name', () => {
		expect(errorTypeOf({ _tag: 'Boom' })).toBe('Boom');
		expect(errorTypeOf(new TypeError('x'))).toBe('TypeError');
	});

	test('omits unnamed failures', () => {
		expect(errorTypeOf('string')).toBeUndefined();
		expect(errorTypeOf(undefined)).toBeUndefined();
	});
});

describe('makeCallSpan', () => {
	test('finish is synchronous and the first exit wins', () => {
		const tracer = makeTestTracer();
		return Effect.gen(function* () {
			const call = yield* makeCallSpan('op', { kind: 'client' });
			call.finish(Exit.fail(new TypeError('first')));
			call.finish(Exit.void);

			const span = spanNamed(tracer.spans, 'op');
			expect(span.kind).toBe('client');
			expect(Exit.isFailure(endedExit(span))).toBe(true);
			expect(attributesOf(span)['error.type']).toBe('TypeError');
		}).pipe(Effect.provide(tracer.layer), Effect.runPromise);
	});

	test('drops undefined attributes', () => {
		const tracer = makeTestTracer();
		return Effect.gen(function* () {
			const call = yield* makeCallSpan('op', {
				attributes: { kept: 1, dropped: undefined },
			});
			call.annotate({ later: 'x', skipped: undefined });
			call.finish(Exit.void);

			expect(attributesOf(spanNamed(tracer.spans, 'op'))).toEqual({
				kept: 1,
				later: 'x',
			});
		}).pipe(Effect.provide(tracer.layer), Effect.runPromise);
	});

	test('uses the current span as parent', () => {
		const tracer = makeTestTracer();
		return Effect.gen(function* () {
			const call = yield* makeCallSpan('child', {});
			call.finish(Exit.void);
			const parent = spanNamed(tracer.spans, 'parent');
			expect(spanNamed(tracer.spans, 'child').parent).toMatchObject({
				value: { spanId: parent.spanId },
			});
		}).pipe(
			Effect.withSpan('parent'),
			Effect.provide(tracer.layer),
			Effect.runPromise,
		);
	});
});

describe('textRequestAttributes', () => {
	test('maps set request settings and skips unset ones', () => {
		expect(
			textRequestAttributes(
				{ temperature: 0.2, topP: 0.9, maxOutputTokens: 50, seed: 1 },
				{ stream: true, agentName: 'helper' },
			),
		).toEqual({
			'gen_ai.operation.name': 'invoke_agent',
			'gen_ai.agent.name': 'helper',
			'gen_ai.request.temperature': 0.2,
			'gen_ai.request.top_p': 0.9,
			'gen_ai.request.max_tokens': 50,
			'gen_ai.request.seed': 1,
			'gen_ai.request.stream': true,
		});
	});

	test('omits the stream flag for non-streams', () => {
		expect(textRequestAttributes({}, { stream: false })).toEqual({
			'gen_ai.operation.name': 'invoke_agent',
		});
	});
});

describe('composeCallbacks', () => {
	test('runs tracing first, then the user callback, and returns its result', async () => {
		const order: string[] = [];
		const composed = composeCallbacks(
			{
				onEnd: async () => {
					order.push('user');
					return 'user-result';
				},
			},
			{ onEnd: () => order.push('trace') },
			{},
		);

		const onEnd = composed.onEnd as (event: unknown) => Promise<string>;
		expect(await onEnd({})).toBe('user-result');
		expect(order).toEqual(['trace', 'user']);
	});

	test('falls back to the alias when only the alias is set, and clears it', () => {
		const user = vi.fn();
		const composed = composeCallbacks(
			{ onFinish: user },
			{ onEnd: () => undefined },
			{ onEnd: 'onFinish' },
		);

		expect(composed.onFinish).toBeUndefined();
		(composed.onEnd as (event: unknown) => void)('event');
		expect(user).toHaveBeenCalledWith('event');
	});

	test('canonical wins over the alias, like the SDK', () => {
		const canonical = vi.fn();
		const alias = vi.fn();
		const composed = composeCallbacks(
			{ onEnd: canonical, onFinish: alias },
			{ onEnd: () => undefined },
			{ onEnd: 'onFinish' },
		);
		(composed.onEnd as (event: unknown) => void)('event');
		expect(canonical).toHaveBeenCalled();
		expect(alias).not.toHaveBeenCalled();
	});

	test('uses the fallback only when the user set nothing', () => {
		const fallback = vi.fn();
		const none = composeCallbacks(
			{},
			{ onError: () => undefined },
			{},
			{
				onError: fallback,
			},
		);
		(none.onError as (event: unknown) => void)('e');
		expect(fallback).toHaveBeenCalledWith('e');

		const user = vi.fn();
		const some = composeCallbacks(
			{ onError: user },
			{ onError: () => undefined },
			{},
			{ onError: fallback },
		);
		(some.onError as (event: unknown) => void)('e2');
		expect(user).toHaveBeenCalledWith('e2');
		expect(fallback).toHaveBeenCalledTimes(1);
	});
});

describe('streamLifecycleCallbacks', () => {
	const setup = (options: {
		signal: AbortSignal;
		userSignal?: AbortSignal;
		errorMayBeRetried?: boolean;
	}) => {
		const tracer = makeTestTracer();
		return Effect.gen(function* () {
			const call = yield* makeCallSpan('stream', {});
			const callbacks = streamLifecycleCallbacks(call, {
				method: 'streamText',
				signal: options.signal,
				userSignal: options.userSignal,
				errorMayBeRetried: options.errorMayBeRetried ?? false,
			});
			return { callbacks, span: spanNamed(tracer.spans, 'stream') };
		}).pipe(Effect.provide(tracer.layer), Effect.runPromise);
	};

	test('a user abort fails with an AbortedError reason', async () => {
		const user = new AbortController();
		const combined = AbortSignal.any([user.signal]);
		const result = await setup({ signal: combined, userSignal: user.signal });

		user.abort(new Error('stop'));

		const error = Option.getOrUndefined(
			Exit.findErrorOption(endedExit(result.span)),
		);
		expect(error).toBeInstanceOf(AiError);
		expect((error as AiError).reason).toBeInstanceOf(AbortedError);
		expect(attributesOf(result.span)['error.type']).toBe(
			'ai-effect/AbortedError',
		);
	});

	test('any other abort is an interrupt', async () => {
		const controller = new AbortController();
		const result = await setup({ signal: controller.signal });

		controller.abort();

		const exit = endedExit(result.span);
		expect(Exit.hasInterrupts(exit)).toBe(true);
		expect(attributesOf(result.span)['error.type']).toBeUndefined();
	});

	test('an already-aborted signal ends the span at setup', async () => {
		const controller = new AbortController();
		controller.abort();
		const result = await setup({ signal: controller.signal });
		expect(Exit.hasInterrupts(endedExit(result.span))).toBe(true);
	});

	test('onEnd after the span has failed does not overwrite the failure', async () => {
		const result = await setup({ signal: new AbortController().signal });
		result.callbacks.onError({ error: new Error('real') });
		result.callbacks.onEnd({
			finishReason: 'error',
			totalUsage: {
				inputTokens: undefined,
				outputTokens: undefined,
				inputTokenDetails: {
					cacheReadTokens: undefined,
					cacheWriteTokens: undefined,
				},
				outputTokenDetails: { reasoningTokens: undefined },
			},
			finalStep: { response: { id: 'r', modelId: 'm' } },
		});
		expect(Exit.isFailure(endedExit(result.span))).toBe(true);
		expect(attributesOf(result.span)['error.type']).toBe(
			'ai-effect/UnknownError',
		);
	});

	describe('when the SDK may retry an error', () => {
		const endEvent = (finishReason: string) => ({
			finishReason,
			totalUsage: {
				inputTokens: undefined,
				outputTokens: undefined,
				inputTokenDetails: {
					cacheReadTokens: undefined,
					cacheWriteTokens: undefined,
				},
				outputTokenDetails: { reasoningTokens: undefined },
			},
			finalStep: { response: { id: 'r', modelId: 'm' } },
		});
		const retryable = () =>
			setup({
				signal: new AbortController().signal,
				errorMayBeRetried: true,
			});

		test('onError only records the error', async () => {
			const result = await retryable();
			result.callbacks.onError({ error: new Error('transient') });
			expect(result.span.status._tag).toBe('Started');
		});

		test('onEnd with an error finish reason fails with the recorded error', async () => {
			const result = await retryable();
			result.callbacks.onError({ error: new TypeError('first') });
			result.callbacks.onError({ error: new Error('last') });
			result.callbacks.onEnd(endEvent('error'));
			expect(Exit.isFailure(endedExit(result.span))).toBe(true);
			expect(attributesOf(result.span)['error.type']).toBe(
				'ai-effect/UnknownError',
			);
		});

		test('onEnd with any other finish reason succeeds, the retry recovered', async () => {
			const result = await retryable();
			result.callbacks.onError({ error: new Error('transient') });
			result.callbacks.onEnd(endEvent('stop'));
			expect(Exit.isSuccess(endedExit(result.span))).toBe(true);
		});

		test('onStepError still ends the span as a failure', async () => {
			const result = await retryable();
			result.callbacks.onStepError({ error: new Error('step') });
			expect(Exit.isFailure(endedExit(result.span))).toBe(true);
		});
	});
});
