import * as AiSdk from 'ai';
import { Duration } from 'effect';
import { afterEach, describe, expect, test, vi } from 'vitest';
import {
	AbortedError,
	AuthenticationError,
	InvalidOutputError,
	InvalidRequestError,
	ProviderError,
	RateLimitError,
	UnknownError,
} from '#src/ai-error.ts';
import { reasonFromSdkError } from '#src/internal/ai-error-from-sdk.ts';

const apiCallError = (options: {
	statusCode?: number;
	responseHeaders?: Record<string, string>;
	isRetryable?: boolean;
}) =>
	new AiSdk.APICallError({
		message: 'call failed',
		url: 'https://example.com',
		requestBodyValues: {},
		...options,
	});

describe('reasonFromSdkError', () => {
	afterEach(() => {
		vi.useRealTimers();
	});

	test('maps 429 to RateLimitError with retry-after seconds', () => {
		const error = apiCallError({
			statusCode: 429,
			responseHeaders: { 'retry-after': '30' },
		});

		const reason = reasonFromSdkError(error);

		expect(reason).toBeInstanceOf(RateLimitError);
		const rateLimit = reason as RateLimitError;
		expect(rateLimit.isRetryable).toBe(true);
		expect(rateLimit.statusCode).toBe(429);
		expect(rateLimit.message).toBe('call failed');
		expect(rateLimit.cause).toBe(error);
		expect(Duration.toMillis(rateLimit.retryAfter as Duration.Duration)).toBe(
			30_000,
		);
	});

	test('parses an HTTP-date retry-after', () => {
		vi.useFakeTimers();
		vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));

		const reason = reasonFromSdkError(
			apiCallError({
				statusCode: 429,
				responseHeaders: { 'Retry-After': 'Thu, 01 Jan 2026 00:01:00 GMT' },
			}),
		) as RateLimitError;

		expect(Duration.toMillis(reason.retryAfter as Duration.Duration)).toBe(
			60_000,
		);
	});

	test.each([
		['unparsable', { 'retry-after': 'soon' }],
		['missing', {}],
		['no headers', undefined],
	])('leaves retryAfter undefined when %s', (_name, responseHeaders) => {
		const reason = reasonFromSdkError(
			apiCallError({ statusCode: 429, responseHeaders }),
		) as RateLimitError;

		expect(reason.retryAfter).toBeUndefined();
	});

	test.each([401, 403])('maps %i to AuthenticationError', (statusCode) => {
		const reason = reasonFromSdkError(apiCallError({ statusCode }));

		expect(reason).toBeInstanceOf(AuthenticationError);
		expect(reason.isRetryable).toBe(false);
		expect(reason.statusCode).toBe(statusCode);
	});

	test.each([400, 404, 409, 422])(
		'maps %i to InvalidRequestError',
		(statusCode) => {
			const reason = reasonFromSdkError(apiCallError({ statusCode }));

			expect(reason).toBeInstanceOf(InvalidRequestError);
			expect(reason.isRetryable).toBe(false);
		},
	);

	test('maps 5xx to ProviderError using the SDK isRetryable', () => {
		const retryable = reasonFromSdkError(
			apiCallError({ statusCode: 500, isRetryable: true }),
		);
		const notRetryable = reasonFromSdkError(
			apiCallError({ statusCode: 503, isRetryable: false }),
		);

		expect(retryable).toBeInstanceOf(ProviderError);
		expect(retryable.statusCode).toBe(500);
		expect(retryable.isRetryable).toBe(true);
		expect(notRetryable).toBeInstanceOf(ProviderError);
		expect(notRetryable.isRetryable).toBe(false);
	});

	test('maps an APICallError without a status to ProviderError', () => {
		const reason = reasonFromSdkError(apiCallError({ isRetryable: true }));

		expect(reason).toBeInstanceOf(ProviderError);
		expect(reason.statusCode).toBeUndefined();
		expect(reason.isRetryable).toBe(true);
	});

	test('maps other statuses to UnknownError', () => {
		const reason = reasonFromSdkError(apiCallError({ statusCode: 418 }));

		expect(reason).toBeInstanceOf(UnknownError);
		expect(reason.statusCode).toBe(418);
		expect(reason.isRetryable).toBe(false);
	});

	test('unwraps RetryError to its last error', () => {
		const last = apiCallError({
			statusCode: 429,
			responseHeaders: { 'retry-after': '1' },
		});
		const retryError = new AiSdk.RetryError({
			message: 'Failed after 3 attempts',
			reason: 'maxRetriesExceeded',
			errors: [apiCallError({ statusCode: 500 }), last],
		});

		const reason = reasonFromSdkError(retryError);

		expect(reason).toBeInstanceOf(RateLimitError);
		expect(reason.cause).toBe(last);
	});

	test('keeps the RetryError as cause when there is no last error', () => {
		const retryError = new AiSdk.RetryError({
			message: 'Failed after 0 attempts',
			reason: 'abort',
			errors: [],
		});

		const reason = reasonFromSdkError(retryError);

		expect(reason).toBeInstanceOf(UnknownError);
		expect(reason.cause).toBe(retryError);
		expect(reason.message).toBe('Failed after 0 attempts');
	});

	test('maps an AbortError DOMException to AbortedError', () => {
		const error = new DOMException('The operation was aborted', 'AbortError');

		const reason = reasonFromSdkError(error);

		expect(reason).toBeInstanceOf(AbortedError);
		expect(reason.cause).toBe(error);
		expect(reason.message).toBe('The operation was aborted');
		expect(reason.isRetryable).toBe(false);
	});

	test('maps an Error named TimeoutError to AbortedError', () => {
		const error = new Error('timed out');
		error.name = 'TimeoutError';

		const reason = reasonFromSdkError(error);

		expect(reason).toBeInstanceOf(AbortedError);
		expect(reason.cause).toBe(error);
	});

	test('maps an APICallError caused by an abort to AbortedError, not ProviderError', () => {
		const abort = new DOMException('aborted', 'AbortError');
		const error = new AiSdk.APICallError({
			message: 'Cannot connect to API',
			url: 'https://example.com',
			requestBodyValues: {},
			cause: abort,
		});

		const reason = reasonFromSdkError(error);

		expect(reason).toBeInstanceOf(AbortedError);
		expect(reason.cause).toBe(error);
		expect(reason.message).toBe('Cannot connect to API');
	});

	test('maps a RetryError whose last error is an abort to AbortedError', () => {
		const abort = new DOMException('aborted', 'AbortError');
		const retryError = new AiSdk.RetryError({
			message: 'Failed after 2 attempts',
			reason: 'abort',
			errors: [apiCallError({ statusCode: 500 }), abort],
		});

		const reason = reasonFromSdkError(retryError);

		expect(reason).toBeInstanceOf(AbortedError);
		expect(reason.cause).toBe(abort);
	});

	test('maps NoOutputGeneratedError to InvalidOutputError', () => {
		const error = new AiSdk.NoOutputGeneratedError();

		const reason = reasonFromSdkError(error);

		expect(reason).toBeInstanceOf(InvalidOutputError);
		expect(reason.isRetryable).toBe(true);
		expect(reason.message).toBe(error.message);
		expect(reason.cause).toBe(error);
	});

	test('maps InvalidPromptError to InvalidRequestError', () => {
		const reason = reasonFromSdkError(
			new AiSdk.InvalidPromptError({ prompt: 'x', message: 'bad prompt' }),
		);

		expect(reason).toBeInstanceOf(InvalidRequestError);
		expect(reason.message).toBe('Invalid prompt: bad prompt');
	});

	test('maps LoadAPIKeyError to UnknownError', () => {
		const reason = reasonFromSdkError(
			new AiSdk.LoadAPIKeyError({ message: 'no key' }),
		);

		expect(reason).toBeInstanceOf(UnknownError);
		expect(reason.message).toBe('no key');
	});

	test('maps StreamProviderError to UnknownError with its status code', () => {
		const reason = reasonFromSdkError(
			new AiSdk.StreamProviderError({
				message: 'stream failed',
				statusCode: 503,
			}),
		);

		expect(reason).toBeInstanceOf(UnknownError);
		expect(reason.statusCode).toBe(503);
	});

	test('maps a plain Error to UnknownError', () => {
		const error = new Error('boom');

		const reason = reasonFromSdkError(error);

		expect(reason).toBeInstanceOf(UnknownError);
		expect(reason.message).toBe('boom');
		expect(reason.cause).toBe(error);
	});

	test('stringifies non-Error throws', () => {
		const reason = reasonFromSdkError('nope');

		expect(reason).toBeInstanceOf(UnknownError);
		expect(reason.message).toBe('nope');
		expect(reason.cause).toBe('nope');
	});
});
