import * as AiSdk from 'ai';
import { describe, expect, test } from 'vitest';
import { AiError } from '#src/ai-error.ts';
import { reasonFromSdkError } from '#src/internal/ai-error-from-sdk.ts';

const findStatusCode = (error: unknown): number | undefined => {
	let current = error;
	while (current instanceof Error) {
		if ('statusCode' in current && typeof current.statusCode === 'number') {
			return current.statusCode;
		}
		current = current.cause;
	}
	return undefined;
};

describe('AiError', () => {
	const sdkError = new AiSdk.APICallError({
		message: 'rate limited',
		url: 'https://example.com',
		requestBodyValues: {},
		statusCode: 429,
		responseBody: '{"error":"slow down"}',
	});
	const error = new AiError({
		method: 'generateText',
		reason: reasonFromSdkError(sdkError),
	});

	test('cause is the raw SDK error', () => {
		expect(error.cause).toBe(sdkError);
		expect(error.cause).toBe(error.reason.cause);
	});

	test('cause is a getter, not an own property', () => {
		expect(Object.getOwnPropertyDescriptor(error, 'cause')).toBeUndefined();
	});

	test('a generic walker over .cause reaches statusCode and responseBody', () => {
		expect(findStatusCode(error)).toBe(429);
		expect((error.cause as AiSdk.APICallError).responseBody).toBe(
			'{"error":"slow down"}',
		);
	});
});
