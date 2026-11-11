import * as AiSdk from 'ai';
import { Duration } from 'effect';
import {
	type AiErrorReason,
	AuthenticationError,
	InvalidOutputError,
	InvalidRequestError,
	ProviderError,
	RateLimitError,
	UnknownError,
} from '#src/ai-error.ts';

const INVALID_REQUEST_STATUSES = new Set([400, 404, 409, 422]);

const INVALID_REQUEST_ERRORS = [
	AiSdk.InvalidPromptError,
	AiSdk.InvalidArgumentError,
	AiSdk.UnsupportedFunctionalityError,
	AiSdk.NoSuchModelError,
];

const INVALID_OUTPUT_ERRORS = [
	AiSdk.NoOutputGeneratedError,
	AiSdk.NoObjectGeneratedError,
	AiSdk.TypeValidationError,
	AiSdk.JSONParseError,
	AiSdk.InvalidResponseDataError,
	AiSdk.EmptyResponseBodyError,
	AiSdk.NoContentGeneratedError,
	AiSdk.InvalidToolInputError,
	AiSdk.NoSuchToolError,
];

const messageOf = (error: unknown) =>
	error instanceof Error ? error.message : String(error);

/** `Retry-After` is either delay-seconds or an HTTP date. */
function parseRetryAfter(
	headers: Record<string, string> | undefined,
): Duration.Duration | undefined {
	if (headers === undefined) {
		return undefined;
	}
	const entry = Object.entries(headers).find(
		([name]) => name.toLowerCase() === 'retry-after',
	);
	if (entry === undefined) {
		return undefined;
	}
	const value = entry[1].trim();
	if (/^\d+$/.test(value)) {
		return Duration.seconds(Number(value));
	}
	const date = Date.parse(value);
	if (Number.isNaN(date)) {
		return undefined;
	}
	return Duration.millis(Math.max(0, date - Date.now()));
}

function reasonFromApiCallError(error: AiSdk.APICallError): AiErrorReason {
	const fields = {
		message: error.message,
		cause: error,
		statusCode: error.statusCode,
	};
	const status = error.statusCode;
	if (status === undefined || status >= 500) {
		return new ProviderError({ ...fields, isRetryable: error.isRetryable });
	}
	if (status === 429) {
		return new RateLimitError({
			...fields,
			retryAfter: parseRetryAfter(error.responseHeaders),
		});
	}
	if (status === 401 || status === 403) {
		return new AuthenticationError(fields);
	}
	if (INVALID_REQUEST_STATUSES.has(status)) {
		return new InvalidRequestError(fields);
	}
	return new UnknownError(fields);
}

export function reasonFromSdkError(error: unknown): AiErrorReason {
	const unwrapped =
		AiSdk.RetryError.isInstance(error) && error.lastError !== undefined
			? error.lastError
			: error;

	if (AiSdk.APICallError.isInstance(unwrapped)) {
		return reasonFromApiCallError(unwrapped);
	}
	const fields = { message: messageOf(unwrapped), cause: unwrapped };
	if (
		INVALID_REQUEST_ERRORS.some((sdkError) => sdkError.isInstance(unwrapped))
	) {
		return new InvalidRequestError(fields);
	}
	if (
		INVALID_OUTPUT_ERRORS.some((sdkError) => sdkError.isInstance(unwrapped))
	) {
		return new InvalidOutputError(fields);
	}
	if (AiSdk.StreamProviderError.isInstance(unwrapped)) {
		return new UnknownError({ ...fields, statusCode: unwrapped.statusCode });
	}
	return new UnknownError(fields);
}
