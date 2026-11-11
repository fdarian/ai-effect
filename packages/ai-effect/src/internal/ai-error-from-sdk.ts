import * as AiSdk from 'ai';
import { Duration } from 'effect';
import {
	AbortedError,
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

const ABORT_ERROR_NAMES = new Set([
	'AbortError',
	'ResponseAborted',
	'TimeoutError',
]);

const MAX_ABORT_CAUSE_DEPTH = 5;

const isAbortLike = (error: unknown) =>
	(error instanceof Error ||
		(typeof DOMException === 'function' && error instanceof DOMException)) &&
	ABORT_ERROR_NAMES.has(error.name);

/** Mirrors ai's unexported `isCausedByAbort`; providers wrap aborts as the `cause` of an `APICallError`. */
function isCausedByAbort(error: unknown): boolean {
	let current = error;
	for (let depth = 0; depth < MAX_ABORT_CAUSE_DEPTH; depth++) {
		if (isAbortLike(current)) {
			return true;
		}
		if (!(current instanceof Error) || current.cause === undefined) {
			return false;
		}
		current = current.cause;
	}
	return false;
}

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

	if (isCausedByAbort(unwrapped)) {
		return new AbortedError({
			message: messageOf(unwrapped),
			cause: unwrapped,
		});
	}
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
