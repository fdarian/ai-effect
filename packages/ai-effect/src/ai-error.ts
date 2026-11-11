import { Data, type Duration } from 'effect';

type ReasonFields = {
	message: string;
	/** The raw SDK error, after `RetryError` unwrapping. */
	cause: unknown;
	statusCode?: number;
};

export class RateLimitError extends Data.TaggedClass(
	'ai-effect/RateLimitError',
)<ReasonFields & { retryAfter?: Duration.Duration }> {
	get isRetryable() {
		return true;
	}
}

export class AuthenticationError extends Data.TaggedClass(
	'ai-effect/AuthenticationError',
)<ReasonFields> {
	get isRetryable() {
		return false;
	}
}

export class InvalidRequestError extends Data.TaggedClass(
	'ai-effect/InvalidRequestError',
)<ReasonFields> {
	get isRetryable() {
		return false;
	}
}

export class ProviderError extends Data.TaggedClass('ai-effect/ProviderError')<
	ReasonFields & { isRetryable: boolean }
> {}

export class InvalidOutputError extends Data.TaggedClass(
	'ai-effect/InvalidOutputError',
)<ReasonFields> {
	get isRetryable() {
		return true;
	}
}

export class AbortedError extends Data.TaggedClass(
	'ai-effect/AbortedError',
)<ReasonFields> {
	get isRetryable() {
		return false;
	}
}

export class UnknownError extends Data.TaggedClass(
	'ai-effect/UnknownError',
)<ReasonFields> {
	get isRetryable() {
		return false;
	}
}

export type AiErrorReason =
	| RateLimitError
	| AuthenticationError
	| InvalidRequestError
	| ProviderError
	| InvalidOutputError
	| AbortedError
	| UnknownError;

export class AiError extends Data.TaggedError('ai-effect/AiError')<{
	method:
		| 'generateText'
		| 'streamText'
		| 'embed'
		| 'embedMany'
		| 'rerank'
		| 'ToolLoopAgent.generate'
		| 'ToolLoopAgent.stream'
		| 'createUIMessageStream';
	reason: AiErrorReason;
}> {
	override get message() {
		return `${this.method}: ${this.reason.message}`;
	}

	get isRetryable() {
		return this.reason.isRetryable;
	}
}
