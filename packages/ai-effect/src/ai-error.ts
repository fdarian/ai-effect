import { Data } from 'effect';

export class AiError extends Data.TaggedError('ai-effect/AiError')<{
	message: string;
	cause?: unknown;
}> {}
