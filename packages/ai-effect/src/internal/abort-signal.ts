import { Effect, Exit, Stream } from 'effect';

export function combineAbortSignals(
	...signals: ReadonlyArray<AbortSignal | undefined>
): AbortSignal {
	const defined = signals.filter((signal) => signal !== undefined);
	const first = defined[0];
	if (first !== undefined && defined.length === 1) return first;
	return AbortSignal.any(defined);
}

// `Stream.take` and friends close the upstream with a success Exit, so only a
// real interruption aborts the controller.
export const abortOnInterrupt =
	(controller: AbortController) =>
	<A, E, R>(stream: Stream.Stream<A, E, R>) =>
		Stream.onExit(stream, (exit) =>
			Exit.hasInterrupts(exit)
				? Effect.sync(() => controller.abort())
				: Effect.void,
		);
