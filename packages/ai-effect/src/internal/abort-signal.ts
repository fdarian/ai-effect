export function combineAbortSignals(
	...signals: ReadonlyArray<AbortSignal | undefined>
): AbortSignal {
	const defined = signals.filter((signal) => signal !== undefined);
	const first = defined[0];
	if (first !== undefined && defined.length === 1) return first;
	return AbortSignal.any(defined);
}
