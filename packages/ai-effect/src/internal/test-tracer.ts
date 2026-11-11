import { Layer, Option, Tracer } from 'effect';

/** Collects every span the Effect runtime creates, so tests can assert on them. */
export function makeTestTracer() {
	const spans: Array<Tracer.NativeSpan> = [];
	const tracer = Tracer.make({
		span: (options) => {
			const span = new Tracer.NativeSpan(options);
			spans.push(span);
			return span;
		},
	});
	return { spans, layer: Layer.succeed(Tracer.Tracer, tracer) };
}

export function spanNamed(
	spans: ReadonlyArray<Tracer.NativeSpan>,
	name: string,
) {
	const found = spans.filter((span) => span.name === name);
	const span = found.at(0);
	if (span === undefined || found.length > 1) {
		throw new Error(
			`expected exactly one span named "${name}", got ${found.length}: ${spans.map((s) => s.name).join(', ')}`,
		);
	}
	return span;
}

export function spansNamed(
	spans: ReadonlyArray<Tracer.NativeSpan>,
	name: string,
) {
	return spans.filter((span) => span.name === name);
}

export function endedExit(span: Tracer.NativeSpan) {
	if (span.status._tag !== 'Ended') {
		throw new Error(`span "${span.name}" has not ended`);
	}
	return span.status.exit;
}

export const parentSpanId = (span: Tracer.NativeSpan) =>
	Option.getOrUndefined(span.parent)?.spanId;

export const attributesOf = (span: Tracer.NativeSpan) =>
	Object.fromEntries(span.attributes);
