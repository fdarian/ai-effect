import { jsonSchema as aiJsonSchema } from 'ai';
import { JsonSchema, Result, Schema } from 'effect';

/**
 * Flattens a v4 JSON Schema document into the single draft-07 root object the
 * AI SDK expects, hoisting generated definitions onto `$defs`.
 */
function toDraft07(schema: Schema.Constraint) {
	const document = JsonSchema.toDocumentDraft07(
		Schema.toJsonSchemaDocument(schema),
	);
	if (Object.keys(document.definitions).length === 0) return document.schema;
	return { ...document.schema, $defs: document.definitions };
}

export function effectSchema<A, I>(schema: Schema.Codec<A, I>) {
	const decode = Schema.decodeUnknownResult(schema);
	return aiJsonSchema<A>(toDraft07(schema), {
		validate: (value) => {
			const result = decode(value);
			if (Result.isSuccess(result)) {
				return { success: true as const, value: result.success };
			}
			return {
				success: false as const,
				error: new Error(String(result.failure)),
			};
		},
	});
}

export const describe =
	(d: string) =>
	<S extends Schema.Top>(self: S) =>
		self.annotate({ description: d });
