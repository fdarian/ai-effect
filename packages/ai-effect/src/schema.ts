import * as AiSdk from 'ai';
import { JsonSchema, Schema } from 'effect';

// Not `Schema.toStandardSchemaV1`: the AI SDK would run its JSON through
// `addAdditionalPropertiesToJsonSchema`, which closes every object, including
// open records like `Record(String, Unknown)`. A `jsonSchema` passes through
// `asSchema` untouched, so Effect alone decides which objects are closed.
export function effectSchema<A, I>(schema: Schema.Codec<A, I>) {
	const standard = Schema.toStandardSchemaV1(schema);

	return AiSdk.jsonSchema<A>(
		() => {
			const document = JsonSchema.toDocumentDraft07(
				Schema.toJsonSchemaDocument(schema, { onExcessProperty: 'error' }),
			);
			const jsonSchema: JsonSchema.JsonSchema = {
				$schema: JsonSchema.META_SCHEMA_URI_DRAFT_07,
				...document.schema,
			};
			if (Object.keys(document.definitions).length > 0) {
				jsonSchema.definitions = document.definitions;
			}
			return jsonSchema as AiSdk.JSONSchema7;
		},
		{
			validate: async (value) => {
				const result = await standard['~standard'].validate(value);
				return 'value' in result
					? { success: true, value: result.value }
					: {
							success: false,
							error: new AiSdk.TypeValidationError({
								value,
								cause: result.issues,
							}),
						};
			},
		},
	);
}
