import { Schema } from 'effect';

export function effectSchema<A, I>(schema: Schema.Codec<A, I>) {
	return Schema.toStandardSchemaV1(Schema.toStandardJSONSchemaV1(schema));
}
