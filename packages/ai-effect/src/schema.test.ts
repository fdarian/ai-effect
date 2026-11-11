import * as AiSdk from 'ai';
import { Effect, Schema } from 'effect';
import { describe, expect, test } from 'vitest';
import { effectSchema } from '#src/schema.ts';

const draft07 = 'http://json-schema.org/draft-07/schema#';

const validate = (schema: AiSdk.Schema<unknown>, value: unknown) => {
	const run = schema.validate;
	if (run === undefined) {
		throw new Error('effectSchema must provide validate');
	}
	return Effect.promise(async () => run(value));
};

describe('effectSchema', () => {
	test('closes structs but keeps free-form records open', () => {
		const schema = Schema.Struct({
			a: Schema.String,
			ctx: Schema.Record(Schema.String, Schema.Unknown),
		});

		expect(AiSdk.asSchema(effectSchema(schema)).jsonSchema).toEqual({
			$schema: draft07,
			type: 'object',
			properties: {
				a: { type: 'string' },
				ctx: { type: 'object' },
			},
			required: ['a', 'ctx'],
			additionalProperties: false,
		});
	});

	test('keeps the value schema of typed records', () => {
		const schema = Schema.Record(Schema.String, Schema.Finite);

		expect(AiSdk.asSchema(effectSchema(schema)).jsonSchema).toEqual({
			$schema: draft07,
			type: 'object',
			additionalProperties: { type: 'number' },
		});
	});

	test('validate returns the decoded value', () =>
		Effect.gen(function* () {
			const schema = AiSdk.asSchema(
				effectSchema(Schema.Struct({ n: Schema.NumberFromString })),
			);
			const result = yield* validate(schema, { n: '4' });

			expect(result).toEqual({ success: true, value: { n: 4 } });
		}).pipe(Effect.runPromise));

	test('validate fails with a TypeValidationError', () =>
		Effect.gen(function* () {
			const schema = AiSdk.asSchema(
				effectSchema(Schema.Struct({ n: Schema.Number })),
			);
			const result = yield* validate(schema, { n: 'x' });

			expect(result.success).toBe(false);
			expect(!result.success && result.error).toBeInstanceOf(
				AiSdk.TypeValidationError,
			);
		}).pipe(Effect.runPromise));
});
