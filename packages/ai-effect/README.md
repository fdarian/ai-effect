# ai-effect

Effect wrappers for the [AI SDK](https://ai-sdk.dev)'s `generateText`, `streamText` and `tool`. Callbacks and tool `execute` functions are Effects that can use your services, and calls fail with a typed `AiError` that says why they failed.

The package version tracks the AI SDK major: `ai-effect@7` wraps `ai@7`.

## Install

```sh
pnpm add ai-effect ai effect
```

`ai` and `effect` (v4) are peer dependencies, so your app shares one copy of each with this package.

## Setup

Every function requires the `Ai` service. Provide `Ai.layer` once, at the edge of your app, next to your other layers:

```ts
import { Ai, generateText } from 'ai-effect';
import { Effect } from 'effect';

const program = Effect.gen(function* () {
	const result = yield* generateText({
		model: 'openai/gpt-5',
		prompt: 'What is the capital of France?',
	});
	console.log(result.text);
});

program.pipe(Effect.provide(Ai.layer), Effect.runPromise);
```

Interrupting the `generateText` effect aborts the underlying request. An `abortSignal` you pass is still honored.

<details>
<summary>Why a layer?</summary>

Callbacks run after the call that registered them has returned (a stream's `onEnd` fires long after `streamText` hands you the stream). They need somewhere to run that lives longer than the call, so `Ai.layer` owns that place. Its lifetime is the layer's, which is why you provide it once for the whole app rather than per request. The functions never require `Scope`, so there is no `Effect.scoped` to place.

</details>

## Callbacks

Pass Effects where the AI SDK takes callbacks. They can use any service the call site has:

```ts
import { generateText } from 'ai-effect';
import { Effect } from 'effect';

const summary = generateText({
	model: 'openai/gpt-5',
	prompt: 'Summarize the water cycle.',
	onEnd: (event) => Effect.log(`Tokens used: ${event.usage.totalTokens}`),
});
```

## `streamText`

`streamText` returns the AI SDK's stream result right away. You don't need `Effect.scoped`:

```ts
import { streamText } from 'ai-effect';
import { Effect } from 'effect';

const chat = (prompt: string) =>
	Effect.gen(function* () {
		const result = yield* streamText({
			model: 'openai/gpt-5',
			prompt,
			onEnd: (event) =>
				Effect.log(`Streamed ${event.usage.totalTokens} tokens`),
		});
		return result.toTextStreamResponse();
	});
```

The callbacks run in the `Ai` layer, not in the effect that called `streamText`. An HTTP handler can return the response while the stream keeps going and `onEnd` still fires. When the `Ai` layer closes (e.g. at app shutdown), the stream is aborted and callbacks and tool executions still running are interrupted. An `abortSignal` you pass is honored too; either one stops the stream.

## `tool`

`tool` takes an Effect `execute` and an Effect `Schema` as `inputSchema` (any AI SDK schema works too). It returns an Effect, so yield it before passing the tool to `generateText` or `streamText`:

```ts
import { Ai, generateText, tool } from 'ai-effect';
import { Context, Effect, Layer, Schema } from 'effect';

class Weather extends Context.Service<Weather>()('app/Weather', {
	make: Effect.succeed({
		getTemperature: (city: string) => Effect.succeed(city.length + 10),
	}),
}) {
	static readonly layer = Layer.effect(Weather, Weather.make);
}

const program = Effect.gen(function* () {
	const weather = yield* tool({
		description: 'Get the current temperature for a city',
		inputSchema: Schema.Struct({
			city: Schema.String.annotate({ description: 'City name' }),
		}),
		execute: ({ city }) =>
			Effect.gen(function* () {
				const service = yield* Weather;
				const temperature = yield* service.getTemperature(city);
				return { city, temperature };
			}),
	});

	const result = yield* generateText({
		model: 'openai/gpt-5',
		prompt: 'What is the temperature in Paris?',
		tools: { weather },
	});
	return result.text;
});

program.pipe(
	Effect.provide(Layer.mergeAll(Ai.layer, Weather.layer)),
	Effect.runPromise,
);
```

<details>
<summary>Effect schemas outside <code>tool</code></summary>

`effectSchema` converts an Effect `Schema` into an AI SDK schema, for places that take one directly, such as structured output:

```ts
import { effectSchema, generateText } from 'ai-effect';
import { Output } from 'ai';
import { Effect, Schema } from 'effect';

const Recipe = Schema.Struct({
	name: Schema.String,
	steps: Schema.Array(Schema.String),
});

const recipe = generateText({
	model: 'openai/gpt-5',
	prompt: 'Give me a pancake recipe.',
	output: Output.object({ schema: effectSchema(Recipe) }),
}).pipe(Effect.map((result) => result.output));
```

</details>

## Errors

A failed `generateText` or `streamText` call fails the effect with one `AiError` (tag `ai-effect/AiError`). It has the `method` that failed and a `reason` describing why, plus `message` and `isRetryable` derived from the reason.

| `reason._tag`         | Cause                                                                                    | `isRetryable`               |
| --------------------- | ---------------------------------------------------------------------------------------- | --------------------------- |
| `RateLimitError`      | HTTP 429; `retryAfter` is a `Duration` parsed from the `retry-after` header when present | yes                         |
| `AuthenticationError` | HTTP 401 or 403                                                                          | no                          |
| `InvalidRequestError` | HTTP 400, 404, 409 or 422, or a bad prompt, argument, model or unsupported feature       | no                          |
| `ProviderError`       | HTTP 5xx, or a network failure with no status                                            | the SDK's own `isRetryable` |
| `InvalidOutputError`  | the model returned no output, or output that failed parsing, validation or tool lookup   | yes                         |
| `UnknownError`        | anything else, including errors thrown by your own code                                  | no                          |

Every reason has a `message` and a `cause` holding the raw SDK error, and a `statusCode` when the provider returned one. If the SDK retried and gave up, `cause` is the last underlying error rather than the SDK's retry wrapper.

Handle one reason with `Effect.catchReason`; other reasons stay in the error channel:

```ts
import { generateText } from 'ai-effect';
import { Effect } from 'effect';

const reply = generateText({
	model: 'openai/gpt-5',
	prompt: 'Hello!',
}).pipe(
	Effect.map((result) => result.text),
	Effect.catchReason('ai-effect/AiError', 'RateLimitError', (reason) =>
		Effect.logWarning('rate limited', reason.retryAfter).pipe(
			Effect.as('Fallback response'),
		),
	),
);
```

Use `isRetryable` to drive a retry policy:

```ts
const resilient = generateText({
	model: 'openai/gpt-5',
	prompt: 'Hello!',
}).pipe(
	Effect.retry({
		while: (error) => error.isRetryable,
		times: 3,
	}),
);
```

`streamText` returns the stream synchronously, so its error channel only reports failures that happen before the stream starts. Errors during streaming arrive as `error` parts on the stream or through `onError`, as raw SDK errors.

## License

Apache-2.0
