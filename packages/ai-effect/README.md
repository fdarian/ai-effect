# ai-effect

Effect wrappers for the [AI SDK](https://ai-sdk.dev)'s `generateText`, `streamText`, `tool`, `ToolLoopAgent` and the UI message stream helpers. Callbacks and tool `execute` functions are Effects that can use your services, and calls fail with a typed `AiError` that says why they failed.

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

`streamText` returns right away with an Effect-typed view of the AI SDK's stream result: `stream`, `textStream` and the other streams are `Stream<_, AiError>`, and `text`, `usage`, `output` and the other promise fields are `Effect<_, AiError>`. You don't need `Effect.scoped`:

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
		return result.raw.toTextStreamResponse();
	});
```

The callbacks run in the `Ai` layer, not in the effect that called `streamText`. An HTTP handler can return the response while the stream keeps going and `onEnd` still fires. When the `Ai` layer closes (e.g. at app shutdown), the stream is aborted and callbacks and tool executions still running are interrupted. An `abortSignal` you pass is honored too; either one stops the stream. Interrupting a fiber that consumes one of the result's streams or effects aborts the call too, but stopping early (`Stream.take`) does not. The SDK's own result, with its response helpers, is on `result.raw`.

## UI message streams

For chat UIs, `createUIMessageStream` is the Effect version of the SDK's function of the same name. `execute` is an Effect, and `writer.merge` takes an Effect `Stream`, such as the result of `toUIMessageStream()` on a `streamText` result:

```ts
import { createUIMessageStream, streamText } from 'ai-effect';
import { Effect } from 'effect';

const chatStream = (prompt: string) =>
	createUIMessageStream({
		execute: ({ writer }) =>
			Effect.gen(function* () {
				const result = yield* streamText({ model: 'openai/gpt-5', prompt });
				yield* writer.merge(result.toUIMessageStream());
			}),
	});
```

The result is a `Stream<_, AiError>` that calls the SDK each time it runs. If `execute` fails, the stream ends with an `error` chunk rather than failing.

`createAgentUIStream` does the same for a `ToolLoopAgent` and the `UIMessage`s a client sends. It returns an Effect of the stream, so invalid messages fail before anything is streamed and your handler can still answer with an error status:

```ts
import { createAgentUIStream, ToolLoopAgent } from 'ai-effect';
import { Effect } from 'effect';

const chat = (uiMessages: unknown[]) =>
	Effect.gen(function* () {
		const agent = yield* ToolLoopAgent.make({ model: 'openai/gpt-5' });
		return yield* createAgentUIStream({ agent, uiMessages });
	});
```

There are no wrappers for the SDK's response helpers. Convert the stream with `Stream.toReadableStreamEffect` and pass it to `createUIMessageStreamResponse` (or `pipeUIMessageStreamToResponse` for Node responses):

```ts
import { createUIMessageStreamResponse } from 'ai';
import { Effect, Stream } from 'effect';

const handler = (uiMessages: unknown[]) =>
	Effect.gen(function* () {
		const stream = yield* chat(uiMessages);
		const readable = yield* Stream.toReadableStreamEffect(stream);
		return createUIMessageStreamResponse({ stream: readable });
	});
```

<details>
<summary>Caveats</summary>

The stream from `createAgentUIStream` wraps a single SDK stream, so it can be run once. Interrupting a fiber that consumes it aborts the model call. Invalid `uiMessages` fail with `InvalidRequestError`, which is not retryable.

</details>

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

| `reason._tag`                   | Cause                                                                                    | `isRetryable`               |
| ------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------- |
| `ai-effect/RateLimitError`      | HTTP 429; `retryAfter` is a `Duration` parsed from the `retry-after` header when present | yes                         |
| `ai-effect/AuthenticationError` | HTTP 401 or 403                                                                          | no                          |
| `ai-effect/InvalidRequestError` | HTTP 400, 404, 409 or 422, or a bad prompt, argument, model or unsupported feature       | no                          |
| `ai-effect/ProviderError`       | HTTP 5xx, or a network failure with no status                                            | the SDK's own `isRetryable` |
| `ai-effect/InvalidOutputError`  | the model returned no output, or output that failed parsing, validation or tool lookup   | yes                         |
| `ai-effect/UnknownError`        | anything else, including errors thrown by your own code                                  | no                          |

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
	Effect.catchReason(
		'ai-effect/AiError',
		'ai-effect/RateLimitError',
		(reason) =>
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

`streamText` returns the stream synchronously, so its own error channel only reports failures that happen before the stream starts. Errors during streaming fail the result's streams and effects with an `AiError` too. `stream` is the exception: it emits `error` and `abort` parts as data, which `textStream`, `partialOutputStream` and `elementStream` turn into failures. `onError` still receives the raw SDK error.

## License

Apache-2.0
