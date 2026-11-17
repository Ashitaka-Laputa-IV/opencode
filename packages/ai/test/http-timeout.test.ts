import { describe, expect } from "bun:test"
import { Deferred, Effect, Fiber } from "effect"
import * as TestClock from "effect/testing/TestClock"
import { ai } from "../src/effect.js"
import { Auth, LLMClient, mergeTransport } from "../src/route.js"
import { AnthropicMessages } from "../src/protocols/anthropic-messages.js"
import { configure } from "../src/providers/openai.js"
import { dynamicResponse } from "./lib/http.js"
import { deltaChunk, finishChunk } from "./lib/openai-chunks.js"
import { sseEvents, sseNamedEvent, sseRaw } from "./lib/sse.js"
import { it } from "./lib/effect.js"

const model = configure({ baseURL: "https://api.openai.test/v1/", apiKey: "test" }).chat("gpt-4.1-mini")
const SSE = { headers: { "content-type": "text/event-stream" } }

const silentServer = dynamicResponse(() => Effect.never)

const slowHeadersServer = dynamicResponse((input) =>
  Effect.sleep("10 minutes").pipe(
    Effect.as(input.respond(sseEvents(deltaChunk({ role: "assistant", content: "Hi" }), finishChunk("stop")), SSE)),
  ),
)

// Sends one chunk, then stalls until `resume` releases the rest of the body.
const stalledServer = Effect.gen(function* () {
  const stalled = yield* Deferred.make<void>()
  let resume = () => {}
  const released = new Promise<void>((resolve) => {
    resume = resolve
  })
  const encoder = new TextEncoder()
  const layer = dynamicResponse((input) =>
    Effect.sync(() =>
      input.respond(
        new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(encoder.encode(sseRaw(`data: ${JSON.stringify(deltaChunk({ content: "Hi" }))}`)))
          },
          async pull(controller) {
            Deferred.doneUnsafe(stalled, Effect.void)
            await released
            controller.enqueue(encoder.encode(sseEvents(finishChunk("stop"))))
            controller.close()
          },
        }),
        SSE,
      ),
    ),
  )
  return { layer, stalled, resume: () => resume() }
})

const anthropic = AnthropicMessages.route
  .with({ endpoint: { baseURL: "https://api.anthropic.test/v1/" }, auth: Auth.header("x-api-key", "test") })
  .model({ id: "claude-sonnet-4-5" })

// Streams only what the test pushes, so it controls what arrives between clock adjustments.
const pushedServer = Effect.gen(function* () {
  const opened = yield* Deferred.make<ReadableStreamDefaultController<Uint8Array>>()
  const encoder = new TextEncoder()
  const layer = dynamicResponse((input) =>
    Effect.sync(() =>
      input.respond(
        new ReadableStream<Uint8Array>({
          start(controller) {
            Deferred.doneUnsafe(opened, Effect.succeed(controller))
          },
        }),
        SSE,
      ),
    ),
  )
  const push = (...events: ReadonlyArray<string>) =>
    Effect.gen(function* () {
      const controller = yield* Deferred.await(opened)
      controller.enqueue(encoder.encode(sseRaw(...events)))
      yield* Effect.promise(() => new Promise((resolve) => setTimeout(resolve, 0)))
    })
  return { layer, push, close: Deferred.await(opened).pipe(Effect.map((controller) => controller.close())) }
})

const textStart = [
  sseNamedEvent("message_start", { type: "message_start", message: { usage: { input_tokens: 5 } } }),
  sseNamedEvent("content_block_start", {
    type: "content_block_start",
    index: 0,
    content_block: { type: "text", text: "" },
  }),
]

const textDelta = (text: string) =>
  sseNamedEvent("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } })

describe("HTTP transport timeouts", () => {
  it.effect("fails when response headers take longer than five minutes", () =>
    Effect.gen(function* () {
      const fiber = yield* LLMClient.generate(ai.request({ model, prompt: "Hello" })).pipe(
        Effect.provide(silentServer),
        Effect.flip,
        Effect.forkChild({ startImmediately: true }),
      )
      yield* TestClock.adjust("5 minutes")
      const error = yield* Fiber.join(fiber)

      expect(error.reason).toMatchObject({
        _tag: "Transport",
        transport: "http",
        operation: "request",
        code: "Timeout",
      })
    }),
  )

  it.effect("fails when the response body stalls for fifteen minutes", () =>
    Effect.gen(function* () {
      const server = yield* stalledServer
      const fiber = yield* LLMClient.generate(ai.request({ model, prompt: "Hello" })).pipe(
        Effect.provide(server.layer),
        Effect.flip,
        Effect.forkChild({ startImmediately: true }),
      )
      yield* Deferred.await(server.stalled)
      yield* Effect.yieldNow
      yield* TestClock.adjust("15 minutes")
      const error = yield* Fiber.join(fiber)

      expect(error.reason).toMatchObject({ _tag: "Transport", transport: "http", operation: "read", code: "Timeout" })
      expect(error.reason.http).toMatchObject({ status: 200 })
    }),
  )

  it.effect("fails when only keepalives and pings arrive for fifteen minutes", () =>
    Effect.gen(function* () {
      const server = yield* pushedServer
      const fiber = yield* LLMClient.generate(ai.request({ model: anthropic, prompt: "Hello" })).pipe(
        Effect.provide(server.layer),
        Effect.flip,
        Effect.forkChild({ startImmediately: true }),
      )
      yield* server.push(...textStart)
      yield* TestClock.adjust("10 minutes")
      yield* server.push(": keep-alive", sseNamedEvent("ping", { type: "ping" }))
      yield* TestClock.adjust("10 minutes")
      const error = yield* Fiber.join(fiber)

      expect(error.reason).toMatchObject({ _tag: "Transport", transport: "http", operation: "read", code: "Timeout" })
    }),
  )

  it.effect("keeps streaming while model output arrives inside each chunk window", () =>
    Effect.gen(function* () {
      const server = yield* pushedServer
      const fiber = yield* LLMClient.generate(ai.request({ model: anthropic, prompt: "Hello" })).pipe(
        Effect.provide(server.layer),
        Effect.forkChild({ startImmediately: true }),
      )
      yield* server.push(...textStart)
      for (const text of ["a", "b", "c"]) {
        yield* TestClock.adjust("10 minutes")
        yield* server.push(textDelta(text))
      }
      yield* server.push(
        sseNamedEvent("content_block_stop", { type: "content_block_stop", index: 0 }),
        sseNamedEvent("message_delta", {
          type: "message_delta",
          delta: { stop_reason: "end_turn" },
          usage: { output_tokens: 3 },
        }),
        sseNamedEvent("message_stop", { type: "message_stop" }),
      )
      yield* server.close
      const response = yield* Fiber.join(fiber)

      expect(response.text).toBe("abc")
    }),
  )

  it.effect("applies a configured header timeout", () =>
    Effect.gen(function* () {
      const fiber = yield* LLMClient.generate(ai.request({ model, prompt: "Hello", timeout: { headers: 1_000 } })).pipe(
        Effect.provide(silentServer),
        Effect.flip,
        Effect.forkChild({ startImmediately: true }),
      )
      yield* TestClock.adjust("1 second")
      const error = yield* Fiber.join(fiber)

      expect(error.reason).toMatchObject({ _tag: "Transport", operation: "request", code: "Timeout" })
    }),
  )

  it.effect("disables the header timeout with false", () =>
    Effect.gen(function* () {
      const fiber = yield* LLMClient.generate(ai.request({ model, prompt: "Hello", timeout: { headers: false } })).pipe(
        Effect.provide(slowHeadersServer),
        Effect.forkChild({ startImmediately: true }),
      )
      yield* TestClock.adjust("10 minutes")
      const response = yield* Fiber.join(fiber)

      expect(response.text).toBe("Hi")
    }),
  )

  it.effect("disables the chunk timeout with false", () =>
    Effect.gen(function* () {
      const server = yield* stalledServer
      const fiber = yield* LLMClient.generate(ai.request({ model, prompt: "Hello", timeout: { chunk: false } })).pipe(
        Effect.provide(server.layer),
        Effect.forkChild({ startImmediately: true }),
      )
      yield* Deferred.await(server.stalled)
      yield* Effect.yieldNow
      yield* TestClock.adjust("10 minutes")
      server.resume()
      const response = yield* Fiber.join(fiber)

      expect(response.text).toBe("Hi")
    }),
  )

  it.effect("applies a whole-request timeout while the body is still streaming", () =>
    Effect.gen(function* () {
      const server = yield* stalledServer
      const fiber = yield* LLMClient.generate(
        ai.request({ model, prompt: "Hello", timeout: { total: 60_000, chunk: false } }),
      ).pipe(Effect.provide(server.layer), Effect.flip, Effect.forkChild({ startImmediately: true }))
      yield* Deferred.await(server.stalled)
      yield* Effect.yieldNow
      yield* TestClock.adjust("1 minute")
      const error = yield* Fiber.join(fiber)

      expect(error.reason).toMatchObject({ _tag: "Transport", operation: "read", code: "Timeout" })
    }),
  )

  it.effect("bounds the header wait by a shorter whole-request timeout", () =>
    Effect.gen(function* () {
      const fiber = yield* LLMClient.generate(
        ai.request({ model, prompt: "Hello", timeout: { total: 1_000, headers: false } }),
      ).pipe(Effect.provide(silentServer), Effect.flip, Effect.forkChild({ startImmediately: true }))
      yield* TestClock.adjust("1 second")
      const error = yield* Fiber.join(fiber)

      expect(error.reason).toMatchObject({ _tag: "Transport", operation: "request", code: "Timeout" })
    }),
  )

  it.effect("merges timeouts with later values winning", () =>
    Effect.sync(() => {
      const merged = mergeTransport(
        { timeout: { total: 5_000, headers: 1_000, chunk: 2_000 } },
        { headers: { a: "b" } },
        { timeout: { chunk: false } },
        { timeout: 3_000 },
      )

      expect(merged).toMatchObject({ headers: { a: "b" }, timeout: { total: 3_000, headers: 1_000, chunk: false } })
      expect(mergeTransport({}, undefined)).toEqual({})
    }),
  )
})
