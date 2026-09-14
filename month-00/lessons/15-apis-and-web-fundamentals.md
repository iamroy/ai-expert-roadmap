# 0.15 APIs and Web Fundamentals — SKIM

Almost every LLM you use in this roadmap arrives over HTTP, and almost every system you build serves one over HTTP. This lesson covers the request/response mechanics, the reliability patterns that matter when calling a slow and rate-limited service, and the streaming protocols behind token-by-token output.

| | |
|---|---|
| **Mode** | SKIM |
| **Time** | 60–75 minutes |
| **Assumes** | 0.1 Python (async, JSON, configuration) |
| **Used by** | Month 3 LLM APIs and structured generation, Month 5 agents and tools, Month 8 production serving |

[Month 0 roadmap](../README.md) · [Previous: Basic Data Engineering](14-basic-data-engineering.md) · [Next: Linux, Containers, and Git](16-linux-containers-and-git.md)

## Learning objectives

After this lesson you can:

- read an HTTP request and response, and choose the right method and status code
- authenticate safely with API keys and explain what not to do with them
- implement retries with exponential backoff and jitter, and know which errors to retry
- handle rate limits, timeouts, and pagination correctly
- explain SSE, WebSockets, and how LLM token streaming works
- choose between synchronous, streaming, and asynchronous job endpoints for model work

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Sections [0.15.4](#0154--retries-timeouts-and-backoff) and [0.15.6](#0156--streaming-sse-and-websockets) are the ones that matter most in practice.
3. Record gaps in [`progress.md`](../progress.md).

## 0.15.1 — HTTP and REST

An HTTP exchange is a request and a response, both with headers and an optional body.

```text
POST /v1/messages HTTP/1.1
Host: api.example.com
Authorization: Bearer sk-…
Content-Type: application/json

{"model": "some-model", "messages": [{"role": "user", "content": "Hello"}]}
```

```text
HTTP/1.1 200 OK
Content-Type: application/json

{"id": "msg_123", "content": [{"type": "text", "text": "Hi there."}]}
```

**Methods** and the properties that matter operationally:

| Method | Purpose | Safe | Idempotent |
|---|---|---|---|
| GET | read | yes | yes |
| POST | create or invoke | no | no |
| PUT | replace | no | yes |
| PATCH | partial update | no | no |
| DELETE | remove | no | yes |

"Idempotent" means repeating the request has the same effect as sending it once. This is directly relevant to retries: retrying a GET is free, while retrying a POST may create a second resource or bill you twice. APIs that expect retries offer an **idempotency key**, a client-generated ID the server uses to deduplicate.

**REST** is a style: resources identified by URLs, manipulated with standard methods, with representations usually in JSON. Most LLM APIs are only loosely RESTful; they are mostly a single POST endpoint that invokes a model.

## 0.15.2 — Status codes

Learn the ranges, then the specific codes that change your behavior.

| Range | Meaning |
|---|---|
| 2xx | success |
| 3xx | redirection |
| 4xx | client error: the request was wrong |
| 5xx | server error: the request may have been fine |

| Code | Meaning | Retry? |
|---|---|---|
| 200 / 201 | OK / Created | — |
| 400 | bad request; malformed input | no, fix the request |
| 401 | unauthenticated; bad or missing key | no |
| 403 | authenticated but not permitted | no |
| 404 | not found | no |
| 408 | request timeout | yes |
| 409 | conflict | usually no |
| 422 | valid JSON, invalid content | no |
| 429 | rate limited | yes, after the retry delay |
| 500 | server error | yes |
| 502 / 503 / 504 | gateway, unavailable, gateway timeout | yes |

The rule that follows: **retry 408, 429, and 5xx; do not retry other 4xx.** Retrying a 400 just sends the same broken request again.

## 0.15.3 — Authentication and API keys

Most LLM APIs use a bearer token in a header:

```text
Authorization: Bearer sk-…
```

Non-negotiables:

- **Load keys from the environment or a secret manager**, never from source (0.1.11). A key committed to Git is compromised even after you delete the commit, because the history retains it.
- **Never put a key in a URL query string.** URLs end up in logs, proxies, browser history, and referrer headers.
- **Never log request headers wholesale.** Redact the `Authorization` value.
- **Never ship a provider key in client-side code.** Anything in a browser or mobile app is readable. Proxy through your own backend, which is also where you put per-user rate limiting and auditing.
- **Rotate on any suspicion**, and prefer short-lived tokens where the provider offers them.

Other schemes you will meet: OAuth 2.0 for acting on a user's behalf, HMAC signatures for webhook verification, and mTLS inside service meshes.

## 0.15.4 — Retries, timeouts, and backoff

LLM APIs are slow, occasionally overloaded, and rate limited. A client without retry and timeout handling will fail constantly.

**Timeouts.** Always set one. A request with no timeout can hang indefinitely and consume a worker forever. Distinguish the connect timeout (a few seconds) from the read timeout, which for a long generation may need to be minutes.

**Exponential backoff with jitter.** Doubling the wait after each failure gives an overloaded server room to recover. **Jitter** is not optional: without randomization, all failed clients retry in lockstep and re-create the spike they are recovering from, which is the thundering-herd problem.

```python
import random
import time

class RateLimited(Exception):
    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after


def call_with_retry(operation, attempts: int = 5, base: float = 0.5, cap: float = 30.0):
    for attempt in range(attempts):
        try:
            return operation()
        except RateLimited as error:
            if attempt == attempts - 1:
                raise
            # honor the server's own hint when it sends one
            delay = error.retry_after if error.retry_after is not None else min(
                cap, base * 2 ** attempt
            )
            delay += random.uniform(0, delay * 0.1)      # full-ish jitter
            time.sleep(min(delay, 0.01))                 # shortened for this demo
    raise RuntimeError("unreachable")


calls = {"n": 0}
def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise RateLimited()
    return "ok"

print(call_with_retry(flaky), "after", calls["n"], "attempts")
```

If the response carries a `Retry-After` header or a reset timestamp, **obey it** rather than guessing; the server knows when capacity returns.

**Rate limits** usually come in several dimensions at once: requests per minute, tokens per minute, and concurrent requests. Handle them proactively with a concurrency semaphore (0.1.14) rather than only reacting to 429s. When you also hold a queue of work, a **circuit breaker** that stops calling a persistently failing service for a cool-off period prevents you from burning your entire retry budget on a dead endpoint.

**Idempotency on retries.** If the request is a POST that creates or bills something, send an idempotency key so a retry after a timeout does not duplicate the work. This matters most in exactly the ambiguous case: the request succeeded but the response never arrived.

## 0.15.5 — Pagination

Large collections come in pages. Two common styles:

| Style | Request | Trade-off |
|---|---|---|
| Offset / page | `?limit=100&offset=200` | simple; drifts and can skip or repeat items when the collection changes mid-traversal |
| Cursor / token | `?limit=100&cursor=abc123` | stable under concurrent writes; cannot jump to an arbitrary page |

Most modern APIs use cursors. The generator pattern from 0.1.8 fits perfectly, since the caller gets a flat stream and never sees the paging:

```python
def paginate(fetch_page, limit: int = 100):
    """Yield items across pages; fetch_page(cursor) -> (items, next_cursor)."""
    cursor = None
    while True:
        items, cursor = fetch_page(cursor)
        yield from items
        if not cursor:
            return


pages = {None: ([1, 2], "c1"), "c1": ([3, 4], "c2"), "c2": ([5], None)}
print(list(paginate(lambda cursor: pages[cursor])))   # [1, 2, 3, 4, 5]
```

Always guard against an infinite loop from a server that keeps returning the same cursor.

## 0.15.6 — Streaming: SSE and WebSockets

Waiting for a full LLM response before showing anything feels broken. Streaming sends tokens as they are produced, which changes perceived latency far more than any model optimization.

**Server-Sent Events (SSE)** is the mechanism most LLM APIs use. It is plain HTTP, one-directional (server to client), text-based, and auto-reconnecting. The wire format is simple:

```text
data: {"type":"content_block_delta","delta":{"text":"Hello"}}

data: {"type":"content_block_delta","delta":{"text":" there"}}

data: [DONE]
```

Each event is `data: ` plus a payload, terminated by a blank line.

```python
raw_stream = (
    'data: {"delta": "Hello"}\n'
    "\n"
    'data: {"delta": " there"}\n'
    "\n"
    "data: [DONE]\n"
    "\n"
)

import json

def parse_sse(lines):
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            return
        yield json.loads(payload)["delta"]

print("".join(parse_sse(raw_stream.splitlines())))   # Hello there
```

**WebSockets** are bidirectional and persistent, and are the right choice for genuinely two-way interaction such as live voice or collaborative editing. For request/response generation, SSE is simpler, works through more proxies, and reconnects on its own.

Practical notes on streaming:

- Time to first token and total time are different metrics. Streaming improves the first dramatically and the second not at all.
- A stream can fail partway through, after you have already shown output. Decide in advance whether to keep partial text or discard it.
- Intermediate proxies sometimes buffer responses and defeat streaming entirely. If tokens arrive in one burst, suspect a proxy.
- Structured output and streaming interact awkwardly: partial JSON is invalid JSON, so you need an incremental parser or you buffer until complete (Month 3).
- Token counts and cost usually arrive in a final event, not the first, so bill and log at the end.

## 0.15.7 — Designing an endpoint for a slow model

Inference can take longer than a reasonable HTTP request. There are three standard shapes, and picking the wrong one causes most of the pain:

| Shape | Fits | Mechanics |
|---|---|---|
| Synchronous | fast, bounded work (under ~30s) | `POST /v1/predict` returns `200` with the result |
| Streaming | long generation shown progressively | `POST` returns `200` and an SSE body |
| Asynchronous job | batch or very long work | `POST /v1/jobs` returns `202 Accepted` with a job ID; the client polls `GET /v1/jobs/{id}` or receives a webhook |

The async pattern is worth knowing concretely, because it is how batch embedding and evaluation jobs are exposed:

```text
POST /v1/jobs            -> 202 Accepted   {"job_id": "job_123", "status": "queued"}
GET  /v1/jobs/job_123    -> 200 OK         {"status": "running", "progress": 0.4}
GET  /v1/jobs/job_123    -> 200 OK         {"status": "succeeded", "result_url": "..."}
```

Design points that carry across all three shapes:

- **Return errors in a consistent, machine-readable shape.** RFC 9457 problem details is a reasonable default: a `type`, `title`, `status`, and `detail`. A client should never have to parse a human sentence to decide whether to retry.
- **Distinguish a failed job from a failed request.** A job that ran and produced an error is a `200` with `"status": "failed"`, not a `500`. Conflating them makes retry logic incorrect.
- **Version the API** (`/v1/`), because model behavior changes are breaking changes for someone.
- **Accept an idempotency key on job creation**, so a retried submission does not queue the work twice (0.15.4).
- **Set explicit limits** on input size and return `413` or `422` rather than timing out, and document the limit.

## Exercises

### Exercise 1 — A resilient client

Write a client wrapper with a timeout, retries on 408/429/5xx only, exponential backoff with jitter, `Retry-After` support, and a concurrency semaphore. Use a fake transport that fails in scripted ways, and assert that a 400 is not retried while a 429 is.

### Exercise 2 — Parse an SSE stream

Parse a realistic SSE byte stream that includes multi-line events, comment lines beginning with `:`, and a terminating sentinel. Reassemble the full text, then handle a stream that stops abruptly and report how much was received.

### Exercise 3 — Paginate safely

Implement cursor pagination with a maximum page count and repeated-cursor detection. Show it terminating cleanly against a server that returns the same cursor forever.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
import asyncio
import random


class HTTPError(Exception):
    def __init__(self, status: int, retry_after: float | None = None):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


RETRYABLE = {408, 429, 500, 502, 503, 504}


async def request_with_retry(send, *, attempts=5, base=0.05, cap=2.0, limit=None):
    for attempt in range(attempts):
        try:
            if limit is not None:
                async with limit:
                    async with asyncio.timeout(10):
                        return await send()
            async with asyncio.timeout(10):
                return await send()
        except HTTPError as error:
            if error.status not in RETRYABLE or attempt == attempts - 1:
                raise
            delay = error.retry_after if error.retry_after is not None else min(
                cap, base * 2 ** attempt
            )
            await asyncio.sleep(delay * (0.5 + random.random() * 0.5))   # jitter


def scripted(statuses: list[int]):
    """Fail with each status in turn, then succeed."""
    remaining = list(statuses)

    async def send():
        if remaining:
            raise HTTPError(remaining.pop(0), retry_after=0.01)
        return "ok"

    return send


async def main() -> None:
    limit = asyncio.Semaphore(4)

    result = await request_with_retry(scripted([429, 503]), limit=limit)
    print("retried then succeeded:", result)

    try:
        await request_with_retry(scripted([400, 400]), limit=limit)
    except HTTPError as error:
        print("not retried:", error)

    try:
        await request_with_retry(scripted([500] * 10), attempts=3, limit=limit)
    except HTTPError as error:
        print("gave up after attempts:", error)


asyncio.run(main())
```

The 429 and 503 are retried and the call eventually succeeds. The 400 raises immediately, because retrying a malformed request only wastes quota. The permanent 500 exhausts the attempt budget and surfaces the error rather than looping forever.

### Exercise 2

```python
def parse_sse_stream(chunks):
    """Yield decoded events from raw SSE text chunks."""
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in block.split("\n"):
                if line.startswith(":") or not line.strip():
                    continue                                   # comment / keep-alive
                if line.startswith("data: "):
                    data_lines.append(line[len("data: "):])
            if not data_lines:
                continue
            payload = "\n".join(data_lines)
            if payload == "[DONE]":
                return
            yield json.loads(payload)


chunks = [
    ': keep-alive\n\ndata: {"delta": "Hel',                    # split mid-token
    'lo"}\n\ndata: {"delta": " wor"}\n\n',
    'data: {"delta": "ld"}\n\ndata: [DONE]\n\n',
]
print("".join(event["delta"] for event in parse_sse_stream(chunks)))

truncated = ['data: {"delta": "Hello"}\n\ndata: {"delta": " wor']
received = [event["delta"] for event in parse_sse_stream(truncated)]
print(f"truncated stream delivered {len(received)} event(s): {''.join(received)!r}")
```

Two details matter and both appear here. Network chunks do not align with event boundaries, so you must buffer and split on the blank line rather than assuming one chunk is one event. And a stream can simply stop: the partial event is silently dropped, so the caller has to notice the missing terminator itself rather than assuming the text is complete.

### Exercise 3

```python
def paginate_safely(fetch_page, max_pages: int = 1_000):
    cursor, seen, pages = None, set(), 0
    while True:
        items, next_cursor = fetch_page(cursor)
        yield from items
        pages += 1
        if not next_cursor:
            return
        if next_cursor in seen:
            raise RuntimeError(f"pagination loop: cursor {next_cursor!r} repeated")
        if pages >= max_pages:
            raise RuntimeError(f"pagination exceeded {max_pages} pages")
        seen.add(next_cursor)
        cursor = next_cursor


good = {None: ([1, 2], "c1"), "c1": ([3, 4], "c2"), "c2": ([5], None)}
print(list(paginate_safely(lambda cursor: good[cursor])))

try:
    list(paginate_safely(lambda cursor: ([1], "stuck")))
except RuntimeError as error:
    print("caught:", error)
```

Both guards are needed. Repeated-cursor detection catches the common buggy-server case immediately, and the page cap bounds the damage when a server cycles through cursors instead of repeating one.

</details>

## Exit test

1. What are the parts of an HTTP request and response?
2. What does idempotent mean, and which methods are?
3. Which status codes should you retry, and which should you not?
4. What is the difference between 401 and 403?
5. Name three things you must never do with an API key.
6. Why is jitter necessary in exponential backoff?
7. Why set both a connect and a read timeout?
8. When does an idempotency key matter most?
9. Compare offset and cursor pagination.
10. What is SSE, and why do LLM APIs use it rather than WebSockets?
11. Which latency metric does streaming improve, and which does it not?
12. Name two ways streaming complicates structured output and error handling.
13. When would you expose inference as an async job, and why is a failed job not a 500?

<details>
<summary>Show answers</summary>

1. A method, URL, headers, and optional body for the request; a status code, headers, and optional body for the response.
2. Repeating the request has the same effect as sending it once. GET, PUT, and DELETE are idempotent; POST and PATCH generally are not.
3. Retry 408, 429, and 5xx. Do not retry 400, 401, 403, 404, or 422; the request itself is wrong and will fail identically.
4. 401 means the request is not authenticated, usually a missing or invalid key. 403 means it is authenticated but not permitted to do this.
5. Never commit it to source control, never put it in a URL query string, never log headers containing it, and never ship it in client-side code.
6. Without randomization, all failed clients retry at the same instants and recreate the load spike they are recovering from.
7. They fail for different reasons and need different budgets: connecting should take a few seconds at most, while reading a long generation may legitimately take minutes.
8. When a request that creates or bills something times out, so you cannot tell whether it succeeded. The key lets the server deduplicate the retry.
9. Offset is simple and supports jumping to any page, but can skip or duplicate items if the collection changes during traversal. Cursor pagination is stable under concurrent writes but only moves forward.
10. Server-Sent Events: a one-directional, text-based streaming format over plain HTTP with automatic reconnection. Generation is request/response, so bidirectional WebSockets add complexity without benefit, and SSE passes through more infrastructure.
11. It improves time to first token dramatically; it does not improve total generation time.
12. Partial JSON is invalid JSON, so structured output needs an incremental parser or buffering. A stream can fail after partial output has been displayed, and usage or cost data arrives only in the final event.
13. When the work exceeds a reasonable request lifetime — batch embedding, evaluation runs, large document processing — so the server returns `202` with a job ID that the client polls or receives a webhook for. A job that ran and failed is a successful request reporting a failed job, so it returns `200` with a failed status; using `500` would tell a client's retry logic that the request itself was unreliable.

</details>

## Completion criteria

You are done when:

- you can classify any status code as retryable or not
- you can write a retry wrapper with backoff, jitter, and `Retry-After` support from memory
- you can parse an SSE stream where chunks do not align with event boundaries
- you can state the API key rules without hedging
- your three exercises run, including the truncated stream and the pagination loop

## References

**HTTP and REST**
- [RFC 9110: HTTP semantics](https://www.rfc-editor.org/rfc/rfc9110)
- [RFC 9457: problem details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457)
- [MDN: HTTP overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)
- [MDN: HTTP response status codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)
- [MDN: HTTP authentication](https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication)

**Reliability**
- [AWS: timeouts, retries, and backoff with jitter](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/)
- [Exponential backoff and jitter, AWS architecture blog](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
- [Stripe: idempotent requests](https://docs.stripe.com/api/idempotent_requests)
- [Release It!, Michael Nygard](https://pragprog.com/titles/mnee2/release-it-second-edition/) — circuit breakers and bulkheads

**Streaming**
- [MDN: Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
- [WHATWG: the event stream format](https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation)
- [MDN: WebSockets API](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
- [RFC 6455: the WebSocket protocol](https://www.rfc-editor.org/rfc/rfc6455)
- [RFC 9700: best current practice for OAuth 2.0 security](https://www.rfc-editor.org/rfc/rfc9700)

**Python clients**
- [httpx](https://www.python-httpx.org/) — sync and async, with timeouts and connection limits
- [tenacity](https://tenacity.readthedocs.io/) — retry decorators with backoff strategies
- [FastAPI](https://fastapi.tiangolo.com/) — for the serving side in later months

## About this lesson

Written to cover section 0.15 of the [Month 0 curriculum](../README.md). Code examples were checked with Python 3.12.
