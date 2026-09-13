# 0.15 — APIs and Web Fundamentals

**Depth: SKIM**

**Goal:** build and consume reliable model-facing HTTP APIs with authentication, limits, retries, pagination, and streaming.

[Month 0 roadmap](../README.md) · [Previous: Data Engineering](14-basic-data-engineering.md) · [Next: Linux, Containers, and Git](16-linux-containers-and-git.md)

## HTTP request and response

An HTTP request has a method, target URI, headers, and optional body. A response has a status code, headers, and optional body. HTTP semantics are distinct from JSON; JSON is one possible representation named by `Content-Type: application/json`.

Common methods:

- `GET` retrieves a representation and should be safe and idempotent.
- `POST` submits data or creates/processes a resource and is not inherently idempotent.
- `PUT` replaces state at a known resource and is idempotent by intended semantics.
- `PATCH` applies a partial change; idempotence depends on the patch operation.
- `DELETE` requests removal and is idempotent in intended effect, even if repeated responses differ.

Idempotence matters for retries. A POST that charges money or launches a model job should accept an idempotency key or expose a resumable job resource.

## Status codes and error contracts

Use status families consistently: 2xx success, 4xx client/request problems, and 5xx server failures. Useful examples include 201 for created resources, 202 for accepted asynchronous work, 400 for malformed requests, 401 for missing/invalid authentication, 403 for authenticated but unauthorized access, 404 for absent resources, 409 for state conflict, 422 for semantically invalid content, 429 for rate limiting, and 503 for temporary unavailability.

Return a stable machine-readable error code, human-readable message, request ID, and field-level details where safe. Do not expose stack traces, credentials, internal prompts, or private provider responses.

## Authentication and authorization

Authentication establishes an identity; authorization determines permitted actions and data. API keys identify a caller or project but need storage, rotation, scope, and revocation. OAuth-style access tokens support delegated authorization and expiration.

Use TLS, keep secrets out of URLs and logs, validate token audience/issuer/expiry, and enforce authorization on every resource lookup. A valid user ID supplied by the client is not authorization to access that user's data.

## Rate limits, timeouts, and retries

Rate limits protect capacity and fairness. Communicate quotas and retry timing through documented headers or error fields. Client libraries should distinguish retryable failures from permanent validation errors.

Set connection, read, and total deadlines. A timeout does not prove the server stopped processing; retrying a non-idempotent action can duplicate work. Use exponential backoff with jitter, cap attempts, honor server retry guidance, and budget retries within an end-to-end deadline.

Model APIs add token and concurrency limits. Apply backpressure rather than allowing an unbounded request queue. Track queue time separately from inference time.

## Pagination

Offset pagination is simple but can skip or duplicate items when records change. Cursor pagination encodes a stable continuation position, often based on a deterministic sort key. Treat cursors as opaque and include filters/sort in their validity contract.

A response should provide items and a next cursor or link. Set maximum page size and define ordering. Consumers must handle an empty page, final page, repeated cursor protection, and retry without processing duplicate items.

## Streaming: SSE and WebSockets

Server-Sent Events provide a long-lived HTTP response carrying server-to-client events. They work well for token streams and reconnect semantics. Events are text-framed; applications must buffer partial UTF-8/data and define completion/error events.

WebSockets provide full-duplex messages after an HTTP upgrade, useful when both sides send ongoing events. They require connection lifecycle, heartbeat, backpressure, authentication renewal, and per-message authorization design.

Streaming improves perceived latency but not necessarily total compute time. Once bytes are shown, retracting unsafe or invalid output is difficult. Buffer when the application needs schema validation or moderation before display.

## REST resource design for model jobs

Long inference or batch work can use an asynchronous resource:

```text
POST /jobs          → 202 + job ID
GET  /jobs/{id}     → state and progress
GET  /jobs/{id}/result
DELETE /jobs/{id}   → cancellation request
```

Define states and transitions, idempotency, retention, cancellation semantics, and result provenance. A cancellation response should not claim work stopped until the worker acknowledges it.

## Checkpoint

1. Why can retrying a timed-out POST be dangerous?
2. What is the difference between 401 and 403?
3. Why is cursor pagination more stable under concurrent inserts?
4. When might token streaming be inappropriate?

<details>
<summary>Show answers</summary>

1. The server may have completed the action even though the response was lost; a retry can duplicate the effect without an idempotency mechanism.
2. 401 concerns missing/invalid authentication; 403 means the recognized principal lacks permission.
3. It continues from a stable ordered key rather than an offset whose rows shift when inserts occur.
4. When output must be validated as a complete schema, moderated before exposure, or handled transactionally before users see it.

</details>

## Exercise — Specify an inference endpoint

Design a JSON API for an idempotent asynchronous document-classification job. Include request/response fields, errors, retry policy, and observability metadata.

<details>
<summary>Show exercise solution</summary>

`POST /classification-jobs` accepts an idempotency key, document reference or bounded content, requested model alias, and client metadata. It returns 202 with job ID, accepted model/version, status URL, and request ID. Repeating the same key and equivalent payload returns the same job; a conflicting payload returns 409.

`GET /classification-jobs/{id}` enforces resource authorization and returns queued/running/succeeded/failed/cancelled plus timestamps. A successful result includes label, score interpretation, model version, input hash, and finish reason. Validation errors are 400/422; rate limits are 429 with retry guidance; temporary capacity failures are 503. Clients retry safe reads and idempotent creation with bounded exponential backoff and jitter. Logs use request/job IDs and omit raw sensitive text.

</details>

## Completion criteria

Explain HTTP methods/statuses, separate authentication from authorization, design bounded retries and pagination, and choose SSE/WebSockets/buffering deliberately.

## Primary references

- [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110)
- [RFC 9457: Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457)
- [OAuth 2.0 Security Best Current Practice](https://www.rfc-editor.org/rfc/rfc9700)
- [Server-Sent Events specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [WebSocket Protocol](https://www.rfc-editor.org/rfc/rfc6455)
