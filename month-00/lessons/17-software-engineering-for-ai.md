# 0.17 — Software Engineering for AI

**Depth: SKIM**

**Goal:** structure AI code around explicit contracts, test the right boundaries, and make failures diagnosable and reproducible.

[Month 0 roadmap](../README.md) · [Previous: Linux, Containers, and Git](16-linux-containers-and-git.md) · [Next: MLOps](18-mlops-fundamentals.md)

## Modules, interfaces, and dependency direction

Separate domain logic from infrastructure. A classifier should accept typed input and return a defined result without knowing whether requests arrived through HTTP or a batch file. Adapters handle model providers, storage, queues, and frameworks.

```text
API / batch entry points → application service → domain contracts
                                 ↓
                    model, storage, telemetry adapters
```

An interface describes behavior and failure semantics, not merely method names. Include units, shapes, schema versions, timeout/cancellation, and ownership of resources. Validate at boundaries and keep the internal representation simple.

## Dependency injection and configuration

Dependency injection supplies collaborators rather than constructing them deep inside business logic. Tests can use a deterministic fake model; production can use a remote client. This also makes time, randomness, storage, and feature flags controllable.

Configuration should be typed, validated at startup, and immutable during a request. Separate secrets from ordinary config, version prompt/model settings, and log a safe configuration fingerprint. Avoid hidden global clients and import-time network calls.

## Testing layers

- Unit tests check pure transformations, shape rules, prompt assembly, and parsing quickly.
- Integration tests check real boundaries such as serialization, database transactions, or a provider sandbox.
- Contract tests verify that consumers and providers agree on schemas and errors.
- End-to-end tests cover a small number of critical workflows in a realistic environment.
- Evaluation tests use datasets and metrics to measure probabilistic model behavior.

Mock the boundary, not every internal call. Over-mocking reproduces implementation details and can pass while the real integration fails. Fakes implement useful behavior and state; stubs return fixed responses; spies record calls. For nondeterministic systems, assert invariants and metric thresholds over representative cases rather than one exact prose response.

## Error handling and resilience

Classify validation, authentication, rate-limit, timeout, dependency, model-output, and internal failures. Retry only transient idempotent operations within a deadline. Preserve causes internally and map them to safe public errors.

Use bounded queues, concurrency limits, circuit breakers where justified, and graceful degradation. Every fallback changes semantics and needs evaluation. A smaller backup model may be faster but produce a different schema or calibration.

## Observability

Logs describe discrete events, metrics summarize numeric behavior, and traces connect work across services. Include request/job IDs, model and prompt versions, latency stages, token/record counts, finish reasons, and categorized failures. Avoid raw sensitive prompts or model outputs by default.

Monitor both software and model behavior: request rate, errors, latency, queue depth, resource saturation, schema-valid rate, task success, refusals, and cost. High HTTP success with invalid model output is still a failure.

Cardinality matters. User IDs and prompt text should not become metric labels. Put high-cardinality identifiers in controlled logs/traces and aggregate metrics by bounded dimensions.

## Reproducibility and versioned contracts

A reproducible AI result depends on code, configuration, model, tokenizer, data, prompt, tool versions, sampling settings, seed, and runtime. Record immutable identifiers where possible. Deterministic replay can still be limited by external tools or numerical kernels; document the practical guarantee.

Schema evolution needs compatibility tests. Producers may add optional fields before consumers require them. Removing or redefining a field needs a migration. Treat prompt templates and evaluation sets as reviewed software artifacts.

## Checkpoint

1. What does dependency injection make easier in an AI service?
2. Why can conventional unit tests be insufficient for an LLM feature?
3. What is the difference among logs, metrics, and traces?
4. Why should prompt text not be a metric label?

<details>
<summary>Show answers</summary>

1. Substituting deterministic fakes, changing providers, controlling time/randomness, and testing failure behavior without hidden global state.
2. The model is probabilistic and quality depends on a distribution of inputs. Dataset-based evaluation and integration/contract tests cover behavior that isolated deterministic tests miss.
3. Logs capture events, metrics aggregate numeric signals, and traces show the path and timing of one distributed operation.
4. It creates unbounded label cardinality, cost, and privacy risk; aggregate using bounded categories and keep sensitive details in controlled storage.

</details>

## Exercise — Design a testable LLM component

Design a component that extracts a typed invoice record using a model. Define its interface, dependencies, tests, errors, and telemetry.

<details>
<summary>Show exercise solution</summary>

Accept a document reference plus authorized content and return a versioned `Invoice` schema or categorized error. Inject a model client, clock, retry policy, and telemetry sink. The service builds a versioned request, invokes the client under a deadline, validates the structured response, and records provenance.

Unit tests cover prompt/context assembly, schema validation, retry classification, and redaction with deterministic fakes. Contract tests verify provider request/response mapping. Integration tests exercise storage and timeout behavior. An evaluation set measures field accuracy, schema validity, abstention, latency, and cost across layouts and languages. Logs contain request ID and versions; metrics use bounded outcome categories; traces separate preprocessing, model, and validation latency.

</details>

## Completion criteria

Define module boundaries and interfaces, inject dependencies, choose testing layers, and design safe logs, metrics, traces, and reproducibility metadata.

## Primary references

- [The Twelve-Factor App](https://12factor.net/)
- [OpenTelemetry specifications](https://opentelemetry.io/docs/specs/)
- [Google Testing Blog: Test Sizes](https://testing.googleblog.com/2010/12/test-sizes.html)
- [Semantic Versioning](https://semver.org/)
