# 0.14 — Basic Data Engineering

**Depth: SKIM**

**Goal:** design a small, reproducible data pipeline with appropriate formats, storage, validation, lineage, and versioning.

[Month 0 roadmap](../README.md) · [Previous: Information Retrieval](13-information-retrieval-fundamentals.md) · [Next: APIs](15-apis-and-web-fundamentals.md)

## Structured and unstructured data

Structured data has an explicit schema: rows, fields, types, and constraints. Unstructured data such as prose, images, audio, and PDFs lacks one uniform tabular shape, but production systems still attach structured metadata for identity, provenance, permissions, timestamps, and processing status.

Semi-structured records such as JSON have named fields but can vary between records. A schema remains valuable even when storage does not enforce it: validate required fields, types, enumerations, nullability, and semantic constraints at boundaries.

## JSON, JSONL, and Parquet

JSON represents nested objects and arrays and is convenient for APIs and small configuration artifacts. It has no native date or binary type and numeric precision must be handled deliberately.

JSON Lines stores one JSON value per line. It supports streaming and record-level recovery: a reader can process large files incrementally and identify a malformed line. Newlines inside string values must remain escaped, and each record should carry an ID.

Parquet is a columnar, typed format suited to analytical scans. Column projection and statistics can reduce IO, while compression exploits values of similar type. It is less convenient for frequent single-row mutation and requires schema-evolution discipline.

```text
JSON:    interoperable nested object or small payload
JSONL:   append/stream independent records
Parquet: typed columnar datasets and analytical reads
```

Partitioning by a commonly filtered, bounded-cardinality field such as date can prune reads. Partitioning by user ID may create millions of tiny directories/files. File size and metadata overhead matter alongside total bytes.

## Batch, streaming, ETL, and ELT

Batch processing handles a bounded collection on a schedule. Streaming processes events continuously or in small windows. A stream is not automatically real-time: ingestion lag, windows, retries, and downstream processing determine freshness.

ETL transforms before loading into the destination. ELT loads raw or lightly processed data first and transforms inside the target platform. Choose based on governance, destination capability, cost, and whether retaining raw inputs is allowed.

Exactly-once business effects are difficult across distributed boundaries. Systems often use at-least-once delivery plus idempotent consumers, stable event IDs, deduplication, and transactional writes. Idempotence means retrying the same logical input does not duplicate its effect.

## Object storage and caching

Object storage uses keys and immutable-like blobs rather than a mounted filesystem's assumptions. Write new versioned keys, validate checksums, then update a small manifest or catalog pointer. Do not let readers discover a partially published dataset.

A cache trades freshness and consistency for latency and cost. Define the key, value, TTL, invalidation trigger, and authority. Model-output caches must include every input that affects behavior: model/version, prompt/template, decoding settings, tool state where relevant, and authorization scope. Never share private cached responses across tenants through an incomplete key.

## Metadata, versioning, and lineage

Dataset metadata should include source, license/consent, collection time, schema, record counts, split rules, preprocessing version, quality checks, and known limitations. Content hashes can identify immutable artifacts.

Lineage answers: which sources and code produced this artifact, with what parameters, and which models/evaluations consumed it? Version raw, processed, and split manifests. A random seed alone cannot reproduce a split if upstream row order or source content changes.

Schema evolution should preserve meaning. Adding an optional field is usually easier than changing a field's unit or semantics. Use explicit versioning or migrations, test old and new readers, and distinguish missing, null, empty, and unknown values.

## Data-quality checks

Validate schema, uniqueness, range/domain, referential integrity, null rate, volume, distribution, freshness, and duplicates. Establish which failures block publication and which generate warnings. Compare against a prior accepted baseline with tolerances rather than asserting that every count is constant.

For ML data, also test label distribution, group leakage, temporal leakage, near duplicates, unsupported content, and train/validation overlap. Log rejected record counts and samples without exposing sensitive data.

## Checkpoint

1. Why is JSONL easier to stream than one large JSON array?
2. When is Parquet preferable to JSONL?
3. How does idempotence help an at-least-once pipeline?
4. Why are a dataset filename and random seed insufficient lineage?

<details>
<summary>Show answers</summary>

1. Each line is an independent record, so readers need not parse or retain the whole array and can identify line-level failures.
2. For typed, large-scale analytical reads where column projection, compression, and predicate pruning reduce IO.
3. Duplicate delivery produces the same state as one delivery when consumers identify the logical event and avoid repeating effects.
4. Content, schema, code, parameters, source versions, and ordering can change under the same filename. The seed controls only defined random operations.

</details>

## Exercise — Design a versioned ingestion pipeline

Design a pipeline that ingests support tickets, stores raw records, produces a redacted training dataset, and publishes train/validation manifests. Include retries, validation, lineage, and deletion.

<details>
<summary>Show exercise solution</summary>

Assign each source ticket a stable ID and ingestion timestamp. Write immutable encrypted raw objects under versioned keys, subject to access and retention policy. Validate schema and quarantine invalid records. Redact with a versioned transform, record input/output hashes and code/configuration version, and write partitioned processed data.

Split by customer and time before making training windows. Publish content-addressed train/validation manifests only after count, null, duplication, redaction, and overlap checks pass. Use event IDs and upserts for retry safety. Maintain a source-to-derived lineage index so deletion removes raw, processed, cached, and indexed copies and triggers manifest/model-impact review.

</details>

## Completion criteria

Choose among JSON/JSONL/Parquet, distinguish batch/streaming and ETL/ELT, design idempotent publication, and document versioning and lineage.

## Primary references

- [The JSON Data Interchange Syntax](https://www.rfc-editor.org/rfc/rfc8259)
- [JSON Lines](https://jsonlines.org/)
- [Apache Parquet documentation](https://parquet.apache.org/docs/)
- [The Data Engineering Lifecycle](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/)
