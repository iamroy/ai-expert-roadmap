# 0.14 Basic Data Engineering — SKIM

Model quality is bounded by data quality, and most of the work in a real AI system is moving, cleaning, and versioning data rather than training. This lesson covers enough vocabulary and enough of the file formats to make the rest of the roadmap legible.

| | |
|---|---|
| **Mode** | SKIM |
| **Time** | 45–60 minutes |
| **Assumes** | 0.1 Python (JSON, generators) |
| **Used by** | the tiny text classifier project, Month 4 document pipelines, Month 7 training infrastructure, Month 8 production systems |

[Month 0 roadmap](../README.md) · [Previous: Information Retrieval](13-information-retrieval-fundamentals.md) · [Next: APIs and Web Fundamentals](15-apis-and-web-fundamentals.md)

## Learning objectives

After this lesson you can:

- distinguish structured, semi-structured, and unstructured data, and say why it matters for AI systems
- choose between JSON, JSONL, Parquet, and Arrow for a given job
- explain batch versus streaming, and ETL versus ELT
- describe object storage and the caching layers in an AI system
- explain why dataset versioning and lineage are not optional

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Section [0.14.2](#0142--file-formats-json-jsonl-parquet-arrow) is the practical one; you will use JSONL immediately.
3. Record gaps in [`progress.md`](../progress.md).

Examples assume:

```python
import json
from pathlib import Path
```

## 0.14.1 — Structured, semi-structured, and unstructured

| Type | Examples | Handling |
|---|---|---|
| **Structured** | SQL tables, CSV, Parquet | fixed schema, queryable directly |
| **Semi-structured** | JSON, JSONL, XML, logs | flexible fields, needs parsing and validation |
| **Unstructured** | text, PDFs, images, audio, video | no schema at all; needs a model to extract meaning |

The relevant point for this roadmap: **unstructured data is exactly where AI adds value, and it is also where it is hardest to work.** Classical analytics needed data to be structured first, usually by people. Embeddings and LLMs convert unstructured input into something queryable, which is the engine behind semantic search and document extraction.

A practical middle ground worth naming: most production AI pipelines end up storing unstructured payloads alongside structured metadata, such as a text chunk plus its source, page, timestamp, permissions, and embedding. Month 4 uses exactly this shape.

## 0.14.2 — File formats: JSON, JSONL, Parquet, Arrow

| Format | Shape | Good for | Weak at |
|---|---|---|---|
| **CSV** | rows of text | interchange, spreadsheets | no types, quoting and encoding pain |
| **JSON** | one nested object | configs, API payloads | must load the whole file to read any of it |
| **JSONL** | one JSON object per line | training data, logs, batch jobs | no compression or column pruning |
| **Parquet** | columnar, compressed, typed | analytics, large datasets | not human-readable, poor for appends |
| **Arrow** | columnar, in memory | zero-copy sharing between tools | an in-memory format, not a storage one |

**JSONL is the workhorse of ML data.** It streams (0.1.8), appends cheaply, survives partial writes, and is readable with any text tool:

```python
records = [
    {"id": 1, "text": "great film", "label": 1},
    {"id": 2, "text": "dull plot", "label": 0},
]

path = Path("/tmp/train.jsonl")
with path.open("w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

with path.open(encoding="utf-8") as file:
    loaded = [json.loads(line) for line in file if line.strip()]
print(loaded[0], len(loaded))
```

**Parquet** stores by column rather than by row. Three consequences follow:

- Reading one column of a 200-column dataset touches only that column's bytes.
- Columns compress far better than rows, because similar values sit together. Ten-fold reductions over JSONL are common.
- Types and a schema are stored in the file, so numbers stay numbers.

The rule of thumb: **JSONL while data is being produced and inspected; Parquet once it is large and read repeatedly.**

## 0.14.3 — Batch and streaming, ETL and ELT

**Batch** processes accumulated data on a schedule: nightly embedding jobs, weekly retraining, offline evaluation. It is simple, easy to retry, and adds latency.

**Streaming** processes records as they arrive: live ingestion, incremental index updates, online feature computation. It gives freshness, at the cost of harder failure handling, ordering, and exactly-once semantics.

Most real systems are hybrid: a large batch backfill plus a stream for new records.

**ETL** extracts, transforms, then loads the cleaned result. **ELT** loads raw data first and transforms inside the destination system. ELT has become the default in modern warehouses because storage is cheap and it preserves the raw data, which means a transformation bug is fixable without re-ingesting.

For AI specifically, prefer keeping the **raw source** alongside the processed form. Chunking strategies, embedding models, and cleaning rules all change, and every change means reprocessing from the original. A pipeline that discards its input is a pipeline you will rebuild.

**Idempotency** is the property that makes any of this survivable: running a job twice produces the same result as once. Without it, a retry after a partial failure silently duplicates records, and duplicates in a training set are both a leakage risk (0.7.6) and a retrieval quality problem.

## 0.14.4 — Object storage and caching

**Object storage** (S3, GCS, Azure Blob) is the default home for training data, checkpoints, and artifacts. The characteristics that matter:

- Flat key-value storage, not a filesystem. "Directories" are just key prefixes.
- Cheap, effectively unlimited, and highly durable.
- High latency per request, so many small objects are slow and expensive. Prefer a few large files over millions of tiny ones; this is why sharded formats such as WebDataset tar shards exist.
- Objects are immutable in practice; you replace rather than edit, which pairs naturally with versioned artifacts.

**Caching** appears at several layers in an AI system, and each is worth recognizing now:

| Cache | Holds | Why |
|---|---|---|
| Data loader prefetch | upcoming batches | keeps the GPU fed (0.8.4) |
| Local disk cache | data pulled from object storage | avoids repeated network fetches |
| Embedding cache | vectors keyed by content hash | embedding is expensive and deterministic |
| LLM response cache | outputs keyed by prompt | avoids paying twice for identical requests |
| KV cache | attention keys and values during generation | avoids recomputing the prefix per token (0.9) |

A cache needs a key that captures **everything** affecting the result. An embedding cache keyed on text alone returns stale vectors the moment you change embedding model, so the key must include the model name and version. That mistake is common and quiet.

## 0.14.5 — Metadata, versioning, and lineage

**Metadata** is data about the data: source, timestamp, license, language, chunk boundaries, permissions. In retrieval systems this is not decoration. Filtering by permission or recency at query time is a metadata operation, and getting it wrong is how a RAG system leaks a document to the wrong user.

**Dataset versioning** means being able to say exactly which data produced a given model. Without it you cannot reproduce a result, attribute a regression to a data change, or comply with a deletion request. Content-addressed hashing, manifests, and tools such as DVC, LakeFS, or Hugging Face dataset revisions all serve this.

**Lineage** is the traceable path from raw source to model artifact: which documents, which cleaning code, which chunking parameters, which embedding model. When quality drops you need to answer "what changed?" and lineage is what makes the question answerable.

A minimal practice that costs little and pays immediately: write a small manifest next to every processed dataset.

```python
import hashlib

def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

manifest = {
    "dataset": "reviews-train",
    "version": "2026-09-13",
    "source": "s3://raw/reviews/2026-09/",
    "records": 2,
    "sha256_prefix": file_digest(path),
    "code_commit": "abc1234",
    "notes": "regex tokenizer, min_count=2, dedup by normalized text",
}
print(json.dumps(manifest, indent=2))
```

That is enough to answer, months later, which data a checkpoint saw.

**Data quality checks** worth running on any dataset before training: exact and near-duplicate detection, train/test overlap (0.7.6), label distribution, length distribution and outliers, encoding problems, and personally identifiable information. Duplicates deserve special attention, because near-duplicates across splits inflate validation scores in exactly the way that is hardest to notice.

## Exercises

### Exercise 1 — JSONL round trip with validation

Write a JSONL file of 1,000 synthetic records, then stream it back with a generator that validates each record, reports the line number of any malformed line, and counts label distribution without loading the file into memory. Deliberately corrupt one line and confirm the error names it.

### Exercise 2 — Compare formats on the same data

Write the same 50,000 records as JSON, JSONL, and CSV, and compare file sizes and the time to read a single field from each. If `pyarrow` is available, add Parquet. Explain the pattern.

### Exercise 3 — Cache keys that are actually correct

Build an embedding cache keyed only on text, then show it returning a stale vector after switching "models". Fix the key and confirm the problem disappears.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
import random
from collections import Counter
from collections.abc import Iterator

path = Path("/tmp/exercise.jsonl")
random.seed(0)

with path.open("w", encoding="utf-8") as file:
    for i in range(1_000):
        record = {"id": i, "text": f"review number {i}", "label": random.randint(0, 1)}
        file.write(json.dumps(record) + "\n")


def read_validated(path: Path) -> Iterator[dict]:
    required = {"id", "text", "label"}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from error
            missing = required - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
            yield record


labels = Counter(record["label"] for record in read_validated(path))
print(f"clean file: {sum(labels.values())} records, distribution {dict(labels)}")

lines = path.read_text(encoding="utf-8").splitlines()
lines[499] = '{"id": 499, "text": "broken"'                 # truncated JSON
Path("/tmp/broken.jsonl").write_text("\n".join(lines), encoding="utf-8")

try:
    list(read_validated(Path("/tmp/broken.jsonl")))
except ValueError as error:
    print("caught:", error)
```

Streaming keeps memory flat regardless of file size, and the line number turns "the job failed" into a one-second fix. Validating at the boundary is the runtime check that type hints cannot give you (0.1.6).

### Exercise 2

```python
import csv
import time

records = [{"id": i, "text": f"document number {i}", "label": i % 2} for i in range(50_000)]

json_path, jsonl_path, csv_path = Path("/tmp/a.json"), Path("/tmp/a.jsonl"), Path("/tmp/a.csv")
json_path.write_text(json.dumps(records), encoding="utf-8")
with jsonl_path.open("w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record) + "\n")
with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["id", "text", "label"])
    writer.writeheader()
    writer.writerows(records)

for path in [json_path, jsonl_path, csv_path]:
    start = time.perf_counter()
    if path.suffix == ".json":
        total = sum(record["label"] for record in json.loads(path.read_text()))
    elif path.suffix == ".jsonl":
        total = sum(json.loads(line)["label"] for line in path.open(encoding="utf-8"))
    else:
        total = sum(int(row["label"]) for row in csv.DictReader(path.open(encoding="utf-8")))
    elapsed = time.perf_counter() - start
    print(f"{path.name:9s} {path.stat().st_size / 1e6:6.2f} MB  sum in {elapsed * 1000:6.1f} ms")
```

```text
a.json      2.98 MB  sum in    35-83 ms
a.jsonl     2.93 MB  sum in    69-79 ms
a.csv       1.53 MB  sum in    55-65 ms
```

Sizes are stable across runs; timings are not, so treat them as rough. JSON and JSONL come out nearly identical in size because both repeat every field name on every record. CSV writes the field names once and is roughly half the size.

The timings are the less interesting half here, because at 3 MB everything fits in memory and the differences are mostly parser overhead. What matters is how each behaves when the file does *not* fit: JSONL and CSV still stream a line at a time with flat memory, while `json.loads` on a single 50 GB object simply cannot run. Parquet would beat all three on size and would win the single-column read outright, since summing one column touches only that column's compressed bytes instead of parsing every field of every record.

### Exercise 3

```python
import hashlib

def fake_embed(text: str, model: str) -> list[float]:
    seed = int(hashlib.sha256(f"{model}:{text}".encode()).hexdigest()[:8], 16)
    return [((seed >> shift) & 0xFF) / 255 for shift in (0, 8, 16)]

text = "the quick brown fox"

bad_cache: dict[str, list[float]] = {}
def embed_bad(text: str, model: str) -> list[float]:
    if text not in bad_cache:                                  # key ignores the model
        bad_cache[text] = fake_embed(text, model)
    return bad_cache[text]

v1 = embed_bad(text, "encoder-v1")
v2 = embed_bad(text, "encoder-v2")                             # different model, same key
print(f"bad cache:  v1 == v2 -> {v1 == v2}   (stale: v2 never computed)")

good_cache: dict[tuple[str, str], list[float]] = {}
def embed_good(text: str, model: str) -> list[float]:
    key = (model, hashlib.sha256(text.encode()).hexdigest())
    if key not in good_cache:
        good_cache[key] = fake_embed(text, model)
    return good_cache[key]

print(f"good cache: v1 == v2 -> {embed_good(text, 'encoder-v1') == embed_good(text, 'encoder-v2')}")
```

The broken cache silently serves v1 vectors for v2 requests. Nothing errors; retrieval just degrades, because the index now mixes vectors from two different spaces, and comparing them is meaningless. Any cache key must include every input that changes the output, which for embeddings means at minimum the model identifier, its version, and any normalization or truncation settings.

</details>

## Exit test

1. Distinguish structured, semi-structured, and unstructured data.
2. Why is unstructured data both the opportunity and the difficulty in AI systems?
3. When would you choose JSONL over JSON? Over Parquet?
4. Name two advantages of columnar storage.
5. What is the difference between batch and streaming, and what does each cost?
6. What is the difference between ETL and ELT, and why has ELT become common?
7. Why keep raw data after processing it?
8. What is idempotency, and why does it matter for data pipelines?
9. Why are many small objects a problem in object storage?
10. Name three caches in an AI system, and state what must be in an embedding cache key.
11. Why does dataset versioning matter beyond reproducibility?
12. Name four data quality checks to run before training.

<details>
<summary>Show answers</summary>

1. Structured data has a fixed schema (tables, Parquet). Semi-structured data has flexible fields but some self-describing shape (JSON, logs). Unstructured data has no schema (text, images, audio).
2. It is where most real information lives and where classical tooling could not operate, so AI adds the most value there. It is also schema-free, noisy, and requires a model just to extract anything queryable.
3. JSONL over JSON when the file is large or appended to, because it streams and survives partial writes. JSON over JSONL for a single nested config. Parquet over JSONL once the data is large and read repeatedly, for compression and column pruning.
4. Reading only the needed columns, and much better compression because similar values are stored adjacently. Stored types and schema are a third.
5. Batch processes accumulated data on a schedule and is simple and easy to retry, at the cost of latency. Streaming processes records on arrival and is fresh, at the cost of harder ordering, failure handling, and exactly-once semantics.
6. ETL transforms before loading; ELT loads raw and transforms in the destination. ELT is common because storage is cheap and keeping the raw data means a transformation bug can be fixed without re-ingesting.
7. Because chunking, cleaning, and embedding choices change, and every change requires reprocessing from the original source.
8. Running a job twice yields the same result as running it once. Without it, a retry after partial failure duplicates records, creating both leakage risk and retrieval quality problems.
9. Object storage has high per-request latency, so millions of small objects are slow and costly. Fewer, larger, sharded files are much more efficient.
10. Data loader prefetch, local disk cache, embedding cache, LLM response cache, KV cache. An embedding cache key must include the text, the model identifier and version, and any normalization or truncation settings.
11. It underpins regression attribution, compliance and deletion requests, and lineage: knowing which data produced which model is what makes "what changed?" answerable.
12. Duplicate and near-duplicate detection, train/test overlap, label distribution, length distribution and outliers, encoding problems, and PII scanning.

</details>

## Completion criteria

You are done when:

- you can choose a format for a given dataset and justify it
- you can stream and validate a JSONL file without loading it into memory
- you can explain idempotency and why caches need complete keys
- you can describe what a dataset manifest should contain
- your three exercises run, including the stale-cache demonstration

## References

**Formats**
- [JSON Lines](https://jsonlines.org/)
- [RFC 8259: the JSON data interchange format](https://www.rfc-editor.org/rfc/rfc8259)
- [Fundamentals of Data Engineering, Reis and Housley](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/)
- [Apache Parquet](https://parquet.apache.org/docs/)
- [Apache Arrow](https://arrow.apache.org/overview/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) — Arrow-backed, memory-mapped

**Pipelines and storage**
- [Designing Data-Intensive Applications, Martin Kleppmann](https://dataintensive.net/) — the standard text; chapters 1, 3, 10, 11
- [AWS S3 performance guidelines](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html)
- [WebDataset](https://github.com/webdataset/webdataset) — sharded data for large-scale training

**Versioning, lineage, and quality**
- [DVC: data version control](https://dvc.org/doc/start/data-management)
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

## Videos and code to read

- [huggingface/datasets](https://github.com/huggingface/datasets) — Arrow-backed, memory-mapped dataset handling; a practical model for streaming data larger than RAM
- [webdataset/webdataset](https://github.com/webdataset/webdataset) — sharded storage for large-scale training, and the answer to the many-small-objects problem in 0.14.4
- [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML) — data pipeline, versioning, and quality checks in one working repository

## About this lesson

Written to cover section 0.14 of the [Month 0 curriculum](../README.md). Code examples were checked with Python 3.12.
