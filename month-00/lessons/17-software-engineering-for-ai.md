# 0.17 Software Engineering for AI — SKIM

ML code has a reputation for being research scripts that escaped into production. This lesson is about the engineering practices that make an AI system maintainable, testable, and debuggable — and about the ways AI systems differ from ordinary software, which is what makes them harder to test than most code you have written.

| | |
|---|---|
| **Mode** | SKIM |
| **Time** | 45–60 minutes |
| **Assumes** | 0.1 Python (classes, type hints, modules, logging, configuration) |
| **Used by** | the tiny text classifier project, 0.18 MLOps, Month 5 agent systems, Month 8 production |

[Month 0 roadmap](../README.md) · [Previous: Linux, Containers, and Git](16-linux-containers-and-git.md) · [Next: MLOps Fundamentals](18-mlops-fundamentals.md)

## Learning objectives

After this lesson you can:

- structure an AI project into modules with clear boundaries
- define an interface and inject a dependency, and say what that buys for testing
- drive behavior from configuration instead of edited constants
- write unit and integration tests for code that calls a model, including mocking
- distinguish programmer errors from environment errors and design a degraded path
- instrument a system with structured logging and the metrics that matter
- explain why ML systems are harder to test than ordinary software

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Section [0.17.4](#0174--testing-ml-code) is the one that changes how you work; most people under-test ML code because the obvious assertions are unavailable.
3. Record gaps in [`progress.md`](../progress.md).

## 0.17.1 — Modular structure

The default project layout for everything in this roadmap:

```text
project/
├── pyproject.toml
├── configs/
│   └── default.yaml
├── src/
│   └── package/
│       ├── data.py          # loading, tokenizing, batching
│       ├── model.py         # architecture only
│       ├── train.py         # the training loop
│       ├── evaluate.py      # metrics
│       └── predict.py       # inference entry point
└── tests/
```

The boundaries matter more than the names. Each module should be replaceable without touching the others: swapping the tokenizer should not require editing `model.py`, and changing the architecture should not require editing `train.py`.

Two rules that do most of the work:

- **Separate I/O from computation.** A function that both reads a file and transforms its contents cannot be tested without a file. Split it: one function reads, one transforms, and the transform is testable with a literal.
- **No work at import time.** A module that loads a model or reads a config when imported makes tests slow and startup unpredictable. Do that inside a function or a `main()` guard (0.1.15).

The `src/` layout specifically prevents a common failure: without it, Python may import your package from the working directory rather than the installed copy, so your tests pass against code that is not what ships.

## 0.17.2 — Interfaces and dependency injection

An **interface** is a contract: what a component accepts and returns, independent of how it does it. In Python, `typing.Protocol` expresses this without inheritance.

**Dependency injection** means passing a component's dependencies in rather than constructing them inside. It sounds bureaucratic; the payoff is concrete:

```python
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SearchService:
    def __init__(self, embedder: Embedder, index):
        self.embedder = embedder          # injected, not constructed here
        self.index = index

    def search(self, query: str, k: int = 5):
        vector = self.embedder.embed([query])[0]
        return self.index.query(vector, k)


class FakeEmbedder:
    """Deterministic stand-in: no network, no model, no GPU."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class FakeIndex:
    def query(self, vector, k):
        return [("doc-1", 0.9), ("doc-2", 0.8)][:k]


service = SearchService(FakeEmbedder(), FakeIndex())
print(service.search("hello world", k=1))       # [('doc-1', 0.9)]
```

`SearchService` never learns whether it is talking to a real embedding API or a fake, so its logic is testable in milliseconds with no model, no network, and no GPU. The same seam lets you switch providers in production without touching the service.

This is the highest-leverage pattern in the lesson. In AI systems the expensive, slow, non-deterministic parts — model calls, vector databases, external APIs — are exactly the parts you want behind an interface.

## 0.17.3 — Configuration-driven systems

Hyperparameters, paths, and model names belong in configuration, not scattered through the code. The test: **can you reproduce a run from one file?**

```python
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "tiny-classifier"
    embedding_dim: int = 128
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 10
    seed: int = 42


config = TrainConfig(batch_size=64)
print(asdict(config))
```

Practices worth adopting now:

- **A typed config object**, not a loose dictionary. A typo in `config["bath_size"]` returns `None` at 3 a.m.; a typo in a dataclass field fails immediately.
- **Precedence: defaults, then file, then environment, then command-line flags.** The most specific source wins.
- **Save the resolved config with every run's artifacts** (0.7.8, 0.14.5). The config *as used*, after all overrides, is what makes a result reproducible.
- **Secrets are not config.** They come from the environment or a secret manager and never appear in a file you commit (0.16.8).

Tools you will meet: `pydantic-settings` for validated settings, Hydra and OmegaConf for composable experiment configs.

## 0.17.4 — Testing ML code

The difficulty is that the obvious assertion is unavailable. You cannot assert a model's output equals an expected value, because the output depends on learned weights and often on sampling. So you test everything around it, and test the model with properties rather than exact values.

**What to test, in rough order of value:**

| Target | Example assertion |
|---|---|
| Shapes | `model(batch).shape == (B, num_classes)` |
| Invariants | extra padding does not change a prediction (0.10.5) |
| Boundaries | empty string, one token, a sequence longer than the maximum, unknown tokens |
| Determinism | the same seed and input produce the same output |
| The data pipeline | vocabulary lookup, encode/decode round trip, mask sums equal true lengths |
| Learning capacity | the model can overfit 8 examples to near-zero loss (0.7.7) |
| Serialization | save, reload, and get identical predictions |

```python
import torch
from torch import nn


def test_shapes_and_padding_invariance():
    torch.manual_seed(0)
    embedding = nn.Embedding(50, 8, padding_idx=0)
    head = nn.Linear(8, 3)

    def forward(tokens):
        mask = (tokens != 0).unsqueeze(-1).float()
        pooled = (embedding(tokens) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return head(pooled)

    short = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0]])

    assert forward(short).shape == (1, 3)
    assert torch.allclose(forward(short), forward(padded), atol=1e-6)


test_shapes_and_padding_invariance()
print("passed")
```

That single invariance test catches the most common bug in the entire Month 0 project.

**Unit versus integration.** A unit test covers one function in isolation with fakes, and should run in milliseconds. An integration test runs several real components together — tokenizer, model, and prediction path — and catches the interface mismatches unit tests cannot. You want many of the first and a few of the second.

**Mocking** replaces a slow or non-deterministic dependency with a controlled stand-in. Mock the LLM API, the vector database, and the filesystem; do not mock the thing you are actually testing. The dependency-injection seam from 0.17.2 usually makes an explicit mocking library unnecessary — passing a fake is simpler and less brittle.

A subtlety specific to this domain: **mocking hides contract drift.** Your fake embedder returns 3 dimensions forever while the real API moves to 1,024. That is why you keep a small number of tests that hit the real interface, run on a schedule rather than on every commit.

Finally, **test the data, not only the code**. Assertions about label distribution, duplicate rate, and train/test overlap (0.14.5) catch the failures that produce a perfectly functioning pipeline with a meaningless result.

## 0.17.5 — Error handling and resilience

AI systems depend on slow, remote, occasionally wrong components, so error handling is a design concern rather than an afterthought.

**Fail fast on programmer errors, degrade gracefully on environment errors.** A shape mismatch or a missing config key should raise immediately and loudly. A rate-limited API call or a timed-out vector store should be retried, then handled.

```python
class RetrievalUnavailable(Exception):
    """The index could not be reached; the caller may still answer without it."""


def answer(query: str, retriever, generator, *, logger=None) -> dict:
    context, degraded = [], False
    try:
        context = retriever.search(query, k=5)
    except RetrievalUnavailable:
        degraded = True                  # answer without context rather than failing entirely
        if logger:
            logger.warning('{"event": "retrieval_degraded"}')

    return {"text": generator.generate(query, context), "degraded": degraded}


class DeadRetriever:
    def search(self, query, k):
        raise RetrievalUnavailable


class StubGenerator:
    def generate(self, query, context):
        return f"answer to {query!r} using {len(context)} sources"


print(answer("what is BM25?", DeadRetriever(), StubGenerator()))
# {'text': "answer to 'what is BM25?' using 0 sources", 'degraded': True}
```

Practices specific to this domain:

- **Define your own exception types** at module boundaries. Letting a raw `httpx.ReadTimeout` escape into business logic couples every caller to your HTTP client.
- **Never swallow an exception silently.** A bare `except: pass` converts a loud failure into a silent quality regression, which is the failure mode ML systems are already worst at detecting.
- **Decide what degraded looks like before you need it.** Retrieval down, so answer without context and flag it. Reranker down, so return retrieval order. Primary model down, so fall back to a smaller one. Each is a product decision, and the flag in the response is what makes the degradation measurable.
- **Set a timeout on every external call** (0.15.4), and make it shorter than your caller's timeout, or you will hold connections nobody is waiting for.
- **Validate model output before trusting it.** A structured-output call can return malformed JSON or a value outside the allowed set. Parse, validate, and define the failure path: retry once, then fall back.
- **Make partial failure explicit.** In a batch job, one bad record should not abort 10,000 good ones; collect failures, report them, and continue.

## 0.17.6 — Observability: logging, metrics, tracing

**Logging.** Use the `logging` module, never `print` (0.1.10). Prefer **structured** logs — key-value or JSON — so they can be queried rather than grepped:

```python
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

logger.info(json.dumps({
    "event": "inference",
    "request_id": "req-123",
    "model": "tiny-classifier-v2",
    "input_tokens": 84,
    "latency_ms": 143,
}))
```

Level discipline: DEBUG for development detail, INFO for normal milestones, WARNING for recoverable anomalies, ERROR for failed operations. Log the **request ID** on every line so one request can be reconstructed from a million lines. Never log secrets, and be deliberate about logging user content — it may be sensitive, and it lands in systems with different retention rules.

**Metrics** for an AI system fall into three groups, and teams routinely instrument only the first:

| Group | Examples |
|---|---|
| System | latency (p50/p95/p99), throughput, error rate, saturation |
| Model | token counts, cost per request, cache hit rate, retrieval recall, refusal or fallback rate |
| Quality | user feedback, task success, human review scores, drift indicators |

Track **percentiles, not averages**. A mean latency of 200 ms hides a p99 of 8 seconds, and the p99 is what users complain about.

**Tracing** connects the stages of one request across services: retrieve, rerank, generate, post-process. In a multi-step agent or RAG pipeline it is the only practical way to answer "where did the 4 seconds go?"

**Reproducibility** is the observability of the past: seeds, versions, config, data version, and code commit recorded with every run (0.7.8). If you cannot rebuild a result, you cannot debug a regression in it.

## 0.17.7 — What makes AI systems different

Worth stating explicitly, because it explains why ordinary engineering practice is necessary but not sufficient:

- **Behavior lives in data and weights, not only code.** A correct codebase can produce a bad system, so code review alone is not quality control.
- **Outputs are often non-deterministic**, so exact-match assertions mostly do not apply.
- **Failures are silent.** A broken mask, a stale cache, or a leaked split degrades quality without raising an exception. Ordinary software crashes; ML software quietly gets worse.
- **Quality drifts** as the world changes, even with frozen code and weights (0.18).
- **The feedback loop is slow and expensive**: a full training run or evaluation is not a unit test.

The practices in this lesson exist to counter those properties. Interfaces and injection make the expensive parts fakeable, invariant tests catch the silent failures, and observability catches the drift.

## Exercises

### Exercise 1 — Refactor for injection

Take a function that constructs an embedding client inside itself and calls it directly. Refactor so the client is passed in behind a `Protocol`, then write a test using a fake that runs with no network and no model.

### Exercise 2 — A test suite for the Month 0 project

For the tiny text classifier, write tests covering: vocabulary lookup including `<unk>`, encode/decode round trip, batch shapes, mask sums equal true lengths, padding invariance of the pooled output, and checkpoint save/reload producing identical predictions.

### Exercise 3 — Structured logging and percentiles

Instrument a fake inference function to emit one structured log line per request with a request ID and latency. Generate 1,000 requests where 3% are slow, then compute mean, p50, p95, and p99 and explain what the mean hides.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
from typing import Protocol


# before: untestable without network access
def classify_before(text: str) -> str:
    client = RealEmbeddingClient(api_key="...")        # constructed inside
    vector = client.embed([text])[0]
    return "positive" if vector[0] > 0 else "negative"


# after: the dependency is a parameter
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def classify(text: str, embedder: Embedder) -> str:
    vector = embedder.embed([text])[0]
    return "positive" if vector[0] > 0 else "negative"


class FakeEmbedder:
    def __init__(self, value: float):
        self.value = value
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[self.value, 0.0]] * len(texts)


fake = FakeEmbedder(1.0)
assert classify("anything", fake) == "positive"
assert classify("anything", FakeEmbedder(-1.0)) == "negative"
assert fake.calls == [["anything"]]                  # also asserts how it was called
print("passed")
```

The fake records its calls, so the test can assert not just the result but that the collaborator was used correctly — batching one text rather than one call per character, for instance.

### Exercise 2

```python
import torch
from torch import nn
from collections import Counter


def build_vocabulary(documents: list[str]) -> dict[str, int]:
    counts = Counter(token for document in documents for token in document.lower().split())
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token, _ in counts.most_common():
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode(text: str, vocabulary: dict[str, int]) -> list[int]:
    return [vocabulary.get(token, 1) for token in text.lower().split()]


def pad_batch(sequences: list[list[int]]):
    longest = max(len(sequence) for sequence in sequences)
    tokens = torch.zeros(len(sequences), longest, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tokens[row, :len(sequence)] = torch.tensor(sequence)
    return tokens, tokens != 0


class Classifier(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 16, classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.head = nn.Linear(dim, classes)

    def forward(self, tokens, mask):
        weights = mask.unsqueeze(-1).float()
        pooled = (self.embedding(tokens) * weights).sum(1) / weights.sum(1).clamp(min=1)
        return self.head(pooled)


corpus = ["the movie was great", "dull and slow plot"]
vocabulary = build_vocabulary(corpus)
torch.manual_seed(0)
model = Classifier(len(vocabulary)).eval()


def test_vocabulary():
    assert vocabulary["<pad>"] == 0 and vocabulary["<unk>"] == 1
    assert encode("the unseenword", vocabulary) == [vocabulary["the"], 1]


def test_batch_shapes_and_mask():
    sequences = [encode(document, vocabulary) for document in corpus]
    tokens, mask = pad_batch(sequences)
    assert tokens.shape == (2, 4)
    assert mask.sum(dim=1).tolist() == [len(sequence) for sequence in sequences]


def test_padding_invariance():
    short = torch.tensor([[2, 3, 4]])
    padded = torch.tensor([[2, 3, 4, 0, 0, 0]])
    with torch.inference_mode():
        assert torch.allclose(model(short, short != 0), model(padded, padded != 0), atol=1e-6)


def test_checkpoint_round_trip():
    tokens = torch.tensor([[2, 3, 4]])
    with torch.inference_mode():
        before = model(tokens, tokens != 0)

    torch.save(model.state_dict(), "/tmp/classifier.pt")
    restored = Classifier(len(vocabulary))
    restored.load_state_dict(torch.load("/tmp/classifier.pt", weights_only=True))
    restored.eval()

    with torch.inference_mode():
        assert torch.allclose(before, restored(tokens, tokens != 0), atol=1e-6)


def test_empty_and_single_token():
    for sequence in [[1], [2]]:
        tokens, mask = pad_batch([sequence])
        with torch.inference_mode():
            assert model(tokens, mask).shape == (1, 2)


for test in [test_vocabulary, test_batch_shapes_and_mask, test_padding_invariance,
             test_checkpoint_round_trip, test_empty_and_single_token]:
    test()
    print(f"{test.__name__} passed")
```

Under pytest these become a real suite unchanged; the loop at the bottom is only so the example runs standalone. Note that not one of them asserts a specific prediction value — every assertion is a shape, an invariant, or a round trip, which is what testing ML code looks like.

### Exercise 3

```python
import json
import logging
import random
import statistics

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("inference")

random.seed(0)
latencies = []

for i in range(1_000):
    latency = random.gauss(120, 20) if random.random() > 0.03 else random.gauss(4_000, 500)
    latency = max(1.0, latency)
    latencies.append(latency)
    if i < 2:                                     # log a couple for illustration
        logger.info(json.dumps({
            "event": "inference",
            "request_id": f"req-{i:04d}",
            "model": "tiny-classifier-v2",
            "latency_ms": round(latency, 1),
        }))

latencies.sort()
def percentile(p: float) -> float:
    return latencies[min(len(latencies) - 1, int(len(latencies) * p))]

print(f"mean {statistics.mean(latencies):7.1f} ms")
print(f"p50  {percentile(0.50):7.1f} ms")
print(f"p95  {percentile(0.95):7.1f} ms")
print(f"p99  {percentile(0.99):7.1f} ms")
print(f"max  {latencies[-1]:7.1f} ms")
```

```text
mean   237.7 ms
p50    121.2 ms
p95    164.2 ms
p99   4123.9 ms
max   4732.5 ms
```

The mean of 238 ms looks like a slightly slow but healthy service, and the p95 of 164 ms looks fine. The p99 is 4.1 seconds — thirty-four times the median — because 3% of requests take four seconds. Those are real users having a bad experience, and a dashboard showing only the mean displays nothing unusual at all.

Note also how much the cutoff matters: at a 1% slow rate the p99 would sit right at the boundary and miss the tail entirely. Track p95, p99, and max together, alert on the high percentiles, and use the mean only for capacity math.

</details>

## Exit test

1. Why separate I/O from computation?
2. Why should a module do no work at import time?
3. What is dependency injection, and what does it buy in an AI system specifically?
4. Why prefer a typed config object over a dictionary?
5. What is the precedence order for configuration sources?
6. Why can you rarely assert an exact model output, and what do you assert instead?
7. Name five things worth testing in an ML pipeline.
8. What is the difference between a unit and an integration test here?
9. What should you mock, and what risk does mocking introduce?
10. Why structured logging rather than plain text, and what field matters most?
11. Name the three groups of metrics for an AI system.
12. Why track percentiles rather than averages?
13. What is the difference between a programmer error and an environment error, and why is `except: pass` especially dangerous in an ML system?

<details>
<summary>Show answers</summary>

1. So the computation can be tested with literal values, with no file, network, or database required.
2. Import-time work makes tests slow, makes startup order significant, and can trigger side effects such as loading a model or reading a config simply because a module was imported.
3. Passing a component's dependencies in rather than constructing them internally. In AI systems it puts the slow, expensive, non-deterministic parts — model calls, vector stores, external APIs — behind a seam that can be replaced with a fake in tests or a different provider in production.
4. A typo in a dictionary key silently returns `None` or a default; a typo in a dataclass field raises immediately, and the types are checkable and self-documenting.
5. Defaults, then a config file, then environment variables, then command-line flags, with the most specific winning.
6. Because outputs depend on learned weights and often on sampling, so they are not stable known values. You assert shapes, invariants, boundary behavior, determinism under a fixed seed, round trips, and the ability to overfit a tiny batch.
7. Output shapes, padding invariance, boundary inputs, vocabulary and encode/decode round trips, mask correctness, checkpoint save/reload, and the capacity to overfit 8 examples.
8. A unit test exercises one function in isolation with fakes and runs in milliseconds. An integration test runs several real components together and catches interface mismatches between them.
9. Mock slow or non-deterministic dependencies such as LLM APIs, vector databases, and the filesystem, never the code under test. The risk is contract drift: the fake keeps matching an interface the real service has changed, so you need a few real-interface tests on a schedule.
10. Structured logs can be queried and aggregated by field rather than grepped. The request ID matters most, because it is what lets you reconstruct one request's path through a large log volume.
11. System metrics (latency, throughput, errors, saturation), model metrics (tokens, cost, cache hits, retrieval recall, fallback rate), and quality metrics (user feedback, task success, drift).
12. Averages hide the tail. A healthy-looking mean can coexist with a p99 many times larger, and the tail is what users actually experience.
13. A programmer error is a bug such as a shape mismatch or missing config, and it should fail fast and loudly. An environment error is external and transient, such as a timeout or rate limit, and it should be retried and then degraded gracefully. Swallowing an exception is especially dangerous here because ML systems already fail silently: the pipeline keeps returning plausible output while quality drops, and the one loud signal you had is gone.

</details>

## Completion criteria

You are done when:

- you can structure a project so each module is replaceable in isolation
- you can refactor a hard-coded dependency into an injected one and test it with a fake
- you can list what to test in an ML pipeline without reaching for exact output assertions
- you can explain why ML failures are usually silent, and what catches them
- your three exercises run, including the full Month 0 project test suite

## References

**Structure and design**
- [Architecture Patterns with Python, Percival and Gregory](https://www.cosmicpython.com/) — free online; dependency inversion and ports/adapters
- [The Twelve-Factor App](https://12factor.net/) — config, logs, and dependencies
- [Python Packaging User Guide: src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)

**Configuration**
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Hydra](https://hydra.cc/docs/intro/)

**Testing**
- [pytest documentation](https://docs.pytest.org/)
- [Effective testing for machine learning systems, Jeremy Jordan](https://www.jeremyjordan.me/testing-ml/)
- [How to Test Machine Learning Code, Chase Roberts](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765)
- [hypothesis](https://hypothesis.readthedocs.io/) — property-based testing, a natural fit for invariants

**Resilience and observability**
- [Python `logging` HOWTO](https://docs.python.org/3/howto/logging.html)
- [Google Testing Blog: test sizes](https://testing.googleblog.com/2010/12/test-sizes.html)
- [Semantic Versioning](https://semver.org/) — versioned contracts between components
- [OpenTelemetry specification](https://opentelemetry.io/docs/specs/)
- [structlog](https://www.structlog.org/)
- [OpenTelemetry](https://opentelemetry.io/docs/) — tracing across services
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) — the canonical account of why ML systems rot

## About this lesson

Written to cover section 0.17 of the [Month 0 curriculum](../README.md). Code examples were checked with Python 3.12 and PyTorch 2.14.
