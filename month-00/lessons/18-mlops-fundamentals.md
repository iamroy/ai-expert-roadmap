# 0.18 MLOps Fundamentals — SKIM

MLOps is what keeps a model useful after it works once on your machine. The difference from ordinary DevOps is that an ML system has three moving parts instead of one — code, data, and model — and quality can degrade while all three sit perfectly still.

| | |
|---|---|
| **Mode** | SKIM |
| **Time** | 45–60 minutes |
| **Assumes** | 0.7 Training (checkpoints, reproducibility), 0.14 Data Engineering, 0.17 Software Engineering |
| **Used by** | Month 7 training infrastructure, Month 8 deployment and serving, Month 11 the capstone |

[Month 0 roadmap](../README.md) · [Previous: Software Engineering for AI](17-software-engineering-for-ai.md)

## Learning objectives

After this lesson you can:

- explain what experiment tracking records and why "I'll remember" fails
- describe a model registry, artifact storage, and promotion stages
- explain versioning across code, data, model, and environment
- describe CI/CD for an ML system and what tests gate a deployment
- name the deployment patterns and when each applies
- distinguish data drift from concept drift and describe how each is detected
- explain what makes rollback possible

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Section [0.18.6](#0186--monitoring-and-drift) is the one that separates a demo from a system.
3. Record gaps in [`progress.md`](../progress.md).

## 0.18.1 — Experiment tracking

A training run produces a number. Two weeks and forty runs later, the question "what produced the 0.91?" is unanswerable unless something recorded it.

What to record for **every** run:

| Category | Items |
|---|---|
| Config | all hyperparameters, resolved after overrides (0.17.3) |
| Code | git commit hash, and whether the tree was dirty |
| Data | dataset version or manifest hash (0.14.5) |
| Environment | library versions, CUDA version, hardware |
| Metrics | train and validation loss per step, final evaluation numbers |
| Artifacts | checkpoints, plots, sample predictions |
| Context | a one-line note on what this run was testing |

```python
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


def start_run(name: str, config: dict, data_version: str) -> Path:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{name}-{uuid.uuid4().hex[:6]}"
    run_directory = Path("/tmp/runs") / run_id          # unique: runs can start in the same second
    run_directory.mkdir(parents=True, exist_ok=True)

    (run_directory / "meta.json").write_text(json.dumps({
        "name": name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "code_commit": commit,
        "data_version": data_version,
    }, indent=2))
    return run_directory


run = start_run("baseline", {"lr": 3e-4, "batch_size": 32, "seed": 42}, "reviews-2026-09-13")
(run / "metrics.jsonl").write_text(
    '{"step": 100, "train_loss": 0.42, "val_loss": 0.51}\n'
    '{"step": 200, "train_loss": 0.31, "val_loss": 0.48}\n'
)
print(run.name)
print(json.loads((run / "meta.json").read_text())["code_commit"])
```

That is the minimum viable tracker, and it is already enough to answer most questions. Real tools — MLflow, Weights & Biases, Neptune — add a UI, run comparison, and artifact handling, but the discipline is the same: **record the inputs, not just the outputs**.

The "I'll remember which config that was" assumption fails reliably, usually a week before someone asks you to reproduce the result.

## 0.18.2 — Model registries and artifact storage

**Artifact storage** holds the large binary outputs: checkpoints, tokenizers, embeddings, evaluation reports. Object storage is the usual home (0.14.4), addressed by run ID.

A **model registry** sits above it and tracks *which* model is which, with:

- a name and a version number
- a pointer to the artifact and to the run that produced it
- a stage: `staging`, `production`, `archived`
- evaluation metrics and approval metadata
- lineage back to the data and code

The registry is what makes "which model is serving traffic right now, and what was it trained on?" a one-query question rather than an archaeology project.

Practical points:

- Prefer **safetensors** over pickle-based formats for weights. A pickle executes arbitrary code on load, which is a real risk for any model you did not produce yourself (0.1.16, 0.8.6).
- Store the **tokenizer and preprocessing config with the weights**. A model and a mismatched tokenizer produce fluent nonsense, and this is a common production incident.
- Model artifacts are immutable. A new set of weights is a new version, never an overwrite.

## 0.18.3 — Versioning: code, data, model, environment

Reproducing a result requires all four to be pinned:

| Dimension | Mechanism |
|---|---|
| Code | git commit hash |
| Data | dataset version, manifest, or content hash (0.14.5) |
| Model | registry version, or a checkpoint hash |
| Environment | lockfile, plus a container image digest (0.16.6) |

The failure that motivates this: a model's quality drops and every obvious thing looks unchanged. Without data versioning you cannot tell whether the training set changed; without an environment pin you cannot tell whether a transitive dependency upgraded and changed a tokenizer's behavior.

Note that a **lockfile is necessary but not sufficient** — it pins Python packages, not the CUDA driver, the base image, or the system libraries. A container image digest pins the rest.

## 0.18.4 — CI/CD for ML

Continuous integration for an ML repository runs on every commit and should be fast:

```text
lint and type-check  →  unit tests (fakes, no model)  →  data validation  →  a tiny training smoke run
```

The **smoke run** is the ML-specific piece: train for a handful of steps on a few examples and assert the loss decreases and nothing crashes. It catches shape errors, device errors, and broken configs in a minute rather than six hours into a real run.

Continuous delivery for a model is different from shipping code, because the artifact is not built by CI — it is produced by a training run and *promoted*:

```text
train  →  evaluate against a held-out suite  →  compare to the current production model
       →  human approval  →  staging  →  canary  →  production
```

The gate that matters is the comparison. A new model must beat the incumbent on the evaluation suite *and* not regress on the specific cases you care about. A model with a better average and a new failure on your highest-value query is not an improvement.

**Deployment patterns:**

| Pattern | Mechanism | Use when |
|---|---|---|
| Shadow | new model receives a copy of live traffic; output is logged, not served | validating on real traffic with zero user risk |
| Canary | a small percentage of traffic goes to the new model | you want real feedback with bounded blast radius |
| Blue/green | two full environments; switch traffic at once | you need instant, complete rollback |
| A/B test | traffic split with measurement | comparing on a business metric, not an offline one |

Shadow deployment is underused and especially valuable for LLM systems, where offline evaluation correlates imperfectly with real usage.

## 0.18.5 — Evaluation as a gate

Offline evaluation for AI systems has a specific failure mode: the suite drifts out of alignment with what users do. Practices that help:

- **Keep a regression set** of cases that previously broke. Every incident adds one. This set never shrinks.
- **Separate capability from safety evaluation.** A model that improves on task accuracy while regressing on refusals is not promotable.
- **Track per-slice results**, not only aggregates. Average improvement plus a large regression on one segment is a common and harmful pattern.
- **Version the evaluation suite itself.** A metric change is not a model change, and confusing the two wastes days.

Month 6 covers evaluation properly; for now, the point is that the gate must exist and must be automated enough to run on every candidate.

## 0.18.6 — Monitoring and drift

Deployment is where ML diverges most from ordinary software. A web service that is up and returning 200s is working. An ML service can be up, fast, error-free, and producing steadily worse answers.

**Drift** comes in two kinds:

| Type | What changed | Example |
|---|---|---|
| **Data drift** (covariate shift) | the input distribution | users start asking about a product that did not exist at training time |
| **Concept drift** | the relationship between input and correct output | "positive review" language changes; fraud patterns adapt |

Data drift is detectable without labels: compare the distribution of current inputs to the training distribution, using embedding statistics, a population stability index, or a KS test on features. Concept drift generally is not, because detecting it requires knowing the right answer — which is why you need a feedback channel: user ratings, downstream outcomes, or periodic human review of a sample.

**What to monitor**, building on 0.17.5:

- System: latency percentiles, error rate, throughput, saturation
- Model: prediction distribution, confidence distribution, fallback and refusal rates, token cost, cache hit rate
- Retrieval systems: recall proxies, empty-result rate, chunk age
- Quality: user feedback, task success, human review scores on a sample

A shifting **prediction distribution** is the cheapest early warning available. If a classifier's positive rate moves from 30% to 55% over a week with no product change, something upstream changed and you want to know before users tell you.

**Rollback** is the last line of defense, and it is a property you build in rather than a procedure you write down. It requires: immutable versioned artifacts, the previous version still deployable, config as data rather than code, and no irreversible migration coupled to the release. Practice it. A rollback path that has never been executed is a hypothesis.

Finally, **feedback loops** deserve explicit attention in AI systems. A model that influences the data it is later trained on can amplify its own biases — a recommender that only ever shows what it already ranks highly will never learn that anything else was good. Breaking the loop usually means deliberate exploration and holding out a slice of unmodified traffic.

## Exercises

### Exercise 1 — A minimal experiment tracker

Extend the tracker above: log metrics per step to JSONL, save the final checkpoint path, mark a run as finished with a status, then write a function that loads all runs and prints a comparison table sorted by validation loss. Run it three times with different learning rates.

### Exercise 2 — A promotion gate

Write `should_promote(candidate, incumbent, regression_set)` that returns a decision and a reason. It should require the candidate to beat the incumbent on the aggregate metric by a margin, fail no case in the regression set, and regress no slice by more than a threshold. Test it against a candidate that improves the average while breaking one slice.

### Exercise 3 — Detect data drift

Generate a baseline distribution and three current distributions: identical, slightly shifted, and heavily shifted. Compute a population stability index for each and set a threshold that flags the last two but not the first.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
def log_metrics(run_directory: Path, **values) -> None:
    with (run_directory / "metrics.jsonl").open("a") as file:
        file.write(json.dumps(values) + "\n")


def finish_run(run_directory: Path, status: str, checkpoint: str | None = None) -> None:
    meta = json.loads((run_directory / "meta.json").read_text())
    meta.update({"status": status, "checkpoint": checkpoint,
                 "finished_at": datetime.now(timezone.utc).isoformat()})
    (run_directory / "meta.json").write_text(json.dumps(meta, indent=2))


def load_runs(root: Path = Path("/tmp/runs")) -> list[dict]:
    runs = []
    for directory in sorted(root.iterdir()):
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        metrics_path = directory / "metrics.jsonl"
        history = [json.loads(line) for line in metrics_path.read_text().splitlines()] \
            if metrics_path.exists() else []
        best = min((row["val_loss"] for row in history if "val_loss" in row), default=float("inf"))
        runs.append({"run": directory.name, "lr": meta["config"].get("lr"),
                     "status": meta.get("status", "running"), "best_val": best})
    return sorted(runs, key=lambda row: row["best_val"])


import shutil
shutil.rmtree("/tmp/runs", ignore_errors=True)

for learning_rate, losses in [(1e-2, [0.9, 0.7]), (3e-4, [0.6, 0.42]), (1e-5, [1.1, 1.05])]:
    run = start_run("sweep", {"lr": learning_rate, "batch_size": 32}, "reviews-2026-09-13")
    for step, val_loss in enumerate(losses, start=1):
        log_metrics(run, step=step * 100, val_loss=val_loss)
    finish_run(run, "completed", checkpoint=str(run / "model.pt"))

print(f"{'run':32s} {'lr':>8s} {'status':>10s} {'best val':>9s}")
for row in load_runs():
    print(f"{row['run']:32s} {row['lr']:>8.0e} {row['status']:>10s} {row['best_val']:>9.4f}")
```

```text
run                                    lr     status  best val
20260914T013720-sweep-672f3c        3e-04  completed    0.4200
20260914T013720-sweep-92683a        1e-02  completed    0.7000
20260914T013720-sweep-8945b7        1e-05  completed    1.0500
```

The comparison table is the entire point: forty runs later, this is the difference between a five-second answer and re-running experiments you already ran.

Note the random suffix on each run ID. Three runs launched in the same second would otherwise collide on a timestamp alone and silently overwrite each other's metadata, leaving one run in the table instead of three.

### Exercise 2

```python
from dataclasses import dataclass, field


@dataclass
class Evaluation:
    accuracy: float
    slices: dict[str, float] = field(default_factory=dict)
    regression_passed: set[str] = field(default_factory=set)


def should_promote(candidate: Evaluation, incumbent: Evaluation, regression_set: set[str],
                   margin: float = 0.005, max_slice_regression: float = 0.02):
    if candidate.accuracy < incumbent.accuracy + margin:
        return False, (f"aggregate {candidate.accuracy:.3f} does not beat "
                       f"{incumbent.accuracy:.3f} by {margin}")

    missing = regression_set - candidate.regression_passed
    if missing:
        return False, f"regression cases failed: {sorted(missing)}"

    for name, score in candidate.slices.items():
        previous = incumbent.slices.get(name)
        if previous is not None and previous - score > max_slice_regression:
            return False, (f"slice {name!r} regressed {previous:.3f} -> {score:.3f}, "
                           f"beyond {max_slice_regression}")

    return True, f"promote: {incumbent.accuracy:.3f} -> {candidate.accuracy:.3f}"


regression_set = {"empty-input", "very-long-input", "unicode-name", "known-incident-4412"}
incumbent = Evaluation(0.880, {"short": 0.90, "long": 0.86, "non-english": 0.81},
                       regression_set)

good = Evaluation(0.905, {"short": 0.92, "long": 0.89, "non-english": 0.82}, regression_set)
sneaky = Evaluation(0.910, {"short": 0.95, "long": 0.93, "non-english": 0.74}, regression_set)
broken = Evaluation(0.930, {"short": 0.95, "long": 0.93, "non-english": 0.90},
                    regression_set - {"known-incident-4412"})

for label, candidate in [("good", good), ("sneaky", sneaky), ("broken", broken)]:
    decision, reason = should_promote(candidate, incumbent, regression_set)
    print(f"{label:8s} {'PROMOTE' if decision else 'BLOCK':8s} {reason}")
```

```text
good     PROMOTE  promote: 0.880 -> 0.905
sneaky   BLOCK    slice 'non-english' regressed 0.810 -> 0.740, beyond 0.02
broken   BLOCK    regression cases failed: ['known-incident-4412']
```

The "sneaky" candidate is the instructive one. It has the best aggregate accuracy of the three and would sail through any gate that checks a single number, while quietly getting seven points worse for non-English users. Aggregate-only gates ship exactly this model, and the failure surfaces weeks later as a support pattern rather than a metric.

### Exercise 3

```python
import math
import random


def population_stability_index(baseline: list[float], current: list[float], bins: int = 10) -> float:
    low, high = min(baseline), max(baseline)
    width = (high - low) / bins or 1.0

    def histogram(values: list[float]) -> list[float]:
        counts = [0] * bins
        for value in values:
            index = min(bins - 1, max(0, int((value - low) / width)))
            counts[index] += 1
        return [max(count / len(values), 1e-6) for count in counts]      # avoid log(0)

    expected, actual = histogram(baseline), histogram(current)
    return sum((a - e) * math.log(a / e) for e, a in zip(expected, actual))


random.seed(0)
baseline = [random.gauss(0, 1) for _ in range(10_000)]
scenarios = {
    "identical":   [random.gauss(0.0, 1.0) for _ in range(10_000)],
    "slight shift":[random.gauss(0.5, 1.0) for _ in range(10_000)],
    "heavy shift": [random.gauss(1.5, 1.4) for _ in range(10_000)],
}

for name, current in scenarios.items():
    psi = population_stability_index(baseline, current)
    verdict = "stable" if psi < 0.1 else "investigate" if psi < 0.25 else "ALERT"
    print(f"{name:14s} PSI {psi:6.4f}  {verdict}")
```

```text
identical      PSI 0.0033  stable
slight shift   PSI 0.2423  investigate
heavy shift    PSI 1.8578  ALERT
```

The conventional thresholds are 0.1 and 0.25, and they separate the three cases cleanly here. The important property is that **none of this needs labels**: you are comparing input distributions, so drift detection runs continuously in production where ground truth is unavailable. Concept drift is the harder case and cannot be caught this way, which is why a feedback channel is not optional.

</details>

## Exit test

1. What must be recorded for a training run to be reproducible?
2. What does a model registry track that artifact storage alone does not?
3. Why prefer safetensors over pickle-based checkpoints?
4. Why must the tokenizer be stored with the weights?
5. Name the four dimensions of versioning, and why a lockfile alone is insufficient.
6. What is a training smoke test, and what does it catch?
7. Why is CD for a model different from CD for code?
8. Describe shadow, canary, and blue/green deployment.
9. Why is an aggregate metric improvement not enough to promote a model?
10. Distinguish data drift from concept drift, and say which is detectable without labels.
11. What is the cheapest early warning signal in production, and what does it indicate?
12. What four properties make rollback possible?

<details>
<summary>Show answers</summary>

1. The resolved config, the code commit, the data version, the environment (library and CUDA versions, hardware), the random seeds, the metrics, and the output artifacts.
2. Which model version is in which stage, its lineage back to the run, data, and code, its evaluation metrics, and approval metadata — that is, identity and governance rather than bytes.
3. Pickle-based formats execute arbitrary code on load, so an untrusted checkpoint is a remote code execution risk. safetensors stores only tensor data.
4. A model and a mismatched tokenizer produce confident nonsense rather than an error, because the token IDs mean something different than the embeddings expect.
5. Code, data, model, and environment. A lockfile pins Python packages but not CUDA drivers, base images, or system libraries, so a container image digest is needed as well.
6. A training run of a few steps on a handful of examples in CI, asserting that it runs and the loss decreases. It catches shape errors, device mismatches, and broken configs in a minute instead of hours into a real run.
7. The deployable artifact is produced by a training run rather than built by CI, so the pipeline promotes an existing artifact through evaluation and approval gates rather than compiling one.
8. Shadow sends a copy of live traffic to the new model and logs the output without serving it. Canary routes a small traffic percentage to the new model. Blue/green runs two complete environments and switches all traffic at once, enabling instant rollback.
9. Because an average can improve while a specific slice or previously fixed failure case regresses. That pattern harms a subset of users while every headline number looks better.
10. Data drift is a change in the input distribution; concept drift is a change in the relationship between input and correct output. Data drift is detectable without labels by comparing input distributions; concept drift requires ground truth or a feedback channel.
11. The prediction distribution. A sizeable shift with no corresponding product change signals that something upstream — the data, the users, or a dependency — has changed.
12. Immutable versioned artifacts, the previous version still deployable, configuration held as data rather than code, and no irreversible migration coupled to the release.

</details>

## Completion criteria

You are done when:

- you can list what a run must record to be reproducible, without notes
- you can describe the promotion path from a training run to production traffic
- you can explain drift, its two kinds, and how each is detected
- you can state the preconditions for a working rollback
- your three exercises run, including the blocked "sneaky" promotion

This is the last lesson in Month 0. Return to [the curriculum](../README.md) for the core knowledge check and the [tiny text classifier project](../project/tiny-text-classifier/README.md).

## References

**Experiment tracking and registries**
- [MLflow documentation](https://mlflow.org/docs/latest/index.html) — tracking, registry, and projects
- [Weights & Biases](https://docs.wandb.ai/)
- [safetensors](https://huggingface.co/docs/safetensors/index)

**Versioning**
- [DVC](https://dvc.org/doc/start/data-management)
- [Hugging Face Hub: repository versioning](https://huggingface.co/docs/hub/repositories-getting-started)

**CI/CD and deployment**
- [Continuous Delivery for Machine Learning, Martin Fowler](https://martinfowler.com/articles/cd4ml.html) — the clearest overview of the whole pipeline
- [MLOps: Continuous delivery and automation pipelines, Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) — a rubric for production readiness
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

**Monitoring and drift**
- [Evidently AI: monitoring and drift detection](https://docs.evidentlyai.com/)
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- [Designing Machine Learning Systems, Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — chapters 8 and 9 on drift and monitoring

## About this lesson

Written to cover section 0.18 of the [Month 0 curriculum](../README.md). Code examples were checked with Python 3.12.
