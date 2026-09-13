# 0.18 — MLOps Fundamentals

**Depth: SKIM**

**Goal:** connect experiments, datasets, models, deployment, monitoring, and rollback into one traceable model lifecycle.

[Month 0 roadmap](../README.md) · [Previous: Software Engineering](17-software-engineering-for-ai.md) · [Month 0 project](../project/tiny-text-classifier/README.md)

## The lifecycle

MLOps applies software, data, and operational discipline to systems whose behavior depends on learned artifacts.

```text
data/version → train/run → evaluate → register → approve → deploy
      ↑                                                │
      └──── monitor ← observe ← serve ← rollback ──────┘
```

Each arrow needs identity, validation, and ownership. A code commit alone cannot identify a model; data, features, environment, hyperparameters, random state, and base checkpoint also matter.

## Experiment tracking

An experiment run should record code revision, data/split version, model/configuration, hyperparameters, seed, environment, metrics over time, artifacts, and notes. Use generated run IDs and immutable artifact locations.

Tracking creates evidence, not truth. Comparing dozens of runs on one validation set can overfit it. Mark the selection metric and keep a final untouched test or prospective evaluation. Failed runs are useful when their status and cause are recorded rather than silently deleted.

## Model registries and artifact storage

Artifact storage holds model weights, tokenizers, preprocessing assets, reports, and manifests. Use checksums, access control, encryption, retention, and immutable versions. Do not put large weights directly in ordinary Git history.

A model registry adds lifecycle metadata such as candidate, approved, production, or retired status and connects artifacts to evaluation evidence. A stage label is a mutable pointer; deployments should also record the immutable model digest/version it resolved to.

## Dataset, model, and environment versioning

Dataset versions need source manifests, schema, transformations, split rules, and content identifiers. Model versions need architecture, weights, tokenizer/preprocessor, base model, and adaptation details. Environment versions need dependencies, system libraries, container digest, and hardware/runtime assumptions.

Version these as a compatibility set. Loading a new tokenizer with old weights or a feature transform with a different unit can produce valid tensors and invalid behavior.

## CI/CD and deployment

Continuous integration checks code, schemas, training/inference contracts, data validation, and a small deterministic model fixture. Full training rarely belongs in every code commit; schedule representative smoke training and separate expensive qualification pipelines.

Continuous delivery promotes an already built, tested artifact. Deployment strategies include canary traffic, shadow evaluation, blue/green environments, and staged regional rollout. Define acceptance and automatic-stop criteria before exposure.

For LLM applications, the deployed unit includes model/provider version, prompt and context policy, tool definitions, output schema, safety controls, and evaluation thresholds. Changing only the prompt can still be a production model-behavior release.

## Monitoring and drift

Data drift means input distributions change. Concept drift means the relationship between inputs and desired outputs changes. Prediction drift means output distributions change. None alone proves quality degradation; connect them to delayed labels, sampled human review, or task-specific evaluations.

Monitor availability, latency, throughput, errors, resource use, token/cost measures, schema validity, safety events, and model-quality proxies. Slice by bounded relevant dimensions. Establish baselines and alert on actionable conditions rather than every statistical difference.

Feedback loops need special care: model decisions can change which labels are later observed. For example, rejected applications may never reveal repayment outcomes, biasing monitoring data.

## Rollback and incident response

Rollback must restore a compatible set of model, tokenizer, preprocessing, prompt, schema, and infrastructure. Database or feature changes may make a simple binary rollback unsafe; use backward-compatible migrations and retained artifacts.

Define owners, severity, stop/rollback triggers, and evidence collection. Preserve request IDs, versions, and safe samples for analysis. After recovery, identify whether the control failure was in data validation, evaluation coverage, rollout, monitoring, or authorization—not only which model metric moved.

## Governance and reproducibility

Record intended use, excluded uses, data provenance, evaluation limits, approval, and change history. Access to training data and model artifacts should follow least privilege. Reproducibility includes the ability to explain and reconstruct a release even when bitwise retraining is impractical.

## Checkpoint

1. What does an experiment tracker record that a Git commit does not?
2. How does a model registry differ from artifact storage?
3. Distinguish data, concept, and prediction drift.
4. Why must rollback restore more than model weights?

<details>
<summary>Show answers</summary>

1. Data versions, configuration, hyperparameters, seeds, environment, metrics, and generated artifacts associated with one run.
2. Storage retains bytes; a registry adds model identity, lineage, evaluation, approval, and lifecycle state.
3. Data drift changes inputs, concept drift changes the input-to-target relationship, and prediction drift changes model outputs. Each needs contextual quality evidence.
4. Tokenizers, features, prompts, schemas, and infrastructure form compatibility contracts with the weights; mixed versions can fail silently.

</details>

## Exercise — Create a release and rollback plan

Design promotion for a new text classifier. Include tracked evidence, gates, canary monitoring, and rollback criteria.

<details>
<summary>Show exercise solution</summary>

Register immutable model, tokenizer, dataset/split, code, environment, and evaluation-report digests. Gates require schema/contract tests, minimum held-out metrics and slice floors, calibration limits, latency/memory budgets, and documented approval.

Shadow the candidate on representative traffic without affecting decisions, then canary a small authorized share. Compare errors, latency, prediction distribution, task labels when available, and operational saturation against the current version. Stop or roll back on predeclared quality, safety, schema, or reliability thresholds. Rollback changes the deployment pointer to the prior compatible bundle, verifies health and behavior, and preserves incident evidence. Expand only after the observation window passes.

</details>

## Completion criteria

Trace a model from experiment to registry and deployment, version all compatibility artifacts, distinguish drift types, and define measurable rollout and rollback gates.

## Primary references

- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)
- [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
