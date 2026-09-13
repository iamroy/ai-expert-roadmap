# 0.10 — Representation Learning

**Depth: REFRESH**

**Goal:** reason about embeddings and latent spaces as learned interfaces, evaluate their geometry, and connect vision, language, and multimodal representations.

[Month 0 roadmap](../README.md) · [Previous: Architecture Concepts](09-deep-learning-architecture-concepts.md) · [Next: Self-Supervised Learning](11-self-supervised-learning.md)

## Representations and latent variables

A representation transforms raw input into features that preserve information useful for a task. A latent representation is internal rather than directly observed. An embedding is usually a fixed-width vector representing an item, token, image, passage, user, or other entity.

```text
raw input → encoder → vector representation → comparison or task head
```

An embedding's individual dimensions rarely have stable standalone meanings. The geometry is learned jointly with an objective: vectors close under one model need not be close under another, and arbitrary rotation of the space can preserve many relationships.

## Similarity and dimensionality

Dot product combines angle and magnitude. Cosine similarity divides out vector norms and compares direction. Euclidean distance measures absolute displacement. The choice must match how the model was trained and evaluated.

High dimensionality increases representational capacity but also storage and search cost. Distances can concentrate in high dimensions, and unused/noisy dimensions can harm downstream work. Dimensionality is a design parameter, not a quality score.

Normalize embeddings when the intended metric is cosine and the search system expects unit vectors. After normalization, maximizing dot product and cosine similarity are equivalent. Never mix model versions or preprocessing variants in one index without a migration plan.

## What training objectives shape

Classification training encourages features that separate labeled classes. Contrastive training pulls specified positives together and pushes negatives apart. Autoregressive language modeling learns contextual states useful for predicting future tokens. Reconstruction objectives preserve information needed to rebuild input.

These objectives produce different invariances. A vision representation trained to ignore color changes may fail a product task where exact color is essential. A sentence embedding trained for semantic similarity may discard wording details needed for legal comparison.

## Pretrained representations and transfer

Pretrained embeddings reduce data requirements for downstream tasks, but quality depends on domain, language, granularity, pooling, and objective. Evaluate the actual retrieval or classification task, not only a generic embedding benchmark.

For token-level language representations, pooling must be deliberate: a special token, mean of non-padding tokens, or learned pooling head can produce different sentence vectors. Padding-aware mean pooling is covered in [0.2](02-numpy-and-tensor-manipulation.md).

## From CNN and DINO features to LLM and multimodal embeddings

- CNN features summarize spatial patterns with local shared filters and increasing receptive fields.
- DINO-style self-supervision trains a student to match a teacher's view-level representations, producing transferable visual features without human class labels.
- LLM token embeddings begin as vocabulary lookups and become context-dependent through transformer blocks.
- Multimodal systems align image and text encoders in a compatible space or connect modalities through learned projections and attention.

A shared vector dimension does not prove shared semantics. Alignment requires a training signal and evaluation on cross-modal tasks.

## Evaluating representation quality

Intrinsic checks examine neighbor quality, clustering, isotropy, or known similarity pairs. Extrinsic checks train a linear probe, retrieve relevant items, or measure target-task performance. A linear probe tests information accessible through a simple boundary; full fine-tuning can adapt the representation and answers a different question.

Avoid leakage: near-duplicate images, passages from the same document, or multiple records from one identity must respect group splits. Inspect performance by domain and subgroup. Embeddings can encode sensitive attributes even when they are not explicit labels.

## Production connections

Version the encoder, tokenizer/preprocessor, pooling, normalization, and vector dimension as one contract. Store source IDs and timestamps alongside vectors so they can be deleted and re-embedded. An embedding index is derived data: changing the model normally requires a backfill or dual-index migration.

## Checkpoint

1. Why is an embedding dimension not necessarily human-interpretable?
2. When do cosine similarity and dot product give the same ranking?
3. What different questions do a frozen linear probe and full fine-tuning answer?
4. Why must an embedding migration include the entire index?

<details>
<summary>Show answers</summary>

1. Representations are jointly learned and can be rotated or redistributed while preserving useful relations; objectives do not generally assign named meaning to each coordinate.
2. When all compared vectors are normalized to equal norm, especially unit norm.
3. A linear probe asks what is already simply accessible; full fine-tuning asks what can be learned after adapting the representation.
4. Coordinates from different encoders or preprocessing contracts are not guaranteed to share geometry, even when vector dimensions match.

</details>

## Exercise — Audit an embedding system

Create an evaluation plan for passage retrieval. Include positive/negative construction, metrics, leakage prevention, and a model-version migration.

<details>
<summary>Show exercise solution</summary>

Build queries with independently judged relevant passages and hard topical negatives. Split by source document or time. Report recall@k, mean reciprocal rank when one early relevant result matters, latency, and performance slices by domain/language/length. Inspect nearest neighbors qualitatively and include queries with no answer.

Version encoder, tokenizer, pooling, and normalization. Backfill a new index while the old index serves traffic, compare both on the same evaluation and a shadow workload, switch reads after acceptance, retain rollback briefly, and delete old derived vectors under a documented retention policy.

</details>

## Completion criteria

Explain embedding geometry, choose a similarity metric, distinguish training objectives and probes, and design a versioned representation pipeline.

## Primary references

- [A Tutorial on Representation Learning with Deep Generative Models](https://arxiv.org/abs/2006.12013)
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
