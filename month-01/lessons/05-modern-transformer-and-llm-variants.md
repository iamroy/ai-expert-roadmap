# 1.5 — Modern Transformer / LLM Variants

**Depth: LEARN**

**Goal:** read a model configuration and connect architectural substitutions to training behavior, computation, and serving memory.

[Month 1 roadmap](../README.md) · [Previous: Training Objectives](04-llm-training-objectives.md) · [Next: Inference and Decoding](06-llm-inference-and-decoding.md)

## Lesson 1.5.1 — Keep the stable skeleton in view

Many modern language models preserve this basic path:

```text
IDs → token embeddings → repeated causal blocks → final norm → vocabulary logits
```

Changes often happen inside the block: a different normalization, a gated FFN, rotary position information, shared key/value heads, or sparse expert routing. Knowing a model's family name is less useful than identifying exactly which operators and dimensions its checkpoint uses.

Use the project's learned positions, LayerNorm, standard multi-head attention, and GELU FFN as a baseline. Learn each substitution independently before combining them. A model named “transformer” need not have the same cache layout, normalization parameters, or attention visibility as another transformer.

## Lesson 1.5.2 — LayerNorm versus RMSNorm

For one token vector `x` of width `D`, LayerNorm subtracts its mean and divides by its standard deviation, then applies learned scale and typically bias. [RMSNorm](https://arxiv.org/abs/1910.07467) instead rescales by root mean square:

```text
RMSNorm(x) = g ⊙ x / sqrt(mean(x²) + epsilon)
```

It does not subtract the feature mean. Both preserve tensor shape and use a learned per-feature scale. They are not batch normalization and do not average over different examples.

For `x=[3,4]`, the RMS is `sqrt(12.5)`. With unit scale and negligible epsilon, the output retains the relative values and a nonzero mean. LayerNorm produces approximately `[-1,1]`. This illustrates the semantic difference rather than proving one method is universally better.

Accumulating normalization statistics in adequate precision matters when activations have large magnitudes. Replacing a trained checkpoint's normalization operator without adapting its weights is not a harmless inference optimization.

## Lesson 1.5.3 — SwiGLU and gated feature transformations

A conventional FFN has an expansion matrix, an activation, and a contraction matrix. A [SwiGLU](https://arxiv.org/abs/2002.05202) FFN introduces two expansion branches:

```text
up   = x Wup
gate = SiLU(x Wgate)
y    = (gate ⊙ up) Wdown
SiLU(a) = a × sigmoid(a)
```

The gate modulates the other branch element by element. This is feature gating, not routing tokens among MoE experts. With hidden width `F`, a bias-free SwiGLU FFN has approximately `3DF` parameters; a two-matrix FFN has `2DF`.

Thus keeping `F` unchanged increases parameters and computation. To compare against a conventional `F=4D` FFN at similar matrix-parameter count, choose SwiGLU width near `8D/3`, then account for hardware-friendly rounding. Practical architectures may choose a different width for their own budget.

## Lesson 1.5.4 — MHA, MQA, and GQA

In ordinary multi-head attention (MHA), each query head has its own key and value projections. During autoregressive decoding, a KV cache retains previous keys and values for every layer. Reading that cache can dominate parts of inference.

[Grouped-query attention](https://arxiv.org/abs/2305.13245) separates the number of query heads `Hq` from the number of KV heads `Hkv`:

| Form | KV heads | What is shared? |
|---|---|---|
| MHA | `Hkv = Hq` | No sharing across query heads |
| MQA | `Hkv = 1` | All query heads share one K head and one V head |
| GQA | `1 < Hkv < Hq` | Each query-head group shares a K/V pair |

MQA is the extreme sharing case. Queries remain distinct, so it is incorrect to say the whole attention operation has only one head. An implementation maps each query head to a KV group; it need not permanently duplicate cached K/V to match query-head count.

For ordinary cached self-attention with equal key/value head dimensions, an approximate cache size is:

```text
bytes = 2 × layers × batch × cached_tokens × Hkv × head_dim × bytes_per_element
```

The factor two is for K and V. This excludes weights, temporary activations, allocation overhead, and any specialized cache representation. Reducing `Hkv` reduces cache capacity and bandwidth needs, but does not proportionally reduce all model computation or weight memory. Query-head score calculations still exist. Quality and latency must be measured after changing the attention design.

## Lesson 1.5.5 — Dense versus sparse mixture of experts

A dense FFN applies the same parameters to every token. A sparse mixture-of-experts (MoE) layer contains multiple FFNs and a router. The router chooses a subset for each token and combines their outputs:

```text
token representation → router scores → select k experts
                    → expert FFNs → weighted combination → residual update
```

A common router uses a linear projection and top-k selection. Routing probabilities can be normalized in several ways; use the architecture's actual definition. Sparse MoE usually refers to sparse expert activation, not a sparse attention map.

Two parameter counts answer different questions:

- **Total parameters** describe all stored experts and shared components.
- **Active parameters per token** describe the subset used for that token's computation.

A model with eight experts and two active experts does not necessarily use one quarter of its total parameters per token: shared attention, embeddings, routers, and other components are still active. All experts usually still need to be resident somewhere for efficient serving.

[Mixtral of Experts](https://arxiv.org/abs/2401.04088) is a concrete sparse-MoE architecture report. Expert routing adds engineering concerns: load imbalance, token dispatch, capacity policies, communication across devices, and batching efficiency. More total parameters can increase capacity at a given active compute budget, but communication and memory can erode that benefit. Experts should not be assumed to have clean human-readable specialties.

## Lesson 1.5.6 — Reading a configuration as an engineering exercise

Consider this hypothetical model, not a named vendor checkpoint:

```text
layers=24, width=1024, query_heads=16, kv_heads=4
head_dim=64, norm=RMSNorm, positions=RoPE
ffn=SwiGLU, ffn_hidden=2816, vocab=32000
```

Check `query_heads × head_dim = width` for this conventional setup and divisibility of query heads into KV groups. The model stores fewer K/V channels than query channels. Its three FFN matrices have about `3×1024×2816` parameters per layer. The final vocabulary projection still produces 32,000 logits per position regardless of head sharing.

A practical architecture audit records normalization placement, attention type, position configuration, FFN type, tied versus untied embeddings, and context/training limits. Model-family labels such as Llama, Qwen, and Mistral cover multiple versions and variants; use the exact checkpoint configuration rather than memorizing one permanent list of features for a family.

## Checkpoint

1. Does RMSNorm force the feature mean to zero?
2. Why is using the same hidden width an unfair parameter-budget comparison between GELU FFN and SwiGLU?
3. Does MQA give every query the same query vector?
4. Why is active MoE parameter count insufficient for determining whether a model fits on a device?

<details>
<summary>Show answers</summary>

1. No; it rescales by RMS without centering.
2. SwiGLU uses three main matrices rather than two. Equal hidden widths give different parameter counts.
3. No. It shares K/V projections; query heads remain distinct.
4. All expert weights, shared weights, the KV cache, and runtime overhead require storage. Dispatch may distribute storage across devices, but active count alone omits it.

</details>

## Hands-on exercises

1. Estimate the FP16 KV cache for 24 layers, batch 1, 4,096 tokens, 16 query heads, 4 KV heads, and head dimension 64. Compare with MHA at 16 KV heads.
2. For width 768, find a SwiGLU hidden width that matches the matrix-parameter count of a conventional FFN with hidden width 3,072. Ignore biases and rounding.
3. Propose a comparison of a dense model and an MoE model for a low-latency assistant. Specify what to measure beyond token accuracy.

<details>
<summary>Show exercise solutions</summary>

1. `2×24×1×4096×4×64×2 = 100,663,296` bytes, or 96 MiB. MHA uses four times as much: 384 MiB. These are cache-only estimates.
2. Set `3×768×F = 2×768×3072`, giving `F=2048`. Runtime can still differ because matrix shapes, activation work, and kernels differ.
3. Match the task set and quality requirements. Measure time to first token, decode throughput, end-to-end p50/p95 latency, concurrency, total device memory, active compute, dispatch/communication overhead, cost, and failure slices. Use the same prompt/output distributions; average throughput alone can hide tail-latency problems.

</details>

## Completion criteria

Explain RMSNorm, SwiGLU, GQA/MQA, and sparse MoE; use a configuration to estimate cache and FFN sizes; distinguish stored capacity from per-token computation.

## Primary references

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).
- [GQA](https://arxiv.org/abs/2305.13245).
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088).

## Mapped companion lessons

- [Mixture of Experts](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/11-mixture-of-experts) maps to sparse routing and active-parameter reasoning.
- [KV Cache and Flash Attention](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/12-kv-cache-flash-attention) and [Attention Variants](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/15-attention-variants) extend attention efficiency and architecture choices.
- [Open Models: Architecture Walkthroughs](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs) applies the components to real configurations.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).
