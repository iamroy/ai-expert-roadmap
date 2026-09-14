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

Put numbers on it, because this is the decision GQA exists to make. A 32-layer model with 32 query heads and `head_dim=128`, in `bfloat16`:

```python
def kv_cache_bytes(layers, tokens, kv_heads, head_dim, batch=1, dtype_bytes=2):
    """K and V, per layer, per cached token."""
    return 2 * layers * batch * tokens * kv_heads * head_dim * dtype_bytes


layers, query_heads, head_dim = 32, 32, 128
for name, kv_heads in [("MHA", 32), ("GQA-8", 8), ("MQA", 1)]:
    for tokens in (4_096, 32_768):
        gb = kv_cache_bytes(layers, tokens, kv_heads, head_dim) / 1e9
        print(f"{name:6s} Hkv={kv_heads:2d}  {tokens:>6,} tokens  {gb:7.3f} GB per sequence")

per_sequence = kv_cache_bytes(layers, 32_768, 8, head_dim) / 1e9
print(f"\nGQA-8 at 32k, 64 concurrent sequences: {per_sequence * 64:.1f} GB of cache alone")
```

```text
MHA    Hkv=32   4,096 tokens    2.147 GB per sequence
MHA    Hkv=32  32,768 tokens   17.180 GB per sequence
GQA-8  Hkv= 8   4,096 tokens    0.537 GB per sequence
GQA-8  Hkv= 8  32,768 tokens    4.295 GB per sequence
MQA    Hkv= 1   4,096 tokens    0.067 GB per sequence
MQA    Hkv= 1  32,768 tokens    0.537 GB per sequence

GQA-8 at 32k, 64 concurrent sequences: 274.9 GB of cache alone
```

Read the last line carefully. One sequence at 32k under MHA needs 17 GB of cache, which by itself exceeds many accelerators before a single weight is loaded. GQA-8 cuts that by exactly the head ratio, 4×, and MQA by 32×. But cache scales linearly with *concurrency* too, so even the GQA configuration needs 275 GB to serve 64 long-context users at once. That arithmetic, not model quality, is why GQA is near-universal in models built to be served, and it is the same constraint that motivates paged and quantized caches in Month 9.

Two properties worth noting from the numbers: the cache is independent of vocabulary and FFN size entirely, and it grows with *cached tokens*, so it keeps growing during generation rather than being fixed at prefill.

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

The gap between those two counts is worth computing rather than asserting:

```python
def swiglu_ffn_params(width, hidden):
    return 3 * width * hidden                      # gate, up, and down matrices


def attention_params(width, query_heads, kv_heads, head_dim):
    q = width * query_heads * head_dim
    kv = 2 * width * kv_heads * head_dim
    out = query_heads * head_dim * width
    return q + kv + out


layers, width, hidden = 32, 4096, 14336
experts, active = 8, 2
attn = attention_params(width, query_heads=32, kv_heads=8, head_dim=128)

dense_layer = attn + swiglu_ffn_params(width, hidden)
moe_total_layer = attn + experts * swiglu_ffn_params(width, hidden)
moe_active_layer = attn + active * swiglu_ffn_params(width, hidden)

print(f"dense        total {layers * dense_layer / 1e9:6.2f}B   active {layers * dense_layer / 1e9:6.2f}B")
print(f"MoE 8x top-2 total {layers * moe_total_layer / 1e9:6.2f}B   active {layers * moe_active_layer / 1e9:6.2f}B")
print(f"active fraction: {moe_active_layer / moe_total_layer:.1%}  (not {active}/{experts} = {active / experts:.0%})")
```

```text
dense        total   6.98B   active   6.98B
MoE 8x top-2 total  46.44B   active  12.62B
active fraction: 27.2%  (not 2/8 = 25%)
```

Three readings, and each corresponds to a different engineering question. **What must I store?** 46B — you provision memory for all experts. **What computes per token?** 12.6B — roughly what sets per-token latency. **What does it cost against a dense model of the same active size?** The MoE holds 6.6× the parameters of the 7B dense model while computing on about 1.8× as many per token.

Note the active fraction is 27.2%, not the 25% the expert ratio suggests, because attention runs for every token regardless of routing. The gap widens as attention grows relative to the FFN, so never estimate active parameters from the expert ratio alone.

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

## Videos and code to read

- [huggingface/transformers](https://github.com/huggingface/transformers) — read `modeling_llama.py` for GQA, RMSNorm, SwiGLU, and RoPE in one production file, then `modeling_mixtral.py` for MoE routing
- [labmlai annotated implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) — annotated RMSNorm, GQA, and mixture-of-experts
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — how GQA and cache layout are exploited in a real serving engine

## Mapped companion lessons

- [Mixture of Experts](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/11-mixture-of-experts) maps to sparse routing and active-parameter reasoning.
- [KV Cache and Flash Attention](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/12-kv-cache-flash-attention) and [Attention Variants](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/15-attention-variants) extend attention efficiency and architecture choices.
- [Open Models: Architecture Walkthroughs](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs) applies the components to real configurations.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).
