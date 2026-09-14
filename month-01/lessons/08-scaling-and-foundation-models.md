# 1.8 — Scaling & Foundation Models

**Depth: LEARN**

**Goal:** reason about parameters, data, compute, and deployment together without treating model size as a substitute for evidence.

[Month 1 roadmap](../README.md) · [Previous: Context Windows](07-context-windows-and-limits.md) · [Next: Month 1 Project](../project/tiny-gpt/README.md)

## Lesson 1.8.1 — What makes a model a foundation model?

A foundation model is broadly pretrained and can support a range of downstream tasks through adaptation or conditioning. The term describes its role and transfer potential, not a fixed architecture or minimum parameter threshold. It can be a language, vision, multimodal, or other model.

For language models, large-scale next-token training learns reusable representations and continuation patterns. [Few-shot language-model experiments](https://arxiv.org/abs/2005.14165) illustrate adapting behavior through examples in context without updating parameters. Fine-tuning changes weights; prompting changes the input. Neither guarantees that all desired behavior is reliable.

A tiny GPT in this month's project uses related computation but is trained on a tiny corpus for mechanistic learning. Completing that project does not create a broadly capable foundation model.

## Lesson 1.8.2 — Three separate scale variables

| Variable | Meaning | Common misunderstanding |
|---|---|---|
| Parameters `N` | Learned weights in the model | More weights guarantee better performance on every task |
| Training tokens `D` | Token instances processed during training | Every token is unique, equally useful, or high quality |
| Compute `C` | Arithmetic and hardware effort | Equal FLOPs mean equal wall-clock time or monetary cost |

Here `D` means training-token count, not the model width used in earlier lessons. Keep notation local and explicit.

Parameters provide capacity; data provides learning signal; compute pays for processing that data through the model. Repeated tokens count as compute even though they do not provide the same diversity as new high-quality data. Duplicates, contamination, domain balance, tokenizer choice, and training stability affect the useful signal.

A larger model can be undertrained for its capacity. A smaller model trained on more useful data can outperform a larger one with a worse allocation. Conversely, too little capacity can limit the patterns a model represents even with substantial data.

## Lesson 1.8.3 — Scaling laws describe empirical trends

[Scaling-law studies](https://arxiv.org/abs/2001.08361) fit smooth relationships between language-model loss and scale over measured regimes. A schematic form is:

```text
loss(N, D) ≈ irreducible term + A/N^alpha + B/D^beta
```

The coefficients and exponents depend on the study and assumptions. This is a model of observed behavior, not an equation that predicts every downstream task or every architecture without calibration.

Power-law improvement has diminishing absolute returns: multiplying resources can yield a progressively smaller loss reduction. Loss trends can help estimate a training budget, but downstream success also depends on how the benchmark maps model outputs into scores. A sharp change in an exact-match task score need not imply a discontinuity in all underlying capabilities.

Use small-scale experiments to check data quality, optimization, and scaling direction before a costly run. Extrapolation far beyond a measured range carries substantial uncertainty.

A rough but useful accounting rule is that training cost is about `6 × parameters × tokens` FLOPs: roughly two FLOPs per parameter for the forward pass and four for the backward. That single formula lets you see the allocation problem directly. Given a fixed budget, every parameter you add must be paid for with tokens you no longer train on:

```python
def training_flops(params, tokens):
    return 6 * params * tokens


budget = 1e23                                       # a fixed compute budget
for params in (7e9, 70e9, 280e9):
    tokens = budget / (6 * params)
    print(f"{params / 1e9:6.0f}B params -> {tokens / 1e12:7.2f}T tokens  "
          f"({tokens / params:6.1f} tokens/param)")

print()
for params in (7e9, 70e9):
    tokens = 20 * params                            # the Chinchilla-style ratio
    print(f"{params / 1e9:5.0f}B at 20 tokens/param needs "
          f"{training_flops(params, tokens):.2e} FLOPs ({tokens / 1e12:.2f}T tokens)")
```

```text
     7B params ->    2.38T tokens  ( 340.1 tokens/param)
    70B params ->    0.24T tokens  (   3.4 tokens/param)
   280B params ->    0.06T tokens  (   0.2 tokens/param)

    7B at 20 tokens/param needs 5.88e+21 FLOPs (0.14T tokens)
   70B at 20 tokens/param needs 5.88e+23 FLOPs (1.40T tokens)
```

The first block is the trade-off that scaling-law work exists to resolve. At a fixed budget, a 7B model gets 340 tokens per parameter while a 280B model gets 0.2 — wildly overtrained against severely undertrained, from the same compute. [Chinchilla](https://arxiv.org/abs/2203.15556) found the loss-minimizing point for its setup was far more balanced than prevailing practice, near 20 tokens per parameter, which implied that many well-known models of that era were too large for the data they saw.

Two cautions on using this. The `6ND` rule ignores attention's quadratic term, which matters at long sequence lengths, and it ignores real utilization: achieved FLOPs are a fraction of peak. And compute-optimal is optimal *for training loss only*. If a model will serve billions of requests, a smaller model trained past its compute-optimal point is often the better total-cost decision, which is exactly the argument 1.8.5 develops.

## Lesson 1.8.4 — Compute-optimal training

For a conventional dense decoder, a frequently used rough training estimate is:

```text
training FLOPs ≈ 6 × parameter_count × training_token_count
```

This approximates dominant parameter-matrix work in forward and backward passes. It omits important context-dependent attention costs and implementation details. It is less direct for sparse expert models and unsuitable as an exact hardware runtime estimate.

At a fixed `N×D` budget, a larger model trained on fewer tokens and a smaller model trained on more tokens can consume similar approximate compute. [Chinchilla's study](https://arxiv.org/abs/2203.15556) found that jointly scaling model size and training data more evenly improved allocation in its experimental regime. Its result is an empirical guide, not a permanent rule that every model should train on exactly one fixed token-to-parameter ratio.

For example, doubling `N` while halving `D` preserves the approximate product. Whether quality improves depends on where the original model sat relative to the fitted optimum, along with data and optimization quality.

## Lesson 1.8.5 — Training optimality versus lifetime cost

The allocation minimizing pretraining loss for a training budget need not minimize total system cost. A smaller model trained for longer may be attractive when it will serve a very large number of requests. Inference happens repeatedly; one-time training and ongoing serving costs should be considered separately.

The crossover is computable. Training costs roughly `6ND` once; generation costs roughly `2N` FLOPs per token, forever:

```python
def training_flops(params, tokens):
    return 6 * params * tokens


def inference_flops(params, tokens):
    return 2 * params * tokens


for params in (7e9, 70e9):
    train = training_flops(params, 20 * params)          # Chinchilla-style allocation
    print(f"{params / 1e9:5.0f}B: training {train:.2e} FLOPs")
    for daily_tokens in (1e9, 1e11):
        serving = inference_flops(params, daily_tokens * 365)
        print(f"        at {daily_tokens:.0e} tokens/day -> serving {serving:.2e} FLOPs/year "
              f"({serving / train:5.2f}x training)")
```

```text
    7B: training 5.88e+21 FLOPs
        at 1e+09 tokens/day -> serving 5.11e+21 FLOPs/year ( 0.87x training)
        at 1e+11 tokens/day -> serving 5.11e+23 FLOPs/year (86.90x training)
   70B: training 5.88e+23 FLOPs
        at 1e+09 tokens/day -> serving 5.11e+22 FLOPs/year ( 0.09x training)
        at 1e+11 tokens/day -> serving 5.11e+24 FLOPs/year ( 8.69x training)
```

At low volume, training dominates and compute-optimal allocation is the right objective. At high volume it inverts completely: a 7B model serving 100B tokens a day burns 87× its own training cost every year. Once serving dominates by that margin, spending *more* than the compute-optimal amount on training — a smaller model trained on far more tokens than the 20:1 ratio suggests — lowers total cost, because every FLOP saved per token is paid back billions of times.

That is the reasoning behind widely deployed small models trained far past their compute-optimal point. Note what the arithmetic does not include: memory bandwidth, which often binds decode before FLOPs do (1.6.2), utilization well below peak, and the KV cache (1.5). Use this to see the shape of the trade-off, not to quote a budget.

At deployment, weight memory, KV cache, batch size, context length, output length, and hardware all matter. A rough unquantized FP16 weight estimate is `2N` bytes. This excludes runtime allocations and cache. Training memory is larger because it can also include gradients, optimizer states, master weights, and activations.

Quantization changes storage and arithmetic formats and can affect quality. MoE separates total stored parameters from active computation. Neither makes the parameter count alone an adequate capacity or cost estimate.

## Lesson 1.8.6 — Why larger models often improve, and where that stops helping

More capacity and appropriate training can model richer dependencies and reuse patterns across more tasks. Better pretraining loss often accompanies stronger downstream capability, but correlations are not guarantees. Domain mismatch, rare factual requirements, weak instruction tuning, context misuse, and evaluation leakage can dominate a specific application.

For production model selection, establish a task-specific acceptance threshold first. Compare candidates on held-out examples, difficult slices, latency, cost, context needs, structured-output behavior, and operational constraints. A smaller model with well-selected evidence may outperform a larger model with irrelevant context on a particular workflow. That is an experiment to run, not a universal claim.

## Lesson 1.8.7 — An architecture review worksheet

When looking at a model report or configuration, record:

- Model family and objective; exact checkpoint and tokenizer versions.
- Dense or MoE; total and active parameters; layer/width/head configuration.
- Training-token accounting, data coverage, and known evaluation limitations.
- Context training range and actual tested context quality.
- Weight precision, cache estimate, and target serving concurrency.
- Held-out task quality and the latency/cost measurement conditions.

Several of those lines are derivable from the configuration rather than looked up, and computing them is the fastest way to catch a spec that does not add up:

```python
def count_parameters(layers, width, ffn_hidden, vocab,
                     query_heads, kv_heads, head_dim, gated=True):
    attention = (width * query_heads * head_dim          # Q
                 + 2 * width * kv_heads * head_dim       # K and V, shrunk by GQA
                 + query_heads * head_dim * width)       # output projection
    ffn = (3 if gated else 2) * width * ffn_hidden       # SwiGLU uses three matrices
    norms = 2 * width
    per_layer = attention + ffn + norms

    return {
        "per layer": per_layer,
        "all layers": layers * per_layer,
        "embeddings": vocab * width,
        "total (untied)": layers * per_layer + 2 * vocab * width,
        "total (tied)": layers * per_layer + vocab * width,
    }


config = dict(layers=24, width=1024, ffn_hidden=2816, vocab=32000,
              query_heads=16, kv_heads=4, head_dim=64)
counts = count_parameters(**config)
for name, value in counts.items():
    print(f"{name:<16} {value:>14,}")

print(f"\nembeddings as a share of the tied total: "
      f"{config['vocab'] * config['width'] / counts['total (tied)']:.1%}")
```

```text
per layer            11,274,240
all layers          270,581,760
embeddings           32,768,000
total (untied)      336,117,760
total (tied)        303,349,760

embeddings as a share of the tied total: 10.8%
```

This is the configuration from 1.5.6, so you can check it against the cache arithmetic there and see the two costs are independent: GQA shrinks K and V here *and* shrinks the cache, while the vocabulary projection affects parameters and compute but never the cache.

Two audit habits come out of this. Tying embeddings changes the reported total by 33M — 10.8% of the model — so "how many parameters" is ambiguous until you know whether input and output embeddings are shared. And at small widths the embedding table is a large fraction of the model, which is why parameter count alone is a poor proxy for capability when comparing across vocabulary sizes.

Then explain which observations support your choice and which claims remain untested. This connects Month 1's internal mechanics to later inference systems, evaluation, and platform architecture modules.

## Checkpoint

1. Why are parameter count and training-token count different resources?
2. What does a scaling law predict directly, and what needs separate validation?
3. Why might a heavily used service favor a smaller model trained longer?
4. Does a tiny model's falling loss establish that it has general-purpose reasoning ability?

<details>
<summary>Show answers</summary>

1. Parameters are learned capacity; tokens are training instances processed to fit that capacity. Both consume compute, and their balance matters.
2. A fitted law predicts an empirical quantity such as loss within its assumptions/range. Application accuracy, safety, latency, cost, and out-of-distribution behavior need direct tests.
3. Higher one-time training cost may be offset by lower per-request inference cost over many requests, provided quality requirements are met.
4. No. It may merely memorize or model narrow local patterns. Generalization and reasoning require appropriate independent tasks and controls.

</details>

## Hands-on exercises

1. Estimate dense training FLOPs for 100 million parameters and 2 billion tokens. Compare with 200 million parameters trained on 1 billion tokens. State what cannot be concluded from the equality.
2. Estimate FP16 weight-only storage for 7 billion parameters in decimal GB and binary GiB. List additional memory a serving system needs.
3. Train two widths in the tiny GPT project under a fixed approximate token-processing budget. Design a report that distinguishes undertraining, overfitting, and genuine held-out improvement.

<details>
<summary>Show exercise solutions</summary>

1. Both give `6×10^8×2×10^9 = 1.2×10^18` FLOPs under the approximation. Equal estimated compute does not establish equal quality, runtime, memory, or cost; attention and implementation differences are omitted.
2. `7×10^9×2 = 14×10^9` bytes: 14 GB, about 13.04 GiB. Add KV cache, temporary activations/workspaces, allocator overhead, and other service allocations.
3. Count parameters and actual processed tokens, record seeds and wall time, use the same held-out split/tokenizer, and plot train/validation loss against processed tokens and estimated compute. Rising validation loss with falling train loss suggests overfitting; slow improvement in both may suggest inadequate training or optimization issues. Repeat runs and avoid drawing foundation-model scaling conclusions from two tiny models on toy data.

</details>

## Completion criteria

Use parameters, tokens, compute, memory, and task quality as separate quantities. Explain compute-optimal allocation and why deployment economics can favor a different choice.

## Primary references

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).

## Videos and code to read

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the scaling arithmetic in this lesson applied to a run you can actually afford; its README reports concrete compute and token budgets
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — standard evaluation, so capability claims rest on measurement rather than parameter count
- [huggingface/transformers](https://github.com/huggingface/transformers) — model config files are the fastest way to compare real architecture and scale choices across families

## Mapped companion lessons

- [Scaling Laws](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/13-scaling-laws) maps directly to empirical loss-versus-compute relationships.
- [Scaling: Distributed Training](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/05-scaling-distributed) extends compute allocation into system design.
- [Inference Platform Economics](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/17-infrastructure-and-production/02-inference-platform-economics) connects model selection to serving cost and operational constraints.

- [Quantization: Making Models Fit](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/11-quantization) maps to the precision and memory tradeoffs named in 1.8.5.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).
