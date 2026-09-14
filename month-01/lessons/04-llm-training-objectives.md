# 1.4 — LLM Training Objectives

**Depth: MASTER**

**Goal:** connect the probability objective, the target tensors, and the optimizer update; recognize training bugs that can make an apparently successful model unusable.

[Month 1 roadmap](../README.md) · [Previous: Tokenization](03-tokenization.md) · [Next: Modern Variants](05-modern-transformer-and-llm-variants.md)

## Lesson 1.4.1 — Autoregressive factorization

For a token sequence `x1, ..., xT`, the chain rule gives:

```text
P(x1, ..., xT) = ∏ P(xt | x1, ..., x(t−1))
```

The model parameterizes these conditionals. A start token or an initial prefix supplies the first context when required by the data convention. Maximizing the training sequences' likelihood is equivalent to minimizing their negative log-likelihood. With one correct token ID per prediction, this is categorical cross-entropy:

```text
loss = −(1/N) Σ log P(correct target token | preceding tokens)
```

`N` is the number of scored, non-ignored target positions, not necessarily the number of sequences. The causal architecture and the shifted targets must agree about what “preceding” means.

## Lesson 1.4.2 — Shift the sequence exactly once

Take the toy ID stream `[8, 3, 6, 2, 9]`. A four-position training example is:

```text
input:    [8, 3, 6, 2]
target:   [3, 6, 2, 9]
position:  0  1  2  3
```

The logit at input position 2 sees the prefix `[8,3,6]` and predicts ID 2. A `[B,T,V]` output supplies `B×T` predictions in one forward pass. The causal mask prevents later input tokens from leaking into earlier predictions.

Some high-level model APIs accept unshifted labels and perform the shift internally. A custom PyTorch loop normally shifts explicitly. Shifting both in the dataset and inside a model skips targets; shifting nowhere trains the wrong task. Write down a five-token example whenever debugging alignment.

The standard tensor path is:

```python
import torch
import torch.nn.functional as F

stream = torch.tensor([[8, 3, 6, 2, 9]])
inputs, targets = stream[:, :-1], stream[:, 1:]
logits = torch.randn(1, 4, 10, requires_grad=True)
loss = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
loss.backward()
assert logits.grad.shape == logits.shape
```

Use `reshape` when slices may be non-contiguous. The flattening must preserve the pairing of each logit row and its target ID.

## Lesson 1.4.3 — Cross-entropy, numerics, and gradients

For one target with assigned probability 0.25, loss is `−ln(0.25) ≈ 1.3863` nats. Giving it probability 0.5 reduces loss to about 0.6931. Probabilities assigned to incorrect tokens affect the normalization and therefore the correct token's probability.

From logits `z`:

```text
loss(z, y) = logsumexp(z) − z[y]
∂loss/∂z[k] = softmax(z)[k] − 1[k=y]
```

The gradient raises the correct logit relative to others during gradient descent. This does not imply each individual parameter always moves in a direction with an obvious linguistic meaning; it participates in many examples and layers.

Pass **raw logits** to [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html). It combines log-softmax and the loss in a numerically stable way. Applying softmax first makes the library treat probabilities as logits, optimizing a different expression. Implementing `log(softmax(z))` naively can also underflow for extreme logits.

Perplexity is `exp(mean token loss)` when loss uses natural logarithms. A uniform distribution over `V` tokens has loss `ln(V)` and perplexity `V`. Perplexity is neither classification accuracy nor a directly comparable score across arbitrary tokenizers and datasets.

## Lesson 1.4.4 — Teacher forcing and the parallel-training distinction

Teacher forcing supplies the true previous tokens during training. Every prefix is already available, so a causal transformer evaluates all target positions in parallel. At generation time, the next prefix contains the model's own sampled tokens:

```text
training:  true prefix → distribution → score true next token
inference: current prefix → distribution → choose token → extend prefix
```

A small mistake can change later contexts during generation. This mismatch is often called exposure bias. Teacher forcing remains a standard, efficient way to optimize autoregressive likelihood; it is not an error to replace every training prefix with samples by default.

Evaluation mode and disabled gradients are separate controls. `model.eval()` changes modules such as dropout. `torch.no_grad()` or inference mode disables gradient tracking. Validation normally needs both. Neither performs a training update.

## Lesson 1.4.5 — Masked language modeling and related objectives

A masked language model sees context on both sides of selected hidden tokens and predicts those tokens. [BERT](https://arxiv.org/abs/1810.04805) is a representative encoder model. Its selected-token corruption includes more than simply replacing every selected token with a mask symbol; distinguish the objective from an oversimplified implementation.

| Objective | Visible information | Scored targets | Natural model pattern |
|---|---|---|---|
| Autoregressive LM | Earlier tokens | Next token at each scored position | Causal decoder |
| Masked LM | Unmasked tokens on both sides | Selected original tokens | Bidirectional encoder |
| Span denoising | Corrupted source and generated target prefix | Missing spans or reconstructed target | Often encoder-decoder |

[T5](https://arxiv.org/abs/1910.10683) provides a concrete text-to-text denoising example. These objectives learn useful representations through different prediction problems. A bidirectional encoder cannot simply be used as an ordinary causal decoder without changing how information flows and how it was trained.

## Lesson 1.4.6 — Why token prediction supports broader capabilities

Predicting realistic continuations rewards representing syntax, entities, discourse, world regularities, code structure, and patterns of problem solving present in data. The model can reuse these internal representations across tasks. This explains why a simple objective can support complex behavior without a separate labeled dataset for each capability.

The objective still rewards predictive fit to a data distribution. It does not directly guarantee truth, robust reasoning, instruction following, or trustworthy uncertainty. A fluent continuation can be wrong. Pretraining, instruction tuning, preference optimization, and application-level evaluation serve different roles; this month establishes the pretraining mechanics before later roadmap modules examine adaptation.

## Lesson 1.4.7 — Loss masking, splits, and reliable validation

For padded targets, mark ignored positions and average only over valid ones. An all-ignored batch has no meaningful mean loss; reject or skip it deliberately. With unequal valid-token counts per batch, aggregate the sum of token losses and divide by the total valid-token count. Averaging batch means gives each batch equal weight regardless of how many predictions it contains.

PyTorch spells this `ignore_index`. The convention is to set ignored target positions to `-100` and pass `ignore_index=-100`; those positions then contribute neither to the numerator nor to the denominator. The three ways to get this wrong all produce a plausible number, which is what makes the bug survive:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
V = 32
logits = torch.randn(3, 6, V)
targets = torch.randint(0, V, (3, 6))
for row, valid_length in enumerate([6, 3, 1]):          # very unequal sequences
    targets[row, valid_length:] = -100

flat_logits, flat_targets = logits.reshape(-1, V), targets.reshape(-1)

correct = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)

per_sequence = []
for row in range(3):
    keep = targets[row] != -100
    per_sequence.append(F.cross_entropy(logits[row][keep], targets[row][keep]))
mean_of_means = torch.stack(per_sequence).mean()

counted_as_class_zero = F.cross_entropy(flat_logits, flat_targets.clamp(min=0))

print(f"correct (token-weighted):   {correct:.4f}  over {(flat_targets != -100).sum().item()}/18 tokens")
print(f"mean of per-sequence means: {mean_of_means:.4f}")
print(f"padding counted as class 0: {counted_as_class_zero:.4f}")
```

```text
correct (token-weighted):   4.0838  over 10/18 tokens
mean of per-sequence means: 3.9209
padding counted as class 0: 4.1080
```

All three are the same order of magnitude, near `log 32 ≈ 3.47` for this untrained tensor, so none looks obviously broken on a loss curve. But they optimize different things:

- **Token-weighted** is almost always what you want: every real prediction counts once.
- **Mean of per-sequence means** gives a 1-token sequence the same weight as a 6-token one. With sorted or bucketed batching this systematically over-weights short sequences.
- **Counting padding as class 0** trains the model to predict the pad token, wasting capacity and corrupting the distribution. Since `-100` is not a valid class index, forgetting `ignore_index` raises an error rather than doing this — but `clamp`, `abs`, or using `0` as the pad id silently converts the crash into this quiet corruption.

The related reduction bug is at the outer level: if you average each batch's mean across the epoch, batches with few valid tokens count as much as full ones. Accumulate the summed token loss and the valid-token count, and divide once.

Separate documents or sources before creating training windows. Randomly splitting overlapping windows leaks near-identical text into validation. Duplicate content can also cross document boundaries, so deduplication and task-relevant evaluation matter.

A healthy tiny-model debugging sequence is: inspect one batch, verify causality, overfit one small training batch, then train with held-out data. Low validation loss with a broken mask is not evidence of language-model quality: the model may be seeing the answer in its input.

The optimizer loop should make every transition visible:

```text
zero gradients → forward → loss → backward → optional gradient clipping
               → optimizer step → record metrics
```

Gradient accumulation intentionally changes when gradients are cleared and when the optimizer steps. Do not accidentally accumulate gradients because `zero_grad()` was forgotten.

## Checkpoint

1. What target aligns with the logit at input index `i` in an explicitly shifted next-token dataset?
2. Why does causal masking still matter when inputs and targets are shifted correctly?
3. Why pass logits directly to cross-entropy?
4. Why does low pretraining loss not prove that the model is a reliable assistant?

<details>
<summary>Show answers</summary>

1. The stream token at `i+1`.
2. That target often also appears as the next input position. Without the mask, earlier logits can attend to it and learn to copy the answer.
3. Cross-entropy implements stable log-softmax plus negative log-likelihood. Passing already normalized probabilities changes the calculation.
4. The objective fits observed continuations. Truthfulness, instruction adherence, domain accuracy, and robust behavior require their own data and evaluation.

</details>

## Hands-on exercises

1. For target probabilities `[0.5, 0.25, 0.125]`, calculate mean loss in nats and perplexity. Derive the logit gradient for a target at index 1 when predicted probabilities are `[0.2, 0.5, 0.3]`.
2. Create shifted inputs/targets from `[4,7,1,8,2,5]` using windows of length 3 at starts 0 and 2. Explain which token each final logit predicts.
3. Batch A has 2 valid tokens with mean loss 1.0; batch B has 6 with mean loss 3.0. Compute the corpus mean and explain the error in averaging the two means.
4. Design two tests that would catch an implementation with suspiciously excellent training loss caused by future-token leakage.

<details>
<summary>Show exercise solutions</summary>

1. Mean loss is `(ln 2 + ln 4 + ln 8)/3 = ln 4 ≈ 1.386294`; perplexity is 4. The gradient is `[0.2, -0.5, 0.3]`.
2. Start 0: input `[4,7,1]`, target `[7,1,8]`. Start 2: input `[1,8,2]`, target `[8,2,5]`. Final logits predict 8 and 5 respectively.
3. `(2×1 + 6×3)/8 = 2.5`. The unweighted average 2.0 overrepresents the smaller batch.
4. Change only future tokens and verify that earlier logits remain unchanged in evaluation mode. Also inspect attention's strictly upper triangle and assert zero mass there. A loss-only test can miss a leakage bug; these tests target the information-flow invariant.

</details>

## Completion criteria

Build shifted batches, derive token cross-entropy and its gradient, explain teacher forcing versus generation, and validate a training run with causal and data-isolation checks. Carry those checks into the project.

## Primary references

- [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) — input/target and reduction semantics.
- [BERT](https://arxiv.org/abs/1810.04805) — masked pretraining.
- [T5](https://arxiv.org/abs/1910.10683) — text-to-text objectives.
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — capabilities from autoregressive pretraining at scale.

## Videos and code to read

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — `train.py` shows shifted targets and the loss reduction in production form
- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — the makemore lectures build next-token prediction up from counting, which makes teacher forcing concrete
- [huggingface/transformers](https://github.com/huggingface/transformers) — `DataCollatorForLanguageModeling` is the canonical reference for label shifting and `-100` masking

## Mapped companion lessons

- [BERT — Masked Language Modeling](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/06-bert-masked-language-modeling) and [GPT — Causal Language Modeling](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling) map to the two principal objectives.
- [Pre-Training a Mini GPT](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/04-pre-training-mini-gpt) applies shifted targets and cross-entropy end to end.
- [Training Loop and Evaluation](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/36-training-loop-eval) provides a focused training artifact.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).
