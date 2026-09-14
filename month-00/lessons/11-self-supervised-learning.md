# 0.11 Self-Supervised Learning — LEARN / REFRESH

Self-supervised learning is the reason foundation models exist. Labels are expensive and scarce; raw text, images, and audio are effectively unlimited. SSL turns unlabeled data into a supervised problem by hiding part of the input and predicting it from the rest. This is a LEARN-mode topic, so aim to explain the mechanisms, not just recognize the names.

| | |
|---|---|
| **Mode** | LEARN / REFRESH |
| **Time** | 60–90 minutes, including the exercises |
| **Assumes** | 0.4 Probability (cross-entropy), 0.10 Representation Learning |
| **Used by** | 0.12 NLP, Month 1 pretraining objectives, Month 2 fine-tuning, Month 4 embedding models |

[Month 0 roadmap](../README.md) · [Previous: Representation Learning](10-representation-learning.md) · [Next: Basic NLP Concepts](12-basic-nlp-concepts.md)

## Learning objectives

After this lesson you can:

- distinguish supervised, unsupervised, and self-supervised learning, and explain why SSL is not simply unsupervised
- explain a pretext task and what makes one useful
- describe contrastive learning, including why negatives matter and how collapse is avoided
- compare masked prediction, autoregressive prediction, and teacher/student distillation
- state what BERT, GPT, and DINO each predict, and what representation each produces
- explain why next-token prediction on enough data yields general capability

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Section [0.11.5](#0115--the-three-way-comparison-bert-gpt-dino) is what the curriculum asks you to compare explicitly.
3. Record gaps in [`progress.md`](../progress.md).

Examples assume:

```python
import torch
import torch.nn.functional as F
from torch import nn
```

## 0.11.1 — Three learning paradigms

| Paradigm | Supervision | Example |
|---|---|---|
| Supervised | human-provided labels | image → "cat"; review → "positive" |
| Unsupervised | none; find structure | k-means clustering, PCA |
| Self-supervised | labels derived automatically from the data itself | predict a masked word from its context |

The distinction that matters: SSL still trains with a **supervised loss and explicit targets**, exactly like supervised learning. What is different is where the targets come from. Nobody labeled anything; the target was extracted from the input by hiding part of it. That is why SSL scales, and why it is not the same thing as unsupervised learning.

The standard pipeline:

```text
huge unlabeled corpus → pretraining on a pretext task → general representations
                                                      → fine-tune on a small labeled set (0.9)
```

A **pretext task** is a training objective whose solution is not interesting in itself, but which forces the model to learn something that is. Nobody needs a system that fills in blanks. We want the language understanding that filling in blanks requires.

A good pretext task is cheap to generate at scale, impossible to solve with a shortcut, and demands the structure you actually care about. Early vision pretext tasks such as predicting image rotation or jigsaw permutations were partly abandoned because models found shortcuts, like reading JPEG artifacts or edge continuity, instead of learning about objects.

## 0.11.2 — Masked prediction

Hide part of the input and reconstruct it from the surrounding context.

**BERT's masked language modeling** replaces about 15% of tokens with `[MASK]` and predicts the originals. Because the model sees both sides, its representations are **bidirectional**, which is ideal for understanding tasks like classification and retrieval:

```text
input:  The cat sat on the [MASK].
target: mat
```

BERT actually splits that 15%: 80% become `[MASK]`, 10% a random token, and 10% stay unchanged. The reason is a train/inference mismatch: `[MASK]` never appears at inference time, so a model that only ever conditions on the literal `[MASK]` token learns a distribution it will never use. The random and unchanged cases force it to build a useful representation for every token.

```python
torch.manual_seed(0)
tokens = torch.tensor([5, 12, 7, 30, 2, 18])
mask_id, vocab = 103, 100

probability = torch.rand(tokens.shape)
selected = probability < 0.15
inputs, labels = tokens.clone(), torch.full_like(tokens, -100)   # -100 = ignored (0.6.4)
labels[selected] = tokens[selected]

replace = selected & (torch.rand(tokens.shape) < 0.8)
inputs[replace] = mask_id
print(inputs.tolist(), labels.tolist())
```

Only the masked positions contribute to the loss, so BERT learns from roughly 15% of tokens per pass. That is a large part of why masked models need more epochs over their data than autoregressive ones.

The same idea transfers to vision. **Masked autoencoders (MAE)** mask 75% or more of image patches and reconstruct the pixels. Images are far more redundant than text, so a much higher masking ratio is needed to make the task hard.

## 0.11.3 — Autoregressive prediction

Predict the next token from everything before it. This is GPT's objective, and the loss is exactly the cross-entropy of 0.4.5:

```text
input:  The cat sat on the
target: mat
```

Three properties follow:

- **Every position is a training signal.** For a sequence of `N` tokens you get `N` predictions, not the 15% of masked modeling. This is a major efficiency advantage.
- **It requires causal masking** so position `i` cannot see position `i+1` (0.2.14). Without it the task is trivial, and a subtly wrong mask produces a model that looks superb in training and fails at generation.
- **It is directly generative.** Sampling from the learned conditional distribution produces text; nothing else is needed.

```python
torch.manual_seed(0)
sequence = torch.randint(0, 100, (1, 8))
inputs, targets = sequence[:, :-1], sequence[:, 1:]      # shift by one

logits = torch.randn(1, 7, 100)                          # stand-in for a model
loss = F.cross_entropy(logits.reshape(-1, 100), targets.reshape(-1))
print(inputs.shape, targets.shape, round(loss.item(), 3))   # [1,7] [1,7] near log 100 = 4.6
```

**Why does this produce general capability?** Because minimizing prediction error on a sufficiently broad corpus requires modeling whatever generates that text. Predicting the last token of "The capital of France is" requires a fact. Predicting the answer after "17 × 24 =" requires arithmetic. Predicting the next line of a proof requires following an argument. The objective is narrow; the competence needed to do it well across a whole corpus is not. This is the central bet behind LLMs, and it is an empirical finding rather than a theoretical guarantee.

## 0.11.4 — Contrastive learning

Instead of reconstructing inputs, learn a space where related things are close and unrelated things are far apart (0.10.2). Take two augmented views of the same image as a **positive** pair, and other images in the batch as **negatives**.

The InfoNCE loss is a softmax over similarities, and the structure should look familiar: it is cross-entropy where "the correct class" is the matching pair.

```text
loss = −log[ exp(sim(a, a⁺)/τ) / Σⱼ exp(sim(a, xⱼ)/τ) ]
```

```python
def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """a, b: [B, D] matched pairs; a[i] corresponds to b[i]."""
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temperature                       # [B, B] similarity matrix
    targets = torch.arange(len(a))                       # the diagonal is correct
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

torch.manual_seed(0)
anchors = torch.randn(8, 32)
random_pairs = torch.randn(8, 32)
matched_pairs = anchors + 0.1 * torch.randn(8, 32)       # views of the same items

print(round(info_nce(anchors, random_pairs).item(), 3))   # 4.937: unrelated pairs are unidentifiable
print(round(info_nce(anchors, matched_pairs).item(), 3))  # 0.0: matched pairs stand out clearly
```

Three things to understand:

- **Negatives prevent collapse.** Without them, mapping every input to the same constant vector makes positives maximally similar and the loss perfect. Negatives are what make that solution costly. SimCLR needs large batches for this reason; MoCo uses a momentum-updated queue of negatives instead.
- **Temperature `τ` controls sharpness** exactly as in 0.4.7. Low temperature concentrates the loss on the hardest negatives.
- **Augmentation defines what "similar" means.** If your augmentations include color jitter, you are instructing the model that color is irrelevant. The choice of augmentation *is* the specification of the invariance you want.

**CLIP** applies this across modalities: images and their captions are the positive pairs, and every other caption in the batch is a negative. The result is the shared image-text space from 0.10.8.

Some methods avoid negatives entirely. **BYOL** and **DINO** use two networks, a student and a teacher, where the teacher's weights are an exponential moving average of the student's. The student predicts the teacher's output for a different view. Collapse is prevented architecturally instead, through the momentum teacher plus centering and sharpening of the outputs. DINO's attention maps segment objects without ever being given a segmentation label, which is a striking demonstration that the representation, not the pretext task, is the product.

## 0.11.5 — The three-way comparison: BERT, GPT, DINO

The comparison the curriculum asks for:

| | BERT | GPT | DINO |
|---|---|---|---|
| Modality | text | text | images |
| Objective | masked token prediction | next-token prediction | match a teacher's output across views |
| Context | bidirectional | causal, left to right | global and local crops |
| Supervision from | the masked tokens | the following token | another view, via a momentum teacher |
| Loss | cross-entropy on 15% of positions | cross-entropy on every position | cross-entropy against teacher probabilities |
| Negatives | not applicable | not applicable | none; collapse avoided by the momentum teacher and centering |
| Produces | bidirectional encodings | a generative model | features good enough for nearest-neighbor classification |
| Best at | classification, retrieval, token tagging | generation, in-context learning | transferable visual features, segmentation |

The unifying view: all three are **prediction problems built from unlabeled data**, and all three are optimized with cross-entropy. They differ in what is hidden and what does the hiding.

One important asymmetry: BERT's bidirectionality makes it a better encoder but not a generator, since it has no natural way to produce text left to right. GPT's causal constraint costs it bidirectional context but makes generation and in-context learning fall out for free. That trade-off, plus scale, is why decoder-only models dominate today, and why encoder models remain the better choice for embedding and retrieval work (Month 4).

## Exercises

### Exercise 1 — Build a masking function

Write `mask_tokens(tokens, mask_id, vocab_size, probability=0.15)` implementing BERT's 80/10/10 split, returning inputs and labels with `-100` at unmasked positions. Verify on 10,000 tokens that about 15% are selected and the sub-splits are close to 80/10/10, and that loss is computed only at masked positions.

### Exercise 2 — Autoregressive targets and causal masking

Build inputs and shifted targets from a sequence. Then show that without a causal mask a trivial "copy the next input" model achieves near-zero loss, which is the leakage a wrong mask creates.

### Exercise 3 — Contrastive learning and collapse

Using the `info_nce` function above, train a small encoder on paired views and show the loss falling. Then replace the loss with positives-only similarity. Measure collapse two ways: the spread of the representations across the batch, and the mean cosine similarity between *different* items, which should stay near zero in a healthy space.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
def mask_tokens(tokens: torch.Tensor, mask_id: int, vocab_size: int, probability: float = 0.15):
    labels = torch.full_like(tokens, -100)
    selected = torch.rand(tokens.shape) < probability
    labels[selected] = tokens[selected]

    inputs = tokens.clone()
    roll = torch.rand(tokens.shape)
    inputs[selected & (roll < 0.8)] = mask_id                              # 80% -> [MASK]
    random_slot = selected & (roll >= 0.8) & (roll < 0.9)                  # 10% -> random
    inputs[random_slot] = torch.randint(0, vocab_size, (int(random_slot.sum()),))
    return inputs, labels                                                  # remaining 10% unchanged


torch.manual_seed(0)
tokens = torch.randint(0, 1000, (10_000,))
inputs, labels = mask_tokens(tokens, mask_id=103, vocab_size=1000)

selected = labels != -100
print(f"selected           {selected.float().mean():.3f}")
print(f"of those, [MASK]   {(inputs[selected] == 103).float().mean():.3f}")
print(f"of those, unchanged{(inputs[selected] == tokens[selected]).float().mean():.3f}")

logits = torch.randn(10_000, 1000)
print(f"loss on masked only {F.cross_entropy(logits, labels, ignore_index=-100).item():.3f}")
```

About 15% are selected, roughly 80% of those become `[MASK]`, and about 10% stay unchanged (slightly more, since a random replacement can coincidentally draw the original token). The loss uses only masked positions.

### Exercise 2

```python
torch.manual_seed(0)
vocab, N = 100, 12
sequence = torch.randint(0, vocab, (4, N))
inputs, targets = sequence[:, :-1], sequence[:, 1:]

# a model that can see the next token trivially wins
leaked = F.one_hot(targets, vocab).float() * 20.0
print(f"with leakage      {F.cross_entropy(leaked.reshape(-1, vocab), targets.reshape(-1)):.6f}")

uniform = torch.zeros(4, N - 1, vocab)
print(f"no information    {F.cross_entropy(uniform.reshape(-1, vocab), targets.reshape(-1)):.4f}")

scores = torch.randn(N, N)
causal = torch.tril(torch.ones(N, N, dtype=torch.bool))
weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
assert torch.all(weights.triu(diagonal=1) == 0)      # no attention to the future
print("causal mask verified")
```

Leakage gives a loss near zero; no information gives `log 100 ≈ 4.6`. A training loss far below what the task should allow is the signature of a masking or shifting bug, and it is the first thing to check when results look too good.

### Exercise 3

```python
torch.manual_seed(0)
encoder = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)

base = torch.randn(64, 32)
view_a = base + 0.8 * torch.randn(64, 32)              # two noisy views of the same items
view_b = base + 0.8 * torch.randn(64, 32)


def off_diagonal_similarity(embeddings: torch.Tensor) -> float:
    """Mean cosine similarity between different items; near 0 is healthy."""
    normalized = F.normalize(embeddings, dim=-1)
    similarity = normalized @ normalized.T
    return similarity[~torch.eye(len(embeddings), dtype=torch.bool)].mean().item()


for step in range(400):
    loss = info_nce(encoder(view_a), encoder(view_b))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"InfoNCE step {step:3d}: loss {loss.item():.4f}")

# positives only: with no negatives, collapse is the optimal solution
torch.manual_seed(0)
collapsed = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
optimizer = torch.optim.AdamW(collapsed.parameters(), lr=1e-3)

for step in range(400):
    a = F.normalize(collapsed(view_a), dim=-1)
    b = F.normalize(collapsed(view_b), dim=-1)
    loss = -(a * b).sum(-1).mean()                     # maximize positive similarity only
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.inference_mode():
    for name, model in [("contrastive", encoder), ("positives-only", collapsed)]:
        embeddings = model(view_a)
        spread = F.normalize(embeddings, dim=-1).std(dim=0).mean().item()
        print(f"{name:16s} spread {spread:.4f}   similarity between different items "
              f"{off_diagonal_similarity(embeddings):+.4f}")
```

```text
InfoNCE step   0: loss 1.7052
InfoNCE step 100: loss 0.0085
InfoNCE step 300: loss 0.0033
contrastive      spread 0.1771   similarity between different items -0.0126
positives-only   spread 0.0424   similarity between different items +0.9397
```

The contrastive model learns to tell the 64 items apart, and unrelated items end up nearly orthogonal, which is what a healthy embedding space looks like (0.3.9).

The positives-only model reaches a positive-pair similarity of about 0.9997 and looks like a triumph by its own loss. But every *different* item is now at 0.94 similarity too: it has pushed everything toward one direction, so the space cannot distinguish anything. That is collapse. The loss curve alone would never reveal it, which is why you measure the spread. Negatives make collapse expensive, and DINO's momentum teacher with centering achieves the same end without them.

</details>

## Exit test

1. Distinguish supervised, unsupervised, and self-supervised learning.
2. Why is SSL not just unsupervised learning?
3. What is a pretext task, and what makes a good one?
4. What does BERT predict, and why the 80/10/10 split?
5. Why does masked language modeling learn from only about 15% of tokens per pass?
6. What does GPT predict, and why does it need a causal mask?
7. Why does next-token prediction produce general capability?
8. What are positives and negatives in contrastive learning?
9. What is collapse, and what prevents it?
10. What do temperature and augmentation each control in contrastive learning?
11. How does DINO avoid needing negatives?
12. Which of BERT, GPT, and DINO would you choose for retrieval, for generation, and for transferable visual features?

<details>
<summary>Show answers</summary>

1. Supervised learning uses human-provided labels. Unsupervised learning finds structure with no targets. Self-supervised learning derives targets automatically from the data itself.
2. Because it still optimizes a supervised loss against explicit targets. Only the source of those targets differs; they are extracted from the input rather than annotated.
3. A training objective whose solution is uninteresting but which forces the model to learn useful structure. A good one is cheap to generate at scale, has no shortcut solution, and requires the structure you care about.
4. Randomly masked tokens, given bidirectional context. The split exists because `[MASK]` never appears at inference, so random and unchanged tokens force the model to represent every position rather than only the literal mask token.
5. Because the loss is computed only at masked positions, so roughly 85% of the tokens in each pass provide context but no training signal.
6. The next token, given all previous ones. The causal mask prevents any position from attending to future positions, which would make the task trivial and break generation.
7. Because predicting text well across a broad corpus requires modeling the facts, reasoning, and structure that produced it. The objective is narrow, but the competence required to minimize it at scale is not.
8. A positive is a pair that should be close, such as two augmentations of one image or an image and its caption. Negatives are pairs that should be far apart, usually the other items in the batch.
9. Mapping every input to the same vector, which trivially satisfies a positives-only objective while carrying no information. Negatives make it costly; alternatively a momentum teacher with centering and sharpening prevents it architecturally.
10. Temperature controls how sharply the loss focuses on the hardest negatives. Augmentation defines what the model is told to treat as irrelevant, and therefore what "similar" means.
11. It uses a student network and a momentum-averaged teacher, with the student matching the teacher's output distribution across different views. Centering and sharpening of the teacher outputs keep the solution from degenerating.
12. BERT-style encoders for retrieval, GPT-style decoders for generation, DINO for transferable visual features.

</details>

## Completion criteria

You are done when:

- you can explain why SSL scales where supervised learning does not
- you can implement BERT-style masking and autoregressive target shifting from scratch
- you can explain collapse and every mechanism used to prevent it
- you can fill in the BERT/GPT/DINO comparison table from memory
- your three exercises run, including the collapse demonstration

## References

**Masked prediction**
- [BERT](https://arxiv.org/abs/1810.04805)
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [The Illustrated BERT, Jay Alammar](https://jalammar.github.io/illustrated-bert/)

**Autoregressive prediction**
- [Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)

**Contrastive and teacher/student**
- [SimCLR](https://arxiv.org/abs/2002.05709)
- [MoCo](https://arxiv.org/abs/1911.05722)
- [BYOL: Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733)
- [DINO](https://arxiv.org/abs/2104.14294) and [DINOv2](https://arxiv.org/abs/2304.07193)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) — the InfoNCE loss

**Overviews**
- [Self-supervised learning: The dark matter of intelligence, Meta AI](https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/)
- [Lilian Weng: Self-supervised representation learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)

## Videos and code to read

- [facebookresearch/dino](https://github.com/facebookresearch/dino) — the teacher/student implementation, including the centering and sharpening that prevent collapse
- [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers) — contrastive losses (MultipleNegativesRanking) as used in production embedding training
- [labmlai annotated implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) — annotated masked-modeling and contrastive code

## Mapped companion lessons

- [Self-Supervised Vision — SimCLR, DINO, MAE](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/04-computer-vision/17-self-supervised-vision) maps to contrastive, teacher-student, and masked-image objectives.
- [BERT — Masked Language Modeling](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/06-bert-masked-language-modeling) and [GPT — Causal Language Modeling](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling) map to the language objectives.
- [CLIP and Contrastive Pretraining](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/12-multimodal-ai/02-clip-contrastive-pretraining) extends the objective to paired modalities.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).

## About this lesson

Written to cover section 0.11 of the [Month 0 curriculum](../README.md). Code examples were checked with PyTorch 2.14 on CPU.
