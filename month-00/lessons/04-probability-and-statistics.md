# 0.4 Probability and Statistics — SKIM

Deep learning is applied probability. A classifier outputs a distribution over classes, a language model outputs a distribution over the vocabulary, and the loss that trains both is a statement about probability. This lesson covers only the probability you need to read that loss and reason about sampling.

| | |
|---|---|
| **Mode** | SKIM |
| **Time** | 60–90 minutes, including the exercises |
| **Assumes** | 0.2 NumPy and Tensor Manipulation, 0.3 Linear Algebra |
| **Used by** | 0.6 Neural-Network Fundamentals, 0.7 Training, the tiny text classifier project, Month 1 sampling, Month 3 structured generation, Month 6 evaluation |

[Month 0 roadmap](../README.md) · [Previous: Linear Algebra](03-linear-algebra-for-deep-learning.md) · [Next: Calculus](05-calculus-for-neural-networks.md)

## Learning objectives

After this lesson you can:

- read a distribution, a conditional probability, and Bayes' theorem, and say what each term means
- compute expectation, variance, and covariance, and explain what each measures
- explain entropy, cross-entropy, and KL divergence, and how they relate
- derive why a language model's loss is `-log P(correct token)`
- explain how temperature, top-k, and top-p change sampling
- state why a confident model is not necessarily a calibrated one

## How to use this lesson

1. Attempt the [exit test](#exit-test) first and read only the sections you miss.
2. The single most important section is [0.4.5](#045--cross-entropy-and-why-the-loss-is--log-pcorrect-token). Do not skim it, even in SKIM mode; the whole roadmap uses it.
3. Record gaps in [`progress.md`](../progress.md).

Examples assume:

```python
import math
import torch
import torch.nn.functional as F
```

## 0.4.1 — Distributions, conditional probability, and Bayes' theorem

A **random variable** takes values with specified probabilities. A **discrete distribution** assigns a probability to each possible value, and those probabilities are nonnegative and sum to 1. A **continuous distribution** uses a density instead, which integrates to 1; a density may exceed 1 at a point, so it is not itself a probability.

Distributions you will meet in this roadmap:

| Distribution | Used for |
|---|---|
| Bernoulli | a single binary outcome, such as a sigmoid output |
| Categorical | one of `V` classes; every softmax output is one of these |
| Gaussian (normal) | weight initialization, noise, diffusion models |
| Uniform | random seeds, dropout masks, sampling baselines |

**Joint** probability `P(A, B)` is the probability of both. **Conditional** probability is the probability of one given the other:

```text
P(A | B) = P(A, B) / P(B)
```

`P(A, B) = P(A)P(B)` exactly when `A` and `B` are independent. A language model is a conditional distribution: `P(next token | all previous tokens)`. The chain rule turns that into the probability of a whole sequence:

```text
P(t₁, t₂, …, t_N) = P(t₁) · P(t₂ | t₁) · … · P(t_N | t₁…t_{N−1})
```

That factorization is the entire idea behind autoregressive generation.

**Bayes' theorem** reverses a conditional:

```text
P(A | B) = P(B | A) · P(A) / P(B)
```

`P(A)` is the prior, `P(B | A)` the likelihood, and `P(A | B)` the posterior. The practical lesson is that a rare condition stays rare even after a positive test: if 0.1% of documents are fraudulent and a detector has a 5% false-positive rate, most positive flags are still false. Base rates dominate. You will use this reasoning when reading evaluation results on imbalanced data.

## 0.4.2 — Expectation, variance, and covariance

**Expectation** is the probability-weighted average, `E[X] = Σ x·P(x)`. **Variance** is the expected squared deviation from the mean, `Var(X) = E[(X − E[X])²]`, and **standard deviation** is its square root, back in the original units.

Two properties get used constantly:

```text
E[aX + b] = a·E[X] + b
Var(aX)   = a²·Var(X)
```

The second explains why scaling a tensor by `1/√d` divides its standard deviation by `√d`, which is the basis of both attention scaling (0.3) and weight initialization (0.6).

**Covariance** measures whether two variables vary together, and **correlation** is covariance normalized to `[-1, 1]`. A covariance matrix collects the pairwise covariances of every feature; its eigenvectors are the principal components (0.3.13).

```python
x = torch.randn(10_000)
print(x.mean(), x.var(), x.std())        # about 0, 1, 1

scaled = x / math.sqrt(64)
print(scaled.std())                      # about 1/8 = 0.125
```

> **Pitfall:** A mini-batch estimate is noisy. Batch statistics, a validation score on 200 examples, and a benchmark run on one seed are all samples, not truths. Smaller batches mean noisier gradients, which is one reason batch size interacts with learning rate (0.7).

## 0.4.3 — Softmax turns scores into a distribution

A model's final layer produces **logits**: unbounded real scores, one per class. Softmax converts them into a categorical distribution:

```text
P(i) = exp(zᵢ) / Σⱼ exp(zⱼ)
```

Exponentiation makes every value positive, and dividing by the sum makes them add to 1. Two consequences matter:

- Only *differences* between logits matter. Adding a constant to every logit leaves the probabilities unchanged, which is what makes the max-subtraction trick in 0.2.16 safe.
- Softmax is monotonic, so the argmax of the logits equals the argmax of the probabilities. You never need softmax to pick the top class, only to interpret or sample.

```python
logits = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(logits, dim=-1)
print(probs, probs.sum())                          # about [0.659, 0.242, 0.099], sums to 1
print(torch.softmax(logits + 100, dim=-1))         # unchanged
```

## 0.4.4 — Entropy, cross-entropy, and KL divergence

**Entropy** measures the average surprise in a distribution:

```text
H(p) = −Σᵢ p(i) log p(i)
```

It is largest when the distribution is uniform, meaning maximum uncertainty, and zero when one outcome is certain. Using base-2 logs gives bits; machine learning uses natural logs, giving nats.

**Cross-entropy** measures the average surprise of data drawn from `p` when you predict with `q`:

```text
H(p, q) = −Σᵢ p(i) log q(i)
```

**KL divergence** is the excess cost of using `q` instead of `p`:

```text
KL(p ‖ q) = H(p, q) − H(p) = Σᵢ p(i) log(p(i) / q(i))
```

KL is nonnegative, and zero exactly when the distributions are identical. It is *not* symmetric and is not a distance.

```python
p = torch.tensor([0.7, 0.2, 0.1])
q = torch.tensor([0.5, 0.3, 0.2])

entropy = -(p * p.log()).sum()
cross = -(p * q.log()).sum()
kl = (p * (p / q).log()).sum()

assert torch.allclose(kl, cross - entropy, atol=1e-6)
print(entropy.item(), cross.item(), kl.item())     # 0.802, 0.887, 0.085
```

Because `H(p)` is fixed by the data, minimizing cross-entropy is the same as minimizing KL divergence from the true distribution. That equivalence is why one loss serves both framings. KL returns directly in Month 2, where RLHF and DPO penalize drift from a reference model.

## 0.4.5 — Cross-entropy and why the loss is `-log P(correct token)`

This is the most important derivation in Month 0.

In classification, the true distribution for one example is **one-hot**: probability 1 on the correct class `c`, and 0 everywhere else. Substituting that into the cross-entropy formula collapses the sum, because every term with `p(i) = 0` vanishes:

```text
H(p, q) = −Σᵢ p(i) log q(i)
        = −1 · log q(c)
        = −log P(correct class)
```

So the loss for one example is the negative log of the probability the model assigned to the right answer. The batch loss is the average over examples.

The shape of `-log` explains the training signal:

| `P(correct)` | Loss |
|---|---|
| 1.0 | 0.00 |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 0.01 | 4.61 |
| → 0 | → ∞ |

Confident and right costs nothing. Confident and wrong is punished without limit, which is exactly the gradient signal you want.

This also gives a sanity check you will use for the rest of the roadmap: **an untrained model over `C` classes should start at a loss near `log C`**, because it spreads probability roughly uniformly. For 2 classes that is 0.69; for a 50,000-token vocabulary, about 10.8. A training run that starts far from `log C` usually signals a bug in the labels, the masking, or the initialization.

```python
vocab_size = 50_000
logits = torch.zeros(4, vocab_size)                       # untrained: all classes equal
targets = torch.randint(0, vocab_size, (4,))
loss = F.cross_entropy(logits, targets)

print(loss.item(), math.log(vocab_size))                  # both about 10.82

logits = torch.tensor([[2.0, 1.0, 0.1]])
target = torch.tensor([0])
manual = -torch.log_softmax(logits, dim=-1)[0, 0]
assert torch.allclose(F.cross_entropy(logits, target), manual)
```

Two derived quantities you will see reported:

- **Perplexity** is `exp(cross-entropy)`, roughly "how many tokens the model is effectively choosing between". A loss of 2.3 is a perplexity of about 10.
- **Negative log-likelihood** is the same quantity viewed as [maximum likelihood estimation](#046--likelihood-and-maximum-likelihood).

> **Pitfall:** `F.cross_entropy` expects raw logits and integer class targets, and applies `log_softmax` internally. Passing probabilities applies softmax twice and quietly trains a flatter objective. Use `F.nll_loss` if you already have log-probabilities.

## 0.4.6 — Likelihood and maximum likelihood

**Likelihood** is the probability of the observed data viewed as a function of the parameters. **Maximum likelihood estimation** picks the parameters that make the observed data most probable.

Products of many small probabilities underflow, so we maximize the log-likelihood instead. Maximizing a log-likelihood is the same as minimizing its negative, which is the cross-entropy loss from 0.4.5:

```text
maximize  Σ log P(token | context)      ≡      minimize  −Σ log P(token | context)
```

So "train a language model on next-token prediction" and "maximize the likelihood of the corpus" are the same instruction. Month 2's supervised fine-tuning is the same objective on a narrower dataset.

## 0.4.7 — Sampling: temperature, top-k, and top-p

At inference, a language model gives you a distribution and you must choose a token. **Greedy decoding** always takes the argmax; it is deterministic and tends to be repetitive. **Sampling** draws from the distribution, which is more varied but sometimes incoherent.

**Temperature** `T` divides the logits before softmax:

```text
P(i) = softmax(zᵢ / T)
```

- `T < 1` sharpens the distribution toward the top tokens; `T → 0` approaches greedy.
- `T = 1` leaves the model's own distribution unchanged.
- `T > 1` flattens it, raising the chance of rare tokens.

**Top-k** keeps only the `k` highest-probability tokens and renormalizes. **Top-p** (nucleus) keeps the smallest set whose cumulative probability reaches `p`, so the candidate set grows when the model is uncertain and shrinks when it is confident.

```python
logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.1])

for T in [0.5, 1.0, 2.0]:
    print(T, torch.softmax(logits / T, dim=-1).round(decimals=3).tolist())
# 0.5 [0.860, 0.116, 0.016, 0.006, 0.003]   sharper: the top token dominates
# 1.0 [0.610, 0.224, 0.083, 0.050, 0.034]   the model's own distribution
# 2.0 [0.401, 0.243, 0.147, 0.115, 0.094]   flatter: the tail gains probability

def top_p_filter(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    sorted_logits, sorted_index = logits.sort(descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    remove = probs.cumsum(dim=-1) - probs >= p        # keep the token that crosses p
    sorted_logits[remove] = float("-inf")
    return sorted_logits.scatter(0, sorted_index, sorted_logits)

filtered = top_p_filter(logits.clone(), p=0.9)
print(torch.softmax(filtered, dim=-1).round(decimals=3).tolist())
# [0.665, 0.245, 0.090, 0.0, 0.0]: the tail is dropped and the rest renormalized
```

Temperature, top-k, and top-p stack, and the order matters in real implementations. Month 1 covers decoding strategies in full.

## 0.4.8 — Statistics of an evaluation result

Probability describes uncertainty under a model. Statistics is the other direction: using a finite sample to say something about the underlying process. Every evaluation number you report is a sample, and treating it as exact is the most common statistical error in ML practice.

**Accuracy on an imbalanced set is misleading.** Report per-class recall and **balanced accuracy**, the unweighted mean of per-class recalls, alongside it:

```python
majority_correct, majority_total = 70, 80
minority_correct, minority_total = 5, 20

accuracy = (majority_correct + minority_correct) / (majority_total + minority_total)
recalls = [majority_correct / majority_total, minority_correct / minority_total]
balanced = sum(recalls) / len(recalls)

print(f"accuracy          {accuracy:.3f}")            # 0.750
print(f"per-class recall  {[round(r, 3) for r in recalls]}")   # [0.875, 0.25]
print(f"balanced accuracy {balanced:.4f}")            # 0.5625
```

75% accuracy sounds respectable. Balanced accuracy of 56% says the model is barely better than chance on the minority class, which is usually the class that matters.

**A metric has a standard error.** Treating each prediction as an independent Bernoulli trial gives a rough interval:

```python
n = 100
standard_error = math.sqrt(accuracy * (1 - accuracy) / n)
print(f"accuracy {accuracy:.3f} ± {1.96 * standard_error:.3f} (95%)")   # 0.750 ± 0.085
```

On 100 examples the 95% interval spans roughly 8.5 points in each direction, so a model scoring 0.75 and one scoring 0.78 are indistinguishable on this data. Two caveats: the approximation assumes independent, identically distributed examples, which fails for grouped or time-ordered data; and it ignores the uncertainty added by selecting the model on the same set, which is the multiple-comparisons problem behind many irreproducible results.

The practical rule: **before claiming an improvement, check whether the evaluation set is large enough to see it.**

## 0.4.9 — Confidence versus probability

A softmax output of 0.98 is a number the model produced, not evidence that it is right 98% of the time. A model is **calibrated** when predictions made with confidence `c` are correct about `c` of the time. Neural networks are often overconfident, and this gets worse with capacity and with training past the point of fitting.

Practical consequences:

- Measure calibration rather than assuming it, with reliability diagrams or expected calibration error.
- Temperature scaling on a validation set is a simple, effective post-hoc fix.
- For an LLM, a fluent, confident-sounding answer carries no probabilistic guarantee at all. Token probabilities describe the model's next-token distribution, not the truth of a claim. This is the root of why Month 6 evaluation and Month 4 retrieval grounding exist.

## Exercises

### Exercise 1 — Verify the `log C` starting point

For `C` in `[2, 10, 1000, 50_000]`, build uniform logits, compute the cross-entropy against random targets, and confirm it equals `log C`. Then shift every logit by +100 and confirm the loss is unchanged.

### Exercise 2 — Cross-entropy from first principles

Given `logits: [8, 5]` and integer `targets: [8]`, compute the loss three ways and show all three agree: manual `-log_softmax` selection with `gather`, `F.nll_loss` on log-probabilities, and `F.cross_entropy` on the logits.

### Exercise 3 — Entropy and the temperature curve

Sample a random logit vector of length 100. For temperatures from 0.1 to 5.0, compute the entropy of the resulting distribution and the probability of the top token. Describe the trend and explain the two limits.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
for C in [2, 10, 1_000, 50_000]:
    logits = torch.zeros(16, C)
    targets = torch.randint(0, C, (16,))
    loss = F.cross_entropy(logits, targets)
    assert torch.allclose(loss, torch.tensor(math.log(C)), atol=1e-4)
    assert torch.allclose(F.cross_entropy(logits + 100, targets), loss, atol=1e-4)
    print(f"C={C:6d}  loss={loss.item():.4f}  log C={math.log(C):.4f}")
```

Uniform logits give every class probability `1/C`, so the loss is `-log(1/C) = log C`. Shifting all logits equally cancels inside softmax.

### Exercise 2

```python
torch.manual_seed(0)
logits = torch.randn(8, 5)
targets = torch.randint(0, 5, (8,))

log_probs = torch.log_softmax(logits, dim=-1)
manual = -log_probs.gather(1, targets[:, None]).squeeze(1).mean()
nll = F.nll_loss(log_probs, targets)
built_in = F.cross_entropy(logits, targets)

assert torch.allclose(manual, nll) and torch.allclose(nll, built_in)
print(built_in.item())
```

`F.cross_entropy` is exactly `log_softmax` followed by `nll_loss`.

### Exercise 3

```python
torch.manual_seed(0)
logits = torch.randn(100) * 2

for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
    probs = torch.softmax(logits / T, dim=-1)
    entropy = -(probs * probs.log()).sum()
    print(f"T={T:4.1f}  entropy={entropy.item():.3f}  top prob={probs.max().item():.3f}")
```

Entropy rises with temperature and the top probability falls. As `T → 0` the distribution approaches one-hot: entropy 0 and greedy decoding. As `T` grows it approaches uniform: entropy approaches `log 100 ≈ 4.6` and sampling becomes noise.

</details>

## Exit test

1. What is the difference between a joint, a marginal, and a conditional probability?
2. Write Bayes' theorem and name each term.
3. Why does a rare condition stay rare after a positive test?
4. What does variance measure, and what happens to it when you scale a variable by `a`?
5. What does softmax do, and why do only logit differences matter?
6. Define entropy, cross-entropy, and KL divergence, and give the relationship between them.
7. Derive `-log P(correct token)` from the cross-entropy formula.
8. What loss should an untrained model over 50,000 tokens show, and why?
9. What is perplexity?
10. How does maximum likelihood relate to cross-entropy?
11. What do temperature, top-k, and top-p each change about sampling?
12. Why is a softmax probability of 0.98 not the same as being right 98% of the time?
13. Why report balanced accuracy alongside accuracy, and what does the standard error of a metric tell you?

<details>
<summary>Show answers</summary>

1. Joint is the probability of two events together, `P(A, B)`. Marginal is the probability of one alone, summing the joint over the other. Conditional is the probability of one given the other, `P(A | B) = P(A, B) / P(B)`.
2. `P(A | B) = P(B | A)P(A) / P(B)`: posterior equals likelihood times prior, divided by the evidence.
3. Because the prior dominates. When the base rate is far below the false-positive rate, most positives come from the large negative population.
4. Variance is the expected squared deviation from the mean. Scaling by `a` multiplies variance by `a²` and standard deviation by `|a|`.
5. It maps unbounded logits to a categorical distribution by exponentiating and normalizing. Adding a constant to every logit multiplies every exponential by the same factor, which cancels in the normalization.
6. Entropy `H(p) = −Σ p log p` is average surprise under `p`. Cross-entropy `H(p, q) = −Σ p log q` is average surprise when predicting with `q`. KL is their difference, `KL(p ‖ q) = H(p, q) − H(p)`, nonnegative and asymmetric.
7. With a one-hot target, `p(i)` is 0 for every class except the correct one, so all terms vanish except `−1 · log q(c)`.
8. About `log 50000 ≈ 10.82`, because an untrained model assigns roughly uniform probability, making `P(correct) ≈ 1/50000`.
9. `exp(cross-entropy)`, interpretable as the effective number of choices the model is deciding among.
10. Maximizing log-likelihood is minimizing negative log-likelihood, which is exactly cross-entropy against the observed data.
11. Temperature rescales logits before softmax, sharpening (`T < 1`) or flattening (`T > 1`) the distribution. Top-k restricts sampling to the `k` most probable tokens. Top-p restricts it to the smallest set whose cumulative probability reaches `p`, so the candidate set adapts to the model's certainty.
12. Softmax outputs are not calibrated by default. Neural networks are commonly overconfident, so predictions at 0.98 confidence may be correct far less than 98% of the time unless calibration has been measured and corrected.
13. Accuracy on an imbalanced set is dominated by the majority class, while balanced accuracy averages per-class recall and exposes weak minority-class performance. The standard error says how much the number would move on a different sample of the same size, so it tells you whether an apparent improvement is real or within noise.

</details>

## Completion criteria

You are done when:

- you can derive `-log P(correct token)` on paper without notes
- you can state the expected initial loss for any number of classes and explain why
- you can explain the relationship between entropy, cross-entropy, KL divergence, and maximum likelihood
- you can describe what temperature, top-k, and top-p change
- you can explain why model confidence is not calibrated probability
- your three exercises run with passing assertions

## References

**Distributions, conditional probability, Bayes**
- [Seeing Theory](https://seeing-theory.brown.edu/) — visual introduction to probability
- [3Blue1Brown: Bayes' theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)

**Expectation, variance, covariance**
- [Mathematics for Machine Learning](https://mml-book.github.io/), chapter 6

**Entropy, cross-entropy, KL divergence**
- [Visual Information Theory, Chris Olah](https://colah.github.io/posts/2015-09-Visual-Information/)
- [Deep Learning Book](https://www.deeplearningbook.org/contents/prob.html), sections 3.13 and 5.5

**Cross-entropy in PyTorch**
- [`F.cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [`F.nll_loss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html)

**Sampling and decoding**
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — the top-p paper
- [How to generate text, Hugging Face](https://huggingface.co/blog/how-to-generate)

**Evaluation statistics and calibration**
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- [NIST/SEMATECH e-Handbook of Statistical Methods](https://www.itl.nist.gov/div898/handbook/)
- [PyTorch distributions](https://docs.pytorch.org/docs/stable/distributions.html)

## Videos and code to read

- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) — the single best video source for this lesson; the entropy, cross-entropy, and softmax explanations are short and exact
- [3Blue1Brown: Bayes' theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) — the geometric picture that makes base rates obvious
- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — `makemore` builds a language model up from raw counts to negative log-likelihood, which is this lesson made concrete

## Mapped companion lessons

- [Probability and Distributions](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/06-probability-and-distributions), [Bayes' Theorem](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/07-bayes-theorem), and [Statistics for Machine Learning](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/15-statistics-for-ml) reinforce the probability and estimation sections.
- [Information Theory](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/09-information-theory) maps to entropy, cross-entropy, and KL divergence.
- [Sampling Methods](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/16-sampling-methods) extends the sampling and generation connection.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).

## About this lesson

Written to cover section 0.4 of the [Month 0 curriculum](../README.md). Code examples were checked with PyTorch 2.14 on CPU.
