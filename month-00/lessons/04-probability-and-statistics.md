# 0.4 — Probability and Statistics

**Depth: SKIM**

**Goal:** read model probabilities and evaluation statistics correctly, and recognize when uncertainty, dependence, or sampling invalidates a conclusion.

[Month 0 roadmap](../README.md) · [Previous: Linear Algebra](03-linear-algebra-for-deep-learning.md) · [Next: Calculus](05-calculus-for-neural-networks.md)

## The working mental model

Probability describes uncertainty under a model; statistics uses observed samples to learn about an underlying process. In AI systems, the model's token probability, the empirical frequency of an event, and confidence that a metric will generalize are three different quantities.

For events `A` and `B`:

```text
P(A | B) = P(A ∩ B) / P(B)
P(A | B) = P(B | A) P(A) / P(B)      Bayes' theorem
```

Bayes' theorem updates a prior belief `P(A)` using the likelihood of evidence `P(B|A)`. A high likelihood does not imply a high posterior when the event is extremely rare. This base-rate effect matters in fraud, safety, and medical classifiers.

## Distributions and summary statistics

A random variable maps outcomes to numbers. A probability mass function applies to discrete outcomes; a density describes continuous outcomes, where probability is the area over an interval rather than the density at one exact point.

The expectation is a probability-weighted average:

```text
E[X] = Σx x P(X=x)
Var(X) = E[(X − E[X])²] = E[X²] − E[X]²
Std(X) = sqrt(Var(X))
Cov(X,Y) = E[(X−E[X])(Y−E[Y])]
```

Variance is in squared units; standard deviation returns to the original units. Covariance's sign indicates whether variables tend to move together, but its magnitude depends on scale. Correlation normalizes covariance and still does not establish causation.

Common AI distributions include Bernoulli for a binary event, categorical for one of several classes or tokens, Gaussian for continuous noise or approximate uncertainty, and empirical distributions formed from observed counts. A vector of logits is not a probability distribution until normalized.

## Entropy, cross-entropy, and KL divergence

For a discrete distribution `p`:

```text
H(p)       = −Σx p(x) log p(x)
H(p, q)    = −Σx p(x) log q(x)
KL(p || q) =  Σx p(x) log(p(x)/q(x)) = H(p,q) − H(p)
```

Entropy measures uncertainty in `p`. Cross-entropy measures the coding or prediction cost when the true distribution is `p` but predictions use `q`. KL divergence measures their directed mismatch. KL is nonnegative under its standard conditions, asymmetric, and can be infinite if `q` assigns zero probability where `p` assigns positive probability.

With a one-hot correct token `y`, cross-entropy reduces to:

```text
−log q(y)
```

If the model assigns the correct token probability `0.8`, the loss is about `0.223` nats. At `0.1`, it is about `2.303`. Pass raw logits to a stable cross-entropy implementation; avoid calculating `log(softmax(logits))` naively.

## Likelihood, sampling, and calibration

Likelihood treats observed data as fixed and asks which parameter values make it probable. Maximum likelihood chooses parameters maximizing the product of example probabilities, equivalently minimizing summed negative log-likelihood. This does not mean the fitted parameters themselves have a probability unless a Bayesian model defines one.

Softmax converts logits `z` into categorical probabilities:

```text
softmax(z_i) = exp(z_i) / Σj exp(z_j)
```

Subtract the maximum logit before exponentiation for numerical stability. Sampling draws an outcome from the resulting distribution; greedy selection takes the maximum. Temperature changes distribution sharpness but does not calibrate truthfulness.

A classifier is calibrated when predictions assigned confidence near `0.8` are correct about 80% of the time over an appropriate evaluation population. One probability cannot validate its own calibration. Distribution shift can invalidate calibration measured on older data.

## Production checks

- Report sample count and uncertainty, not only a point estimate.
- Split correlated examples by user, document, or time when independent generalization matters.
- Treat repeated trials from one prompt or one user as dependent unless justified otherwise.
- Inspect class prevalence before interpreting accuracy.
- Separate model confidence, empirical correctness, and business risk.
- Use log probabilities for products of many small probabilities to avoid underflow.

## Checkpoint

1. A detector has 90% sensitivity and a 5% false-positive rate. The event prevalence is 1%. Why is a positive result not 90% certain?
2. What is the relationship among entropy, cross-entropy, and KL divergence?
3. Why is a token probability of 0.95 not the same as 95% confidence that a complete answer is true?
4. When should validation examples be grouped rather than randomly split row by row?

<details>
<summary>Show answers</summary>

1. Bayes' theorem includes the base rate. Out of 10,000 cases, about 90 of 100 true events test positive, while about 495 of 9,900 negatives do. The positive predictive value is about `90/(90+495) = 15.4%`.
2. `H(p,q) = H(p) + KL(p||q)`. Entropy is inherent uncertainty in `p`; cross-entropy includes both that uncertainty and the penalty for using `q`.
3. It is a conditional probability for one next token under the model and prompt. Truth of a multi-token claim is a different event, and model probabilities can be miscalibrated.
4. When rows share a source that could leak information—such as chunks from one document, records from one patient, or repeated events from one user. Group-level splitting better tests new-source generalization.

</details>

## Exercise — Analyze a small evaluation

A model makes 100 predictions: 70 correct among 80 majority-class examples and 5 correct among 20 minority-class examples. Calculate overall accuracy, per-class recall, balanced accuracy, and the standard error of the overall accuracy using the simple independent Bernoulli approximation. State two limitations of that standard error.

<details>
<summary>Show exercise solution</summary>

Overall accuracy is `75/100 = 75%`. Recalls are `70/80 = 87.5%` and `5/20 = 25%`. Balanced accuracy is `(87.5% + 25%)/2 = 56.25%`. The approximate standard error is `sqrt(0.75×0.25/100) ≈ 0.0433`, or 4.33 percentage points.

The approximation assumes independent, identically distributed Bernoulli outcomes and ignores uncertainty introduced by choosing or tuning the model on the same evaluation. Correlated examples, distribution shift, and class-conditional concerns can make it misleading.

</details>

## Completion criteria

Explain conditional probability and Bayes' theorem, compute expectation/variance, connect cross-entropy to correct-token probability, and distinguish confidence from calibration and correctness.

## Primary references

- [NIST/SEMATECH e-Handbook of Statistical Methods](https://www.itl.nist.gov/div898/handbook/)
- [PyTorch distributions](https://docs.pytorch.org/docs/stable/distributions.html)
- [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
