# 0.7 Training Fundamentals — SKIM

0.5 gave you one gradient step. This lesson is about everything that turns a sequence of those steps into a training run that converges, generalizes, and can be reproduced.

| | |
|---|---|
| **Mode** | SKIM |
| **Time** | 60–90 minutes, including the exercises |
| **Assumes** | 0.5 Calculus, 0.6 Neural-Network Fundamentals |
| **Used by** | 0.8 PyTorch, the tiny text classifier project, Month 2 fine-tuning, Month 7 training infrastructure |

[Month 0 roadmap](../README.md) · [Previous: Neural-Network Fundamentals](06-neural-network-fundamentals.md) · [Next: PyTorch Fundamentals](08-pytorch-fundamentals.md)

## Learning objectives

After this lesson you can:

- explain SGD, momentum, Adam, and AdamW, and why AdamW is the transformer default
- explain what a learning-rate schedule and warmup do, and why decoupled weight decay matters
- define batch, epoch, and step, and convert between them
- recognize overfitting from a loss curve and name the standard responses
- save and restore a checkpoint correctly, and know what reproducibility does and does not guarantee
- use gradient accumulation and clipping, and state what each costs

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Section [0.7.7](#077--reading-a-loss-curve) is the practical payoff; it is how you debug every training run from here on.
3. Record gaps in [`progress.md`](../progress.md).

Examples assume:

```python
import math
import torch
import torch.nn.functional as F
from torch import nn
```

## 0.7.1 — Optimizers

Every optimizer answers the same question: given `.grad`, how should the parameter actually move?

**SGD** takes the plain step from 0.5:

```text
w ← w − η · g
```

**Momentum** accumulates an exponentially weighted average of past gradients, which damps oscillation across steep directions and accelerates along consistent ones:

```text
v ← βv + g          (β typically 0.9)
w ← w − η · v
```

**Adam** keeps two running averages per parameter: the mean of gradients `m` and the mean of squared gradients `v`. It divides the step by `√v`, giving each parameter its own effective learning rate. Parameters with consistently large gradients take smaller steps, and rarely updated ones take larger steps.

```text
m ← β₁m + (1−β₁)g                (β₁ = 0.9)
v ← β₂v + (1−β₂)g²               (β₂ = 0.999 or 0.95 for LLMs)
m̂, v̂ = bias-corrected m, v
w ← w − η · m̂ / (√v̂ + ε)
```

Bias correction matters because `m` and `v` start at zero and would otherwise be biased toward zero for the first steps.

**AdamW** changes one thing: it applies weight decay directly to the weights instead of adding it to the gradient.

```text
Adam + L2:   g ← g + λw,  then the adaptive step       (decay gets scaled by 1/√v̂)
AdamW:       w ← w − η·(m̂/(√v̂ + ε) + λw)              (decay is independent of gradient history)
```

Inside Adam, adding an L2 term to the gradient means the decay is divided by `√v̂` along with everything else, so parameters with large gradient history are barely regularized. Decoupling restores the intended behavior, and this is why **AdamW is the default for transformers**.

The cost is memory: Adam and AdamW store `m` and `v` for every parameter, so optimizer state is roughly **two extra copies of the model** in `float32`. For a 7-billion-parameter model, that is the difference between a job that fits and one that does not; Month 7 covers ZeRO and 8-bit optimizers for exactly this reason.

```python
sizes = {"SGD": 0, "SGD+momentum": 1, "Adam/AdamW": 2}
params = 7_000_000_000
for name, extra in sizes.items():
    print(f"{name:14s} optimizer state: {extra * params * 4 / 1e9:.0f} GB in fp32")
```

## 0.7.2 — Learning-rate schedules and warmup

A fixed learning rate is rarely best. A **schedule** changes it over training: large early steps to make progress, small late steps to settle.

- **Warmup** ramps the learning rate from near zero over the first few hundred or thousand steps. Adam's `v` estimate is unreliable at the start, so full-size early steps can destabilize or diverge a run. Warmup is close to mandatory for transformers.
- **Cosine decay** smoothly anneals from the peak to near zero over the run, and is the common default for LLM pretraining.
- **Step decay** cuts the rate by a factor at fixed milestones, and is common in vision.
- **Linear decay** is a simple, strong choice for fine-tuning.

```python
def lr_at(step: int, peak: float = 3e-4, warmup: int = 100, total: int = 1000) -> float:
    if step < warmup:
        return peak * step / warmup                                   # linear warmup
    progress = (step - warmup) / (total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))            # cosine decay

for step in [0, 50, 100, 300, 600, 1000]:
    print(f"step {step:4d}: lr {lr_at(step):.2e}")
```

In PyTorch these come from `torch.optim.lr_scheduler`, and `scheduler.step()` is called once per optimizer step (or once per epoch, depending on the scheduler).

The learning rate is the hyperparameter to tune first. If you tune nothing else, tune this.

## 0.7.3 — Weight decay and gradient clipping

**Weight decay** shrinks weights toward zero on every step, penalizing large weights and improving generalization. A typical transformer value is 0.01 to 0.1. By convention, decay is *not* applied to biases or normalization parameters, since shrinking those distorts the model without regularizing capacity:

```python
model = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 2))

decay, no_decay = [], []
for name, parameter in model.named_parameters():
    (no_decay if parameter.ndim < 2 else decay).append(parameter)     # biases and norms are 1D

optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4,
)
print(len(decay), len(no_decay))                                      # 2 weight matrices, 4 1D tensors
```

**Gradient clipping** caps the global gradient norm before the optimizer step, which prevents one bad batch from destroying the run:

```python
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

`max_norm=1.0` is the standard transformer value. Clipping happens **after** `backward()` and **before** `step()`. It treats a symptom rather than a cause, so a run that clips constantly usually has a learning rate that is too high or bad data.

## 0.7.4 — Batches, epochs, and steps

- A **batch** is the group of examples in one forward and backward pass.
- A **step** (iteration) is one optimizer update, one per batch.
- An **epoch** is one full pass over the training set.

```text
steps per epoch = ceil(dataset size / batch size)
total steps     = steps per epoch × epochs
```

Batch size trades off gradient noise against hardware efficiency. Larger batches give lower-variance gradients and better GPU utilization, but generalize slightly worse at a fixed learning rate and cost memory. A common heuristic when you change batch size is to scale the learning rate with it, linearly or by the square root, then re-tune.

LLM pretraining is usually measured in **tokens** and steps rather than epochs, because a corpus is typically seen roughly once.

## 0.7.5 — Gradient accumulation

To get the statistics of a large batch on hardware that cannot hold one, run several smaller batches and update once:

```python
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
accumulation_steps = 4

optimizer.zero_grad()
for micro_step in range(accumulation_steps):
    x = torch.randn(8, 8)
    targets = torch.randint(0, 2, (8,))
    loss = F.cross_entropy(model(x), targets) / accumulation_steps    # scale before backward
    loss.backward()                                                   # gradients accumulate

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()                                                      # one update from 32 examples
```

Dividing the loss by `accumulation_steps` is essential; without it the accumulated gradient is the *sum* rather than the mean, effectively multiplying your learning rate by four. This is the intentional twin of the `zero_grad()` bug from 0.5: the same accumulation, used deliberately. Note that accumulation buys batch statistics, not speed; it costs the same compute as the micro-batches it replaces.

## 0.7.6 — Validation, overfitting, and early stopping

Split data into **train**, **validation**, and **test**. Train fits parameters; validation guides decisions such as hyperparameters, checkpoints, and stopping; test is touched once, at the end. Tuning against the test set silently converts it into a second validation set and inflates your reported numbers.

**Overfitting** is the point where the model memorizes training data rather than learning generalizable patterns: training loss keeps falling while validation loss turns upward. Standard responses: more data, augmentation, weight decay, dropout, a smaller model, and early stopping.

**Early stopping** ends training when validation loss has not improved for `patience` evaluations, and restores the best checkpoint.

> **Pitfall:** Data leakage produces validation numbers that look excellent and collapse in production. Common causes are duplicate or near-duplicate examples across splits, splitting randomly when the data is a time series, fitting a scaler or vocabulary on the full dataset before splitting, and grouped data such as several rows per user landing on both sides of the split.

## 0.7.7 — Reading a loss curve

| Pattern | Likely cause | Try |
|---|---|---|
| Loss is flat at `log C` from step 0 | no gradient signal: frozen parameters, `zero_grad()` after `backward()`, detached tensor, or learning rate near 0 | check `p.grad` is non-`None` and nonzero; check the step order |
| Loss diverges or becomes `NaN` | learning rate too high, no warmup, `inf` in the data, fp16 overflow | lower the rate, add warmup, add clipping, check inputs for `NaN` |
| Loss drops then plateaus high | learning rate too low or model too small | raise the rate, add capacity, check the schedule |
| Loss spikes periodically | a bad data shard or duplicate batch | inspect the batches at the spike; add clipping |
| Train falls, validation rises | overfitting | regularize, get more data, stop early |
| Train and validation both stay high | underfitting or a label bug | verify labels and the pipeline on a tiny subset first |

**The single best debugging move is to overfit a tiny batch.** Take 8 examples, disable regularization, and train until the loss reaches nearly zero. If a model cannot memorize 8 examples, the bug is in the code, not the hyperparameters. Do this before any full run.

```python
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 4))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

x, targets = torch.randn(8, 16), torch.randint(0, 4, (8,))
losses = []
for step in range(200):
    loss = F.cross_entropy(model(x), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(round(losses[0], 3), round(losses[-1], 6))   # 1.493 -> 0.000049: the pipeline works
```

The first value sits near `log 4 ≈ 1.386`, the expected untrained loss from 0.4.5, and the last is essentially zero. Both facts together say the pipeline is wired correctly.

## 0.7.8 — Checkpoints and reproducibility

A checkpoint must contain enough to *resume*, not just to run inference:

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),      # momentum and variance estimates
    "step": 1000,
    "config": {"lr": 3e-4, "batch_size": 32},
}
torch.save(checkpoint, "/tmp/checkpoint.pt")

restored = torch.load("/tmp/checkpoint.pt", weights_only=False)   # trusted, self-created file
model.load_state_dict(restored["model"])
optimizer.load_state_dict(restored["optimizer"])
print(restored["step"], restored["config"])
```

Omitting optimizer state is a classic mistake: the model resumes with Adam's moments reset to zero, which produces a visible loss spike. The scheduler state and data-loader position matter too. Keep `weights_only=True` (the default since PyTorch 2.6) for any checkpoint you did not create.

**Reproducibility** means seeding Python, NumPy, and PyTorch, and recording versions:

```python
import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(0)
first = torch.randn(3)
set_seed(0)
assert torch.equal(first, torch.randn(3))
```

Seeding is necessary but not sufficient. Bit-exact reproduction across GPU runs also needs `torch.use_deterministic_algorithms(True)`, a fixed `cudnn.benchmark`, matching library and driver versions, and identical hardware, and it usually costs speed. Aim for *statistical* reproducibility by default, and report results across several seeds rather than the best one.

## Exercises

### Exercise 1 — Optimizers on the same problem

Minimize `(w − 3)²` from `w = 5` using SGD, SGD with momentum, and AdamW at the same learning rate. Print `w` every 10 steps for 100 steps and describe how the trajectories differ.

### Exercise 2 — Overfit a tiny batch

Train on 8 random examples until the loss is essentially zero, as in 0.7.7. Then break the pipeline deliberately: call `zero_grad()` *after* `backward()`, and separately set the learning rate to 10. Record the first loss, the maximum loss, and the mean of the last 50 steps in each case.

### Exercise 3 — Gradient accumulation equivalence

Show that four micro-batches of 8 with loss scaling produce the same gradient as one batch of 32. Then omit the division by `accumulation_steps` and measure how much larger the gradient norm becomes.

### Exercise 4 — Schedule shapes

Implement warmup-plus-cosine and warmup-plus-linear schedules, print the learning rate at 10 points across 1,000 steps for each, and explain what warmup protects against.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
def run(make_optimizer, steps: int = 100) -> list[float]:
    w = torch.tensor([5.0], requires_grad=True)
    optimizer = make_optimizer([w])
    history = []
    for step in range(steps):
        loss = ((w - 3) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            history.append(round(w.item(), 4))
    return history

print("SGD        ", run(lambda p: torch.optim.SGD(p, lr=0.1)))
print("SGD+mom    ", run(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)))
print("AdamW      ", run(lambda p: torch.optim.AdamW(p, lr=0.1)))
```

SGD converges smoothly and monotonically. Momentum overshoots past 3, then oscillates back with decaying amplitude; the accumulated velocity carries it through the minimum. AdamW moves in near-constant-size steps at first, because dividing by `√v̂` normalizes the step magnitude, so its early progress barely depends on how steep the gradient is.

### Exercise 2

```python
def train(steps: int = 200, lr: float = 1e-2, zero_after_backward: bool = False) -> list[float]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    x, targets = torch.randn(8, 16), torch.randint(0, 4, (8,))

    losses = []
    for _ in range(steps):
        loss = F.cross_entropy(model(x), targets)
        if not zero_after_backward:
            optimizer.zero_grad()
        loss.backward()
        if zero_after_backward:
            optimizer.zero_grad()          # bug: wipes gradients before the step
        optimizer.step()
        losses.append(loss.item())
    return losses

for label, kwargs in [("healthy", {}), ("zero after", {"zero_after_backward": True}), ("lr=10", {"lr": 10.0})]:
    losses = train(**kwargs)
    tail = sum(losses[-50:]) / 50
    print(f"{label:11s} first {losses[0]:.3f}  max {max(losses):10.3f}  mean of last 50 {tail:.4f}")
```

```text
healthy     first 1.493  max      1.493  mean of last 50 0.0001
zero after  first 1.493  max      1.493  mean of last 50 1.4926
lr=10       first 1.493  max   3134.508  mean of last 50 8.0793
```

The healthy run falls to essentially zero and stays there. Zeroing after `backward()` wipes every gradient before the step, so the loss never moves from `log 4 ≈ 1.386`; that is the "flat from step 0" row of the table. The `lr=10` run is the more instructive failure: its *final* loss happens to look small, but the maximum is over 3,000 and the tail average is 8.08, so it is bouncing violently rather than converging. Judging a run by its last loss alone would have hidden that completely, which is why you read the curve, not the endpoint.

### Exercise 3

```python
torch.manual_seed(0)
x, targets = torch.randn(32, 8), torch.randint(0, 2, (32,))

def gradient_norm(accumulation_steps: int, scale_loss: bool = True) -> float:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 2))
    model.zero_grad()
    chunk = 32 // accumulation_steps
    for i in range(accumulation_steps):
        xb = x[i * chunk:(i + 1) * chunk]
        tb = targets[i * chunk:(i + 1) * chunk]
        loss = F.cross_entropy(model(xb), tb)
        (loss / accumulation_steps if scale_loss else loss).backward()
    return torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()

full = gradient_norm(1)
accumulated = gradient_norm(4)
unscaled = gradient_norm(4, scale_loss=False)

print(f"one batch of 32     : {full:.6f}")
print(f"4 micro-batches     : {accumulated:.6f}")
print(f"4 without scaling   : {unscaled:.6f}  ({unscaled / full:.1f}x too large)")
assert abs(full - accumulated) < 1e-5
```

With scaling, accumulation reproduces the full-batch gradient exactly. Without it, the gradient is four times too large, which silently quadruples the effective learning rate.

### Exercise 4

```python
def cosine(step, peak=3e-4, warmup=100, total=1000):
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))

def linear(step, peak=3e-4, warmup=100, total=1000):
    if step < warmup:
        return peak * step / warmup
    return peak * max(0.0, 1 - (step - warmup) / (total - warmup))

for step in range(0, 1001, 100):
    print(f"step {step:4d}  cosine {cosine(step):.2e}  linear {linear(step):.2e}")
```

Both ramp to the peak at step 100. Cosine holds a high rate longer and then decays quickly near the end; linear falls at a constant rate. Warmup protects the first steps, when Adam's second-moment estimate `v` is still based on almost no data and is therefore unreliable; a full-size step then can push the model into a region it never recovers from.

</details>

## Exit test

1. Write the SGD update rule and explain what momentum adds.
2. What two running statistics does Adam maintain, and what does dividing by `√v̂` accomplish?
3. What exactly does AdamW change relative to Adam with L2 regularization, and why does it matter?
4. How much extra memory does AdamW use relative to the model?
5. What is warmup, and why do transformers need it?
6. Which parameters should be excluded from weight decay, and why?
7. Where in the training step does gradient clipping go?
8. Define batch, step, and epoch, and compute the steps in 3 epochs over 50,000 examples with batch size 32.
9. Why must the loss be divided by the accumulation count?
10. What does overfitting look like on a loss curve, and what are three responses?
11. What must a checkpoint contain to resume training, and what breaks if you save only the weights?
12. What is the fastest way to tell whether a training bug is in your code rather than your hyperparameters?

<details>
<summary>Show answers</summary>

1. `w ← w − η·g`. Momentum accumulates an exponentially weighted average of past gradients, `v ← βv + g`, and steps along `v`, which damps oscillation and accelerates consistent directions.
2. An exponentially weighted mean of gradients `m` and of squared gradients `v`. Dividing by `√v̂` gives each parameter its own effective step size, normalizing for how large its gradients typically are.
3. Adam with L2 adds `λw` to the gradient, so the decay is then divided by `√v̂` along with the gradient. AdamW subtracts `ηλw` from the weight directly, decoupling regularization from gradient history and making it behave as intended.
4. Two extra copies of the parameters, for `m` and `v`, so about 8 bytes per parameter in fp32.
5. A linear ramp of the learning rate over the first steps. Adam's variance estimate is unreliable at the start, and large early steps can destabilize or diverge a transformer run.
6. Biases and normalization parameters, conventionally identified as the 1D tensors. Decaying them distorts the model without regularizing capacity.
7. After `backward()` and before `optimizer.step()`.
8. A batch is the examples in one forward/backward pass; a step is one optimizer update; an epoch is one pass over the data. `ceil(50000/32) = 1563` steps per epoch, so 4,689 steps.
9. Because `.grad` accumulates a sum. Without dividing, the gradient is `N` times the mean, which multiplies the effective learning rate by `N`.
10. Training loss keeps falling while validation loss turns upward. Respond with more data or augmentation, stronger regularization such as weight decay or dropout, a smaller model, or early stopping.
11. Model weights, optimizer state, scheduler state, step or epoch count, and the run config, ideally with the data-loader position and RNG state. With weights only, Adam's moments reset to zero on resume and the loss spikes.
12. Overfit a tiny batch of about 8 examples with regularization off. If the loss does not go to nearly zero, the bug is in the code.

</details>

## Completion criteria

You are done when:

- you can explain AdamW's decoupled decay and why transformers use it
- you can convert between batches, steps, and epochs without hesitation
- you can diagnose a flat, diverging, or overfitting loss curve from its shape
- you can write a correct accumulation loop, including the loss scaling
- you can save and restore a checkpoint that resumes training cleanly
- your four exercises run, including the two deliberately broken runs in Exercise 2

## References

**Optimizers**
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — the AdamW paper
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/)
- [PyTorch: `torch.optim`](https://docs.pytorch.org/docs/stable/optim.html)

**Schedules and warmup**
- [PyTorch: learning-rate schedulers](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), section 5.3 — the original warmup schedule

**Batch size and scaling**
- [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677) — linear scaling and warmup
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)

**Debugging and practice**
- [A Recipe for Training Neural Networks, Andrej Karpathy](https://karpathy.github.io/2019/04/25/recipe/) — read this one in full
- [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)

**Checkpoints and reproducibility**
- [PyTorch: saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch: reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch: automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)

## Videos and code to read

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — `train.py` is a ~300-line training loop with warmup, cosine decay, clipping, accumulation, and checkpointing all visible at once. Note the repo is now archived in favor of [karpathy/nanochat](https://github.com/karpathy/nanochat), which is worth reading for a current end-to-end pipeline
- [google-research/tuning_playbook](https://github.com/google-research/tuning_playbook) — what to tune, in what order, and how to tell whether a change helped
- [StatQuest: gradient descent and SGD](https://www.youtube.com/@statquest) — if the optimizer intuition is thin

## Mapped companion lessons

- [Optimizers](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/03-deep-learning-core/06-optimizers) and [Regularization](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/03-deep-learning-core/07-regularization) map to update rules, weight decay, and generalization controls.
- [Weight Initialization and Training Stability](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/03-deep-learning-core/08-weight-initialization) maps to stable signal flow.
- [Learning Rate Schedules and Warmup](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/03-deep-learning-core/09-learning-rate-schedules) maps directly to the scheduling section.
- [Bias-Variance Tradeoff](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/02-ml-fundamentals/10-bias-variance) and [Model Evaluation](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/02-ml-fundamentals/09-model-evaluation) map to overfitting diagnosis and validation (0.7.6–0.7.7).
- [Hyperparameter Tuning](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/02-ml-fundamentals/12-hyperparameter-tuning) maps to searching the settings this lesson tells you to tune first.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).

## About this lesson

Written to cover section 0.7 of the [Month 0 curriculum](../README.md). Code examples were checked with PyTorch 2.14 on CPU.
