# 0.5 Calculus for Neural Networks — SKIM

You will almost never differentiate by hand again; autograd does it. What you do need is the ability to say what a gradient *is*, why the chain rule makes deep networks trainable, and what is actually happening when a loss stops decreasing.

| | |
|---|---|
| **Mode** | SKIM |
| **Time** | 45–75 minutes, including the exercises |
| **Assumes** | 0.3 Linear Algebra, 0.4 Probability (for the loss) |
| **Used by** | 0.6 Neural-Network Fundamentals, 0.7 Training, 0.8 PyTorch autograd, Month 2 fine-tuning |

[Month 0 roadmap](../README.md) · [Previous: Probability and Statistics](04-probability-and-statistics.md) · [Next: Neural-Network Fundamentals](06-neural-network-fundamentals.md)

## Learning objectives

After this lesson you can:

- explain a derivative, a partial derivative, and a gradient in one sentence each
- apply the chain rule to a composed function and explain why backpropagation is exactly this
- read a computational graph and say which nodes need gradients
- trace loss → gradients → parameter update in order, and name what each step touches
- explain vanishing and exploding gradients and the architectural fixes for them

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Section [0.5.6](#056--the-full-trace-loss--gradients--parameters--optimizer-update) is the one to know cold; it is the backbone of 0.7 and 0.8.
3. Record gaps in [`progress.md`](../progress.md).

Examples assume:

```python
import torch
import torch.nn.functional as F
from torch import nn
```

## 0.5.1 — Derivatives and partial derivatives

A **derivative** `df/dx` is the rate of change of `f` with respect to `x`: if you nudge `x` by a tiny amount, how much does `f` move, and in which direction? Geometrically it is the slope of the tangent line.

The handful of rules worth recognizing:

```text
d/dx (xⁿ)      = n·xⁿ⁻¹
d/dx (eˣ)      = eˣ
d/dx (log x)   = 1/x
d/dx (f + g)   = f′ + g′
d/dx (f·g)     = f′g + fg′
```

A **partial derivative** `∂f/∂x` is the same idea for a function of several variables, holding the others fixed. A neural network's loss is a function of millions of parameters, so every parameter has a partial derivative.

## 0.5.2 — The gradient

The **gradient** `∇f` collects all the partial derivatives into a vector:

```text
∇f = [∂f/∂w₁, ∂f/∂w₂, …, ∂f/∂wₙ]
```

Three properties matter:

- It points in the direction of steepest *increase*. That is why gradient descent steps in the direction `−∇f`.
- Its magnitude says how steep the surface is. A near-zero gradient means a flat region, where learning stalls.
- **It has the same shape as the thing it differentiates.** The gradient of a scalar loss with respect to a `[768, 256]` weight matrix is itself `[768, 256]`, one number per parameter, each answering "if I nudge this single weight, how does the loss change?"

```python
w = torch.randn(3, requires_grad=True)
loss = (w ** 2).sum()          # ∂loss/∂wᵢ = 2wᵢ
loss.backward()

assert torch.allclose(w.grad, 2 * w)
print(w.grad.shape)            # torch.Size([3]): same shape as w
```

The loss must be a **scalar** for `backward()` to work without arguments, which is why training loops reduce per-example losses to a mean.

## 0.5.3 — The chain rule

The chain rule differentiates composed functions:

```text
If y = f(u) and u = g(x):   dy/dx = (dy/du) · (du/dx)
```

A neural network *is* a composition: input → layer 1 → activation → layer 2 → … → loss. The chain rule says the gradient at any early layer is the product of the local derivatives along the path to the loss.

Worked example, `L = (3x + 1)²` at `x = 2`:

```text
u = 3x + 1 = 7
L = u²     = 49

dL/du = 2u = 14
du/dx = 3
dL/dx = 14 · 3 = 42
```

```python
x = torch.tensor(2.0, requires_grad=True)
u = 3 * x + 1
loss = u ** 2
loss.backward()
print(x.grad)                  # tensor(42.)
```

**Backpropagation is the chain rule applied in reverse across the whole network**, reusing shared intermediate results instead of recomputing them for each parameter. That reuse is why one backward pass costs roughly the same as one or two forward passes, no matter how many millions of parameters there are.

The multiplicative structure also explains a failure mode. Multiply many factors below 1 and the gradient **vanishes** toward zero, so early layers stop learning. Multiply many factors above 1 and it **explodes**, producing `NaN` losses. Residual connections, careful normalization, and gradient clipping in 0.6 and 0.7 exist to keep that product near 1.

## 0.5.4 — Computational graphs

A framework records each operation as a node in a directed graph, with tensors as edges. The forward pass computes values and saves what the backward pass will need; `backward()` then walks the graph in reverse, applying each node's local derivative rule.

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
w = torch.tensor([3.0, 4.0], requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

y = (x * w).sum() + b          # graph: mul → sum → add
y.backward()

print(x.grad, w.grad, b.grad)  # tensor([3., 4.]) tensor([1., 2.]) tensor(1.)
```

The gradient with respect to `x` is `w`, and with respect to `w` is `x`; that symmetry of multiplication is the whole backward rule for that node.

Points that carry directly into PyTorch (0.8):

- **`requires_grad`** marks which tensors need gradients. Model parameters have it by default; input data normally does not.
- **The graph is built on every forward pass** and freed by `backward()`. This is what "define-by-run" means, and it is why a Python `if` inside `forward` is allowed.
- **`torch.no_grad()` and `torch.inference_mode()`** skip graph construction entirely, saving memory and time during evaluation.
- **`.detach()`** cuts a tensor out of the graph, which is how you log a value or stop a gradient from flowing into a frozen branch.
- **Activation memory** is the hidden cost: the saved intermediates are usually what exhausts GPU memory during training, not the parameters. Gradient checkpointing trades recomputation for that memory.

## 0.5.5 — Gradient descent

Gradient descent repeatedly steps downhill:

```text
w ← w − η · ∇L(w)
```

`η` is the **learning rate**. Too small and training crawls; too large and the loss oscillates or diverges. It is the single most important hyperparameter in 0.7.

| Variant | Gradient computed on | Character |
|---|---|---|
| Batch | the entire dataset | stable, and infeasible at scale |
| Stochastic (SGD) | one example | very noisy |
| Mini-batch | a batch of 8 to 1024 | the practical default |

Mini-batch gradients are noisy estimates of the full gradient (0.4.2), and that noise is not purely harmful: it helps escape sharp regions of the loss surface.

Real loss surfaces in high dimensions are non-convex, but exact local minima are rarely the problem. Saddle points and long flat plateaus are more common, which is why momentum and adaptive methods (0.7) matter.

```python
w = torch.tensor(5.0, requires_grad=True)   # minimize (w - 3)^2; the answer is w = 3
learning_rate = 0.1

for step in range(25):
    loss = (w - 3) ** 2
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
    w.grad.zero_()                           # gradients accumulate; clear them each step

print(round(w.item(), 4))                    # 3.0076: converging on the minimum
```

## 0.5.6 — The full trace: loss → gradients → parameters → optimizer update

Memorize this order. Every training loop in the rest of the roadmap is this loop.

1. **Forward.** Inputs flow through the model to produce logits, and the graph is recorded.
2. **Loss.** Logits and targets are reduced to one scalar (0.4.5).
3. **Zero gradients.** `optimizer.zero_grad()`, because PyTorch *accumulates* into `.grad` rather than overwriting.
4. **Backward.** `loss.backward()` walks the graph in reverse, filling `.grad` for every parameter with `requires_grad=True`.
5. **Optimizer step.** `optimizer.step()` updates each parameter using its gradient and the optimizer's own state, such as momentum.
6. **Repeat** with the next batch.

```python
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

x = torch.randn(16, 4)
targets = torch.randint(0, 2, (16,))

first_loss = None
for step in range(200):
    logits = model(x)                        # 1. forward
    loss = F.cross_entropy(logits, targets)  # 2. loss
    optimizer.zero_grad()                    # 3. clear old gradients
    loss.backward()                          # 4. backward
    optimizer.step()                         # 5. update
    first_loss = first_loss or loss.item()

print(round(first_loss, 4), round(loss.item(), 4))   # 0.7634 -> 0.0143
```

The starting loss near `log 2 ≈ 0.69` is the sanity check from 0.4.5.

> **Pitfall:** Forgetting `zero_grad()` silently sums gradients across steps, so the model takes increasingly large and wrong steps. The loss often still moves, which is what makes it hard to spot. Deliberate gradient *accumulation* (0.7) is the same mechanism used on purpose: skip `zero_grad()` and `step()` for several batches to simulate a larger batch.

## Exercises

### Exercise 1 — Verify autograd against hand-computed derivatives

For `f(x) = x³ + 2x` at `x = 4`, compute `df/dx` by hand, then confirm it with `backward()`. Do the same for `L = (3x + 1)²` at `x = 2` and for `L = Σ wᵢ²`.

### Exercise 2 — Gradient descent by hand

Minimize `(w - 3)²` from `w = 5` using a manual loop. Run it at learning rates 0.01, 0.1, 0.5, and 1.1, and describe what each does over 25 steps.

### Exercise 3 — Watch the chain rule vanish and explode

Build a stack of `depth` linear layers with no activations, normalizations, or residual connections, and initialize every weight from a normal distribution with standard deviation `scale`. For a width of 64, each layer multiplies the signal by roughly `scale · √64`, so `scale = 0.125` gives a per-layer gain near 1.

Print the output activation standard deviation and the first layer's gradient norm for depths 5, 10, 20, and 30 at scales 0.10, 0.125, and 0.15. Explain the three regimes.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
x = torch.tensor(4.0, requires_grad=True)
f = x ** 3 + 2 * x
f.backward()
assert torch.allclose(x.grad, torch.tensor(3 * 4.0 ** 2 + 2))    # 50
print(x.grad.item())
```

### Exercise 2

```python
for learning_rate in [0.01, 0.1, 0.5, 1.1]:
    w = torch.tensor(5.0, requires_grad=True)
    for _ in range(25):
        loss = (w - 3) ** 2
        loss.backward()
        with torch.no_grad():
            w -= learning_rate * w.grad
        w.grad.zero_()
    print(f"lr={learning_rate:4.2f} -> w={w.item():.4f}")
```

`0.01` is still far from 3 after 25 steps; `0.1` converges cleanly; `0.5` overshoots past the minimum each step but still converges; `1.1` diverges, with `w` growing without bound. The gradient of this function is `2(w − 3)`, so any learning rate at or above 1 overshoots by more than the remaining distance.

### Exercise 3

```python
def probe(scale: float, depth: int, width: int = 64) -> tuple[float, float]:
    torch.manual_seed(0)
    layers = [nn.Linear(width, width, bias=False) for _ in range(depth)]
    for layer in layers:
        nn.init.normal_(layer.weight, std=scale)

    output = nn.Sequential(*layers)(torch.randn(8, width))
    output.pow(2).mean().backward()
    return output.std().item(), layers[0].weight.grad.norm().item()

for scale in [0.10, 0.125, 0.15]:
    print(f"scale={scale}  per-layer gain about {scale * 8:.1f}")
    for depth in [5, 10, 20, 30]:
        activation_std, grad_norm = probe(scale, depth)
        print(f"   depth {depth:2d}: activation std {activation_std:.2e}   layer-1 grad norm {grad_norm:.2e}")
```

```text
scale=0.1    per-layer gain about 0.8
   depth  5: activation std 3.24e-01   layer-1 grad norm 2.25e-01
   depth 30: activation std 8.16e-04   layer-1 grad norm 4.05e-06
scale=0.125  per-layer gain about 1.0
   depth  5: activation std 9.90e-01   layer-1 grad norm 1.67e+00
   depth 30: activation std 6.59e-01   layer-1 grad norm 2.11e+00
scale=0.15   per-layer gain about 1.2
   depth  5: activation std 2.46e+00   layer-1 grad norm 8.64e+00
   depth 30: activation std 1.56e+02   layer-1 grad norm 9.92e+04
```

A gain slightly below 1 vanishes: over 30 layers the gradient falls by five orders of magnitude, and the early layers stop learning. A gain slightly above 1 explodes by the same logic in reverse. Only a gain near 1 stays stable across depth, and the effect is exponential in depth, so the three regimes separate further the deeper the network gets.

This is why initialization schemes target a per-layer gain of 1 (0.6), why normalization layers re-center activations at every step, why residual connections give the gradient an additive path around the multiplication, and why gradient clipping (0.7) exists as a last line of defense.

</details>

## Exit test

1. What is a derivative, in one sentence?
2. What is the difference between a partial derivative and a gradient?
3. What shape does the gradient of a scalar loss with respect to a `[768, 256]` weight have?
4. Why must the loss be a scalar before calling `backward()`?
5. State the chain rule and apply it to `L = (3x + 1)²` at `x = 2`.
6. What is backpropagation in terms of the chain rule?
7. What is a computational graph, and when is it built and freed?
8. What do `requires_grad`, `no_grad()`, and `detach()` each control?
9. Write the gradient descent update rule and name each symbol.
10. What breaks if the learning rate is far too large, and far too small?
11. List the five steps of a training iteration in order.
12. Why does forgetting `zero_grad()` cause a subtle bug rather than an obvious one?

<details>
<summary>Show answers</summary>

1. The rate of change of a function's output with respect to a small change in its input.
2. A partial derivative is the rate of change with respect to one variable, holding the others fixed. The gradient is the vector of all partial derivatives.
3. `[768, 256]`, the same shape as the weight; one partial derivative per parameter.
4. Because `backward()` propagates a single starting value of `dL/dL = 1`. A non-scalar output has no single such value, so you would have to supply a vector explicitly.
5. `dy/dx = (dy/du)(du/dx)`. With `u = 3x + 1 = 7`: `dL/du = 2u = 14`, `du/dx = 3`, so `dL/dx = 42`.
6. It is the chain rule applied in reverse through the network's computational graph, reusing shared intermediate gradients instead of recomputing them for each parameter.
7. A directed record of operations built during the forward pass; `backward()` traverses it in reverse and then frees it. It is rebuilt on the next forward pass.
8. `requires_grad` marks a tensor as needing gradients. `no_grad()` disables graph construction for a block. `detach()` returns a tensor cut out of the graph so gradients do not flow through it.
9. `w ← w − η·∇L(w)`: `w` is the parameter, `η` the learning rate, and `∇L(w)` the gradient of the loss with respect to that parameter.
10. Too large and the update overshoots, causing oscillation, divergence, or `NaN`. Too small and training is slow and can stall on plateaus.
11. Forward pass, compute loss, `zero_grad()`, `backward()`, `optimizer.step()`.
12. Because PyTorch accumulates gradients rather than erasing them, so the model still trains, just with inflated, stale-contaminated gradients. The loss keeps moving, so nothing errors out.

</details>

## Completion criteria

You are done when:

- you can state the five training steps in order and explain what each one touches
- you can explain backpropagation as the chain rule in reverse
- you can predict the shape of any parameter's gradient
- you can explain vanishing and exploding gradients and name the architectural fixes
- your three exercises run, including the diverging learning rate in Exercise 2

## References

**Derivatives, partial derivatives, chain rule**
- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) — chapters 1–4
- [Khan Academy: Multivariable derivatives](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives)

**Gradients and backpropagation**
- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)
- [CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) — the clearest treatment of local gradients
- [Andrej Karpathy: The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) — builds an autograd engine from scratch

**Computational graphs and autograd**
- [PyTorch: Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [`torch.autograd` reference](https://docs.pytorch.org/docs/stable/autograd.html)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
- [PyTorch: A gentle introduction to `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

**Gradient descent**
- [Deep Learning Book](https://www.deeplearningbook.org/contents/numerical.html), chapter 4
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/)

## Videos and code to read

- [karpathy/micrograd](https://github.com/karpathy/micrograd) — a scalar autograd engine in about 100 readable lines; read it alongside [the lecture](https://www.youtube.com/watch?v=VMj-3S1tku0) and you will not be confused about backpropagation again
- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — the notebooks for that lecture, including the exercises
- [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/topics/neural-networks) — chapters 3 and 4 are gradient descent and backpropagation

- [ai-engineering-from-scratch: chain rule and autodiff](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/05-chain-rule-and-autodiff) — a second pass at this lesson that builds an autograd engine end to end

## About this lesson

Written to cover section 0.5 of the [Month 0 curriculum](../README.md). Code examples were checked with PyTorch 2.14 on CPU.
