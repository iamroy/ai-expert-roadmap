# 0.5 — Calculus for Neural Networks

**Depth: SKIM**

**Goal:** trace how a scalar loss produces parameter gradients and recognize the practical failure modes of automatic differentiation.

[Month 0 roadmap](../README.md) · [Previous: Probability](04-probability-and-statistics.md) · [Next: Neural Networks](06-neural-network-fundamentals.md)

## Derivatives and partial derivatives

A derivative measures local sensitivity. If `y=f(x)`, then `dy/dx` approximates how much `y` changes for a small change in `x`. For a function of many inputs, a partial derivative varies one input while holding the others fixed.

For scalar loss `L(θ)` and parameter vector `θ`, the gradient collects all partial derivatives:

```text
∇θ L = [∂L/∂θ1, ..., ∂L/∂θn]
```

The gradient points in the direction of steepest local increase under the ordinary Euclidean metric. Gradient descent moves in the opposite direction:

```text
θ ← θ − learning_rate × ∇θ L
```

This is a local first-order step, not a promise of reaching a global optimum.

## Chain rule and computational graphs

Neural networks compose functions, so the chain rule is the central operation. If `L=f(g(x))`:

```text
dL/dx = dL/dg × dg/dx
```

For `y=wx+b` and `L=(y−t)²`:

```text
∂L/∂y = 2(y−t)
∂L/∂w = 2(y−t)x
∂L/∂b = 2(y−t)
∂L/∂x = 2(y−t)w
```

A computational graph records operations and dependencies during the forward pass. Reverse-mode automatic differentiation starts from a scalar output and propagates vector-Jacobian products backward. It is efficient when many parameters influence one scalar loss, which is the usual training setup.

```text
inputs + parameters → predictions → loss
                          ↑          │
parameter update ← gradients ← backward
```

Backpropagation calculates gradients. The optimizer decides how to use them. These are separate responsibilities.

## PyTorch autograd essentials

```python
import torch

x = torch.tensor([2.0])
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
target = torch.tensor([10.0])

prediction = w * x + b       # 7
loss = (prediction - target).pow(2).mean()  # 9
loss.backward()

assert w.grad.item() == -12.0
assert b.grad.item() == -6.0
```

Gradients accumulate into `.grad`. Clear them before the next independent update. Operations inside `torch.no_grad()` are not recorded for backward. `detach()` returns a tensor disconnected from the current graph; using it accidentally in a loss path silently prevents upstream gradients.

In-place mutation can invalidate values saved for backward. A graph is normally freed after `.backward()`; requesting repeated backward passes requires retaining or rebuilding it and can consume significant memory.

## Gradient pathologies

Repeated multiplication by derivatives smaller than one can produce vanishing gradients; derivatives larger than one can produce exploding gradients. Activations, initialization, residual connections, normalization, and architecture all affect these paths.

Gradient clipping limits update magnitude, often by global norm. It can prevent an occasional explosion from destabilizing training, but persistent clipping may conceal a poor learning rate, invalid data, or numerical overflow. Log gradient norms and nonfinite values.

Finite differences provide a slow independent check:

```text
df/dx ≈ [f(x+ε) − f(x−ε)] / (2ε)
```

Choose `ε` carefully: too large gives approximation error, too small suffers floating-point cancellation. Gradient checking is useful for small differentiable functions, not an efficient training method.

## Checkpoint

1. What is the difference between backpropagation and an optimizer step?
2. Why must gradients usually be cleared between batches?
3. What happens if an intermediate tensor is detached before computing the loss?
4. Why can a finite-difference check disagree when `ε` is extremely small?

<details>
<summary>Show answers</summary>

1. Backpropagation computes derivatives through the graph. The optimizer transforms those gradients into parameter updates, possibly using momentum or adaptive state.
2. PyTorch adds new gradients to existing `.grad` values. Without clearing, unrelated batches accumulate unintentionally.
3. Operations before the detach receive no gradient through that path because the graph connection is broken.
4. Subtraction of nearly equal floating-point values loses precision; rounding error can dominate the small signal.

</details>

## Exercise — Manual and automatic gradients

For `x=2`, `w=3`, `b=1`, and `t=10`, calculate `L=(wx+b−t)²` and gradients for `w`, `b`, and `x`. Verify them with autograd and a centered finite difference for `w`.

<details>
<summary>Show exercise solution</summary>

`y=7`, error is `−3`, and `L=9`. The gradients are `dL/dw=2(−3)(2)=−12`, `dL/db=−6`, and `dL/dx=2(−3)(3)=−18`.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
loss = (w * x + b - 10).pow(2)
loss.backward()
assert (w.grad.item(), b.grad.item(), x.grad.item()) == (-12.0, -6.0, -18.0)

def f(value):
    return (value * 2 + 1 - 10) ** 2

epsilon = 1e-4
estimate = (f(3 + epsilon) - f(3 - epsilon)) / (2 * epsilon)
assert abs(estimate + 12) < 1e-6
```

</details>

## Completion criteria

Apply the chain rule to a small graph, interpret a gradient, explain accumulation and detach, and trace `loss → gradients → optimizer update`.

## Primary references

- [PyTorch automatic differentiation](https://docs.pytorch.org/docs/stable/autograd.html)
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
