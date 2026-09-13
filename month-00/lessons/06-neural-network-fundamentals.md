# 0.6 — Neural-Network Fundamentals

**Depth: SKIM**

**Goal:** connect layers, activations, losses, and backpropagation into one trainable system, with special attention to GELU, LayerNorm, residual connections, and dropout.

[Month 0 roadmap](../README.md) · [Previous: Calculus](05-calculus-for-neural-networks.md) · [Next: Training](07-training-fundamentals.md)

## From affine maps to networks

A linear layer computes an affine transformation:

```text
y = xW + b
[B, Din] @ [Din, Dout] + [Dout] → [B, Dout]
```

The bias makes it affine rather than strictly linear. Stacking affine maps without nonlinearities collapses to one affine map. An activation lets depth represent more complex functions.

ReLU uses `max(0,x)`. GELU smoothly scales values according to magnitude and is common in transformers. Sigmoid maps to `(0,1)` and is useful for independent binary probabilities; softmax normalizes mutually exclusive categorical logits. Activation choice belongs to the model design, while the loss must match the output/target contract.

## Forward pass, loss, and backward pass

```text
features → layers → logits → task loss
                              │
parameters ← optimizer ← gradients
```

Logits are raw scores. For multiclass classification, use categorical cross-entropy with one target class per example. For independent multilabel decisions, use a binary-logit loss per label. Applying softmax before a cross-entropy function that expects logits changes the objective and reduces numerical stability.

Backpropagation computes each parameter's contribution to the loss. The optimizer updates parameters. Evaluation must disable training-only stochastic behavior and gradient tracking when gradients are unnecessary.

## Initialization

If every neuron starts with identical weights, many receive identical gradients and remain redundant. Random initialization breaks symmetry. Its scale should preserve usable activation and gradient variance through depth.

Xavier/Glorot initialization is designed around fan-in and fan-out; Kaiming/He initialization targets rectifier-like activations. Exact choices depend on activation, architecture, residual scaling, and framework defaults. Initialization is an initial condition, not a substitute for normalization or a sound optimizer.

## Normalization

LayerNorm normalizes features within one example/token:

```text
normalized = (x − mean_features) / sqrt(var_features + epsilon)
output = scale ⊙ normalized + bias
```

It behaves the same with respect to batch statistics in training and evaluation, unlike BatchNorm. It stabilizes feature scale but does not guarantee equal gradient scale or eliminate every optimization problem.

Transformers commonly place LayerNorm before a sublayer (pre-norm) or after residual addition (post-norm). The choice changes gradient paths and must match checkpoint weights.

## Residual connections

A residual update is `y = x + F(x)`. The shapes must match, or a deliberate projection must reconcile them. The identity path lets a block learn an incremental change and provides a direct additive route for gradients.

Residuals do not mean all layers are optional: the learned updates accumulate through depth, and normalization placement changes what each sublayer sees.

## Dropout and train/evaluation mode

Dropout randomly zeros activations during training and rescales survivors so their expectation is preserved. During evaluation it becomes an identity operation. It regularizes co-adaptation but can slow optimization or harm small-data models if too strong.

```python
model.train()  # dropout active
model.eval()   # dropout inactive
```

`model.eval()` does not disable gradients. Use `torch.no_grad()` for ordinary validation. A reproducibility test should compare outputs in evaluation mode; repeated training-mode outputs can differ by design.

## Capacity, underfitting, and overfitting

Underfitting means the model cannot adequately fit training data under the current representation, capacity, or optimization. Overfitting means training performance improves while held-out performance degrades or fails to generalize. More parameters can increase capacity but do not determine either outcome alone.

Track training and validation curves, use data splits that reflect deployment, and compare against a simple baseline. A small neural network beating chance on leaked validation data proves little.

## Checkpoint

1. Why does a deep stack of linear layers without activations collapse to one layer?
2. What is the difference between logits and probabilities?
3. Why do residual additions require compatible shapes?
4. Why are `model.eval()` and `torch.no_grad()` both used in validation?

<details>
<summary>Show answers</summary>

1. Matrix multiplication and bias composition remain affine: `(xW1+b1)W2+b2 = x(W1W2)+(b1W2+b2)`.
2. Logits are unrestricted scores; probabilities are normalized under a defined mapping such as sigmoid or softmax.
3. Element-wise addition needs corresponding features. A projection can intentionally map one path to the required width.
4. Evaluation mode disables training behavior such as dropout; `no_grad` avoids recording a backward graph. Neither replaces the other.

</details>

## Exercise — Build and diagnose a residual MLP

Implement a block `x + Linear(GELU(Linear(LayerNorm(x))))` that maps width 16 back to width 16. Test output shape, deterministic evaluation, stochastic training with dropout, and finite gradients.

<details>
<summary>Show exercise solution</summary>

```python
import torch
from torch import nn

class ResidualMLP(nn.Module):
    def __init__(self, width=16, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * width, width),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))

torch.manual_seed(7)
model = ResidualMLP()
x = torch.randn(4, 16, requires_grad=True)
assert model(x).shape == x.shape
model.eval()
torch.testing.assert_close(model(x), model(x))
model.train()
loss = model(x).square().mean()
loss.backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
```

To test dropout variation, compare several training-mode forwards rather than assuming two random masks must differ. With dropout zero, training and evaluation are intentionally identical for this block.

</details>

## Completion criteria

Trace a forward/backward pass, select an output/loss contract, explain initialization, LayerNorm, residuals, and dropout, and diagnose train-versus-validation curves.

## Primary references

- [PyTorch neural-network modules](https://docs.pytorch.org/docs/stable/nn.html)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415)
- [Dropout](https://jmlr.org/papers/v15/srivastava14a.html)
