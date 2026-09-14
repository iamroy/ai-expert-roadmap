# 1.1 — Transformer Architecture

**Depth: MASTER**

**Goal:** derive and implement the path from token representations through one transformer block, then explain how stacking blocks produces a language model.

[Month 1 roadmap](../README.md) · [Next: Position Information](02-position-information.md)

## Lesson 1.1.1 — Three architecture families

A transformer repeatedly updates one vector per token. Attention mixes information across positions; a feed-forward network transforms the features at each position. Different visibility rules and connections produce different model families.

| Family | What can a token attend to? | Typical use |
|---|---|---|
| Encoder-only | All non-padding input positions | Representations, classification, masked-token prediction |
| Decoder-only | Its own position and preceding positions | Autoregressive text generation |
| Encoder-decoder | Bidirectional source encoder; causal target decoder plus cross-attention to source | Translation and other sequence-to-sequence tasks |

In cross-attention, queries come from the decoder while keys and values come from the encoder. Source and target lengths need not match. A decoder-only model can still translate: it conditions on a serialized source and task description, then continues with the target. Architecture constrains information flow; it does not assign exactly one task to a model.

The original [Transformer paper](https://arxiv.org/abs/1706.03762) introduced an encoder-decoder design. The project's small GPT uses a decoder-only stack, which makes the training-to-generation connection easy to inspect.

## Lesson 1.1.2 — Q, K, and V are learned projections

Let `X` have shape `[B, T, D]`: batch, sequence length, model width. With one head, project it into queries, keys, and values:

```text
Q = X Wq   [B, T, Dh]
K = X Wk   [B, T, Dh]
V = X Wv   [B, T, Dv]
```

A query represents the information a position seeks, a key represents what a position can match, and a value is the information it can contribute. These are useful intuitions, not hand-assigned linguistic meanings. Their projections are learned through the training objective.

For query position `i` and key position `j`, the dot product `qi · kj` is a compatibility score. Softmax turns that row of scores into nonnegative weights summing to one. Multiplying by `V` creates a weighted mixture of values:

```text
scores  = Q Kᵀ / sqrt(Dh)       [B, T, T]
weights = softmax(scores, -1)   [B, T, T]
output  = weights V            [B, T, Dv]
```

The softmax axis is **keys for each query**. Normalizing over queries instead silently changes the operation while keeping plausible tensor shapes.

Why divide by `sqrt(Dh)`? If query/key components are independent with mean zero and variance one, a dot product's variance grows with `Dh`. The division keeps that scale approximately stable, reducing premature softmax saturation. Learned components do not strictly satisfy those assumptions, but the calculation explains the design.

## Lesson 1.1.3 — A small numeric walkthrough

Suppose one query has scores `[0, log(2), 0]`. Softmax gives weights `[0.25, 0.5, 0.25]`. For values `[[2, 0], [0, 4], [2, 2]]`, its output is:

```text
0.25 × [2, 0] + 0.5 × [0, 4] + 0.25 × [2, 2] = [1, 2.5]
```

The output mixes **values**, not token IDs or probability labels. Multiple positions can contribute. Attention weights are internal routing coefficients and are not, by themselves, a faithful explanation of the final prediction: later projections, residual paths, and other layers also matter.

## Lesson 1.1.4 — Multi-head attention and shapes

For `H` heads with `Dh = D/H`, separate subspaces can learn different matching patterns. A common implementation projects to `3D`, splits Q/K/V, and reshapes each into heads:

```text
X                      [B, T, D]
combined QKV           [B, T, 3D]
Q, K, V                [B, T, D] each
split heads            [B, T, H, Dh]
move head axis         [B, H, T, Dh]
Q Kᵀ                   [B, H, T, T]
weighted values        [B, H, T, Dh]
merge heads            [B, T, D]
output projection Wo   [B, T, D]
```

`Wo` mixes features across heads. Holding `D` fixed while changing `H` does not multiply the four full-width projection matrices by `H`; it repartitions their outputs. It does change head dimension, attention-map count, and execution characteristics.

For `B=2, T=16, D=64, H=4`, each head has width 16 and there are `2 × 4 × 16 × 16` attention coefficients. The attention map indexes positions, not vocabulary items.

## Lesson 1.1.5 — Causality, padding, and mask conventions

At position `i`, a causal model may use positions `j <= i`. Including the diagonal is correct because the logit at `i` predicts the **next** token, not the input token at `i`.

```text
         key position
         0  1  2  3
query 0  ✓  ×  ×  ×
      1  ✓  ✓  ×  ×
      2  ✓  ✓  ✓  ×
      3  ✓  ✓  ✓  ✓
```

Replace disallowed scores with negative infinity **before** softmax. Replacing them with zero allows positive probability to leak to forbidden positions. Multiplying probabilities by a mask afterward loses row normalization unless deliberately renormalized.

Padding and causality solve different problems. A key-padding mask removes artificial padded keys. A loss mask excludes padded prediction targets. Neither substitutes for the other. All-masked query rows require explicit handling; softmax of all negative infinities is undefined. For the first implementation, use fixed-length, unpadded windows.

Here is a teaching implementation with inspectable intermediate values:

```python
import math
import torch


def causal_attention(q, k, v):
    # q, k, v: [B, H, T, Dh]; this example is square self-attention.
    length = q.size(-2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    allowed = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


torch.manual_seed(7)
q, k, v = [torch.randn(2, 4, 16, 16) for _ in range(3)]
context, weights = causal_attention(q, k, v)
assert context.shape == (2, 4, 16, 16)
assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 16))
assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
```

Library masks need care: in PyTorch's [scaled dot-product attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html), Boolean `True` means allowed; some higher-level attention APIs use `True` to mean blocked. The functional operation also applies the provided dropout probability regardless of an enclosing module's evaluation mode; explicitly use zero during evaluation. Inspect the API rather than transferring a mask by intuition.

## Lesson 1.1.6 — FFNs, residuals, and normalization

A conventional position-wise feed-forward network is:

```text
FFN(x) = GELU(x W1 + b1) W2 + b2
D → Dff → D, often with Dff around 4D in a basic GPT-style block
```

The same weights are applied at every position. The FFN mixes features within a token representation, whereas attention mixes information across positions. Nonlinearity is essential: two linear maps without an activation collapse into a single linear map. [GELU](https://arxiv.org/abs/1606.08415) is a smooth activation; modern gated alternatives appear in lesson 1.5.

A pre-norm block can be written:

```text
x ──→ LayerNorm ──→ causal multi-head attention ──→ + ──→ h
└────────────────────────────────────────────────┘
h ──→ LayerNorm ──→ FFN ─────────────────────────→ + ──→ y
└────────────────────────────────────────────────┘
```

Equivalently, `h = x + Attention(LN(x))`, then `y = h + FFN(LN(h))`. The residual stream carries the accumulated representation and gives gradients a direct additive path. It does not mean attention and FFNs can be omitted without consequences.

LayerNorm normalizes features within each token, using a feature mean and variance plus learned scale and usually bias. It does not normalize across the batch or replace attention softmax. Post-norm instead uses `h = LN(x + Attention(x))`. Both are valid designs; normalization placement affects gradient behavior and training stability. Pre-norm is a useful starting point for the project, followed by a final normalization before the LM head.

## Lesson 1.1.7 — Why transformers changed language modeling

An RNN propagates state through positions sequentially, making training across a sequence hard to parallelize. Causal transformers can compute representations for all known training positions together using a triangular mask. Attention also provides a direct path between distant positions within a layer, instead of requiring information to traverse every intermediate recurrent step.

That advantage has a cost: full attention compares all pairs of positions, and generation remains sequential because each new token becomes part of the next input. Transformer training parallelism does not imply that 100 unknown output tokens can be generated in one ordinary autoregressive forward pass.

## Checkpoint

1. For `B=3, T=10, D=96, H=6`, give the Q shape after splitting heads and the score shape.
2. Why is the diagonal of a causal mask allowed? What predicts the token at input position 5?
3. Why can neither `softmax(scores, dim=-2)` nor masking scores with zero implement the intended attention?
4. What different work do attention, the FFN, residual connections, and LayerNorm perform?

<details>
<summary>Show answers</summary>

1. `Dh=16`; Q is `[3, 6, 10, 16]`, scores are `[3, 6, 10, 10]`.
2. The input at `i` is known when predicting `i+1`. The logit at position 4 predicts the token at position 5, assuming the ordinary shifted-target layout.
3. `dim=-2` normalizes across queries instead of keys. A score of zero has exponential one and therefore retains probability; negative infinity excludes it.
4. Attention mixes positions, the FFN nonlinearly transforms features, residuals carry and update the representation, and LayerNorm controls feature scale within each token.

</details>

## Hands-on exercises

1. Implement a multi-head wrapper around `causal_attention`, including projections and merging. Use `D=48, H=3`. Record each intermediate shape.
2. Design a test in which you change only future input tokens. Compare the earlier output logits in evaluation mode. Then deliberately remove the causal mask and repeat.
3. Count the weight parameters, excluding biases and norms, for attention and a two-matrix FFN with `Dff=4D`. Explain why increasing the head count at fixed `D` does not multiply this count.

<details>
<summary>Show exercise solutions</summary>

### Exercises 1 and 2 — multi-head attention and the causal-invariance test

```python
from torch import nn


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        assert width % heads == 0, "width must divide evenly into heads"
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)      # one projection, split three ways
        self.out = nn.Linear(width, width)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)        # each [B, H, T, Dh]
        context, weights = causal_attention(q, k, v)
        merged = context.transpose(1, 2).reshape(B, T, D)
        return self.out(merged), weights


torch.manual_seed(0)
model = MultiHeadCausalAttention(width=48, heads=3).eval()

x = torch.randn(2, 7, 48)
y, w = model(x)
print(f"x {tuple(x.shape)} -> per-head q/k/v [2, 3, 7, 16] -> scores {tuple(w.shape)} -> y {tuple(y.shape)}")

assert y.shape == (2, 7, 48)
assert w.shape == (2, 3, 7, 7)
assert torch.allclose(w.sum(-1), torch.ones(2, 3, 7), atol=1e-6)   # rows are distributions
assert torch.count_nonzero(w.triu(diagonal=1)) == 0                # nothing attends forward

# Exercise 2: change only the future, and check the past is unmoved.
a = torch.randn(1, 9, 48)
b = a.clone()
b[:, 5:] = torch.randn(1, 4, 48)
with torch.inference_mode():
    ya, _ = model(a)
    yb, _ = model(b)

print("prefix outputs identical:", torch.allclose(ya[:, :5], yb[:, :5], atol=1e-6))
print("suffix outputs differ:   ", not torch.allclose(ya[:, 5:], yb[:, 5:], atol=1e-6))
```

```text
x (2, 7, 48) -> per-head q/k/v [2, 3, 7, 16] -> scores (2, 3, 7, 7) -> y (2, 7, 48)
prefix outputs identical: True
suffix outputs differ:    True
```

The second test is the one that matters, and it is the same check the project requires. Both halves are needed: if the prefix changed, the mask leaks the future; if the suffix did *not* change, your model is ignoring its input somewhere. Delete the `masked_fill` line and rerun — the prefix assertion fails immediately, which is the fastest way to see what the mask is actually doing.

### Exercise 3 — parameter count

```python
model = MultiHeadCausalAttention(width=48, heads=3)
weight_params = sum(p.numel() for p in model.parameters() if p.dim() > 1)
print(f"attention weight params {weight_params:,} = 4*D^2 = {4 * 48 * 48:,}")

for heads in (1, 2, 4, 8, 16):
    m = MultiHeadCausalAttention(width=48, heads=heads) if 48 % heads == 0 else None
    if m:
        total = sum(p.numel() for p in m.parameters() if p.dim() > 1)
        print(f"heads={heads:2d} head_dim={48 // heads:2d}  weight params {total:,}")
```

Q/K/V and the output projection total `4D²`; a two-matrix FFN with `Dff=4D` adds `8D²`, so a block is about `12D²` excluding biases and norms. The loop shows why head count does not appear in that formula: heads partition the same `D` channels into `H` groups of `D/H`, so the projections keep their size and only the *grouping* of channels changes. More heads means more, narrower attention patterns at identical parameter cost — the trade is representational, not budgetary.

</details>

## Completion criteria

Derive scaled dot-product attention, trace every dimension, implement the mask and a pre-norm block, and distinguish parallel training from sequential generation. Carry the causal-invariance test into the [project](../project/tiny-gpt/README.md).

## Primary references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — architecture and attention.
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) — mask and dropout semantics.
- [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) — GELU.

## Videos and code to read

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy builds exactly this lesson's block live; the best two hours available on this topic
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — `model.py` is a complete, readable decoder-only transformer in ~300 lines; read `CausalSelfAttention` against your Exercise 1
- [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) — the original paper with the implementation interleaved line by line
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the reference diagrams for shapes and head splitting

## Mapped companion lessons

- [Why Transformers](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/01-why-transformers) establishes the architecture transition from recurrence.
- [Self-Attention from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/02-self-attention-from-scratch), [Multi-Head Attention](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/03-multi-head-attention), and [The Full Transformer](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/05-full-transformer) map to the core derivation and block implementation.
- [Build a Transformer from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/14-build-a-transformer-capstone) provides an additional end-to-end build.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).
