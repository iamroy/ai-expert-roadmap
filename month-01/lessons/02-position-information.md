# 1.2 — Position Information

**Depth: LEARN**

**Goal:** explain how transformers represent order and why position schemes affect context extension.

[Month 1 roadmap](../README.md) · [Previous: Transformer Architecture](01-transformer-architecture.md) · [Next: Tokenization](03-tokenization.md)

## Lesson 1.2.1 — Content does not specify order

Token embeddings identify vocabulary entries. If the same token occurs twice, its initial embedding is the same both times. In unmasked self-attention without any positional signal, permuting input positions simply permutes corresponding outputs: the operation has no independent representation of order.

A causal mask introduces an asymmetric visibility pattern, so a causal transformer is not wholly permutation-equivariant. Nevertheless, visibility alone is not an explicit, flexible encoding of exact position or relative distance. “Dog bites person” and “Person bites dog” require order-sensitive representations, not merely the presence of the same words.

Think of a position scheme as answering one of two questions: “Where is this token?” or “How far apart are these two positions?” Different schemes put that information into different parts of attention.

The permutation claim is easy to state and easy to doubt, so measure it. Shuffle the input positions and check whether the outputs are merely shuffled the same way:

```python
import math

import torch


def attention(x, Wq, Wk, Wv):
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    weights = (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))).softmax(-1)
    return weights @ v


torch.manual_seed(0)
D = 8
Wq, Wk, Wv = (torch.randn(D, D) for _ in range(3))
x = torch.randn(1, 4, D)
perm = torch.tensor([2, 0, 3, 1])

out = attention(x, Wq, Wk, Wv)
out_shuffled = attention(x[:, perm], Wq, Wk, Wv)
print("no positions:  permuting inputs just permutes outputs:",
      torch.allclose(out[:, perm], out_shuffled, atol=1e-6))

positions = torch.randn(4, D) * 0.5
with_pos = attention(x + positions, Wq, Wk, Wv)
with_pos_shuffled = attention(x[:, perm] + positions, Wq, Wk, Wv)
print("with positions: same check:                          ",
      torch.allclose(with_pos[:, perm], with_pos_shuffled, atol=1e-6))
```

```text
no positions:  permuting inputs just permutes outputs: True
with positions: same check:                           False
```

The first `True` is the problem in one line: without a position signal, attention cannot distinguish "dog bites person" from "person bites dog", because reordering the input reorders the output and changes nothing else. The `False` is the fix — once each slot carries a distinct vector, the same tokens in a different order become genuinely different inputs.

## Lesson 1.2.2 — Learned absolute embeddings

Create a table `P` of shape `[Tmax, D]`, and add its row to each token embedding:

```text
x[i] = token_embedding[token_id[i]] + P[i]
```

Addition preserves the model width. Gradients learn both tables. A learned table is easy to implement and serves the project well. It also has a concrete capacity limit: position `Tmax` has no row. Expanding the table allocates parameters; it does not teach the new positions what to mean.

For cached generation, position numbers must continue from the existing prefix. Restarting every generated token at position zero changes the model input even if the token IDs and cache are otherwise correct. If you deliberately truncate a window and renumber its positions, document that policy and compare it against the training setup.

## Lesson 1.2.3 — Sinusoidal positions

A fixed scheme uses sine and cosine waves at different frequencies. For feature-pair index `j`:

```text
PE(pos, 2j)   = sin(pos / 10000^(2j/D))
PE(pos, 2j+1) = cos(pos / 10000^(2j/D))
```

Low-frequency pairs change slowly; high-frequency pairs distinguish nearby positions. The vectors can be computed beyond the training length without adding learned rows. This is **representability**, not evidence that a trained model uses unfamiliar distances correctly.

Illustrative code for an even model width:

```python
import math
import torch


def sinusoidal_positions(length, width):
    if width <= 0 or width % 2:
        raise ValueError("width must be a positive even number")
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
    angles = positions * frequencies
    table = torch.empty(length, width)
    table[:, 0::2] = angles.sin()
    table[:, 1::2] = angles.cos()
    return table


assert sinusoidal_positions(12, 8).shape == (12, 8)
```

The table has a property worth seeing directly, because it is the ancestor of the RoPE result in the next section:

```python
import torch.nn.functional as F

table = F.normalize(sinusoidal_positions(64, 32), dim=-1)

print("cosine similarity to position 0:")
for p in [0, 1, 2, 5, 10, 20, 40, 63]:
    print(f"  pos {p:2d}: {(table[0] @ table[p]).item():+.4f}")

print("\nsame offset at different absolute positions (p vs p+5):")
for p in [0, 10, 30]:
    print(f"  {p:2d} vs {p + 5:2d}: {(table[p] @ table[p + 5]).item():+.4f}")
```

```text
cosine similarity to position 0:
  pos  0: +1.0000
  pos  1: +0.9571
  pos  2: +0.8581
  pos  5: +0.7361
  pos 10: +0.6287
  pos 20: +0.6544
  pos 40: +0.4869
  pos 63: +0.5530

same offset at different absolute positions (p vs p+5):
   0 vs  5: +0.7361
  10 vs 15: +0.7361
  30 vs 35: +0.7361
```

The second block is the interesting one: a displacement of 5 produces the same similarity wherever it occurs. The encoding carries relative information even though each row is computed from an absolute index, which is what allows a fixed offset to be expressed as a linear function of the position encoding.

The first block is a caution against over-reading that. Similarity falls with distance overall, but not monotonically — position 20 is *more* similar to position 0 than position 10 is, because summing periodic functions produces interference rather than a clean decay curve. Do not describe sinusoidal encodings as giving the model a tidy sense of "nearer means more similar."

## Lesson 1.2.4 — RoPE rotates queries and keys

[Rotary position embeddings](https://arxiv.org/abs/2104.09864) act on pairs of Q/K features rather than simply adding a position vector to the residual stream. A two-dimensional rotation is:

```text
R(θ) [a, b] = [a cos θ − b sin θ, a sin θ + b cos θ]
```

At position `m`, rotate a query pair by `mθ`; at position `n`, rotate a key pair by `nθ`. Orthogonality gives:

```text
(R(mθ) q)ᵀ (R(nθ) k) = qᵀ R((n−m)θ) k
```

Their dot product therefore depends on relative displacement. Multiple feature pairs use different frequencies. Standard RoPE applies to Q and K; values need not be rotated. Rotary dimension and pairing conventions vary, so implementations must match the checkpoint's configuration.

RoPE does not change the basic `[B, H, T, Dh]` shape. It changes the geometric relationship used by attention. Extrapolating to unseen positions can introduce unfamiliar phase patterns. Context-extension approaches may rescale frequencies or positions and use additional training; changing a frequency setting is not a universal quality fix.

Here it is applied to real tensors, with the identity checked rather than asserted:

```python
def rope_angles(head_dim, positions, base=10000.0):
    inverse_frequencies = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    return positions[:, None].float() * inverse_frequencies[None, :]


def apply_rope(x, positions):
    """x: [B, H, T, Dh]; rotates consecutive feature pairs by position-dependent angles."""
    angles = rope_angles(x.size(-1), positions)
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]

    rotated = torch.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


torch.manual_seed(0)
head_dim = 16
q = torch.randn(1, 1, 1, head_dim)
k = torch.randn(1, 1, 1, head_dim)


def rotated_score(m, n):
    """Score between a query at position m and a key at position n."""
    return (apply_rope(q, torch.tensor([m])) * apply_rope(k, torch.tensor([n]))).sum().item()


print(f"{'(m, n)':>12} {'offset':>7} {'score':>11}")
for m, n in [(0, 3), (5, 8), (100, 103), (0, 5), (10, 15)]:
    print(f"{str((m, n)):>12} {n - m:>7} {rotated_score(m, n):>11.6f}")
```

```text
      (m, n)  offset       score
      (0, 3)       3   -3.160644
      (5, 8)       3   -3.160644
  (100, 103)       3   -3.160643
      (0, 5)       5    2.395697
    (10, 15)       5    2.395697
```

That table *is* the RoPE property. Positions 0 and 3 score identically to positions 100 and 103, because only the offset of 3 matters — absolute location cancels out of the dot product. The tiny difference at position 100 is float32 rounding, not a real effect.

This is why RoPE is called a relative scheme despite being applied to each position independently. Nothing computes `n - m` anywhere; the rotation makes it fall out of the geometry. It is also why RoPE needs no extra parameters and no lookup table, and why the attention score, not the residual stream, is where the position information ends up.

The same machinery exposes the cached-decoding bug from the checkpoint. If a cached decoder gives each new token position 0 instead of its true offset:

```python
prefix_length = 6
print(f"correct: query at {prefix_length}, key at 2, offset {prefix_length - 2}: "
      f"{rotated_score(2, prefix_length):.6f}")
print(f"bug:     query at 0, key at 2, offset {0 - 2}: {rotated_score(2, 0):.6f}")
```

```text
correct: query at 6, key at 2, offset 4: -1.315803
bug:     query at 0, key at 2, offset -2: -1.895645
```

The identical query and key now score differently, because the model has been told the wrong distance — and it degrades output without raising anything. Positions must come from the cache length (1.6.2), which is exactly what the cached-versus-uncached logit comparison is testing for.

## Lesson 1.2.5 — ALiBi biases attention scores

[ALiBi](https://arxiv.org/abs/2108.12409) adds a distance-dependent bias to attention scores. For a causal head with positive slope `m` and `j <= i`:

```text
score[i, j] = qi · kj / sqrt(Dh) − m × (i − j)
```

Older positions receive a larger penalty. Different heads use different slopes, giving different distance preferences. Unlike a learned position table, this formula is defined for arbitrary distances. Unlike RoPE, it changes the score directly rather than rotating Q/K vectors.

An ALiBi bias is not a causal mask: future positions still need to be excluded. Likewise, a finite negative distance penalty does not absolutely forbid attending to a distant token.

Both points are visible in the numbers:

```python
def alibi_slopes(heads):
    start = 2 ** (-8 / heads)
    return torch.tensor([start ** (i + 1) for i in range(heads)])


def alibi_bias(heads, length):
    positions = torch.arange(length)
    distance = positions[None, :] - positions[:, None]          # negative into the past
    bias = distance[None].float() * alibi_slopes(heads)[:, None, None]
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    return bias.masked_fill(~causal, float("-inf"))              # the mask is still required


print("slopes (8 heads):", [round(s, 4) for s in alibi_slopes(8).tolist()])
bias = alibi_bias(heads=8, length=6)
print("head 0 (steepest), query 5 over keys 0-5:", bias[0, 5].tolist())
print("head 7 (flattest), query 5 over keys 0-5:", [round(v, 5) for v in bias[7, 5].tolist()])

weights = (torch.zeros(8, 6, 6) + bias).softmax(-1)              # uniform scores, bias only
print(f"head 0: key 5 gets {weights[0, 5, 5]:.3f}, key 0 gets {weights[0, 5, 0]:.5f}")
print(f"head 7: key 5 gets {weights[7, 5, 5]:.3f}, key 0 gets {weights[7, 5, 0]:.5f}")
```

```text
slopes (8 heads): [0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039]
head 0 (steepest), query 5 over keys 0-5: [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
head 7 (flattest), query 5 over keys 0-5: [-0.01953, -0.01562, -0.01172, -0.00781, -0.00391, 0.0]
head 0: key 5 gets 0.414, key 0 gets 0.03399
head 7: key 5 gets 0.168, key 0 gets 0.16504
```

Three things to take from this. The slopes form a geometric series, so heads span a range of distance preferences rather than all sharing one: head 0 concentrates hard on recent tokens while head 7 is nearly distance-neutral, and a model can use both at once.

The bias is finite, so distance discourages rather than forbids. Even under the steepest head, key 0 still receives 3.4% of the attention mass — a strong enough content match can overcome the penalty, which is the intended behavior.

And the `masked_fill` line is not optional. Without it, the bias for future positions is *positive* (distance is positive forward), so future tokens would be actively favored. ALiBi shapes scores; it does not enforce causality.

## Lesson 1.2.6 — Choosing and debugging a position scheme

| Method | Injection point | Learned position table? | Main extension concern |
|---|---|---|---|
| Learned absolute | Add to token embeddings | Yes | Untrained or absent rows |
| Sinusoidal | Add to token embeddings | No | Unfamiliar distances/patterns |
| RoPE | Rotate Q/K feature pairs | Usually fixed frequencies | Phase distribution and scaling configuration |
| ALiBi | Add bias to attention scores | No table | Distance bias and task-dependent quality |

In production, check tokenizer and special-token offsets, left/right padding, position IDs, cached-prefix length, and training/inference configuration consistency. A position bug can produce fluent but substantially worse output. Tests should compare cached and uncached logits for the same prefix, not just inspect whether generated text “looks fine.”

Long-context evaluation needs both information retrieval and tasks requiring several distant facts. A model might recover a single name at a long distance yet fail to combine two constraints. This becomes the context-engineering problem in Month 2.

## Checkpoint

1. How do learned absolute positions, RoPE, and ALiBi enter computation differently?
2. Why is “the formula accepts position 100,000” insufficient evidence of useful context at that length?
3. A cached decoder gives every new token position ID zero. What changes even if token IDs are correct?

<details>
<summary>Show answers</summary>

1. Learned positions add vectors to input representations; RoPE rotates Q/K pairs; ALiBi adds distance biases to attention scores.
2. Training may never have covered those distances or interactions. Mathematical availability does not establish retrieval, integration, numerical stability, or task accuracy there.
3. The position signal is inconsistent with the prefix: absolute embeddings repeat the wrong row, or relative phases in RoPE become incorrect. Compare cached/uncached logits with continuing position IDs.

</details>

## Hands-on exercises

1. Implement the two-dimensional rotation above. For `q=[1,2]`, `k=[3,4]`, `m=2`, `n=5`, and `θ=0.1`, verify the relative-displacement identity numerically.
2. Calculate the ALiBi biases for query index 3 attending to keys 0–3 with slope 0.5. Explain why those biases alone do not define a full attention distribution.
3. Design an experiment comparing learned and sinusoidal positions on sequences longer than training sequences. Specify what would and would not be a fair comparison.

<details>
<summary>Show exercise solutions</summary>

1. Both sides are approximately `11.09974`; use the same pairing and sign convention on each side:

    ```python
    import math
    import torch

    def rotate(x, angle):
        c, s = math.cos(angle), math.sin(angle)
        return torch.stack((x[0] * c - x[1] * s, x[0] * s + x[1] * c))

    q, k = torch.tensor([1., 2.]), torch.tensor([3., 4.])
    left = rotate(q, 0.2) @ rotate(k, 0.5)
    right = q @ rotate(k, 0.3)
    assert torch.allclose(left, right, atol=1e-6)
    ```

2. Biases are `[-1.5, -1.0, -0.5, 0.0]`. Add the Q/K compatibility scores and apply softmax across allowed keys; biases are not probabilities.
3. Hold model width, data, optimizer, and training lengths fixed. Use several seeds, evaluate across lengths and evidence positions, and report short-context quality too. A learned table with untrained extra rows has a different failure mode from a sinusoidal scheme; report it explicitly. Do not claim either model has learned long-range behavior merely because inference runs.

</details>

## Completion criteria

Locate position information in the forward pass, explain RoPE's relative-position identity, and design a quality test for context extension. Deriving every RoPE scaling method is outside this month's scope.

## Primary references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — sinusoidal position encoding.
- [RoFormer](https://arxiv.org/abs/2104.09864) — rotary embeddings.
- [Train Short, Test Long](https://arxiv.org/abs/2108.12409) — ALiBi.

## Videos and code to read

- [lucidrains/rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch) — a compact, readable RoPE implementation; compare its rotation against your own
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — learned absolute positions, the simplest baseline to contrast with RoPE
- [labmlai annotated implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) — annotated RoPE and ALiBi side by side with the equations

## Mapped companion lessons

- [Positional Encoding — Sinusoidal, RoPE, ALiBi](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/04-positional-encoding) maps directly to every mechanism compared here.
- [Token and Positional Embeddings](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/32-token-positional-embeddings) provides a focused implementation artifact.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).
