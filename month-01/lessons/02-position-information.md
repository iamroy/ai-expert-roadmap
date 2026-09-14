# 1.2 — Position Information

**Depth: LEARN**

**Goal:** explain how transformers represent order and why position schemes affect context extension.

[Month 1 roadmap](../README.md) · [Previous: Transformer Architecture](01-transformer-architecture.md) · [Next: Tokenization](03-tokenization.md)

## Lesson 1.2.1 — Content does not specify order

Token embeddings identify vocabulary entries. If the same token occurs twice, its initial embedding is the same both times. In unmasked self-attention without any positional signal, permuting input positions simply permutes corresponding outputs: the operation has no independent representation of order.

A causal mask introduces an asymmetric visibility pattern, so a causal transformer is not wholly permutation-equivariant. Nevertheless, visibility alone is not an explicit, flexible encoding of exact position or relative distance. “Dog bites person” and “Person bites dog” require order-sensitive representations, not merely the presence of the same words.

Think of a position scheme as answering one of two questions: “Where is this token?” or “How far apart are these two positions?” Different schemes put that information into different parts of attention.

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

## Lesson 1.2.5 — ALiBi biases attention scores

[ALiBi](https://arxiv.org/abs/2108.12409) adds a distance-dependent bias to attention scores. For a causal head with positive slope `m` and `j <= i`:

```text
score[i, j] = qi · kj / sqrt(Dh) − m × (i − j)
```

Older positions receive a larger penalty. Different heads use different slopes, giving different distance preferences. Unlike a learned position table, this formula is defined for arbitrary distances. Unlike RoPE, it changes the score directly rather than rotating Q/K vectors.

An ALiBi bias is not a causal mask: future positions still need to be excluded. Likewise, a finite negative distance penalty does not absolutely forbid attending to a distant token.

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

## Mapped companion lessons

- [Positional Encoding — Sinusoidal, RoPE, ALiBi](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/04-positional-encoding) maps directly to every mechanism compared here.
- [Token and Positional Embeddings](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/19-capstone-projects/32-token-positional-embeddings) provides a focused implementation artifact.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).
