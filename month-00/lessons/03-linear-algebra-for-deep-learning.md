# 0.3 — Linear Algebra for Deep Learning

The goal is practical geometric and shape intuition. You should understand what deep-learning operations do, why they are useful, and how their dimensions fit together. Proof-heavy mathematics is outside this prerequisite review.

## 1. Scalars, vectors, matrices, and tensors

- A **scalar** is one number.
- A **vector** is an ordered list of numbers, often representing features or a direction.
- A **matrix** is a rectangular grid of numbers. It can represent a dataset, a collection of vectors, or a linear transformation.
- A **tensor** generalizes these objects to more axes.

Typical AI shapes:

```text
scalar loss:                    []
one embedding:                 [embedding_dim]
token embeddings:              [sequence, embedding_dim]
batched token embeddings:      [batch, sequence, embedding_dim]
multi-head attention tensors:  [batch, heads, sequence, head_dim]
```

“Tensor” describes structure; “vector,” “matrix,” and “linear map” add mathematical meaning.

## 2. Matrix multiplication as transformation

Let a row vector `x` contain `d_in` input features and let `W` have shape `[d_in, d_out]`:

```text
xW: [d_in] @ [d_in, d_out] → [d_out]
```

Every output coordinate is a weighted sum of all input coordinates. The matrix maps a vector from one feature space into another.

For a batch:

```text
XW: [batch, d_in] @ [d_in, d_out] → [batch, d_out]
```

An affine neural-network layer also adds a bias:

```text
y = xW + b
```

Strictly, the bias makes this an affine transformation rather than a purely linear one. Deep-learning libraries may store weights using a transposed convention, but the same dimensional reasoning applies.

## 3. Dot products: alignment and weighted sums

For equal-length vectors:

```text
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

The dot product serves two closely related roles:

1. It measures directional alignment, scaled by vector lengths.
2. It computes a weighted sum, where one vector supplies values and the other supplies weights.

Geometrically:

```text
a · b = ||a|| ||b|| cos(θ)
```

- positive: vectors point broadly in the same direction
- zero: vectors are orthogonal
- negative: vectors point broadly in opposite directions

Raw dot product is affected by both direction and magnitude. That is useful in attention, where learned magnitudes can matter, but it is not a pure direction-only similarity measure.

## 4. Norms and distances

The L2 norm is the Euclidean length of a vector:

```text
||x||₂ = √(x₁² + x₂² + ... + xₙ²)
```

Common norms:

- L1: sum of absolute values
- L2: Euclidean length
- Frobenius: L2-like norm over all entries of a matrix

Euclidean distance measures the length of the difference:

```text
distance(a, b) = ||a - b||₂
```

Distance and similarity are related but not identical. The best choice depends on how embeddings were trained and whether their lengths carry useful information.

## 5. Cosine similarity

Cosine similarity normalizes out vector length:

```text
cosine(a, b) = (a · b) / (||a|| ||b||)
```

For nonzero vectors, it ranges from `-1` to `1`. If embeddings are L2-normalized, cosine similarity equals their dot product:

```text
||a|| = ||b|| = 1  ⇒  cosine(a, b) = a · b
```

Cosine similarity is common in retrieval because it compares direction in embedding space. It is undefined for a zero vector, so implementations use safeguards or prevent zero embeddings.

## 6. Projections

The projection of vector `x` onto nonzero vector `u` is:

```text
projᵤ(x) = (x · u / u · u) u
```

The scalar coefficient tells how much of `x` lies along `u`; multiplying by `u` produces the vector component in that direction.

Projection provides intuition for learned features. A neuron can respond strongly when an input has a large component along its learned weight direction.

If `u` is a unit vector, the formula simplifies to `(x · u)u`.

## 7. Basis and dimensionality

A basis is a set of independent directions that can construct every vector in a space. Coordinates describe a vector relative to that basis.

A learned linear layer can rotate, stretch, compress, expand, or mix feature coordinates. Changing coordinates does not necessarily change the underlying information, while reducing dimensions may discard information.

An embedding dimension of 768 means each item is represented by 768 coordinates. It does not imply that the data uses 768 independent directions; the effective structure can lie near a lower-dimensional subspace.

## 8. Linear independence and rank

Vectors are linearly independent when none can be constructed from the others. Matrix rank counts the number of independent directions represented by its rows or columns.

For a matrix shaped `[m, n]`:

```text
rank(A) ≤ min(m, n)
```

A low-rank matrix can be described using fewer independent factors. This matters for compression and parameter-efficient adaptation.

For example, LoRA represents a weight update as a product of two thin matrices:

```text
ΔW = AB
```

If the inner dimension is small, `ΔW` has low rank and requires far fewer trainable parameters than a full weight matrix.

## 9. Transpose

The transpose swaps matrix rows and columns:

```text
A:   [m, n]
Aᵀ:  [n, m]
```

Useful identities:

```text
(AB)ᵀ = BᵀAᵀ
(Aᵀ)ᵀ = A
```

In attention, transposing keys changes `[sequence, head_dim]` to `[head_dim, sequence]`, allowing all query-key dot products to be computed at once.

## 10. Inverses and solving systems

If a square matrix `A` is invertible:

```text
A⁻¹A = I
```

and the system `Ax = b` has solution `x = A⁻¹b`.

Conceptually, an inverse undoes a transformation. In numerical code, do not normally compute the inverse explicitly just to solve a system. A solver such as `torch.linalg.solve(A, b)` is usually faster and more stable.

Many matrices are not invertible, and rectangular matrices do not have ordinary inverses. The pseudoinverse gives a least-squares-related generalization.

## 11. Eigenvalues and eigenvectors

An eigenvector is a nonzero direction that a square transformation does not rotate away from itself:

```text
Av = λv
```

The eigenvalue `λ` tells how much that direction is scaled, including a possible sign reversal.

Conceptual uses include understanding repeated transformations, stability, covariance structure, and optimization curvature. For Month 0, recognize the equation and geometric meaning; do not spend time on hand calculation.

## 12. Singular value decomposition

Every matrix `A` can be factored as:

```text
A = UΣVᵀ
```

- columns of `V` describe input directions
- singular values in `Σ` describe the strength of each direction
- columns of `U` describe corresponding output directions

SVD works for rectangular and rank-deficient matrices. Keeping only the largest `k` singular values produces the best rank-`k` approximation under common matrix norms.

This supports:

- dimensionality reduction
- compression
- denoising
- analysis of learned weight matrices and representations
- intuition for low-rank adaptation

## 13. Embeddings as a matrix lookup

An embedding table with vocabulary size `V` and embedding dimension `d` is a matrix:

```text
E: [V, d]
```

A token ID selects one row. For token IDs shaped `[batch, sequence]`, lookup returns:

```text
[batch, sequence] → [batch, sequence, d]
```

One-hot multiplication gives the same mathematical result but is wasteful:

```text
one_hot(token) @ E = E[token]
```

Training changes rows of `E` so tokens useful in similar contexts can acquire related representations.

## 14. Attention as linear algebra

For hidden states `X` shaped `[sequence, model_dim]`, learned projections create queries, keys, and values:

```text
Q = XW_Q
K = XW_K
V = XW_V
```

For one head:

```text
Q: [sequence, head_dim]
K: [sequence, head_dim]
V: [sequence, value_dim]
```

All query-key dot products are computed by:

```text
QKᵀ: [sequence, head_dim] @ [head_dim, sequence]
    → [sequence, sequence]
```

After scaling, masking, and row-wise softmax, the attention matrix contains weights:

```text
P = softmax(QKᵀ / √head_dim)
```

Then:

```text
PV: [sequence, sequence] @ [sequence, value_dim]
  → [sequence, value_dim]
```

Each output token is a weighted combination of value vectors. Queries and keys determine *where to look*; values determine *what information to collect*.

The `√head_dim` scaling keeps dot-product magnitudes from growing too large as dimension increases, which helps prevent softmax from becoming excessively saturated.

## 15. Shape-first debugging

Before debugging numerical values, write the intended shapes:

```text
X:       [batch, sequence, model_dim]
W_Q:     [model_dim, heads × head_dim]
Q:       [batch, sequence, heads × head_dim]
Q split: [batch, heads, sequence, head_dim]
scores:  [batch, heads, sequence, sequence]
```

For every matrix multiplication, verify that the inner dimensions match and that the remaining dimensions describe the output you intend. Most early attention bugs are axis-order or masking errors rather than failures of the underlying math.

## Exit test

Answer without notes.

1. In geometric terms, what does multiplying a vector by a matrix do?
2. Why is the raw dot product not purely a measure of directional similarity?
3. Two nonzero vectors are L2-normalized. How are their dot product and cosine similarity related?
4. What does it mean if two vectors have dot product zero?
5. What is the projection of `x = [3, 4]` onto `u = [1, 0]`?
6. A matrix has shape `[100, 20]`. What is the largest possible rank?
7. Why should numerical code generally solve `Ax = b` instead of calculating `A⁻¹b` explicitly?
8. State the geometric meaning of `Av = λv`.
9. What does truncating an SVD to its largest singular values accomplish?
10. An embedding table has shape `[50_000, 768]`. What is the output shape for token IDs shaped `[8, 128]`?
11. If `Q`, `K`, and `V` each have shape `[32, 64]`, what are the shapes of `QKᵀ` and `softmax(QKᵀ)V`?
12. In attention, what distinct roles do queries/keys and values play?
13. A linear layer maps 768 input features to 3,072 output features for 16 tokens. Give compatible shapes for `X`, `W`, and `XW` using row-vector convention.
14. Explain why `QKᵀ` grows quadratically with sequence length.

<details>
<summary>Show answers</summary>

### Answers

1. It maps the vector into another coordinate/feature space by producing weighted combinations of its input coordinates. Depending on the matrix, it can rotate, reflect, stretch, compress, expand, or project directions.
2. `a · b = ||a|| ||b|| cos(θ)`, so the result depends on magnitudes as well as the angle between the vectors.
3. They are equal because both vector norms are 1.
4. The vectors are orthogonal under the standard inner product. In geometric terms they meet at 90 degrees, provided neither is the zero vector.
5. `[3, 0]`. Since `u` is a unit vector, the projection is `(x · u)u = 3[1, 0]`.
6. 20, because rank cannot exceed the smaller matrix dimension.
7. Direct solvers avoid unnecessary computation and usually have better numerical stability. Forming an explicit inverse can amplify floating-point error.
8. `v` is a direction preserved by the transformation `A`; `A` only scales it by `λ`, with a negative value also reversing direction.
9. It produces a low-rank approximation that preserves the strongest directions while reducing storage and computation. Under common matrix norms, truncated SVD gives the best approximation of that rank.
10. `[8, 128, 768]`. Every token ID is replaced by its 768-value embedding row.
11. `QKᵀ` has shape `[32, 32]`. Multiplying its row-wise softmax by `V` produces `[32, 64]`.
12. Queries and keys produce compatibility scores that decide how strongly positions attend to one another. Values carry the information combined using those attention weights.
13. `X` is `[16, 768]`, `W` is `[768, 3072]`, and `XW` is `[16, 3072]`.
14. With sequence length `n`, every one of `n` queries is compared with all `n` keys, producing an `[n, n]` score matrix with `n²` entries per head and batch item.

</details>

## Primary references

- [Deep Learning — Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)
- [Dive into Deep Learning — Linear Algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html)
- [PyTorch Linear Algebra](https://docs.pytorch.org/docs/stable/linalg.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
