# 0.3 — Linear Algebra for Deep Learning

The goal is practical geometric intuition for embeddings, attention, transformers, and neural networks. Proof-heavy mathematics and hand calculation are outside this prerequisite review.

This file preserves all 14 lessons, the four-priority checkpoint, transformer connection, exit test, and answer guide from the roadmap conversation. It retains a few useful extensions from the original repository lesson.

**Source conversation:** [Create Month Zero Learning List](https://chatgpt.com/share/6aa28c17-a29c-83ea-ae5b-9202ec5ba987)

## Lesson 0.3.1 — Scalars, vectors, matrices, and tensors

Map these mathematical objects to ML concepts immediately:

```text
Scalar  → one number
Vector  → one feature vector, embedding, point, or direction
Matrix  → collection of vectors or linear transformation
Tensor  → higher-dimensional data
```

Typical AI shapes:

```text
scalar loss:                    []
embedding vector:               [768]
token embeddings:               [128, 768]
batch of token embeddings:      [32, 128, 768]
multi-head attention tensor:    [32, 12, 128, 64]
```

“Tensor” describes the multidimensional structure. “Vector,” “matrix,” and “linear transformation” add mathematical meaning.

## Lesson 0.3.2 — Vector magnitude and L2 norm

A vector has direction and magnitude. For:

```text
v = [3, 4]
```

the L2 norm is:

```text
||v||₂ = √(3² + 4²) = 5
```

In general:

```text
||v||₂ = √(v₁² + v₂² + ... + vₙ²)
```

Norms matter for:

- similarity and distance
- vector normalization
- regularization
- gradient clipping

The L2 norm is the main norm to know for this lesson. The L1 norm is the sum of absolute values, and the Frobenius norm is an L2-like norm over all entries of a matrix.

Euclidean distance between two vectors is the norm of their difference:

```text
distance(a, b) = ||a - b||₂
```

## Lesson 0.3.3 — Dot product

For:

```text
a = [1, 2]
b = [3, 4]
```

the dot product is:

```text
a · b = 1×3 + 2×4 = 11
```

Geometrically:

```text
a · b = ||a|| ||b|| cos(θ)
```

Interpretation:

- large positive: broadly aligned directions
- near zero: roughly orthogonal
- negative: broadly opposite directions

The dot product depends on both direction and vector magnitude. It can also be viewed as a weighted sum, where one vector provides values and the other provides weights.

In attention, `Query · Key` becomes a compatibility score.

## Lesson 0.3.4 — Cosine similarity

Cosine similarity removes vector magnitude and compares direction:

```text
cosine(a, b) = (a · b) / (||a|| ||b||)
```

For nonzero vectors, its range is:

```text
-1  → opposite direction
 0  → orthogonal
 1  → same direction
```

If both vectors are L2-normalized, cosine similarity equals their dot product:

```text
||a|| = ||b|| = 1  ⇒  cosine(a, b) = a · b
```

Embedding vectors for “car” and “automobile” may point in similar directions even though the words differ. This makes cosine similarity useful in semantic search, vector databases, and RAG.

Cosine similarity is undefined for a zero vector, so implementations must prevent or handle zero norms.

## Lesson 0.3.5 — Matrix multiplication as transformation

Do not think of matrix multiplication as arithmetic alone. A matrix maps vectors from one feature space into another.

Suppose:

```text
x = [768]
W = [768, 512]
```

Then:

```text
xW = [512]
```

Every output coordinate is a weighted sum of the 768 inputs. Depending on the matrix, a transformation can rotate, reflect, stretch, compress, expand, mix, or project directions.

For a batch:

```text
XW: [batch, 768] @ [768, 512] → [batch, 512]
```

The inner dimensions must match. Deep-learning libraries may store weights using a transposed convention, but the same dimensional reasoning applies.

## Lesson 0.3.6 — Linear layers

A neural-network linear layer is:

```text
y = xW + b
```

For example:

```text
x = [batch, 768]
W = [768, 256]
b = [256]
y = [batch, 256]
```

`W` performs the learned linear transformation. Broadcasting adds the learned offset `b` to every sample.

Strictly, `xW + b` is an affine transformation because of the bias. Deep-learning libraries conventionally call it a linear layer.

Transformers are built largely from repeated:

- matrix multiplication
- addition
- normalization
- nonlinear activation

## Lesson 0.3.7 — Projection

In transformer code, a projection usually means a learned linear mapping into another feature space.

The same token representation is mapped using three learned matrices:

```text
Q = XWq
K = XWk
V = XWv
```

These query, key, and value projections serve different roles:

- query: what a token is looking for
- key: what a token can be matched against
- value: what information a token contributes

The word “projection” also has a narrower geometric meaning. The orthogonal projection of `x` onto nonzero vector `u` is:

```text
projᵤ(x) = (x · u / u · u)u
```

You need the learned-mapping intuition for transformers; geometric projection proofs are unnecessary here.

## Lesson 0.3.8 — Basis and dimensions

A basis is a set of independent directions that can construct every vector in a space. Coordinates describe a vector relative to a basis.

```text
[0.2, -1.1, 0.7]  → a vector in a 3-dimensional space
[768]              → an embedding in a 768-dimensional space
```

You do not need to interpret each learned embedding dimension separately. Meaning is distributed across many dimensions, unlike a handcrafted feature vector where one coordinate may explicitly mean height and another width.

A learned linear layer can change the coordinate system and dimensionality. Reducing dimensions may discard information if the representation does not lie in a sufficiently lower-dimensional subspace.

## Lesson 0.3.9 — Orthogonality

Two vectors are orthogonal when:

```text
a · b = 0
```

For nonzero vectors under the standard Euclidean inner product, they meet at 90 degrees.

In high-dimensional embedding spaces, near-orthogonality can indicate low directional similarity. It does not always mean “unrelated” in every learned model; interpretation depends on how the model was trained and how similarity is calibrated.

## Lesson 0.3.10 — Transpose

Transpose swaps matrix rows and columns:

```text
A  = [N, D]
Aᵀ = [D, N]
```

Useful identities:

```text
(AB)ᵀ = BᵀAᵀ
(Aᵀ)ᵀ = A
```

Attention computes `QKᵀ`. If:

```text
Q  = [N, D]
K  = [N, D]
Kᵀ = [D, N]
```

then:

```text
[N, D] @ [D, N] = [N, N]
```

This computes every query token's dot product with every key token.

## Lesson 0.3.11 — Rank

For this roadmap, think of rank as the amount of independent directional information in a matrix.

A matrix may be large but contain redundant structure. If:

```text
row₂ = 2 × row₁
```

the second row does not introduce a new independent direction.

For a matrix shaped `[m, n]`:

```text
rank(A) ≤ min(m, n)
```

You do not need to calculate rank by hand. The concept matters later for low-rank adaptation, LoRA, dimensionality reduction, and model compression.

## Lesson 0.3.12 — Low-rank approximation

A large matrix can sometimes be approximated with two thinner matrices.

Instead of learning a full update:

```text
ΔW = [4096, 4096]
```

represent it as:

```text
A = [4096, r]
B = [r, 4096]
r << 4096

ΔW ≈ AB
```

The product has rank at most `r` and uses:

```text
4096r + r4096
```

parameters instead of:

```text
4096 × 4096
```

This is the central parameter-saving intuition behind LoRA. The later fine-tuning module will cover how the update is applied and trained.

## Lesson 0.3.13 — Eigenvalues and eigenvectors

Only conceptual awareness is required.

An eigenvector is a nonzero direction that a square transformation does not rotate away from itself:

```text
Av = λv
```

- `v` is the eigenvector
- `λ` is the eigenvalue

The eigenvalue tells how much that direction is scaled, including a possible direction reversal when negative.

This is useful background for PCA, dimensionality reduction, repeated transformations, stability, covariance structure, and optimization. It is not a transformer prerequisite worth deriving by hand.

## Lesson 0.3.14 — Singular value decomposition

Singular value decomposition factors any matrix as:

```text
A = UΣVᵀ
```

Intuitively:

- columns of `V` identify important input directions
- singular values in `Σ` give the strength of those directions
- columns of `U` identify corresponding output directions

SVD works for rectangular and rank-deficient matrices. Keeping only the largest `k` singular values produces a rank-`k` approximation that preserves the strongest structure.

Relevant uses include:

- compression
- dimensionality reduction
- denoising
- low-rank approximation
- analysis of representations and weight matrices
- LoRA intuition

Do not spend Month 0 deriving SVD.

## The four concepts to know cold

Spend most of the 0.3 review on:

1. **Dot product** — measures alignment scaled by magnitude.
2. **Cosine similarity** — compares direction after removing magnitude.
3. **Matrix multiplication** — transforms representations through learned weighted combinations.
4. **Projection** — maps the same representation into another learned feature space.

Everything else supports these four ideas.

## How this connects to transformers

Start with a token embedding:

```text
x = [768]
```

Project it into three roles:

```text
q = xWq
k = xWk
v = xWv
```

Compare a query and key:

```text
q · k
```

Across every token:

```text
QKᵀ = [N, N]
```

After scaling, masking, and row-wise softmax:

```text
P = softmax(QKᵀ / √D)
```

Use the weights to combine values:

```text
PV: [N, N] @ [N, Dv] → [N, Dv]
```

The core flow is:

```text
linear projection
      ↓
dot-product compatibility
      ↓
normalization
      ↓
weighted combination of value vectors
```

The `√D` scaling keeps dot-product magnitudes from growing excessively as dimension increases, which helps prevent softmax saturation.

## Useful extensions

These concepts were retained from the earlier repository version because they strengthen the main lesson.

### Inverses and solving systems

If square matrix `A` is invertible:

```text
A⁻¹A = I
```

Conceptually, an inverse undoes a transformation. In numerical code, solve `Ax = b` with a routine such as `torch.linalg.solve(A, b)` instead of explicitly forming `A⁻¹b`; direct solvers are normally faster and more stable.

Rectangular and rank-deficient matrices do not have ordinary inverses. A pseudoinverse provides a least-squares-related generalization.

### Embeddings as matrix lookup

An embedding table with vocabulary size `V` and embedding dimension `D` is:

```text
E = [V, D]
```

Token IDs shaped `[B, N]` select rows and produce:

```text
[B, N] → [B, N, D]
```

One-hot multiplication is mathematically equivalent but wasteful:

```text
one_hot(token) @ E = E[token]
```

### Shape-first debugging

Write intended shapes before inspecting numerical values:

```text
X:       [batch, sequence, model_dim]
Wq:      [model_dim, heads × head_dim]
Q:       [batch, sequence, heads × head_dim]
Q split: [batch, heads, sequence, head_dim]
scores:  [batch, heads, sequence, sequence]
```

For each matrix multiplication, verify matching inner dimensions and interpret every remaining output dimension.

## Exit test

Answer without notes:

1. What is a vector?
2. What does the L2 norm represent?
3. What does a dot product tell us?
4. Why is cosine similarity different from dot product?
5. What does a matrix do geometrically?
6. Why is a linear layer `xW + b`?
7. What is a projection?
8. Why does attention create separate Q, K, and V projections?
9. Why does `QKᵀ` create an `N × N` matrix?
10. What does low-rank mean?
11. Why is low-rank approximation useful for LoRA?
12. At a high level, what are eigenvectors and SVD?

<details>
<summary>Show answers</summary>

### Answers

1. **What is a vector?**
   A vector is an ordered list of numbers representing a point, direction, or learned representation in a feature space. An embedding shaped `[768]` is a vector.

2. **What does the L2 norm represent?**
   The L2 norm is the magnitude or length of a vector:

   ```text
   ||v||₂ = √(v₁² + v₂² + ... + vₙ²)
   ```

   It tells how large the vector is without specifying its direction.

3. **What does a dot product tell us?**
   The dot product measures alignment scaled by vector magnitudes:

   ```text
   a · b = ||a|| ||b|| cos(θ)
   ```

   Large positive values indicate alignment, values near zero indicate near-orthogonality, and negative values indicate opposing directions. In attention, `Query · Key` is a compatibility score.

4. **Why is cosine similarity different from dot product?**
   Dot product depends on direction and magnitude. Cosine similarity divides by both norms and compares direction:

   ```text
   cosine(a, b) = (a · b) / (||a|| ||b||)
   ```

5. **What does a matrix do geometrically?**
   A matrix transforms vectors. It can rotate, scale, combine, reflect, project, or change dimensionality. For example, `[768] @ [768, 512] → [512]` maps a 768-dimensional representation into a 512-dimensional one.

6. **Why is a linear layer `xW + b`?**
   `W` performs the learned linear transformation and `b` adds a learned offset. Training learns both `W` and `b`.

7. **What is a projection?**
   In a transformer, a projection is a learned mapping of a representation into another feature space:

   ```text
   Q = XWq
   K = XWk
   V = XWv
   ```

8. **Why does attention create separate Q, K, and V projections?**
   Each has a different learned role: the query represents what a token seeks, the key represents what it can match, and the value represents the information it contributes.

9. **Why does `QKᵀ` create an `N × N` matrix?**
   If `Q = [N, D]` and `K = [N, D]`, then `Kᵀ = [D, N]`, so:

   ```text
   [N, D] @ [D, N] = [N, N]
   ```

   Entry `(i, j)` measures how strongly token `i`'s query matches token `j`'s key.

10. **What does low-rank mean?**
    A low-rank matrix contains fewer independent directions than its full dimensions allow. Much of its structure can be represented with a smaller number of factors.

11. **Why is low-rank approximation useful for LoRA?**
    Instead of training a full update `ΔW = [4096, 4096]`, LoRA learns `A = [4096, r]` and `B = [r, 4096]` for small `r`, then uses `ΔW ≈ AB`. This dramatically reduces trainable parameters while allowing useful adaptation.

12. **At a high level, what are eigenvectors and SVD?**
    An eigenvector is a direction that a square transformation preserves and only scales: `Av = λv`. SVD factors a matrix as `A = UΣVᵀ`, exposing important directions and their strengths so dimensionality reduction and low-rank approximation become possible.

</details>

## Completion criteria

0.3 is complete when:

- dot product, cosine similarity, matrix transformation, and projection feel intuitive
- you can reason about `xW + b`, `QKᵀ`, and `PV` by both meaning and shape
- you can explain orthogonality, rank, low-rank approximation, eigenvectors, and SVD at a conceptual level
- you can give the LoRA parameter-saving intuition
- you can answer the 12 exit questions without notes

## Primary references

- [Deep Learning — Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)
- [Dive into Deep Learning — Linear Algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html)
- [PyTorch Linear Algebra](https://docs.pytorch.org/docs/stable/linalg.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
