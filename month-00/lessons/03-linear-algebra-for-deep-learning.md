# 0.3 — Linear Algebra for Deep Learning

The goal is practical geometric intuition for embeddings, attention, transformers, and neural networks. Proof-heavy mathematics and hand calculation are outside this prerequisite review.

| | |
|---|---|
| **Mode** | SKIM / REFRESH |
| **Time** | 60–90 minutes; add about 45 minutes for the hands-on exercises |
| **Assumes** | 0.2 NumPy and Tensor Manipulation |
| **Used by** | 0.4 Probability (softmax, covariance), 0.10 Representation Learning, Month 1 attention, Month 2 LoRA, Month 4 vector search |

## Learning objectives

After this lesson you can:

- explain dot product, cosine similarity, and Euclidean distance geometrically, and choose among them for embeddings
- read `y = xW + b`, `QKᵀ`, and `softmax(QKᵀ / √d_k)V` by both meaning and shape
- explain why depth needs nonlinearity, why attention scales by `√d_k`, and how logits arise from dot products
- describe rank, low-rank factorization, eigenvectors, and SVD conceptually, and connect them to LoRA and compression
- check each of these ideas numerically with a few lines of PyTorch

## How to use this lesson

1. **Diagnose first.** Answer the [exit test](#exit-test) without notes.
2. **Read selectively.** Study the lessons behind any miss, plus every **Pitfall** callout.
3. **Verify numerically.** The [hands-on exercises](#hands-on-exercises) turn each key claim into an assertion you can run.
4. **Record gaps** in [`progress.md`](../progress.md).

## Notation and conventions

This lesson uses the row-vector convention common in deep-learning code:

```text
x: [D_in]                 one row vector
X: [N, D_in]              N row vectors stacked
W: [D_in, D_out]
y = xW + b: [D_out]
```

Two other conventions appear in references:

- Mathematics texts usually use column vectors and write `y = Wx + b`, with `W: [D_out, D_in]`.
- PyTorch's `nn.Linear` stores `weight` as `[D_out, D_in]` and computes `x @ weight.T + bias`.

All three describe the same transformation. When reading a formula, first establish its convention, then check that inner dimensions match.

Code examples assume:

```python
import math
import torch
import torch.nn.functional as F
from torch import nn
```

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

```python
v = torch.tensor([3.0, 4.0])
torch.linalg.vector_norm(v)   # tensor(5.)
F.normalize(v, dim=-1)        # tensor([0.6000, 0.8000]): same direction, unit length
```

Norms also sit inside normalization layers. RMSNorm, used in many recent LLMs, divides each vector by its root-mean-square value, `||x||₂ / √D`, then applies a learned per-feature scale. LayerNorm also subtracts the mean first. Both return in 0.6.

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

Dot products grow with dimension. If the entries of `q` and `k` are independent with mean 0 and variance 1, then `q · k` is a sum of `d` such products, so it has mean 0 and variance `d`; its typical magnitude is about `√d`. This is why attention divides scores by `√d_k` (see [How this connects to transformers](#how-this-connects-to-transformers)).

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

For unit-length vectors, Euclidean distance carries the same ranking information as cosine similarity:

```text
||a − b||² = ||a||² + ||b||² − 2(a · b) = 2 − 2 cosine(a, b)
```

On normalized embeddings, nearest neighbors by distance and by cosine similarity are therefore identical. On unnormalized embeddings, dot product, cosine similarity, and distance can rank the same candidates differently.

> **Pitfall:** Use the similarity measure an embedding model was trained with. Many text-embedding models are trained with cosine similarity on normalized vectors; indexing their raw outputs by unnormalized dot product or distance can silently degrade retrieval. This choice returns in Month 4.

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

The inner dimensions must match. PyTorch stores this weight transposed; see [Notation and conventions](#notation-and-conventions).

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

In PyTorch:

```python
layer = nn.Linear(768, 256)
x = torch.randn(32, 768)
y = layer(x)                                                   # [32, 256]
assert torch.allclose(y, x @ layer.weight.T + layer.bias, atol=1e-5)
```

### Why depth needs nonlinearity

Composing linear transformations produces another linear transformation:

```text
(xW₁)W₂ = x(W₁W₂)
```

Two stacked linear layers, `768 → 3072 → 768`, can therefore represent nothing that a single `768 → 768` matrix cannot. With biases, the composition is still one affine map. Inserting a nonlinear activation between the layers breaks this collapse, which is what lets depth represent more complex functions. Classic transformer feed-forward blocks are `Linear → activation → Linear`, using GELU in GPT-2 and BERT and gated variants such as SwiGLU in many recent LLMs.

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

These roles are helpful intuition, not instructions the model receives; training determines what each projection actually encodes. Multi-head attention also ends with an **output projection** `W_o` that maps the concatenated heads back to the model dimension.

The word “projection” also has a narrower geometric meaning. The orthogonal projection of `x` onto nonzero vector `u` is:

```text
projᵤ(x) = (x · u / u · u)u
```

You need the learned-mapping intuition for transformers; geometric projection proofs are unnecessary here.

## Lesson 0.3.8 — Basis and dimensions

A set of vectors is **linearly independent** when none of them can be written as a weighted sum of the others. Their **span** is the set of all vectors those weighted sums can reach. A **basis** is a linearly independent set that spans the whole space, and the **dimension** is the number of vectors in any basis. Coordinates describe a vector relative to a basis.

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

High dimensions make near-orthogonality the default. Two random vectors in 768 dimensions have cosine similarity close to zero, with a typical magnitude around `1/√768 ≈ 0.036`:

```python
a = torch.randn(10_000, 768)
b = torch.randn(10_000, 768)
cos = F.cosine_similarity(a, b, dim=-1)
print(cos.mean(), cos.std())   # about 0.000 and 0.036
```

This is one reason a large space can hold many nearly independent directions. It is also why raw similarity scores need a baseline: a cosine similarity of 0.3 can be a strong match in one embedding model and near noise in another.

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

Rank also bounds products:

```text
rank(AB) ≤ min(rank(A), rank(B))
```

A product that passes through an `r`-dimensional bottleneck therefore has rank at most `r`. The next lesson relies on exactly this fact.

You do not need to calculate rank by hand; `torch.linalg.matrix_rank` estimates it numerically from singular values. The concept matters later for low-rank adaptation (LoRA), dimensionality reduction, and model compression.

## Lesson 0.3.12 — Low-rank approximation

A large matrix can sometimes be approximated, or deliberately constrained, by the product of two thinner matrices.

Consider fine-tuning a pretrained weight `W₀: [4096, 4096]`. Instead of learning a full update:

```text
ΔW: [4096, 4096]          16,777,216 parameters
```

LoRA learns two factors. Using the LoRA paper's notation and column-vector convention, with `W₀: [d, k]`:

```text
B: [d, r] = [4096, r]
A: [r, k] = [r, 4096]
r << 4096

ΔW = BA                   rank at most r
W  = W₀ + BA
```

The factors hold `4096·r + r·4096 = 8192·r` parameters. With `r = 8`, that is 65,536 parameters, about 0.39% of the full update.

Two distinctions are easy to blur:

- **Approximation versus constraint.** Truncated SVD (0.3.14) finds the best low-rank *approximation* of a matrix you already have. LoRA does not approximate a known `ΔW`. It *constrains* the learned update to rank `r` and trains `B` and `A` directly, relying on the empirical finding that useful fine-tuning updates have low intrinsic rank.
- **The pretrained weight is frozen.** Only `A` and `B` receive gradients. `B` is initialized to zero, so `W = W₀` when training starts.

Month 2 covers where LoRA adapters are placed, their scaling factor, and how they are merged back into the base weights.

## Lesson 0.3.13 — Eigenvalues and eigenvectors

Only conceptual awareness is required.

An eigenvector is a nonzero direction that a square transformation does not rotate away from itself:

```text
Av = λv
```

- `v` is the eigenvector
- `λ` is the eigenvalue

The eigenvalue tells how much that direction is scaled, including a possible direction reversal when negative.

Two facts explain where eigenvectors are useful:

- Not every real matrix has real eigenvectors. A 2D rotation by 90° moves every direction, so its eigenvalues are complex.
- A **symmetric** matrix, such as a covariance matrix, always has real eigenvalues and a full set of mutually orthogonal eigenvectors. PCA relies on this: the eigenvectors of the data's covariance matrix are the directions of greatest variance, ordered by eigenvalue. Use `torch.linalg.eigh` for symmetric matrices.

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

More precisely, `U` and `V` have orthonormal columns, and the singular values on the diagonal of `Σ` are nonnegative and sorted from largest to smallest. The singular values of `A` are the square roots of the eigenvalues of `AᵀA`, which links SVD to 0.3.13.

The **Eckart–Young theorem** makes low-rank approximation precise. Keeping the top `k` singular values gives the best rank-`k` approximation of `A` in both the Frobenius and spectral norms, and the Frobenius error equals the square root of the sum of the discarded squared singular values:

```python
A = torch.randn(512, 256)
U, S, Vh = torch.linalg.svd(A, full_matrices=False)    # Vh is Vᵀ
k = 32
A_k = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]

error = torch.linalg.matrix_norm(A - A_k)              # Frobenius norm
assert torch.allclose(error, S[k:].pow(2).sum().sqrt(), rtol=1e-4)
```

PCA is SVD applied to mean-centered data: the right singular vectors are the principal directions.

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
P = softmax(QKᵀ / √d_k)
```

Here `d_k` is the query and key dimension of one attention head. In a 768-dimensional model with 12 heads, `d_k = 64`, not 768.

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

The `√d_k` scaling cancels the `√d` growth in dot-product magnitude described in 0.3.3. Without it, large scores push softmax toward a nearly one-hot distribution, where gradients become vanishingly small; this is softmax saturation.

Together, one attention head computes:

```text
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

Multi-head attention runs `H` heads in parallel on smaller projections, concatenates their outputs, and applies the output projection `W_o`. Lesson 0.2 traces the exact tensor shapes.

### From hidden states to next-token logits

The last step of a language model is also a dot product. A final hidden state `h: [D]` is compared with every row of an output matrix `W_out: [V, D]`:

```text
logits = h W_outᵀ:   [D] @ [D, V] → [V]
```

Logit `i` is the dot product between the hidden state and token `i`'s output vector, so the model scores highly the tokens whose vectors align with its current representation. Many models *tie* `W_out` to the input embedding table `E: [V, D]`, reusing one matrix for both lookup and prediction. Softmax then turns the logits into a next-token distribution (0.4).

## Useful extensions

These concepts are not required for the exit test, but they come up regularly in practice.

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

## Hands-on exercises

Turn each key claim into an assertion. Call `torch.manual_seed(0)` first for reproducible results.

### Exercise 1 — Similarity measures

Create five random embeddings shaped `[5, 64]`. Compute the full `[5, 5]` cosine-similarity matrix with a single matrix multiplication on normalized vectors, and verify it against `F.cosine_similarity`. Then verify that `||a − b||² = 2 − 2 cosine(a, b)` for normalized vectors.

### Exercise 2 — Why attention scales by √d_k

For `d` in `[16, 64, 256, 1024]`, sample 32,000 pairs of random vectors and measure the standard deviation of their dot products. Then group the dot products into 1,000 rows of 32 attention scores, and compare the average largest softmax weight per row with and without dividing by `√d`.

### Exercise 3 — Stacked linear layers collapse

Build `nn.Sequential(nn.Linear(16, 64), nn.Linear(64, 16))`. Compute a single equivalent weight matrix and bias, and verify that this one affine map reproduces the model's outputs. Explain why inserting `nn.GELU()` between the layers breaks the equivalence.

### Exercise 4 — Low-rank approximation and LoRA arithmetic

Create a `[256, 256]` matrix of rank 8 plus small noise. Print its leading singular values, reconstruct it at ranks 4, 8, and 32, and confirm the Eckart–Young error formula at each rank. Then compute LoRA's trainable parameters for `d = k = 4096` at ranks 4, 8, 16, and 64 as a percentage of a full update.

### Exercise 5 — Embedding lookup and logits

With an embedding table `E: [100, 16]` and token IDs `[3, 17, 42]`, verify that `E[ids]` equals `one_hot(ids) @ E`. Then compute tied next-token logits for a hidden state `h: [16]` as `E @ h`, and confirm that the argmax is the token whose embedding has the largest dot product with `h`.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
torch.manual_seed(0)
X = torch.randn(5, 64)
Xn = F.normalize(X, dim=-1)

cos_matrix = Xn @ Xn.T                                                  # [5, 5]
reference = F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1)   # broadcast to [5, 5]
assert torch.allclose(cos_matrix, reference, atol=1e-6)

a, b = Xn[0], Xn[1]
assert torch.allclose((a - b).pow(2).sum(), 2 - 2 * torch.dot(a, b), atol=1e-6)
```

The diagonal of `cos_matrix` is 1 because every vector points in its own direction.

### Exercise 2

```python
torch.manual_seed(0)
for d in [16, 64, 256, 1024]:
    q = torch.randn(32_000, d)
    k = torch.randn(32_000, d)
    dots = (q * k).sum(dim=-1)                  # [32000]
    rows = dots.reshape(1_000, 32)              # 1,000 rows of 32 attention scores

    raw_max = torch.softmax(rows, dim=-1).max(dim=-1).values.mean().item()
    scaled_max = torch.softmax(rows / math.sqrt(d), dim=-1).max(dim=-1).values.mean().item()
    print(f"d={d:4d}  std={dots.std().item():6.2f}  √d={math.sqrt(d):5.2f}  "
          f"mean max weight: raw={raw_max:.2f}  scaled={scaled_max:.2f}")
```

The standard deviation tracks `√d` (about 4, 8, 16, and 32). Without scaling, the average largest weight climbs from about 0.66 to 0.96 as `d` grows, so one key takes almost all of the attention and gradients through the other positions vanish. With scaling, it stays near 0.17 at every dimension.

### Exercise 3

```python
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(16, 64), nn.Linear(64, 16))
first, second = model

W = second.weight @ first.weight                  # [16, 16] in PyTorch's [out, in] convention
b = second.weight @ first.bias + second.bias      # [16]

x = torch.randn(10, 16)
assert torch.allclose(model(x), x @ W.T + b, atol=1e-5)
```

`GELU` is applied element-wise and is not linear, so `second(gelu(first(x)))` cannot be rewritten as a single `x @ W.T + b` for all inputs.

### Exercise 4

```python
torch.manual_seed(0)
low_rank = torch.randn(256, 8) @ torch.randn(8, 256)    # rank 8 by construction
A = low_rank + 0.01 * torch.randn(256, 256)

U, S, Vh = torch.linalg.svd(A)
print(S[:10])   # eight large values, then a sharp drop to the noise level

total = torch.linalg.matrix_norm(A)
for rank in [4, 8, 32]:
    A_r = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
    error = torch.linalg.matrix_norm(A - A_r)
    assert torch.allclose(error, S[rank:].pow(2).sum().sqrt(), rtol=1e-3)
    print(f"rank {rank:2d}: relative error {(error / total).item():.4f}")

full = 4096 * 4096
for r in [4, 8, 16, 64]:
    lora = 4096 * r + r * 4096
    print(f"r={r:2d}: {lora:,} parameters = {100 * lora / full:.2f}% of a full update")
```

Rank 4 leaves a large error (about 64%), rank 8 captures almost everything (about 0.35%), and rank 32 improves only slightly because the extra components fit noise. LoRA uses about 0.20%, 0.39%, 0.78%, and 3.12% of a full update's parameters at ranks 4, 8, 16, and 64.

### Exercise 5

```python
torch.manual_seed(0)
E = torch.randn(100, 16)
ids = torch.tensor([3, 17, 42])

one_hot = F.one_hot(ids, num_classes=100).float()    # [3, 100]
assert torch.allclose(one_hot @ E, E[ids])           # [3, 16]

h = torch.randn(16)
logits = E @ h                                       # [100]: one dot product per token
best = max(range(100), key=lambda i: torch.dot(E[i], h).item())
assert logits.argmax().item() == best
```

</details>

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
    Instead of training a full update `ΔW: [4096, 4096]` (about 16.8 million parameters), LoRA freezes the pretrained weight and trains `B: [4096, r]` and `A: [r, 4096]`, with `ΔW = BA`. At `r = 8` that is 65,536 parameters, under 0.4% of the full update. This works because useful fine-tuning updates tend to have low intrinsic rank.

12. **At a high level, what are eigenvectors and SVD?**
    An eigenvector is a direction that a square transformation preserves and only scales: `Av = λv`. SVD factors a matrix as `A = UΣVᵀ`, exposing important directions and their strengths so dimensionality reduction and low-rank approximation become possible.

</details>

### Extended check

These questions go beyond the original 12 and cover the material added to this lesson.

13. When do cosine similarity and Euclidean distance produce the same nearest neighbors?
14. Why does attention divide scores by `√d_k`, and what is `d_k` in a 12-head, 768-dimensional model?
15. Why can a stack of linear layers without activations represent nothing more than a single linear layer?
16. How does LoRA's low-rank update differ from a truncated-SVD approximation?
17. How are a language model's next-token logits computed from dot products?

<details>
<summary>Show answers</summary>

13. When all vectors are L2-normalized. Then `||a − b||² = 2 − 2 cosine(a, b)`, so a smaller distance always means a higher cosine similarity.
14. With independent unit-variance entries, a dot product of `d`-dimensional vectors has standard deviation about `√d`. Dividing by `√d_k` keeps scores in a range where softmax does not saturate. With 12 heads over 768 dimensions, `d_k = 64`.
15. Because `(xW₁)W₂ = x(W₁W₂)`: the composition is a single matrix, and with biases a single affine map. A nonlinearity between layers prevents this collapse.
16. Truncated SVD finds the best rank-`k` approximation of an existing matrix. LoRA never computes a full `ΔW`; it constrains the update to `BA` with rank at most `r` and learns `B` and `A` by gradient descent while the pretrained weight stays frozen.
17. The final hidden state is multiplied by an output matrix with one row per vocabulary token, so each logit is the dot product between the hidden state and that token's vector. With weight tying, those vectors are the input embeddings.

</details>

## Completion criteria

0.3 is complete when:

- dot product, cosine similarity, matrix transformation, and projection feel intuitive
- you can reason about `xW + b`, `QKᵀ`, and `PV` by both meaning and shape
- you can explain orthogonality, rank, low-rank approximation, eigenvectors, and SVD at a conceptual level
- you can give the LoRA parameter-saving intuition and distinguish LoRA's rank constraint from SVD approximation
- you can explain the `√d_k` scaling, the need for nonlinearity, and logits as dot products
- your solutions to the five hands-on exercises run with passing assertions
- you can answer the 12 exit questions without notes

## Primary references

- [Deep Learning — Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)
- [Dive into Deep Learning — Linear Algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html)
- [PyTorch Linear Algebra](https://docs.pytorch.org/docs/stable/linalg.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Mathematics for Machine Learning](https://mml-book.github.io/) — chapters 2–4 for deeper background
- [MIT 18.06 Linear Algebra (Gilbert Strang)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — optional depth on rank, eigenvectors, and SVD
- [PyTorch: `torch.linalg.svd`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html)

## About this lesson

Adapted from the roadmap planning conversation, [Create Month Zero Learning List](https://chatgpt.com/share/6aa28c17-a29c-83ea-ae5b-9202ec5ba987), and an earlier repository version. The original 14 lessons, transformer connection, and 12-question exit test are preserved. Notation conventions, runnable checks, corrected LoRA notation and attention scaling, the logits connection, hands-on exercises, and the extended check were added. Code examples were checked with PyTorch 2.14 on CPU.
