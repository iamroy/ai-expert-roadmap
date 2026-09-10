# 0.2 — NumPy and Tensor Manipulation

The goal is tensor fluency: look at a shape and understand what every dimension represents, predict the result of an operation, and manipulate transformer tensors without trial and error.

This file preserves all 18 lessons, the transformer shape walkthrough, five hands-on exercises, and the 12-question exit test from the roadmap conversation.

**Source conversation:** [Create Month Zero Learning List](https://chatgpt.com/share/6aa28c17-a29c-83ea-ae5b-9202ec5ba987)

**Highest-priority ideas:** shapes, reshape/permute, broadcasting, matrix multiplication, reductions, and masking.

```python
import numpy as np
import torch
```

## Lesson 0.2.1 — Shapes, dimensions, and axes

A scalar has no axes, a vector has one, a matrix has two, and higher-dimensional arrays are commonly called tensors.

```text
scalar:     []
vector:     [D]
matrix:     [N, D]
3D tensor:  [B, N, D]
4D tensor:  [B, H, N, D]
```

In NumPy and PyTorch:

```python
scalar = torch.tensor(3.0)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.zeros(2, 3)
x = torch.zeros(32, 128, 768)

print(x.shape)   # torch.Size([32, 128, 768])
print(x.ndim)    # 3
print(x.numel()) # 32 * 128 * 768
```

Interpret `x.shape == [32, 128, 768]` as:

```text
32   = batch size
128  = sequence length
768  = embedding dimension
```

- **shape** gives the length of every axis
- **ndim** gives the number of axes
- **size** can mean the length of one dimension; in PyTorch, `x.size()` returns the full shape
- **numel** gives the total number of elements

Shape alone does not record meaning. Two tensors can have the same shape and represent different concepts, so name tensors and document axes clearly.

## Lesson 0.2.2 — Indexing and slicing

Indexing selects parts of a tensor.

```python
x = torch.randn(32, 128, 768)

x[0]          # [128, 768]: first batch item
x[:, 0]       # [32, 768]: first token from every batch item
x[:, :, 0]    # [32, 128]: first feature
x[0:4]        # [4, 128, 768]
x[:, 10:20]   # [32, 10, 768]
x[:, 0, :]    # [32, 768]
```

Integer indexing removes an axis, while slicing preserves it:

```python
x[0].shape    # [128, 768]
x[0:1].shape  # [1, 128, 768]
```

Also recognize:

```python
x[-1]             # last batch item
x[..., -1]        # last feature; ... fills omitted leading axes
x[:, :, [0, 2]]   # select specific features
x[x > 0]          # Boolean indexing; returns matching values
```

For a transformer tensor `[B, N, D]`, `x[:, 0, :]` selects the first-token representation from every sample.

## Lesson 0.2.3 — Reshape, view, and flatten

These operations change tensor organization without changing the total number of elements.

```python
x = torch.randn(32, 128, 768)
flat_tokens = x.reshape(32 * 128, 768)

assert flat_tokens.shape == (4096, 768)
assert x.numel() == flat_tokens.numel()
```

The critical rule is:

```text
product of old dimensions = product of new dimensions
```

`reshape` may return a view or copy data when needed. PyTorch's `view` requires a compatible memory layout:

```python
y = x.transpose(1, 2)          # often non-contiguous
z = y.contiguous().view(32, -1)
```

`flatten` combines a range of dimensions:

```python
x.flatten(start_dim=1).shape   # [32, 128 * 768]
```

Never reshape only to silence an error. First identify what every axis means.

## Lesson 0.2.4 — `-1` shape inference

Use one `-1` to ask NumPy or PyTorch to infer the missing dimension:

```python
x = torch.randn(32, 128, 768)

x.reshape(-1, 768).shape       # [4096, 768]
x.reshape(32, 128, -1).shape   # [32, 128, 768]
```

Only one dimension can be inferred. The known dimensions and total element count must determine it exactly.

This pattern appears constantly when splitting and merging attention heads.

## Lesson 0.2.5 — Transpose and permute

Transpose and permutation reorder axes.

```python
matrix = torch.randn(4, 8)
matrix.T.shape                  # [8, 4]
matrix.transpose(0, 1).shape   # [8, 4]
```

For higher-dimensional tensors:

```python
x = torch.randn(2, 3, 4)

x.transpose(1, 2).shape        # [2, 4, 3]: swap two axes
x.permute(0, 2, 1).shape      # [2, 4, 3]: specify every axis
x.mT.shape                     # [2, 4, 3]: swap final two axes
```

For transformer representations:

```text
[B, N, D] → [B, D, N]
```

and, after splitting the embedding dimension:

```text
[B, N, H, Dh] → [B, H, N, Dh]
```

`reshape` changes how elements are grouped. `permute` changes axis order.

## Lesson 0.2.6 — Broadcasting

Broadcasting lets compatible shapes interact without explicitly copying values.

```python
x = torch.randn(32, 128, 768)
bias = torch.randn(768)
y = x + bias

assert y.shape == (32, 128, 768)
```

Conceptually, the 768-value bias is applied to every token in every batch:

```text
[32, 128, 768]
+          [768]
----------------
[32, 128, 768]
```

Compare dimensions from right to left. Two dimensions are compatible when they are equal or one is `1`. A missing leading dimension behaves like `1`.

```text
[32, 128, 768]
     [128,   1]
----------------
[32, 128, 768]
```

The second example also works: `1` expands over features and the missing leading dimension expands over the batch.

An incompatible example:

```text
[2, 3, 4]
      [3]
```

The final dimensions 4 and 3 conflict.

Broadcasting often avoids copying the smaller input, but the operation can still create a large output. Always predict the result shape.

## Lesson 0.2.7 — Element-wise operations

Ordinary arithmetic is element-wise when shapes are compatible:

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

x + y   # [5, 7, 9]
x - y   # [-3, -3, -3]
x * y   # [4, 10, 18]
x / y
x**2    # [1, 4, 9]
```

`x * y` multiplies corresponding values. It is not matrix multiplication.

Vectorized tensor operations are normally clearer and faster than Python loops:

```python
error = ((predictions - targets) ** 2).mean()
```

## Lesson 0.2.8 — Dot product

For two equal-length vectors:

```text
a = [1, 2, 3]
b = [4, 5, 6]

a · b = 1×4 + 2×5 + 3×6 = 32
```

In NumPy and PyTorch:

```python
np.dot(np.array([1, 2, 3]), np.array([4, 5, 6]))

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
a @ b
torch.dot(a, b)
```

A dot product measures alignment scaled by vector magnitude. Later, a query-key dot product becomes an attention compatibility score.

## Lesson 0.2.9 — Matrix multiplication

Suppose:

```text
X = [N, D]
W = [D, K]
```

Then:

```python
Y = X @ W
```

has shape `[N, K]`. The inner dimensions must match:

```text
[N, D] @ [D, K] → [N, K]
     matching
```

Each output entry is a dot product between one row of `X` and one column of `W`.

A neural-network linear layer is an affine transformation:

```text
Y = XW + b
```

Transformers use the same operation to create queries, keys, and values:

```text
Q = XWq
K = XWk
V = XWv
```

## Lesson 0.2.10 — Batched matrix multiplication

Now let every batch item contain its own matrices:

```text
Q = [B, N, D]
K = [B, N, D]
Kᵀ = [B, D, N]
```

PyTorch can multiply the final two axes for every batch:

```python
scores = torch.matmul(Q, K.transpose(-2, -1))
# equivalently for exactly 3D inputs:
scores = torch.bmm(Q, K.transpose(1, 2))
```

The result is:

```text
[B, N, D] @ [B, D, N] → [B, N, N]
```

That `N × N` matrix contains token-to-token scores for each batch item.

`torch.matmul` supports broadcasting over leading dimensions. `torch.bmm` expects two 3D tensors with the same batch size and does not broadcast.

With attention heads:

```python
Q = torch.randn(8, 12, 32, 64)
K = torch.randn(8, 12, 32, 64)
scores = Q @ K.transpose(-2, -1)

assert scores.shape == (8, 12, 32, 32)
```

## Lesson 0.2.11 — Concatenation versus stacking

Suppose:

```text
a.shape = [4, 768]
b.shape = [4, 768]
```

Concatenation extends an existing axis:

```python
torch.cat([a, b], dim=0).shape  # [8, 768]
torch.cat([a, b], dim=1).shape  # [4, 1536]
```

Stacking creates a new axis:

```python
torch.stack([a, b], dim=0).shape  # [2, 4, 768]
```

For `cat`, every dimension other than the concatenation dimension must match. For `stack`, all input shapes must match.

## Lesson 0.2.12 — `unsqueeze` and `squeeze`

Add a batch axis to one sequence:

```python
x = torch.randn(128, 768)
x = x.unsqueeze(0)

assert x.shape == (1, 128, 768)
```

Remove that size-one axis:

```python
x = x.squeeze(0)
assert x.shape == (128, 768)
```

`unsqueeze(dim)` inserts a new dimension of size 1. `squeeze(dim)` removes that dimension only when its size is 1.

This appears constantly when moving between one sample and a batch. Prefer naming a dimension in `squeeze`; calling `squeeze()` without a dimension can accidentally remove other meaningful size-one axes.

## Lesson 0.2.13 — Reductions

Reductions combine values along one or more dimensions:

```python
x = torch.randn(8, 100, 768)

x.sum()                          # scalar
x.mean()                         # scalar
x.max()                          # scalar
x.min()                          # scalar
x.mean(dim=1).shape              # [8, 768]
x.max(dim=-1).values.shape       # [8, 100]
```

For `[B, N, D]`, `mean(dim=1)` averages across tokens and produces `[B, D]`. This is simple mean pooling.

`keepdim=True` preserves the reduced axis with length 1:

```python
x.mean(dim=1, keepdim=True).shape  # [8, 1, 768]
```

Keeping the axis is often useful for later broadcasting.

## Lesson 0.2.14 — Boolean masks

```python
scores = np.array([0.8, 0.1, 0.9])
mask = scores > 0.5               # [True, False, True]
selected = scores[mask]           # [0.8, 0.9]
```

Transformer masks determine:

- which tokens are real rather than padding
- which positions may attend to one another
- which future tokens a causal model must hide

Attention masks normally preserve the score tensor and replace disallowed entries before softmax:

```python
attention_scores = torch.randn(4, 4)
allowed = torch.tril(torch.ones(4, 4, dtype=torch.bool))
masked_scores = attention_scores.masked_fill(~allowed, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)
```

Disallowed positions receive zero probability after softmax. A row in which every position is masked needs deliberate handling because softmax over all negative infinities is undefined.

## Lesson 0.2.15 — Softmax across a dimension

Suppose attention scores have shape `[B, H, N, N]`:

```python
weights = torch.softmax(scores, dim=-1)
```

`dim=-1` normalizes over the final dimension. For each batch, head, and query token, the weights over all key tokens sum to 1.

Always ask:

> Which axis am I normalizing over?

Applying softmax over the wrong axis can return a tensor with the expected shape while giving it the wrong meaning.

## Lesson 0.2.16 — Numerical stability

Computers have finite precision. Large exponentials can overflow:

```python
np.exp(1000)
```

A naive softmax is unstable:

```text
exp(x) / sum(exp(x))
```

Subtract the largest value before exponentiation:

```python
def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)
```

The probabilities do not change because the same constant is subtracted from every logit and the common exponential factor cancels.

Use library operations such as `torch.softmax`, `torch.logsumexp`, and `torch.nn.functional.cross_entropy` in real code. Mathematically equivalent formulas can behave very differently with finite-precision numbers.

## Lesson 0.2.17 — NumPy arrays versus PyTorch tensors

The APIs and tensor concepts are similar:

```python
numpy_array = np.array([1, 2, 3])
torch_tensor = torch.tensor([1, 2, 3])

numpy_array.reshape(3, 1)
torch_tensor.reshape(3, 1)
```

PyTorch adds deep-learning capabilities:

- GPU and accelerator support
- automatic differentiation
- neural-network layers and optimizers

On CPU, NumPy arrays and PyTorch tensors can share memory:

```python
array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
tensor = torch.from_numpy(array)
back_to_numpy = tensor.numpy()
```

An in-place change through one view can affect the other. Use `.copy()` or `.clone()` for independent storage. Before converting a gradient-tracked tensor to NumPy:

```python
array = tensor.detach().cpu().numpy()
```

For this roadmap, tensor concepts matter more than memorizing NumPy-specific APIs.

## Lesson 0.2.18 — Device and dtype awareness

Every PyTorch tensor has three fundamental properties:

```text
shape + dtype + device
```

```python
token_ids = torch.tensor([12, 5, 91], dtype=torch.int64)
activations = torch.randn(3, 768, dtype=torch.float32)
mask = torch.tensor([True, True, False], dtype=torch.bool)

print(activations.dtype)
print(activations.device)
```

Common types include `float32`, `float16`, `bfloat16`, `int64`, and `bool`. Data type affects precision, memory, and which operations are valid.

Move or cast a tensor with `to`:

```python
x = x.to(device)
x = x.to(torch.bfloat16)
```

Tensors involved in one operation generally need compatible devices and dtypes. Token IDs used for embedding lookup are normally integer tensors, while model activations and parameters are floating point.

When creating a related tensor, preserve properties when appropriate:

```python
bias = torch.zeros_like(activations)
```

## The most important walkthrough — Transformer shape manipulation

Start with token embeddings:

```text
X = [B, N, D] = [32, 128, 768]
```

Suppose there are 12 heads:

```text
H = 12
Dh = D / H = 768 / 12 = 64
```

Split the embedding dimension:

```text
[B, N, D] → [B, N, H, Dh]
[32, 128, 768] → [32, 128, 12, 64]
```

Move the head axis before sequence:

```text
[B, N, H, Dh] → [B, H, N, Dh]
[32, 128, 12, 64] → [32, 12, 128, 64]
```

In PyTorch:

```python
x = torch.randn(32, 128, 768)
q = x.reshape(32, 128, 12, 64).permute(0, 2, 1, 3)

assert q.shape == (32, 12, 128, 64)
```

Queries and keys now have:

```text
Q  = [32, 12, 128, 64]
K  = [32, 12, 128, 64]
Kᵀ = [32, 12, 64, 128]
```

Matrix multiplication produces:

```text
[32, 12, 128, 64] @ [32, 12, 64, 128]
= [32, 12, 128, 128]
```

The final tensor contains a score between every query token and every key token, separately for every attention head and every batch item.

Reverse the head transformation after attention:

```python
context = q.permute(0, 2, 1, 3).contiguous()
context = context.reshape(32, 128, 768)
```

The reshape splits or merges feature axes. The permutation changes axis order. Neither operation computes attention by itself.

## Hands-on exercises

Complete these five exercises in PyTorch.

### Exercise 1 — Shape manipulation

Create:

```python
x = torch.randn(32, 128, 768)
```

Convert it to `[32, 12, 128, 64]` using `reshape` and `permute`. Predict the intermediate shape before running the code.

### Exercise 2 — Batched matrix multiplication

Create `Q` and `K` with shape `[2, 4, 8]`. Calculate `Q @ Kᵀ` and predict the result shape first.

### Exercise 3 — Broadcasting

Given `X = [4, 10, 768]` and `bias = [768]`, calculate `X + bias` and explain why it works.

### Exercise 4 — Reduction

Given `embeddings = [8, 100, 768]`, calculate `embeddings.mean(dim=1)` and explain why the result is `[8, 768]`.

### Exercise 5 — Masking

Create an attention-score matrix, replace disallowed positions with negative infinity, apply softmax over keys, and verify that masked positions receive zero probability.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
x = torch.randn(32, 128, 768)
x = x.reshape(32, 128, 12, 64)
x = x.permute(0, 2, 1, 3)
assert x.shape == (32, 12, 128, 64)
```

The intermediate shape is `[32, 128, 12, 64]`.

### Exercise 2

```python
Q = torch.randn(2, 4, 8)
K = torch.randn(2, 4, 8)
scores = Q @ K.transpose(-2, -1)
assert scores.shape == (2, 4, 4)
```

Each of four queries is compared with four keys.

### Exercise 3

```python
X = torch.randn(4, 10, 768)
bias = torch.randn(768)
result = X + bias
assert result.shape == (4, 10, 768)
```

The final dimension matches; missing leading dimensions expand across batch and sequence.

### Exercise 4

```python
embeddings = torch.randn(8, 100, 768)
pooled = embeddings.mean(dim=1)
assert pooled.shape == (8, 768)
```

The sequence axis is averaged away, leaving one embedding per batch item.

### Exercise 5

```python
scores = torch.randn(2, 4, 4)
allowed = torch.tril(torch.ones(4, 4, dtype=torch.bool))
masked_scores = scores.masked_fill(~allowed, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)

assert torch.all(weights.masked_select(~allowed) == 0)
```

</details>

## Exit test

Answer these quickly without notes:

1. What does `[B, N, D]` mean?
2. What is the difference between `reshape` and `permute`?
3. Why can `[32, 128, 768] + [768]` work?
4. What is the difference between `*` and `@`?
5. What is the shape of `[10, 768] @ [768, 512]`?
6. What is the difference between `cat` and `stack`?
7. What does `unsqueeze(0)` do?
8. What does `mean(dim=1)` do to `[B, N, D]`?
9. Why are masks needed in attention?
10. Why does `Q @ Kᵀ` produce an `N × N` matrix?
11. How do you transform `[B, N, 768]` into 12 attention heads of dimension 64?
12. What do `dtype` and `device` mean?

<details>
<summary>Show answers</summary>

### Answers

1. `B` is batch size, `N` is sequence length, and `D` is the feature or embedding dimension.
2. `reshape` changes how the same elements are grouped into dimensions. `permute` changes the order of the dimensions.
3. Broadcasting aligns dimensions from the right. The final 768 dimensions match, and the missing leading dimensions behave like ones, so the bias applies to every token and batch item.
4. `*` performs element-wise multiplication. `@` performs matrix multiplication.
5. `[10, 512]`; the matching inner dimension 768 is contracted.
6. `cat` extends an existing dimension. `stack` creates a new dimension.
7. It inserts a size-one dimension at axis 0. For example, `[128, 768]` becomes `[1, 128, 768]`.
8. It averages across the `N` sequence positions and returns `[B, D]`.
9. Masks prevent padding, disallowed positions, or future tokens from receiving attention probability.
10. With `Q = [N, D]` and `Kᵀ = [D, N]`, matrix multiplication returns `[N, N]`. Entry `(i, j)` scores query token `i` against key token `j`.
11. Reshape to `[B, N, 12, 64]`, then permute to `[B, 12, N, 64]`:

    ```python
    q = x.reshape(B, N, 12, 64).permute(0, 2, 1, 3)
    ```

12. `dtype` specifies the representation and precision of each value, such as `float32` or `int64`. `device` specifies where tensor storage and computation live, such as CPU or a CUDA GPU.

</details>

## Completion criteria

0.2 is complete when you can:

- predict shapes before running tensor operations
- distinguish indexing from slicing and element-wise from matrix multiplication
- explain broadcasting from right to left
- use reshape, inference with `-1`, transpose, permute, squeeze, and unsqueeze deliberately
- identify the reduction or softmax axis by meaning
- apply attention masks before softmax
- trace `[B, N, D] → [B, H, N, Dh] → [B, H, N, N]`

## Primary references

- [NumPy: Array fundamentals](https://numpy.org/doc/stable/user/basics.html)
- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [PyTorch: Tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [PyTorch: Tensor attributes](https://docs.pytorch.org/docs/stable/notes/tensor_attributes.html)
- [PyTorch: Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [PyTorch: `matmul`](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
- [PyTorch: `softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
