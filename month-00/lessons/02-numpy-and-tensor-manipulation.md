# 0.2 — NumPy and Tensor Manipulation

The goal is to reason about tensor shapes without trial and error. Transformer code becomes much easier when you can predict the result of an indexing, broadcasting, reshape, or matrix-multiplication operation before running it.

Examples use NumPy and PyTorch side by side where their behavior is similar.

```python
import numpy as np
import torch
```

## 1. Tensors, shapes, axes, and data types

A scalar has no axes, a vector has one, a matrix has two, and a higher-dimensional array is commonly called a tensor.

```python
scalar = torch.tensor(3.0)                         # shape: []
vector = torch.tensor([1.0, 2.0, 3.0])           # shape: [3]
matrix = torch.zeros(2, 3)                        # shape: [2, 3]
batch = torch.zeros(4, 16, 768)                   # shape: [4, 16, 768]
```

For the batch above, a useful interpretation is `[batch, sequence, embedding]`:

- axis 0 contains 4 examples
- axis 1 contains 16 token positions
- axis 2 contains 768 features per token

Shape alone does not record meaning. Two tensors can both have shape `[4, 16, 768]` while representing different concepts, so good variable names and shape comments matter.

Data types affect precision, memory, and valid operations:

```python
token_ids = torch.tensor([12, 5, 91], dtype=torch.long)
activations = torch.randn(3, 768, dtype=torch.float32)
mask = torch.tensor([True, True, False], dtype=torch.bool)
```

Token IDs are integers, activations are floating point, and masks are Boolean. In PyTorch, a tensor also belongs to a device such as CPU, CUDA, or MPS.

## 2. Creating arrays and tensors

```python
np.zeros((2, 3))
np.ones((2, 3))
np.arange(6).reshape(2, 3)

torch.zeros(2, 3)
torch.ones(2, 3)
torch.arange(6).reshape(2, 3)
torch.randn(2, 3)
```

When creating a tensor from another tensor, preserve its device and dtype when appropriate:

```python
bias = torch.zeros_like(activations)
```

## 3. Indexing and slicing

```python
x = torch.arange(24).reshape(2, 3, 4)

x[0]          # shape [3, 4]: first item on axis 0
x[:, 1]       # shape [2, 4]: item 1 on axis 1 for every batch
x[:, :, -1]   # shape [2, 3]: last feature
x[..., -1]    # same result; ... fills the omitted leading axes
x[:, 1:3, :]  # shape [2, 2, 4]: slice preserves the sliced axis
x[:, 1, :]    # shape [2, 4]: integer indexing removes that axis
```

That last distinction matters: slicing with `1:2` preserves an axis of length 1, while indexing with `1` removes it.

## 4. Element-wise operations and vectorization

Ordinary arithmetic operates element by element when shapes are compatible.

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

x + y      # [5, 7, 9]
x * y      # [4, 10, 18], element-wise multiplication
x**2       # [1, 4, 9]
```

Vectorized operations express work over whole tensors. They are clearer and normally much faster than Python loops because optimized native kernels perform the computation.

```python
# Vectorized squared error
error = ((predictions - targets) ** 2).mean()
```

## 5. Broadcasting

Broadcasting allows element-wise operations on tensors with different but compatible shapes. Compare dimensions from right to left. Each pair must be equal, or one of them must be `1`, or one shape must have no corresponding leading dimension.

```python
x = torch.zeros(2, 3, 4)
bias = torch.arange(4)
y = x + bias
```

The shapes align as:

```text
x:     [2, 3, 4]
bias:        [4]
result: [2, 3, 4]
```

The four-value bias is applied to every sequence position in every batch item.

Another example:

```text
[2, 3, 4]
[1, 3, 1]
-----------
[2, 3, 4]
```

An incompatible example:

```text
[2, 3, 4]
      [3]
```

The final dimensions `4` and `3` conflict, so the operation fails.

Broadcasting often avoids copying data, but the resulting operation can still produce a large output. Always reason about the result shape.

## 6. Reshape, view, flatten, and unsqueeze

Reshaping changes how elements are grouped without changing their number or order.

```python
x = torch.arange(24)
x.reshape(2, 3, 4)       # 2 × 3 × 4 = 24
x.reshape(6, 4)
x.reshape(2, -1)         # infer the second dimension: [2, 12]
```

Useful axis operations:

```python
y = x.unsqueeze(0)       # add an axis at position 0: [1, 24]
y.squeeze(0)             # remove axis 0 because its size is 1: [24]

batch = torch.zeros(2, 3, 4)
batch.flatten(start_dim=1)  # preserve batch axis: [2, 12]
```

PyTorch's `view` also changes shape, but it requires compatible memory layout. Operations such as transpose can produce non-contiguous tensors. `reshape` may return a view or make a copy when needed; `contiguous().view(...)` makes that choice explicit.

Never reshape merely to silence a shape error. First confirm what every axis means.

## 7. Transpose and permutation

Transpose reorders axes; it does not rearrange values arbitrarily.

```python
x = torch.zeros(2, 3, 4)

x.transpose(1, 2).shape  # [2, 4, 3]: swap two axes
x.permute(2, 0, 1).shape # [4, 2, 3]: specify every axis in new order
x.mT                     # transpose the final two dimensions
```

For a matrix, `A.T` changes shape `[m, n]` to `[n, m]`. For batched matrices, PyTorch's `mT` is often safer than reversing every axis.

## 8. Concatenation and stacking

Concatenation joins existing axes. Stacking creates a new axis.

```python
a = torch.zeros(2, 3)
b = torch.ones(2, 3)

torch.cat([a, b], dim=0).shape    # [4, 3]
torch.cat([a, b], dim=1).shape    # [2, 6]
torch.stack([a, b], dim=0).shape  # [2, 2, 3]
```

For `cat`, every dimension except the concatenation dimension must match. For `stack`, all input shapes must match.

## 9. Reductions

Reductions combine values along one or more axes.

```python
x = torch.randn(4, 16, 768)

x.mean()                         # scalar
x.mean(dim=-1).shape             # [4, 16]
x.mean(dim=1).shape              # [4, 768]
x.sum(dim=1, keepdim=True).shape # [4, 1, 768]
x.max(dim=-1).values.shape       # [4, 16]
```

`keepdim=True` retains reduced axes with size 1. This often makes later broadcasting easier and less error-prone.

## 10. Matrix multiplication

Element-wise multiplication and matrix multiplication are different operations.

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B                         # shape [2, 4]
```

The inner dimensions must match. Each output entry is a dot product between a row of `A` and a column of `B`.

PyTorch and NumPy treat the final two dimensions as matrices and broadcast preceding dimensions:

```python
Q = torch.randn(8, 12, 32, 64)
K = torch.randn(8, 12, 32, 64)
scores = Q @ K.transpose(-2, -1) # [8, 12, 32, 32]
```

Here the axes represent `[batch, heads, sequence, head_dim]`. Each head compares every query token with every key token.

## 11. Boolean masks and indexed updates

```python
scores = torch.tensor([0.2, -1.0, 0.7, -0.3])
positive = scores[scores > 0]     # [0.2, 0.7]
```

Attention masks usually preserve the full tensor shape and replace disallowed scores before softmax:

```python
masked_scores = scores.masked_fill(~allowed, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)
```

After softmax, positions with negative infinity receive probability zero.

## 12. Numerical stability

Exponentials grow quickly. A direct softmax implementation can overflow:

```python
def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)
```

Subtracting the maximum does not change the softmax probabilities because the same constant is subtracted from every logit. Use library implementations such as `torch.softmax`, `torch.logsumexp`, and `torch.nn.functional.cross_entropy` in real code because they handle stability carefully.

Avoid adding tiny constants without understanding the operation. Prefer a stable formulation designed for the computation.

## 13. The transformer head-shape transformation

Suppose hidden states have shape `[batch, sequence, embedding]`:

```python
batch_size = 2
sequence_length = 10
embedding_dim = 768
num_heads = 12
head_dim = embedding_dim // num_heads  # 64

x = torch.randn(batch_size, sequence_length, embedding_dim)
```

Split the embedding dimension into heads, then move the head axis before sequence:

```python
q = x.reshape(batch_size, sequence_length, num_heads, head_dim)
q = q.transpose(1, 2)
assert q.shape == (2, 12, 10, 64)
```

After attention, reverse the operation:

```python
context = q.transpose(1, 2).contiguous()
context = context.reshape(batch_size, sequence_length, embedding_dim)
assert context.shape == (2, 10, 768)
```

The reshape splits or merges feature axes. The transpose changes axis order. Neither operation computes attention by itself.

## 14. NumPy and PyTorch interoperability

On CPU, the libraries can sometimes share memory:

```python
array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
tensor = torch.from_numpy(array)
back_to_numpy = tensor.numpy()
```

Because memory may be shared, an in-place change through one view can affect the other. Use `.copy()` or `.clone()` when independent storage is required. A tensor requiring gradients must be detached and moved to CPU before conversion:

```python
array = tensor.detach().cpu().numpy()
```

## Exit test

Answer without running the code.

1. What are the shapes of `x[0]`, `x[:, 1]`, and `x[:, 1:2]` when `x.shape == [4, 8, 16]`?
2. Can shapes `[5, 1, 7]` and `[3, 7]` broadcast? What is the result shape?
3. Can shapes `[2, 4, 8]` and `[4]` broadcast? Why?
4. What is the difference between `torch.cat([a, b], dim=0)` and `torch.stack([a, b], dim=0)` for two tensors shaped `[2, 3]`?
5. If `x.shape == [32, 128, 768]`, what is the shape of `x.mean(dim=1)`? What information did that reduction combine?
6. If `A.shape == [10, 20]` and `B.shape == [20, 5]`, what is `(A @ B).shape`?
7. Given `Q` and `K` shaped `[2, 12, 10, 64]`, what is the shape of `Q @ K.transpose(-2, -1)` and what does each final-axis value represent?
8. Why is subtracting the maximum logit before exponentiation safe in softmax?
9. Write the two shape operations that convert `[2, 10, 768]` to `[2, 12, 10, 64]`.
10. Identify the bug:

```python
logits = torch.randn(4, 10)
normalizer = logits.exp().sum(dim=0)
probabilities = logits.exp() / normalizer
```

<details>
<summary>Show answers</summary>

### Answers

1. `x[0]` is `[8, 16]`; integer indexing removes axis 0. `x[:, 1]` is `[4, 16]`; integer indexing removes axis 1. `x[:, 1:2]` is `[4, 1, 16]`; slicing retains axis 1.
2. Yes. Aligning from the right gives `[5, 1, 7]` and `[1, 3, 7]`, so the result is `[5, 3, 7]`.
3. No. The final dimensions are `8` and `4`; neither is `1`, so they are incompatible.
4. Concatenation extends an existing axis and returns `[4, 3]`. Stacking creates a new axis and returns `[2, 2, 3]`.
5. The result is `[32, 768]`. It averages the 128 sequence-position vectors, producing one 768-feature vector per batch item.
6. `[10, 5]`. The shared inner dimension is 20.
7. `[2, 12, 10, 10]`. For each batch and head, every query position has ten dot-product scores, one for each key position.
8. Softmax is unchanged by adding or subtracting the same constant from all logits. Algebraically, the common exponential factor cancels between numerator and denominator. Subtracting the maximum makes the largest exponent zero and prevents overflow.
9. First reshape to `[2, 10, 12, 64]`, then transpose axes 1 and 2:

   ```python
   q = x.reshape(2, 10, 12, 64).transpose(1, 2)
   ```

10. The code normalizes along the batch axis because `sum(dim=0)` returns one total for each class across four examples. Classification probabilities normally normalize classes for each example, so it should use the final axis. It also repeats an unstable exponential calculation. Use:

    ```python
    probabilities = torch.softmax(logits, dim=-1)
    ```

</details>

## Primary references

- [NumPy: Array fundamentals](https://numpy.org/doc/stable/user/basics.html)
- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [PyTorch: Tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [PyTorch: Tensor semantics](https://docs.pytorch.org/docs/stable/notes/tensor_attributes.html)
- [PyTorch: Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
