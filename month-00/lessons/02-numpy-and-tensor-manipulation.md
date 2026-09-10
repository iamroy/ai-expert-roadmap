# 0.2 — NumPy and Tensor Manipulation

The goal is tensor fluency: look at a shape and understand what every dimension represents, predict the result of an operation, and manipulate transformer tensors without trial and error.

| | |
|---|---|
| **Mode** | SKIM, but validate carefully: tensor-level fluency is one of the three Month 0 areas most likely to slow down later months |
| **Time** | 90–120 minutes including core exercises 1–5; add about 60 minutes for stretch exercises 6–8 |
| **Assumes** | 0.1 Python for Modern AI, with NumPy and PyTorch installed |
| **Used by** | 0.3 Linear Algebra, 0.8 PyTorch Fundamentals, the tiny text classifier project, Month 1 transformer internals, Month 9 inference |

**Highest-priority ideas:** shapes, reshape/permute, broadcasting, matrix multiplication, reductions, masking, and selecting with index tensors.

## Learning objectives

After this lesson you can:

- name every axis of a tensor and predict an operation's output shape before running it
- choose correctly among reshape, view, permute, squeeze/unsqueeze, cat, stack, and split
- apply broadcasting deliberately and recognize when it silently produces the wrong shape
- select values with slices, masks, and index tensors, and know which return views and which return copies
- implement padding-aware pooling and masked multi-head attention with explicit shapes
- keep dtype, device, and numerical stability correct in tensor code

## How to use this lesson

1. **Diagnose first.** Answer the [exit test](#exit-test) and attempt [Exercise 7](#exercise-7--multi-head-attention-from-scratch-stretch) without notes.
2. **Read selectively.** Study the lessons behind any miss. Read every **Pitfall** callout even if the topic is familiar; they describe bugs that run without errors.
3. **Run code rather than only reading it.** Keep a Python session open and check each predicted shape with `assert`.
4. **Record gaps** in [`progress.md`](../progress.md).

Every example assumes:

```python
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
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
print(x.numel()) # 3110400 = 32 * 128 * 768
```

Interpret `x.shape == [32, 128, 768]` as:

```text
32   = batch size
128  = sequence length
768  = embedding dimension
```

- **shape** gives the length of every axis
- **ndim** gives the number of axes
- **numel** gives the total number of elements
- **size** means different things in the two libraries: in PyTorch, `x.size()` returns the shape and `x.size(1)` returns one axis length; in NumPy, `array.size` is the total element count, the equivalent of PyTorch's `numel()`
- `len(x)` returns the length of the first axis only

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

### Views versus copies

Basic slicing returns a **view** that shares storage with the original tensor. Indexing with a list, an index tensor, or a Boolean mask returns a **copy**:

```python
x = torch.zeros(5)
view = x[1:3]
view += 1        # modifies x: tensor([0., 1., 1., 0., 0.])

x = torch.zeros(5)
copy = x[[1, 2]]
copy += 1        # x is unchanged
```

NumPy follows the same rule. Boolean indexing also flattens its result: `x[x > 0]` on a `[32, 128, 768]` tensor returns a 1D tensor whose length is the number of matching elements.

> **Pitfall:** Modifying a slice in place changes the original tensor, and autograd raises an error if an in-place change overwrites a value needed for the backward pass. Use `.clone()` when you need an independent tensor.

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

> **Pitfall:** `reshape` cannot reorder axes, even when the target shape looks right. For `[B, N, D] → [B, D, N]`, it produces the requested shape but scrambles which numbers belong to which token and feature, without any error:

```python
x = torch.arange(6).reshape(1, 2, 3)   # 1 sample, 2 tokens, 3 features
# tensor([[[0, 1, 2],
#          [3, 4, 5]]])

x.permute(0, 2, 1)   # correct: column j holds token j's features
# tensor([[[0, 3],
#          [1, 4],
#          [2, 5]]])

x.reshape(1, 3, 2)   # right shape, wrong meaning
# tensor([[[0, 1],
#          [2, 3],
#          [4, 5]]])
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

Use `.mT` or `transpose(-2, -1)` for batched matrices. `.T` reverses *every* axis and is deprecated for tensors that are not 2D.

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

To broadcast deliberately, insert size-one axes with `None` or `unsqueeze`:

```python
a = torch.randn(4, 3)
b = torch.randn(5, 3)
pairwise = a[:, None, :] - b[None, :, :]   # [4, 1, 3] - [1, 5, 3] → [4, 5, 3]
```

> **Pitfall:** Broadcasting can turn a shape bug into a wrong answer instead of an error. A classic case is a loss where predictions have shape `[N, 1]` and targets have shape `[N]`:

```python
predictions = torch.randn(8, 1)
targets = torch.randn(8)

(predictions - targets).shape                        # [8, 8]: every prediction minus every target
((predictions - targets) ** 2).mean()                 # runs, but computes the wrong loss
((predictions.squeeze(-1) - targets) ** 2).mean()     # correct: [8] - [8]
```

Assert shapes at boundaries such as losses, pooling, and attention, where a silent broadcast does the most damage.

## Lesson 0.2.7 — Element-wise operations

Ordinary arithmetic is element-wise when shapes are compatible:

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

x + y   # [5, 7, 9]
x - y   # [-3, -3, -3]
x * y   # [4, 10, 18]
x / y   # [0.25, 0.4, 0.5]
x**2    # [1, 4, 9]
```

`x * y` multiplies corresponding values. It is not matrix multiplication.

Methods ending in an underscore, such as `x.add_(y)` or `x.zero_()`, modify a tensor in place. They save memory but interact with views and autograd, so avoid them in model code unless you have a specific reason.

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

PyTorch's `nn.Linear(in_features, out_features)` stores its weight with shape `[out_features, in_features]` and computes `X @ W.T + b`. The dimensional reasoning is identical; only the storage convention differs. It also accepts any number of leading axes, so `[32, 128, 768]` becomes `[32, 128, 256]`:

```python
layer = nn.Linear(768, 256)
X = torch.randn(10, 768)

assert layer.weight.shape == (256, 768)
assert torch.allclose(layer(X), X @ layer.weight.T + layer.bias, atol=1e-5)
assert layer(torch.randn(32, 128, 768)).shape == (32, 128, 256)
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

The inverse operations split a tensor apart. `split` takes a chunk size (or a list of sizes), and `chunk` takes a number of chunks:

```python
qkv = torch.randn(32, 128, 3 * 768)                 # one fused Q/K/V projection
q, k, v = qkv.split(768, dim=-1)                    # three [32, 128, 768] tensors
first, second = torch.randn(8, 10).chunk(2, dim=0)  # two [4, 10] tensors
```

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

Indexing with `None` does the same thing and is common in research code: `x[None]` equals `x.unsqueeze(0)`, and `mask[:, :, None]` adds a trailing axis. NumPy's equivalent of `unsqueeze` is `np.expand_dims`.

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

`argmax` returns the index of the largest value rather than the value itself. It turns classifier logits into predicted classes:

```python
logits = torch.randn(16, 3)                  # [batch, classes]
labels = torch.randint(0, 3, (16,))          # [batch]
predictions = logits.argmax(dim=-1)          # [batch]
accuracy = (predictions == labels).float().mean()
```

Without `dim`, `argmax` returns a single index into the flattened tensor, which is almost never what a batch computation needs.

### Padding-aware mean pooling

Sequences in a batch are padded to a common length, so a plain `mean(dim=1)` averages padding vectors into the result. Weight the average by the attention mask instead:

```python
hidden = torch.randn(2, 5, 4)                       # [B, N, D]
mask = torch.tensor([[1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1]])              # [B, N]: 1 = real token

mask_f = mask.unsqueeze(-1).to(hidden.dtype)        # [B, N, 1]
summed = (hidden * mask_f).sum(dim=1)               # [B, D]
counts = mask_f.sum(dim=1).clamp(min=1.0)           # [B, 1]; avoids 0 / 0
pooled = summed / counts                            # [B, D]

assert torch.allclose(pooled[0], hidden[0, :3].mean(dim=0))
```

The Month 0 project requires exactly this operation, along with a test that padding does not change the pooled embedding.

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

Disallowed positions receive zero probability after softmax. A row in which every position is masked produces `NaN` values, because softmax over only negative infinities is undefined. This can happen with fully padded sequences and needs deliberate handling.

### Padding masks and broadcasting

A padding mask usually has shape `[B, N]`, while attention scores have shape `[B, H, N, N]`. Add axes so that the mask selects **keys**, the last axis, for every head and every query:

```python
B, H, N = 2, 4, 5
scores = torch.randn(B, H, N, N)
is_real = torch.tensor([[True, True, True, False, False],
                        [True, True, True, True,  True]])   # [B, N]

key_mask = is_real[:, None, None, :]                        # [B, 1, 1, N]
weights = torch.softmax(scores.masked_fill(~key_mask, float("-inf")), dim=-1)

assert torch.all(weights[0, :, :, 3:] == 0)                 # padded keys receive no weight
```

Causal and padding masks combine with logical AND: `allowed = causal[None, None] & key_mask`, giving shape `[B, 1, N, N]`.

> **Pitfall:** Libraries disagree about what `True` means in a Boolean mask. In `torch.nn.functional.scaled_dot_product_attention`, `True` marks positions that **may** be attended to. In `nn.MultiheadAttention`, `key_padding_mask=True` marks positions to **ignore**. Check the documentation for every attention API you call, and test it with a padded example.

> **Pitfall:** Filling with `-inf` turns a fully masked row into `NaN`, and so does mask arithmetic that mixes `+inf` and `-inf`. Many implementations fill with the most negative finite value, `torch.finfo(scores.dtype).min`, instead; a fully masked row then gets uniform weights that you can zero out explicitly.

## Lesson 0.2.15 — Softmax across a dimension

Suppose attention scores have shape `[B, H, N, N]`:

```python
weights = torch.softmax(scores, dim=-1)
```

`dim=-1` normalizes over the final dimension. For each batch, head, and query token, the weights over all key tokens sum to 1.

Always ask:

> Which axis am I normalizing over?

Applying softmax over the wrong axis can return a tensor with the expected shape while giving it the wrong meaning.

When you need log-probabilities, as in a cross-entropy loss, use `torch.log_softmax(scores, dim=-1)` rather than `torch.softmax(...).log()`. The next lesson shows why.

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

The same idea gives the log-sum-exp trick, which computes log-probabilities safely:

```text
log softmax(x)ᵢ = xᵢ − logsumexp(x)
logsumexp(x)    = m + log Σⱼ exp(xⱼ − m),  where m = max(x)
```

```python
logits = torch.tensor([1000.0, 0.0, -1000.0])

torch.softmax(logits, dim=-1).log()   # tensor([0., -inf, -inf]): probabilities underflowed
torch.log_softmax(logits, dim=-1)     # tensor([0., -1000., -2000.]): exact and finite
```

This is why `torch.nn.functional.cross_entropy` takes raw **logits**, not probabilities: it applies `log_softmax` internally and computes `-log P(correct class)` stably. Passing probabilities, or applying softmax twice, still runs and trains, but it optimizes a distorted objective.

Floating-point format matters too:

| dtype | Largest finite value | Typical use |
|---|---|---|
| `float32` | about 3.4 × 10³⁸ | default for training and inference |
| `float16` | 65,504 | mixed precision; overflows easily |
| `bfloat16` | about 3.4 × 10³⁸ | mixed precision with float32's range but fewer precision bits |

Two further habits prevent subtle bugs:

- Compare floating-point tensors with `torch.allclose` or `torch.testing.assert_close`, not `==`. Even `0.1 + 0.2 == 0.3` is `False`.
- Add a small `eps`, or use `clamp(min=...)`, before dividing by a norm or count that can be zero.

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

An in-place change through one view can affect the other. `torch.tensor(array)` always copies, while `torch.from_numpy(array)` and `tensor.numpy()` share memory on CPU. Use `.copy()` or `.clone()` for independent storage. Before converting a gradient-tracked tensor to NumPy:

```python
array = tensor.detach().cpu().numpy()
```

> **Pitfall:** NumPy creates `float64` arrays by default, while PyTorch parameters are `float32`. A tensor made with `torch.from_numpy(np.array([1.0, 2.0]))` is `float64`, and passing it to a layer fails with an error such as `mat1 and mat2 must have the same dtype`. Convert with `.float()` or create the array with `dtype=np.float32`.

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
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

x = torch.randn(4, 768)
x = x.to(device)
x = x.to(torch.bfloat16)
```

Tensors involved in one operation generally need compatible devices and dtypes. Token IDs used for embedding lookup are integer tensors, while model activations and parameters are floating point. `nn.Embedding` requires integer indices, and `F.cross_entropy` expects class targets as `int64` (`torch.long`).

> **Pitfall:** Calling `.item()`, `.tolist()`, `.cpu()`, or `print` on a GPU tensor makes the CPU wait for all queued GPU work to finish. Inside a training loop, do this every N steps rather than on every step.

When creating a related tensor, preserve properties when appropriate:

```python
bias = torch.zeros_like(activations)
```

## Lesson 0.2.19 — Selecting with index tensors

An integer tensor can index another tensor. This is how token IDs become embeddings:

```python
vocab_size, D = 1000, 16
embedding_table = torch.randn(vocab_size, D)     # [V, D]
token_ids = torch.tensor([[5, 42, 7],
                          [9, 0, 0]])            # [B, N], int64

vectors = embedding_table[token_ids]             # [B, N, D]
assert vectors.shape == (2, 3, 16)
```

`nn.Embedding` performs the same lookup with a trainable table.

Picking one value per row, such as each example's logit for its correct class, pairs a row index with a column index:

```python
logits = torch.randn(4, 10)                      # [B, C]
targets = torch.tensor([3, 0, 9, 3])             # [B]

correct = logits[torch.arange(4), targets]       # [B]
same = logits.gather(dim=1, index=targets[:, None]).squeeze(1)
assert torch.equal(correct, same)
```

`torch.gather(input, dim, index)` generalizes this to any axis. The output has the shape of `index`, and along `dim` each value is read from the position that `index` names. Language models use it to select the log-probability of each actual next token from `[B, N, V]` log-probabilities:

```python
log_probs = torch.log_softmax(torch.randn(2, 5, 100), dim=-1)                  # [B, N, V]
next_tokens = torch.randint(0, 100, (2, 5))                                    # [B, N]
token_log_probs = log_probs.gather(-1, next_tokens[..., None]).squeeze(-1)     # [B, N]
loss = -token_log_probs.mean()

reference = F.cross_entropy(log_probs.reshape(-1, 100), next_tokens.reshape(-1))
assert torch.allclose(loss, reference, atol=1e-5)
```

That loss is the cross-entropy you will derive in 0.4: the average of `-log P(correct token)`.

## Lesson 0.2.20 — `einsum` as executable shape notation

`torch.einsum` (and `np.einsum`) describes an operation with labeled axes. Values are multiplied along axes that share a label, and labels missing from the output are summed away:

```python
Q = torch.randn(8, 12, 32, 64)   # [B, H, N, Dh]
K = torch.randn(8, 12, 32, 64)

scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)   # [B, H, N, N]
assert torch.allclose(scores, Q @ K.transpose(-2, -1), atol=1e-4)
```

Common patterns:

```text
"i,i->"        dot product
"nd,dk->nk"    matrix multiplication
"bnd->bd"      sum over the sequence axis
"bi,bj->bij"   batched outer product
```

You do not need to write everything with `einsum`, but you should be able to read it; attention implementations and research code use it heavily. The `einops` library provides a related notation for reshaping and permuting, such as `rearrange(x, "b n d -> b d n")`.

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

First, project the embeddings into queries, keys, and values. Each projection is a linear layer whose output size is also `D`, so the shape does not change:

```text
Q = XWq, K = XWk, V = XWv:   [32, 128, 768] → [32, 128, 768]
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
B, N, D, H = 32, 128, 768, 12
Dh = D // H

x = torch.randn(B, N, D)
w_q = nn.Linear(D, D)
q = w_q(x).reshape(B, N, H, Dh).permute(0, 2, 1, 3)

assert q.shape == (B, H, N, Dh)
```

Keys and values are split the same way, so:

```text
Q  = [32, 12, 128, 64]
K  = [32, 12, 128, 64]
Kᵀ = [32, 12, 64, 128]   (last two axes transposed)
```

Matrix multiplication produces:

```text
[32, 12, 128, 64] @ [32, 12, 64, 128]
= [32, 12, 128, 128]
```

The final tensor contains a score between every query token and every key token, separately for every attention head and every batch item.

Scale the scores, apply any mask, normalize over keys, and use the weights to mix the values:

```text
weights = softmax(scores / √Dh, dim=-1):                  [32, 12, 128, 128]
weights @ V:  [32, 12, 128, 128] @ [32, 12, 128, 64]  →  [32, 12, 128, 64]
```

Merge the heads by reversing the split, then apply the output projection:

```python
attn_output = torch.randn(B, H, N, Dh)                    # stands in for weights @ V
merged = attn_output.permute(0, 2, 1, 3).contiguous()     # [B, N, H, Dh]
merged = merged.reshape(B, N, D)                          # [B, N, D]
output = nn.Linear(D, D)(merged)                          # [B, N, D]
```

(`reshape` would copy a non-contiguous tensor automatically; `.contiguous()` is required if you use `view` instead.)

The reshape splits or merges feature axes. The permutation changes axis order. Neither operation computes attention by itself; the projections, `Q @ Kᵀ`, the mask, softmax, and `weights @ V` do. [Exercise 7](#exercise-7--multi-head-attention-from-scratch-stretch) assembles the full computation and checks it against PyTorch's built-in implementation.

## Hands-on exercises

Exercises 1–5 are the core validation for 0.2. Exercises 6–8 are stretch exercises that build exactly what the Month 0 project and Month 1 attention work need. Call `torch.manual_seed(0)` first for reproducible values.

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

### Exercise 6 — Padding-aware pooling (stretch)

Write `masked_mean_pool(hidden, mask)` for `hidden: [B, N, D]` and `mask: [B, N]`. Verify that changing the values at padded positions does not change the output, and that an all-padding row returns zeros instead of `NaN`.

### Exercise 7 — Multi-head attention from scratch (stretch)

Write a multi-head self-attention forward pass using only linear layers, `reshape`, `permute`, `@`, `masked_fill`, and `softmax`. Support an optional `[B, N]` key padding mask and an optional causal mask, and assert the shape after every step. Verify that your result matches `F.scaled_dot_product_attention` applied to the same projected heads.

### Exercise 8 — Find the silent shape bugs (stretch)

Each line runs without an error but computes the wrong thing. Explain the bug and fix it.

```python
# Given: x, hidden: [B, N, D]; scores: [B, H, N, N]; logits: [B, C]
#        probs: [B, 1]; labels: [B]; batches contain padding

x_bdn = x.reshape(B, D, N)                     # a) swap sequence and feature axes
weights = torch.softmax(scores, dim=-2)        # b) attention weights over keys
pooled = hidden.mean(dim=1)                    # c) sentence embedding
loss = ((probs - labels) ** 2).mean()          # d) per-example squared error
predicted_class = logits.argmax()              # e) one prediction per example
```

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

<details>
<summary>Show stretch exercise solutions</summary>

### Exercise 6

```python
def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """hidden: [B, N, D]; mask: [B, N] with 1/True for real tokens -> [B, D]."""
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)          # [B, N, 1]
    summed = (hidden * mask_f).sum(dim=1)                 # [B, D]
    counts = mask_f.sum(dim=1).clamp(min=1.0)             # [B, 1]
    return summed / counts

torch.manual_seed(0)
hidden = torch.randn(3, 6, 8)
mask = torch.tensor([[1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0]])                  # third row is all padding

pooled = masked_mean_pool(hidden, mask)

changed = hidden.clone()
changed[mask == 0] = 999.0                                 # corrupt only padded positions
assert torch.allclose(pooled, masked_mean_pool(changed, mask))
assert torch.all(pooled[2] == 0) and not torch.isnan(pooled).any()
```

### Exercise 7

```python
import math

def split_heads(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    B, N, D = t.shape
    return t.reshape(B, N, num_heads, D // num_heads).permute(0, 2, 1, 3)   # [B, H, N, Dh]

def build_allowed(B: int, N: int, key_mask=None, causal=False) -> torch.Tensor:
    allowed = torch.ones(N, N, dtype=torch.bool)
    if causal:
        allowed = torch.tril(allowed)
    allowed = allowed[None, None]                           # [1, 1, N, N]
    if key_mask is not None:
        allowed = allowed & key_mask[:, None, None, :]      # [B, 1, N, N]
    return allowed

def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads, key_mask=None, causal=False):
    B, N, D = x.shape
    assert D % num_heads == 0
    Dh = D // num_heads

    q = split_heads(w_q(x), num_heads)                      # [B, H, N, Dh]
    k = split_heads(w_k(x), num_heads)                      # [B, H, N, Dh]
    v = split_heads(w_v(x), num_heads)                      # [B, H, N, Dh]

    scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)        # [B, H, N, N]
    allowed = build_allowed(B, N, key_mask, causal)
    scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)                 # [B, H, N, N]
    assert weights.shape == (B, num_heads, N, N)

    context = weights @ v                                   # [B, H, N, Dh]
    merged = context.permute(0, 2, 1, 3).reshape(B, N, D)   # [B, N, D]
    return w_o(merged)                                      # [B, N, D]

torch.manual_seed(0)
B, N, D, H = 2, 6, 32, 4
x = torch.randn(B, N, D)
w_q, w_k, w_v, w_o = (nn.Linear(D, D) for _ in range(4))
key_mask = torch.tensor([[True] * 6,
                         [True] * 4 + [False] * 2])

ours = multi_head_attention(x, w_q, w_k, w_v, w_o, H, key_mask=key_mask, causal=True)

q, k, v = (split_heads(w(x), H) for w in (w_q, w_k, w_v))
allowed = build_allowed(B, N, key_mask, causal=True)
reference = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)   # True = may attend
reference = w_o(reference.permute(0, 2, 1, 3).reshape(B, N, D))

assert ours.shape == (B, N, D)
assert torch.allclose(ours, reference, atol=1e-5)
```

The causal mask guarantees that every query can attend at least to position 0, which is a real token here, so no row is fully masked.

### Exercise 8

- **a)** `reshape` regroups elements without reordering axes, so tokens and features are scrambled. Use `x.permute(0, 2, 1)` or `x.transpose(1, 2)`.
- **b)** `dim=-2` normalizes over queries, so each column sums to 1 instead of each row. Use `dim=-1` so that every query's weights over keys sum to 1.
- **c)** Plain `mean` averages padding vectors into the embedding. Use masked mean pooling from Exercise 6.
- **d)** `[B, 1] - [B]` broadcasts to `[B, B]`, comparing every prediction with every label. Use `probs.squeeze(-1) - labels`.
- **e)** Without `dim`, `argmax` returns one index into the flattened `[B, C]` tensor. Use `logits.argmax(dim=-1)` to get `[B]` predictions.

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

### Extended check

These questions go beyond the original 12 and cover the pitfalls added to this lesson.

13. Which indexing operations return views, and which return copies?
14. Why does `[N, 1] - [N]` not raise an error, and why is that dangerous?
15. What do `logits[torch.arange(B), targets]` and `logits.gather(1, targets[:, None])` compute?
16. Why does `F.cross_entropy` expect logits rather than probabilities?
17. How must a `[B, N]` padding mask be reshaped to mask `[B, H, N, N]` attention scores, and which axis does it mask?

<details>
<summary>Show answers</summary>

13. Basic slicing (`x[1:3]`, `x[:, 0]`) returns views that share storage. Indexing with a list, an integer tensor, or a Boolean mask returns a copy.
14. Broadcasting treats `[N]` as `[1, N]`, and both size-one axes expand, producing `[N, N]`. A loss computed on that tensor runs normally but compares every prediction with every target.
15. Both select, for each row `i`, the logit of that row's target class `targets[i]`. The first returns `[B]`; `gather` returns `[B, 1]` until squeezed.
16. It applies `log_softmax` internally using the log-sum-exp trick, which stays finite for extreme logits and computes `-log P(correct class)` exactly. Probabilities that have already underflowed to 0 cannot be recovered.
17. Reshape it to `[B, 1, 1, N]` so it broadcasts across heads and queries and masks the last axis, the **keys**. A padded token then receives no attention from any query.

</details>

## Completion criteria

0.2 is complete when you can:

- predict shapes before running tensor operations
- distinguish indexing from slicing and element-wise from matrix multiplication
- explain broadcasting from right to left
- use reshape, inference with `-1`, transpose, permute, squeeze, and unsqueeze deliberately
- identify the reduction or softmax axis by meaning
- apply causal and padding masks before softmax, with the correct broadcast shape and polarity
- select values with index tensors and `gather`, and explain which indexing operations return views
- implement padding-aware pooling and a masked multi-head attention forward pass that matches PyTorch's implementation
- trace `[B, N, D] → [B, H, N, Dh] → [B, H, N, N] → [B, H, N, Dh] → [B, N, D]`
- explain why cross-entropy takes logits and why `float16` overflows

## Primary references

- [NumPy: Array fundamentals](https://numpy.org/doc/stable/user/basics.html)
- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [PyTorch: Tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [PyTorch: Tensor attributes](https://docs.pytorch.org/docs/stable/notes/tensor_attributes.html)
- [PyTorch: Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [PyTorch: `matmul`](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
- [PyTorch: `softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
- [NumPy: Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy: Copies and views](https://numpy.org/doc/stable/user/basics.copies.html)
- [PyTorch: `gather`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)
- [PyTorch: `einsum`](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html)
- [PyTorch: `cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [PyTorch: `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch: Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

## About this lesson

Adapted from the roadmap planning conversation, [Create Month Zero Learning List](https://chatgpt.com/share/6aa28c17-a29c-83ea-ae5b-9202ec5ba987). The original 18 lessons, five exercises, and 12-question exit test are preserved; index selection, `einsum`, the pitfall callouts, the complete attention walkthrough, stretch exercises, and the extended check were added. Code examples were checked with NumPy 2.4 and PyTorch 2.14 on CPU.
