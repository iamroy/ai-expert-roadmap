# 0.8 PyTorch Fundamentals — SKIM / REFRESH

This is where 0.2 through 0.7 become code. The curriculum flags tensor-level PyTorch fluency as one of the three areas to validate most carefully, so treat the exercise as mandatory rather than optional: **implement a tiny MLP training loop with no high-level trainer**.

| | |
|---|---|
| **Mode** | SKIM / REFRESH, but validate with code |
| **Time** | 90–120 minutes, including the MLP exercise |
| **Assumes** | 0.1 Python, 0.2 Tensors, 0.5 Calculus, 0.6 Neural Networks, 0.7 Training |
| **Used by** | the tiny text classifier project, and every month that follows |

[Month 0 roadmap](../README.md) · [Previous: Training Fundamentals](07-training-fundamentals.md) · [Next: Architecture Concepts](09-deep-learning-architecture-concepts.md)

## Learning objectives

After this lesson you can:

- create, move, and inspect tensors, and reason about device and dtype
- explain what autograd records and when it is freed
- write an `nn.Module` with `__init__` and `forward`, and inspect its parameters
- build a `Dataset` and `DataLoader` with a custom `collate_fn` for padding
- write training and validation loops from scratch, with correct mode switching
- save and load checkpoints, and know where mixed precision, `torch.compile`, and GPU memory fit

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Then do [the MLP exercise](#exercise-1--a-tiny-mlp-training-loop-from-scratch) with the lesson closed. Typing it from memory is the validation.
3. Record gaps in [`progress.md`](../progress.md).

Examples assume:

```python
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
```

## 0.8.1 — Tensors, devices, and dtypes

```python
x = torch.zeros(2, 3)
y = torch.randn(2, 3)                       # standard normal
z = torch.arange(6).reshape(2, 3)           # int64
w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # from data

print(y.shape, y.dtype, y.device, y.requires_grad)   # [2, 3] torch.float32 cpu False
```

Pick the device once, at the top of a script, and move both model and batch:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(device)
```

The common errors are a device mismatch (`Expected all tensors to be on the same device`) and a dtype mismatch, usually `float64` arriving from NumPy where `float32` was expected (0.2.17). Both are fixed with `.to(device)` and `.float()`.

## 0.8.2 — Autograd in practice

```python
w = torch.randn(3, requires_grad=True)
x = torch.randn(3)

loss = (w * x).sum()
loss.backward()
print(w.grad.shape, x.grad)                 # torch.Size([3]) None: x needs no gradient

w.grad.zero_()                              # gradients accumulate; clear before reuse

with torch.no_grad():                       # no graph recorded
    w += 0.1

detached = (w * x).detach()                 # cut out of the graph, e.g. for logging
print(detached.requires_grad)               # False
```

Rules worth internalizing:

- Only leaf tensors with `requires_grad=True` accumulate into `.grad`.
- `backward()` frees the graph. Calling it twice on the same graph raises an error unless you pass `retain_graph=True`, and needing that usually means the loop is structured wrong.
- Use `torch.inference_mode()` rather than `no_grad()` for pure inference; it is stricter and slightly faster.
- In-place operations on tensors autograd needs for the backward pass raise a clear runtime error. `.clone()` is the fix.

## 0.8.3 — `nn.Module`

A module owns parameters and defines computation:

```python
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()                              # first, always (0.1.4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] -> [B, out_dim] logits."""
        return self.net(x)

model = MLP(16, 64, 4)
print(sum(p.numel() for p in model.parameters()))       # 1,348
print([name for name, _ in model.named_parameters()])
```

- Call `model(x)`, not `model.forward(x)`; the former runs registered hooks (0.1.3).
- Submodules assigned as attributes are registered automatically. A plain Python list of layers is **not**; use `nn.ModuleList` or `nn.Sequential`, or the parameters will silently never train.
- For a non-parameter tensor that should move with the model and be saved, use `self.register_buffer("mask", tensor)`. Causal attention masks are the canonical example.
- `model.to(device)` moves parameters in place; tensors need `x = x.to(device)` because tensor `.to()` returns a new tensor.

## 0.8.4 — `Dataset` and `DataLoader`

A map-style `Dataset` implements `__len__` and `__getitem__` (0.1.3):

```python
class ToyTextDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(self.sequences[index]), self.labels[index]
```

Variable-length sequences need a **`collate_fn`** to pad each batch. Padding per batch rather than to a global maximum saves a lot of computation:

```python
def collate(batch: list[tuple[torch.Tensor, int]], pad_id: int = 0):
    sequences, labels = zip(*batch)
    longest = max(len(s) for s in sequences)

    padded = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    for i, sequence in enumerate(sequences):
        padded[i, :len(sequence)] = sequence

    mask = padded != pad_id                       # [B, N] True for real tokens
    return padded, mask, torch.tensor(labels)

dataset = ToyTextDataset([[1, 2, 3], [4, 5], [6, 7, 8, 9]], [0, 1, 0])
loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate)

tokens, mask, labels = next(iter(loader))
print(tokens.shape, mask.sum(dim=1).tolist())     # [3, 4] [3, 2, 4]
```

`DataLoader` options worth knowing: `shuffle=True` for training and `False` for validation, `num_workers` for parallel loading (0.1.15, remember the `__main__` guard), `pin_memory=True` with CUDA, and `drop_last=True` when a ragged final batch would cause trouble.

## 0.8.5 — The training loop

Every element here came from 0.7. This is the reference shape:

```python
def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()                                        # dropout on
    total, count = 0.0, 0
    for x, targets in loader:
        x, targets = x.to(device), targets.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item() * len(targets)
        count += len(targets)
    return total / count


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()                                         # dropout off
    loss_sum, correct, count = 0.0, 0, 0
    for x, targets in loader:
        x, targets = x.to(device), targets.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == targets).sum().item()
        count += len(targets)
    return loss_sum / count, correct / count
```

Four details that are easy to get wrong:

- `model.train()` and `model.eval()` switch dropout and normalization behavior. Set them explicitly in both functions rather than assuming.
- `@torch.no_grad()` on evaluation saves memory and time; without it you build a graph you never use.
- Weight the loss by batch size when averaging, or a smaller final batch skews the epoch average.
- `.item()` forces a GPU synchronization (0.2.18). Accumulate on the device and call `.item()` once per epoch when it matters.

## 0.8.6 — Checkpoints

```python
model = MLP(16, 64, 4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

torch.save(
    {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 3},
    "/tmp/mlp.pt",
)

checkpoint = torch.load("/tmp/mlp.pt", weights_only=True)     # safe default
restored = MLP(16, 64, 4)
restored.load_state_dict(checkpoint["model"])
restored.eval()
print(checkpoint["epoch"])
```

`state_dict()` returns tensors only, not the class, so you must construct the model first with the same architecture. Save the config alongside the weights so you can. For untrusted checkpoints, keep `weights_only=True` (0.1.16).

## 0.8.7 — Mixed precision, `torch.compile`, and GPU memory

These are GPU-facing; know what they are and reach for them in Month 7.

**Mixed precision** runs most operations in `bfloat16` or `float16` while keeping a `float32` master copy of the weights, cutting memory and increasing throughput:

```python
# CUDA only
# scaler = torch.amp.GradScaler("cuda")
# with torch.autocast("cuda", dtype=torch.bfloat16):
#     loss = F.cross_entropy(model(x), targets)
# scaler.scale(loss).backward()
# scaler.step(optimizer)
# scaler.update()
```

`bfloat16` has `float32`'s exponent range and needs no loss scaling; `float16` has a maximum of 65,504 (0.2.16) and does, which is what `GradScaler` provides.

**`torch.compile(model)`** traces and fuses the graph ahead of time, typically giving a solid speedup for a one-time compilation cost. It is a single line and is worth trying on any long run.

**GPU memory** during training holds four things: parameters, gradients, optimizer state (two extra copies for AdamW, 0.7.1), and activations saved for the backward pass. Activations scale with batch size and sequence length and are usually what overflows first. Levers, in the order to try them: reduce batch size and use gradient accumulation, enable mixed precision, turn on gradient checkpointing, then shard state across devices. `torch.cuda.max_memory_allocated()` tells you where you actually are.

## Exercises

### Exercise 1 — A tiny MLP training loop from scratch

This is the curriculum's required exercise. Write it without a high-level trainer and without looking at the solution:

1. Generate a synthetic classification dataset: `X: [1000, 16]` and 4 classes with genuinely learnable structure.
2. Wrap it in a `Dataset`, split 80/20, and build two `DataLoader`s.
3. Define an `MLP` as an `nn.Module`.
4. Train with AdamW, cross-entropy, gradient clipping, and a cosine schedule with warmup.
5. Evaluate each epoch, tracking loss and accuracy.
6. Save the best checkpoint by validation loss, reload it, and confirm the metrics match.

**Check your work:** the first training loss should be near `log 4 ≈ 1.386`, validation accuracy should end well above the 25% random baseline, and the reloaded checkpoint should reproduce the validation numbers exactly.

### Exercise 2 — Padding-aware batching

Using the `collate` function from 0.8.4, build a text model that embeds token IDs, mean-pools with the mask (0.2.13), and classifies. Verify that appending extra padding to a sequence does not change its prediction.

### Exercise 3 — Four common PyTorch bugs

Reproduce each bug, observe its symptom, then fix it: layers stored in a plain Python list; forgetting `model.eval()` before validation with dropout at 0.5; forgetting `optimizer.zero_grad()`; and passing softmax probabilities to `F.cross_entropy`.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
import math

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. synthetic data with real structure: each class has its own centroid
centroids = torch.randn(4, 16) * 0.6            # overlapping enough to be non-trivial
labels = torch.randint(0, 4, (1000,))
X = centroids[labels] + torch.randn(1000, 16)


class TensorDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


# 2. split and load
dataset = TensorDataset(X, labels)
train_set, val_set = random_split(dataset, [800, 200],
                                  generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# 3. model
model = MLP(16, 64, 4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

epochs, warmup = 15, 20
total_steps = epochs * len(train_loader)


def lr_scale(step: int) -> float:
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

# 4-6. train, evaluate, checkpoint the best
best_val = float("inf")
for epoch in range(epochs):
    model.train()
    running, seen = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running += loss.item() * len(y)
        seen += len(y)

    val_loss, val_accuracy = evaluate(model, val_loader, device)
    if val_loss < best_val:
        best_val = val_loss
        torch.save({"model": model.state_dict(), "epoch": epoch}, "/tmp/best.pt")

    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:2d}  train {running / seen:.4f}  "
              f"val {val_loss:.4f}  acc {val_accuracy:.3f}")

# reload and confirm
reloaded = MLP(16, 64, 4).to(device)
reloaded.load_state_dict(torch.load("/tmp/best.pt", weights_only=True)["model"])
print("reloaded:", evaluate(reloaded, val_loader, device))
```

```text
epoch  0  train 1.2257  val 0.9202  acc 0.815
epoch  5  train 0.2691  val 0.2620  acc 0.895
epoch 10  train 0.2485  val 0.2522  acc 0.905
epoch 14  train 0.2387  val 0.2514  acc 0.900
reloaded: (0.2514…, 0.9)
```

Three things to read here. The first epoch's average sits just under `log 4 ≈ 1.386` because it already includes learning inside that epoch; print the very first batch's loss to see the baseline exactly. Validation accuracy of about 0.90 is far above the 0.25 random baseline and close to the ceiling for this much class overlap, so the model is learning the real structure rather than memorizing. And the reloaded checkpoint reproduces the best validation numbers exactly, which is the actual point of the exercise.

### Exercise 2

```python
torch.manual_seed(0)


class PooledClassifier(nn.Module):
    def __init__(self, vocab: int, dim: int, classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim, padding_idx=0)
        self.head = nn.Linear(dim, classes)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)                       # [B, N, D]
        weights = mask.unsqueeze(-1).to(hidden.dtype)         # [B, N, 1]
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        return self.head(pooled)


model = PooledClassifier(vocab=50, dim=8, classes=3).eval()

short = torch.tensor([[5, 6, 7]])
padded = torch.tensor([[5, 6, 7, 0, 0, 0]])

with torch.inference_mode():
    a = model(short, short != 0)
    b = model(padded, padded != 0)

assert torch.allclose(a, b, atol=1e-6)
print("padding does not change the prediction")
```

### Exercise 3

```python
# Bug 1: layers in a plain list are invisible to the optimizer
class Broken(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(4, 4), nn.Linear(4, 4)]      # not registered

class Fixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

print(sum(p.numel() for p in Broken().parameters()))          # 0: nothing would train
print(sum(p.numel() for p in Fixed().parameters()))           # 40

# Bug 2: dropout left on during evaluation makes results non-deterministic
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5))
x = torch.randn(4, 8)
model.train()
print(torch.allclose(model(x), model(x)))                     # False: two different answers
model.eval()
print(torch.allclose(model(x), model(x)))                     # True

# Bug 3: missing zero_grad inflates gradients every step
model = nn.Linear(4, 2)
x, y = torch.randn(8, 4), torch.randint(0, 2, (8,))
norms = []
for step in range(3):
    F.cross_entropy(model(x), y).backward()                   # no zero_grad
    norms.append(model.weight.grad.norm().item())
print([round(n, 4) for n in norms])                           # grows: 1x, 2x, 3x

# Bug 4: probabilities passed to cross_entropy
logits = torch.randn(8, 4)
targets = torch.randint(0, 4, (8,))
print(F.cross_entropy(logits, targets).item())                        # correct
print(F.cross_entropy(torch.softmax(logits, -1), targets).item())     # softmax applied twice
```

Bug 1 is the quietest: the model runs and the loss barely moves, because the optimizer received no parameters. Bug 4 also runs without error and still trains, just against a flattened objective, which is why it survives into real codebases.

</details>

## Exit test

1. What does `requires_grad=True` do, and which tensors have it by default?
2. Why call `model(x)` instead of `model.forward(x)`?
3. What happens to layers stored in a plain Python list inside a module?
4. What is `register_buffer` for?
5. What two methods must a map-style `Dataset` implement?
6. What problem does `collate_fn` solve, and why pad per batch?
7. What do `model.train()` and `model.eval()` change?
8. Why decorate an evaluation function with `@torch.no_grad()`?
9. List the five steps of a training iteration in order.
10. What does `state_dict()` contain, and why must you build the model before loading one?
11. When would you choose `bfloat16` over `float16`?
12. What four things consume GPU memory during training, and which usually overflows first?

<details>
<summary>Show answers</summary>

1. It marks a tensor for gradient tracking so operations on it are recorded and `.grad` is populated by `backward()`. Module parameters have it by default; input data and buffers do not.
2. `nn.Module.__call__` runs registered forward and backward hooks around `forward`. Calling `forward` directly skips them.
3. They are not registered as submodules, so they do not appear in `parameters()`, are not moved by `.to(device)`, and never receive updates. Use `nn.ModuleList` or `nn.Sequential`.
4. Registering a non-parameter tensor that should move with the model and be saved in the `state_dict`, such as a causal mask or running statistics.
5. `__len__` and `__getitem__`.
6. It assembles a list of individual examples into a batch, which is where variable-length sequences get padded to a common length. Padding to the longest sequence in the batch rather than a global maximum avoids computing on unnecessary padding.
7. They switch modules with training-specific behavior: dropout is active in training and disabled in eval, and BatchNorm uses batch statistics versus running statistics.
8. It disables graph construction, saving memory and time, since no gradients are needed for evaluation.
9. Forward, compute loss, `zero_grad()`, `backward()`, `optimizer.step()`.
10. A dictionary of parameter and buffer tensors, keyed by name. It contains no architecture, so you must instantiate the matching model class first and then load the tensors into it.
11. When the hardware supports it. `bfloat16` keeps `float32`'s exponent range, so it avoids the overflow and underflow that `float16` suffers, and it needs no loss scaler.
12. Parameters, gradients, optimizer state, and saved activations. Activations usually overflow first, since they scale with batch size and sequence length.

</details>

## Completion criteria

You are done when:

- you can write the MLP training loop from memory, including mode switching and the step order
- you can write a `Dataset` and a padding `collate_fn` without looking them up
- you can explain what `state_dict` does and does not contain
- you can name the four consumers of GPU memory and the order in which to attack them
- your Exercise 1 checkpoint reloads and reproduces its validation metrics
- you have reproduced and fixed all four bugs in Exercise 3

You are now ready for [the Month 0 project](../project/tiny-text-classifier/README.md), which is this lesson applied to real text.

## References

**Core tutorials**
- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) — the official end-to-end path
- [What is `torch.nn` really?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html) — builds up from raw tensors to `Module` and `DataLoader`
- [Neural Networks: Zero to Hero, Andrej Karpathy](https://karpathy.ai/zero-to-hero.html)

**Modules, data, and autograd**
- [`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html)
- [`torch.Tensor` reference](https://docs.pytorch.org/docs/stable/tensors.html)
- [`nn.Module` reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [`torch.compiler` overview](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [`torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html) — `Dataset`, `DataLoader`, `collate_fn`, workers
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)

**Checkpoints**
- [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [safetensors](https://huggingface.co/docs/safetensors/index)

**Performance**
- [Automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)
- [`torch.compile` tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA semantics and memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html)

## About this lesson

Written to cover section 0.8 of the [Month 0 curriculum](../README.md). Code examples were checked with PyTorch 2.14 on CPU.
