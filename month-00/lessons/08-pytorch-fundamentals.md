# 0.8 — PyTorch Fundamentals

**Depth: SKIM / REFRESH**

**Goal:** build, train, validate, save, and reload a small model without a high-level trainer while maintaining correct shapes, devices, modes, and gradients.

[Month 0 roadmap](../README.md) · [Previous: Training](07-training-fundamentals.md) · [Next: Architecture Concepts](09-deep-learning-architecture-concepts.md)

## Tensors, shapes, devices, and autograd

Every tensor has a shape, data type, device, and storage/layout. Integer token IDs commonly use `torch.long`; model parameters and activations use floating types; masks are Boolean or deliberately additive. Operations require compatible shapes, devices, and dtypes.

Move a model and its input to the same device. Avoid creating new CPU tensors inside a GPU forward pass; prefer `x.new_zeros(...)`, registered buffers, or an explicit device. Review [0.2](02-numpy-and-tensor-manipulation.md) for shape operations.

Autograd records operations involving tensors that require gradients. Leaf parameters accumulate `.grad` after backward. `detach()` breaks a gradient path; `no_grad()` disables recording for a scope. Avoid `.item()` on a value that must still contribute to the loss.

## `nn.Module` and `forward`

A module registers parameters, buffers, and child modules assigned as attributes. Registration lets `.parameters()`, `.state_dict()`, `.to(device)`, and train/evaluation mode traverse the model.

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, features):
        if features.ndim != 2:
            raise ValueError("expected [batch, features]")
        return self.network(features)

model = MLP(8, 16, 3)
logits = model(torch.randn(4, 8))
assert logits.shape == (4, 3)
```

Call `model(inputs)`, not `model.forward(inputs)`, so module hooks and wrapper behavior run. Store layers in `ModuleList` or `Sequential`; a plain Python list does not register its parameters.

## `Dataset` and `DataLoader`

A map-style `Dataset` implements `__len__` and `__getitem__`. An `IterableDataset` streams records and must partition work correctly across workers to avoid duplication. A `DataLoader` handles batching, shuffling, worker processes, and collation.

Variable-length sequences need a `collate_fn` that pads a batch and returns lengths or masks. Shuffle training data when appropriate, but not when temporal order is part of the task. More workers can increase throughput while complicating reproducibility and consuming memory.

```python
from torch.utils.data import DataLoader, TensorDataset

features = torch.randn(100, 8)
labels = torch.randint(0, 3, (100,))
loader = DataLoader(TensorDataset(features, labels), batch_size=16, shuffle=True)
```

## A complete training and validation loop

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(8, 16, 3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

model.train()
for features, labels in loader:
    features, labels = features.to(device), labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(features)
    loss = loss_fn(logits, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

Validation uses `model.eval()` and a no-gradient scope. Sum per-example losses or correct counts, then divide by the number of examples. Restore training mode before the next training batch. `eval()` affects dropout and normalization behavior but does not freeze parameters.

## Checkpoint save and load

Prefer `state_dict` plus an explicit, versioned configuration over serializing an entire live model object:

```python
path = "mlp.pt"
torch.save({"model": model.state_dict(), "config": {"in": 8, "hidden": 16, "classes": 3}}, path)
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
restored = MLP(checkpoint["config"]["in"], checkpoint["config"]["hidden"],
               checkpoint["config"]["classes"])
restored.load_state_dict(checkpoint["model"])
restored.eval()
```

Treat untrusted serialized artifacts as untrusted input. Verify expected keys, tensor shapes, model/tokenizer versions, and provenance. Exact training resumption also needs optimizer and run state, as covered in 0.7.

## Mixed precision, compilation, and memory

Automatic mixed precision uses lower-precision operations where appropriate while retaining higher precision where needed. On CUDA, gradient scaling can prevent small float16 gradients from underflowing; bfloat16 has a wider exponent range and typically does not use the same scaling. Follow the current PyTorch AMP API for the target device.

`torch.compile` can capture and optimize compatible execution. Dynamic shapes, graph breaks, and initial compilation cost affect whether it helps. Measure steady-state results after warmup and keep an eager-mode correctness baseline.

GPU memory includes parameters, gradients, optimizer states, activations saved for backward, temporary workspaces, and allocator cache. `empty_cache()` does not free live tensors or fix a retained computation graph. Common leaks include appending graph-connected losses or outputs to a Python list; store `.item()` or detached CPU summaries instead.

## Checkpoint

1. Why can a layer stored in a plain list remain untrained?
2. What separate effects do `eval()` and `no_grad()` have?
3. Why might validation run out of memory even though backward is never called?
4. What should be tested after reloading a checkpoint?

<details>
<summary>Show answers</summary>

1. A plain list does not register child modules, so their parameters may be absent from `.parameters()`, device moves, and the state dictionary. Use `ModuleList` or another registered container.
2. `eval()` changes module behavior such as dropout; `no_grad()` stops autograd graph recording.
3. Graphs can still be recorded unless gradients are disabled, and retained outputs can keep large tensors alive. Large batches and temporary activations also consume memory.
4. Configuration and key compatibility, inference mode, expected shapes, identical or tolerance-close logits on a fixed fixture, and the tokenizer/preprocessing contract.

</details>

## Exercise — Implement the required tiny MLP loop

Create a synthetic three-class dataset, train an MLP, compute token/example-weighted validation loss and accuracy, save/reload the state, and assert restored logits match on a fixed batch.

<details>
<summary>Show exercise solution</summary>

Use `TensorDataset`, split before creating loaders, and seed generation. The essential validation pattern is:

```python
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += loss_fn(logits, y).item() * y.numel()
            correct += (logits.argmax(-1) == y).sum().item()
            count += y.numel()
        if count == 0:
            raise ValueError("empty validation loader")
        return loss_sum / count, correct / count
    finally:
        model.train(was_training)
```

Use a loss configured with mean reduction as shown. Save the CPU-compatible state and compare both models in evaluation mode on the same fixture with `torch.testing.assert_close`.

</details>

## Completion criteria

Implement the loop without a trainer, explain registration and modes, save/reload a checkpoint, and identify memory and mixed-precision failure modes.

## Primary references

- [PyTorch tensors](https://docs.pytorch.org/docs/stable/tensors.html)
- [PyTorch modules](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch data utilities](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch compilation](https://docs.pytorch.org/docs/stable/torch.compiler.html)
