# 0.9 — Deep-Learning Architecture Concepts

**Depth: SKIM**

**Goal:** compare how architectures encode locality and context, and reason about pretrained backbones, freezing, and fine-tuning.

[Month 0 roadmap](../README.md) · [Previous: PyTorch](08-pytorch-fundamentals.md) · [Next: Representation Learning](10-representation-learning.md)

## Features and hierarchical representations

A learned feature is a representation useful to downstream computation. Early vision layers often respond to local edges or textures; deeper features can combine them into motifs and object-level evidence. This hierarchy is an observed tendency, not a guarantee that every neuron has a clean human label.

In language models, token embeddings become contextual representations through repeated attention and feed-forward updates. The same word can have different later representations depending on surrounding text.

## Receptive fields and parameter sharing

A unit's receptive field is the part of the input that can affect it. A convolutional kernel sees a local neighborhood and shares the same weights across spatial positions. Stacking layers expands the theoretical receptive field, though the effective influence may be concentrated near the center.

Self-attention can connect each query to every allowed key in one layer. It shares projection parameters across positions but computes content-dependent mixing weights. Full attention's global connectivity costs pairwise work; convolution encodes locality and translation-related structure more directly.

```text
convolution: fixed local pattern of connections + content-independent kernel weights
attention:   allowed connection pattern + content-dependent mixing weights
```

Neither mechanism is always superior. Vision systems often combine convolutional inductive bias, patch embeddings, attention, or hierarchical windows.

## Transfer learning and pretrained backbones

A pretrained backbone supplies representations learned on a source objective and dataset. A task-specific head maps them to the target. Transfer helps when features generalize and target labels are limited.

Domain mismatch can make a backbone ineffective or misleading. Validate target-domain slices rather than assuming large-scale pretraining transfers uniformly. The preprocessing contract—image normalization, tokenizer, input size, channel order—must match the checkpoint.

## Frozen versus trainable layers

Freezing a parameter means excluding it from gradient updates by disabling gradients or omitting it from the optimizer. `model.eval()` is different: it changes dropout/normalization behavior but does not freeze weights.

Common strategies are:

- Train a new head on frozen backbone features for a fast baseline.
- Unfreeze upper layers or the full model with a smaller learning rate.
- Use different learning rates for backbone and head.
- Apply parameter-efficient adapters while keeping most base weights fixed.

After changing `requires_grad`, construct or update optimizer parameter groups intentionally. A frozen backbone can still consume activation memory if gradients are needed through it for earlier trainable components; ordinary head-only training can run the backbone under `no_grad()` and cache features when augmentation/preprocessing permits.

## Architectural tradeoffs

Inductive bias narrows the set of functions a model learns easily. Locality and weight sharing make convolutions data-efficient for spatial patterns. Attention's flexible content-dependent routing helps integrate distant information. Recurrence maintains sequential state with parameter sharing over time.

Choose architecture using data, task, latency, memory, resolution/sequence length, and deployment hardware. Parameter count alone misses activation cost and operator efficiency.

## Checkpoint

1. How does a theoretical receptive field differ from effective influence?
2. What does convolution share, and what does attention compute dynamically?
3. Why does `model.eval()` not freeze a backbone?
4. What can invalidate transfer even when tensor shapes match?

<details>
<summary>Show answers</summary>

1. Theoretical receptive field lists all inputs with a graph path to a unit; effective influence describes how strongly they affect it after learning.
2. Convolution shares kernel parameters across locations. Attention shares Q/K/V projections but computes input-dependent weights between allowed positions.
3. Evaluation mode changes layer behavior but leaves `requires_grad` and optimizer membership unchanged.
4. Different preprocessing, tokenization, domain, label meaning, or training objective can make compatible shapes semantically incompatible.

</details>

## Exercise — Design a transfer-learning experiment

Design three phases for a small image classifier: frozen backbone, partial unfreezing, and full fine-tuning. Specify controls, metrics, and evidence that would justify the added cost of unfreezing.

<details>
<summary>Show exercise solution</summary>

Use the same train/validation split, preprocessing, head capacity, and evaluation slices. Phase 1 trains only the head. Phase 2 unfreezes a defined upper block with a lower backbone learning rate. Phase 3 unfreezes all layers, again controlling learning rates and schedule.

Record trainable parameters, peak memory, examples/second, wall time, validation metric, calibration, and difficult target-domain slices. Added cost is justified only when repeated runs show a meaningful held-out improvement under the deployment constraints. Ensure optimizer groups contain exactly the intended trainable parameters after each transition.

</details>

## Completion criteria

Explain hierarchical features, receptive fields, parameter sharing, transfer learning, and the difference among frozen parameters, evaluation mode, and cached features.

## Primary references

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
