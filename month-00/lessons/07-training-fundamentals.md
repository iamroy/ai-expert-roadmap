# 0.7 — Training Fundamentals

**Depth: SKIM**

**Goal:** operate and diagnose a training loop, including optimizers, schedules, clipping, validation, checkpointing, and reproducibility.

[Month 0 roadmap](../README.md) · [Previous: Neural Networks](06-neural-network-fundamentals.md) · [Next: PyTorch](08-pytorch-fundamentals.md)

## Batches, epochs, and steps

An example is one record. A batch is the set used for one gradient estimate. A step usually means one optimizer update. An epoch is one pass through the training dataset, but streaming, sampling with replacement, and token-based training can make epochs less useful than processed examples or tokens.

With dataset size `N` and batch size `B`, one ordinary epoch has `ceil(N/B)` batches. If gradients accumulate across `K` microbatches, an optimizer step represents up to `B×K` examples. Loss scaling during accumulation must preserve the intended effective-batch average.

## SGD, momentum, Adam, and AdamW

SGD updates parameters using the current gradient. Momentum maintains a moving direction, smoothing noisy gradients and accelerating persistent ones. Adam maintains moving estimates of the gradient and squared gradient, applying coordinate-wise adaptive scaling.

AdamW decouples weight decay from the adaptive gradient update. This differs from simply adding an L2 penalty inside Adam's gradient because adaptive scaling changes that penalty's effect. Biases and normalization parameters are often excluded from decay, but this is a design choice to validate.

Optimizer state consumes memory. Adam-like methods normally keep two state tensors per trained parameter in addition to parameters and gradients; mixed-precision training may retain master weights too.

## Learning-rate schedules and warmup

The learning rate controls update scale. Too high can diverge; too low can waste compute or settle slowly. Warmup increases the rate over initial steps while activations, gradients, and optimizer estimates stabilize. A decay schedule then reduces it, often linearly or with cosine shape.

Schedulers disagree on whether they advance per batch, optimizer step, or epoch. With gradient accumulation, step an update-based schedule when the optimizer actually updates. Log the effective rate so an off-by-one error is visible.

## Gradient clipping and accumulation

Global-norm clipping rescales all gradients when their combined norm exceeds a threshold. Clip after backward and after any mixed-precision unscaling, immediately before the optimizer step. Clipping every step can indicate a deeper issue.

For `K` equal-size microbatches, divide each mean loss by `K` before backward, or sum gradients and divide equivalently before the update. Unequal token counts require token-weighted normalization. Zero gradients at the start of each accumulation cycle, not each microbatch.

## Validation, overfitting, and early stopping

Training loss measures optimization data; validation estimates generalization to an independent deployment-like sample. Use evaluation mode, disable gradients, and aggregate losses by valid example/token count. Track metrics meaningful to the task alongside loss.

Early stopping selects a checkpoint based on validation behavior. Reusing the same validation set for many decisions can overfit it. Keep a final test set untouched until model and thresholds are fixed.

Overfitting appears as improving train metrics with stagnant or worsening validation metrics. Responses include better data splits, regularization, data augmentation, reduced capacity, or earlier stopping. Underfitting may need capacity, better features, longer training, or optimization fixes.

## Checkpoints and reproducibility

An inference checkpoint needs model architecture/configuration, model state, tokenizer/preprocessing contract, and version metadata. Exact training resumption additionally needs optimizer and scheduler state, step/epoch position, random-number-generator states, data sampler state, and mixed-precision scaler state where used.

A random seed is necessary but not sufficient for reproducibility. Hardware, library versions, nondeterministic kernels, thread scheduling, data ordering, and floating-point reduction order can change results. Record the environment and report variation across runs when conclusions depend on small differences.

## A reliable step order

```text
clear gradients at cycle start → load microbatch → forward
    → scaled loss → backward → repeat microbatches if accumulating
    → unscale if needed → clip → optimizer step → scheduler step
    → record metrics
```

Validate periodically, save the best qualifying checkpoint plus the latest resumable state, and verify reload behavior before a long run finishes.

## Checkpoint

1. How do microbatch size, accumulation steps, and effective batch size differ?
2. Why is AdamW's weight decay called decoupled?
3. At what point should gradients be clipped in mixed-precision training?
4. What state is missing if a checkpoint stores only model weights and must resume exactly?

<details>
<summary>Show answers</summary>

1. A microbatch is one forward/backward unit. Several may accumulate before one optimizer update; their total valid examples or tokens form the effective batch.
2. The decay is applied directly to parameters rather than entering the gradient that Adam rescales using its adaptive statistics.
3. After scaled gradients are unscaled and before the optimizer step.
4. At least optimizer, scheduler, RNG, data-order/sampler, current step, and any precision-scaler state, plus the matching code/configuration.

</details>

## Exercise — Plan a training run

A dataset has 10,000 examples. Microbatch size is 25, accumulation is 4, and training lasts 3 epochs with no dropped remainder. Calculate batches per epoch, optimizer steps per full epoch, nominal effective batch, and approximate total optimizer steps. Then explain how a final partial accumulation should be handled.

<details>
<summary>Show exercise solution</summary>

There are `10,000/25 = 400` microbatches per epoch. Four per update gives 100 optimizer steps per epoch, an effective batch of 100 examples, and 300 steps over three epochs.

If the count were not divisible by four, do not silently discard or overweight the remainder. Normalize gradients by its actual example/token count and take a final update, or deliberately drop it and document that policy. A scheduler based on optimizer steps must use the resulting update count.

</details>

## Completion criteria

Explain one update end to end, choose among basic optimizers, place warmup/scheduling/clipping correctly, aggregate validation metrics, and define a reproducible resumable checkpoint.

## Primary references

- [PyTorch optimization](https://docs.pytorch.org/docs/stable/optim.html)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html)
