# 1.6 — LLM Inference & Decoding

**Depth: MASTER**

**Goal:** implement the transition from vocabulary logits to output tokens, and connect sampling choices to quality, reproducibility, and serving cost.

[Month 1 roadmap](../README.md) · [Previous: Modern Variants](05-modern-transformer-and-llm-variants.md) · [Next: Context Windows](07-context-windows-and-limits.md)

## Lesson 1.6.1 — Prefill and decode

A model forward pass over a prompt returns a distribution at every position, but ordinary generation starts with the **last position's logits**. Choose a token, append it, and repeat until a stop condition is reached.

```text
prompt → prefill → last logits → select ID → append ID
                              ↑                  │
                              └──── decode ──────┘
```

Prefill processes the known prompt. With a KV cache, later decode steps compute new-token states and attend to saved keys and values. The cache contains activations, not new learned weights. It saves recomputing previous states, but the new query still reads preceding keys/values.

The project's first implementation recomputes a bounded prefix for clarity. It is correct for its documented window policy but less efficient than cached decoding. Cache extensions need continuing position IDs, correct masking for nonsquare query/key lengths, and cached-versus-uncached equivalence tests.

## Lesson 1.6.2 — Logits, probabilities, greedy selection

Logits are unnormalized real-valued scores, not probabilities. For temperature `τ > 0`:

```text
P(token=i) = exp(zi/τ) / Σj exp(zj/τ)
```

Subtracting the largest scaled logit before exponentiation leaves probabilities unchanged and improves numerical stability. Library softmax handles this.

Greedy decoding selects `argmax(logits)` at each step. It needs no softmax because softmax preserves ordering. Greedy chooses the locally most likely next token, not necessarily the globally most likely full sequence. One early choice changes all subsequent distributions.

Temperature below one sharpens a distribution; above one flattens it. It does not improve the model's underlying knowledge. Temperature zero is commonly an API shorthand for greedy decoding, not valid arithmetic in the softmax formula. Define that case separately.

## Lesson 1.6.3 — Top-k and top-p restrict the candidate set

Top-k keeps the `k` highest-scoring vocabulary items, masks others to negative infinity, then renormalizes. Its candidate count is fixed even when the model is highly certain or uncertain. For `k=1`, sampling selects a highest-logit token just like greedy, with tie behavior dependent on implementation.

Top-p, or nucleus sampling, sorts candidates by probability and retains the smallest leading set whose cumulative probability reaches or exceeds `p`. Its set size adapts to distribution sharpness. Retain the token that crosses the threshold and always keep at least one token.

For sorted probabilities `[0.50, 0.25, 0.15, 0.10]` with `p=0.80`, keep the first three: the cumulative mass reaches 0.90. Renormalization then gives approximately `[0.556, 0.278, 0.167]` over the retained candidates. [The nucleus-sampling paper](https://arxiv.org/abs/1904.09751) motivates restricting the unreliable tail in open-ended generation.

If combining temperature, top-k, and top-p, the ordering is part of the algorithm. Temperature changes probabilities used by top-p. Top-k truncation and renormalization also change those probabilities. Record the actual sequence, not merely three parameter values.

## Lesson 1.6.4 — A transparent sampler

This teaching implementation accepts a one-dimensional finite logit vector. It applies temperature, then top-k, then top-p using the post-top-k distribution. Setting `top_p=1` disables nucleus filtering. It rejects invalid parameters rather than silently guessing.

```python
import math
import torch


def sampling_probs(logits, temperature=1.0, top_k=None, top_p=1.0):
    if logits.ndim != 1 or logits.numel() == 0 or not torch.isfinite(logits).all():
        raise ValueError("expected a nonempty finite 1D logit vector")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    scores = logits.float() / temperature
    if not torch.isfinite(scores).all():
        raise ValueError("temperature caused logit overflow")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= scores.numel():
            raise ValueError("top_k must be an integer in [1, vocabulary size]")
        keep = torch.topk(scores, top_k).indices
        filtered = torch.full_like(scores, float("-inf"))
        filtered[keep] = scores[keep]
        scores = filtered
    ordered, indices = torch.sort(scores, descending=True)
    probs = ordered.softmax(dim=-1)
    # Mass strictly before each candidate: keep the crossing token.
    mass_before = probs.cumsum(dim=-1) - probs
    if top_p < 1:
        ordered = ordered.masked_fill(mass_before >= top_p, float("-inf"))
    ordered_probs = ordered.softmax(dim=-1)
    result = torch.zeros_like(ordered_probs)
    result.scatter_(0, indices, ordered_probs)
    return result


logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
probs = sampling_probs(logits, temperature=0.8, top_k=3, top_p=0.9)
assert torch.isclose(probs.sum(), torch.tensor(1.0))
generator = torch.Generator().manual_seed(42)
chosen = torch.multinomial(probs, 1, generator=generator).item()
```

Floating-point values near a cumulative threshold can affect membership; tests should use tolerances and unambiguous fixtures. This simple function is for teaching, not a drop-in replacement for a batched production decoding engine.

## Lesson 1.6.5 — Beam search and sequence scores

Beam search retains several partial sequences. At each step, extend them with candidates, sum log probabilities, and keep the best scoring continuations under the chosen scoring rule. A beam width of one reduces to greedy for the corresponding scoring/stopping setup.

Suppose first-token probabilities are `A=0.6, B=0.4`; the best next-token probabilities after them are `0.4` and `0.9`. Greedy commits to A, producing a two-token path with probability 0.24. A beam retaining B can discover a 0.36 path. Wider search still does not guarantee that a finite beam finds the globally best full sequence.

Raw summed log probabilities tend to favor short completed sequences because each added log probability is nonpositive. Length normalization and stopping policies influence results. Beam search can be useful for constrained sequence tasks, but maximizing likelihood can produce repetitive or uninteresting open-ended prose. Compare task success, not just the score the search optimized.

## Lesson 1.6.6 — Repetition penalties and stopping

A frequency penalty subtracts a term based on how often a token already occurred; a presence penalty subtracts once for any prior occurrence. A multiplicative repetition penalty is a different rule and requires sign-aware handling of negative logits. Provider implementations vary, so name the exact rule in experiments.

Penalties may reduce loops while harming legitimate repetition in code, names, or structured records. They are not correctness or safety guarantees.

Stopping can use a learned EOS ID, an output-token limit, a time budget, or a configured stop sequence. Multi-token stop strings may cross generation or stream-chunk boundaries. Define whether stop markers are included in returned text, and avoid displaying a partial marker before detection if the interface promises to omit it. A length limit can leave text or JSON incomplete; return a termination reason to the application.

## Lesson 1.6.7 — Determinism and production tradeoffs

A fixed seed controls random sampling in a fixed environment. It does not promise identical outputs across hardware, kernels, library versions, model revisions, or batching behavior. Greedy eliminates sampling randomness but may still differ when numerical changes alter close scores. Log model/tokenizer versions, prompt serialization, sampling order, seeds, and execution settings.

Measure time to first token separately from time per subsequent token. Long prompts increase prefill work; long outputs require more sequential decode steps. Large batches can improve throughput while increasing per-request waiting time. The best strategy depends on task success, diversity, repeatability, cost, and latency budgets.

Month 2's constrained generation adds another operation at this interface: a grammar can mask disallowed next tokens before sampling. It can control syntax without making the underlying facts true.

## Checkpoint

1. Which output positions are used for next-token training and which start ordinary generation?
2. Why is greedy not guaranteed to find the most likely full sequence?
3. How do top-k and top-p respond differently to a flat distribution?
4. What does a KV cache save, and what does it not eliminate?

<details>
<summary>Show answers</summary>

1. Training scores every valid target position. Generation begins with the last prompt position's logits, then iterates.
2. A locally likely token can lead to low-probability later choices. Search over complete paths differs from repeated local maximization.
3. Top-k keeps exactly k items (under this implementation); top-p may need many items to reach its mass threshold.
4. It reuses previous K/V activations and avoids recomputing their prefix states. It does not eliminate new-token computation, reads of preceding K/V, sequential token dependencies, or memory costs.

</details>

## Hands-on exercises

1. For sorted probabilities `[0.55,0.25,0.15,0.05]`, compute the retained set and renormalized probabilities for top-k=2 and top-p=0.90 separately.
2. Test the sampler's row sum, masked probabilities, top-k=1 behavior, top-p=1 behavior, and invalid parameters. Add a case in which the first token alone exceeds top-p.
3. Compare greedy, temperature-only, top-k, and top-p on five fixed project prompts with three seeds each where sampling applies. Keep output length fixed and record repetition and failure cases.
4. Design a correct detector for a stop string that arrives partly in one streamed chunk and partly in the next.

<details>
<summary>Show exercise solutions</summary>

1. Top-k=2 keeps the first two, renormalized to `[0.6875,0.3125]`. Top-p=0.90 keeps the first three (mass 0.95), yielding approximately `[0.578947,0.263158,0.157895]`.
2. Use finite fixtures without threshold ties; compare top-p=1 and no top-k to `softmax(logits/temperature)`. With a dominant first token and small p, only it remains. Reject empty/nonfinite vectors, nonpositive temperature, invalid k, and p outside `(0,1]`. Verify a token that crosses the threshold is retained.
3. There is no universal winning setting. Record prompts, seeds, sampler order, lengths, and observed text. A small byte model may be poor under every strategy; report that rather than interpreting random variation as a quality breakthrough.
4. Preserve sufficient trailing text or incremental matcher state across chunks. Search the accumulated pending suffix plus new text, stop at the first full match, and emit only the portion known not to be part of a marker. Do not assume each chunk is independently searchable.

</details>

## Completion criteria

Implement the four required generation modes, validate filtering edge cases, explain beam-search scoring, and report a reproducible comparison with termination reasons and latency context.

## Primary references

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — nucleus sampling and generation quality.
- [PyTorch softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html).
- [PyTorch multinomial](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html).
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
