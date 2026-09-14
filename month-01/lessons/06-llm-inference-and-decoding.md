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

## Lesson 1.6.2 — The KV cache, demonstrated

Prefill and decode differ because of one data structure. Without it, generating token `n` reprocesses all `n-1` previous tokens, so producing `N` tokens costs `O(N²)` work. The cache stores each layer's keys and values as they are computed, so each new token attends to stored state and only ever computes its own K and V.

The claim that this changes nothing about the output is worth proving rather than believing:

```python
import math

import torch
from torch import nn

torch.manual_seed(0)
V, D, H, L = 64, 32, 4, 2


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.qkv = nn.ModuleList([nn.Linear(D, 3 * D) for _ in range(L)])
        self.out = nn.Linear(D, V)

    def block(self, i, x, cache=None):
        B, T, _ = x.shape
        projection = self.qkv[i]            # bind first, then call
        q, k, v = projection(x).reshape(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
        new = None
        if cache is not None:
            past_k, past_v = cache
            if past_k is not None:                       # append to what we already have
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new = (k, v)

        Tq, Tk = q.size(2), k.size(2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D // H)
        # queries sit at the END of the key range, so the mask is offset by Tk - Tq
        mask = torch.ones(Tq, Tk, dtype=torch.bool).tril(diagonal=Tk - Tq)
        weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        context = (weights @ v).transpose(1, 2).reshape(B, Tq, D)
        return x + context, new

    def forward(self, ids, caches=None):
        x = self.emb(ids)
        new_caches = []
        for i in range(L):
            x, c = self.block(i, x, None if caches is None else caches[i])
            new_caches.append(c)
        return self.out(x), new_caches


model = TinyDecoder().eval()
prompt = torch.randint(0, V, (1, 6))

with torch.inference_mode():
    ids, uncached = prompt.clone(), []
    for _ in range(5):                                   # recompute everything each step
        logits, _ = model(ids)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        uncached.append(nxt.item())
        ids = torch.cat([ids, nxt], dim=1)

    logits, caches = model(prompt, caches=[(None, None)] * L)   # prefill
    nxt, cached = logits[:, -1].argmax(-1, keepdim=True), []
    for _ in range(5):                                   # decode one token at a time
        cached.append(nxt.item())
        logits, caches = model(nxt, caches=caches)
        nxt = logits[:, -1].argmax(-1, keepdim=True)

print("no cache:", uncached)
print("cached:  ", cached)
print("identical:", uncached == cached)
print("cached K per layer:", tuple(caches[0][0].shape))
```

```text
no cache: [44, 62, 17, 44, 39]
cached:   [44, 62, 17, 44, 39]
identical: True
cached K per layer: (1, 4, 11, 8)
```

Identical tokens, so the cache is a pure optimization with no effect on the distribution. Three details in that code are where real implementations go wrong:

- **The mask offset.** During decode, `Tq=1` and `Tk` is the full history, so a plain `tril` would mask almost everything. `diagonal=Tk - Tq` places the single query at the end of the key range. Getting this wrong is the most common KV-cache bug, and it usually shows as generation that degrades after the first token rather than as an exception.
- **The cache grows every step.** `(1, 4, 11, 8)` after 6 prompt tokens and 5 generated ones. Its size is the 1.5 formula, and it is why memory climbs during a long generation rather than being fixed at prefill.
- **Positions come from the cache length**, not from a counter you maintain separately, which matters as soon as RoPE (1.2) is applied.

This is why prefill and decode have different performance characters. Prefill is compute-bound: many tokens at once, high arithmetic intensity. Decode is memory-bandwidth-bound: one token, but the entire cache must be read at every step for every layer. Batching helps decode substantially because the weight reads are shared across sequences, while each sequence keeps its own cache — which is exactly the tension Month 9's serving work is about.

## Lesson 1.6.3 — Logits, probabilities, greedy selection

Logits are unnormalized real-valued scores, not probabilities. For temperature `τ > 0`:

```text
P(token=i) = exp(zi/τ) / Σj exp(zj/τ)
```

Subtracting the largest scaled logit before exponentiation leaves probabilities unchanged and improves numerical stability. Library softmax handles this.

Greedy decoding selects `argmax(logits)` at each step. It needs no softmax because softmax preserves ordering. Greedy chooses the locally most likely next token, not necessarily the globally most likely full sequence. One early choice changes all subsequent distributions.

Temperature below one sharpens a distribution; above one flattens it. It does not improve the model's underlying knowledge. Temperature zero is commonly an API shorthand for greedy decoding, not valid arithmetic in the softmax formula. Define that case separately.

## Lesson 1.6.4 — Top-k and top-p restrict the candidate set

Top-k keeps the `k` highest-scoring vocabulary items, masks others to negative infinity, then renormalizes. Its candidate count is fixed even when the model is highly certain or uncertain. For `k=1`, sampling selects a highest-logit token just like greedy, with tie behavior dependent on implementation.

Top-p, or nucleus sampling, sorts candidates by probability and retains the smallest leading set whose cumulative probability reaches or exceeds `p`. Its set size adapts to distribution sharpness. Retain the token that crosses the threshold and always keep at least one token.

For sorted probabilities `[0.50, 0.25, 0.15, 0.10]` with `p=0.80`, keep the first three: the cumulative mass reaches 0.90. Renormalization then gives approximately `[0.556, 0.278, 0.167]` over the retained candidates. [The nucleus-sampling paper](https://arxiv.org/abs/1904.09751) motivates restricting the unreliable tail in open-ended generation.

If combining temperature, top-k, and top-p, the ordering is part of the algorithm. Temperature changes probabilities used by top-p. Top-k truncation and renormalization also change those probabilities. Record the actual sequence, not merely three parameter values.

## Lesson 1.6.5 — A transparent sampler

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

## Lesson 1.6.6 — Beam search and sequence scores

Beam search retains several partial sequences. At each step, extend them with candidates, sum log probabilities, and keep the best scoring continuations under the chosen scoring rule. A beam width of one reduces to greedy for the corresponding scoring/stopping setup.

Suppose first-token probabilities are `A=0.6, B=0.4`; the best next-token probabilities after them are `0.4` and `0.9`. Greedy commits to A, producing a two-token path with probability 0.24. A beam retaining B can discover a 0.36 path. Wider search still does not guarantee that a finite beam finds the globally best full sequence.

Raw summed log probabilities tend to favor short completed sequences because each added log probability is nonpositive. Length normalization and stopping policies influence results. Beam search can be useful for constrained sequence tasks, but maximizing likelihood can produce repetitive or uninteresting open-ended prose. Compare task success, not just the score the search optimized.

## Lesson 1.6.7 — Repetition penalties and stopping

A frequency penalty subtracts a term based on how often a token already occurred; a presence penalty subtracts once for any prior occurrence. A multiplicative repetition penalty is a different rule and requires sign-aware handling of negative logits. Provider implementations vary, so name the exact rule in experiments.

Penalties may reduce loops while harming legitimate repetition in code, names, or structured records. They are not correctness or safety guarantees.

Stopping can use a learned EOS ID, an output-token limit, a time budget, or a configured stop sequence. Multi-token stop strings may cross generation or stream-chunk boundaries. Define whether stop markers are included in returned text, and avoid displaying a partial marker before detection if the interface promises to omit it. A length limit can leave text or JSON incomplete; return a termination reason to the application.

## Lesson 1.6.8 — Determinism and production tradeoffs

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

### Exercises 1 and 2 — the filter, and tests that actually constrain it

```python
def filter_logits(logits, temperature=1.0, top_k=None, top_p=None):
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    if top_p is not None and not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1]")

    out = logits / temperature
    if top_k is not None:
        k = min(top_k, out.size(-1))
        threshold = out.topk(k, dim=-1).values[..., -1:]
        out = out.masked_fill(out < threshold, float("-inf"))
    if top_p is not None:
        ordered, index = out.sort(dim=-1, descending=True)
        probs = ordered.softmax(-1)
        drop = (probs.cumsum(-1) - probs) >= top_p      # keep the token that crosses p
        ordered = ordered.masked_fill(drop, float("-inf"))
        out = torch.full_like(out, float("-inf")).scatter(-1, index, ordered)
    return out


probabilities = torch.tensor([0.55, 0.25, 0.15, 0.05])
logits = probabilities.log()

print("top-k=2:   ", filter_logits(logits, top_k=2).softmax(-1).round(decimals=6).tolist())
print("top-p=0.90:", filter_logits(logits, top_p=0.90).softmax(-1).round(decimals=6).tolist())

# identity cases: neither filter should alter the distribution
assert torch.allclose(filter_logits(logits).softmax(-1), probabilities, atol=1e-6)
assert torch.allclose(filter_logits(logits, top_p=1.0).softmax(-1), probabilities, atol=1e-6)

# top-k=1 is greedy: all mass on the argmax
assert torch.allclose(filter_logits(logits, top_k=1).softmax(-1).max(), torch.tensor(1.0))

# a first token that alone exceeds p must still be kept, never an empty set
dominant = torch.tensor([0.97, 0.02, 0.01]).log()
assert (filter_logits(dominant, top_p=0.5).softmax(-1) > 0).sum().item() == 1

# survivors always renormalize to a valid distribution
assert torch.allclose(filter_logits(logits, top_k=2).softmax(-1).sum(), torch.tensor(1.0))

for bad in [dict(temperature=0), dict(top_k=0), dict(top_p=0), dict(top_p=1.5)]:
    try:
        filter_logits(logits, **bad)
        raise AssertionError(f"should have rejected {bad}")
    except ValueError:
        pass

print("all sampler tests pass")
```

```text
top-k=2:    [0.6875, 0.3125, 0.0, 0.0]
top-p=0.90: [0.578947, 0.263158, 0.157895, 0.0]
```

These match the hand-computed values in Exercise 1. The dominant-token case is the one worth keeping: a naive `cumsum >= top_p` mask drops *every* token when the first already exceeds `p`, leaving nothing to sample from. Subtracting `probs` before comparing keeps the crossing token, which is why that filter is written the way it is.

### Exercise 3 — comparing strategies

Hold everything fixed except the sampler: same prompts, same seeds, same output length, same sampler ordering. Record the settings alongside the text, and count concrete failures — verbatim repetition, truncation, topic drift — rather than judging fluency by eye.

There is no universal winning setting, and a small byte-level model may look poor under all of them. Report that honestly instead of reading random variation as a breakthrough. The useful output of this exercise is a table you can rerun, not a favorite temperature.

### Exercise 4 — a stop sequence split across chunks

```python
class StopDetector:
    """Emits text safe to show, holding back any suffix that might begin the marker."""

    def __init__(self, stop: str):
        self.stop, self.pending = stop, ""

    def feed(self, chunk: str) -> tuple[str, bool]:
        self.pending += chunk
        hit = self.pending.find(self.stop)
        if hit != -1:
            emit, self.pending = self.pending[:hit], ""
            return emit, True

        keep = len(self.stop) - 1                  # any shorter suffix may start a match
        if keep:
            emit, self.pending = self.pending[:-keep], self.pending[-keep:]
        else:
            emit, self.pending = self.pending, ""
        return emit, False


for chunks in (["Hello <|", "END|> ignored"],
               ["Hello <|END|> ignored"],
               ["Hel", "lo <", "|EN", "D|>x"],
               ["no marker here"]):
    detector, shown, stopped = StopDetector("<|END|>"), "", False
    for chunk in chunks:
        emitted, done = detector.feed(chunk)
        shown += emitted
        if done:
            stopped = True
            break
    print(f"{str(chunks)[:40]:42} -> {shown!r:14} stopped={stopped}")
```

```text
['Hello <|', 'END|> ignored']              -> 'Hello '       stopped=True
['Hello <|END|> ignored']                  -> 'Hello '       stopped=True
['Hel', 'lo <', '|EN', 'D|>x']             -> 'Hello '       stopped=True
['no marker here']                         -> 'no marke'     stopped=False
```

The marker is caught identically whether it arrives whole, split in two, or spread across four chunks. The essential move is holding back `len(stop) - 1` characters: searching each chunk independently misses any marker straddling a boundary, and emitting everything immediately leaks marker fragments to the user. The last row shows the cost — the final few characters stay buffered until the stream ends, so flush `pending` on completion or the tail is silently dropped.

</details>

## Completion criteria

Implement the four required generation modes, validate filtering edge cases, explain beam-search scoring, and report a reproducible comparison with termination reasons and latency context.

## Primary references

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — nucleus sampling and generation quality.
- [PyTorch softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html).
- [PyTorch multinomial](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html).
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Videos and code to read

- [huggingface/transformers](https://github.com/huggingface/transformers) — the `generate` implementation and its logits processors are the reference for every sampler in this lesson
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — `model.py`'s `generate` is the whole decode loop in about 20 lines
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — paged KV cache and continuous batching, which is where 1.6.2's cache arithmetic leads in production

## Mapped companion lessons

- [Sampling Methods](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/01-math-foundations/16-sampling-methods) reinforces the probability mechanics beneath decoding.
- [Inference Optimization](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/12-inference-optimization) maps to autoregressive serving, caching, batching, and latency.
- [Speculative Decoding](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/16-speculative-decoding) extends the lesson beyond single-model token selection.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).
