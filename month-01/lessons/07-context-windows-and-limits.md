# 1.7 — Context Windows & Limits

**Depth: LEARN**

**Goal:** distinguish context capacity, computational cost, and useful information processing, then design tests that measure all three.

[Month 1 roadmap](../README.md) · [Previous: Inference and Decoding](06-llm-inference-and-decoding.md) · [Next: Scaling](08-scaling-and-foundation-models.md)

## Lesson 1.7.1 — What a context window contains

A context window is a bound on the token sequence available to a model under a particular inference configuration. It is not a persistent memory of all past conversations. An application may retain history externally, but only the content supplied or represented in the current computation can affect that request through context.

For a conventional decoder with a shared total sequence limit:

```text
instructions + serialized messages + examples + retrieved/tool content
+ generated continuation <= total context capacity
```

Serving systems may also impose separate input/output limits or account for additional internal tokens. Verify the exact model/service contract rather than assuming one advertised number covers every kind of token identically.

Count the final serialized request with its actual tokenizer. User-visible text alone omits message wrappers, special tokens, and tool schemas. Reserve output capacity before filling the input window. Hitting the capacity limit mid-response can truncate the result even when the prompt was accepted.

## Lesson 1.7.2 — Where quadratic attention comes from

For one dense self-attention head over `T` positions, the score matrix has `T²` entries. For `B` batches and `H` heads, materializing the scores requires `BHT²` elements. QK and attention-times-V arithmetic scales roughly as `O(BT²D)` across heads; dense feature projections and FFNs also contribute, typically with terms proportional to `BTD²`.

Doubling sequence length therefore quadruples the pairwise attention work and score storage of the naive implementation, but does not necessarily quadruple total runtime. Other operations, hardware utilization, and memory access affect the measured result.

An example: one layer with batch 1, 8 heads, and 4,096 positions has `134,217,728` score entries. At two bytes per entry, that is 256 MiB for one materialized score tensor, before probabilities, other activations, gradients, or weights. The project exposes attention maps for inspection; that design becomes expensive at long lengths.

## Lesson 1.7.3 — Efficient attention is not automatically sparse attention

[FlashAttention](https://arxiv.org/abs/2205.14135) computes exact attention with an IO-aware tiled algorithm, avoiding storage of the full attention matrix in high-bandwidth memory. It improves the memory/execution pattern without making dense attention cease to compare all relevant pairs. Floating-point ordering can still produce small numerical differences from another implementation.

Sliding-window attention restricts a token to nearby positions and can reduce pairwise work to approximately `O(Tw)` for a fixed window `w`. Global tokens or selected global layers can restore some long-range communication. Sparse attention changes which connections exist; it is an architectural choice, not the same as implementing dense attention more efficiently.

Other long-context designs may use recurrent state or compressed memory. Their ability to retain exact distant details depends on the design and training. Do not infer attention structure from an advertised context length.

## Lesson 1.7.4 — Prefill memory and decode cache are different

The KV cache grows approximately linearly with the number of retained tokens. For the conventional setup in lesson 1.5:

```text
KV bytes ≈ 2 × L × B × T × Hkv × Dh × element_bytes
```

Prefill may process a whole prompt at once. With caching, a decode step creates one new query per head and attends across the retained prefix. Its attention work grows with prefix length, while many projection/FFN operations apply only to the new token.

Long context can therefore pressure serving capacity even if one request fits comfortably: multiple simultaneous requests each need cache storage. GQA, cache quantization, sliding windows, and scheduling affect that budget differently. Record the assumptions behind each estimate.

## Lesson 1.7.5 — Advertised capacity versus useful capacity

A model may accept a long input yet fail to use information evenly across it. [Lost in the Middle](https://arxiv.org/abs/2307.03172) demonstrated position-sensitive performance in studied models and tasks, including poorer use of information located in the middle of context. Treat this as a failure mode to test, not a universal curve that every current model must have.

Recency effects, distractors, conflicting evidence, and the number of reasoning steps can all influence performance. Adding more text can introduce misleading correlations or hide the relevant evidence among plausible alternatives. Increasing capacity does not establish better reasoning.

Context extrapolation adds another issue: a model may encounter position ranges, rotary phases, or dependency distances outside training. A successful forward pass verifies shape and allocation compatibility, not competence at those distances.

## Lesson 1.7.6 — A context-quality experiment

Build synthetic documents with a fact whose answer you know. Vary total length and the fact's position independently, keeping the question and evidence meaning fixed. Use multiple fact templates and distractor sets to avoid memorizing one wording. Then add a second task requiring two distant facts to be combined.

Measure answer correctness, evidence identification, refusal/abstention, input/output tokens, latency, and run-to-run variation. Include a no-evidence case to detect guessing. Split development templates from evaluation templates so tuning the prompt does not merely overfit a benchmark.

A useful result table has one row per length and evidence-position slice. Reporting only the overall mean may hide a severe middle-position failure. Document which model/tokenizer revision and prompt format were tested; position sensitivity can change with both.

## Lesson 1.7.7 — Production implications and the Month 2 bridge

A practical context builder selects information under a capacity and quality budget:

```text
available sources → authorization → relevance/deduplication
                  → budget allocation → ordering → serialized prompt
```

Truncation, summarization, retrieval, and persistent memory are distinct mechanisms. Truncation discards text; summarization changes its representation and can omit details; retrieval selects evidence for the current request; external memory stores information between requests. Each requires provenance and evaluation.

For a legalistic policy clause, lossy summarization might remove an exception that changes the answer. For repeated conversational filler, compression may help. The right decision depends on the task, not a blanket instruction to “use more context.” Month 2 turns these observations into a context-engineering pipeline.

## Checkpoint

1. Why can a long-context model still fail on a question whose answer is present in its input?
2. What scales quadratically in naive attention, and what scales linearly in the conventional KV cache?
3. Does FlashAttention change dense attention into sliding-window attention?
4. Why reserve output tokens before packing retrieved documents?

<details>
<summary>Show answers</summary>

1. Capacity does not guarantee retrieval or reasoning quality; evidence position, distractors, conflicts, and unfamiliar distances can all matter.
2. Pairwise scores and dense attention arithmetic have quadratic length dependence. Retained K/V entries have linear length dependence for fixed layers, batch, and head configuration.
3. No. It changes how exact attention is computed and stored; sliding-window attention changes connectivity.
4. In a shared total-context budget, the continuation also occupies positions. An input that consumes the entire budget leaves no room for the requested response.

</details>

## Hands-on exercises

1. A total limit is 8,192 tokens. Reserve 1,024 for output and 512 for serialization/safety margin. Instructions use 700 and conversation history uses 1,500. How much remains for retrieved context?
2. Increase sequence length from 2,048 to 8,192 at fixed model/batch settings. Compare naive attention coefficients and KV-cache storage. Explain why this is not a complete latency prediction.
3. Design a six-slice long-context test using two lengths and three evidence positions. Include a control that separates retrieval from multi-fact reasoning.

<details>
<summary>Show exercise solutions</summary>

1. `8192−1024−512−700−1500 = 4456` tokens. Count final serialization with the real tokenizer and revise the margin based on observed overhead.
2. Length grows fourfold, pairwise coefficients sixteenfold, and conventional KV storage fourfold. Runtime also depends on projections, kernels, memory traffic, batching, and prefill/decode phase.
3. Use short/long documents with evidence near the beginning, middle, and end. Match distractor themes and answer difficulty across slices. Compare single-fact extraction against a task requiring two separated facts. Add no-evidence controls and repeated templates/seeds; record accuracy and latency per slice.

</details>

## Completion criteria

Estimate input/output capacity, attention storage, and KV storage separately. Design a test of useful context that includes position, distractors, and multi-fact reasoning.

## Primary references

- [FlashAttention](https://arxiv.org/abs/2205.14135) — exact attention and memory traffic.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — context utilization experiments.
- [Train Short, Test Long](https://arxiv.org/abs/2108.12409) — extrapolation and position bias.
