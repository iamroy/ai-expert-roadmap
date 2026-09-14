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

Doing that accounting explicitly is the difference between a system that degrades gracefully and one that truncates mid-answer in production:

```python
def plan_context(window, system, tools, history, retrieved_chunk, max_output, reserve=0.1):
    overhead = int(window * reserve)          # wrappers, special tokens, safety margin
    fixed = system + tools + history + overhead + max_output
    room = window - fixed
    chunks = max(0, room // retrieved_chunk)
    return {
        "window": window, "system": system, "tools": tools, "history": history,
        "overhead+safety": overhead, "output reserved": max_output,
        "left for retrieval": room, "chunks that fit": chunks,
        "unused": room - chunks * retrieved_chunk,
    }


for window in (8_192, 32_768, 128_000):
    plan = plan_context(window, system=600, tools=1_400, history=2_500,
                        retrieved_chunk=700, max_output=1_000)
    print(f"window {window:>7,}: retrieval room {plan['left for retrieval']:>7,} "
          f"-> {plan['chunks that fit']:>3} chunks of 700")

print()
for key, value in plan_context(8_192, 600, 1_400, 2_500, 700, 1_000).items():
    print(f"  {key:<20} {value:>8,}")
```

```text
window   8,192: retrieval room   1,873 ->   2 chunks of 700
window  32,768: retrieval room  23,992 ->  34 chunks of 700
window 128,000: retrieval room 109,700 -> 156 chunks of 700

  window                  8,192
  system                    600
  tools                   1,400
  history                 2,500
  overhead+safety           819
  output reserved         1,000
  left for retrieval      1,873
  chunks that fit             2
  unused                    473
```

The 8k row is the one worth sitting with. An 8,192-token window sounds generous, but a realistic system prompt, tool schemas, and conversation history consume three quarters of it before a single retrieved document is added — leaving room for two chunks. A naive retriever configured to return the top 5 would overflow, and depending on the framework that either errors or silently drops chunks, usually the last ones.

Two implications for Month 4 and 5. Growing the window from 8k to 32k multiplies retrieval room by 13×, not by 4×, because the fixed overhead is paid once. And the accounting must run *before* retrieval, so `k` adapts to remaining room instead of being a constant that happens to fit during development.

## Lesson 1.7.2 — Where quadratic attention comes from

For one dense self-attention head over `T` positions, the score matrix has `T²` entries. For `B` batches and `H` heads, materializing the scores requires `BHT²` elements. QK and attention-times-V arithmetic scales roughly as `O(BT²D)` across heads; dense feature projections and FFNs also contribute, typically with terms proportional to `BTD²`.

Doubling sequence length therefore quadruples the pairwise attention work and score storage of the naive implementation, but does not necessarily quadruple total runtime. Other operations, hardware utilization, and memory access affect the measured result.

An example: one layer with batch 1, 8 heads, and 4,096 positions has `134,217,728` score entries. At two bytes per entry, that is 256 MiB for one materialized score tensor, before probabilities, other activations, gradients, or weights. The project exposes attention maps for inspection; that design becomes expensive at long lengths.

The growth rate is the part worth feeling rather than reading:

```python
def attention_scores_bytes(tokens, heads, dtype_bytes=2):
    return heads * tokens * tokens * dtype_bytes


print(f"{'tokens':>9} {'score matrix':>16} {'relative cost':>14}")
base = None
for tokens in (1_024, 4_096, 16_384, 131_072):
    gb = attention_scores_bytes(tokens, heads=32) / 1e9
    base = base or gb
    print(f"{tokens:>9,} {gb:>13.2f} GB {gb / base:>13.0f}x")
```

```text
   tokens     score matrix  relative cost
    1,024          0.07 GB             1x
    4,096          1.07 GB            16x
   16,384         17.18 GB           256x
  131,072       1099.51 GB         16384x
```

A 128× longer sequence costs 16,384× the score memory. At 128k the materialized scores for a single layer would need about 1.1 TB, which is why nobody materializes them: FlashAttention computes the same result in tiles and never stores the full matrix, and that is a memory-access change, not an approximation.

Contrast this with the KV cache from 1.5, which grows *linearly* in length. Long-context serving therefore has two different cost curves running at once — quadratic compute against linear cache — and which one binds depends on whether you are prefilling a long prompt or decoding token by token.

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

The analysis side is small enough to write once and reuse. Every cell needs a confidence interval, because per-slice sample sizes are small enough that noise imitates a finding:

```python
import math
import random
from collections import defaultdict


def summarize(results):
    by_slice = defaultdict(lambda: [0, 0])
    for r in results:
        cell = by_slice[(r["length"], r["position"])]
        cell[0] += int(r["correct"])
        cell[1] += 1

    rows = []
    for (length, position), (hits, n) in sorted(by_slice.items()):
        p = hits / n
        half_width = 1.96 * math.sqrt(p * (1 - p) / n)      # normal approximation
        rows.append((length, position, p, half_width, n))
    return rows


random.seed(0)
underlying = {("4k", "start"): 0.95, ("4k", "middle"): 0.92, ("4k", "end"): 0.96,
              ("32k", "start"): 0.90, ("32k", "middle"): 0.55, ("32k", "end"): 0.88}
results = [{"length": length, "position": position, "correct": random.random() < p}
           for (length, position), p in underlying.items() for _ in range(60)]

print(f"{'length':>7} {'position':>9} {'accuracy':>9} {'95% CI':>16} {'n':>4}")
for length, position, p, half_width, n in summarize(results):
    print(f"{length:>7} {position:>9} {p:>9.3f}   +/- {half_width:.3f}      {n:>4}")

overall = sum(r["correct"] for r in results) / len(results)
print(f"\noverall accuracy: {overall:.3f}  <- hides the 32k middle result")
```

```text
 length  position  accuracy           95% CI    n
    32k       end     0.817   +/- 0.098        60
    32k    middle     0.583   +/- 0.125        60
    32k     start     0.950   +/- 0.055        60
     4k       end     0.933   +/- 0.063        60
     4k    middle     0.917   +/- 0.070        60
     4k     start     0.950   +/- 0.055        60

overall accuracy: 0.858  <- hides the 32k middle result
```

The simulated data has a real middle-of-context failure at 32k and none at 4k. An 85.8% headline looks like a mostly-working system; the slice table shows a 0.583 cell that a user would experience as the model ignoring the middle of their document. This is why the deliverable is a table, not a number.

Note the interval widths. At 60 trials per cell the 95% interval spans roughly ±0.10, so a 4-point difference between two slices is not evidence of anything. Size each cell for the effect you need to detect before running the experiment, and treat the intervals as a lower bound on uncertainty, since repeated templates and shared distractors make trials less independent than this formula assumes.

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

## Videos and code to read

- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — the tiled exact-attention kernel that makes the 1.1 TB score matrix in this lesson unnecessary
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — PagedAttention, the answer to cache fragmentation at long context
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — infrastructure for running the position-sensitive evaluation this lesson asks you to design

## Mapped companion lessons

- [KV Cache and Flash Attention](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/12-kv-cache-flash-attention) maps to attention memory and serving cost.
- [Native Sparse Attention](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/10-llms-from-scratch/17-native-sparse-attention) extends the architectural approaches to longer sequences.
- [Long-Context Evaluation](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/28-long-context-evaluation) and [Context Engineering](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/11-llm-engineering/05-context-engineering) map to useful-context measurement and budget design.

- [Prompt Caching and Context Caching](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/11-llm-engineering/15-prompt-caching) maps to prefix reuse against the budget in 1.7.1.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).
