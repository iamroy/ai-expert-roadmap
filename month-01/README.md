# Month 01 — Foundation Models & LLM Internals

**Purpose:** build a working mental model of what happens from the moment a prompt enters a language model until the next token comes out.

**Suggested duration:** 4 weeks

**Suggested effort:** 30–40 hours, including the project; adjust after the checkpoints.

This curriculum follows the eight-topic Month 1 roadmap in [Start Month 1](https://chatgpt.com/c/6aa21b71-4eb4-83ea-8ec8-2e03861de632) (the source conversation requires access). It assumes the [Month 0 prerequisites](../month-00/README.md), especially tensor shapes, matrix multiplication, gradients, and a basic PyTorch training loop. Familiar prerequisites should be skimmed and validated, not removed from the learning path.

## Learning approach

- **SKIM** — briefly review familiar prerequisites and validate them.
- **LEARN** — explain a mechanism, apply it, and diagnose its production tradeoffs.
- **MASTER** — derive the mechanism, implement it, test it, and defend alternatives.

Month 1 has three MASTER topics and five LEARN topics. All eight are required. The linked Jupyter notebooks combine the teaching text, runnable examples, learner work cells, and section-level readings or videos. Work through each lesson's examples before attempting its checkpoints without notes. Every supplied checkpoint, exercise, project solution, and exit-test answer is inside a collapsed disclosure. Open it only after writing your own attempt.

## Curriculum

### 1.1 Transformer Architecture — MASTER

- [Lesson: Transformer Architecture](lessons/01-transformer-architecture.ipynb)
- Encoder-only, decoder-only, encoder-decoder; transformer blocks; self-attention; multi-head attention; Q/K/V; scaled dot products; causal and padding masks; FFNs; residuals; LayerNorm; pre-norm versus post-norm; transformers versus recurrence.
- Validation: trace tensor shapes through a full block, implement causal attention, and verify that future tokens cannot affect earlier outputs.

### 1.2 Position Information — LEARN

- [Lesson: Position Information](lessons/02-position-information.ipynb)
- Why order needs representation; sinusoidal and learned positions; rotary position embeddings (RoPE); ALiBi; extrapolation; long-context implications.
- Validation: explain where each method enters computation and why extending an index range does not establish long-context quality.

### 1.3 Tokenization — LEARN

- [Lesson: Tokenization](lessons/03-tokenization.ipynb)
- Tokens versus words/characters/bytes; BPE; WordPiece; SentencePiece; vocabulary construction; special tokens; efficiency; multilingual cost; inference implications.
- Validation: audit a tokenizer's round trip, unknown-input policy, vocabulary compatibility, and token budget.

### 1.4 LLM Training Objectives — MASTER

- [Lesson: LLM Training Objectives](lessons/04-llm-training-objectives.ipynb)
- Autoregressive and masked language modeling; next-token targets; teacher forcing; cross-entropy; general capabilities; training versus inference.
- Validation: construct correctly shifted targets, calculate loss by hand, and debug leakage and invalid padding reductions.

### 1.5 Modern Transformer / LLM Variants — LEARN

- [Lesson: Modern Transformer and LLM Variants](lessons/05-modern-transformer-and-llm-variants.ipynb)
- Decoder-only stacks; dense versus sparse models; MoE routing; SwiGLU; RMSNorm; grouped-query attention (GQA); multi-query attention (MQA).
- Validation: read an architecture configuration and reason about active computation, resident weights, and KV-cache size separately.

### 1.6 LLM Inference & Decoding — MASTER

- [Lesson: LLM Inference and Decoding](lessons/06-llm-inference-and-decoding.ipynb)
- Autoregressive inference; logits and probabilities; greedy, temperature, top-k, top-p, beam search; repetition penalties; stop sequences; determinism; quality, diversity, and latency.
- Validation: implement and test samplers, describe prefill/decode and caching, and compare generation settings with fixed inputs.

### 1.7 Context Windows & Limits — LEARN

- [Lesson: Context Windows and Limits](lessons/07-context-windows-and-limits.ipynb)
- Context accounting; quadratic attention; long-context architectures; usable versus advertised context; lost-in-the-middle; extrapolation and reasoning limits.
- Validation: estimate attention/cache costs and design a position-sensitive context evaluation.

### 1.8 Scaling & Foundation Models — LEARN

- [Lesson: Scaling and Foundation Models](lessons/08-scaling-and-foundation-models.ipynb)
- Foundation models; parameters, tokens, compute; scaling laws; compute-optimal training; model size and capability; production selection.
- Validation: compare training allocations and serving costs without treating parameter count as a quality guarantee.

## Suggested weekly sequence

| Week | Study | Practical evidence |
|---|---|---|
| 1 | 1.1–1.3 | Attention shapes, position experiment, tokenizer audit |
| 2 | 1.4–1.5 | Shifted batches, cross-entropy, model/block implementation |
| 3 | 1.6–1.7 | Sampler checks, generation comparison, context budget |
| 4 | 1.8, project, exit test | Training report, architecture review, gap analysis |

## Hands-on validation project

[Build a small GPT-like transformer in PyTorch](project/tiny-gpt/README.md):

```text
text → tokenizer → token IDs → embeddings + positions
     → causal transformer blocks → final norm → LM head → logits
     ├─ training: shifted targets → cross-entropy → gradients → AdamW
     └─ inference: final-position logits → sampling → append token → repeat
```

The project includes milestones, diagnostic tests, experiments, and a collapsed runnable reference. Start with explicit attention so Q/K/V and attention weights remain inspectable. Efficient kernels and a KV cache are extensions after correctness.

## Exit criteria

Complete the [Month 1 exit test](exit-test.md), the project evidence checklist, and [progress notes](progress.md). You should be able to trace an entire forward pass and training step without relying on a framework diagram, explain the principal modern architecture substitutions, and distinguish model behavior from decoding behavior.

Next planned module: **Month 02 — Prompting, Context Engineering & Structured Generation**. It builds on context behavior and the logits-to-sampling interface studied here.

[Back to the full roadmap](../README.md)
