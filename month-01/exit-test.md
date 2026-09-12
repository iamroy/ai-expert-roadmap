# Month 01 — Exit Test

[Month 1 roadmap](README.md) · [Project](project/tiny-gpt/README.md) · [Progress notes](progress.md)

Attempt this after the eight lessons and the project. Spend about 60–90 minutes without notes. Show tensor shapes, arithmetic, and assumptions where requested. All supplied answers are collapsed below the questions.

## Part A — Mental models

1. Compare encoder-only, decoder-only, and encoder-decoder information flow. Give one appropriate task for each.
2. Explain Q, K, and V. Which axis receives softmax in attention, and why?
3. Explain the purpose of scaling dot products, residual connections, the FFN, and LayerNorm.
4. Compare learned positions, sinusoidal positions, RoPE, and ALiBi by injection point and extrapolation concern.
5. Distinguish BPE, WordPiece, and SentencePiece. Why must the tokenizer be versioned with a checkpoint?
6. Contrast autoregressive and masked language modeling. Explain teacher forcing and the training/inference difference.
7. Explain RMSNorm, SwiGLU, GQA/MQA, and sparse MoE without equating total parameters with active parameters.
8. Compare greedy, temperature, top-k, top-p, and beam search. State one limitation of each.
9. Distinguish a long advertised context, useful context, and persistent application memory.
10. Explain a foundation model, an empirical scaling law, and the distinction between training-optimal and serving-optimal model choices.

## Part B — Shapes and calculations

11. For `B=2, T=32, D=128, H=8, V=256`, trace token embeddings through Q/K/V head splitting, scores, merged attention, and vocabulary logits.
12. For stream `[9,4,7,2,6]` and sequence length 4, write inputs and targets. Which logit predicts 6? Why is the diagonal allowed in causal attention?
13. Correct-target probabilities are `[0.5,0.25,0.125]`. Compute mean cross-entropy and perplexity. For probabilities `[0.1,0.7,0.2]` and target index 1, give the logit gradient.
14. Sorted probabilities are `[0.6,0.2,0.15,0.05]`. What does top-p=0.85 retain, and what is its renormalized distribution?
15. Estimate FP16 KV storage for 12 layers, batch 2, length 2,048, 4 KV heads, and head dimension 64. How would 16 KV heads change it?
16. A model has a shared total context limit of 16,384. Output reservation is 2,048, formatting/margin 512, instructions 800, history 3,000, and examples 1,024. What remains for evidence?
17. Sequence length doubles at fixed batch, width, and heads. Compare naive attention-map size and conventional KV-cache size. Why might measured runtime have a different multiplier?
18. Compare approximate dense training compute for 300 million parameters × 4 billion tokens and 600 million parameters × 2 billion tokens. What quality conclusion is justified?

## Part C — Debugging and architecture

19. A developer masks future attention scores with zero, then applies softmax. Explain the failure and propose a test that would catch it.
20. A training loop feeds `softmax(logits)` into cross-entropy and uses unshifted inputs as targets in a custom model with no internal shift. Identify both problems.
21. Validation averages batch mean losses even though batches contain different numbers of valid targets. Correct the aggregation and handle an all-ignored batch.
22. A cached decoder restarts position IDs at zero for each new token. What comparison would reveal the bug? What assumptions must match?
23. A long-context assistant succeeds when evidence is at the end but fails in the middle. Design a controlled evaluation and two interventions to compare.
24. An MoE model advertises low active parameter count but does not fit within the serving memory budget. Explain plausible causes and the measurements needed.
25. Trace your project from raw text through one training update and one generated token. Identify at least three differences between it and a production pretrained language-model service.

## Scoring and exit criteria

Each question is worth 2 points: 2 for a correct explanation plus the requested calculation/example or caveat, 1 for partial understanding, and 0 for a missing or fundamentally wrong answer. Maximum: 50.

Aim for **40/50 or higher**, with full credit on questions **11, 12, 13, 19, 20, and 25**. These cover the core forward-pass, target, loss, causality, and end-to-end mechanics. Also complete the project's diagnostic tests and evidence checklist; a written score alone does not establish MASTER-level implementation skill. Revisit weak lessons and reattempt missed questions with fresh examples.

<details>
<summary>Show answers and grading guidance</summary>

### Part A answers

1. Encoder-only uses bidirectional input context, suitable for classification/representations. Decoder-only uses causal self-attention for generation. Encoder-decoder uses a source encoder plus causal target decoding with cross-attention, suitable for translation. Tasks are examples, not exclusive assignments.
2. Queries seek matches, keys supply matchable features, values supply mixed content. Softmax normalizes across keys for each query so its value mixture has nonnegative weights summing to one.
3. Scaling controls dot-product magnitude as head width changes; residuals carry representations and provide additive gradient paths; the FFN transforms token features nonlinearly; LayerNorm normalizes features within a token. They do different work.
4. Learned and sinusoidal vectors add to input representations; RoPE rotates Q/K; ALiBi biases scores. Learned tables have a row limit; formula-based schemes can represent new indices but still face unfamiliar distances and unverified quality.
5. BPE learns pair merges; WordPiece uses learned subword pieces and commonly greedy longest-match encoding; SentencePiece is a system supporting BPE/unigram methods. Vocabulary ID meanings, normalization, and special-token/chat-template conventions must match the weights.
6. Autoregressive LM predicts next tokens from prefixes; masked LM predicts selected hidden tokens using bidirectional visible context. Teacher forcing uses true previous tokens in training; inference conditions on previously generated tokens. Training positions can be evaluated in parallel under a causal mask.
7. RMSNorm rescales without mean subtraction. SwiGLU gates one FFN branch with another. GQA shares K/V among query groups, with MQA using one KV head. Sparse MoE routes tokens to subsets of experts, but shared components remain active and all experts still need storage somewhere.
8. Greedy takes a local maximum and can miss better sequence paths. Temperature adjusts sharpness without fixing knowledge. Top-k fixes candidate count regardless of uncertainty. Top-p adapts candidate count but depends on probability calibration/settings. Finite beam search explores alternatives but can favor short/repetitive high-likelihood text and is not exhaustive.
9. Advertised context is an acceptance/configuration capacity; useful context is task performance on information within it; application memory is externally retained state that must be selected and supplied or otherwise used in computation. They are not interchangeable.
10. A broadly pretrained adaptable model is foundational in role. Scaling laws fit trends such as loss versus resources under assumptions. A training-optimal allocation need not minimize lifetime serving cost, especially at high request volume.

### Part B answers

11. IDs `[2,32]`; embeddings `[2,32,128]`; combined QKV `[2,32,384]`; each Q/K/V `[2,32,128]` then `[2,8,32,16]`; scores/weights `[2,8,32,32]`; head outputs `[2,8,32,16]`; merged/projected attention `[2,32,128]`; final logits `[2,32,256]`.
12. Inputs `[9,4,7,2]`, targets `[4,7,2,6]`. Logit index 3 predicts 6. Each position can see its own known input token while predicting the next token.
13. Mean loss is `ln(4) ≈ 1.386294` nats; perplexity is 4. Gradient is `p−one_hot(y) = [0.1,-0.3,0.2]`.
14. Keep the first three (cumulative mass 0.95), renormalized to approximately `[0.631579,0.210526,0.157895]`. The crossing token must remain.
15. `2×12×2×2048×4×64×2 = 50,331,648` bytes = 48 MiB. With 16 KV heads it is 192 MiB. This excludes weights/workspaces.
16. `16384−2048−512−800−3000−1024 = 9000` tokens, subject to checking actual serialized overhead.
17. Naive pairwise coefficients grow fourfold; KV entries double. Projection/FFN work, kernels, memory access, batching, and inference phase affect runtime.
18. Both are approximately `6×300×10^6×4×10^9 = 7.2×10^18` FLOPs. Equal estimated compute does not establish equal quality; data, capacity balance, optimizer behavior, and approximation omissions matter.

### Part C answers

19. Zero scores still have positive exponential weight and leak future information. Use negative infinity before softmax. Check upper-triangle weights and perturb only future input positions while requiring unchanged prefix logits.
20. Cross-entropy expects logits and already applies stable log-softmax. Unshifted targets with no internal shift train reconstruction/copying at the same position, not next-token prediction. Pass raw logits and shift exactly once.
21. Sum losses over valid targets and divide by their total count. Reject or skip a batch with no valid targets, and report if no targets remain overall. Do not assign a fictitious zero mean loss to an undefined evaluation.
22. Compare last-position logits from cached incremental decoding with an uncached forward pass on exactly the same prefix. Match positions, model state, mask, tokenization, precision/tolerances, and dropout-free evaluation mode; avoid comparing different cropping policies.
23. Cross length with evidence position while controlling task difficulty and distractors; include missing-evidence and multi-fact cases. Compare relevance filtering/retrieval and reordered evidence; optionally compare summarization while checking preserved details. Report per-slice quality and latency, not just an average.
24. Total experts, shared weights, cache, workspaces, and replication consume memory even when few experts activate per token. Measure resident bytes, cache at target concurrency/length, precision, dispatch, and device placement; active count alone is insufficient.
25. Encode text, sample a shifted window, embed IDs/positions, apply causal blocks, normalize/project, compute raw-logit loss, backpropagate, and update with AdamW. For generation, use final-position logits, sample, append, and repeat. Differences include tiny synthetic data, a byte tokenizer, no post-training, explicit inefficient attention, no KV cache, basic stopping, and no production observability or batching. Full credit requires tracing the learner's actual implementation and its limitations.

</details>
