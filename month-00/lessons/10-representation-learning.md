# 0.10 Representation Learning — REFRESH

The curriculum names representation learning as one of the three areas to validate most carefully, because it is the concept that connects everything you already know about CNN features to embeddings, attention, and retrieval. If you come from computer vision, you have been doing representation learning for years under a different name.

| | |
|---|---|
| **Mode** | REFRESH |
| **Time** | 60–90 minutes, including the exercises |
| **Assumes** | 0.3 Linear Algebra (cosine similarity), 0.6 Neural Networks, 0.9 Architecture Concepts |
| **Used by** | 0.11 Self-Supervised Learning, 0.12 NLP, 0.13 Information Retrieval, Month 4 embeddings and vector search |

[Month 0 roadmap](../README.md) · [Previous: Architecture Concepts](09-deep-learning-architecture-concepts.md) · [Next: Self-Supervised Learning](11-self-supervised-learning.md)

## Learning objectives

After this lesson you can:

- define a representation, a latent space, and an embedding, and say how they relate
- explain what a pretrained representation buys you and when it stops working
- explain what makes an embedding space useful and how geometry encodes meaning
- describe the trade-offs of embedding dimensionality
- distinguish static embeddings from contextual ones and explain why it matters
- trace the through-line from CNN features to token embeddings to multimodal embeddings
- choose a pooling strategy and explain when to normalize
- evaluate whether a representation is any good, and recognize collapse

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Section [0.10.8](#0108--the-through-line-cnn-features--token-embeddings--multimodal) is what the Month 0 exit criteria ask you to explain.
3. Record gaps in [`progress.md`](../progress.md).

Examples assume:

```python
import torch
import torch.nn.functional as F
from torch import nn
```

## 0.10.1 — Representations, latent spaces, and embeddings

A **representation** is any encoding of an input that a model works with. Raw pixels are a representation; so is the output of a CNN's penultimate layer.

A **latent space** is the vector space where learned representations live. "Latent" means the dimensions are not directly observed; they emerged from training. A **latent vector**, **feature vector**, and **embedding** are the same object viewed from different angles: a fixed-length vector of floats that stands in for a complex input.

In practice, "embedding" usually implies one more property: **the geometry is meaningful**. Distances and directions in the space correspond to relationships between the things represented.

```text
"A photograph of a golden retriever"  →  [0.21, −1.04, 0.88, …]   768 floats
"A photo of a labrador"               →  [0.19, −0.97, 0.91, …]   nearby
"Quarterly revenue projections"       →  [1.52,  0.33, −2.10, …]  far away
```

The whole point of a good representation is that hard problems become easy in the new space. Similarity becomes a dot product. Classification becomes a linear boundary. Retrieval becomes nearest-neighbor search.

## 0.10.2 — What makes an embedding space useful

Three properties, roughly in order of importance:

**1. Semantic similarity maps to geometric proximity.** Related inputs land near each other, which is what makes cosine similarity (0.3.4) meaningful. Note that "similar" is defined by the training objective, not by intuition: a model trained on paraphrase pairs and one trained on topic labels will disagree about what belongs together.

**2. Linear structure.** Meaningful relationships often correspond to consistent directions. The classic Word2Vec example is `king − man + woman ≈ queen`. The effect is real but frequently overstated; it holds for some relations, chiefly frequent syntactic and geographic ones, and fails for many others, and the original evaluations excluded the query words from the answer set, which flattered the results.

**3. Smoothness.** Small input changes produce small representation changes, which is what lets a downstream classifier generalize instead of memorizing.

```python
torch.manual_seed(0)
vocab = ["dog", "puppy", "cat", "invoice", "receipt"]
embeddings = F.normalize(torch.randn(5, 64), dim=-1)

# force the geometry a well-trained model would learn
embeddings[1] = F.normalize(embeddings[0] + 0.15 * torch.randn(64), dim=-1)   # puppy near dog
embeddings[4] = F.normalize(embeddings[3] + 0.15 * torch.randn(64), dim=-1)   # receipt near invoice

similarity = embeddings @ embeddings.T
for i, word in enumerate(vocab):
    ranked = similarity[i].argsort(descending=True)[1:3]
    print(f"{word:8s} nearest: {[vocab[j] for j in ranked]}")
```

## 0.10.3 — Dimensionality

Embedding dimension `D` is a capacity and cost decision.

| Too small | Too large |
|---|---|
| distinct concepts collide | more memory, slower search |
| an information bottleneck | risk of overfitting on small data |
| poor downstream accuracy | diminishing returns in quality |

Typical values: 128 to 384 for compact retrieval embeddings, 768 for BERT-base and many sentence encoders, 1,024 to 4,096 for larger models, and `D` equal to the model width for LLM token embeddings.

Two facts make high dimensions work better than intuition suggests:

- Random vectors in high dimensions are nearly orthogonal (0.3.9), so a 768-dimensional space can hold an enormous number of nearly independent directions.
- **Superposition**: models appear to represent many more features than they have dimensions, by accepting slight interference between rarely co-occurring features. This is a major theme in interpretability research.

The cost is concrete. One million documents at 768 dimensions in `float32` is about 3 GB before any index overhead, which is why Month 4 spends time on quantization and approximate nearest-neighbor search.

```python
for dim in [128, 384, 768, 1536]:
    gb = 1_000_000 * dim * 4 / 1e9
    print(f"D={dim:5d}: {gb:5.2f} GB per million vectors in float32")
```

## 0.10.4 — Static versus contextual embeddings

This distinction is the single most important idea in this lesson.

A **static** embedding assigns one fixed vector per word. Word2Vec and GloVe (0.12) work this way. The word "bank" gets one vector that blends the riverbank and the financial sense, and no context can change it.

A **contextual** embedding is computed by a model that has seen the surrounding text, so the same word gets different vectors in different sentences:

```text
"I sat on the river bank"          → bank ≈ [geography-ish vector]
"I deposited it at the bank"       → bank ≈ [finance-ish vector]
```

This is what attention buys. Each layer lets a token's representation absorb information from the rest of the sequence, so representations become progressively more context-dependent with depth. Resolving polysemy, pronouns, and long-range dependencies all depend on it.

Note the layered structure inside a transformer: the *input* embedding table is static, one vector per token ID, and everything after the first attention layer is contextual. Both are called embeddings, which causes constant confusion.

## 0.10.5 — Pretrained representations

Everything above assumed a useful embedding space already exists. Pretraining is where it comes from, and it is why representation learning matters practically rather than only conceptually.

The economics are the point. Learning a good space requires enormous data, and almost nobody has labeled data at that scale. Self-supervised pretraining (0.11) sidesteps the requirement by deriving its targets from the data itself, so a model can learn from essentially unlimited unlabeled text or images. What you download is the result of that compute: a space where the geometry is already meaningful.

Three ways to use one, matching the adaptation strategies in 0.9.4:

| Use | What you do | When |
|---|---|---|
| **Frozen features** | embed inputs, train a small model on the vectors | little labeled data; the domain resembles pretraining |
| **Fine-tuned** | continue training the encoder on your task | enough data, and a domain the pretrained model handles poorly |
| **Off the shelf** | embed and compare directly, no training at all | retrieval and clustering, which need no labels |

The third row is worth dwelling on, because it is what makes semantic search possible. You are not training anything. You embed documents once, embed a query, and rank by cosine similarity (0.3.4). All of the intelligence was paid for during pretraining, by someone else.

Two constraints govern whether a pretrained space actually works for you:

- **The training objective defines similarity** (0.10.2). A model pretrained on paraphrase pairs groups differently from one pretrained on citations. Read what an embedding model was trained to do before assuming it matches your notion of "related".
- **Domain distance degrades quality.** An encoder trained on web text handles railway maintenance logs or clinical notes less well, and the failure is quiet: you get plausible rankings that are subtly wrong. This is the distribution-mismatch case from 0.10.7, and it is the usual reason to fine-tune an embedding model rather than take one off the shelf.

The through-line in the next section is really a statement about pretraining: the same recipe — a self-supervised objective on a large unlabeled corpus — produces CNN features, token embeddings, and CLIP's shared space alike.

## 0.10.6 — From token vectors to one vector for a document

A transformer gives you `[B, N, D]`: one vector per token. Retrieval and classification usually need `[B, D]`: one vector per input. The reduction is **pooling**.

| Strategy | How | Notes |
|---|---|---|
| Mean pooling | average over real tokens | strong, simple default; must be padding-aware (0.2.13) |
| CLS pooling | take the special first token's vector | only meaningful if the model was trained that way |
| Max pooling | element-wise maximum | occasionally useful; loses magnitude information |
| Last token | take the final position | the convention for decoder-only embedding models |
| Attention pooling | learned weighted average | more capacity, needs training |

```python
torch.manual_seed(0)
hidden = torch.randn(2, 6, 8)                       # [B, N, D]
mask = torch.tensor([[1, 1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 1, 1]])

weights = mask.unsqueeze(-1).float()
pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)

corrupted = hidden.clone()
corrupted[mask == 0] = 99.0                         # garbage at padded positions
assert torch.allclose(pooled, (corrupted * weights).sum(1) / weights.sum(1).clamp(min=1.0))
print(pooled.shape)                                 # [2, 8]
```

> **Pitfall:** Two mistakes account for most bad retrieval results. First, pooling over padding, which drags every short document's vector toward the padding vector. Second, using a similarity the model was not trained with. If a model was trained with cosine similarity, normalize before comparing (0.3.4).

## 0.10.7 — Evaluating representation quality

"Good embeddings" is not a property you can read off a loss curve. The standard evaluations:

| Method | What it measures | Notes |
|---|---|---|
| **Linear probe** | train only a linear classifier on frozen features | the standard test: if a linear boundary separates classes, the representation did the work |
| **k-NN classification** | label by nearest neighbors, no training at all | tests local geometry directly; DINO's strong k-NN result is why it was notable (0.11) |
| **Retrieval metrics** | recall@K, nDCG on a labeled query set | the right measure when retrieval is the actual task (0.13.4) |
| **Clustering agreement** | do clusters align with known labels | unsupervised, coarse |
| **Alignment and uniformity** | are positives close, and is the space well spread | diagnoses collapse directly (0.11.4) |

Two failure modes are worth naming because both produce a good-looking loss:

- **Collapse**, where the spread across the batch approaches zero and everything is similar to everything (0.11, Exercise 3).
- **Distribution mismatch**, where an encoder evaluated on data unlike its training set degrades sharply. This is why benchmarks like MTEB report many task types rather than one number, and why a model at the top of a leaderboard can still be wrong for your corpus.

The practical rule: **evaluate on your task and your data.** A linear probe on your own labels, or recall@K on fifty hand-written queries against your own documents, tells you more than any leaderboard.

## 0.10.8 — The through-line: CNN features → token embeddings → multimodal

The curriculum asks you to connect these three explicitly. They are the same idea applied to different inputs.

**CNN features.** Take a ResNet trained on ImageNet, remove the classification head, and the penultimate layer gives a 2,048-dimensional vector per image. Similar images land near each other. You can train a linear classifier on top of frozen features for a new task with very little data. Self-supervised vision models such as DINO learn these without labels (0.11), and produce features good enough that a nearest-neighbor lookup nearly matches a trained classifier.

**LLM token embeddings.** A token ID indexes an embedding table `[V, D]` to give a static vector, transformer blocks make it contextual, and the final hidden state is compared against output vectors to produce logits (0.3). Same object as the CNN feature: a learned vector where geometry carries meaning.

**Multimodal embeddings.** CLIP trains an image encoder and a text encoder jointly so that matching image-text pairs land near each other in *one shared space*. An image of a dog and the caption "a photo of a dog" become nearby vectors. That shared space is what enables zero-shot classification (compare an image against embedded label descriptions), cross-modal retrieval (search images with text), and much of what modern vision-language models are built on.

The one-sentence version: **representation learning is learning a space where the geometry does the work.** CNN features, token embeddings, sentence embeddings, and CLIP vectors are that idea applied to pixels, tokens, documents, and pairs of modalities.

## Exercises

### Exercise 1 — Geometry of a learned space

Build embeddings for a few concept clusters, compute the full cosine-similarity matrix, and verify that within-cluster similarity exceeds cross-cluster similarity. Then compare the ranking you get from unnormalized dot product against cosine similarity, and explain when they differ.

### Exercise 2 — Static embeddings cannot disambiguate

Simulate a static embedding table and a context-dependent encoder. Show that the static vector for an ambiguous word is identical in both sentences, while the contextual one differs, and relate this to what attention does.

### Exercise 3 — Pooling strategies compared

For a batch with padding, compute mean pooling done correctly, mean pooling done naively over padding, and last-token pooling. Measure how far the naive version drifts as padding grows.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
torch.manual_seed(0)
concepts = {"animals": ["dog", "cat", "horse"], "finance": ["invoice", "revenue", "ledger"]}

centers = {name: F.normalize(torch.randn(64), dim=-1) for name in concepts}
words, vectors, groups = [], [], []
for name, members in concepts.items():
    for word in members:
        words.append(word)
        groups.append(name)
        vectors.append(F.normalize(4.0 * centers[name] + torch.randn(64), dim=-1))

E = torch.stack(vectors)
similarity = E @ E.T

same = [similarity[i, j].item() for i in range(len(words)) for j in range(len(words))
        if i != j and groups[i] == groups[j]]
different = [similarity[i, j].item() for i in range(len(words)) for j in range(len(words))
             if groups[i] != groups[j]]

print(f"within-cluster mean  {sum(same) / len(same):.3f}")
print(f"cross-cluster mean   {sum(different) / len(different):.3f}")
assert sum(same) / len(same) > sum(different) / len(different)

# unnormalized vectors let magnitude override direction
scaled = E.clone()
scaled[3] *= 5.0
print("cosine ranking for 'dog':", [words[j] for j in (E @ E[0]).argsort(descending=True)[:3]])
print("dot ranking for 'dog':   ", [words[j] for j in (scaled @ scaled[0]).argsort(descending=True)[:3]])
```

Cosine ignores magnitude and ranks by direction alone. Raw dot product multiplies by length, so a single long vector can dominate every ranking regardless of its meaning. Normalize when the model was trained with cosine similarity.

### Exercise 2

```python
torch.manual_seed(0)
vocab = {"river": 0, "bank": 1, "money": 2, "the": 3}
static_table = nn.Embedding(len(vocab), 16)

sentence_a = torch.tensor([vocab["the"], vocab["river"], vocab["bank"]])
sentence_b = torch.tensor([vocab["the"], vocab["money"], vocab["bank"]])

static_a = static_table(sentence_a)[2]
static_b = static_table(sentence_b)[2]
assert torch.equal(static_a, static_b)
print("static: identical vector for 'bank' in both sentences")

encoder = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, batch_first=True)
with torch.inference_mode():
    contextual_a = encoder(static_table(sentence_a)[None])[0, 2]
    contextual_b = encoder(static_table(sentence_b)[None])[0, 2]

similarity = F.cosine_similarity(contextual_a, contextual_b, dim=0).item()
print(f"contextual: cosine similarity between the two 'bank' vectors = {similarity:.3f}")
assert not torch.allclose(contextual_a, contextual_b)
```

The encoder is untrained here, so the vectors differ only because attention mixed in different neighbors. That is the mechanism; training is what makes the difference *meaningful*, separating the financial sense from the geographic one.

### Exercise 3

```python
torch.manual_seed(0)
D, N = 8, 12
hidden = torch.randn(1, N, D)

print(f"{'real tokens':>12} {'drift of naive mean':>22}")
for real in [12, 8, 4, 2]:
    mask = torch.zeros(1, N)
    mask[0, :real] = 1
    padded = hidden.clone()
    padded[0, real:] = 0.0                                   # padding embeds to zero

    weights = mask.unsqueeze(-1)
    correct = (padded * weights).sum(1) / weights.sum(1).clamp(min=1.0)
    naive = padded.mean(1)                                   # averages the zeros too
    last = padded[:, real - 1]

    drift = (correct - naive).norm().item()
    print(f"{real:>12} {drift:>22.4f}")
```

The naive average divides by the full padded length, so it shrinks toward zero as padding grows: at 2 real tokens out of 12 it has scaled the true embedding by 1/6. Short documents are affected most, which in a retrieval system shows up as short passages systematically ranking oddly. Last-token pooling is unaffected by padding when the mask is respected, but it discards everything before the final position unless the model was trained to summarize there.

</details>

## Exit test

1. Define representation, latent space, and embedding.
2. What makes an embedding space useful?
3. What defines "similar" in a given embedding space?
4. Why is `king − man + woman ≈ queen` a weaker result than it is often presented as?
5. What goes wrong if the embedding dimension is too small? Too large?
6. Why can a 768-dimensional space hold so many distinct concepts?
7. What is the difference between static and contextual embeddings?
8. Which part of a transformer produces each kind?
9. Name four pooling strategies and when each is appropriate.
10. What are the two most common causes of bad retrieval results from pooling?
11. How do CNN features, token embeddings, and CLIP embeddings relate?
12. What does a shared image-text space enable?
13. What is a linear probe, and why is a strong k-NN result on frozen features notable?

<details>
<summary>Show answers</summary>

1. A representation is any encoding of an input a model operates on. A latent space is the learned vector space those encodings live in. An embedding is a vector in that space, usually implying its geometry is meaningful.
2. Semantic similarity corresponds to geometric proximity, meaningful relationships correspond to consistent directions, and small input changes produce small vector changes.
3. The training objective. A model trained on paraphrases, one trained on topics, and one trained on citations will each group different things together.
4. The effect holds for some relation types, mainly frequent syntactic and geographic ones, and fails for many others. The original evaluations also excluded the query words from candidate answers, which improved the apparent results.
5. Too small causes concepts to collide and creates an information bottleneck. Too large costs memory and search time, can overfit on small datasets, and gives diminishing quality returns.
6. Random high-dimensional vectors are nearly orthogonal, so the space supports a vast number of nearly independent directions. Superposition lets models pack in still more features by tolerating slight interference.
7. A static embedding gives one fixed vector per word regardless of context. A contextual embedding is computed from the surrounding text, so the same word gets different vectors in different sentences.
8. The input embedding table is static; every representation after the first attention layer is contextual.
9. Mean pooling (a strong default, must be padding-aware), CLS pooling (only if trained that way), max pooling (occasionally useful, loses magnitude), and last-token pooling (the decoder-only convention). Attention pooling is a learned fifth option.
10. Averaging over padding tokens, and comparing with a similarity measure the model was not trained with, such as raw dot product on vectors trained for cosine.
11. All three are learned vectors where geometry encodes meaning: over pixels, over tokens, and over images and text placed in one shared space.
12. Zero-shot classification by comparing an image with embedded label descriptions, cross-modal retrieval such as searching images with text, and the vision-language grounding modern multimodal models build on.
13. A linear probe trains only a linear classifier on frozen features, so whatever accuracy it reaches is attributable to the representation rather than to the classifier. A strong k-NN result is a stronger claim still, because it involves no training at all: the raw geometry already groups same-class items together.

</details>

## Completion criteria

You are done when:

- you can explain the static-versus-contextual distinction and point to where each lives in a transformer
- you can state what defines similarity in an embedding space
- you can choose and implement a padding-aware pooling strategy
- you can trace CNN features → token embeddings → multimodal embeddings aloud
- your three exercises run with passing assertions

## References

**Foundations**
- [Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538)
- [Deep Learning Book](https://www.deeplearningbook.org/contents/representation.html), chapter 15

**Word and sentence embeddings**
- [Efficient Estimation of Word Representations (Word2Vec)](https://arxiv.org/abs/1301.3781)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — pooling strategies for sentence embeddings
- [Sentence Transformers documentation](https://sbert.net/)
- [The Illustrated Word2vec, Jay Alammar](https://jalammar.github.io/illustrated-word2vec/)

**Contextual and multimodal**
- [BERT](https://arxiv.org/abs/1810.04805) — contextual representations
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Emerging Properties in Self-Supervised Vision Transformers (DINO)](https://arxiv.org/abs/2104.14294)

**Evaluating representations**
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)
- [Understanding Contrastive Representation Learning through Alignment and Uniformity](https://arxiv.org/abs/2005.10242)
- [What Should Not Be Contrastive in Contrastive Learning](https://arxiv.org/abs/2008.05659)
- [Whitening Sentence Representations](https://arxiv.org/abs/2006.12013)

**Geometry and interpretability**
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Linguistic Regularities in Continuous Space Word Representations](https://aclanthology.org/N13-1090/) — the analogy result, with its caveats

## Videos and code to read

- [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers) — the reference implementation for pooling, normalization, and training embedding models; read the pooling module against 0.10.6
- [embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb) — how embedding quality is actually measured across tasks, which is 0.10.7 in practice
- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — the `makemore` MLP lecture visualizes a learned embedding space directly

## About this lesson

Written to cover section 0.10 of the [Month 0 curriculum](../README.md). Code examples were checked with PyTorch 2.14 on CPU.
