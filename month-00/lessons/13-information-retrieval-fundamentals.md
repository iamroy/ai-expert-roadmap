# 0.13 Information Retrieval Fundamentals — REFRESH

Retrieval predates deep learning by decades, and the classical machinery did not go away when embeddings arrived. Production RAG systems in Month 4 are usually hybrid: BM25 and vector search combined, then reranked. This lesson is the classical half.

| | |
|---|---|
| **Mode** | REFRESH |
| **Time** | 60–75 minutes, including the exercises |
| **Assumes** | 0.10 Representation Learning, 0.12 NLP (TF-IDF) |
| **Used by** | Month 4 embeddings, retrieval, and vector search; Month 5 RAG systems; Month 6 evaluation |

[Month 0 roadmap](../README.md) · [Previous: Basic NLP Concepts](12-basic-nlp-concepts.md) · [Next: Basic Data Engineering](14-basic-data-engineering.md)

## Learning objectives

After this lesson you can:

- describe how an inverted index makes lexical search fast, and what it costs
- explain BM25 and what it adds over TF-IDF
- compute precision, recall, F1, MRR, and nDCG, and choose the right one
- explain why top-K retrieval plus reranking is the standard architecture
- choose a chunking strategy and say what the retrieval unit costs you
- state where lexical search beats semantic search and vice versa, and why hybrid wins

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Section [0.13.5](#0135--lexical-versus-semantic-matching) is the comparison the curriculum asks for.
3. Record gaps in [`progress.md`](../progress.md).

Examples assume:

```python
import math
from collections import Counter, defaultdict
```

## 0.13.1 — The retrieval problem

Given a query and a large collection, return the most relevant documents, fast. Two costs matter and they pull against each other: **quality** (are the right documents returned and ranked highly?) and **latency** (scanning ten million documents per query is not an option).

Everything below is a way of buying quality or latency.

## 0.13.2 — Inverted indexes

A **forward index** maps document to terms. That is the wrong direction: answering a query would mean scanning every document.

An **inverted index** maps term to the documents containing it:

```text
"neural"  → [1, 4, 7, 23]
"network" → [1, 7, 88]
"recipe"  → [3, 12]
```

A query for "neural network" intersects two short lists instead of scanning the corpus. This single data structure is why keyword search is fast, and it is what Lucene, Elasticsearch, OpenSearch, and every classical search engine are built on.

```python
documents = {
    1: "neural network training with gradient descent",
    2: "a recipe for sourdough bread",
    3: "gradient boosting for tabular data",
    4: "training neural networks requires gpus",
}

index = defaultdict(set)
for document_id, text in documents.items():
    for token in text.lower().split():
        index[token].add(document_id)

print(sorted(index["neural"]))                                  # [1, 4]
print(sorted(index["gradient"] & index["training"]))            # [1]: AND query
print(sorted(index["gradient"] | index["recipe"]))              # [1, 2, 3]: OR query
```

Real indexes store more per entry, called a posting list: term frequency, positions for phrase queries, and per-field data. They also apply normalization at index time (lowercasing, stemming, stop-word removal) and **must apply exactly the same normalization to queries**. A mismatch is a classic silent failure.

Two limitations follow directly. The index matches **surface forms**, so "car" does not match "automobile" unless you added synonyms by hand. And updates need care, since documents are typically indexed into immutable segments that are periodically merged.

## 0.13.3 — From TF-IDF to BM25

TF-IDF (0.12.4) has two weaknesses that BM25 fixes.

**Term frequency saturation.** Under TF-IDF, a document mentioning "python" 100 times scores 10 times higher than one mentioning it 10 times. That is wrong: after a handful of occurrences, more occurrences tell you little. BM25 saturates term frequency with a tunable `k₁`.

**Length normalization.** Long documents contain more terms and win by accident. BM25 normalizes by length relative to the corpus average, with `b` controlling the strength.

```text
BM25(q, d) = Σ_t IDF(t) · [ f(t,d) · (k₁ + 1) ] / [ f(t,d) + k₁ · (1 − b + b · |d|/avgdl) ]

IDF(t) = log( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

Typical defaults are `k₁ = 1.2` and `b = 0.75`.

```python
class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.corpus, self.k1, self.b = corpus, k1, b
        self.N = len(corpus)
        self.average_length = sum(len(document) for document in corpus) / self.N
        self.document_frequency = Counter(token for document in corpus for token in set(document))

    def idf(self, term: str) -> float:
        df = self.document_frequency.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: list[str], index: int) -> float:
        document = self.corpus[index]
        counts = Counter(document)
        length_norm = 1 - self.b + self.b * len(document) / self.average_length

        total = 0.0
        for term in query:
            frequency = counts.get(term, 0)
            if frequency:
                total += self.idf(term) * (frequency * (self.k1 + 1)) / (
                    frequency + self.k1 * length_norm
                )
        return total

    def top_k(self, query: list[str], k: int = 3) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(self.N)]
        return sorted(scored, key=lambda pair: -pair[1])[:k]


corpus = [
    "neural network training with gradient descent".split(),
    "a recipe for sourdough bread baking".split(),
    "gradient boosting for tabular data science".split(),
    "training deep neural networks on gpus".split(),
]
bm25 = BM25(corpus)
for index, score in bm25.top_k("neural training".split()):
    print(f"doc {index} score {score:.4f}: {' '.join(corpus[index])}")
```

BM25 is a strong baseline that dense retrievers do not automatically beat, particularly on exact terms, rare entities, and out-of-domain data. Treat any retrieval result that does not compare against BM25 with suspicion.

## 0.13.4 — Evaluation: precision, recall, ranking

For a set of results:

```text
precision = relevant returned / total returned        "how much of what I got is useful?"
recall    = relevant returned / total relevant        "how much of what exists did I get?"
F1        = 2 · precision · recall / (precision + recall)
```

They trade off: returning everything gives perfect recall and terrible precision. Because retrieval returns a ranked list, the metrics are usually computed at a cutoff, **precision@K** and **recall@K**.

Rank-aware metrics account for *where* the relevant results land:

- **MRR** (mean reciprocal rank): `1/rank` of the first relevant result, averaged over queries. Right for "one correct answer" tasks.
- **nDCG**: discounts gains logarithmically by position and supports graded relevance, then normalizes by the ideal ordering. The general-purpose ranking metric.
- **Recall@K** is what matters most for RAG, because a document the retriever never returns cannot be used by the generator, no matter how good it is.

```python
def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    top = retrieved[:k]
    return sum(1 for document in top if document in relevant) / k

def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    top = retrieved[:k]
    return sum(1 for document in top if document in relevant) / len(relevant)

def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    for position, document in enumerate(retrieved, start=1):
        if document in relevant:
            return 1 / position
    return 0.0

def ndcg_at_k(retrieved: list[int], gains: dict[int, float], k: int) -> float:
    def dcg(order: list[int]) -> float:
        return sum(gains.get(document, 0.0) / math.log2(position + 1)
                   for position, document in enumerate(order[:k], start=1))
    ideal = sorted(gains, key=lambda document: -gains[document])
    return dcg(retrieved) / dcg(ideal) if dcg(ideal) else 0.0


retrieved = [7, 2, 9, 1, 4]
relevant = {1, 4, 7}
gains = {7: 3.0, 1: 2.0, 4: 1.0}

print(f"P@3   {precision_at_k(retrieved, relevant, 3):.3f}")
print(f"R@3   {recall_at_k(retrieved, relevant, 3):.3f}")
print(f"P@5   {precision_at_k(retrieved, relevant, 5):.3f}")
print(f"MRR   {reciprocal_rank(retrieved, relevant):.3f}")
print(f"nDCG@5 {ndcg_at_k(retrieved, gains, 5):.3f}")
```

> **Pitfall:** Relevance judgments are expensive and usually incomplete. A document marked irrelevant may simply never have been judged, which penalizes a system that surfaces genuinely good results the annotators missed. Read benchmark numbers with that in mind, and prefer relative comparisons on the same judgments over absolute scores.

## 0.13.5 — Lexical versus semantic matching

The comparison the curriculum asks for.

**Lexical** matching (BM25) scores on term overlap. **Semantic** matching embeds query and documents into a shared space (0.10) and ranks by vector similarity.

| | Lexical (BM25) | Semantic (dense) |
|---|---|---|
| Matches | exact terms, with stemming | meaning, paraphrase, synonyms |
| Rare terms and IDs | excellent | often poor |
| Synonyms and paraphrase | fails without a synonym list | strong |
| Out-of-domain | robust | degrades; the encoder was trained somewhere |
| Index build | cheap, immediate | requires embedding every document |
| Query cost | very low | an encoder forward pass, plus ANN search |
| Explainability | shows matched terms | a similarity score with no explanation |
| Multilingual | needs per-language handling | free with a multilingual encoder |

Each fails where the other succeeds. Searching "ERR_CONN_REFUSED 2847" needs exact matching; a dense encoder may map that to a nearby-but-wrong error. Searching "how do I stop my code from crashing" needs meaning; BM25 finds documents containing "stop", "code", and "crashing" literally.

**Hybrid search** runs both and fuses the results, which is the production default. The simplest robust fusion is **reciprocal rank fusion**, which combines rankings without needing the scores to be comparable:

```python
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for position, document in enumerate(ranking, start=1):
            scores[document] += 1 / (k + position)
    return sorted(scores, key=lambda document: -scores[document])

lexical = [7, 2, 9, 1]
semantic = [1, 7, 5, 3]
print(reciprocal_rank_fusion([lexical, semantic]))   # [7, 1, 2, 9, 5, 3]
```

Documents ranked well by both systems rise to the top, and `k` damps the influence of any single top-1 result.

## 0.13.6 — Chunking and the retrieval unit

Before anything can be retrieved, you must decide *what a document is*. This is the most consequential and most neglected decision in a retrieval system.

The unit you index is the unit you return. Index whole 80-page manuals and a match tells the user the answer is somewhere in 80 pages. Index single sentences and you retrieve fragments with no surrounding context, and the generator cannot tell what they refer to.

| Strategy | Unit | Trade-off |
|---|---|---|
| Whole document | a file | precise topic match, useless granularity |
| Fixed-size chunks | N tokens with overlap | simple and predictable; cuts mid-sentence and mid-table |
| Structural chunks | section, paragraph, slide, function | respects author intent; uneven sizes |
| Semantic chunks | split at topic shifts | best coherence; costs a model call per document |
| Sentence windows | retrieve a sentence, return its neighbors | precise matching with restored context |

Practical guidance that generalizes:

- **Chunk on structure first, size second.** Headings, paragraphs, and list boundaries carry information the author put there deliberately. Fall back to fixed size only within an oversized section.
- **Overlap costs little and prevents a specific failure**: an answer split across a boundary appearing in neither chunk. Ten to twenty percent is typical.
- **Keep chunk metadata** (0.14.5): source, title, section heading, page, and position. Prepending the document title and section heading to a chunk's embedded text measurably improves retrieval, because an isolated paragraph often never names its own subject.
- **Match chunk size to the embedding model.** A model trained on sentence-length input degrades on 2,000-token chunks, and anything past the encoder's limit is silently truncated — a quiet failure that looks like poor relevance.
- **The retrieved unit and the generated-context unit need not be identical.** Retrieve small for precision, then expand to the parent section before passing text to the model. This is the sentence-window and parent-document pattern, and it usually beats picking one size for both jobs.

Chunking is also the first thing to revisit when a RAG system underperforms. It is cheaper to re-chunk than to fine-tune, and a bad chunking strategy caps quality no matter how good the retriever and generator are.

## 0.13.7 — Top-K retrieval and reranking

The standard architecture is two stages, because quality and latency have different shapes:

```text
query → retrieve top 100 (cheap, high recall) → rerank to top 5 (expensive, high precision) → use
```

The **retriever** must be fast enough to touch the whole corpus, so it uses an inverted index or approximate nearest-neighbor search over embeddings. It optimizes recall: get the right documents into the candidate set.

The **reranker** sees only 100 candidates, so it can afford a **cross-encoder**, which runs query and document through a transformer *together*. That lets every query token attend to every document token, which is far more accurate than comparing two independently computed vectors. It cannot be used for retrieval, because scoring ten million documents per query would be impossibly slow.

That distinction is worth holding onto:

- **Bi-encoder**: encode documents once, offline; compare vectors at query time. Fast, scalable, less accurate.
- **Cross-encoder**: encode query and document jointly at query time. Accurate, and far too slow for full-corpus search.

`K` is a real tuning decision. Too small and the reranker never sees the right document; recall@K at the retrieval stage is the ceiling for the whole pipeline. Too large and latency and cost climb.

## Exercises

### Exercise 1 — Build an inverted index

Index ten short documents, support AND, OR, and NOT queries, and measure how many documents are touched versus a linear scan. Then show that a query normalized differently from the index (uppercase, or unstemmed) returns nothing, and fix it.

### Exercise 2 — TF-IDF versus BM25 on saturation and length

Construct two documents: one short with a term appearing twice, one long with the same term 50 times, plus a few unrelated documents so IDF is meaningful. Score both with TF-IDF and BM25, then vary `k₁` and `b` and explain each effect.

### Exercise 3 — Metrics on the same ranking

For a fixed ranked list and relevance set, compute P@1, P@5, R@5, MRR, and nDCG@5. Then swap the relevant document at position 1 with the irrelevant one at position 3 and recompute, noting which metrics change and which do not.

### Exercise 4 — Where lexical fails and where semantic fails

Write four queries: two where BM25 should win (an exact error code, a rare product name) and two where a dense retriever should win (a paraphrase, a synonym). Predict the outcome for each and explain what a hybrid system does.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
corpus = {
    1: "Neural Network Training with Gradient Descent",
    2: "A recipe for sourdough bread",
    3: "Gradient boosting for tabular data",
    4: "Training neural networks requires GPUs",
    5: "Sourdough starter maintenance guide",
}

def normalize(text: str) -> list[str]:
    return [token.lower().rstrip("s") for token in text.split()]      # crude stemming

index = defaultdict(set)
for document_id, text in corpus.items():
    for token in normalize(text):
        index[token].add(document_id)

all_ids = set(corpus)
print("AND:", sorted(index["neural"] & index["training"]))
print("OR: ", sorted(index["recipe"] | index["sourdough"]))
print("NOT:", sorted(index["gradient"] - index["neural"]))

touched = len(index["neural"]) + len(index["training"])
print(f"posting entries touched {touched} vs linear scan {len(corpus)}")

print("unnormalized query 'Networks':", sorted(index.get("Networks", set())))   # empty
print("normalized query:            ", sorted(index[normalize("Networks")[0]]))
```

The uppercase, unstemmed query returns nothing at all, because the index stores normalized terms. Applying the identical normalization to queries and documents is mandatory, and a mismatch fails silently rather than raising an error.

### Exercise 2

```python
short = "python tutorial python".split()                       # 3 terms, 2 occurrences
long_document = ("python " * 50 + "filler " * 450).split()     # 500 terms, 50 occurrences
corpus = [
    short,
    long_document,
    "java programming guide".split(),                          # unrelated, so IDF is meaningful
    "rust memory safety".split(),
    "go concurrency patterns".split(),
]

N = len(corpus)
document_frequency = Counter(token for document in corpus for token in set(document))

def tfidf(document: list[str], term: str) -> float:
    return (Counter(document)[term] / len(document)) * math.log(N / document_frequency[term])

for k1 in [0.5, 1.2, 3.0]:
    scorer = BM25(corpus, k1=k1, b=0.75)
    print(f"k1={k1:<4} BM25 short {scorer.score(['python'], 0):.4f}  "
          f"long {scorer.score(['python'], 1):.4f}")

for b in [0.0, 0.75, 1.0]:
    scorer = BM25(corpus, k1=1.2, b=b)
    print(f"b={b:<5} BM25 short {scorer.score(['python'], 0):.4f}  "
          f"long {scorer.score(['python'], 1):.4f}")

print(f"TF-IDF short {tfidf(short, 'python'):.4f}  long {tfidf(long_document, 'python'):.4f}")
```

```text
k1=0.5  BM25 short 1.2296  long 1.2638
k1=1.2  BM25 short 1.6558  long 1.7607
k1=3.0  BM25 short 2.4872  long 2.8362
b=0.0   BM25 short 1.2038  long 1.8809
b=0.75  BM25 short 1.6558  long 1.7607
b=1.0   BM25 short 1.8928  long 1.7240
TF-IDF short 0.6109  long 0.0916
```

Raising `k₁` weakens saturation, so repeated occurrences keep earning score; at `k₁ = 0.5` the fiftieth occurrence adds almost nothing over the second. Raising `b` penalizes length more: at `b = 0` normalization is off and the 50-occurrence document wins comfortably, while at `b = 1` the short document overtakes it.

The instructive part is the contrast. BM25 keeps the two documents within a factor of about 1.1 of each other at default settings, treating them as comparably relevant, which is the sensible judgment. TF-IDF, which divides by length but never saturates term frequency, rates the short document nearly 7 times higher. Neither tool ranks the keyword-stuffed document first here, but only BM25 lets you tune the two behaviors independently, which is why it became the standard.

### Exercise 3

```python
retrieved = [7, 2, 9, 1, 4]
relevant = {1, 4, 7}
gains = {7: 3.0, 1: 2.0, 4: 1.0}

def report(order: list[int], label: str) -> None:
    print(f"{label:10s} P@1 {precision_at_k(order, relevant, 1):.2f}  "
          f"P@5 {precision_at_k(order, relevant, 5):.2f}  "
          f"R@5 {recall_at_k(order, relevant, 5):.2f}  "
          f"MRR {reciprocal_rank(order, relevant):.3f}  "
          f"nDCG@5 {ndcg_at_k(order, gains, 5):.3f}")

report(retrieved, "original")
swapped = retrieved.copy()
swapped[0], swapped[2] = swapped[2], swapped[0]      # best result drops from rank 1 to rank 3
report(swapped, "swapped")
```

```text
original   P@1 1.00  P@5 0.60  R@5 1.00  MRR 1.000  nDCG@5 0.892
swapped    P@1 0.00  P@5 0.60  R@5 1.00  MRR 0.333  nDCG@5 0.577
```

P@5 and R@5 do not move at all: the same five documents are returned, and set metrics ignore order entirely. P@1, MRR, and nDCG@5 all drop sharply, because the best document fell from rank 1 to rank 3. A system that reports only recall@K can therefore degrade badly in user-visible ranking while its headline metric stays flat, which is why you always pair a set metric with a rank-aware one.

### Exercise 4

| Query | Expected winner | Why |
|---|---|---|
| `ERR_CONN_REFUSED 2847` | lexical | an exact token; embeddings blur it toward similar-looking error strings |
| `Thinkpad X1 Carbon Gen 11 BIOS` | lexical | rare product and model tokens, likely absent from encoder training data |
| `how do I stop my app from crashing on startup` | semantic | the target document probably says "resolve boot-time failures"; almost no term overlap |
| `car insurance` matching documents about `automobile coverage` | semantic | pure synonymy, which BM25 cannot see without a manual synonym list |

Hybrid search runs both and fuses, so each query is answered by whichever system handles it well; reciprocal rank fusion needs no score calibration. A reranking cross-encoder then fixes ordering within the merged candidate set, which is why the three-stage pattern (lexical + dense → fuse → rerank) is the production default.

</details>

## Exit test

1. What is an inverted index, and why is it fast?
2. Why must query normalization match index normalization?
3. What two problems does BM25 fix in TF-IDF?
4. What do `k₁` and `b` control?
5. Define precision and recall, and give a case where each alone is misleading.
6. When is MRR the right metric, and when is nDCG?
7. Why is recall@K the metric that matters most for RAG retrieval?
8. Why can a cross-encoder rerank but not retrieve?
9. What is the difference between a bi-encoder and a cross-encoder?
10. Give two queries where lexical search beats semantic, and two where the reverse holds.
11. What is reciprocal rank fusion, and why not just add the scores?
12. What does the choice of `K` in top-K retrieval trade off?
13. Why is chunk size a quality ceiling, and what does prepending a section heading fix?

<details>
<summary>Show answers</summary>

1. A mapping from term to the list of documents containing it. A query intersects a few short posting lists instead of scanning the entire corpus.
2. The index stores normalized terms, so a query normalized differently will not match anything. The failure is silent: you get zero results rather than an error.
3. Unbounded term-frequency growth, which BM25 saturates, and a bias toward long documents, which BM25 corrects with length normalization.
4. `k₁` controls how quickly term frequency saturates; `b` controls how strongly document length is penalized, with `b = 0` disabling it.
5. Precision is the fraction of returned results that are relevant; recall is the fraction of relevant results that were returned. Returning everything gives perfect recall with terrible precision; returning one certain result gives high precision with terrible recall.
6. MRR when there is essentially one right answer and you care where it lands. nDCG when relevance is graded and the whole ordering matters.
7. Because the generator can only use what the retriever returned. A document missing from the candidate set is unrecoverable no matter how good the model is.
8. It scores a query and document jointly in one forward pass, so scoring a whole corpus would require one pass per document per query. That is fine for 100 candidates and impossible for ten million.
9. A bi-encoder embeds documents independently and offline, then compares vectors, which is fast and scalable. A cross-encoder processes query and document together at query time, which is more accurate and far slower.
10. Lexical wins on exact error codes and rare product or model names. Semantic wins on paraphrased questions and synonym matches such as "car" versus "automobile".
11. A fusion method that scores each document by summing `1/(k + rank)` across rankings. It uses ranks rather than raw scores, so BM25 scores and cosine similarities, which are on incomparable scales, can be combined without calibration.
12. Retrieval recall against latency and cost. A `K` that is too small caps the pipeline's ceiling, since the reranker can never recover a document the retriever missed.
13. The indexed unit is the unit returned, so chunks that are too large give useless granularity and chunks that are too small lose the context needed to interpret them; neither is fixable downstream. Prepending the document title and section heading gives an isolated paragraph the subject it never names itself, which is often the difference between matching and not.

</details>

## Completion criteria

You are done when:

- you can explain an inverted index and the normalization requirement
- you can write the BM25 intuition and say what `k₁` and `b` do
- you can pick the right metric for a given retrieval task and compute it
- you can explain the retrieve-then-rerank architecture and why each stage uses a different model
- you can argue for hybrid search with concrete examples in both directions

## References

**Classical IR**
- [Introduction to Information Retrieval, Manning, Raghavan, Schütze](https://nlp.stanford.edu/IR-book/) — free online; chapters 1, 2, 6, 8
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [Elasticsearch: practical BM25](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-variables)

**Evaluation**
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html), chapter 8
- [BEIR: a heterogeneous benchmark for zero-shot IR](https://arxiv.org/abs/2104.08663) — where dense retrievers do and do not beat BM25

**Neural and hybrid retrieval**
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [ColBERT](https://arxiv.org/abs/2004.12832) — late interaction, between bi- and cross-encoders
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Pretrained Transformers for Text Ranking: BERT and Beyond](https://arxiv.org/abs/2010.06467)

## About this lesson

Written to cover section 0.13 of the [Month 0 curriculum](../README.md). Code examples were checked with Python 3.12.
