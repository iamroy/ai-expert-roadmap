# 0.13 — Information Retrieval Fundamentals

**Depth: REFRESH**

**Goal:** explain indexing and ranking, measure retrieval correctly, and distinguish lexical, semantic, and reranked search.

[Month 0 roadmap](../README.md) · [Previous: NLP](12-basic-nlp-concepts.md) · [Next: Data Engineering](14-basic-data-engineering.md)

## Retrieval as candidate selection and ranking

Given a query, an information-retrieval system returns ranked items from a corpus. Production search commonly uses stages:

```text
query → fast candidate retrieval → filtering → reranking → top results
```

Candidate retrieval optimizes recall under latency constraints. A more expensive reranker spends computation on a smaller candidate set. If the first stage never retrieves a relevant document, later ranking cannot recover it.

## Inverted indexes

An inverted index maps each term to a postings list of documents containing it, often with term frequency and positions. Query execution accesses postings for query terms instead of scanning every document.

Tokenization, case folding, stemming, field boundaries, and stop-word policy define the index contract. Positions enable phrase search. Document updates require consistent postings and metadata; deletion must remove content from every derived index.

## TF-IDF and BM25

TF-IDF rewards terms frequent in a document but rare across the collection. Raw term frequency can overreward repeated terms, and long documents naturally contain more terms.

BM25 adds saturation and document-length normalization. A common term contribution is:

```text
IDF(t) × [tf(t,d)(k1+1)]
         / [tf(t,d) + k1(1 − b + b |d|/avgdl)]
```

`k1` controls term-frequency saturation; `b` controls length normalization. Implementations vary in IDF formula and scoring details. BM25 remains a strong baseline for exact names, identifiers, rare terms, and rapidly changing corpora.

## Precision, recall, and rank-aware metrics

For a retrieved set:

```text
precision = relevant retrieved / retrieved
recall    = relevant retrieved / all relevant
```

Precision@k asks how many of the first `k` are relevant. Recall@k asks how much of known relevant content appears there. Mean reciprocal rank emphasizes the first relevant result. NDCG supports graded relevance and discounts lower ranks.

Metrics require relevance judgments and a defined candidate universe. Judging only results returned by one system can make new systems appear worse because their novel documents are unjudged. Include no-answer queries and separate retrieval relevance from answer-generation correctness.

## Semantic retrieval

Dense retrieval encodes queries and documents into vectors and ranks with dot product, cosine, or another trained metric. It can match paraphrases without shared words but may miss exact rare strings or numerical constraints.

Approximate nearest-neighbor indexes trade some recall for speed and memory. Evaluate index recall separately from embedding-model retrieval quality: exact vector search can isolate whether misses come from the ANN algorithm or representation.

Lexical and semantic search are complementary. Hybrid retrieval fuses their scores or ranks. Score scales are not automatically comparable; reciprocal-rank fusion combines positions without assuming calibrated scores.

## Reranking

A cross-encoder or LLM reranker reads query and candidate together, allowing richer interactions than independent embeddings. It improves precision at higher latency. Rerank enough candidates to preserve recall, batch where appropriate, and measure tail latency.

Reranking retrieved passages can also amplify bias toward plausible text. Keep source authorization, freshness, and trust metadata as explicit filters rather than expecting relevance scores to enforce policy.

## Chunking and retrieval units

Documents are often split into passages. Tiny chunks lose context; huge chunks dilute matching and consume downstream context windows. Preserve parent IDs, headings, offsets, timestamps, and access-control metadata. Consider retrieving passages and expanding to neighboring context after ranking.

Overlapping chunks from one document must stay in the same evaluation split. Otherwise a benchmark can reward memorizing near-duplicates.

## Checkpoint

1. Why can a reranker not fix low first-stage recall?
2. How does BM25 improve on raw term counts?
3. What different questions do precision@k and recall@k answer?
4. Why test exact and approximate vector search separately?

<details>
<summary>Show answers</summary>

1. It only scores supplied candidates; missing relevant items never reach it.
2. It saturates repeated term frequency, normalizes document length, and weights rare terms.
3. Precision@k measures result cleanliness; recall@k measures coverage of known relevant items.
4. Exact search measures representation/ranking quality without ANN approximation. Comparing it to ANN reveals index-induced losses.

</details>

## Exercise — Calculate and design retrieval metrics

A query has four known relevant documents. The top five results have relevance `[1,0,1,1,0]`. Calculate precision@5, recall@5, and reciprocal rank. Then name two reasons this single query is insufficient for system selection.

<details>
<summary>Show exercise solution</summary>

Three of five results are relevant, so precision@5 is `3/5 = 0.6`. Three of four known relevant documents were retrieved, so recall@5 is `3/4 = 0.75`. The first result is relevant, so reciprocal rank is `1/1 = 1`.

One query does not cover variation in query type, frequency, language, or no-answer behavior. Relevance judgments may be incomplete, and these metrics omit latency, freshness, authorization, and downstream answer quality.

</details>

## Completion criteria

Explain inverted indexes and BM25, compute core metrics, distinguish lexical/dense/hybrid retrieval, and design a candidate-plus-reranker evaluation.

## Primary references

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
