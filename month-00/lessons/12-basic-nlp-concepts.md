# 0.12 — Basic NLP Concepts

**Depth: LEARN**

**Goal:** understand the vocabulary of text processing and the architectural progression from sparse counts to contextual transformers.

[Month 0 roadmap](../README.md) · [Previous: Self-Supervised Learning](11-self-supervised-learning.md) · [Next: Information Retrieval](13-information-retrieval-fundamentals.md)

## Corpus, document, sentence, token, vocabulary

A corpus is a collection of text examples. A document is one logical unit inside it; its boundary might be an article, message, code file, or conversation. Sentence segmentation divides text into sentences under language-specific rules. Tokenization converts text into units consumed by a model. The vocabulary maps known units to IDs.

These boundaries are data-model decisions. Splitting train and validation after chunking one document leaks neighboring text. Sentence rules designed for English can fail on abbreviations, code, or languages without the same punctuation. Preserve raw source IDs and offsets so preprocessing remains auditable.

## N-grams and bag of words

An n-gram is a contiguous sequence of `n` units. Word bigrams preserve short local order; character n-grams handle morphology and spelling variation. A bag-of-words vector records counts while discarding global order.

For vocabulary `[cat, dog, bites]`, both “cat bites dog” and “dog bites cat” map to `[1,1,1]`. Adding bigrams distinguishes them at the cost of a much larger sparse feature space.

Count features remain useful: they are fast, interpretable, and strong on many classification/search tasks. They also struggle with synonymy, long-range structure, and context-dependent meaning.

## TF-IDF

Term frequency measures a term within a document. Inverse document frequency downweights corpus-wide common terms. One common variant is:

```text
tfidf(t,d) = tf(t,d) × log(N / df(t))
```

Libraries use smoothing, sublinear term frequency, and normalization variants. Record the exact formula. TF-IDF is a weighting scheme, not a semantic embedding: two synonyms with no shared terms remain orthogonal.

## Word2Vec and GloVe

Word2Vec learns dense word vectors from local context. Skip-gram predicts context words from a center word; continuous bag of words predicts a center from surrounding words. Negative sampling makes training efficient by distinguishing observed pairs from sampled alternatives.

GloVe factorizes information derived from global word co-occurrence statistics. Both can encode useful semantic and syntactic relations, but each word type receives one base vector. “Bank” cannot directly have separate river and finance vectors without context or multiple-sense machinery.

Out-of-vocabulary words and rare terms are additional limitations. Subword models reduce this problem by composing pieces. Static embeddings also inherit corpus stereotypes and frequency artifacts.

## Stemming, lemmatization, and stop words

Stemming uses rules to strip affixes and may produce nonwords. Lemmatization maps inflected forms to dictionary lemmas using linguistic analysis. Stop-word removal drops frequent terms such as articles.

These are historical and still useful tools, but apply them by task. Removing “not” can invert sentiment; stemming can damage names or code; modern subword language models generally expect the tokenizer's trained preprocessing rather than an external stemming pipeline.

## From sequences to context

### RNNs

A recurrent neural network updates a hidden state one token at a time:

```text
h_t = f(x_t, h_(t−1))
```

Parameters share across positions, but computation is sequential and long gradient paths make distant dependencies difficult.

### LSTMs

Long short-term memory networks add gates controlling what to write, retain, and expose from a cell state. They improve gradient flow and long-term memory but remain sequential and compress the prefix into fixed-width state.

### Sequence-to-sequence and attention

Early encoder-decoder systems compressed a source sequence into a final state. Attention lets each decoder step form a weighted combination of encoder states, providing direct access to different source positions. This removed one bottleneck but commonly kept recurrent encoders/decoders.

### Transformers

Transformers replace recurrence with attention and position-wise transformations. Training can process known positions in parallel; self-attention gives direct connections among allowed tokens. Decoder-only models use causal visibility, encoder-only models can use bidirectional context, and encoder-decoder models add cross-attention.

```text
sparse counts → static word vectors → recurrent contextual state
              → attention over recurrent states → transformer contextual vectors
```

This is an evolution of tradeoffs, not a rule that every older method is obsolete. TF-IDF can outperform a large model when exact terminology, cost, or explainability dominates.

## Text evaluation pitfalls

Exact match is appropriate for some structured answers but penalizes valid paraphrases. BLEU and ROUGE measure reference overlap and have task-specific limits. Perplexity depends on tokenization and corpus. Human evaluation needs rubrics and agreement checks. Always connect the metric to the product decision.

## Checkpoint

1. What information does bag of words discard, and what do n-grams recover?
2. Why is TF-IDF not semantic similarity?
3. How do Word2Vec and contextual embeddings differ for a polysemous word?
4. What bottleneck did attention address in recurrent sequence-to-sequence models?

<details>
<summary>Show answers</summary>

1. Bag of words discards order. N-grams retain local contiguous order, not arbitrary long-range structure.
2. It weights lexical overlap using corpus frequency. Different surface forms remain unrelated unless features explicitly connect them.
3. Word2Vec normally assigns one vector per word type; a contextual model updates the token representation using its surrounding tokens.
4. A fixed final encoder state had to summarize the entire source. Attention lets each decoder step access a weighted mixture of source states.

</details>

## Exercise — Build a text baseline ladder

Design an experiment comparing unigram TF-IDF, word bigrams, static embeddings with mean pooling, and a pretrained contextual encoder for support-ticket classification.

<details>
<summary>Show exercise solution</summary>

Split by customer or time to prevent template leakage. Keep labels and evaluation examples fixed. Tune each method only on development data. Record macro-F1 for class imbalance, per-class recall, inference latency, model/index memory, and errors on rare terminology and paraphrases.

The ladder tests increasing representational complexity. Bigram gains suggest local phrases matter; static-embedding gains suggest semantic similarity beyond exact words; contextual gains suggest order or context-dependent meaning. Verify these hypotheses with error slices rather than inferring them from one aggregate score.

</details>

## Completion criteria

Define the core units, explain sparse and static-vector methods, describe preprocessing tradeoffs, and narrate the progression from RNNs through attention to transformers.

## Primary references

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
