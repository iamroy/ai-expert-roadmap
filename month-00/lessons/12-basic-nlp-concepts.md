# 0.12 Basic NLP Concepts — LEARN

This is the third area the curriculum flags for careful validation, and the only one marked LEARN rather than SKIM or REFRESH. If your background is computer vision, this is the largest genuine gap: text has no natural fixed-size input, no pixel grid, and a discrete, open vocabulary. Everything below is how the field dealt with those facts.

| | |
|---|---|
| **Mode** | LEARN |
| **Time** | 90–120 minutes, including the exercises |
| **Assumes** | 0.10 Representation Learning, 0.11 Self-Supervised Learning |
| **Used by** | 0.13 Information Retrieval, the tiny text classifier project, Month 1 tokenization and transformers, Month 4 retrieval |

[Month 0 roadmap](../README.md) · [Previous: Self-Supervised Learning](11-self-supervised-learning.md) · [Next: Information Retrieval](13-information-retrieval-fundamentals.md)

## Learning objectives

After this lesson you can:

- use the vocabulary of NLP precisely: corpus, document, token, vocabulary, type
- build a vocabulary with `<pad>` and `<unk>`, and explain why both exist
- explain bag-of-words, n-grams, and TF-IDF, and compute TF-IDF by hand
- explain what Word2Vec added and what it could not do
- say why stemming, lemmatization, and stop-word removal faded, and where they survive
- trace the progression from bag-of-words to the transformer and say what each step fixed
- explain subword tokenization and why modern models use it

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Section [0.12.7](#0127--the-progression-bag-of-words--transformer) is a required Month 0 exit item.
3. Do [Exercise 1](#exercise-1--build-a-vocabulary-and-encode-text); it is the first component of the Month 0 project.
4. Record gaps in [`progress.md`](../progress.md).

Examples assume:

```python
import math
import re
from collections import Counter

import torch
```

## 0.12.1 — Core vocabulary

| Term | Meaning |
|---|---|
| **Corpus** | the whole collection of text |
| **Document** | one unit of text: an article, a review, a chunk |
| **Sentence** | one sentence; boundary detection is harder than it looks ("Dr. Smith paid $1.5M.") |
| **Token** | one unit fed to the model: a word, subword, or character |
| **Type** | one distinct token; "the cat sat on the mat" has 6 tokens and 5 types |
| **Vocabulary** | the set of types the model knows, mapped to integer IDs |

**Tokenization** is splitting text into tokens, and it is less obvious than it appears. Consider "don't", "New York", "state-of-the-art", "café", "🙂", URLs, and code. Languages such as Chinese and Japanese have no whitespace between words at all.

```python
text = "The cat sat on the mat. The cat didn't move!"

whitespace = text.split()
regex = re.findall(r"\b\w+\b", text.lower())

print(whitespace)
print(regex)
print(f"{len(regex)} tokens, {len(set(regex))} types")
```

## 0.12.2 — Vocabulary, `<unk>`, and `<pad>`

A model needs integers, so the vocabulary maps token to ID. Two special tokens are essential:

- **`<pad>`** fills short sequences so a batch forms a rectangular tensor. It must be masked out of pooling and attention (0.2.14), which is the single most common bug in this area.
- **`<unk>`** stands in for any token not in the vocabulary. Without it, a word-level model simply cannot encode unseen input.

```python
def build_vocabulary(documents: list[str], max_size: int = 10_000, min_count: int = 1):
    counts = Counter(token for document in documents for token in document.lower().split())
    kept = [token for token, count in counts.most_common(max_size - 2) if count >= min_count]
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token in kept:
        vocabulary[token] = len(vocabulary)
    return vocabulary


documents = ["the cat sat on the mat", "the dog sat on the log", "a bird flew away"]
vocabulary = build_vocabulary(documents)
print(vocabulary)


def encode(text: str, vocabulary: dict[str, int]) -> list[int]:
    return [vocabulary.get(token, vocabulary["<unk>"]) for token in text.lower().split()]


print(encode("the cat sat on the rug", vocabulary))   # 'rug' is unseen -> 1
```

`<pad>` is conventionally ID 0 so `padding_idx=0` in `nn.Embedding` works and mask construction is a simple `tokens != 0`.

**The out-of-vocabulary problem** is the fundamental weakness of word-level vocabularies. A 50,000-word vocabulary still misses names, typos, new terms, and most morphology: "running" and "runs" get separate entries while "runnning" gets `<unk>`. Subword tokenization (0.12.7) is the fix.

## 0.12.3 — Bag-of-words and n-grams

**Bag-of-words** represents a document as counts over the vocabulary, discarding all order:

```text
"the cat sat"  →  [0, 0, 1, 1, 1, 0, 0, …]
```

It is simple, interpretable, and sparse, and it remains a genuinely strong baseline for topical classification. But "the dog bit the man" and "the man bit the dog" get identical representations, and it knows nothing about synonyms: "excellent" and "superb" are as unrelated as "excellent" and "tractor".

**N-grams** recover a little local order by treating adjacent sequences as units:

```python
tokens = "the cat sat on the mat".split()
bigrams = list(zip(tokens, tokens[1:]))
print(bigrams[:3])   # [('the', 'cat'), ('cat', 'sat'), ('sat', 'on')]
```

Bigrams capture "not good" as a unit, which unigrams cannot. The cost is combinatorial: vocabulary size grows roughly as `V^n`, and most n-grams are rare, which is why n-gram language models needed heavy smoothing and still could not generalize beyond observed sequences.

## 0.12.4 — TF-IDF

Raw counts over-reward common words. TF-IDF scales each term by how informative it is across the corpus:

```text
TF(t, d)   = count of t in d (often normalized by document length)
IDF(t)     = log(N / df(t))        N = number of documents, df = documents containing t
TF-IDF     = TF × IDF
```

A term appearing in every document gets `IDF = log(1) = 0` and is ignored. A term appearing in few documents gets a high weight.

```python
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
]
tokenized = [document.split() for document in corpus]
N = len(corpus)

document_frequency = Counter(token for document in tokenized for token in set(document))

def tf_idf(document: list[str]) -> dict[str, float]:
    counts = Counter(document)
    return {
        token: (count / len(document)) * math.log(N / document_frequency[token])
        for token, count in counts.items()
    }

scores = tf_idf(tokenized[0])
for token, score in sorted(scores.items(), key=lambda item: -item[1]):
    print(f"{token:8s} {score:.4f}")
```

"mat" scores highest because it is unique to that document; "the" scores 0.0 because it appears everywhere. In practice, smoothed IDF variants avoid division by zero for unseen terms, and lengths are normalized.

TF-IDF is not history. It is the direct ancestor of BM25 (0.13), still the lexical half of most production hybrid-search systems, and still the right first thing to try on a small text-classification problem.

## 0.12.5 — Stemming, lemmatization, and stop words

These three appear in the curriculum as historical context, and knowing why they faded is more useful than knowing how to apply them.

**Stemming** chops affixes with rules, aiming to collapse "running", "runs", and "run" into one token. It is fast and routinely produces non-words. **Lemmatization** maps a word to its dictionary form using a vocabulary and part-of-speech information: "better" becomes "good", "was" becomes "be". It is slower and more accurate.

**Stop words** are very common words — "the", "is", "of" — removed on the theory that they carry little meaning.

```python
def crude_stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token

for token in ["running", "studies", "runs", "bus", "runner"]:
    print(f"{token:9s} -> {crude_stem(token)}")
```

```text
running   -> runn
studies   -> studi
runs      -> run
bus       -> bus
runner    -> runner
```

Note how badly even this behaves, and real stemmers are only somewhat better. "running" becomes "runn" rather than "run", so it never unifies with "runs". "studies" becomes the non-word "studi". "runner" is untouched. The length guard saves "bus" from becoming "bu", which is the kind of special case these rule sets accumulate.

All three exist for the same reason: **the vocabulary was the bottleneck.** When a word-level vocabulary must be small, collapsing "run/runs/running" into one entry and dropping high-frequency noise saves space and reduces sparsity. TF-IDF already handles stop words gracefully, since a term in every document gets `IDF = 0` (0.12.4), so explicit removal was always partly redundant.

Why they are largely obsolete for neural models:

- **Subword tokenization solves the morphology problem better.** "running" splits into pieces that share representation with "run" without a hand-written rule, and without destroying words the rules get wrong.
- **Stop words carry syntax.** "not", "no", and "but" are stop words in many standard lists, and removing them inverts meaning. Transformers use function words to resolve structure; discarding them removes real signal.
- **Stemming is lossy and irreversible.** "university" and "universe" can stem to the same token, silently merging unrelated concepts.

Where they still apply: lexical search indexes (0.13.2) often stem at index time so "running" matches "run", and stop-word lists still trim posting lists. That is the surviving use case, and even there it must be applied identically to queries and documents.

## 0.12.6 — Word2Vec, GloVe, and the limits of static vectors

Bag-of-words and TF-IDF treat words as atomic symbols with no relationships. **Word2Vec** (2013) learned dense vectors from context, on the distributional hypothesis: words appearing in similar contexts have similar meanings.

Two training setups:

- **Skip-gram** predicts context words from a center word. Better for rare words.
- **CBOW** predicts the center word from its context. Faster.

Note what this is: a self-supervised pretext task (0.11) predating the term. The targets come from the text itself, and the vectors, not the prediction, are the product.

**Negative sampling** made it tractable. Predicting over a 1-million-word vocabulary with a full softmax is expensive, so the model instead distinguishes the true context word from a handful of random ones. This is the direct ancestor of the contrastive losses in 0.11.

**GloVe** reached similar vectors differently, by factorizing a global word co-occurrence matrix rather than sliding a local window.

What these gained: dense vectors, meaningful similarity, and some linear analogical structure (0.10.2, with its caveats). What they could not do:

- **One vector per word, forever.** "bank" is a single blurred vector (0.10.4).
- **No composition.** Averaging word vectors for a sentence loses order and negation.
- **Still out-of-vocabulary bound**, though fastText's character n-grams helped.

Removing the first limitation is precisely what the rest of the progression is about.

## 0.12.7 — The progression: bag-of-words → transformer

The curriculum asks you to be able to trace this. Each step exists because the previous one failed at something specific.

| Stage | Idea | Fixed | Still broken |
|---|---|---|---|
| **Bag-of-words / TF-IDF** | count words, weight by informativeness | simple, interpretable, strong lexical baseline | no order, no synonymy, sparse |
| **Word2Vec / GloVe** | dense vectors from context | similarity, dimensionality | one vector per word; no composition |
| **RNN** | process tokens in sequence, carrying a hidden state | order and variable length | vanishing gradients over long spans; strictly sequential |
| **LSTM / GRU** | gates with an additive cell path | much longer dependencies | still sequential, so no parallel training; still a fixed-size state |
| **Seq2seq** | encoder RNN → single context vector → decoder RNN | variable-length input to variable-length output | one fixed vector is an information bottleneck |
| **Attention (2014)** | the decoder attends over *all* encoder states | the bottleneck; direct access to any input position | still built on sequential RNNs |
| **Transformer (2017)** | drop recurrence, keep only attention | full parallel training; constant path length between any two tokens | quadratic cost in sequence length; needs positional encoding |

Three threads run through the whole table.

**Path length.** How many steps must information traverse to connect two tokens? In an RNN, `O(N)`, with each step risking loss. In a transformer, `O(1)` for any pair, which is the deepest reason attention works (0.9.2).

**Parallelism.** An RNN's hidden state at step `t` requires step `t-1`, so training cannot parallelize across the sequence. A transformer processes all positions at once, which is what made training on internet-scale corpora feasible. The transformer's success is as much an engineering result as a modeling one.

**What is thrown away.** Each stage discards a prior. Bag-of-words discards order. RNNs assume recency matters most. The transformer assumes almost nothing, which is why it needs positional encoding, more data, and more compute, and why it scales further than anything before it.

Note also that the LSTM's additive cell path is the same trick as a residual connection (0.6.7): give the gradient a route that adds rather than multiplies.

## 0.12.8 — Subword tokenization

Modern models use neither words nor characters, but **subwords**. Words give a huge vocabulary and constant `<unk>`; characters give no `<unk>` but very long sequences and little meaning per token. Subwords sit in between: frequent words stay whole, rare words split into pieces.

```text
"tokenization"  →  ["token", "ization"]
"unhappiness"   →  ["un", "happiness"]
"Roy"           →  ["Roy"]
"Rrroy"         →  ["R", "rr", "oy"]           still encodable, no <unk>
```

**Byte-pair encoding (BPE)** starts from characters and repeatedly merges the most frequent adjacent pair, learning a merge list of the chosen vocabulary size. **WordPiece** (BERT) and **Unigram** (SentencePiece) differ in the merge criterion. **Byte-level BPE** (GPT-2 onward) operates on raw bytes, so *any* input is encodable and `<unk>` disappears entirely.

Consequences worth knowing now:

- Token counts, not word counts, drive context limits and API pricing. English averages roughly 1.3 tokens per word; code, other languages, and unusual formatting can be far worse.
- Tokenization explains a family of odd LLM behaviors, such as difficulty spelling words or counting letters, since the model never sees characters.
- Numbers tokenize inconsistently, which contributes to arithmetic errors.
- A tokenizer is fixed at pretraining time. You cannot change it during fine-tuning without retraining embeddings.

Month 1 covers BPE in depth, including implementing it.

## Exercises

### Exercise 1 — Build a vocabulary and encode text

This is the first component of the Month 0 project. Write a regex tokenizer, a vocabulary builder with `<pad>` and `<unk>` and a minimum count, and encode/decode functions. Then pad a batch of variable-length sequences into a `[B, N]` tensor with a matching mask, and verify that the mask sums equal the true lengths.

### Exercise 2 — TF-IDF by hand

On a corpus of five short documents, compute TF-IDF for every term. Show that a term appearing in all documents scores 0, and rank each document's most distinctive terms. Then use cosine similarity over the TF-IDF vectors to find the closest pair, and explain one case where it fails.

### Exercise 3 — Bag-of-words loses order

Build bag-of-words vectors for "the dog bit the man" and "the man bit the dog" and show they are identical. Then add bigrams and show they become distinguishable. Discuss the cost.

### Exercise 4 — Subword behavior

Without installing a tokenizer library, implement a tiny BPE trainer: start from characters, merge the most frequent adjacent pair a fixed number of times, and show how a rare word splits while a frequent one stays whole.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

```python
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def build_vocabulary(documents: list[str], min_count: int = 1) -> dict[str, int]:
    counts = Counter(token for document in documents for token in tokenize(document))
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token, count in counts.most_common():
        if count >= min_count:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def encode(text: str, vocabulary: dict[str, int]) -> list[int]:
    return [vocabulary.get(token, 1) for token in tokenize(text)]


def decode(ids: list[int], vocabulary: dict[str, int]) -> str:
    inverse = {index: token for token, index in vocabulary.items()}
    return " ".join(inverse[index] for index in ids)


def pad_batch(sequences: list[list[int]], pad_id: int = 0):
    longest = max(len(sequence) for sequence in sequences)
    tokens = torch.full((len(sequences), longest), pad_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tokens[row, :len(sequence)] = torch.tensor(sequence)
    return tokens, tokens != pad_id


corpus = ["The movie was great!", "I didn't like the plot.", "Great acting, dull story."]
vocabulary = build_vocabulary(corpus)
print(f"{len(vocabulary)} entries; pad={vocabulary['<pad>']}, unk={vocabulary['<unk>']}")

ids = [encode(document, vocabulary) for document in corpus]
tokens, mask = pad_batch(ids)

print(tokens)
assert mask.sum(dim=1).tolist() == [len(sequence) for sequence in ids]
print(decode(ids[0], vocabulary))
print(encode("the cinematography was superb", vocabulary))   # unseen words become 1
```

Note that `didn't` survives as one token thanks to the apostrophe rule, and that the mask sums exactly match the unpadded lengths, which is the invariant the project's pooling test depends on.

### Exercise 2

```python
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
    "a bird flew over the house",
    "the house was quiet",
]
tokenized = [document.split() for document in corpus]
N = len(corpus)
document_frequency = Counter(token for document in tokenized for token in set(document))

def tf_idf(document: list[str]) -> dict[str, float]:
    counts = Counter(document)
    return {token: (count / len(document)) * math.log(N / document_frequency[token])
            for token, count in counts.items()}

vectors = [tf_idf(document) for document in tokenized]
print(f"IDF('the') = {math.log(N / document_frequency['the']):.4f}")

for index, scores in enumerate(vectors):
    top = sorted(scores.items(), key=lambda item: -item[1])[:2]
    print(f"doc {index}: {[(token, round(score, 3)) for token, score in top]}")

vocabulary = sorted(document_frequency)
def densify(scores: dict[str, float]) -> torch.Tensor:
    return torch.tensor([scores.get(token, 0.0) for token in vocabulary])

matrix = torch.stack([densify(scores) for scores in vectors])
normalized = matrix / matrix.norm(dim=1, keepdim=True)
similarity = normalized @ normalized.T
similarity.fill_diagonal_(-1)

best = divmod(similarity.argmax().item(), N)
print(f"most similar pair: {best} at {similarity[best].item():.3f}")
```

"the" appears in all five documents, so its IDF is exactly 0 and it contributes nothing to any vector. Each document's top terms are the ones unique to it: "mat" and "log" distinguish the two otherwise near-identical sentences, and documents 0 and 1 come out as the closest pair at 0.329 because they share "sat" and "on".

The failure case is synonymy. Any two documents sharing no terms score 0 no matter what they mean: "The film was excellent" and "The movie was superb" overlap only on "was", so TF-IDF rates them as nearly unrelated. Lexical matching cannot see meaning, only surface forms, and that is exactly the gap dense embeddings fill (0.13).

### Exercise 3

```python
a, b = "the dog bit the man".split(), "the man bit the dog".split()
vocabulary = sorted(set(a) | set(b))

def bag(tokens: list[str], vocabulary: list[str]) -> list[int]:
    counts = Counter(tokens)
    return [counts.get(token, 0) for token in vocabulary]

print(bag(a, vocabulary), bag(b, vocabulary))
assert bag(a, vocabulary) == bag(b, vocabulary)          # identical: order is gone

def with_bigrams(tokens: list[str]) -> list[str]:
    return tokens + [f"{x}_{y}" for x, y in zip(tokens, tokens[1:])]

bigram_vocabulary = sorted(set(with_bigrams(a)) | set(with_bigrams(b)))
va, vb = bag(with_bigrams(a), bigram_vocabulary), bag(with_bigrams(b), bigram_vocabulary)
assert va != vb
print(f"unigram vocabulary {len(vocabulary)}, +bigram vocabulary {len(bigram_vocabulary)}")
```

Adding bigrams distinguishes the sentences but more than doubles the vocabulary on a five-word example. On a real corpus the bigram space is far larger and far sparser, and trigrams are worse again. That sparsity is what forced the field toward dense representations.

### Exercise 4

```python
def train_bpe(words: dict[str, int], merges: int) -> list[tuple[str, str]]:
    splits = {word: list(word) for word in words}
    learned = []
    for _ in range(merges):
        pairs = Counter()
        for word, frequency in words.items():
            symbols = splits[word]
            for left, right in zip(symbols, symbols[1:]):
                pairs[(left, right)] += frequency
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        learned.append(best)
        for word, symbols in splits.items():
            merged, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            splits[word] = merged
    return learned


def apply_bpe(word: str, learned: list[tuple[str, str]]) -> list[str]:
    symbols = list(word)
    for left, right in learned:
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == (left, right):
                merged.append(left + right)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


words = {"low": 50, "lower": 20, "lowest": 15, "newer": 30, "wider": 25, "new": 40}
learned = train_bpe(words, merges=12)
print("merges:", learned[:6])

for word in ["low", "lower", "lowest", "newest", "xylophone"]:
    print(f"{word:10s} -> {apply_bpe(word, learned)}")
```

Frequent whole words collapse into single tokens. Unseen words such as "newest" reuse learned pieces, and even "xylophone", which shares almost nothing with the training words, still splits into characters rather than becoming `<unk>`. That is the property real tokenizers are built for: complete coverage with a bounded vocabulary.

</details>

## Exit test

1. Define corpus, document, token, type, and vocabulary.
2. Why is tokenization harder than splitting on whitespace?
3. What are `<pad>` and `<unk>` for, and why is `<pad>` usually ID 0?
4. What is the out-of-vocabulary problem, and how do subwords solve it?
5. What does bag-of-words discard, and when is it still a good choice?
6. Write the TF-IDF formula and explain why a term in every document scores 0.
7. What did Word2Vec add over TF-IDF, and what were its two main limitations?
8. Why is stop-word removal risky for a neural model, and where is it still used?
9. What is the distributional hypothesis?
10. What problem did attention solve in seq2seq models?
11. What did the transformer remove, and what two things did that buy?
12. What is path length between two tokens in an RNN versus a transformer?
13. Give three practical consequences of subword tokenization.

<details>
<summary>Show answers</summary>

1. A corpus is the full text collection; a document is one unit of text; a token is one unit fed to the model; a type is a distinct token; the vocabulary is the set of types mapped to integer IDs.
2. Contractions, hyphenation, multi-word expressions, punctuation, accents, emoji, URLs, code, and languages such as Chinese that have no whitespace word boundaries.
3. `<pad>` fills short sequences so a batch is rectangular, and must be masked out of pooling and attention. `<unk>` represents tokens outside the vocabulary. `<pad>` is ID 0 so `padding_idx=0` works and masks are a simple `tokens != 0`.
4. A word-level vocabulary cannot encode unseen words, so names, typos, and new terms all collapse to `<unk>`. Subword tokenization decomposes any word into known pieces, and byte-level BPE covers any input at all.
5. All word order, and therefore negation and syntax. It remains a strong, cheap, interpretable baseline for topical classification with limited data.
6. `TF-IDF = TF(t,d) × log(N / df(t))`. If a term appears in every document, `N/df = 1` and `log 1 = 0`, so it carries no discriminative weight.
7. Dense vectors with meaningful similarity, learned from context rather than counts. Its limits were one fixed vector per word regardless of context, and no way to compose words into sentence meaning.
8. Many standard stop-word lists include "not", "no", and "but", so removal can invert meaning, and transformers rely on function words to resolve syntax. It survives in lexical search indexes, where it trims posting lists and must be applied identically to queries and documents.
9. That words occurring in similar contexts tend to have similar meanings, which is what lets context prediction produce useful vectors.
10. The fixed-size context vector bottleneck: the decoder could attend to all encoder states directly rather than relying on one summary vector.
11. Recurrence. That bought full parallel training across positions and a constant path length between any two tokens.
12. `O(N)` in an RNN, with information degrading at each step, versus `O(1)` in a transformer, where any token can attend directly to any other.
13. Token counts drive context limits and pricing; models cannot easily spell or count characters because they never see them; numbers tokenize inconsistently, hurting arithmetic; and the tokenizer is fixed at pretraining time.

</details>

## Completion criteria

You are done when:

- you can build a vocabulary with `<pad>` and `<unk>` and pad a batch with a correct mask, from scratch
- you can compute TF-IDF by hand and explain the IDF term
- you can trace bag-of-words → Word2Vec → RNN → LSTM → seq2seq → attention → transformer, saying what each step fixed
- you can explain path length and parallelism as the two reasons transformers won
- you can explain subword tokenization and three of its practical consequences
- your four exercises run, including the BPE trainer

You now have every component the [Month 0 project](../project/tiny-text-classifier/README.md) needs.

## References

**Foundations**
- [Speech and Language Processing, Jurafsky and Martin](https://web.stanford.edu/~jurafsky/slp3/) — chapters 2 (regex and tokenization), 4 (naive Bayes), 6 (vector semantics and TF-IDF). The standard reference, free online.
- [scikit-learn: text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

**Word embeddings**
- [Efficient Estimation of Word Representations (Word2Vec)](https://arxiv.org/abs/1301.3781)
- [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546) — negative sampling
- [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) and its [ACL page](https://aclanthology.org/D14-1162/)
- [The Illustrated Word2vec, Jay Alammar](https://jalammar.github.io/illustrated-word2vec/)

**The progression to transformers**
- [The Unreasonable Effectiveness of Recurrent Neural Networks, Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Understanding LSTM Networks, Chris Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention's first appearance
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer, Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

**Tokenization**
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE
- [SentencePiece](https://arxiv.org/abs/1808.06226)
- [Hugging Face: tokenizers summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Let's build the GPT Tokenizer, Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)

## Videos and code to read

- [karpathy/minbpe](https://github.com/karpathy/minbpe) — a minimal, readable BPE implementation; the natural next step after this lesson's Exercise 4, paired with [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the end of the progression table, built live
- [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers) — where the static-to-contextual jump lands in practice


## Mapped companion lessons

- [Text Processing](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/01-text-processing) and [Bag of Words and TF-IDF](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf) map to the preprocessing and sparse-feature baseline.
- [Word2Vec from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec) and [GloVe, FastText, and Subwords](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword) map to dense word representations.
- [CNNs and RNNs for Text](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text) and [Attention Mechanism](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/05-nlp-foundations-to-advanced/10-attention-mechanism) continue the architecture progression.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).

## About this lesson

Written to cover section 0.12 of the [Month 0 curriculum](../README.md). Code examples were checked with Python 3.12 and PyTorch 2.14.
