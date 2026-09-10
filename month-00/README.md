# Month 00 — AI Engineering Prerequisite Review

**Purpose:** close only the prerequisite gaps that would slow down the main roadmap.

**Target duration:** 1–2 weeks  
**Target effort:** 8–10 hours total

Most topics are fast reviews. The three areas to validate most carefully are tensor-level PyTorch fluency, representation and self-supervised learning, and NLP foundations.

## Curriculum

### 0.1 Python for Modern AI — SKIM

Review classes, inheritance, dataclasses, type hints, function arguments, decorators, generators, context managers, comprehensions, exceptions, modules, environments, packaging, async basics, concurrency concepts, serialization, logging, configuration, and environment variables.

- [Lesson: Python for Modern AI](lessons/01-python-for-modern-ai.md)
- Validation: pass both exit tests and explain every required pattern without looking up syntax.

### 0.2 NumPy and Tensor Manipulation — SKIM

- [Lesson: NumPy and Tensor Manipulation](lessons/02-numpy-and-tensor-manipulation.md)
- Validation: solve the shape, broadcasting, masking, and attention-layout exit test.

- dimensions, indexing, slicing, and boolean masks
- broadcasting and vectorization
- reshape/view, transpose/permutation
- concatenation and stacking
- matrix multiplication, dot products, and element-wise operations
- reductions and numerical stability
- shape transition: `[batch, sequence, embedding]` → `[batch, heads, sequence, head_dim]`

### 0.3 Linear Algebra for Deep Learning — SKIM / REFRESH

- [Lesson: Linear Algebra for Deep Learning](lessons/03-linear-algebra-for-deep-learning.md)
- Validation: explain the geometric and tensor-shape reasoning in the exit test.

- vectors, matrices, tensors, transpose, and matrix multiplication
- norms, cosine similarity, and projections
- basis and dimensionality
- eigenvalues/eigenvectors, SVD, and low-rank approximation at a conceptual level
- key connections: dot product → similarity → attention; matrix multiplication → learned linear transformations

### 0.4 Probability and Statistics — SKIM

- distributions, conditional probability, and Bayes' theorem
- expectation, variance, standard deviation, and covariance
- entropy, cross-entropy, KL divergence, likelihood, and maximum likelihood
- sampling, softmax probabilities, and confidence versus probability
- explain why cross-entropy for the correct token is `-log P(correct token)`

### 0.5 Calculus for Neural Networks — SKIM

- derivatives, partial derivatives, gradients, and the chain rule
- computational graphs and gradient descent
- trace: loss → gradients → parameters → optimizer update

### 0.6 Neural-Network Fundamentals — SKIM

- linear layers, activations, forward pass, losses, and backpropagation
- initialization, normalization, residual connections, and dropout
- focus on GELU, LayerNorm, and residual connections

### 0.7 Training Fundamentals — SKIM

- SGD, momentum, Adam, and AdamW
- learning-rate schedules, warmup, weight decay, and gradient clipping
- batches, epochs, steps, validation, overfitting, and early stopping
- checkpoints, reproducibility, random seeds, and gradient accumulation

### 0.8 PyTorch Fundamentals — SKIM / REFRESH

- tensors, devices, shapes, and autograd
- `nn.Module`, `forward()`, linear layers, activations, and losses
- optimizers, `Dataset`, `DataLoader`, training and validation loops
- `model.train()`, `model.eval()`, and `torch.no_grad()`
- checkpoint save/load, mixed precision, `torch.compile`, and GPU memory basics
- Exercise: implement a tiny MLP training loop without a high-level trainer.

### 0.9 Deep-Learning Architecture Concepts — SKIM

- feature extraction, hierarchical representations, and receptive fields
- parameter sharing, transfer learning, pretrained backbones, and fine-tuning
- frozen versus trainable layers
- compare convolution's local receptive field with attention's contextual interaction

### 0.10 Representation Learning — REFRESH

- latent representations, feature vectors, embeddings, and embedding spaces
- semantic similarity and representation dimensionality
- pretrained representations and self-supervised learning
- connect CNN/DINO features → LLM token embeddings → multimodal embeddings

### 0.11 Self-Supervised Learning — LEARN / REFRESH

- distinguish supervised, unsupervised, and self-supervised learning
- contrastive learning, masked prediction, and autoregressive prediction
- compare BERT masked-token prediction, GPT next-token prediction, and DINO teacher/student learning

### 0.12 Basic NLP Concepts — LEARN

- corpus, document, sentence, token, and vocabulary
- n-grams, bag-of-words, TF-IDF, Word2Vec, and GloVe
- stemming, lemmatization, and stop words as historical context
- progression: bag-of-words → Word2Vec → RNN → LSTM → seq2seq → attention → transformer

### 0.13 Information Retrieval Fundamentals — REFRESH

- document retrieval, inverted indexes, lexical search, TF-IDF, and BM25
- precision, recall, ranking, top-K retrieval, and reranking
- distinguish lexical matching from semantic matching

### 0.14 Basic Data Engineering — SKIM

- structured and unstructured data
- JSON, JSONL, and Parquet
- batch, streaming, ETL/ELT, object storage, and caching
- metadata, dataset versioning, and data lineage

### 0.15 APIs and Web Fundamentals — SKIM

- HTTP, REST, methods, headers, status codes, and JSON payloads
- authentication, API keys, rate limits, retries, timeouts, and pagination
- streaming responses, Server-Sent Events, WebSockets, and token streaming

### 0.16 Linux, Containers, and Git — SKIM

- shell, processes, environment variables, permissions, and networking basics
- container images, volumes, and ports
- branches, commits, merge/rebase, and `.gitignore`

### 0.17 Software Engineering for AI — SKIM

- modular code, interfaces, dependency injection, and configuration-driven systems
- unit tests, integration tests, and mocking
- observability, logging, metrics, and reproducibility

### 0.18 MLOps Fundamentals — SKIM

- experiment tracking, model registries, and artifact storage
- dataset, model, and environment versioning
- CI/CD, deployment, monitoring, drift, and rollback

## Core knowledge check

Explain each item without notes:

1. What does matrix multiplication do?
2. Why can a dot product represent similarity?
3. What does softmax do?
4. What is cross-entropy loss?
5. What is a gradient?
6. How does backpropagation work conceptually?
7. How does AdamW differ from SGD?
8. What is an embedding?
9. What is cosine similarity?
10. Why does normalization help training?
11. What is a residual connection?
12. What is representation learning?
13. What makes self-supervised learning different?
14. What is semantic search?
15. How do training and inference differ?

## Hands-on validation project

[Build a tiny text classifier in PyTorch](project/tiny-text-classifier/README.md), initially without Hugging Face:

`raw text → tokenizer → vocabulary → token IDs → embedding → pooling → linear classifier → logits → cross-entropy → backpropagation → AdamW → validation → inference API`

## Exit criteria

Move to Month 1 after you can trace:

`text → tokens → token IDs → embeddings → tensor operations → neural network → logits → softmax → loss → backpropagation → optimizer update`

You should also be able to connect traditional ML/CV representation learning to embeddings, attention, and transformers.
