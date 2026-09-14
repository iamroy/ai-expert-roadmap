# Month 00 — AI Engineering Prerequisite Review

**Purpose:** close only the prerequisite gaps that would slow down the main roadmap.

**Target duration:** 1–2 weeks

**Target effort:** 8–10 hours total

Most topics are fast reviews. The three areas to validate most carefully are tensor-level PyTorch fluency, representation and self-supervised learning, and NLP foundations.

All 18 sections now have complete teaching material, runnable examples, learner work cells, exercises, collapsed solutions, completion criteria, and primary references. The linked Jupyter notebooks are the interactive lessons; each also places a selected reading or video beside the concept it supports. Use the section depth to control time: attempt the validation first for SKIM topics, read weak areas, and spend more time on LEARN topics.

## Curriculum

### 0.1 Python for Modern AI — SKIM

Review classes, inheritance, dataclasses, type hints, function arguments, decorators, generators, context managers, comprehensions, exceptions, modules, environments, packaging, async basics, concurrency concepts, serialization, logging, configuration, and environment variables.

- [Lesson: Python for Modern AI](lessons/01-python-for-modern-ai.ipynb)
- Validation: pass all three exit tests and explain every required pattern without looking up syntax.

### 0.2 NumPy and Tensor Manipulation — SKIM

- [Lesson: NumPy and Tensor Manipulation](lessons/02-numpy-and-tensor-manipulation.ipynb)
- Validation: complete all five exercises and answer the original 12-question exit test.

- dimensions, indexing, slicing, and boolean masks
- broadcasting and vectorization
- reshape/view, transpose/permutation
- concatenation and stacking
- matrix multiplication, dot products, and element-wise operations
- reductions and numerical stability
- shape transition: `[batch, sequence, embedding]` → `[batch, heads, sequence, head_dim]`

### 0.3 Linear Algebra for Deep Learning — SKIM / REFRESH

- [Lesson: Linear Algebra for Deep Learning](lessons/03-linear-algebra-for-deep-learning.ipynb)
- Validation: answer the original 12-question exit test and explain the transformer connection.

- vectors, matrices, tensors, transpose, and matrix multiplication
- norms, cosine similarity, and projections
- basis and dimensionality
- eigenvalues/eigenvectors, SVD, and low-rank approximation at a conceptual level
- key connections: dot product → similarity → attention; matrix multiplication → learned linear transformations

### 0.4 Probability and Statistics — SKIM

- [Lesson: Probability and Statistics](lessons/04-probability-and-statistics.ipynb)
- Validation: calculate a Bayesian base-rate example and connect one-hot cross-entropy to negative log probability.

- distributions, conditional probability, and Bayes' theorem
- expectation, variance, standard deviation, and covariance
- entropy, cross-entropy, KL divergence, likelihood, and maximum likelihood
- sampling, softmax probabilities, and confidence versus probability
- explain why cross-entropy for the correct token is `-log P(correct token)`

### 0.5 Calculus for Neural Networks — SKIM

- [Lesson: Calculus for Neural Networks](lessons/05-calculus-for-neural-networks.ipynb)
- Validation: derive and verify gradients through a small computational graph.

- derivatives, partial derivatives, gradients, and the chain rule
- computational graphs and gradient descent
- trace: loss → gradients → parameters → optimizer update

### 0.6 Neural-Network Fundamentals — SKIM

- [Lesson: Neural-Network Fundamentals](lessons/06-neural-network-fundamentals.ipynb)
- Validation: implement and diagnose a residual MLP with normalization, GELU, and dropout.

- linear layers, activations, forward pass, losses, and backpropagation
- initialization, normalization, residual connections, and dropout
- focus on GELU, LayerNorm, and residual connections

### 0.7 Training Fundamentals — SKIM

- [Lesson: Training Fundamentals](lessons/07-training-fundamentals.ipynb)
- Validation: trace one optimizer update and define a resumable, reproducible checkpoint.

- SGD, momentum, Adam, and AdamW
- learning-rate schedules, warmup, weight decay, and gradient clipping
- batches, epochs, steps, validation, overfitting, and early stopping
- checkpoints, reproducibility, random seeds, and gradient accumulation

### 0.8 PyTorch Fundamentals — SKIM / REFRESH

- [Lesson: PyTorch Fundamentals](lessons/08-pytorch-fundamentals.ipynb)
- Validation: implement a tiny MLP loop, evaluate it correctly, and verify a checkpoint reload.

- tensors, devices, shapes, and autograd
- `nn.Module`, `forward()`, linear layers, activations, and losses
- optimizers, `Dataset`, `DataLoader`, training and validation loops
- `model.train()`, `model.eval()`, and `torch.no_grad()`
- checkpoint save/load, mixed precision, `torch.compile`, and GPU memory basics
- Exercise: implement a tiny MLP training loop without a high-level trainer.

### 0.9 Deep-Learning Architecture Concepts — SKIM

- [Lesson: Deep-Learning Architecture Concepts](lessons/09-deep-learning-architecture-concepts.ipynb)
- Validation: compare convolution and attention, then design a controlled transfer-learning experiment.

- feature extraction, hierarchical representations, and receptive fields
- parameter sharing, transfer learning, pretrained backbones, and fine-tuning
- frozen versus trainable layers
- compare convolution's local receptive field with attention's contextual interaction

### 0.10 Representation Learning — REFRESH

- [Lesson: Representation Learning](lessons/10-representation-learning.ipynb)
- Validation: audit an embedding pipeline's objective, metric, evaluation, and version contract.

- latent representations, feature vectors, embeddings, and embedding spaces
- semantic similarity and representation dimensionality
- pretrained representations and self-supervised learning
- connect CNN/DINO features → LLM token embeddings → multimodal embeddings

### 0.11 Self-Supervised Learning — LEARN / REFRESH

- [Lesson: Self-Supervised Learning](lessons/11-self-supervised-learning.ipynb)
- Validation: compare autoregressive, masked, contrastive, and teacher-student objectives and their failure modes.

- distinguish supervised, unsupervised, and self-supervised learning
- contrastive learning, masked prediction, and autoregressive prediction
- compare BERT masked-token prediction, GPT next-token prediction, and DINO teacher/student learning

### 0.12 Basic NLP Concepts — LEARN

- [Lesson: Basic NLP Concepts](lessons/12-basic-nlp-concepts.ipynb)
- Validation: build a baseline ladder and explain the progression from sparse counts to contextual transformers.

- corpus, document, sentence, token, and vocabulary
- n-grams, bag-of-words, TF-IDF, Word2Vec, and GloVe
- stemming, lemmatization, and stop words as historical context
- progression: bag-of-words → Word2Vec → RNN → LSTM → seq2seq → attention → transformer

### 0.13 Information Retrieval Fundamentals — REFRESH

- [Lesson: Information Retrieval Fundamentals](lessons/13-information-retrieval-fundamentals.ipynb)
- Validation: calculate ranking metrics and design lexical, dense, hybrid, and reranked comparisons.

- document retrieval, inverted indexes, lexical search, TF-IDF, and BM25
- precision, recall, ranking, top-K retrieval, and reranking
- distinguish lexical matching from semantic matching

### 0.14 Basic Data Engineering — SKIM

- [Lesson: Basic Data Engineering](lessons/14-basic-data-engineering.ipynb)
- Validation: design a versioned, idempotent ingestion pipeline with lineage and deletion.

- structured and unstructured data
- JSON, JSONL, and Parquet
- batch, streaming, ETL/ELT, object storage, and caching
- metadata, dataset versioning, and data lineage

### 0.15 APIs and Web Fundamentals — SKIM

- [Lesson: APIs and Web Fundamentals](lessons/15-apis-and-web-fundamentals.ipynb)
- Validation: specify a reliable asynchronous inference API with safe retries and streaming decisions.

- HTTP, REST, methods, headers, status codes, and JSON payloads
- authentication, API keys, rate limits, retries, timeouts, and pagination
- streaming responses, Server-Sent Events, WebSockets, and token streaming

### 0.16 Linux, Containers, and Git — SKIM

- [Lesson: Linux, Containers, and Git](lessons/16-linux-containers-and-git.ipynb)
- Validation: explain a containerized service from process and port through Git commit and deployed image.

- shell, processes, environment variables, permissions, and networking basics
- container images, volumes, and ports
- branches, commits, merge/rebase, and `.gitignore`

### 0.17 Software Engineering for AI — SKIM

- [Lesson: Software Engineering for AI](lessons/17-software-engineering-for-ai.ipynb)
- Validation: design a testable AI component with explicit contracts, dependencies, and telemetry.

- modular code, interfaces, dependency injection, and configuration-driven systems
- unit tests, integration tests, and mocking
- observability, logging, metrics, and reproducibility

### 0.18 MLOps Fundamentals — SKIM

- [Lesson: MLOps Fundamentals](lessons/18-mlops-fundamentals.ipynb)
- Validation: define a traceable model release with gates, canary monitoring, and rollback.

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

<details>
<summary>Show answer guide</summary>

1. Matrix multiplication applies linear transformations and combines features by dot products across matching dimensions.
2. A dot product grows when vectors align, so learned query/key or embedding vectors can use it as a compatibility score. Magnitude also affects it.
3. Softmax converts a vector of logits into nonnegative values summing to one along a chosen axis.
4. Cross-entropy measures prediction cost under a target distribution. For a one-hot class/token, it is the negative log probability assigned to the correct outcome.
5. A gradient collects partial derivatives of a scalar loss with respect to parameters and describes local sensitivity.
6. Backpropagation applies the chain rule in reverse through a computational graph, accumulating gradients for parameters.
7. Plain SGD uses the current gradient; AdamW uses momentum and adaptive second-moment scaling plus decoupled weight decay.
8. An embedding is a learned fixed-width vector representing an item in a geometry shaped by a training objective.
9. Cosine similarity is the dot product divided by both vector norms, measuring directional alignment.
10. Normalization controls activation scale and can improve optimization stability, though its exact effect depends on the operation and placement.
11. A residual connection adds a sublayer's update to its input, preserving a direct representation and gradient path.
12. Representation learning trains features useful for comparison, prediction, or transfer rather than relying only on hand-designed features.
13. Self-supervised learning constructs targets or relationships from the data itself, such as hidden tokens or paired views.
14. Semantic search retrieves by learned meaning-oriented vector similarity rather than requiring exact lexical overlap.
15. Training uses targets and gradients to update parameters; inference holds parameters fixed and produces predictions, often autoregressively for language models.

</details>

## Hands-on validation project

[Build a tiny text classifier in PyTorch](project/tiny-text-classifier/README.md), initially without Hugging Face:

`raw text → tokenizer → vocabulary → token IDs → embedding → pooling → linear classifier → logits → cross-entropy → backpropagation → AdamW → validation → inference API`

## Exit criteria

Move to Month 1 after you can trace:

`text → tokens → token IDs → embeddings → tensor operations → neural network → logits → softmax → loss → backpropagation → optimizer update`

You should also be able to connect traditional ML/CV representation learning to embeddings, attention, and transformers.

[Back to the full roadmap](../README.md)
