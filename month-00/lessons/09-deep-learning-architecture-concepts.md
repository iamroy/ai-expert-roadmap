# 0.9 Deep-Learning Architecture Concepts — SKIM

This lesson is the bridge from classical deep learning to transformers. If you have a computer-vision background, most of it is already familiar; the goal is to make the vocabulary transferable, ending with the comparison the curriculum asks for between a convolution's local receptive field and attention's global one.

| | |
|---|---|
| **Mode** | SKIM |
| **Time** | 45–60 minutes |
| **Assumes** | 0.6 Neural-Network Fundamentals |
| **Used by** | 0.10 Representation Learning, 0.11 Self-Supervised Learning, Month 1 transformer internals, Month 2 fine-tuning |

[Month 0 roadmap](../README.md) · [Previous: PyTorch Fundamentals](08-pytorch-fundamentals.md) · [Next: Representation Learning](10-representation-learning.md)

## Learning objectives

After this lesson you can:

- explain feature extraction and why learned hierarchies replaced hand-engineered features
- define a receptive field and describe how it grows in a CNN versus a transformer
- explain parameter sharing and where it appears in convolution, RNNs, and transformers
- describe transfer learning, backbones, heads, and the freeze-versus-fine-tune decision
- state the practical trade-offs between convolution and attention

## How to use this lesson

1. Attempt the [exit test](#exit-test) first.
2. Section [0.9.5](#095--convolution-versus-attention) is the one the Month 0 exit criteria ask you to explain aloud.
3. Record gaps in [`progress.md`](../progress.md).

## 0.9.1 — Feature extraction and hierarchical representations

Before deep learning, vision pipelines used hand-designed feature extractors (SIFT, HOG, Haar) feeding a separate classifier. Deep learning replaced the design step with learning: the network discovers useful features because they reduce the loss.

What emerges in a trained CNN is a **hierarchy**:

| Depth | Typical features |
|---|---|
| early | edges, color transitions, simple gradients |
| middle | textures, corners, repeated motifs |
| late | object parts such as wheels, eyes, or letters |
| final | whole objects and class-discriminative structure |

Each level composes the previous one. Language models show an analogous progression: early layers track token identity and local syntax, middle layers carry syntactic and semantic relations, and later layers carry task- and context-specific abstractions. The analogy is genuine but loose; do not push it too far.

The practical consequence is **reuse**. Early features are generic enough to serve many tasks, which is the entire basis of transfer learning.

## 0.9.2 — Receptive fields

A unit's **receptive field** is the region of the input that can influence its output.

In a CNN it grows gradually. Stacking 3×3 convolutions adds 2 to the receptive field per layer, and stride or pooling multiplies it:

```text
layer 1: 3×3
layer 2: 5×5
layer 3: 7×7
…
```

So a CNN's early layers are structurally *local*. Relating two distant pixels requires enough depth, downsampling, or dilation for their receptive fields to overlap. This locality is a strong, useful prior for images, where nearby pixels really are related.

```python
def receptive_field(layers: int, kernel: int = 3, stride: int = 1) -> int:
    """Receptive field of a stack of identical convolution layers."""
    field, jump = 1, 1
    for _ in range(layers):
        field += (kernel - 1) * jump
        jump *= stride
    return field


for layers in [1, 3, 5, 10, 50]:
    print(f"{layers:2d} layers of 3x3 stride 1 -> receptive field {receptive_field(layers):3d}")
```

```text
 1 layers of 3x3 stride 1 -> receptive field   3
 3 layers of 3x3 stride 1 -> receptive field   7
 5 layers of 3x3 stride 1 -> receptive field  11
10 layers of 3x3 stride 1 -> receptive field  21
50 layers of 3x3 stride 1 -> receptive field 101
```

Fifty layers to see 101 pixels. That is the price of locality, and it is why real CNNs downsample: with stride, the field grows multiplicatively rather than additively.

In a transformer the situation is different: **every token can attend to every other token in the first layer**. The receptive field is the entire context window immediately. Nothing forces attention to be local, and nothing forces it to be global; it is learned.

The cost is quadratic. Attention computes an `N × N` score matrix (0.2), so doubling sequence length quadruples attention compute and memory, while a convolution stays linear in input size. That single trade-off drives a large amount of research: sliding-window and sparse attention, linear attention, FlashAttention's memory-efficient exact computation, and state-space models. Month 1 and Month 9 return to it.

## 0.9.3 — Parameter sharing

**Parameter sharing** reuses the same weights across positions, which cuts parameter count and builds in an assumption about structure.

| Architecture | What is shared | Prior it encodes |
|---|---|---|
| CNN | one kernel slides across all spatial positions | a useful feature is useful anywhere in the image (translation equivariance) |
| RNN | the same cell is applied at every timestep | the same transition rule applies at all times |
| Transformer | the same projection and feed-forward weights apply to every position | one transformation should work at any position, with order supplied separately by positional encoding |

Two consequences worth holding onto:

- A convolution over a 224×224 image uses one small kernel rather than a separate weight per pixel, which is why CNNs are so parameter-efficient.
- Because a transformer applies identical weights at every position and attention itself is permutation-equivariant, the architecture has no inherent notion of order. Positional information must be injected explicitly, and that is precisely why positional encodings exist (Month 1).

## 0.9.4 — Transfer learning, backbones, and fine-tuning

**Transfer learning** reuses representations learned on a large dataset for a new task with far less data.

The standard decomposition is a **backbone** (the pretrained feature extractor) plus a **head** (a small task-specific layer or two). The Month 0 project is this shape in miniature: an embedding-plus-pooling backbone with a linear classification head.

Adaptation strategies, from cheapest to most expensive:

| Strategy | What trains | Use when |
|---|---|---|
| Feature extraction (frozen backbone) | the head only | very little data; the domain resembles pretraining |
| Partial fine-tuning | the head plus the last N blocks | moderate data; a somewhat different domain |
| Full fine-tuning | everything | plenty of data and compute; a distant domain |
| Parameter-efficient (LoRA, adapters) | small injected matrices | large models where full fine-tuning is impractical |

Practical rules that hold across modalities:

- A fine-tuning learning rate is typically 10 to 100 times smaller than a pretraining rate. Large steps erase pretrained features, which is the origin of **catastrophic forgetting**.
- Freezing is not only about quality; a frozen backbone stores no gradients or optimizer state for those parameters, which is a large memory saving.
- Differential (layer-wise) learning rates are a good middle ground: smaller rates for early, generic layers and larger rates for late, task-specific ones.
- Frozen modules still need `model.eval()` semantics for dropout and normalization, and normalization layers with running statistics will keep updating them unless you explicitly stop that.

LoRA (0.3.12) belongs to the fourth row and is the reason low-rank factorization earned a place in Month 0. Month 2 covers it properly.

## 0.9.5 — Convolution versus attention

This is the comparison the curriculum asks you to be able to make.

| | Convolution | Attention |
|---|---|---|
| Interaction | local, fixed neighborhood | global, all positions |
| Weights | fixed kernel, input-independent | weights computed from the input itself |
| Receptive field | grows with depth | full context at layer 1 |
| Cost in input size `N` | linear | quadratic |
| Built-in prior | locality and translation equivariance | none; order supplied by positional encoding |
| Data efficiency | strong prior, works with less data | weak prior, needs more data or pretraining |

The deepest difference is that a convolution kernel is **static**: the same weights apply regardless of content. Attention weights are **dynamic**, computed per input from query-key similarity, so which positions interact depends on what the input actually contains. A pronoun can attend to its antecedent 200 tokens back without any architectural provision for that distance.

That flexibility is why Vision Transformers need far more data or stronger augmentation than CNNs at comparable scale: they must *learn* locality that a CNN is given for free. It is also why hybrids exist, and why convolution remains a strong choice when data is limited or inputs are large and locally structured.

## Exercises

### Exercise 1 — Compute a receptive field

For a stack of 3×3 convolutions with stride 1, compute the receptive field after 1, 3, 5, and 10 layers. Then recompute with a stride-2 layer inserted every second layer. How many stride-1 layers would it take to relate two pixels 100 apart, and what does that imply for a transformer processing a 2,000-token context?

### Exercise 2 — Parameter counts, convolution versus attention

For a 224×224×3 input, count the parameters in a 3×3 convolution with 64 output channels. Compare with a single attention projection over 196 patches of dimension 768. Explain where the difference comes from.

### Exercise 3 — Quadratic cost

Tabulate the number of entries in the attention score matrix for sequence lengths 512, 2,048, 8,192, and 131,072, and the memory each needs in `bfloat16` for 32 heads. Explain why long-context inference is dominated by this and by the KV cache.

<details>
<summary>Show exercise solutions</summary>

### Exercise 1

Each stride-1 3×3 layer adds 2: layer 1 sees 3, layer 3 sees 7, layer 5 sees 11, layer 10 sees 21. To span 100 pixels you would need about 50 layers. With a stride-2 layer every second layer, the receptive field grows multiplicatively instead and reaches 100 in roughly 8 to 10 layers, which is why real CNNs downsample.

A transformer needs **one** layer to relate positions 2,000 apart, because attention has no distance term at all. The price is the `N × N` score matrix rather than depth.

### Exercise 2

```python
conv = 3 * 3 * 3 * 64 + 64              # kernel h × w × in_channels × out_channels + bias
attention_projection = 768 * 768 + 768

print(f"3x3 conv, 3 -> 64 channels : {conv:,}")
print(f"one attention projection   : {attention_projection:,}")
print(f"ratio                      : {attention_projection / conv:.0f}x")
```

The convolution has 1,792 parameters and reuses them across all 50,176 spatial positions. One attention projection has 590,592, and a block needs four of them plus a feed-forward network. Convolution buys parameter efficiency with a locality assumption; attention spends parameters to keep the interaction pattern learnable.

### Exercise 3

```python
heads, bytes_per_value = 32, 2          # bfloat16

for n in [512, 2_048, 8_192, 131_072]:
    entries = heads * n * n
    print(f"N={n:7,}: {entries:>18,} score entries = {entries * bytes_per_value / 1e9:8.2f} GB")
```

```text
N=    512:         8,388,608 score entries =     0.02 GB
N=  2,048:       134,217,728 score entries =     0.27 GB
N=  8,192:     2,147,483,648 score entries =     4.29 GB
N=131,072:   549,755,813,888 score entries =  1099.51 GB
```

Sixteen times the sequence length costs 256 times the score memory. Materializing the full matrix is impossible at long context, which is what FlashAttention avoids by computing attention in tiles without ever storing all the scores. During generation, the separate **KV cache** stores keys and values for every past token so each new token is not recomputed; that cache grows linearly with context and often becomes the binding memory constraint in serving (Month 9).

</details>

## Exit test

1. What is feature extraction, and what changed when deep learning replaced hand-designed features?
2. Describe the hierarchy of features learned by a CNN.
3. Define a receptive field, and state how it grows in a CNN.
4. What is the receptive field of a transformer's first attention layer?
5. What is parameter sharing, and what does a CNN share?
6. What prior does convolution encode, and what does a transformer lack as a result of its own sharing?
7. What are a backbone and a head?
8. Name four adaptation strategies in order of cost.
9. Why is a fine-tuning learning rate smaller than a pretraining one?
10. Give three differences between convolution and attention.
11. Why are attention weights called dynamic?
12. Why do Vision Transformers typically need more data than CNNs?

<details>
<summary>Show answers</summary>

1. Turning raw input into a representation a model can use. Deep learning learns that transformation from data rather than designing it by hand, so features are optimized for the task.
2. Early layers detect edges and colors, middle layers textures and motifs, later layers object parts, and the final layers whole objects and class-discriminative structure, each composing the previous level.
3. The region of the input that can influence a given unit's output. In a CNN it grows by 2 per stride-1 3×3 layer and multiplicatively with stride or pooling.
4. The entire context window. Every token can attend to every other token immediately.
5. Reusing the same weights at many positions. A CNN slides one kernel across every spatial location.
6. Convolution encodes locality and translation equivariance. A transformer applies the same weights at every position and attention is permutation-equivariant, so it has no inherent sense of order and needs positional encodings.
7. The backbone is the pretrained feature extractor; the head is the small task-specific layer on top.
8. Frozen feature extraction, partial fine-tuning, full fine-tuning, and parameter-efficient methods such as LoRA and adapters, with the last being cheap in trainable parameters but applied to very large models.
9. Because large updates overwrite the pretrained representations that transfer learning is meant to exploit, causing catastrophic forgetting.
10. Local versus global interaction; static kernel weights versus input-dependent weights; linear versus quadratic cost in sequence length. Depth-dependent versus immediate receptive field is a fourth.
11. Because they are computed from the input itself via query-key similarity, so which positions interact changes with the content rather than being fixed by the architecture.
12. Because they lack convolution's built-in locality prior and must learn it from data, so they need more data, stronger augmentation, or large-scale pretraining to reach comparable performance.

</details>

## Completion criteria

You are done when:

- you can explain receptive fields in a CNN and in a transformer, and why the difference matters
- you can name what each architecture shares and the prior that sharing encodes
- you can choose an adaptation strategy for a given data and compute budget
- you can give the convolution-versus-attention comparison aloud, including the quadratic cost

## References

**Feature hierarchies and receptive fields**
- [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [A guide to receptive field arithmetic](https://distill.pub/2019/computing-receptive-fields/)

**Attention versus convolution**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — section 4 compares the two directly
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — the Vision Transformer and its data requirements
- [FlashAttention](https://arxiv.org/abs/2205.14135) — exact attention without materializing the score matrix

**Transfer learning and fine-tuning**
- [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
- [Universal Language Model Fine-tuning (ULMFiT)](https://arxiv.org/abs/1801.06146) — discriminative learning rates
- [LoRA](https://arxiv.org/abs/2106.09685)
- [PyTorch: transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Videos and code to read

- [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en) — the CNN and attention chapters, with runnable receptive-field and parameter-count examples
- [labmlai annotated implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) — convolution and attention side by side, which makes the comparison in 0.9.5 concrete
- [3Blue1Brown: Neural Networks series](https://www.3blue1brown.com/topics/neural-networks) — the later chapters on GPTs and attention visualize what attention actually computes

## Mapped companion lessons

- [Convolutions from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/04-computer-vision/02-convolutions-from-scratch) and [CNNs — LeNet to ResNet](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/04-computer-vision/03-cnns-lenet-to-resnet) map to locality, receptive fields, and residual architectures.
- [Transfer Learning & Fine-Tuning](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/04-computer-vision/05-transfer-learning) maps to frozen and trainable backbones.
- [Why Transformers](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/07-transformers-deep-dive/01-why-transformers) deepens the convolution, recurrence, and attention comparison.

See the [complete Month 0–1 content map](../../references/ai-engineering-from-scratch-map.md).

## About this lesson

Written to cover section 0.9 of the [Month 0 curriculum](../README.md). Code examples were checked with Python 3.12.
