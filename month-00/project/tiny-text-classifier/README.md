# Month 00 Project — Tiny Text Classifier

Build a small text classifier in plain PyTorch before using Hugging Face or a high-level trainer.

## Data flow

`raw text → tokenizer → vocabulary → token IDs → embedding → pooling → linear classifier → logits → cross-entropy → backpropagation → AdamW → validation → inference API`

## Requirements

- Use a small, clearly licensed text-classification dataset or create a tiny synthetic dataset.
- Implement a simple whitespace or regex tokenizer.
- Build a vocabulary with `<pad>` and `<unk>` tokens.
- Convert examples into padded token-ID tensors.
- Implement a PyTorch `Dataset` and `DataLoader`.
- Create an `nn.Module` with an embedding layer, masked mean pooling, and a linear output layer.
- Train with cross-entropy and AdamW.
- Report training and validation loss plus validation accuracy.
- Save and reload a checkpoint.
- Expose one prediction function that accepts raw text.
- Add tests for vocabulary lookup, tensor shapes, padding-aware pooling, and prediction output.

## Suggested structure

```text
tiny-text-classifier/
├── README.md
├── pyproject.toml
├── src/
│   └── text_classifier/
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       └── predict.py
└── tests/
```

## Completion evidence

Record:

- dataset and license
- vocabulary size and maximum sequence length
- model parameter count
- train/validation split
- final training loss, validation loss, and validation accuracy
- one correct and one incorrect prediction
- what changed after reloading the checkpoint
- two limitations and the next improvement you would try

## Architecture review questions

1. Where does padding enter, and how is it prevented from affecting pooled embeddings?
2. Why does the model output logits instead of probabilities during training?
3. What does cross-entropy optimize?
4. Which tensors receive gradients?
5. How does AdamW update parameters?
6. What information does mean pooling discard?
7. What would a transformer change in this pipeline?

