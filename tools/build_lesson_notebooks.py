#!/usr/bin/env python3
"""Build interactive Jupyter notebooks from the Month 0 and Month 1 lessons.

The Markdown lessons remain the maintainable source. The generated notebooks:

* turn visible Python fences into executable cells;
* keep every answer and solution inside its collapsed ``<details>`` block;
* add an empty learner cell after hands-on exercise prompts;
* place selected readings and videos beside the concepts they support; and
* rewrite lesson-to-lesson links to point to notebooks.

Run this script from any directory. It writes notebooks beside their sources.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LESSON_DIRS = (ROOT / "month-00" / "lessons", ROOT / "month-01" / "lessons")


# These resources are inserted directly after the matching section heading in the
# notebook. Primary-reference lists in the source lessons remain available for
# deeper study; these callouts identify the best next reading at the point of use.
SECTION_RESOURCES: dict[str, dict[str, str]] = {
    "month-00/lessons/01-python-for-modern-ai.md": {
        "## 0.1.14 — Async and `await`": (
            "> **Read next:** [Python's `asyncio` documentation]"
            "(https://docs.python.org/3/library/asyncio.html) explains the event loop, "
            "tasks, queues, and synchronization primitives used in concurrent AI services. "
            "The earlier [Python data model chapter]"
            "(https://docs.python.org/3/reference/datamodel.html) is useful when the object "
            "protocols in this lesson feel implicit."
        ),
    },
    "month-00/lessons/02-numpy-and-tensor-manipulation.md": {
        "## Lesson 0.2.6 — Broadcasting": (
            "> **Read next:** Work through NumPy's [broadcasting guide]"
            "(https://numpy.org/doc/stable/user/basics.broadcasting.html), then compare it "
            "with PyTorch's [broadcasting semantics]"
            "(https://docs.pytorch.org/docs/stable/notes/broadcasting.html). The shared rules "
            "explain most shape behavior in model code."
        ),
    },
    "month-00/lessons/03-linear-algebra-for-deep-learning.md": {
        "## How this connects to transformers": (
            "> **Read next:** Revisit the attention equations in [Attention Is All You Need]"
            "(https://arxiv.org/abs/1706.03762) and identify the matrix multiplication, "
            "transpose, scaling, and row-wise normalization operations from this lesson."
        ),
    },
    "month-00/lessons/04-probability-and-statistics.md": {
        "## Entropy, cross-entropy, and KL divergence": (
            "> **Read next:** The [Deep Learning textbook's probability chapter]"
            "(https://www.deeplearningbook.org/contents/prob.html) develops entropy and KL "
            "divergence from probability foundations. For implementation details, see "
            "[`torch.nn.CrossEntropyLoss`]"
            "(https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)."
        ),
    },
    "month-00/lessons/05-calculus-for-neural-networks.md": {
        "## Chain rule and computational graphs": (
            "> **Watch and build:** 3Blue1Brown's [gradient descent explanation]"
            "(https://www.youtube.com/watch?v=IHZwWFHWa-w) gives the geometric intuition; "
            "Andrej Karpathy's [micrograd walkthrough]"
            "(https://www.youtube.com/watch?v=VMj-3S1tku0) turns the same chain rule into a "
            "small automatic-differentiation engine."
        ),
    },
    "month-00/lessons/06-neural-network-fundamentals.md": {
        "## Forward pass, loss, and backward pass": (
            "> **Watch:** [But what is a neural network?]"
            "(https://www.youtube.com/watch?v=aircAruvnKk) develops an approachable visual "
            "model of layers and activations. Follow it with the [micrograd walkthrough]"
            "(https://www.youtube.com/watch?v=VMj-3S1tku0) for a code-level account of the "
            "forward and backward passes."
        ),
    },
    "month-00/lessons/07-training-fundamentals.md": {
        "## SGD, momentum, Adam, and AdamW": (
            "> **Read next:** [Adam: A Method for Stochastic Optimization]"
            "(https://arxiv.org/abs/1412.6980) defines the adaptive moment updates. "
            "[Decoupled Weight Decay Regularization]"
            "(https://arxiv.org/abs/1711.05101) explains why AdamW separates weight decay "
            "from the gradient update."
        ),
    },
    "month-00/lessons/08-pytorch-fundamentals.md": {
        "## A complete training and validation loop": (
            "> **Read next:** PyTorch's [Quickstart]"
            "(https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) "
            "shows the same dataset, model, loss, optimizer, train, and evaluation sequence "
            "using the current library APIs."
        ),
    },
    "month-00/lessons/09-deep-learning-architecture-concepts.md": {
        "## Receptive fields and parameter sharing": (
            "> **Read next:** Use [Deep Residual Learning for Image Recognition]"
            "(https://arxiv.org/abs/1512.03385) as a concrete CNN architecture, then contrast "
            "its spatial inductive bias with [Attention Is All You Need]"
            "(https://arxiv.org/abs/1706.03762)."
        ),
    },
    "month-00/lessons/10-representation-learning.md": {
        "## From CNN and DINO features to LLM and multimodal embeddings": (
            "> **Read and watch:** [CLIP]"
            "(https://arxiv.org/abs/2103.00020) shows a shared image-text embedding space. "
            "Stanford CS224N's [word vectors lecture]"
            "(https://www.youtube.com/watch?v=DzpHeXVSC5I) develops the distributional "
            "intuition behind learned text representations."
        ),
    },
    "month-00/lessons/11-self-supervised-learning.md": {
        "## Contrastive learning": (
            "> **Read next:** [A Simple Framework for Contrastive Learning of Visual "
            "Representations (SimCLR)](https://arxiv.org/abs/2002.05709) is a clean reference "
            "for augmented positive pairs, in-batch negatives, and the contrastive loss."
        ),
    },
    "month-00/lessons/12-basic-nlp-concepts.md": {
        "## Word2Vec and GloVe": (
            "> **Watch and read:** Stanford CS224N's [word vectors lecture]"
            "(https://www.youtube.com/watch?v=DzpHeXVSC5I) explains prediction-based word "
            "embeddings. Pair it with the original [GloVe paper]"
            "(https://aclanthology.org/D14-1162/) to compare local prediction with global "
            "co-occurrence statistics."
        ),
    },
    "month-00/lessons/13-information-retrieval-fundamentals.md": {
        "## TF-IDF and BM25": (
            "> **Read next:** The free [Introduction to Information Retrieval]"
            "(https://nlp.stanford.edu/IR-book/) develops inverted indexes, TF-IDF, and "
            "ranking evaluation. [The Probabilistic Relevance Framework: BM25 and Beyond]"
            "(https://staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) gives the "
            "probabilistic account of BM25."
        ),
    },
    "month-00/lessons/14-basic-data-engineering.md": {
        "## JSON, JSONL, and Parquet": (
            "> **Read next:** The [Apache Parquet documentation]"
            "(https://parquet.apache.org/docs/) explains columnar storage and file layout. "
            "The [JSON Lines specification](https://jsonlines.org/) captures the simple "
            "streaming contract behind JSONL datasets."
        ),
    },
    "month-00/lessons/15-apis-and-web-fundamentals.md": {
        "## HTTP request and response": (
            "> **Read next:** [RFC 9110: HTTP Semantics]"
            "(https://www.rfc-editor.org/rfc/rfc9110) is the authoritative reference for "
            "methods, status codes, headers, representation metadata, and caching semantics."
        ),
    },
    "month-00/lessons/16-linux-containers-and-git.md": {
        "## Shell, processes, and environment": (
            "> **Watch and practice:** MIT's Missing Semester [course overview and shell "
            "lecture](https://www.youtube.com/watch?v=Z56Jmr9Z34Q) demonstrates navigation, "
            "streams, pipes, permissions, and command composition in a real terminal."
        ),
    },
    "month-00/lessons/17-software-engineering-for-ai.md": {
        "## Testing layers": (
            "> **Read next:** Google's [Software Engineering at Google: Testing Overview]"
            "(https://abseil.io/resources/swe-book/html/ch11.html) explains test scope, "
            "fidelity, and maintainability. Translate its principles to data contracts, "
            "model boundaries, and AI service behavior."
        ),
    },
    "month-00/lessons/18-mlops-fundamentals.md": {
        "## The lifecycle": (
            "> **Read next:** [The ML Test Score]"
            "(https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) "
            "turns production readiness into concrete checks across data, models, pipelines, "
            "and monitoring."
        ),
    },
    "month-01/lessons/01-transformer-architecture.md": {
        "## Lesson 1.1.2 — Q, K, and V are learned projections": (
            "> **Read and watch:** Read the attention section of [Attention Is All You Need]"
            "(https://arxiv.org/abs/1706.03762), then use Andrej Karpathy's [build GPT from "
            "scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) to see the projections, "
            "causal mask, softmax, and value aggregation implemented step by step."
        ),
    },
    "month-01/lessons/02-position-information.md": {
        "## Lesson 1.2.4 — RoPE rotates queries and keys": (
            "> **Read next:** [RoFormer]"
            "(https://arxiv.org/abs/2104.09864) derives rotary position embedding and its "
            "relative-position behavior. Compare the mechanism with [ALiBi]"
            "(https://arxiv.org/abs/2108.12409), which adds a distance-dependent attention "
            "bias instead."
        ),
    },
    "month-01/lessons/03-tokenization.md": {
        "## Lesson 1.3.2 — Vocabulary construction and BPE": (
            "> **Read next:** [SentencePiece]"
            "(https://arxiv.org/abs/1808.06226) explains language-independent subword training "
            "directly from raw text. The [Hugging Face tokenizers course chapter]"
            "(https://huggingface.co/learn/llm-course/chapter6/1) adds practical inspection "
            "and training workflows."
        ),
    },
    "month-01/lessons/04-llm-training-objectives.md": {
        "## Lesson 1.4.1 — Autoregressive factorization": (
            "> **Read and watch:** [Language Models are Unsupervised Multitask Learners]"
            "(https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) "
            "connects next-token prediction to broad capabilities. Karpathy's [build GPT from "
            "scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) shows how shifted targets "
            "become an autoregressive training batch."
        ),
    },
    "month-01/lessons/05-modern-transformer-and-llm-variants.md": {
        "## Lesson 1.5.4 — MHA, MQA, and GQA": (
            "> **Read next:** [Fast Transformer Decoding: One Write-Head is All You Need]"
            "(https://arxiv.org/abs/1911.02150) introduces multi-query attention. [GQA]"
            "(https://arxiv.org/abs/2305.13245) develops the grouped compromise between MHA "
            "quality and MQA cache efficiency."
        ),
    },
    "month-01/lessons/06-llm-inference-and-decoding.md": {
        "## Lesson 1.6.3 — Top-k and top-p restrict the candidate set": (
            "> **Read next:** [The Curious Case of Neural Text Degeneration]"
            "(https://arxiv.org/abs/1904.09751) motivates nucleus sampling by showing why "
            "maximization and fixed candidate sets can produce brittle or repetitive text."
        ),
    },
    "month-01/lessons/07-context-windows-and-limits.md": {
        "## Lesson 1.7.5 — Advertised capacity versus useful capacity": (
            "> **Read next:** [Lost in the Middle]"
            "(https://arxiv.org/abs/2307.03172) measures how placement changes retrieval from "
            "long contexts. Pair it with [FlashAttention]"
            "(https://arxiv.org/abs/2205.14135) to separate behavioral context limits from "
            "the systems problem of computing attention efficiently."
        ),
    },
    "month-01/lessons/08-scaling-and-foundation-models.md": {
        "## Lesson 1.8.4 — Compute-optimal training": (
            "> **Read next:** [Scaling Laws for Neural Language Models]"
            "(https://arxiv.org/abs/2001.08361) establishes empirical power-law trends. "
            "[Training Compute-Optimal Large Language Models]"
            "(https://arxiv.org/abs/2203.15556) revises the allocation between model size "
            "and training tokens."
        ),
    },
}


def lesson_sources() -> list[Path]:
    return sorted(path for directory in LESSON_DIRS for path in directory.glob("*.md"))


def rewrite_lesson_links(markdown: str, source_path: Path, known_sources: set[Path]) -> str:
    """Make links between lesson files open their interactive notebook versions."""

    pattern = re.compile(r"(?P<prefix>\]\()(?P<target>[^)#\s]+\.md)(?P<anchor>#[^)]*)?(?P<suffix>\))")

    def replace(match: re.Match[str]) -> str:
        target = match.group("target")
        if "://" in target or target.startswith("/"):
            return match.group(0)
        resolved = (source_path.parent / target).resolve()
        if resolved not in known_sources:
            return match.group(0)
        notebook_target = str(Path(target).with_suffix(".ipynb"))
        return f"]({notebook_target}{match.group('anchor') or ''})"

    return pattern.sub(replace, markdown)


def inject_section_resources(markdown: str, source_path: Path) -> str:
    relative = source_path.relative_to(ROOT).as_posix()
    resources = SECTION_RESOURCES.get(relative, {})
    if not resources:
        return markdown

    output: list[str] = []
    inserted: set[str] = set()
    for line in markdown.splitlines():
        output.append(line)
        heading = line.strip()
        if heading in resources:
            output.extend(("", resources[heading]))
            inserted.add(heading)

    missing = set(resources) - inserted
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Resource heading not found in {relative}: {missing_list}")
    return "\n".join(output) + "\n"


def split_cells(markdown: str) -> list[tuple[str, str]]:
    """Split lesson Markdown into Markdown and executable Python cell sources."""

    cells: list[tuple[str, str]] = []
    buffer: list[str] = []
    lines = markdown.splitlines()
    index = 0
    details_depth = 0
    generic_fence: str | None = None

    def flush_markdown() -> None:
        text = "\n".join(buffer).strip("\n")
        buffer.clear()
        if text.strip():
            cells.append(("markdown", text + "\n"))

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        # Convert visible Python fences into executable cells. Fences inside a
        # details block stay Markdown so hidden solutions remain collapsed.
        if details_depth == 0 and generic_fence is None and stripped == "```python":
            flush_markdown()
            index += 1
            code: list[str] = []
            while index < len(lines) and lines[index].strip() != "```":
                code.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError("Unclosed Python fence")
            cells.append(("code", "\n".join(code).rstrip() + "\n"))
            index += 1
            continue

        # Keep answer blocks in one Markdown cell so their HTML disclosure state
        # remains intact in Jupyter and on GitHub.
        if details_depth == 0 and generic_fence is None and stripped.startswith("<details"):
            flush_markdown()
        if generic_fence is None and stripped.startswith("<details"):
            details_depth += 1

        # Start a fresh explanatory cell at each visible H2/H3 heading.
        if (
            details_depth == 0
            and generic_fence is None
            and re.match(r"^#{2,3}\s+", line)
            and buffer
        ):
            flush_markdown()

        buffer.append(line)

        if generic_fence is None and stripped.startswith("```"):
            generic_fence = stripped[:3]
        elif generic_fence is not None and stripped == generic_fence:
            generic_fence = None

        if generic_fence is None and stripped == "</details>":
            details_depth -= 1
            if details_depth < 0:
                raise ValueError("Closing details tag without opener")
            if details_depth == 0:
                flush_markdown()

        index += 1

    flush_markdown()
    if details_depth:
        raise ValueError("Unclosed details block")
    if generic_fence:
        raise ValueError("Unclosed Markdown fence")
    return cells


def is_exercise_prompt(source: str) -> bool:
    """Return True for a visible exercise section that can use a learner cell."""

    first_heading = next(
        (line for line in source.splitlines() if re.match(r"^#{2,3}\s+", line)),
        "",
    )
    return bool(re.search(r"\b(exercise|hands-on)\b", first_heading, flags=re.IGNORECASE))


def cell_id(relative_source: str, index: int, cell_type: str) -> str:
    material = f"{relative_source}:{index}:{cell_type}".encode()
    return hashlib.sha1(material).hexdigest()[:12]


def build_notebook(source_path: Path, known_sources: set[Path]) -> dict[str, object]:
    relative = source_path.relative_to(ROOT).as_posix()
    markdown = source_path.read_text(encoding="utf-8")
    markdown = inject_section_resources(markdown, source_path)
    markdown = rewrite_lesson_links(markdown, source_path, known_sources)
    split = split_cells(markdown)

    generated: list[tuple[str, str]] = [
        (
            "markdown",
            "> **Interactive lesson:** Run the code cells, change inputs, and record your "
            "observations in the learner cells. This notebook is generated from the "
            f"[Markdown source]({source_path.name}); edit that source and rebuild rather "
            "than editing generated cells by hand.\n",
        )
    ]
    for cell_type, source in split:
        generated.append((cell_type, source))
        if cell_type == "markdown" and is_exercise_prompt(source) and "<details" not in source:
            generated.append(("code", "# Your work here\n"))

    cells: list[dict[str, object]] = []
    for index, (cell_type, source) in enumerate(generated):
        base: dict[str, object] = {
            "cell_type": cell_type,
            "id": cell_id(relative, index, cell_type),
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            base.update({"execution_count": None, "outputs": []})
        cells.append(base)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "roadmap": {"generated_from": source_path.name},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    sources = lesson_sources()
    known_sources = {path.resolve() for path in sources}
    missing_resource_maps = {
        path.relative_to(ROOT).as_posix()
        for path in sources
        if path.relative_to(ROOT).as_posix() not in SECTION_RESOURCES
    }
    if missing_resource_maps:
        raise ValueError(
            "Every lesson needs an inline resource mapping: "
            + ", ".join(sorted(missing_resource_maps))
        )

    for source_path in sources:
        notebook = build_notebook(source_path, known_sources)
        destination = source_path.with_suffix(".ipynb")
        destination.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(destination.relative_to(ROOT))

    print(f"Built {len(sources)} notebooks.")


if __name__ == "__main__":
    main()
