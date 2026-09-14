#!/usr/bin/env python3
"""Validate generated lesson notebooks and repository-local learning links."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LESSON_DIRS = (ROOT / "month-00" / "lessons", ROOT / "month-01" / "lessons")
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_REFERENCE_ROOT = "https://github.com/rohitg00/ai-engineering-from-scratch/"


def text_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def local_link_errors(markdown: str, owner: Path) -> list[str]:
    errors: list[str] = []
    for raw_target in LINK_PATTERN.findall(markdown):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        resolved = (owner.parent / unquote(parsed.path)).resolve()
        if not resolved.exists():
            errors.append(f"{owner.relative_to(ROOT)} -> {target}")
    return errors


def validate_notebook(path: Path) -> tuple[list[str], int, int]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.relative_to(ROOT)}: invalid JSON: {exc}"], 0, 0

    relative = path.relative_to(ROOT)
    if notebook.get("nbformat") != 4:
        errors.append(f"{relative}: nbformat must be 4")
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    if kernelspec.get("name") != "python3":
        errors.append(f"{relative}: missing Python 3 kernelspec")

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        return errors + [f"{relative}: notebook has no cells"], 0, 0

    ids = [cell.get("id") for cell in cells]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        errors.append(f"{relative}: cell IDs must be present and unique")

    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown = "\n".join(text_source(cell) for cell in markdown_cells)
    source_path = path.with_suffix(".md")
    source_markdown = source_path.read_text(encoding="utf-8")

    if "**Read" not in markdown and "**Watch" not in markdown:
        errors.append(f"{relative}: missing an inline reading or video callout")
    if "## Mapped companion lessons" not in markdown:
        errors.append(f"{relative}: missing the mapped companion lesson section")
    if EXTERNAL_REFERENCE_ROOT not in markdown:
        errors.append(f"{relative}: missing an ai-engineering-from-scratch reference")
    if "# Your work here" not in "\n".join(text_source(cell) for cell in code_cells):
        errors.append(f"{relative}: missing a learner work cell")
    if "<details open" in markdown.lower():
        errors.append(f"{relative}: an answer disclosure is expanded by default")
    if markdown.count("<details>") != markdown.count("</details>"):
        errors.append(f"{relative}: unbalanced details blocks")
    if markdown.count("<summary>") != markdown.count("</summary>"):
        errors.append(f"{relative}: unbalanced summary blocks")
    if markdown.count("<details>") != source_markdown.count("<details>"):
        errors.append(f"{relative}: a source answer disclosure was lost during conversion")
    if markdown.count("<summary>") != source_markdown.count("<summary>"):
        errors.append(f"{relative}: a source answer summary was lost during conversion")

    for cell_index, cell in enumerate(cells):
        kind = cell.get("cell_type")
        source = text_source(cell)
        if kind not in {"markdown", "code", "raw"}:
            errors.append(f"{relative}: cell {cell_index} has invalid type {kind!r}")
            continue
        if kind == "code":
            if cell.get("execution_count") is not None or cell.get("outputs"):
                errors.append(f"{relative}: cell {cell_index} contains execution state")
            if "<details" in source or "<summary>" in source:
                errors.append(f"{relative}: cell {cell_index} exposes disclosure content as code")
            try:
                compile(
                    source,
                    f"{relative}:cell-{cell_index}",
                    "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )
            except SyntaxError as exc:
                errors.append(
                    f"{relative}: cell {cell_index} has invalid Python "
                    f"({exc.msg}, line {exc.lineno})"
                )
        if kind == "markdown" and "<summary>" in source:
            if "<details>" not in source or "</details>" not in source:
                errors.append(
                    f"{relative}: cell {cell_index} does not contain a complete collapsed disclosure"
                )

    errors.extend(local_link_errors(markdown, path))
    return errors, len(cells), len(code_cells)


def main() -> None:
    sources = sorted(path for directory in LESSON_DIRS for path in directory.glob("*.md"))
    notebooks = sorted(path for directory in LESSON_DIRS for path in directory.glob("*.ipynb"))
    errors: list[str] = []

    expected = {path.with_suffix(".ipynb") for path in sources}
    actual = set(notebooks)
    content_map_path = ROOT / "references" / "ai-engineering-from-scratch-map.md"
    content_map = content_map_path.read_text(encoding="utf-8") if content_map_path.exists() else ""
    if not content_map:
        errors.append("Missing references/ai-engineering-from-scratch-map.md")
    for missing in sorted(expected - actual):
        errors.append(f"Missing notebook: {missing.relative_to(ROOT)}")
    for orphan in sorted(actual - expected):
        errors.append(f"Notebook without Markdown source: {orphan.relative_to(ROOT)}")
    for notebook in sorted(expected):
        map_target = "../" + notebook.relative_to(ROOT).as_posix()
        if map_target not in content_map:
            errors.append(f"Content map is missing: {notebook.relative_to(ROOT)}")

    total_cells = 0
    total_code_cells = 0
    for notebook in notebooks:
        notebook_errors, cell_count, code_count = validate_notebook(notebook)
        errors.extend(notebook_errors)
        total_cells += cell_count
        total_code_cells += code_count

    # Validate repository-local links in Markdown navigation and source material.
    for markdown_path in ROOT.rglob("*.md"):
        if ".git" in markdown_path.parts:
            continue
        errors.extend(
            local_link_errors(markdown_path.read_text(encoding="utf-8"), markdown_path)
        )

    if errors:
        print(f"Validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(
        f"Validated {len(notebooks)} notebooks, {total_cells} cells, "
        f"{total_code_cells} code cells, collapsed disclosures, Python syntax, "
        "and repository-local links."
    )


if __name__ == "__main__":
    main()
