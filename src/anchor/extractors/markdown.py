from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import mistune

from anchor.extractors import NormalizedDocument


# Design choice: returns one NormalizedDocument per section, where a section begins
# at each heading. Content before the first heading is emitted as its own section
# with no heading metadata. This aligns with the chunk-metadata schema's heading/section
# fields and keeps each document semantically coherent for downstream use.


def _inline_text(children: list[dict]) -> str:
    parts = []
    for child in children:
        if "raw" in child:
            parts.append(child["raw"])
        if child.get("children"):
            parts.append(_inline_text(child["children"]))
    return "".join(parts)


def _tokens_to_text(tokens: list[dict]) -> str:
    parts = []
    for tok in tokens:
        t = tok["type"]
        if t == "blank_line":
            continue
        elif t == "block_code":
            marker = tok.get("marker", "```")
            lang = (tok.get("attrs") or {}).get("info") or ""
            parts.append(f"{marker}{lang}\n{tok['raw']}{marker}")
        elif t == "paragraph":
            parts.append(_inline_text(tok.get("children", [])))
        elif t == "heading":
            level = (tok.get("attrs") or {}).get("level", 1)
            parts.append("#" * level + " " + _inline_text(tok.get("children", [])))
        elif "raw" in tok:
            parts.append(tok["raw"])
        else:
            warnings.warn(
                f"Unhandled markdown token type '{t}' has no 'raw' field; content may be lost",
                stacklevel=2,
            )
    return "\n\n".join(p for p in parts if p)


def _heading_label(token: dict) -> str:
    level = (token.get("attrs") or {}).get("level", 1)
    return "#" * level + " " + _inline_text(token.get("children", []))


@dataclass
class _Section:
    heading_token: dict | None
    body_tokens: list[dict]


class MarkdownExtractor:
    def __init__(self) -> None:
        self._md = mistune.create_markdown(renderer="ast")

    def extract(self, path: str | Path) -> list[NormalizedDocument]:
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        tokens: list[dict] = self._md(raw)  # type: ignore[assignment]

        sections: list[_Section] = []
        current_heading: dict | None = None
        current_body: list[dict] = []

        for tok in tokens:
            if tok["type"] == "heading":
                if current_heading is not None or current_body:
                    sections.append(_Section(current_heading, current_body))
                current_heading = tok
                current_body = []
            else:
                current_body.append(tok)

        if current_heading is not None or current_body:
            sections.append(_Section(current_heading, current_body))

        docs: list[NormalizedDocument] = []
        for section in sections:
            metadata: dict[str, str | int | float | bool]
            if section.heading_token is not None:
                content_tokens = [section.heading_token] + section.body_tokens
                metadata = {"heading": _heading_label(section.heading_token)}
            else:
                content_tokens = section.body_tokens
                metadata = {}

            content = _tokens_to_text(content_tokens)
            if not content.strip():
                continue

            docs.append(
                NormalizedDocument(
                    content=content,
                    source=str(path),
                    source_format="markdown",
                    metadata=metadata,
                )
            )

        return docs
