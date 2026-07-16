from __future__ import annotations

from pathlib import Path

from anchor.extractors import NormalizedDocument


class TextExtractor:
    def extract(self, path: str | Path) -> NormalizedDocument:
        path = Path(path)
        content = path.read_text(encoding="utf-8")

        line_end = len(content.splitlines())

        return NormalizedDocument(
            content=content,
            source=str(Path(path).resolve()),
            source_format="text",
            metadata={
                "line_start": 1,
                "line_end": line_end,
                "char_start": 0,
                "char_end": len(content),
            },
        )
