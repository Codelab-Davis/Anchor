from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NormalizedDocument:
    """Canonical output of an extractor and input to a chunker.

    Sits between raw extraction and the chunking step: extractors produce
    NormalizedDocuments, and chunkers consume them. Metadata values must be
    Chroma-compatible types (str, int, float, bool) — None, lists, and dicts
    are rejected at construction time.
    """

    content: str
    source: str
    source_format: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            raise ValueError(
                f"metadata must be a dict, got {type(self.metadata).__name__!r}"
            )
        for key, value in self.metadata.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"metadata keys must be str, got {type(key).__name__!r}: {key!r}"
                )
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"metadata[{key!r}] has unsupported type {type(value).__name__!r}: {value!r}. "
                    "Only str, int, float, and bool are permitted."
                )
