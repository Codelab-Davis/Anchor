from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NormalizedDocument:
    content: str
    source: str
    source_format: str
    metadata: dict = field(default_factory=dict)
