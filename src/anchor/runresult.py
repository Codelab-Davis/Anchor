from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RunResult:
    kind: Literal["done", "ask"]
    content: str
    stop_reason: Literal["done", "ask", "max_remembers", "error"] = "done"
    retrieved_items: list[dict] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
