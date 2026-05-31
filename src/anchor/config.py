from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnchorConfig:
    """Runtime configuration currently used by the loop."""

    max_remembers: int = 10
    log_path: str | Path | None = None
