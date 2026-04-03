from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnchorConfig:
    """Runtime configuration currently used by the loop."""

    max_remembers: int = 10
