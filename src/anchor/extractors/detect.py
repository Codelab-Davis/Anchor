from __future__ import annotations

from pathlib import Path

_EXT_MAP: dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
}


def detect_format(path: str | Path) -> str:
    """Return the logical format name for *path* based on its file extension."""
    suffix = Path(path).suffix.lower()
    try:
        return _EXT_MAP[suffix]
    except KeyError:
        raise ValueError(
            f"Unsupported file extension {suffix!r}: cannot detect format for {path!r}"
        )
