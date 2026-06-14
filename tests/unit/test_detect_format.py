from __future__ import annotations

import pytest

from anchor.extractors.detect import detect_format


# ---------------------------------------------------------------------------
# known extensions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_txt_returns_text() -> None:
    assert detect_format("notes.txt") == "text"


@pytest.mark.unit
def test_md_returns_markdown() -> None:
    assert detect_format("readme.md") == "markdown"


@pytest.mark.unit
def test_markdown_extension_returns_markdown() -> None:
    assert detect_format("guide.markdown") == "markdown"


# ---------------------------------------------------------------------------
# case-insensitive handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_uppercase_txt_returns_text() -> None:
    assert detect_format("notes.TXT") == "text"


@pytest.mark.unit
def test_uppercase_md_returns_markdown() -> None:
    assert detect_format("readme.MD") == "markdown"


@pytest.mark.unit
def test_uppercase_markdown_returns_markdown() -> None:
    assert detect_format("guide.MARKDOWN") == "markdown"


# ---------------------------------------------------------------------------
# error cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unknown_extension_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"\.pdf"):
        detect_format("report.pdf")


@pytest.mark.unit
def test_no_extension_raises_value_error() -> None:
    with pytest.raises(ValueError):
        detect_format("Makefile")
