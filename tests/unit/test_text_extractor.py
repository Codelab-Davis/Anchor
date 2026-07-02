from __future__ import annotations

from pathlib import Path

import pytest

from anchor.extractors.text import TextExtractor


def run_extract(tmp_path: Path, content: str):
    text_file = tmp_path / "sample.txt"
    text_file.write_text(content, encoding="utf-8")
    return TextExtractor().extract(text_file), text_file


@pytest.mark.unit
def test_source_format_is_text(tmp_path: Path) -> None:
    doc, _ = run_extract(tmp_path, "line one\nline two\n")
    assert doc.source_format == "text"


@pytest.mark.unit
def test_source_is_file_path(tmp_path: Path) -> None:
    doc, text_file = run_extract(tmp_path, "line one\nline two\n")
    assert doc.source == str(text_file)


@pytest.mark.unit
def test_content_matches_file_content(tmp_path: Path) -> None:
    content = "line one\nline two\n"
    doc, _ = run_extract(tmp_path, content)
    assert doc.content == content


@pytest.mark.unit
def test_line_start_is_one(tmp_path: Path) -> None:
    doc, _ = run_extract(tmp_path, "line one\nline two\n")
    assert doc.metadata["line_start"] == 1


@pytest.mark.unit
def test_line_end_matches_actual_line_count(tmp_path: Path) -> None:
    doc, _ = run_extract(tmp_path, "line one\nline two\nline three\n")
    assert doc.metadata["line_end"] == 3


@pytest.mark.unit
def test_char_start_is_zero(tmp_path: Path) -> None:
    doc, _ = run_extract(tmp_path, "line one\nline two\n")
    assert doc.metadata["char_start"] == 0


@pytest.mark.unit
def test_char_end_matches_content_length(tmp_path: Path) -> None:
    content = "line one\nline two\n"
    doc, _ = run_extract(tmp_path, content)
    assert doc.metadata["char_end"] == len(content)


@pytest.mark.unit
def test_empty_file_edge_case(tmp_path: Path) -> None:
    doc, _ = run_extract(tmp_path, "")
    assert doc.metadata["line_end"] == 0
    assert doc.metadata["char_end"] == 0
