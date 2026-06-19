from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from anchor.extractors import NormalizedDocument
from anchor.extractors.markdown import MarkdownExtractor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_extract(tmp_path: Path, content: str) -> list[NormalizedDocument]:
    md_file = tmp_path / "sample.md"
    md_file.write_text(textwrap.dedent(content), encoding="utf-8")
    return MarkdownExtractor().extract(md_file)


# ---------------------------------------------------------------------------
# multiple headings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multiple_headings_produces_one_doc_per_section(tmp_path: Path) -> None:
    docs = run_extract(
        tmp_path,
        """\
        # Alpha

        First section content.

        ## Beta

        Second section content.

        ## Gamma

        Third section content.
        """,
    )
    assert len(docs) == 3


@pytest.mark.unit
def test_heading_metadata_is_correct_per_section(tmp_path: Path) -> None:
    docs = run_extract(
        tmp_path,
        """\
        # Alpha

        First.

        ## Beta

        Second.
        """,
    )
    assert docs[0].metadata["heading"] == "# Alpha"
    assert docs[1].metadata["heading"] == "## Beta"


@pytest.mark.unit
def test_heading_level_reflected_in_metadata(tmp_path: Path) -> None:
    docs = run_extract(
        tmp_path,
        """\
        ### Deep Heading

        Some content.
        """,
    )
    assert docs[0].metadata["heading"] == "### Deep Heading"


@pytest.mark.unit
def test_section_content_contains_expected_text(tmp_path: Path) -> None:
    docs = run_extract(
        tmp_path,
        """\
        # Title

        Introduction paragraph.

        ## Details

        Detail paragraph.
        """,
    )
    assert "Introduction paragraph." in docs[0].content
    assert "Detail paragraph." in docs[1].content


@pytest.mark.unit
def test_source_format_is_markdown(tmp_path: Path) -> None:
    docs = run_extract(tmp_path, "# Hello\n\nWorld.\n")
    assert all(doc.source_format == "markdown" for doc in docs)


@pytest.mark.unit
def test_source_is_file_path(tmp_path: Path) -> None:
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Hello\n\nWorld.\n", encoding="utf-8")
    docs = MarkdownExtractor().extract(md_file)
    assert all(doc.source == str(md_file) for doc in docs)


# ---------------------------------------------------------------------------
# content before first heading
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_preamble_before_first_heading_is_extracted(tmp_path: Path) -> None:
    docs = run_extract(
        tmp_path,
        """\
        Preamble text before any heading.

        ## First Heading

        Section content.
        """,
    )
    assert len(docs) == 2
    assert "Preamble text before any heading." in docs[0].content


@pytest.mark.unit
def test_preamble_section_has_no_heading_metadata(tmp_path: Path) -> None:
    docs = run_extract(
        tmp_path,
        """\
        Preamble.

        ## Heading

        Content.
        """,
    )
    assert "heading" not in docs[0].metadata


@pytest.mark.unit
def test_preamble_only_document_has_no_heading_metadata(tmp_path: Path) -> None:
    docs = run_extract(tmp_path, "No headings here at all.\n")
    assert len(docs) == 1
    assert "heading" not in docs[0].metadata


# ---------------------------------------------------------------------------
# fenced code blocks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fenced_code_block_content_preserved_verbatim(tmp_path: Path) -> None:
    docs = run_extract(
        tmp_path,
        """\
        ## Code Section

        ```python
        def foo():
            pass
        ```
        """,
    )
    assert len(docs) == 1
    content = docs[0].content
    assert "def foo():" in content
    assert "    pass" in content  # 4-space indentation must survive


@pytest.mark.unit
def test_fenced_code_block_without_heading(tmp_path: Path) -> None:
    docs = run_extract(
        tmp_path,
        """\
        Some prose.

        ```
        plain block
            indented line
        ```
        """,
    )
    assert len(docs) == 1
    content = docs[0].content
    assert "plain block" in content
    assert "    indented line" in content


@pytest.mark.unit
def test_fenced_code_with_multiline_body(tmp_path: Path) -> None:
    docs = run_extract(
        tmp_path,
        """\
        ## Snippet

        ```python
        x = 1
        y = 2
        z = x + y
        ```
        """,
    )
    content = docs[0].content
    assert "x = 1" in content
    assert "y = 2" in content
    assert "z = x + y" in content
