from __future__ import annotations

from pathlib import Path

import pytest

from anchor import Anchor
from tests.unit_harness import FakeMemoryStore, deterministic_embedding


def _dummy_ai(_messages: list[dict]) -> str:
    return "DONE"


def _make_anchor() -> Anchor:
    return Anchor(
        ai_fn=_dummy_ai,
        light_ai_fn=_dummy_ai,
        memory_store=FakeMemoryStore(),
        embed_fn=deterministic_embedding,
    )


@pytest.mark.unit
def test_ingest_file_txt_returns_single_chunk_id(tmp_path: Path) -> None:
    anchor = _make_anchor()
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Plain text content.", encoding="utf-8")

    chunk_ids = anchor.ingest_file(file_path)

    assert isinstance(chunk_ids, list)
    assert len(chunk_ids) == 1
    assert chunk_ids[0] in anchor.ingestor.memory_store.items


@pytest.mark.unit
def test_ingest_file_md_with_multiple_sections_returns_multiple_chunk_ids(
    tmp_path: Path,
) -> None:
    anchor = _make_anchor()
    file_path = tmp_path / "doc.md"
    file_path.write_text(
        "# Section One\n\nFirst section body.\n\n# Section Two\n\nSecond section body.\n",
        encoding="utf-8",
    )

    chunk_ids = anchor.ingest_file(file_path)

    assert len(chunk_ids) == 2
    assert len(set(chunk_ids)) == 2
    for chunk_id in chunk_ids:
        assert chunk_id in anchor.ingestor.memory_store.items


@pytest.mark.unit
def test_ingest_file_unsupported_extension_raises_value_error(tmp_path: Path) -> None:
    anchor = _make_anchor()
    file_path = tmp_path / "data.pdf"
    file_path.write_text("not really a pdf", encoding="utf-8")

    with pytest.raises(ValueError):
        anchor.ingest_file(file_path)


@pytest.mark.unit
def test_ingest_text_still_works() -> None:
    anchor = _make_anchor()

    chunk_id = anchor.ingest_text("hello world", source="user")

    assert chunk_id in anchor.ingestor.memory_store.items
