from __future__ import annotations

from pathlib import Path

import pytest

from anchor import Anchor
from tests.unit_harness import FakeMemoryStore, deterministic_embedding, scripted_model


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
def test_ingest_directory_flat_txt_and_md_returns_all_chunk_ids(tmp_path: Path) -> None:
    anchor = _make_anchor()
    (tmp_path / "a.txt").write_text("Plain text content.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title\n\nBody content.\n", encoding="utf-8")

    chunk_ids = anchor.ingest_directory(tmp_path)

    assert isinstance(chunk_ids, list)
    assert len(chunk_ids) == 2
    assert anchor.ingestor is not None
    assert anchor.ingestor.memory_store is not None
    assert isinstance(anchor.ingestor.memory_store, FakeMemoryStore)
    for chunk_id in chunk_ids:
        assert chunk_id in anchor.ingestor.memory_store.items


@pytest.mark.unit
def test_ingest_directory_skips_unsupported_extensions(tmp_path: Path) -> None:
    anchor = _make_anchor()
    (tmp_path / "a.txt").write_text("Plain text content.", encoding="utf-8")
    (tmp_path / "data.pdf").write_text("not really a pdf", encoding="utf-8")
    (tmp_path / "data.csv").write_text("col1,col2\n1,2\n", encoding="utf-8")

    chunk_ids = anchor.ingest_directory(tmp_path)

    assert len(chunk_ids) == 1


@pytest.mark.unit
def test_ingest_directory_handles_nested_structure(tmp_path: Path) -> None:
    anchor = _make_anchor()
    (tmp_path / "a.txt").write_text("Top level content.", encoding="utf-8")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "b.txt").write_text("Nested content.", encoding="utf-8")
    deeper_dir = nested_dir / "deeper"
    deeper_dir.mkdir()
    (deeper_dir / "c.md").write_text("# Deep\n\nDeep content.\n", encoding="utf-8")

    chunk_ids = anchor.ingest_directory(tmp_path)

    assert len(chunk_ids) == 3
    assert anchor.ingestor is not None
    assert anchor.ingestor.memory_store is not None
    assert isinstance(anchor.ingestor.memory_store, FakeMemoryStore)
    for chunk_id in chunk_ids:
        assert chunk_id in anchor.ingestor.memory_store.items


@pytest.mark.unit
def test_ingest_directory_processes_files_in_sorted_order(tmp_path: Path) -> None:
    anchor = _make_anchor()
    (tmp_path / "z.txt").write_text("Z content.", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A content.", encoding="utf-8")
    (tmp_path / "m.txt").write_text("M content.", encoding="utf-8")

    chunk_ids = anchor.ingest_directory(tmp_path)

    expected_order = sorted(tmp_path.rglob("*"))
    expected_sources = [str(p) for p in expected_order if p.is_file()]

    assert anchor.ingestor is not None
    assert anchor.ingestor.memory_store is not None
    assert isinstance(anchor.ingestor.memory_store, FakeMemoryStore)
    actual_sources = [
        anchor.ingestor.memory_store.items[chunk_id]["metadata"]["source"]
        for chunk_id in chunk_ids
    ]
    assert actual_sources == expected_sources


@pytest.mark.unit
def test_ingest_directory_without_memory_store_raises_runtime_error(
    tmp_path: Path,
) -> None:
    anchor = Anchor(
        ai_fn=scripted_model(),
        light_ai_fn=scripted_model(),
    )
    (tmp_path / "a.txt").write_text("Plain text content.", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No memory store configured"):
        anchor.ingest_directory(tmp_path)


@pytest.mark.unit
def test_ingest_directory_empty_returns_empty_list(tmp_path: Path) -> None:
    anchor = _make_anchor()

    chunk_ids = anchor.ingest_directory(tmp_path)

    assert chunk_ids == []
