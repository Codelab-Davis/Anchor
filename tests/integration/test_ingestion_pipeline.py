from __future__ import annotations

from pathlib import Path

import pytest

from anchor.ingestor import Ingestor
from tests.unit_harness import FakeMemoryStore, deterministic_embedding

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "ingestion"


def _run_ingest(filename: str) -> tuple[str, str, FakeMemoryStore]:
    filepath = str(_FIXTURES / filename)
    content = Path(filepath).read_text()
    store = FakeMemoryStore(allow_fallback_query=True)
    ingestor = Ingestor(
        memory_store=store,
        embed_fn=deterministic_embedding,
        question_fn=lambda _msgs: "What is this about?",
    )
    chunk_id = ingestor.ingest(content, source=filepath)
    return chunk_id, filepath, store


@pytest.mark.unit
def test_ingest_txt_file() -> None:
    chunk_id, filepath, store = _run_ingest("sample.txt")

    chunk = store.get(chunk_id)
    assert chunk is not None
    assert chunk["metadata"]["source"] == filepath
    assert chunk["metadata"]["questions"]
    assert chunk["metadata"]["timestamp"]

    embedding = deterministic_embedding(Path(filepath).read_text())
    results = store.query(embedding, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == chunk_id


@pytest.mark.unit
def test_ingest_md_file() -> None:
    chunk_id, filepath, store = _run_ingest("sample.md")

    chunk = store.get(chunk_id)
    assert chunk is not None
    assert chunk["metadata"]["source"] == filepath
    assert chunk["metadata"]["questions"]
    assert chunk["metadata"]["timestamp"]

    embedding = deterministic_embedding(Path(filepath).read_text())
    results = store.query(embedding, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == chunk_id
