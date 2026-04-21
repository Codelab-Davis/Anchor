from __future__ import annotations

import pytest

from anchor.memory import MemoryStore
from anchor.retriever import Retriever


class _SpyStore(MemoryStore):
    """Minimal MemoryStore that records query calls and returns canned results."""

    def __init__(self, results: list[dict]) -> None:
        self._results = results
        self.query_calls: list[tuple[list[float], int]] = []

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        self.query_calls.append((list(embedding), top_k))
        return list(self._results)

    def add(self, id, text, embedding, metadata):  # noqa: A002
        raise NotImplementedError

    def delete(self, id):  # noqa: A002
        raise NotImplementedError

    def get(self, id):  # noqa: A002
        raise NotImplementedError


@pytest.mark.unit
def test_retrieve_passes_query_to_embed_fn_and_returns_chunks() -> None:
    captured: list[str] = []
    embedding = [0.1, 0.2, 0.3]
    chunks = [{"id": "c1", "content": "hello", "metadata": {}, "score": 0.9}]

    def fake_embed(text: str) -> list[float]:
        captured.append(text)
        return embedding

    store = _SpyStore(chunks)
    result = Retriever(store, fake_embed).retrieve("what is X?", top_k=3)

    assert captured == ["what is X?"]
    assert store.query_calls == [(embedding, 3)]
    assert result == chunks


@pytest.mark.unit
def test_retrieve_default_top_k() -> None:
    embedding = [0.5]
    store = _SpyStore([])

    Retriever(store, lambda _: embedding).retrieve("anything")

    assert store.query_calls == [([0.5], 2)]


@pytest.mark.unit
def test_retrieve_chunks_returned_unchanged() -> None:
    chunks = [
        {"id": "a", "content": "foo", "metadata": {"source": "s"}, "score": 0.8},
        {"id": "b", "content": "bar", "metadata": {"source": "s"}, "score": 0.7},
    ]
    store = _SpyStore(chunks)

    result = Retriever(store, lambda _: [0.0]).retrieve("q", top_k=5)

    assert result == chunks


@pytest.mark.unit
def test_retrieve_raises_without_embed_fn() -> None:
    store = _SpyStore([])

    with pytest.raises(RuntimeError, match="No embedding function"):
        Retriever(store, None).retrieve("query")
