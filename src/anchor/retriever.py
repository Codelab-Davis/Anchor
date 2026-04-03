from __future__ import annotations
from anchor.memory import MemoryStore


class Retriever:
    """Retrieves information from the vector store based on semantic similarity."""

    def __init__(self, memory_store: MemoryStore, embed_fn):
        self.memory_store = memory_store
        self.embed_fn = embed_fn

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        if not self.embed_fn:
            raise RuntimeError("No embedding function provided to Retriever.")
        embedding = self.embed_fn(query)
        chunks = self.memory_store.query(embedding, top_k=top_k)
        return chunks
