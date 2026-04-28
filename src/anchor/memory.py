from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryStore(ABC):
    """Abstract storage interface for the vector store.

    Implementations must provide add, get, delete, and query.
    Default implementation is ChromaDB (see ChromaMemoryStore).
    """

    @abstractmethod
    def add(self, id: str, text: str, embedding: list[float], metadata: dict) -> None:
        """Store a chunk with its embedding."""
        pass

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return the top-k most similar chunks.

        Returns:
            List of dicts with keys: ``id``, ``content``, ``metadata``, ``score``.
        """
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        """Remove a chunk by ID."""
        pass

    @abstractmethod
    def get(self, id: str) -> dict | None:
        """Retrieve a single chunk by ID."""
        pass


class ChromaMemoryStore(MemoryStore):
    """ChromaDB implementation of MemoryStore."""

    def __init__(self, collection_name: str = "anchor"):
        import chromadb

        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(collection_name)

    def add(self, id: str, text: str, embedding: list[float], metadata: dict) -> None:
        self.collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        hits = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        results = []
        for i, doc_id in enumerate(hits.get("ids", [[]])[0]):
            results.append(
                {
                    "id": doc_id,
                    "content": hits["documents"][0][i],
                    "metadata": hits["metadatas"][0][i],
                    "source": hits["metadatas"][0][i].get("source", "unknown"),
                    "score": hits["distances"][0][i] if "distances" in hits else None,
                }
            )
        return results

    def delete(self, id: str) -> None:
        self.collection.delete(ids=[id])

    def get(self, id: str) -> dict | None:
        hits = self.collection.get(ids=[id])
        ids = hits.get("ids", [])
        if not ids:
            return None

        doc_ids = ids if isinstance(ids[0], str) else ids[0]
        if not doc_ids:
            return None

        documents = hits.get("documents", [])
        metadatas = hits.get("metadatas", [])
        docs = (
            documents
            if (documents and isinstance(documents[0], str))
            else (documents[0] if documents else [])
        )
        metas = (
            metadatas
            if (metadatas and isinstance(metadatas[0], dict))
            else (metadatas[0] if metadatas else [])
        )

        return {
            "id": doc_ids[0],
            "content": docs[0] if docs else "",
            "metadata": metas[0] if metas else {},
        }


class PersistentChromaMemoryStore(ChromaMemoryStore):
    """Persistent ChromaDB implementation of MemoryStore."""

    def __init__(self, path: str, collection_name: str = "anchor"):
        import chromadb

        self.chroma = chromadb.PersistentClient(path=path)
        self.collection = self.chroma.get_or_create_collection(collection_name)
