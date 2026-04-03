from __future__ import annotations
import uuid
from datetime import datetime, timezone
from anchor.memory import MemoryStore


class Ingestor:
    """Ingests text into the vector store."""

    def __init__(self, memory_store: MemoryStore, embed_fn, question_fn=None):
        self.memory_store = memory_store
        self.embed_fn = embed_fn
        self.question_fn = question_fn  # light_ai fn for generating questions

    def ingest(self, text: str, source: str = "user") -> str:
        """Embed and store text. Returns the chunk id."""
        if not self.embed_fn:
            raise RuntimeError("No embedding function provided to Ingestor.")

        questions = ""
        if self.question_fn:
            questions = self.question_fn(
                [
                    {
                        "role": "user",
                        "content": (
                            "Output 2-3 short questions this content answers, one per line, nothing else.\n"
                            "No preamble. No numbering. No explanation.\n\n"
                            f"{text}"
                        ),
                    }
                ]
            )

        embedding_input = f"{text}\n\n{questions}" if questions else text
        embedding = self.embed_fn(embedding_input)
        chunk_id = str(uuid.uuid4())

        self.memory_store.add(
            id=chunk_id,
            text=text,
            embedding=embedding,
            metadata={
                "source": source,
                "questions": questions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return chunk_id
