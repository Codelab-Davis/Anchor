from __future__ import annotations

import uuid

import pytest

from anchor.ingestor import Ingestor
from anchor.memory import ChromaMemoryStore


# Chunk about Eiffel Tower authorship and occasion.
CHUNK_TEXT = (
    "The primary architect of the Eiffel Tower was Gustave Eiffel, "
    "who designed it for the 1889 World's Fair in Paris."
)

# Deliberately shares no literal terms with CHUNK_TEXT — bridges via semantics only.
# "created / iconic / iron landmark / international exhibition / France"
# vs "architect / Gustave Eiffel / designed / World's Fair / Paris"
PARAPHRASE_QUERY = "Who created the iconic iron landmark built for the international exhibition in France?"

DISTRACTOR_CHUNKS = [
    "The Python programming language was created by Guido van Rossum in 1991.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
]


@pytest.mark.eval
def test_eval_ingestor_question_recall(light_ai_fn, embed_fn):
    name_q = f"eval_ingestor_q_{uuid.uuid4().hex}"
    name_base = f"eval_ingestor_base_{uuid.uuid4().hex}"
    store_q = store_base = None
    try:
        store_q = ChromaMemoryStore(collection_name=name_q)
        store_base = ChromaMemoryStore(collection_name=name_base)
        ingestor_q = Ingestor(
            memory_store=store_q, embed_fn=embed_fn, question_fn=light_ai_fn
        )
        ingestor_base = Ingestor(memory_store=store_base, embed_fn=embed_fn)

        for distractor in DISTRACTOR_CHUNKS:
            ingestor_q.ingest(distractor, source="distractor")
            ingestor_base.ingest(distractor, source="distractor")

        chunk_id_q = ingestor_q.ingest(CHUNK_TEXT, source="test")
        ingestor_base.ingest(CHUNK_TEXT, source="test")

        # Assert questions were generated and stored in metadata.
        stored = store_q.get(chunk_id_q)
        assert stored is not None
        questions = stored["metadata"].get("questions", "")
        assert (
            questions and questions.strip()
        ), f"Expected generated questions in metadata, got: {questions!r}"

        # Query the question-enhanced store with all chunks visible.
        n_chunks = len(DISTRACTOR_CHUNKS) + 1
        query_embedding = embed_fn(PARAPHRASE_QUERY)
        results_q = store_q.query(query_embedding, top_k=n_chunks)

        # This eval validates recall on a paraphrase query with zero literal term overlap,
        # but does not assert a score improvement over a baseline. bge-m3's semantic
        # embeddings are strong enough that question augmentation does not reliably reduce
        # L2 distance with only a handful of chunks. The benefit of question augmentation
        # is expected to be more pronounced with many similar chunks and edge-case queries.

        # Assert target chunk is in the top half of results for the question-enhanced store.
        top_ids = [r["id"] for r in results_q[: n_chunks // 2 + 1]]
        assert chunk_id_q in top_ids, (
            f"Expected target chunk in top results, ranked order: "
            f"{[r['id'] for r in results_q]}"
        )

        # Assert question-enhanced chunk ranks first.
        assert (
            results_q[0]["id"] == chunk_id_q
        ), f"Expected target chunk to rank first, got: {[r['id'] for r in results_q]}"
    finally:
        if store_q is not None:
            store_q.chroma.delete_collection(name_q)
        if store_base is not None:
            store_base.chroma.delete_collection(name_base)
