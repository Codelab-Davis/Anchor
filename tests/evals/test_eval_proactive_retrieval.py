from __future__ import annotations

import uuid

import pytest

from anchor import Anchor, ChromaMemoryStore
from anchor.ingestor import Ingestor


CHUNKS = {
    "eiffel_tower": (
        "Gustave Eiffel designed the Eiffel Tower for the 1889 World's Fair in Paris."
    ),
    "python_lang": (
        "The Python programming language was created by Guido van Rossum in 1991."
    ),
    "photosynthesis": (
        "Photosynthesis is the process by which plants convert sunlight into chemical energy."
    ),
    "speed_of_light": (
        "The speed of light in a vacuum is approximately 299,792 kilometres per second."
    ),
    "amazon_river": (
        "The Amazon River in South America is the world's largest river by discharge volume."
    ),
    "roman_empire": (
        "The Roman Empire reached its greatest territorial extent under Emperor Trajan in 117 AD."
    ),
}


# (query, target_key) — query should surface the target_key chunk proactively.
HIT_CASES = [
    (
        "Who was the architect of the Eiffel Tower and what event prompted its construction?",
        "eiffel_tower",
    ),
    (
        "Who created the Python programming language and in what year was it first released?",
        "python_lang",
    ),
]


def _seed_store(embed_fn) -> tuple[ChromaMemoryStore, str, dict[str, str]]:
    """Create an isolated collection, seed all CHUNKS, return (store, name, {key: id})."""
    name = f"eval_proactive_{uuid.uuid4().hex}"
    store = ChromaMemoryStore(collection_name=name)
    ingestor = Ingestor(memory_store=store, embed_fn=embed_fn)
    ids: dict[str, str] = {}
    for key, text in CHUNKS.items():
        ids[key] = ingestor.ingest(text, source="test")
    return store, name, ids


@pytest.mark.eval
@pytest.mark.parametrize("run_number", range(3))
@pytest.mark.parametrize("query,target_key", HIT_CASES)
def test_eval_proactive_retrieval_hit(
    ai_fn, light_ai_fn, embed_fn, query, target_key, run_number
):
    store = store_name = None
    try:
        store, store_name, ids = _seed_store(embed_fn)
        target_id = ids[target_key]

        anchor = Anchor(
            ai_fn=ai_fn,
            light_ai_fn=light_ai_fn,
            memory_store=store,
            embed_fn=embed_fn,
        )
        result = anchor.run(query)

        retrieved_ids = {item["id"] for item in result.retrieved_items}

        assert target_id in retrieved_ids, (
            f"[run {run_number}] query={query!r}, target={target_key!r}: "
            f"expected chunk in retrieved_items, got ids={retrieved_ids!r}"
        )
        # stop_reason=="done" means the model emitted DONE on its first response —
        # the only retrieval that could have happened is the proactive pass before
        # that response, so the target chunk was retrieved proactively.
        assert result.stop_reason == "done", (
            f"[run {run_number}] query={query!r}, target={target_key!r}: "
            f"expected stop_reason='done' (proactive retrieval, no REMEMBER triggered), "
            f"got {result.stop_reason!r} — content: {result.content!r}"
        )
    finally:
        if store is not None and store_name is not None:
            store.chroma.delete_collection(store_name)
