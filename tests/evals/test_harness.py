# tests/evals/test_harness.py
from __future__ import annotations

import uuid
import pytest
from anchor import Anchor
from anchor.memory import ChromaMemoryStore


SEED_CHUNKS = [
    ("The project codename is ORCHID-7.", "seed"),
    ("The max retry limit is 7 for standard tenants.", "seed"),
    ("Severity levels are SABLE, BRASS, and IVORY in descending order.", "seed"),
]


@pytest.fixture
def anchor(ai_fn, light_ai_fn, embed_fn):
    collection_name = f"test_harness_{uuid.uuid4().hex}"
    memory_store = ChromaMemoryStore(collection_name=collection_name)
    instance = Anchor(
        ai_fn=ai_fn,
        light_ai_fn=light_ai_fn,
        memory_store=memory_store,
        embed_fn=embed_fn,
    )
    for text, source in SEED_CHUNKS:
        instance.ingest_text(text, source=source)
    try:
        yield instance
    finally:
        # Ensure eval state does not leak between test runs.
        memory_store.chroma.delete_collection(collection_name)


@pytest.mark.eval
@pytest.mark.parametrize("run_number", range(5))
def test_harness_single_hop(anchor, run_number):
    """Sanity check — anchor retrieves a seeded fact and returns DONE."""
    result = anchor.run("What is the project codename?")
    assert result.stop_reason == "done", f"Expected done, got {result.stop_reason!r}"
    assert (
        "ORCHID-7" in result.content
    ), f"Expected ORCHID-7 in response, got: {result.content!r}"
