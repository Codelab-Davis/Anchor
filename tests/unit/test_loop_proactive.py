from __future__ import annotations

import pytest

from anchor.anchor import Anchor
from tests.unit_harness import FakeMemoryStore, scripted_model


def _make_anchor_with_memory(
    ai_responses: list[str],
    light_ai_responses: list[str],
    store: FakeMemoryStore,
) -> tuple[Anchor, object, object]:
    ai_model = scripted_model(*ai_responses)
    light_model = scripted_model(*light_ai_responses)
    anchor = Anchor(
        ai_fn=ai_model,
        light_ai_fn=light_model,
        memory_store=store,
        embed_fn=lambda _text: [0.0],
    )
    return anchor, ai_model, light_model


@pytest.mark.unit
def test_proactive_chunks_inject_memory_message() -> None:
    """When the retriever returns chunks, a [MEMORY RETRIEVAL RESULT] message is
    injected as the second user message before the first AI call."""
    chunk = {"id": "c1", "content": "relevant fact", "source": "wiki", "score": 0.9}
    store = FakeMemoryStore(query_results=[[chunk]])

    # Light model drives decompose; returns the same text as the gap so it
    # becomes the single proactive query (no extra insert by the decomposer).
    anchor, ai_model, _light = _make_anchor_with_memory(
        ai_responses=["The answer.\nDONE"],
        light_ai_responses=["Tell me about X."],
        store=store,
    )
    result = anchor.run("Tell me about X.")

    first_call = ai_model.calls[0]
    assert len(first_call) == 3, "expected system + user query + [MEMORY RETRIEVAL RESULT]"
    assert first_call[1] == {"role": "user", "content": "Tell me about X."}
    assert first_call[2]["role"] == "user"
    assert "[MEMORY RETRIEVAL RESULT]" in first_call[2]["content"]

    assert result.metadata["retrieval_scores"] == [0.9]
    assert result.metadata["decomposed_queries"] == ["Tell me about X."]


@pytest.mark.unit
def test_proactive_no_chunks_no_message_injected() -> None:
    """When the retriever returns no chunks, no memory retrieval message is
    injected and the AI receives only the system prompt and user query."""
    store = FakeMemoryStore(query_results=[[]])

    anchor, ai_model, _light = _make_anchor_with_memory(
        ai_responses=["The answer.\nDONE"],
        light_ai_responses=["Tell me about X."],
        store=store,
    )
    result = anchor.run("Tell me about X.")

    first_call = ai_model.calls[0]
    assert len(first_call) == 2, "expected only system + user query; no retrieval injection"
    assert first_call[0]["role"] == "system"
    assert first_call[1] == {"role": "user", "content": "Tell me about X."}

    assert result.metadata["retrieval_scores"] == []
    assert result.metadata["decomposed_queries"] == ["Tell me about X."]


@pytest.mark.unit
def test_proactive_skipped_when_no_retriever_configured() -> None:
    """When Anchor is initialized without memory_store/embed_fn, proactive
    retrieval is skipped entirely — the decomposer is never called and the AI
    receives only the system prompt and user query."""
    ai_model = scripted_model("The answer.\nDONE")
    light_model = scripted_model()  # raises if called unexpectedly

    anchor = Anchor(ai_fn=ai_model, light_ai_fn=light_model)
    result = anchor.run("Tell me about X.")

    first_call = ai_model.calls[0]
    assert len(first_call) == 2, "expected only system + user query; no retrieval injection"
    assert first_call[0]["role"] == "system"
    assert first_call[1]["role"] == "user"

    assert len(light_model.calls) == 0, "decomposer must not be called when no retriever"
    assert result.metadata["decomposed_queries"] == []
    assert result.metadata["retrieval_scores"] == []


@pytest.mark.unit
def test_proactive_chunk_ids_not_readded_during_remember() -> None:
    """Chunk IDs fetched during proactive retrieval are tracked in seen_ids so
    that if the same chunk is returned again during a REMEMBER pass it is not
    added to retrieved_items a second time."""
    chunk = {"id": "c1", "content": "fact about X", "source": "src", "score": 0.9}
    # Two queued results: one for the proactive query, one for the REMEMBER query.
    # Both return the same chunk to exercise the dedup logic.
    store = FakeMemoryStore(query_results=[[chunk], [chunk]])

    anchor, ai_model, _light = _make_anchor_with_memory(
        ai_responses=[
            "GAP: what is X?\nCONTEXT: investigating\nREMEMBER",
            "Final answer.\nDONE",
        ],
        light_ai_responses=[
            "Tell me about X.",  # proactive decompose: single query matching the gap
            "what is X?",        # REMEMBER decompose: single query matching the gap
        ],
        store=store,
    )
    result = anchor.run("Tell me about X.")

    assert result.metadata["remember_count"] == 1
    assert len(result.retrieved_items) == 1, "duplicate chunk must not be added twice"
    assert result.retrieved_items[0]["id"] == "c1"
    assert result.metadata["retrieval_scores"] == [0.9]
    assert result.metadata["decomposed_queries"] == ["Tell me about X.", "what is X?"]
