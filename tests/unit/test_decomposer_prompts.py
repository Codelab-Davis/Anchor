from __future__ import annotations

import pytest

from anchor.decomposer import Decomposer


def _spy_model(response: str = ""):
    """Returns (model_fn, prompts). Each call appends the user-role prompt text."""
    prompts: list[str] = []

    def model_fn(messages: list[dict]) -> str:
        prompts.append(messages[0]["content"])
        return response

    return model_fn, prompts


# No-retrieval prompt path
@pytest.mark.unit
def test_no_retrieval_prompt_contains_grounding_language() -> None:
    gap = "What is the capital of France?"
    model_fn, prompts = _spy_model("query one\nquery two")

    Decomposer(model_fn).decompose(gap)

    prompt = prompts[0]
    assert "grounded strictly in the user question below" in prompt
    assert (
        "Do not invent, assume, or reference entities not explicitly mentioned"
        in prompt
    )
    assert f"User question: {gap}" in prompt


# Retrieval-aware prompt path
@pytest.mark.unit
def test_retrieval_aware_prompt_contains_context_facts_and_rules() -> None:
    gap = "missing info about X"
    context = "We are processing project Alpha."
    retrieved = [{"id": "r1", "questions": "q?", "content": "fact about X"}]
    model_fn, prompts = _spy_model("query one\nquery two")

    Decomposer(model_fn).decompose(gap, context=context, retrieved=retrieved)

    prompt = prompts[0]
    assert context in prompt
    assert f"The AI says it's missing: {gap}" in prompt
    assert "r1" in prompt
    assert "fact about X" in prompt
    assert "Rules:" in prompt


# History truncation — only last 6 of N messages included
@pytest.mark.unit
def test_history_truncation_keeps_last_six() -> None:
    history = [{"role": "user", "content": f"msg{i}"} for i in range(8)]
    model_fn, prompts = _spy_model("q")

    Decomposer(model_fn).decompose("gap", history=history)

    prompt = prompts[0]
    assert "msg0" not in prompt
    assert "msg1" not in prompt
    for i in range(2, 8):
        assert f"msg{i}" in prompt


# Retrieved items truncation — only first 8 of N items serialized
@pytest.mark.unit
def test_retrieved_items_truncated_to_eight() -> None:
    retrieved = [
        {"id": f"item{i}", "questions": "", "content": f"content{i}"} for i in range(10)
    ]
    model_fn, prompts = _spy_model("q")

    Decomposer(model_fn).decompose("gap", retrieved=retrieved)

    prompt = prompts[0]
    for i in range(8):
        assert f"item{i}" in prompt
    assert "item8" not in prompt
    assert "item9" not in prompt


# Gap insertion
@pytest.mark.unit
def test_gap_inserted_at_front_when_absent_from_model_output() -> None:
    gap = "What is project Omega?"
    model_fn, _ = _spy_model("query about something else\nanother query")

    queries = Decomposer(model_fn).decompose(gap)

    assert queries[0] == gap


@pytest.mark.unit
def test_gap_not_duplicated_when_already_in_model_output() -> None:
    gap = "What is project Omega?"
    model_fn, _ = _spy_model(f"{gap}\nsome other query")

    queries = Decomposer(model_fn).decompose(gap)

    assert queries.count(gap) == 1
