from __future__ import annotations

import pytest

from anchor.synthesizer import Synthesizer


def _spy_model(response: str = "synthesized answer"):
    """Returns (model_fn, calls). Each call appends the user-role prompt text."""
    calls: list[str] = []

    def model_fn(messages: list[dict]) -> str:
        calls.append(messages[0]["content"])
        return response

    return model_fn, calls


@pytest.mark.unit
def test_empty_chunks_returns_fallback_without_calling_model() -> None:
    model_fn, calls = _spy_model()
    result = Synthesizer(model_fn).synthesize([])
    assert result == "No relevant information found in memory."
    assert calls == []


@pytest.mark.unit
def test_single_chunk_returns_formatted_content_without_calling_model() -> None:
    model_fn, calls = _spy_model()
    chunk = {"source": "doc_A", "content": "The sky is blue."}
    result = Synthesizer(model_fn).synthesize([chunk])
    assert result == "[Source: doc_A]\nThe sky is blue."
    assert calls == []


@pytest.mark.unit
def test_multi_chunk_with_questions_prompt_contains_chunk_content_and_sources() -> None:
    model_fn, calls = _spy_model()
    chunks = [
        {"source": "src_1", "content": "fact one"},
        {"source": "src_2", "content": "fact two"},
    ]
    Synthesizer(model_fn).synthesize(chunks, questions=["What is X?"])
    assert len(calls) == 1
    prompt = calls[0]
    assert "fact one" in prompt
    assert "fact two" in prompt
    assert "[Source: src_1]" in prompt
    assert "[Source: src_2]" in prompt


@pytest.mark.unit
def test_multi_chunk_with_single_question_marks_it_current() -> None:
    model_fn, calls = _spy_model()
    chunks = [
        {"source": "s1", "content": "alpha"},
        {"source": "s2", "content": "beta"},
    ]
    Synthesizer(model_fn).synthesize(chunks, questions=["Tell me about alpha"])
    assert "Tell me about alpha (current)" in calls[0]


@pytest.mark.unit
def test_multi_chunk_with_multiple_questions_only_last_is_marked_current() -> None:
    model_fn, calls = _spy_model()
    chunks = [
        {"source": "s1", "content": "alpha"},
        {"source": "s2", "content": "beta"},
    ]
    questions = ["First question", "Second question", "Third question"]
    Synthesizer(model_fn).synthesize(chunks, questions=questions)
    prompt = calls[0]
    assert "- Third question (current)" in prompt
    assert "- First question\n" in prompt
    assert "- Second question\n" in prompt
    assert "First question (current)" not in prompt
    assert "Second question (current)" not in prompt


@pytest.mark.unit
def test_multi_chunk_without_questions_prompt_contains_chunk_content_and_sources() -> None:
    model_fn, calls = _spy_model()
    chunks = [
        {"source": "src_A", "content": "content alpha"},
        {"source": "src_B", "content": "content beta"},
    ]
    Synthesizer(model_fn).synthesize(chunks)
    assert len(calls) == 1
    prompt = calls[0]
    assert "content alpha" in prompt
    assert "content beta" in prompt
    assert "[Source: src_A]" in prompt
    assert "[Source: src_B]" in prompt
