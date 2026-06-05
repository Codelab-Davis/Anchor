from __future__ import annotations

import re

import pytest

from anchor.decomposer import Decomposer


_GAP = "What is the maximum data transfer rate of the Orion-5 satellite uplink?"

_CONTEXT = (
    "Helios Corp operates the Orion-5 satellite for government clients. "
    "The mission requires a high-bandwidth uplink with AES-256 encryption."
)

_RETRIEVED = [
    {
        "id": "ret-1",
        "questions": "What encryption does the Orion-5 uplink use?",
        "content": "Orion-5 uplink uses AES-256 encryption as mandated by Helios Corp.",
    },
    {
        "id": "ret-2",
        "questions": "Who operates the Orion-5 satellite?",
        "content": "Helios Corp operates the Orion-5 satellite on behalf of government clients.",
    },
]


def _named_tokens(text: str) -> set[str]:
    """Return lowercased capitalized tokens (≥2 chars) as named-entity proxies."""
    return {
        m.group().lower()
        for m in re.finditer(r"\b[A-Z][A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b", text)
    }


@pytest.mark.eval
@pytest.mark.parametrize("run_number", range(5))
def test_eval_decomposer_no_retrieval(light_ai_fn, run_number):
    decomposer = Decomposer(model_fn=light_ai_fn)
    queries = decomposer.decompose(_GAP)

    assert (
        len(queries) >= 2
    ), f"[run {run_number}] expected at least 2 queries, got {len(queries)}: {queries!r}"
    assert (
        queries[0] == _GAP
    ), f"[run {run_number}] gap must be first query, got {queries[0]!r}"
    allowed_text = _GAP.lower()
    for q in queries:
        for entity in _named_tokens(q):
            assert entity in allowed_text, (
                f"[run {run_number}] query {q!r} contains entity {entity!r} "
                f"not present in gap {_GAP!r}"
            )


@pytest.mark.eval
@pytest.mark.parametrize("run_number", range(5))
def test_eval_decomposer_retrieval_aware(light_ai_fn, run_number):
    decomposer = Decomposer(model_fn=light_ai_fn)

    nr_queries = decomposer.decompose(_GAP)
    ra_queries = decomposer.decompose(_GAP, context=_CONTEXT, retrieved=_RETRIEVED)

    assert nr_queries != ra_queries, (
        f"[run {run_number}] retrieval-aware queries should differ from no-retrieval: "
        f"nr={nr_queries!r} ra={ra_queries!r}"
    )

    allowed_text = " ".join(
        [
            _GAP,
            _CONTEXT,
            *[r["content"] for r in _RETRIEVED],
            *[r["questions"] for r in _RETRIEVED],
        ]
    ).lower()
    for q in ra_queries:
        for entity in _named_tokens(q):
            assert entity in allowed_text, (
                f"[run {run_number}] retrieval-aware query {q!r} contains entity "
                f"{entity!r} not present in gap, context, or retrieved entries"
            )

    retrieved_contents = [r["content"].lower() for r in _RETRIEVED]
    for q in ra_queries:
        assert not any(q.lower() in content for content in retrieved_contents), (
            f"[run {run_number}] query {q!r} is a substring of a retrieved fact: "
            f"ra_queries={ra_queries!r} retrieved_contents={retrieved_contents!r}"
        )
