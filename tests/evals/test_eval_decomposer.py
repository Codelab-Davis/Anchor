from __future__ import annotations

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

# Named entities absent from _GAP, _CONTEXT, and _RETRIEVED.
_ABSENT_ENTITIES = ("Falcon-9", "Starlink", "SpaceX", "Tesla", "Crew Dragon")


@pytest.mark.eval
@pytest.mark.parametrize("run_number", range(5))
def test_eval_decomposer_no_retrieval(light_ai_fn, run_number):
    decomposer = Decomposer(model_fn=light_ai_fn)
    queries = decomposer.decompose(_GAP)

    assert len(queries) >= 2, (
        f"[run {run_number}] expected at least 2 queries, got {len(queries)}: {queries!r}"
    )
    assert queries[0] == _GAP, (
        f"[run {run_number}] gap must be first query, got {queries[0]!r}"
    )
    for entity in _ABSENT_ENTITIES:
        for q in queries:
            assert entity.lower() not in q.lower(), (
                f"[run {run_number}] query references absent entity {entity!r}: {q!r}"
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

    for entity in _ABSENT_ENTITIES:
        for q in ra_queries:
            assert entity.lower() not in q.lower(), (
                f"[run {run_number}] retrieval-aware query references absent entity "
                f"{entity!r}: {q!r}"
            )

    retrieved_contents = {r["content"].lower() for r in _RETRIEVED}
    for q in ra_queries:
        assert q.lower() not in retrieved_contents, (
            f"[run {run_number}] query duplicates a retrieved fact verbatim: {q!r}"
        )
