from __future__ import annotations

import pytest

from anchor import Anchor


# Queries that are genuinely underspecified: no entity names, no clear intent,
# no recoverable context.  The model has no choice but to ask for clarification.
UNDERSPECIFIED_QUERIES = [
    "help me with it",
    "what about the other one",
    "fix this",
]


@pytest.fixture
def anchor(ai_fn, light_ai_fn):
    # No memory store — retrieval is irrelevant for intent-clarification evals.
    return Anchor(ai_fn=ai_fn, light_ai_fn=light_ai_fn)


# NOTE: The model may currently fail these tests if the system prompt needs
# fine-tuning to reliably route to CLARIFY (unclear intent) rather than
# REMEMBER (missing entity/fact) for maximally vague inputs.
# The assertions below are correct per spec; the prompt is the variable.
@pytest.mark.eval
@pytest.mark.parametrize("run_number", range(3))
@pytest.mark.parametrize("query", UNDERSPECIFIED_QUERIES)
def test_eval_clarify(anchor, query, run_number):
    result = anchor.run(query)

    assert result.stop_reason == "ask", (
        f"[run {run_number}] query={query!r}: expected stop_reason='ask', "
        f"got {result.stop_reason!r} — content: {result.content!r}"
    )
    assert result.kind == "ask", (
        f"[run {run_number}] query={query!r}: expected kind='ask', "
        f"got {result.kind!r} — content: {result.content!r}"
    )
    assert "?" in result.content, (
        f"[run {run_number}] query={query!r}: expected a clarifying question directed "
        f"at the user, got: {result.content!r}"
    )
    assert result.stop_reason not in ("done", "max_remembers"), (
        f"[run {run_number}] query={query!r}: model must not emit DONE or REMEMBER "
        f"for genuinely underspecified input, got: {result.stop_reason!r}"
    )
