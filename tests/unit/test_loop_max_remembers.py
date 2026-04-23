from __future__ import annotations

import pytest

from anchor.anchor import Anchor
from tests.unit_harness import scripted_model

# The REMEMBER protocol block the agent emits when it wants another retrieval
# cycle. Pre-built here so each test stays readable and we can assert on the
# exact content returned when the guard strips the terminal marker.
REMEMBER_BODY = "GAP: still missing a fact\nCONTEXT: reasoning incomplete"
REMEMBER_RESPONSE = f"{REMEMBER_BODY}\nREMEMBER"


def _make_anchor(
    ai_responses: list[str],
    light_ai_responses: list[str] | None = None,
) -> Anchor:
    return Anchor(
        ai_fn=scripted_model(*ai_responses),
        light_ai_fn=scripted_model(*(light_ai_responses or [])),
    )


# ---------------------------------------------------------------------------
# guard fires and returns the documented shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_guard_trips_when_remember_count_exceeds_limit() -> None:
    # With MAX_REMEMBERS=2, the guard should fire on the 3rd REMEMBER response
    # (remembers counter is incremented to 3 and then 3 > 2 returns True).
    anchor = _make_anchor(
        ai_responses=[REMEMBER_RESPONSE, REMEMBER_RESPONSE, REMEMBER_RESPONSE],
        light_ai_responses=["q1", "q2"],
    )
    anchor.MAX_REMEMBERS = 2

    result = anchor.run("question that never reaches DONE")

    assert result.kind == "done"
    assert result.stop_reason == "max_remembers"


@pytest.mark.unit
def test_guard_returns_content_without_trailing_remember_marker() -> None:
    # When the guard fires, the final content returned to the caller should
    # have the terminal REMEMBER stripped so users never see protocol tokens.
    anchor = _make_anchor(
        ai_responses=[REMEMBER_RESPONSE, REMEMBER_RESPONSE],
        light_ai_responses=["q1"],
    )
    anchor.MAX_REMEMBERS = 1

    result = anchor.run("question")

    assert "REMEMBER" not in result.content.splitlines()[-1]
    assert result.content == REMEMBER_BODY


# ---------------------------------------------------------------------------
# the guard is precise: it fires on N+1, not on N
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_guard_does_not_trip_on_exactly_max_remembers_when_final_response_is_done() -> (
    None
):
    # MAX_REMEMBERS=2 means two REMEMBER cycles are allowed. A subsequent DONE
    # on the third turn should resolve normally - the guard must NOT fire at
    # remembers == max_remembers.
    anchor = _make_anchor(
        ai_responses=[REMEMBER_RESPONSE, REMEMBER_RESPONSE, "final answer\nDONE"],
        light_ai_responses=["q1", "q2"],
    )
    anchor.MAX_REMEMBERS = 2

    result = anchor.run("question")

    assert result.kind == "done"
    assert result.stop_reason == "done"
    assert result.content == "final answer"


@pytest.mark.unit
@pytest.mark.parametrize("max_remembers", [1, 2, 3, 5])
def test_guard_trips_exactly_at_n_plus_one_remembers(max_remembers: int) -> None:
    # For any MAX_REMEMBERS=N, queuing N+1 REMEMBER responses should end in
    # max_remembers. Queue one extra DONE after the limit to prove the loop
    # actually terminates on the guard and never consumes the DONE.
    ai_responses = [REMEMBER_RESPONSE] * (max_remembers + 1) + ["never used\nDONE"]
    light_ai_responses = ["decomposed"] * max_remembers

    anchor = _make_anchor(
        ai_responses=ai_responses, light_ai_responses=light_ai_responses
    )
    anchor.MAX_REMEMBERS = max_remembers

    result = anchor.run("question")

    assert result.stop_reason == "max_remembers"


# ---------------------------------------------------------------------------
# custom MAX_REMEMBERS on the instance and on a subclass both work
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_guard_respects_instance_level_max_remembers_override() -> None:
    # Mutating MAX_REMEMBERS on the instance should override the class
    # default, since the loop does a plain getattr() lookup.
    anchor = _make_anchor(
        ai_responses=[REMEMBER_RESPONSE, REMEMBER_RESPONSE],
        light_ai_responses=["q1"],
    )
    anchor.MAX_REMEMBERS = 1

    result = anchor.run("question")

    assert result.stop_reason == "max_remembers"


@pytest.mark.unit
def test_guard_respects_subclass_level_max_remembers_override() -> None:
    # Overriding the class attribute on a subclass must also be honored, so
    # library users can configure the limit declaratively without mutating
    # individual instances.
    class TightAnchor(Anchor):
        MAX_REMEMBERS = 1

    anchor = TightAnchor(
        ai_fn=scripted_model(REMEMBER_RESPONSE, REMEMBER_RESPONSE),
        light_ai_fn=scripted_model("q1"),
    )

    result = anchor.run("question")

    assert result.stop_reason == "max_remembers"
