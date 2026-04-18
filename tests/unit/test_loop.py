from __future__ import annotations

import pytest

from anchor.anchor import Anchor
from tests.unit_harness import scripted_model


def _make_anchor(
    ai_responses: list[str],
    light_ai_responses: list[str] | None = None,
) -> Anchor:
    return Anchor(
        ai_fn=scripted_model(*ai_responses),
        light_ai_fn=scripted_model(*(light_ai_responses or [])),
    )


@pytest.mark.unit
def test_done_on_first_call() -> None:
    anchor = _make_anchor(["The answer is 42.\nDONE"])
    result = anchor.run("What is the answer?")
    assert result.kind == "done"
    assert result.stop_reason == "done"
    assert result.content == "The answer is 42."


@pytest.mark.unit
def test_single_remember_then_done() -> None:
    anchor = _make_anchor(
        ai_responses=[
            "GAP: what is X?\nCONTEXT: figuring out X\nREMEMBER",
            "The answer is X.\nDONE",
        ],
        light_ai_responses=["query about X"],
    )
    result = anchor.run("Tell me about X.")
    assert result.kind == "done"
    assert result.stop_reason == "done"
    assert result.content == "The answer is X."


@pytest.mark.unit
def test_two_remembers_then_done() -> None:
    anchor = _make_anchor(
        ai_responses=[
            "GAP: what is X?\nCONTEXT: figuring out X\nREMEMBER",
            "GAP: what is Y?\nCONTEXT: still working through it\nREMEMBER",
            "The final answer.\nDONE",
        ],
        light_ai_responses=["query about X", "query about Y"],
    )
    result = anchor.run("Tell me about X and Y.")
    assert result.kind == "done"
    assert result.stop_reason == "done"
    assert result.content == "The final answer."


@pytest.mark.unit
def test_clarify_path() -> None:
    anchor = _make_anchor(["QUESTION: Which format do you want?\nCLARIFY"])
    result = anchor.run("Process the data.")
    assert result.kind == "ask"
    assert result.stop_reason == "ask"
    assert "Which format do you want?" in result.content


@pytest.mark.unit
def test_no_marker_error_path() -> None:
    # no protocol marker => loop exits with stop_reason="error"; kind is still "done"
    anchor = _make_anchor(["Just some text with no protocol marker."])
    result = anchor.run("What is this?")
    assert result.kind == "done"
    assert result.stop_reason == "error"


@pytest.mark.unit
def test_max_remembers_guard() -> None:
    # max_remembers=2 => the 3rd REMEMBER triggers exit (remembers=3 > 2)
    anchor = _make_anchor(
        ai_responses=[
            "GAP: gap1\nCONTEXT: ctx1\nREMEMBER",
            "GAP: gap2\nCONTEXT: ctx2\nREMEMBER",
            "GAP: gap3\nCONTEXT: ctx3\nREMEMBER",
        ],
        light_ai_responses=["query1", "query2"],
    )
    anchor.MAX_REMEMBERS = 2
    result = anchor.run("Something that requires many lookups.")
    assert result.kind == "done"
    assert result.stop_reason == "max_remembers"
