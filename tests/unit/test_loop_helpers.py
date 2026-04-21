from __future__ import annotations

import pytest

from anchor.loop import Loop


@pytest.fixture
def loop() -> Loop:
    # Helpers under test are pure and do not touch the Anchor instance,
    # so we pass None to keep the tests free of model/embedding wiring.
    return Loop(anchor=None)


# ---------------------------------------------------------------------------
# _extract_gap
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_gap_standard_input(loop: Loop) -> None:
    content = "GAP: what is X?\nCONTEXT: figuring out X\nREMEMBER"
    gap, context = loop._extract_gap(content)
    assert gap == "what is X?"
    assert context == "figuring out X"


@pytest.mark.unit
def test_extract_gap_missing_gap(loop: Loop) -> None:
    content = "CONTEXT: still working through it\nREMEMBER"
    gap, context = loop._extract_gap(content)
    assert gap == ""
    assert context == "still working through it"


@pytest.mark.unit
def test_extract_gap_missing_context(loop: Loop) -> None:
    content = "GAP: what is Y?\nREMEMBER"
    gap, context = loop._extract_gap(content)
    assert gap == "what is Y?"
    assert context == ""


@pytest.mark.unit
def test_extract_gap_both_missing(loop: Loop) -> None:
    content = "Some unrelated reasoning text.\nNo markers here.\nDONE"
    gap, context = loop._extract_gap(content)
    assert gap == ""
    assert context == ""


@pytest.mark.unit
def test_extract_gap_handles_extra_whitespace_and_blank_lines(loop: Loop) -> None:
    content = (
        "\n   GAP:   what is Z?   \n\n   CONTEXT:\treasoning about Z\t\n\nREMEMBER\n"
    )
    gap, context = loop._extract_gap(content)
    assert gap == "what is Z?"
    assert context == "reasoning about Z"


# ---------------------------------------------------------------------------
# _strip_marker
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strip_marker_removes_trailing_marker(loop: Loop) -> None:
    content = "The answer is 42.\nDONE"
    assert loop._strip_marker(content, "DONE") == "The answer is 42."


@pytest.mark.unit
def test_strip_marker_cleans_trailing_blank_lines_before_marker(loop: Loop) -> None:
    content = "The answer is 42.\n\n\nDONE\n\n"
    assert loop._strip_marker(content, "DONE") == "The answer is 42."


@pytest.mark.unit
def test_strip_marker_preserves_content_without_marker(loop: Loop) -> None:
    content = "The answer is 42.\nNo marker here."
    assert loop._strip_marker(content, "DONE") == "The answer is 42.\nNo marker here."


@pytest.mark.unit
def test_strip_marker_only_strips_final_marker_occurrence(loop: Loop) -> None:
    # The marker appears twice: once embedded in prose and once as the true
    # terminal marker. Only the terminal one should be removed.
    content = "I initially wrote DONE by mistake.\nHere's the real answer.\nDONE"
    expected = "I initially wrote DONE by mistake.\nHere's the real answer."
    assert loop._strip_marker(content, "DONE") == expected


@pytest.mark.unit
def test_strip_marker_does_not_strip_when_marker_is_substring_of_last_line(
    loop: Loop,
) -> None:
    # The last line contains the marker text but is not *exactly* the marker,
    # so nothing should be stripped.
    content = "Final thought.\nI am DONE reasoning now."
    assert loop._strip_marker(content, "DONE") == content


@pytest.mark.unit
def test_strip_marker_handles_marker_with_surrounding_whitespace(loop: Loop) -> None:
    # The final line is "   DONE   ", which after stripping equals the marker,
    # so it should still be removed.
    content = "The answer is 42.\n   DONE   "
    assert loop._strip_marker(content, "DONE") == "The answer is 42."
