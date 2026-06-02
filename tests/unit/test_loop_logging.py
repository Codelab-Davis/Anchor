from __future__ import annotations

import json

import pytest

from anchor.anchor import Anchor
from tests.unit_harness import FakeMemoryStore, scripted_model


def _make_anchor(
    ai_responses: list[str],
    log_path=None,
    light_ai_responses: list[str] | None = None,
) -> Anchor:
    return Anchor(
        ai_fn=scripted_model(*ai_responses),
        light_ai_fn=scripted_model(*(light_ai_responses or [])),
        log_path=log_path,
    )


def _events(log_file) -> list[dict]:
    return [json.loads(line) for line in log_file.read_text().splitlines()]


@pytest.mark.unit
def test_logging_done_path(tmp_path) -> None:
    log_file = tmp_path / "run.jsonl"
    anchor = _make_anchor(
        ai_responses=["The answer is 42.\nDONE"],
        log_path=log_file,
    )
    result = anchor.run("What is the answer?")

    assert result.kind == "done"
    assert result.stop_reason == "done"

    events = _events(log_file)
    assert events[0] == {"event": "run_start", "query": "What is the answer?"}
    assert events[-1] == {"event": "stop", "stop_reason": "done"}
    # No proactive events — no retriever configured.
    event_names = [e["event"] for e in events]
    assert event_names == ["run_start", "stop"]


@pytest.mark.unit
def test_logging_proactive_retrieval_path(tmp_path) -> None:
    log_file = tmp_path / "run.jsonl"
    chunk = {"id": "c1", "content": "relevant fact", "source": "wiki", "score": 0.9}
    store = FakeMemoryStore(query_results=[[chunk]])

    anchor = Anchor(
        ai_fn=scripted_model("The answer.\nDONE"),
        light_ai_fn=scripted_model("Tell me about X."),
        memory_store=store,
        embed_fn=lambda _text: [0.0],
        log_path=log_file,
    )
    result = anchor.run("Tell me about X.")

    assert result.kind == "done"
    assert result.stop_reason == "done"

    events = _events(log_file)
    event_names = [e["event"] for e in events]
    assert event_names == [
        "run_start",
        "proactive_queries",
        "proactive_chunks",
        "stop",
    ]

    assert events[0]["query"] == "Tell me about X."
    assert events[1]["queries"] == ["Tell me about X."]
    assert events[2]["chunks"] == [{"id": "c1", "source": "wiki", "score": 0.9}]
    assert events[3]["stop_reason"] == "done"


@pytest.mark.unit
def test_logging_remember_cycle(tmp_path) -> None:
    log_file = tmp_path / "run.jsonl"
    anchor = _make_anchor(
        ai_responses=[
            "GAP: what is X?\nCONTEXT: figuring out X\nREMEMBER",
            "The answer is X.\nDONE",
        ],
        light_ai_responses=["query about X"],
        log_path=log_file,
    )
    result = anchor.run("Tell me about X.")

    assert result.kind == "done"
    assert result.stop_reason == "done"

    events = _events(log_file)
    event_names = [e["event"] for e in events]
    assert event_names == [
        "run_start",
        "remember_gap",
        "remember_queries",
        "remember_chunks",
        "stop",
    ]

    assert events[0]["query"] == "Tell me about X."
    assert events[1] == {
        "event": "remember_gap",
        "gap": "what is X?",
        "context": "figuring out X",
    }
    assert events[2]["queries"] == ["what is X?", "query about X"]
    assert events[3] == {"event": "remember_chunks", "chunks": []}
    assert events[4]["stop_reason"] == "done"


@pytest.mark.unit
def test_logging_max_remembers_path(tmp_path) -> None:
    log_file = tmp_path / "run.jsonl"
    anchor = _make_anchor(
        ai_responses=[
            "GAP: what is X?\nCONTEXT: first attempt\nREMEMBER",
            "GAP: still missing\nCONTEXT: second attempt\nREMEMBER",
        ],
        light_ai_responses=["query about X"],  # only cycle 1 calls decompose
        log_path=log_file,
    )
    anchor.MAX_REMEMBERS = 1
    result = anchor.run("Tell me about X.")

    assert result.stop_reason == "max_remembers"

    events = _events(log_file)
    event_names = [e["event"] for e in events]
    assert event_names == [
        "run_start",
        "remember_gap",
        "remember_queries",
        "remember_chunks",
        "stop",
    ]
    assert events[-1] == {"event": "stop", "stop_reason": "max_remembers"}


@pytest.mark.unit
def test_logging_ask_path(tmp_path) -> None:
    log_file = tmp_path / "run.jsonl"
    anchor = _make_anchor(
        ai_responses=["QUESTION: Which format do you want?\nCLARIFY"],
        log_path=log_file,
    )
    result = anchor.run("Process the data.")

    assert result.stop_reason == "ask"

    events = _events(log_file)
    event_names = [e["event"] for e in events]
    assert event_names == ["run_start", "stop"]
    assert events[-1] == {"event": "stop", "stop_reason": "ask"}


@pytest.mark.unit
def test_no_file_written_when_log_path_none(tmp_path) -> None:
    anchor = _make_anchor(["The answer is 42.\nDONE"])
    result = anchor.run("What is the answer?")

    assert result.kind == "done"
    assert result.stop_reason == "done"
    # No log_path set — tmp_path must remain empty.
    assert not any(tmp_path.iterdir())
