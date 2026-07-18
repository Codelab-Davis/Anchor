from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from anchor.ingestor import Ingestor
from tests.unit_harness import FakeMemoryStore, scripted_model


class RecordingEmbedFn:
    """Capture the input strings passed to embed_fn.

    Returns a deterministic embedding keyed on call order so each call gets
    a distinguishable vector, which keeps the tests assertion-friendly
    without pulling in a real embedding model.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(self.calls)), float(len(text))]


# ---------------------------------------------------------------------------
# embed_fn precondition
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ingest_raises_when_embed_fn_is_missing() -> None:
    store = FakeMemoryStore()
    ingestor = Ingestor(memory_store=store, embed_fn=None)

    with pytest.raises(RuntimeError, match="embedding function"):
        ingestor.ingest("some content")

    # Nothing should have been written on failure.
    assert store.items == {}


# ---------------------------------------------------------------------------
# ingest without question_fn
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ingest_without_question_fn_uses_raw_text_as_embedding_input() -> None:
    store = FakeMemoryStore()
    embed_fn = RecordingEmbedFn()
    ingestor = Ingestor(memory_store=store, embed_fn=embed_fn, question_fn=None)

    ingestor.ingest("the raw chunk text")

    # embed_fn should be called exactly once with just the raw text -
    # no generated questions appended.
    assert embed_fn.calls == ["the raw chunk text"]


@pytest.mark.unit
def test_ingest_without_question_fn_stores_empty_questions_metadata() -> None:
    store = FakeMemoryStore()
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=None
    )

    chunk_id = ingestor.ingest("a fact")

    record = store.items[chunk_id]
    assert record["metadata"]["questions"] == ""


# ---------------------------------------------------------------------------
# ingest with question_fn
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ingest_with_question_fn_appends_questions_to_embedding_input() -> None:
    store = FakeMemoryStore()
    embed_fn = RecordingEmbedFn()
    question_fn = scripted_model("what is X?\nwho uses X?")
    ingestor = Ingestor(memory_store=store, embed_fn=embed_fn, question_fn=question_fn)

    ingestor.ingest("X is a thing")

    # Embedding input is "<text>\n\n<questions>" when question_fn is present.
    assert embed_fn.calls == ["X is a thing\n\nwhat is X?\nwho uses X?"]


@pytest.mark.unit
def test_ingest_with_question_fn_passes_text_to_question_prompt() -> None:
    store = FakeMemoryStore()
    question_fn = scripted_model("q1?")
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=question_fn
    )

    ingestor.ingest("the chunk body")

    # question_fn is called exactly once with a single user message whose
    # content embeds the chunk text.
    assert len(question_fn.calls) == 1
    messages = question_fn.calls[0]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "the chunk body" in messages[0]["content"]


@pytest.mark.unit
def test_ingest_with_question_fn_stores_questions_in_metadata() -> None:
    store = FakeMemoryStore()
    question_fn = scripted_model("what is X?\nwho uses X?")
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=question_fn
    )

    chunk_id = ingestor.ingest("X is a thing")

    questions = store.items[chunk_id]["metadata"]["questions"]
    assert questions == "what is X?\nwho uses X?"


# ---------------------------------------------------------------------------
# memory_store.add contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ingest_writes_expected_record_fields_to_memory_store() -> None:
    store = FakeMemoryStore()

    def embed_fn(_text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    ingestor = Ingestor(memory_store=store, embed_fn=embed_fn, question_fn=None)

    chunk_id = ingestor.ingest("hello world", source="docs")

    # Exactly one record was written, keyed by the returned chunk id.
    assert list(store.items.keys()) == [chunk_id]

    record = store.items[chunk_id]
    # FakeMemoryStore.add stores the text it received under "content", so a
    # non-empty "content" confirms the text kwarg was forwarded. The id and
    # embedding are preserved under their own keys.
    assert record["id"] == chunk_id
    assert record["content"] == "hello world"
    assert record["embedding"] == [0.1, 0.2, 0.3]
    # Metadata is present as a dict (detailed contents are asserted in the
    # metadata-focused tests below).
    assert isinstance(record["metadata"], dict)


@pytest.mark.unit
def test_ingest_returns_uuid_chunk_id_matching_stored_record() -> None:
    store = FakeMemoryStore()
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=None
    )

    chunk_id = ingestor.ingest("data")

    # The returned id must be a valid UUID string and must match the key
    # under which the record was stored.
    parsed = uuid.UUID(chunk_id)
    assert str(parsed) == chunk_id
    assert chunk_id in store.items


# ---------------------------------------------------------------------------
# metadata contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ingest_uses_default_source_when_not_provided() -> None:
    store = FakeMemoryStore()
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=None
    )

    chunk_id = ingestor.ingest("data")

    assert store.items[chunk_id]["metadata"]["source"] == "user"


@pytest.mark.unit
def test_ingest_passes_through_custom_source() -> None:
    store = FakeMemoryStore()
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=None
    )

    chunk_id = ingestor.ingest("data", source="docs")

    assert store.items[chunk_id]["metadata"]["source"] == "docs"


@pytest.mark.unit
def test_ingest_metadata_contains_required_keys() -> None:
    store = FakeMemoryStore()
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=None
    )

    chunk_id = ingestor.ingest("data", source="docs")

    metadata = store.items[chunk_id]["metadata"]
    assert set(metadata.keys()) >= {"source", "questions", "timestamp"}


@pytest.mark.unit
def test_ingest_timestamp_is_iso_format_and_utc() -> None:
    store = FakeMemoryStore()
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=None
    )

    before = datetime.now(timezone.utc)
    chunk_id = ingestor.ingest("data")
    after = datetime.now(timezone.utc)

    timestamp_str = store.items[chunk_id]["metadata"]["timestamp"]
    # Must be a parseable ISO 8601 string.
    parsed = datetime.fromisoformat(timestamp_str)
    # Must be timezone-aware and in UTC.
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)
    # Must fall within the window during which ingest() ran.
    assert before <= parsed <= after


# ---------------------------------------------------------------------------
# metadata passthrough (#75)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ingest_stores_caller_metadata_alongside_required_fields() -> None:
    store = FakeMemoryStore()
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=None
    )

    chunk_id = ingestor.ingest(
        "data", metadata={"author": "alice", "priority": 1, "verified": True}
    )

    metadata = store.items[chunk_id]["metadata"]
    assert metadata["author"] == "alice"
    assert metadata["priority"] == 1
    assert metadata["verified"] is True
    # Required fields are still present.
    assert set(metadata.keys()) >= {
        "source",
        "questions",
        "timestamp",
        "author",
        "priority",
        "verified",
    }


@pytest.mark.unit
def test_ingest_caller_metadata_does_not_override_required_fields() -> None:
    store = FakeMemoryStore()
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=None
    )

    chunk_id = ingestor.ingest(
        "data",
        source="docs",
        metadata={
            "source": "attacker-controlled",
            "questions": "attacker-controlled",
            "timestamp": "attacker-controlled",
        },
    )

    metadata = store.items[chunk_id]["metadata"]
    assert metadata["source"] == "docs"
    assert metadata["questions"] == ""
    assert metadata["timestamp"] != "attacker-controlled"
    parsed = datetime.fromisoformat(metadata["timestamp"])
    assert parsed.tzinfo is not None


@pytest.mark.unit
def test_ingest_metadata_none_behaves_like_no_metadata() -> None:
    store_a = FakeMemoryStore()
    store_b = FakeMemoryStore()
    ingestor_a = Ingestor(
        memory_store=store_a, embed_fn=RecordingEmbedFn(), question_fn=None
    )
    ingestor_b = Ingestor(
        memory_store=store_b, embed_fn=RecordingEmbedFn(), question_fn=None
    )

    chunk_id_a = ingestor_a.ingest("data", source="docs")
    chunk_id_b = ingestor_b.ingest("data", source="docs", metadata=None)

    metadata_a = store_a.items[chunk_id_a]["metadata"]
    metadata_b = store_b.items[chunk_id_b]["metadata"]
    assert set(metadata_a.keys()) == set(metadata_b.keys())
    assert metadata_a["source"] == metadata_b["source"]
    assert metadata_a["questions"] == metadata_b["questions"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_metadata",
    [
        {"note": None},
        {"tags": ["a", "b"]},
        {"nested": {"a": 1}},
    ],
)
def test_ingest_raises_value_error_for_invalid_metadata_value(
    invalid_metadata: dict,
) -> None:
    store = FakeMemoryStore()
    ingestor = Ingestor(
        memory_store=store, embed_fn=RecordingEmbedFn(), question_fn=None
    )

    with pytest.raises(ValueError, match=list(invalid_metadata.keys())[0]):
        ingestor.ingest("data", metadata=invalid_metadata)

    # Nothing should have been written on failure.
    assert store.items == {}
