from __future__ import annotations

import pytest

from anchor.document import NormalizedDocument


@pytest.mark.unit
def test_valid_construction_with_all_fields() -> None:
    doc = NormalizedDocument(
        content="Hello world",
        source="docs/readme.md",
        source_format="markdown",
        metadata={"key": "value", "count": 1, "score": 3.14, "active": True},
    )
    assert doc.content == "Hello world"
    assert doc.source == "docs/readme.md"
    assert doc.source_format == "markdown"
    assert doc.metadata == {"key": "value", "count": 1, "score": 3.14, "active": True}


@pytest.mark.unit
def test_valid_construction_with_empty_metadata() -> None:
    doc = NormalizedDocument(
        content="text",
        source="file.txt",
        source_format="text",
    )
    assert doc.metadata == {}


@pytest.mark.unit
def test_metadata_none_raises() -> None:
    with pytest.raises(ValueError, match="'key'"):
        NormalizedDocument(
            content="text",
            source="file.txt",
            source_format="text",
            metadata={"key": None},  # type: ignore[dict-item]
        )


@pytest.mark.unit
def test_metadata_list_raises() -> None:
    with pytest.raises(ValueError, match="'tags'"):
        NormalizedDocument(
            content="text",
            source="file.txt",
            source_format="text",
            metadata={"tags": ["a", "b"]},  # type: ignore[dict-item]
        )


@pytest.mark.unit
def test_metadata_nested_dict_raises() -> None:
    with pytest.raises(ValueError, match="'nested'"):
        NormalizedDocument(
            content="text",
            source="file.txt",
            source_format="text",
            metadata={"nested": {"inner": "value"}},  # type: ignore[dict-item]
        )


@pytest.mark.unit
def test_metadata_none_container_raises() -> None:
    with pytest.raises(ValueError, match="metadata must be a dict"):
        NormalizedDocument(
            content="text",
            source="file.txt",
            source_format="text",
            metadata=None,  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_metadata_non_str_key_raises() -> None:
    with pytest.raises(ValueError, match="metadata keys must be str"):
        NormalizedDocument(
            content="text",
            source="file.txt",
            source_format="text",
            metadata={1: "value"},  # type: ignore[dict-item]
        )


@pytest.mark.unit
def test_metadata_all_valid_scalar_types() -> None:
    doc = NormalizedDocument(
        content="text",
        source="file.txt",
        source_format="text",
        metadata={"s": "hello", "i": 42, "f": 1.5, "b": False},
    )
    assert doc.metadata["s"] == "hello"
    assert doc.metadata["i"] == 42
    assert doc.metadata["f"] == 1.5
    assert doc.metadata["b"] is False
