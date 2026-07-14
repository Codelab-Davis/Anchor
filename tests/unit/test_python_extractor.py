from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from anchor.extractors import NormalizedDocument
from anchor.extractors.python import PythonExtractor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_extract(tmp_path: Path, content: str) -> tuple[list[NormalizedDocument], Path]:
    py_file = tmp_path / "sample.py"
    py_file.write_text(textwrap.dedent(content), encoding="utf-8")
    return PythonExtractor().extract(py_file), py_file


# ---------------------------------------------------------------------------
# module with a function and a class
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_top_level_function_and_class_produce_separate_docs(tmp_path: Path) -> None:
    docs, _ = run_extract(
        tmp_path,
        """\
        def add(a, b):
            return a + b


        class Calculator:
            def multiply(self, a, b):
                return a * b
        """,
    )
    symbol_names = [doc.metadata.get("symbol_name") for doc in docs]
    assert "add" in symbol_names
    assert "Calculator" in symbol_names


@pytest.mark.unit
def test_function_metadata_has_function_symbol_type(tmp_path: Path) -> None:
    docs, _ = run_extract(
        tmp_path,
        """\
        def add(a, b):
            return a + b
        """,
    )
    assert len(docs) == 1
    assert docs[0].metadata["symbol_type"] == "function"
    assert docs[0].metadata["symbol_name"] == "add"


@pytest.mark.unit
def test_async_function_has_async_function_symbol_type(tmp_path: Path) -> None:
    docs, _ = run_extract(
        tmp_path,
        """\
        async def fetch(x):
            return x
        """,
    )
    assert len(docs) == 1
    assert docs[0].metadata["symbol_type"] == "async_function"
    assert docs[0].metadata["symbol_name"] == "fetch"


@pytest.mark.unit
def test_class_metadata_has_class_symbol_type(tmp_path: Path) -> None:
    docs, _ = run_extract(
        tmp_path,
        """\
        class Calculator:
            def multiply(self, a, b):
                return a * b
        """,
    )
    assert len(docs) == 1
    assert docs[0].metadata["symbol_type"] == "class"
    assert docs[0].metadata["symbol_name"] == "Calculator"


@pytest.mark.unit
def test_class_document_includes_nested_methods(tmp_path: Path) -> None:
    docs, _ = run_extract(
        tmp_path,
        """\
        class Calculator:
            def multiply(self, a, b):
                return a * b
        """,
    )
    assert "def multiply" in docs[0].content


@pytest.mark.unit
def test_decorator_is_included_in_function_content(tmp_path: Path) -> None:
    docs, _ = run_extract(
        tmp_path,
        """\
        import functools


        @functools.cache
        def cached(x):
            return x * 2
        """,
    )
    function_docs = [d for d in docs if d.metadata.get("symbol_type") == "function"]
    assert len(function_docs) == 1
    assert "@functools.cache" in function_docs[0].content


# ---------------------------------------------------------------------------
# module-level content (docstring, imports, top-level statements)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_module_docstring_and_imports_produce_module_doc(tmp_path: Path) -> None:
    docs, _ = run_extract(
        tmp_path,
        """\
        \"\"\"Module docstring.\"\"\"

        import os

        VERSION = "1.0.0"
        """,
    )
    assert len(docs) == 1
    assert docs[0].metadata["symbol_type"] == "module"
    assert "Module docstring." in docs[0].content
    assert "import os" in docs[0].content
    assert 'VERSION = "1.0.0"' in docs[0].content


@pytest.mark.unit
def test_module_statements_interleaved_with_function(tmp_path: Path) -> None:
    docs, _ = run_extract(
        tmp_path,
        """\
        import os

        def greet(name):
            return f"Hello, {name}!"

        print(os.getcwd())
        """,
    )
    symbol_types = [doc.metadata["symbol_type"] for doc in docs]
    assert symbol_types == ["module", "function", "module"]
    assert "import os" in docs[0].content
    assert "print(os.getcwd())" in docs[2].content


@pytest.mark.unit
def test_module_doc_has_no_symbol_name(tmp_path: Path) -> None:
    docs, _ = run_extract(tmp_path, "import os\n")
    assert len(docs) == 1
    assert "symbol_name" not in docs[0].metadata


# ---------------------------------------------------------------------------
# common document fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_source_format_is_python(tmp_path: Path) -> None:
    docs, _ = run_extract(tmp_path, "def add(a, b):\n    return a + b\n")
    assert all(doc.source_format == "python" for doc in docs)


@pytest.mark.unit
def test_source_is_file_path(tmp_path: Path) -> None:
    docs, py_file = run_extract(tmp_path, "def add(a, b):\n    return a + b\n")
    assert all(doc.source == str(py_file) for doc in docs)


@pytest.mark.unit
def test_line_range_matches_function_definition(tmp_path: Path) -> None:
    docs, _ = run_extract(
        tmp_path,
        """\
        def add(a, b):
            return a + b
        """,
    )
    assert docs[0].metadata["line_start"] == 1
    assert docs[0].metadata["line_end"] == 2


@pytest.mark.unit
def test_char_offsets_slice_back_to_original_content(tmp_path: Path) -> None:
    content = textwrap.dedent(
        """\
        import os

        def add(a, b):
            return a + b
        """
    )
    docs, _ = run_extract(tmp_path, content)
    for doc in docs:
        start = doc.metadata["char_start"]
        end = doc.metadata["char_end"]
        assert content[start:end] == doc.content


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_file_produces_no_docs(tmp_path: Path) -> None:
    docs, _ = run_extract(tmp_path, "")
    assert docs == []


@pytest.mark.unit
def test_malformed_python_raises_value_error(tmp_path: Path) -> None:
    py_file = tmp_path / "broken.py"
    py_file.write_text("def broken(:\n    pass\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        PythonExtractor().extract(py_file)

    assert str(py_file) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, SyntaxError)


@pytest.mark.unit
def test_multiple_top_level_functions_each_get_own_doc(tmp_path: Path) -> None:
    docs, _ = run_extract(
        tmp_path,
        """\
        def add(a, b):
            return a + b


        def subtract(a, b):
            return a - b
        """,
    )
    function_docs = [d for d in docs if d.metadata.get("symbol_type") == "function"]
    assert [d.metadata["symbol_name"] for d in function_docs] == ["add", "subtract"]