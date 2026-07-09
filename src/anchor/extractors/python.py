from __future__ import annotations

import ast
from pathlib import Path

from anchor.extractors import NormalizedDocument


# Design choice: mirrors the Markdown extractor's "one document per section"
# approach, using top-level def/class boundaries as the section markers
# instead of headings. Each top-level function or class becomes its own
# document carrying symbol metadata. Runs of other top-level statements
# (module docstring, imports, constants, top-level calls/conditionals) are
# grouped into "module" documents in the order they appear in the file, so
# module-scope content is never silently dropped. Only the standard library
# `ast` module is used, so this stays dependency-free and handles any
# syntactically valid Python file without attempting multi-language or
# semantic parsing.


def _line_offsets(source: str) -> list[int]:
    """offsets[i] is the character offset of the start of line i + 1 (1-indexed lines)."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _decorated_start_line(node: ast.stmt) -> int:
    """Return the first line of *node*, including any decorators."""
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        return min(decorator.lineno for decorator in decorators)
    return node.lineno


class PythonExtractor:
    """Extracts NormalizedDocuments from a Python source file using `ast`.

    Raises ``ValueError`` (chained from the original ``SyntaxError``) if the
    file is not valid Python, so callers get a single, catchable exception
    type instead of an unhandled parser error.
    """

    def extract(self, path: str | Path) -> list[NormalizedDocument]:
        path = Path(path)
        source = path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as err:
            raise ValueError(
                f"Cannot parse {str(path)!r} as Python: {err.msg} (line {err.lineno})"
            ) from err

        offsets = _line_offsets(source)
        lines = source.splitlines(keepends=True)
        total_lines = len(lines)

        def build_doc(
            start_line: int, end_line: int, extra_metadata: dict
        ) -> NormalizedDocument | None:
            start_line = max(1, start_line)
            end_line = min(total_lines, end_line)
            if start_line > end_line:
                return None

            content = "".join(lines[start_line - 1 : end_line])
            if not content.strip():
                return None

            metadata: dict[str, str | int | float | bool] = {
                "line_start": start_line,
                "line_end": end_line,
                "char_start": offsets[start_line - 1],
                "char_end": offsets[end_line],
            }
            metadata.update(extra_metadata)

            return NormalizedDocument(
                content=content,
                source=str(path),
                source_format="python",
                metadata=metadata,
            )

        docs: list[NormalizedDocument] = []
        pending_start: int | None = None
        pending_end: int | None = None

        def flush_pending() -> None:
            nonlocal pending_start, pending_end
            if pending_start is not None and pending_end is not None:
                doc = build_doc(pending_start, pending_end, {"symbol_type": "module"})
                if doc is not None:
                    docs.append(doc)
            pending_start = None
            pending_end = None

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                flush_pending()

                start_line = _decorated_start_line(node)
                end_line = (
                    node.end_lineno if node.end_lineno is not None else node.lineno
                )

                if isinstance(node, ast.ClassDef):
                    symbol_type = "class"
                elif isinstance(node, ast.AsyncFunctionDef):
                    symbol_type = "async_function"
                else:
                    symbol_type = "function"

                doc = build_doc(
                    start_line,
                    end_line,
                    {"symbol_type": symbol_type, "symbol_name": node.name},
                )
                if doc is not None:
                    docs.append(doc)
            else:
                start_line = node.lineno
                end_line = (
                    node.end_lineno if node.end_lineno is not None else node.lineno
                )
                if pending_start is None:
                    pending_start = start_line
                pending_end = end_line

        flush_pending()

        return docs