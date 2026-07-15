# Chunk Metadata Schema

This document describes the proposed metadata schema for document-derived memory chunks stored in Anchor's vector store. Metadata is attached at ingest time via `Ingestor.ingest` and forwarded to `MemoryStore.add`. It travels with the chunk through retrieval and appears in `retrieved_items` returned by `Loop.run`.

The schema covers chunks produced from documents (Markdown files, PDFs, plain text). It does not cover agent write-back memory or code-derived chunks, which carry different structural context.

## Required fields

These fields must be present on every document-derived chunk. Omitting any of them makes retrieval debugging impractical and breaks callers that rely on them.

| Field | Type | Description |
| --- | --- | --- |
| `source` | `str` | Origin of the content — a file path, URL, or other stable identifier. |
| `source_format` | `str` | Format identifier for the source: `"markdown"`, `"pdf"`, `"text"`, etc. |
| `chunk_index` | `int` | Zero-based position of this chunk within the source document. |
| `char_start` | `int` | Character offset of the chunk's first character in the extracted source text. |
| `char_end` | `int` | Character offset one past the chunk's last character in the extracted source text. |
| `timestamp` | `str` | ISO 8601 UTC timestamp recording when the chunk was ingested. |
| `questions` | `str` | Newline-separated retrieval questions generated at ingest time by the light model. Empty string if question generation was skipped. |

## Optional fields

These fields are included when the source format and extraction pipeline can supply them. Omit a field entirely rather than storing `None` or a sentinel value — Chroma does not accept `None` in metadata.

| Field | Type | Description |
| --- | --- | --- |
| `page_number` | `int` | Page number where the chunk appears. Meaningful for PDF and paginated formats. |
| `heading` | `str` | Nearest heading above the chunk in the document structure. |
| `section` | `str` | Full section path, for example `"Chapter 2 > Installation"`. |
| `line_start` | `int` | One-based line number of the chunk's first line in the extracted source text. |
| `line_end` | `int` | One-based line number of the chunk's last line in the extracted source text. |
| `chunk_count` | `int` | Total number of chunks produced from this source document in the same ingest run. |

## Source path convention

`source` is populated differently depending on the ingestion path.

**File ingestion.** `Anchor.ingest_file`, `Anchor.ingest_directory`, and the extractors (`TextExtractor`, `MarkdownExtractor`, `PythonExtractor`) set `source` to the absolute path of the file, computed as `str(Path(path).resolve())`. This holds regardless of whether the caller passed a relative or absolute path, and regardless of the working directory at ingest time, so chunks can always be traced back to their source file consistently.

**Low-level text ingestion.** `Anchor.ingest_text` takes `source` as a caller-provided string with no convention enforced. It can be a file path, a URL, a document title, or any other stable identifier — whatever makes sense for the caller's use case.

These two conventions differ intentionally. File ingestion needs a machine-stable reference that resolves the same way no matter where the ingest was run from. Text ingestion is a lower-level entry point where the caller may not have a filesystem path at all, so it needs the flexibility to supply its own identifier.

This convention applies to newly ingested content only. Existing stored chunks are not migrated and may still contain relative paths or other pre-convention `source` values.

## Examples

### Markdown

```python
{
    "source": "docs/setup.md",
    "source_format": "markdown",
    "chunk_index": 2,
    "char_start": 412,
    "char_end": 789,
    "timestamp": "2026-06-04T09:15:00+00:00",
    "questions": "How do I install Anchor?\nWhat Python version is required?",
    "heading": "## Installation",
    "section": "Setup > Installation",
    "line_start": 34,
    "line_end": 51,
    "chunk_count": 8,
}
```

### PDF

```python
{
    "source": "reports/q1-summary.pdf",
    "source_format": "pdf",
    "chunk_index": 5,
    "char_start": 1024,
    "char_end": 1480,
    "timestamp": "2026-06-04T09:15:00+00:00",
    "questions": "What were the Q1 revenue figures?\nWhich regions exceeded targets?",
    "page_number": 3,
    "heading": "Regional Performance",
    "chunk_count": 22,
}
```

`line_start` and `line_end` are omitted because line numbers are not meaningful in most PDF extraction outputs.

### Plain text

```python
{
    "source": "notes/meeting-2026-05-30.txt",
    "source_format": "text",
    "chunk_index": 0,
    "char_start": 0,
    "char_end": 512,
    "timestamp": "2026-06-04T09:15:00+00:00",
    "questions": "What was discussed in the May 30 meeting?\nWhat action items were assigned?",
    "line_start": 1,
    "line_end": 28,
    "chunk_count": 3,
}
```

`page_number`, `heading`, and `section` are omitted because plain text files have no page or heading structure.

## Chroma compatibility

ChromaDB metadata values must be one of `str`, `int`, `float`, or `bool`. Two constraints follow from this.

**No `None` values.** Omit a field entirely rather than setting it to `None`. Storing `None` raises a Chroma error at ingest time. Callers that read optional fields back from retrieved chunks must use `.get("heading")` rather than direct key access.

**No nested structures.** The `questions` field is stored as a newline-delimited string rather than a list. Callers that need individual questions split on `"\n"` after retrieval:

```python
questions = chunk["metadata"].get("questions", "")
question_list = [q for q in questions.splitlines() if q.strip()]
```

## Formats without pages or headings

Plain text, CSV, and some Markdown files have no page structure and may have no headings. For these sources, omit `page_number`, `heading`, and `section` from the metadata dict entirely. Do not substitute sentinel values like `-1` or `""` — an absent key is unambiguous, while `""` is indistinguishable from a heading that was present but empty.

For Markdown files where a chunk falls before the first heading, omit `heading`. For Markdown files where the heading hierarchy is flat (no section nesting), omit `section` and include only `heading`.

## Retrieval debugging fields

Several fields exist primarily to support post-retrieval inspection rather than retrieval itself.

- `chunk_index`, `char_start`, and `char_end` let callers locate the exact passage in the source document that produced a retrieved chunk.
- `line_start` and `line_end` are useful for text formats where an editor or diff view is the natural debugging interface.
- `chunk_count` lets callers determine whether the full document was ingested in the same run or only a subset.
- `source_format` enables format-specific rendering or post-processing of retrieved content.

These fields are not used by the retriever or synthesizer today. They are included in `retrieved_items` and visible in run logs (when `log_path` is set on `Anchor`) for operator inspection.

## Known tradeoffs and open questions

**Character offset semantics for binary formats.** `char_start` and `char_end` reference offsets in the text extracted from the source, not byte offsets in the raw file. For PDFs this means the offsets are only meaningful relative to the extraction library's output, not the PDF file itself. Callers that need to re-locate content in the original file must re-extract it.

**`questions` as a flat string.** Storing `questions` as a newline-delimited string is a Chroma compatibility workaround. This matches the current `Ingestor` implementation but requires callers to parse the field on readback. A future migration to a store that supports list values could make this cleaner.

**`heading` is the nearest heading, not a path.** `heading` captures only the immediately preceding heading. For deeply nested documents, `section` is needed to express the full path. Both fields require extraction support in the ingestion pipeline.

**Stability of `chunk_index` across re-ingest.** If the chunking strategy or source document changes, `chunk_index` and character offsets from a prior ingest are no longer valid. There is currently no mechanism to detect or reconcile stale chunks.

**Open: `source` as path vs. URI.** The schema does not specify whether `source` should be an absolute path, a path relative to the project root, or a URI. The current `Ingestor` passes whatever string the caller supplies. A convention should be established before shipping multi-user or multi-machine ingestion.

**Open: schema versioning.** There is no version or schema field on chunk metadata. If the required field set changes, there is no way to distinguish old chunks from new ones at retrieval time. Adding a `schema_version` field (for example `"schema_version": 1`) would allow callers to handle both old and new chunks during a migration window.
