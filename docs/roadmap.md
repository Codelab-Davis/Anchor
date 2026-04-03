# Roadmap

This roadmap tracks delivery of Anchor as a production-ready Python package for retrieval-backed agent continuity.

## Phase 1

Goal: ship a stable, test-backed core loop with official default and custom integration paths.

### Core model loop

- Retrieve proactive context from decomposed queries.
- Retrieve context for the current user query.
- Pass retrieval context into the core model loop.
- Enforce protocol markers and valid stop behavior.
- Return user-facing output on each loop iteration.
- Add tests for protocol validity and loop logic correctness.

### Proactive retrieval + decomposer

- Decompose input into semantically relevant retrieval queries.
- Handle unknown terms and exploration cases through decomposition prompts.
- Retrieve and return relevant chunks as context.
- Add tests for decomposition output quality and retrieval logic.

### Synthesizer

- Combine multiple chunks into concise, relevant context for the core loop.
- Filter or down-rank irrelevant chunk content.
- Add tests for synthesis output quality and deterministic logic behavior.

### Ingestor + retriever

- Ingest individual chunks into the vector store.
- Generate semantically similar retrieval questions at ingest time.
- Embed and store chunk content plus generated retrieval hints.
- Retrieve by semantic similarity from embedded queries.
- Add tests for ingest and retrieval logic correctness.

### Official integration patterns

- Keep `examples/chat.py` as the default runnable path.
- Treat `examples/custom_anchor.py` as an official supported override pattern.
- Validate both patterns as part of Phase 1 acceptance.

### Observability and debugging

- Add file-based run logs for core loop decisions.
- Include decomposition queries, retrieval decisions, and final stop reason in logs.
- Add tests for logging behavior where deterministic assertions are possible.

### Testing and evaluation policy

- CI runs deterministic/unit tests only.
- Model-based evals are run manually when needed.
- Evals use tolerant pass criteria (for example substring/rubric checks), not strict exact-string matching.
- Eval memory stores must be isolated and cleaned up after tests.

### Storage direction

- Chroma is the recommended long-term default memory backend.
- Persistent Chroma storage is recommended configuration.
- Non-persistent setup remains acceptable as the default quick-start path.

## Phase 2

Goal: support broad content ingestion, persistent AI memory, and operator-facing visibility into agent behavior.

### Ingestion and chunking

- Add file ingestion support for common document formats (for example PDF, Word docs, plain text, and Markdown).
- Add source code ingestion support for common code formats (for example Python, JavaScript/TypeScript, JSON, and YAML).
- Normalize extracted content into a consistent intermediate representation.
- Build chunking strategies that preserve semantic meaning and source context for both prose and code.
- Attach useful metadata for downstream retrieval and debugging (source, section, page/offset when available).
- Attach code-aware metadata where available (file path, symbol/function/class context, and line ranges).
- Route normalized chunks into the existing ingestor interface with minimal user configuration.
- Add tests for extraction accuracy, chunk quality, and ingestion pipeline logic.

### Persistent AI memory

- Add persistent write-back memory support as a second store, separate from baseline ingested knowledge.
- Retrieve from both stores during answering, with clear attribution of which store each result came from.
- Add evaluation logic for memory lifecycle management (for example pruning or deleting stale/low-value write-back items).
- Add configurable retention policies based on usage signals, including token-processing thresholds.
- Add tests for dual-store retrieval behavior, write-back quality, and memory lifecycle actions.

### Visualization and control UI

- Build a UI to inspect stored vectors/chunks and metadata.
- Show what is being retrieved in real time for each query/iteration.
- Add an interaction trace view so users can inspect agent actions across loop steps.
- Add model selection controls in the UI for core and lightweight model functions.
- Add tests for critical UI state and backend integration paths where deterministic coverage is possible.
