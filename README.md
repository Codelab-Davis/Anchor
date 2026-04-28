# Anchor

Anchor is an early-stage, model-agnostic memory layer for AI agents.

It helps agents keep factual continuity across interactions by combining:

- a protocol-driven reasoning loop (`REMEMBER`, `CLARIFY`, `DONE`)
- semantic retrieval from stored memory chunks
- ingest-time question generation to improve recall

## How Anchor works

At a high level:

1. Anchor receives a user query.
2. It proactively decomposes the query into semantic retrieval queries.
3. Retrieved memory is synthesized into compact context.
4. The core model responds using protocol markers:
   - `REMEMBER` when more memory lookup is needed
   - `CLARIFY` when user intent is ambiguous
   - `DONE` when the answer is complete
5. Anchor loops until completion or a configured remember limit.

## Current scope

- Python package with a low-level orchestrator API
- Local-first workflows (for example Ollama + Chroma), but provider-agnostic by design
- Official custom override pattern for teams that need to swap core components
- Deterministic tests in CI and model evals run manually

Memory write-back and richer ingestion/UI workflows are planned next-stage work.

## Prerequisites

```bash
# Python environment + project tooling
uv sync --group dev

# Install local git hooks
uv run pre-commit install
```

For local examples and evals, install and run Ollama:

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama in a separate terminal
ollama serve

# Pull models used by examples/evals
# examples/chat.py
ollama pull qwen3:1.7b
ollama pull bge-m3

# examples/custom_anchor.py (default model)
ollama pull qwen3:4b-instruct

# tests/evals fixtures
ollama pull qwen3:0.6b
```

For Linux and Windows install steps, use the official Ollama docs: <https://docs.ollama.com/>

If you only run deterministic tests (`-m "not eval"`), Ollama is optional.

## Quickstart

Use `ChromaMemoryStore` for quick demos (in-memory) and `PersistentChromaMemoryStore` when memory should persist across restarts.

```bash
# In-memory store (does NOT persist)
uv run python examples/chat.py

# Persistent store (recommended for real use)
uv run python examples/persistent_chat.py
```

## Run examples

```bash
# Default integration path
uv run python examples/chat.py

# Official custom override pattern
uv run python examples/custom_anchor.py
```

## Test commands

```bash
# Deterministic tests (CI path)
uv run pytest -m "not eval"

# Live model evals (manual)
uv run pytest -m eval tests/evals
```

## Ollama process hygiene

```bash
# Show loaded/running models
ollama ps

# Stop a loaded model
ollama stop qwen3:4b-instruct
```

If you started Ollama with `ollama serve`, stop it with `Ctrl+C` in that terminal.

## Local quality checks

```bash
# Fork the repository on GitHub, then clone your fork.
# Create a feature branch in your fork before running full hooks
git checkout -b yourname/short-topic

# Run hooks on all files
uv run pre-commit run --all-files

# Optional: docs/text whitespace checks
git diff --check
```

## Troubleshooting

- `no-commit-to-branch` failed:
  This hook blocks direct commits to `main`. In your fork, create and commit on a feature branch, for example `git checkout -b yourname/short-topic`.

- `end-of-file-fixer` modified files:
  Re-run `uv run pre-commit run --all-files` and commit the hook changes.

## Repository docs

- Contributor workflow: [CONTRIBUTING.md](CONTRIBUTING.md)
- Maintainer operations: [docs/maintainers.md](docs/maintainers.md)
- Security disclosure path: [SECURITY.md](SECURITY.md)
- Project direction: [docs/vision.md](docs/vision.md), [docs/roadmap.md](docs/roadmap.md)
