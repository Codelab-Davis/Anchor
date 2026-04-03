# Vision

Anchor is a Python package for adding factual continuity to AI agents with minimal setup.

The project is built for developers shipping agents that need to hold and reliably retrieve a large amount of information over time. In practice, Anchor is "RAG made easy": a low-level orchestrator with a clean default path and a full custom override path when teams need to swap components.

## What Anchor provides

- A model-agnostic Python interface (`ai_fn`, `light_ai_fn`, `embed_fn`) that works with local and hosted model providers.
- A protocol-driven loop for retrieval-aware reasoning.
- Semantic retrieval over memory chunks focused on speed and answer quality.
- An officially supported custom override pattern for advanced integrations.
- Local-first ergonomics, with support for non-local deployments through custom components.

## Design intent

- Keep the core interface simple enough to adopt quickly.
- Preserve low-level control for teams running complex agents, tools, or MCP workflows.
- Prioritize deterministic debugging and traceability so behavior can be understood and tuned.
- Grow memory write-back capabilities incrementally, after core retrieval reliability is solid.
