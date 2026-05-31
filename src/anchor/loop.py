from __future__ import annotations
import json
from pathlib import Path
from anchor.runresult import RunResult


class Loop:
    def __init__(self, anchor):
        self.anchor = anchor

    def _extract_gap(self, content: str) -> tuple[str, str]:
        gap = ""
        context = ""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("GAP:"):
                gap = line[len("GAP:") :].strip()
            elif line.startswith("CONTEXT:"):
                context = line[len("CONTEXT:") :].strip()
        return gap, context

    def _strip_marker(self, content: str, marker: str) -> str:
        lines = content.rstrip().splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip() == marker:
            lines.pop()
        return "\n".join(lines).strip()

    def run(self, query: str) -> RunResult:
        max_remembers = getattr(self.anchor, "MAX_REMEMBERS", 10)
        remembers = 0
        retrieved_items: list[dict] = []
        all_decomposed_queries: list[str] = []

        # Logging setup: truncate/create the file once, then append per event.
        _raw_log_path = getattr(self.anchor, "log_path", None)
        _log_path = Path(_raw_log_path) if _raw_log_path is not None else None
        if _log_path is not None:
            _log_path.parent.mkdir(parents=True, exist_ok=True)
            _log_path.write_text("")

        def _log(event: str, **data) -> None:
            if _log_path is None:
                return
            with _log_path.open("a") as f:
                f.write(json.dumps({"event": event, **data}) + "\n")

        def _chunk_summary(chunks: list[dict]) -> list[dict]:
            return [
                {"id": c["id"], "source": c.get("source"), "score": c.get("score")}
                for c in chunks
            ]

        def _metadata() -> dict[str, object]:
            return {
                "remember_count": remembers,
                "decomposed_queries": list(all_decomposed_queries),
                "retrieval_scores": [c["score"] for c in retrieved_items],
            }

        messages = [
            {"role": "system", "content": self.anchor.system_prompt()},
            {"role": "user", "content": query},
        ]

        _log("run_start", query=query)

        if self.anchor.retriever:
            # Pass conversation history (excluding system prompt)
            proactive_queries = self.anchor.decompose(
                query, context=query, retrieved=None, history=messages[1:]
            )
            all_decomposed_queries.extend(proactive_queries)
            _log("proactive_queries", queries=proactive_queries)

            seen_ids = set()
            proactive_chunks = []
            for q in proactive_queries:
                for chunk in self.anchor.retriever.retrieve(q):
                    if chunk["id"] not in seen_ids:
                        seen_ids.add(chunk["id"])
                        proactive_chunks.append(chunk)
            _log("proactive_chunks", chunks=_chunk_summary(proactive_chunks))

            if proactive_chunks:
                retrieved_items.extend(proactive_chunks)
                synthesis = self.anchor.synthesize(proactive_chunks, [query])
                messages.append(
                    {
                        "role": "user",
                        "content": f"[MEMORY RETRIEVAL RESULT]\n{synthesis}",
                    }
                )

        while True:
            response = self.anchor.ai(messages)
            messages.append({"role": "assistant", "content": response})
            content = response.strip()

            if content.endswith(
                self.anchor.DONE_MARKER or self.anchor.DONE_MARKER + "."
            ):
                final = self._strip_marker(content, self.anchor.DONE_MARKER)
                # new_memory = self.anchor.assess(query, final, retrieved_items)
                # if new_memory and self.anchor.ingestor:
                #     self.anchor.ingest_text(new_memory, source="agent_reasoning")
                _log("stop", stop_reason="done")
                return RunResult(
                    kind="done",
                    content=final,
                    stop_reason="done",
                    retrieved_items=retrieved_items,
                    metadata=_metadata(),
                )

            elif content.endswith(self.anchor.REMEMBER_MARKER):
                remembers += 1
                if remembers > max_remembers:
                    _log("stop", stop_reason="max_remembers")
                    return RunResult(
                        kind="done",
                        content=self._strip_marker(
                            content, self.anchor.REMEMBER_MARKER
                        ),
                        stop_reason="max_remembers",
                        retrieved_items=retrieved_items,
                        metadata=_metadata(),
                    )

                gap, context = self._extract_gap(content)
                _log("remember_gap", gap=gap, context=context)

                # Pass conversation history (excluding system prompt)
                queries = self.anchor.decompose(
                    gap,
                    context=f"{query}\n\n{context}",
                    retrieved=retrieved_items,
                    history=messages[1:],
                )
                all_decomposed_queries.extend(queries)
                _log("remember_queries", queries=queries)

                chunks = []
                if self.anchor.retriever:
                    seen_ids = {c["id"] for c in retrieved_items}
                    for q in queries:
                        for chunk in self.anchor.retriever.retrieve(q):
                            if chunk["id"] not in seen_ids:
                                seen_ids.add(chunk["id"])
                                chunks.append(chunk)
                    retrieved_items.extend(chunks)
                _log("remember_chunks", chunks=_chunk_summary(chunks))

                synthesis_chunks = chunks if chunks else retrieved_items
                synthesis = self.anchor.synthesize(synthesis_chunks, [query])
                messages.append(
                    {
                        "role": "user",
                        "content": f"[MEMORY RETRIEVAL RESULT]\n{synthesis}",
                    }
                )

            elif content.endswith(self.anchor.CLARIFY_MARKER):
                _log("stop", stop_reason="ask")
                return RunResult(
                    kind="ask",
                    content=self._strip_marker(content, self.anchor.CLARIFY_MARKER),
                    stop_reason="ask",
                    retrieved_items=retrieved_items,
                    metadata=_metadata(),
                )

            else:
                _log("stop", stop_reason="error")
                return RunResult(
                    kind="done",
                    content=content,
                    stop_reason="error",
                    retrieved_items=retrieved_items,
                    metadata=_metadata(),
                )
