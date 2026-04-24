from __future__ import annotations
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

        if self.anchor.retriever:
            # Pass conversation history (excluding system prompt)
            proactive_queries = self.anchor.decompose(
                query, context=query, retrieved=None, history=messages[1:]
            )
            all_decomposed_queries.extend(proactive_queries)
            seen_ids = set()
            proactive_chunks = []
            for q in proactive_queries:
                for chunk in self.anchor.retriever.retrieve(q):
                    if chunk["id"] not in seen_ids:
                        seen_ids.add(chunk["id"])
                        proactive_chunks.append(chunk)
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
                # Pass conversation history (excluding system prompt)
                queries = self.anchor.decompose(
                    gap,
                    context=f"{query}\n\n{context}",
                    retrieved=retrieved_items,
                    history=messages[1:],
                )
                all_decomposed_queries.extend(queries)

                chunks = []
                if self.anchor.retriever:
                    seen_ids = {c["id"] for c in retrieved_items}
                    for q in queries:
                        for chunk in self.anchor.retriever.retrieve(q):
                            if chunk["id"] not in seen_ids:
                                seen_ids.add(chunk["id"])
                                chunks.append(chunk)
                    retrieved_items.extend(chunks)

                synthesis_chunks = chunks if chunks else retrieved_items
                synthesis = self.anchor.synthesize(synthesis_chunks, [query])
                messages.append(
                    {
                        "role": "user",
                        "content": f"[MEMORY RETRIEVAL RESULT]\n{synthesis}",
                    }
                )

            elif content.endswith(self.anchor.CLARIFY_MARKER):
                return RunResult(
                    kind="ask",
                    content=self._strip_marker(content, self.anchor.CLARIFY_MARKER),
                    stop_reason="ask",
                    retrieved_items=retrieved_items,
                    metadata=_metadata(),
                )

            else:
                return RunResult(
                    kind="done",
                    content=content,
                    stop_reason="error",
                    retrieved_items=retrieved_items,
                    metadata=_metadata(),
                )
