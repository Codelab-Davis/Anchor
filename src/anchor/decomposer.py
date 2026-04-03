from __future__ import annotations
import json


class Decomposer:
    def __init__(self, model_fn):
        self.model_fn = model_fn

    def decompose(
        self,
        gap: str,
        context: str = "",
        retrieved: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> list[str]:
        raw_retrieval_text = "None."
        if retrieved:
            entries = [
                {
                    "id": item.get("id", ""),
                    "questions": item.get("questions", ""),
                    "content": item.get("content", ""),
                }
                for item in retrieved[:8]
            ]
            raw_retrieval_text = json.dumps(entries, ensure_ascii=True, indent=2)

        # Format history if provided
        history_text = ""
        if history:
            formatted = []
            for msg in history[-6:]:  # Limit to last 6 messages for brevity
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted.append(f"[{role}] {content}")
            history_text = "Previous conversation:\n" + "\n".join(formatted) + "\n\n"

        if not retrieved:
            prompt = (
                f"{history_text}"
                "Output 2-3 semantic search queries to retrieve relevant facts for the following user question.\n"
                "No preamble. No numbering. No explanation. No commentary.\n\n"
                "The queries must be grounded strictly in the user question below.\n"
                "Do not invent, assume, or reference entities not explicitly mentioned in the question.\n\n"
                f"User question: {gap}\n\n"
                "Output only the queries now:"
            )
        else:
            prompt = (
                f"{history_text}"
                "Output 2-3 semantic search queries, one per line, nothing else.\n"
                "No preamble. No numbering. No explanation. No commentary.\n\n"
                "The AI is reasoning about a task and has hit uncertainty.\n\n"
                f"Current context (last ~500 chars):\n{context[-500:]}\n\n"
                f"The AI says it's missing: {gap}\n\n"
                "Facts already retrieved:\n"
                f"{raw_retrieval_text}\n\n"
                "Rules:\n"
                "1. Queries must only reference entities and terms present in the context or gap above.\n"
                "2. Do not invent or hallucinate entity names, codenames, or identifiers.\n"
                "3. If retrieved facts mention named entities not yet looked up, query those identifiers.\n"
                "4. If two facts contradict, query for supersession or precedence rules naming both.\n"
                "5. Do not regenerate queries for facts already present above.\n\n"
                "Output only the queries now:"
            )
        raw = self.model_fn([{"role": "user", "content": prompt}])
        queries = [self._normalize(q) for q in raw.splitlines() if self._normalize(q)]
        if gap and gap not in queries:
            queries.insert(0, gap)
        return queries

    def _normalize(self, line: str) -> str:
        normalized = line.strip()
        while True:
            if (
                len(normalized) > 2
                and normalized[0].isdigit()
                and normalized[1] in {".", ")"}
                and normalized[2] == " "
            ):
                normalized = normalized[3:].strip()
                continue
            if normalized.startswith("- ") or normalized.startswith("* "):
                normalized = normalized[2:].strip()
                continue
            break
        return normalized
