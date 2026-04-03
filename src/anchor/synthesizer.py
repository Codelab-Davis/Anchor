from __future__ import annotations


class Synthesizer:
    def __init__(self, model_fn):
        self.model_fn = model_fn

    def synthesize(self, chunks: list[dict], questions: list[str] | None = None) -> str:
        if not chunks:
            return "No relevant information found in memory."
        if len(chunks) == 1:
            c = chunks[0]
            return f"[Source: {c.get('source', 'unknown')}]\n{c.get('content', '')}"

        chunks_text = "\n\n---\n\n".join(
            f"[Chunk {i+1}] [Source: {c.get('source', 'unknown')}]\n{c.get('content', '')}"
            for i, c in enumerate(chunks)
        )

        if questions:
            if len(questions) == 1:
                questions_text = f"- {questions[0]} (current)"
            else:
                questions_text = "\n".join(
                    f"- {q}" if i != len(questions) - 1 else f"- {q} (current)"
                    for i, q in enumerate(questions)
                )
        else:
            questions_text = "None"

        prompt = (
            "Summarize these retrieved memory chunks to help answer the following questions.\n"
            "The final question marked (current) is the one being answered now.\n\n"
            f"Questions:\n{questions_text}\n\n"
            f"Chunks:\n{chunks_text}\n\n"
            "Include all information relevant to the questions. Be concise.\n"
            "Only use information present in the chunks. Do not fill gaps with outside knowledge."
        )

        return self.model_fn([{"role": "user", "content": prompt}])
