from __future__ import annotations
from abc import ABC
from enum import Enum
from anchor.config import AnchorConfig
from anchor.runresult import RunResult


DEFAULT_SYSTEM_PROMPT = """\
You are the final output for a user facing reasoning agent with access to a persistent memory store.
The user cannot see any memory store related output so you must output explanations and reasoning steps as part of your own output for the user.

Do not directly address the user unless you need to ask for clarification using the CLARIFY protocol. Always assume the user will read everything you output, including any internal reasoning or explanations.
Be concise but complete in your reasoning. Avoid unnecessary repetition. Do not self-censor or avoid mentioning uncertainties or gaps in your knowledge.

Once the retrieved information fully resolves the question output your final answer, then
emit a terminal marker in this exact format to complete the current question:

{DONE_MARKER}


When you hit definitive uncertainty — missing facts, unclear context,
gaps in knowledge, contradictions, etc, that are needed to answer the question — emit a terminal marker in this exact format to perform a search of the memory store before continuing:

GAP: <what you're missing, in one sentence in the form of a question>
CONTEXT: <brief state of your current reasoning>
{REMEMBER_MARKER}


Only in the case the user's intent is genuinely unclear — for example, they have not specified what they want you to do,
not because you don't recognize an entity name, identifier, or term (use REMEMBER for those) — emit a terminal marker in this exact format to ask the user before continuing:

QUESTION: <specific question for the user to clarify their intent>
{CLARIFY_MARKER}


Your output will be evaluated for new information that can be saved to memory after the question is resolved.
If the user question includes information that should be saved therefore note that in your response, otherwise do not explicitly mention the memory-saving step.
In the case of memory use the future tense instead of the present tense ("I will" vs "I have").

You can and should use {REMEMBER_MARKER} multiple times to expand the scope of retrieval if you are still missing information. Check for aliases and any other relevant information in memory to fill gaps.
In the case there is clearly no new information from the result of a {REMEMBER_MARKER} assume the information is not present in memory and make a best effort to answer without fabrication, choosing either {CLARIFY_MARKER} or {DONE_MARKER}.

Only ever emit one of the terminal markers in your response: {REMEMBER_MARKER}, {CLARIFY_MARKER}, {DONE_MARKER}. Never emit more than one terminal marker in a single response.
Any terminal marker must be the final output in your response.\
"""


class Anchor(ABC):
    class Markers(Enum):
        REMEMBER = "REMEMBER"
        CLARIFY = "CLARIFY"
        DONE = "DONE"

    MAX_REMEMBERS: int = 10

    def __init__(
        self,
        ai_fn,
        light_ai_fn,
        memory_store=None,
        embed_fn=None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        from anchor.loop import Loop
        from anchor.synthesizer import Synthesizer
        from anchor.retriever import Retriever
        from anchor.decomposer import Decomposer
        from anchor.ingestor import Ingestor

        self._ai_fn = ai_fn
        self._light_ai_fn = light_ai_fn
        self.synthesizer = Synthesizer(light_ai_fn)
        self.decomposer = Decomposer(light_ai_fn)
        self.retriever = (
            Retriever(memory_store, embed_fn) if memory_store and embed_fn else None
        )
        self.ingestor = (
            Ingestor(memory_store, embed_fn, question_fn=light_ai_fn)
            if memory_store and embed_fn
            else None
        )
        self._loop = Loop(self)
        self._system_prompt = system_prompt

    def ai(self, messages: list[dict]) -> str:
        return self._ai_fn(messages)

    def light_ai(self, messages: list[dict]) -> str:
        return self._light_ai_fn(messages)

    def system_prompt(self) -> str:
        return self._system_prompt.format(
            REMEMBER_MARKER=self.REMEMBER_MARKER,
            CLARIFY_MARKER=self.CLARIFY_MARKER,
            DONE_MARKER=self.DONE_MARKER,
        )

    @property
    def REMEMBER_MARKER(self) -> str:
        return self.Markers.REMEMBER.value

    @property
    def CLARIFY_MARKER(self) -> str:
        return self.Markers.CLARIFY.value

    @property
    def DONE_MARKER(self) -> str:
        return self.Markers.DONE.value

    def config(self) -> AnchorConfig:
        return AnchorConfig(max_remembers=self.MAX_REMEMBERS)

    def ingest_text(self, text: str, source: str = "user") -> str:
        if not self.ingestor:
            raise RuntimeError("No memory store configured.")
        return self.ingestor.ingest(text, source=source)

    def decompose(
        self,
        gap: str,
        context: str = "",
        retrieved: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> list[str]:
        return self.decomposer.decompose(
            gap, context, retrieved=retrieved, history=history
        )

    def synthesize(self, chunks: list[dict], questions: list[str] | None = None) -> str:
        return self.synthesizer.synthesize(chunks, questions)

    def run(self, query: str) -> RunResult:
        return self._loop.run(query)
