from anchor import Anchor

import ollama


CUSTOM_PROMPT = """\
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


# --- Safe simple components ---
class SimpleDecomposer:
    def decompose(self, gap, context="", retrieved=None, history=None):
        return [gap]  # must return real query strings


class SimpleSynthesizer:
    def synthesize(self, chunks, questions=None):
        return "\n".join(c["text"] for c in chunks)


class SimpleRetriever:
    def retrieve(self, query):
        return [
            {
                "id": f"id_{hash(query)}",
                "text": f"retrieved info for: {query}",
                "source": "mock",
            }
        ]


class SimpleIngestor:
    def ingest(self, text, source="user"):
        return f"id_{hash(text)}"


# --- Ollama wrapper ---
def ollama_chat(messages, model="qwen3:4b-instruct"):
    response = ollama.chat(model=model, messages=messages)
    print(f"[ollama_chat] got response: {response.message.content}", flush=True)
    return response["message"]["content"]


# --- Custom Anchor ---
class CustomAnchor(Anchor):
    def __init__(self, ai_fn=None, light_ai_fn=None, memory_store=None, embed_fn=None):
        if ai_fn is None:

            def ai_fn(messages):
                return ollama_chat(messages)

        if light_ai_fn is None:

            def light_ai_fn(messages):
                return ollama_chat(messages)

        super().__init__(
            ai_fn, light_ai_fn, memory_store, embed_fn, system_prompt=CUSTOM_PROMPT
        )

        # Override components safely
        self.decomposer = SimpleDecomposer()
        self.synthesizer = SimpleSynthesizer()
        self.retriever = SimpleRetriever()
        self.ingestor = SimpleIngestor()


# --- CLI loop (correct) ---
if __name__ == "__main__":
    anchor = CustomAnchor()

    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        result = anchor.run(user_input)

        print(f"Assistant: {result.content}")
        if result.stop_reason != "done":
            print(f"[stop_reason: {result.stop_reason}]")
