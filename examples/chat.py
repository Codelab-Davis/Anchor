# examples/chat.py
import ollama as _ollama
from anchor import Anchor
from anchor.memory import ChromaMemoryStore


class OllamaFn:
    def __init__(self, model: str):
        self.model = model

    def __call__(self, messages: list[dict]) -> str:
        print("[OllamaFn] calling model...", flush=True)
        response = _ollama.chat(model=self.model, messages=messages, think=False)
        print("[OllamaFn] got response: " + response.message.content, flush=True)
        return response.message.content


class OllamaEmbedFn:
    def __init__(self, model: str = "bge-m3"):
        self.model = model

    def __call__(self, text: str) -> list[float]:
        return _ollama.embeddings(model=self.model, prompt=text).embedding


SEED_CHUNKS = [
    ("The project codename is ORCHID-7.", "seed"),
    ("The max retry limit is 7 for standard tenants.", "seed"),
    ("Severity levels are SABLE, BRASS, and IVORY in descending order.", "seed"),
    ("The release train KITE-DELTA ships on Thursdays at 19:40 UTC.", "seed"),
    ("Tenant COBALT has a payout holdback of 4 percent.", "seed"),
]


def main():
    ai = OllamaFn("qwen3:1.7b")
    light_ai = OllamaFn("qwen3:1.7b")
    embed_fn = OllamaEmbedFn()
    memory_store = ChromaMemoryStore()

    anchor = Anchor(
        ai_fn=ai,
        light_ai_fn=light_ai,
        memory_store=memory_store,
        embed_fn=embed_fn,
    )

    print("Ingesting seed chunks...")
    for text, source in SEED_CHUNKS:
        chunk_id = anchor.ingest_text(text, source=source)
        print(f"  ingested [{source}]: {text[:60]} -> {chunk_id}")
    print()

    print("Chat with Anchor. Type 'exit' to quit.\n")
    while True:
        query = input("You> ").strip()
        if not query or query.lower() == "exit":
            break
        result = anchor.run(query)
        print(f"Anchor> {result.content}")
        if result.stop_reason != "done":
            print(f"  [stop_reason: {result.stop_reason}]")
        print()


if __name__ == "__main__":
    main()
