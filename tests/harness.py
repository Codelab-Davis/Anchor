# tests/harness.py
from __future__ import annotations

import ollama as _ollama


def is_ollama_running() -> bool:
    try:
        _ollama.list()
        return True
    except Exception:
        return False


class OllamaFn:
    def __init__(self, model: str = "qwen3:4b-instruct"):
        self.model = model

    def __call__(self, messages: list[dict]) -> str:
        response = _ollama.chat(model=self.model, messages=messages, think=False)
        print(f"[OllamaFn] got response: {response.message.content}", flush=True)
        return response.message.content.strip()


class OllamaEmbedFn:
    def __init__(self, model: str = "bge-m3"):
        self.model = model

    def __call__(self, text: str) -> list[float]:
        return _ollama.embeddings(model=self.model, prompt=text).embedding
