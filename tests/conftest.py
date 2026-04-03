# tests/conftest.py
from __future__ import annotations

import pytest
from tests.harness import OllamaEmbedFn, OllamaFn, is_ollama_running


@pytest.fixture(scope="session")
def ai_fn():
    if not is_ollama_running():
        pytest.skip("Ollama not running")
    return OllamaFn("qwen3:4b-instruct")


@pytest.fixture(scope="session")
def light_ai_fn():
    if not is_ollama_running():
        pytest.skip("Ollama not running")
    return OllamaFn("qwen3:0.6b")


@pytest.fixture(scope="session")
def embed_fn():
    if not is_ollama_running():
        pytest.skip("Ollama not running")
    return OllamaEmbedFn("bge-m3")
