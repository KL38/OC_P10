"""Shared pytest fixtures.

Provides an isolated ``Settings`` (dummy key, temp paths, no throttle) and a
deterministic ``FakeLLM`` so the whole RAG chain is testable without any network
call or API key.
"""

from __future__ import annotations

import pytest

from sportsee_rag.config import Settings

DEFAULT_ANSWER = "FAKE LLM ANSWER"


class FakeLLM:
    """Deterministic stand-in for ``MistralLLM`` (no network, no key).

    ``embed`` maps a keyword in the text to a fixed orthogonal vector, so cosine
    similarity is fully predictable in tests. ``chat`` returns a canned answer.
    """

    def __init__(self, answer: str = DEFAULT_ANSWER) -> None:
        self._answer = answer

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        return self._answer


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Isolated settings: dummy key, temp data/vector dirs, no throttle."""
    return Settings(
        mistral_api_key="test-key",
        data_dir=tmp_path,
        vector_db_dir=tmp_path / "vdb",
        embedding_throttle_seconds=0.0,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
