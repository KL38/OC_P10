"""Unit tests for the RAG pipeline (manager + LLM mocked)."""

from __future__ import annotations

from sportsee_rag.models import AskRequest, RetrievedChunk
from sportsee_rag.rag.pipeline import answer_question


class FakeManager:
    """Returns a fixed set of retrieved chunks, no FAISS / network."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results

    def search(self, query: str, k=None) -> list[RetrievedChunk]:
        return self._results


def test_answer_question_grounds_and_types(settings, fake_llm) -> None:
    fake_llm._answer = "MOCKED ANSWER"
    manager = FakeManager([
        RetrievedChunk(text="ctx one", score=90.0, raw_score=0.9, metadata={"source": "s1"}),
        RetrievedChunk(text="ctx two", score=80.0, raw_score=0.8, metadata={"source": "s2"}),
    ])

    answer = answer_question(
        AskRequest(question="who is best?"),
        manager=manager,
        llm=fake_llm,
        settings=settings,
    )

    assert answer.answer == "MOCKED ANSWER"
    assert answer.contexts == ["ctx one", "ctx two"]  # raw texts feed RAGAS
    assert answer.sources == ["s1", "s2"]
    assert answer.used_tool == "rag"


def test_answer_question_handles_no_context(settings, fake_llm) -> None:
    manager = FakeManager([])
    answer = answer_question(
        AskRequest(question="obscure question?"),
        manager=manager,
        llm=fake_llm,
        settings=settings,
    )
    assert answer.contexts == []
    assert answer.sources == []
    assert answer.answer  # still produces an answer from the LLM
