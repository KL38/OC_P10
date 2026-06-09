"""Unit tests for the Pydantic I/O schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sportsee_rag.models import AskRequest, Chunk, RagAnswer, RetrievedChunk


def test_ask_request_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="")


def test_ask_request_rejects_k_below_one() -> None:
    with pytest.raises(ValidationError):
        AskRequest(question="hi", k=0)


def test_ask_request_accepts_valid() -> None:
    req = AskRequest(question="who is best?", k=3)
    assert req.question == "who is best?"
    assert req.k == 3


def test_models_are_frozen() -> None:
    chunk = Chunk(id="0_0", text="content")
    with pytest.raises(ValidationError):
        chunk.text = "mutated"  # type: ignore[misc]


def test_metadata_default_is_independent_per_instance() -> None:
    a = Chunk(id="1", text="t")
    b = Chunk(id="2", text="t")
    assert a.metadata == {}
    assert a.metadata is not b.metadata  # default_factory -> fresh dict each time


def test_rag_answer_defaults() -> None:
    answer = RagAnswer(answer="hello")
    assert answer.used_tool == "rag"
    assert answer.contexts == []
    assert answer.sources == []


def test_retrieved_chunk_fields() -> None:
    rc = RetrievedChunk(text="t", score=90.0, raw_score=0.9, metadata={"source": "s"})
    assert rc.score == 90.0
    assert rc.metadata["source"] == "s"
