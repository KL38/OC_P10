"""Pydantic I/O schemas for the RAG chain.

These models validate data at the system boundaries (requests in, answers out)
and give the pipeline — and later the Pydantic AI agent — typed, predictable
shapes to work with. Value objects are frozen (immutable) by design.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A single indexed text chunk, as persisted alongside the FAISS index.

    ``metadata`` stays a free-form dict on purpose: it carries heterogeneous
    keys inherited from the loader (source, filename, category, sheet,
    start_index, chunk_id_in_doc...) and we do not want to over-constrain it.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned by a similarity search, with its score."""

    model_config = ConfigDict(frozen=True)

    text: str
    score: float  # cosine similarity as a percentage (0-100), as in the proto
    raw_score: float  # raw inner-product score, kept for debugging
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    """An incoming user question."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1)  # None -> use settings.search_k


class RagAnswer(BaseModel):
    """A typed answer produced by the RAG pipeline (and later the agent).

    ``contexts`` holds the raw retrieved texts, which RAGAS consumes directly
    for faithfulness / context metrics. ``used_tool`` is forward-looking: in
    Phase 3 the agent will route between the RAG and SQL tools.
    """

    model_config = ConfigDict(frozen=True)

    answer: str
    contexts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    used_tool: Literal["rag", "sql"] = "rag"
