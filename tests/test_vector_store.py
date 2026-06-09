"""Unit tests for the FAISS vector store (embeddings mocked)."""

from __future__ import annotations

from sportsee_rag.models import Chunk
from sportsee_rag.retrieval.vector_store import VectorStoreManager

from tests.conftest import FakeLLM

# Short docs (< chunk_size) -> one chunk each, with predictable fake embeddings.
DOCS = [
    {"page_content": "alpha content about teams", "metadata": {"source": "a", "filename": "a"}},
    {"page_content": "beta content about stats", "metadata": {"source": "b", "filename": "b"}},
    {"page_content": "gamma other content", "metadata": {"source": "c", "filename": "c"}},
]


def test_split_creates_typed_chunks_with_ids(settings) -> None:
    manager = VectorStoreManager(llm=FakeLLM(), settings=settings)
    docs = [{"page_content": "x " * 2000, "metadata": {"filename": "big", "source": "big"}}]
    chunks = manager._split_documents_to_chunks(docs)
    assert len(chunks) > 1  # long doc split into several chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].id == "0_0"
    assert chunks[0].metadata["chunk_id_in_doc"] == 0


def test_build_and_search_returns_best_match(settings) -> None:
    manager = VectorStoreManager(llm=FakeLLM(), settings=settings)
    manager.build_index(DOCS)
    assert manager.index is not None
    assert manager.index.ntotal == 3

    results = manager.search("an alpha question", k=1)
    assert len(results) == 1
    assert "alpha" in results[0].text  # fake embeddings make alpha the top hit
    assert isinstance(results[0].score, float)


def test_search_respects_k(settings) -> None:
    manager = VectorStoreManager(llm=FakeLLM(), settings=settings)
    manager.build_index(DOCS)
    assert len(manager.search("anything", k=2)) == 2


def test_persistence_roundtrip(settings) -> None:
    VectorStoreManager(llm=FakeLLM(), settings=settings).build_index(DOCS)
    assert settings.faiss_index_file.exists()
    assert settings.document_chunks_file.exists()

    # A fresh manager reloads the persisted index and typed chunks.
    reloaded = VectorStoreManager(llm=FakeLLM(), settings=settings)
    assert reloaded.index is not None
    assert reloaded.index.ntotal == 3
    assert len(reloaded.document_chunks) == 3
    assert isinstance(reloaded.document_chunks[0], Chunk)


def test_empty_documents_does_not_build(settings) -> None:
    manager = VectorStoreManager(llm=FakeLLM(), settings=settings)
    manager.build_index([])
    assert manager.index is None
