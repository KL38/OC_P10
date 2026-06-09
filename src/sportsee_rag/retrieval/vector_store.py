"""FAISS vector store (ported from the prototype, algorithm unchanged).

Faithful port of ``brief/P10_DSML/utils/vector_store.py``. The retrieval logic
is identical — cosine similarity via ``IndexFlatIP`` + ``normalize_L2``, chunking
1500/150, same scoring/sorting/min-score filtering. Only the *plumbing* changes,
and all changes are output-neutral:

- Embeddings go through ``MistralLLM`` (migrated SDK + retry/backoff/throttle).
  On definitive failure it raises ``MistralError`` instead of injecting *null
  vectors* — the prototype's index-corrupting bug is gone.
- Chunks are validated as ``Chunk`` and persisted as **JSONL** (not pickle):
  portable and safe, with no effect on what gets retrieved.
- ``search`` returns typed ``RetrievedChunk`` objects (Pydantic I/O).
"""

from __future__ import annotations

import logging
import time

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Settings, get_settings
from ..llm.client import MistralLLM
from ..models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Creates, persists, loads and searches a FAISS cosine-similarity index."""

    def __init__(self, llm: MistralLLM | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or MistralLLM(settings=self._settings)
        self.index: faiss.Index | None = None
        self.document_chunks: list[Chunk] = []
        self._load_index_and_chunks()

    # --- persistence ----------------------------------------------------

    def _load_index_and_chunks(self) -> None:
        """Load the FAISS index and chunks from disk if both files exist."""
        index_file = self._settings.faiss_index_file
        chunks_file = self._settings.document_chunks_file
        if not (index_file.exists() and chunks_file.exists()):
            logger.warning("FAISS index or chunks file not found. Index is empty.")
            return
        try:
            self.index = faiss.read_index(str(index_file))
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.document_chunks = [
                    Chunk.model_validate_json(line) for line in f if line.strip()
                ]
            logger.info(
                "Loaded index (%d vectors) and %d chunks.",
                self.index.ntotal, len(self.document_chunks),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load index/chunks: %s", exc)
            self.index = None
            self.document_chunks = []

    def _save_index_and_chunks(self) -> None:
        """Persist the FAISS index and chunks (chunks as JSONL)."""
        if self.index is None or not self.document_chunks:
            logger.warning("Refusing to save an empty index or chunk list.")
            return
        self._settings.vector_db_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._settings.faiss_index_file))
        with open(self._settings.document_chunks_file, "w", encoding="utf-8") as f:
            for chunk in self.document_chunks:
                f.write(chunk.model_dump_json() + "\n")
        logger.info("Index and %d chunks saved.", len(self.document_chunks))

    # --- build ----------------------------------------------------------

    def _split_documents_to_chunks(self, documents: list[dict]) -> list[Chunk]:
        """Split documents into overlapping chunks (1500/150), with metadata."""
        logger.info(
            "Splitting %d document(s) (size=%d, overlap=%d)...",
            len(documents), self._settings.chunk_size, self._settings.chunk_overlap,
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            length_function=len,  # measure in characters
            add_start_index=True,
        )
        chunks: list[Chunk] = []
        for doc_counter, doc in enumerate(documents):
            lc_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            pieces = splitter.split_documents([lc_doc])
            logger.info(
                "  '%s' -> %d chunks.", doc["metadata"].get("filename", "N/A"), len(pieces)
            )
            for i, piece in enumerate(pieces):
                chunks.append(Chunk(
                    id=f"{doc_counter}_{i}",
                    text=piece.page_content,
                    metadata={
                        **piece.metadata,
                        "chunk_id_in_doc": i,
                        "start_index": piece.metadata.get("start_index", -1),
                    },
                ))
        logger.info("Total %d chunks created.", len(chunks))
        return chunks

    def _generate_embeddings(self, chunks: list[Chunk]) -> np.ndarray:
        """Embed all chunks batch by batch, throttling between batches.

        Unlike the prototype, a failed batch is NOT padded with null vectors:
        ``MistralLLM.embed`` raises ``MistralError`` after its retries, which
        aborts the build cleanly (no corrupt index is ever written).
        """
        batch_size = self._settings.embedding_batch_size
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        logger.info("Embedding %d chunks (model=%s)...", len(chunks), self._settings.embedding_model)

        all_embeddings: list[list[float]] = []
        for batch_num, start in enumerate(range(0, len(chunks), batch_size), start=1):
            batch = chunks[start:start + batch_size]
            logger.info("  Batch %d/%d (%d chunks)", batch_num, total_batches, len(batch))
            all_embeddings.extend(self._llm.embed([c.text for c in batch]))
            if batch_num < total_batches:  # throttle to respect free-tier limits
                time.sleep(self._settings.embedding_throttle_seconds)

        embeddings = np.array(all_embeddings).astype("float32")
        logger.info("Embeddings generated. Shape: %s", embeddings.shape)
        return embeddings

    def build_index(self, documents: list[dict]) -> None:
        """Build and persist the FAISS index from documents."""
        if not documents:
            logger.warning("No documents provided to build the index.")
            return

        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logger.error("Splitting produced no chunks. Aborting index build.")
            return

        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings.shape[0] != len(self.document_chunks):
            logger.error("Embedding count mismatch; aborting to avoid a corrupt index.")
            self.document_chunks = []
            self.index = None
            return

        # Cosine similarity = inner product on L2-normalised vectors.
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logger.info("FAISS index built with %d vectors.", self.index.ntotal)

        self._save_index_and_chunks()

    # --- search ---------------------------------------------------------

    def search(
        self, query_text: str, k: int | None = None, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``k`` most relevant chunks for ``query_text``.

        ``min_score`` (0-1) optionally filters out low-similarity results.
        Algorithm identical to the prototype; only the return type is typed.
        """
        k = k if k is not None else self._settings.search_k
        if self.index is None or not self.document_chunks:
            logger.warning("Search impossible: index not loaded or empty.")
            return []

        logger.info("Searching %d most relevant chunks for: '%s'", k, query_text)
        # Embed + normalise the query the same way as the indexed vectors.
        query_embedding = np.array([self._llm.embed([query_text])[0]]).astype("float32")
        faiss.normalize_L2(query_embedding)

        # Ask for more candidates when a min-score filter is active.
        search_k = k * 3 if min_score is not None else k
        scores, indices = self.index.search(query_embedding, search_k)

        results: list[RetrievedChunk] = []
        if indices.size > 0:
            min_score_pct = min_score * 100 if min_score is not None else 0
            for i, idx in enumerate(indices[0]):
                if not (0 <= idx < len(self.document_chunks)):
                    continue
                raw_score = float(scores[0][i])
                similarity = raw_score * 100  # normalised IP in [-1, 1] -> percentage
                if min_score is not None and similarity < min_score_pct:
                    continue
                chunk = self.document_chunks[idx]
                results.append(RetrievedChunk(
                    text=chunk.text,
                    score=similarity,
                    raw_score=raw_score,
                    metadata=chunk.metadata,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        if len(results) > k:
            results = results[:k]
        logger.info("%d relevant chunks found.", len(results))
        return results
