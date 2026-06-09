"""Build the FAISS index over the local corpus — single-run entrypoint.

Ported from the prototype's ``indexer.py`` (minus the ZIP download). Indexes the
whole ``data/`` directory by default: the Match PDFs *and* the flattened Excel.
The Excel is included on purpose so the RAGAS baseline exhibits the prototype's
numeric-question failure (see RAPPORT / plan); Phase 3 loads it into SQL instead.

Usage:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --input-dir data/pdf
"""

from __future__ import annotations

import argparse
import logging

from sportsee_rag.config import get_settings
from sportsee_rag.ingestion.data_loader import load_and_parse_files
from sportsee_rag.observability import setup_observability
from sportsee_rag.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def run_indexing(input_dir: str) -> None:
    """Load & parse the corpus, then build and persist the FAISS index."""
    logger.info("--- Indexing started ---")
    documents = load_and_parse_files(input_dir)
    if not documents:
        logger.warning("No documents loaded from %s. Nothing to index.", input_dir)
        return

    manager = VectorStoreManager()
    manager.build_index(documents)

    if manager.index is not None:
        logger.info("--- Indexing complete: %d chunks indexed ---", manager.index.ntotal)
    else:
        logger.warning("--- Indexing finished but the index is empty ---")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the FAISS index for the RAG system.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(settings.data_dir),
        help=f"Source directory to index (default: {settings.data_dir})",
    )
    args = parser.parse_args()

    setup_observability()
    run_indexing(args.input_dir)


if __name__ == "__main__":
    main()
