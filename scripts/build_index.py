"""Build the FAISS index over the local corpus — single-run entrypoint.

Ported from the prototype's ``indexer.py`` (minus the ZIP download). Indexes the
whole ``data/`` directory by default: the Match PDFs *and* the flattened Excel.
The Excel is included on purpose so the RAGAS baseline exhibits the prototype's
numeric-question failure (see RAPPORT / plan); Phase 3 loads it into SQL instead.

Usage:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --pdf-only   # PDF-only index (no flattened Excel)
"""

from __future__ import annotations

import argparse
import logging

from sportsee_rag.config import Settings, get_settings
from sportsee_rag.ingestion.data_loader import load_and_parse_files
from sportsee_rag.observability import setup_observability
from sportsee_rag.retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

EXCEL_SUFFIXES = {".xlsx", ".xls"}


def run_indexing(
    input_dir: str, *, settings: Settings, exclude_suffixes: set[str] | None = None
) -> None:
    """Load & parse the corpus, then build and persist the FAISS index."""
    logger.info("--- Indexing started (variant='%s') ---", settings.index_variant or "full")
    documents = load_and_parse_files(input_dir, exclude_suffixes=exclude_suffixes)
    if not documents:
        logger.warning("No documents loaded from %s. Nothing to index.", input_dir)
        return

    manager = VectorStoreManager(settings=settings)
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
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Exclude the flattened Excel and write to the '_pdf' index variant "
             "(faiss_index_pdf.idx). Used for the enriched_v2 evaluation.",
    )
    args = parser.parse_args()

    setup_observability()
    exclude = EXCEL_SUFFIXES if args.pdf_only else None
    if args.pdf_only:
        settings = settings.model_copy(update={"index_variant": "_pdf"})
    run_indexing(args.input_dir, settings=settings, exclude_suffixes=exclude)


if __name__ == "__main__":
    main()
