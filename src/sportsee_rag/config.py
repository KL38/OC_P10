"""Typed application configuration (pydantic-settings).

Single source of truth for settings, read from the environment / a `.env`
file. Paths are anchored to the project root so scripts run correctly from
any working directory.

No side effects at import time (unlike the prototype's ``config.py``): call
``get_settings()`` to load. A missing required variable (``MISTRAL_API_KEY``)
raises a clear validation error on first call.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# OC_P10/  <-  src/  <-  sportsee_rag/  <-  config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings.

    Field names map to environment variables case-insensitively
    (``mistral_api_key`` <- ``MISTRAL_API_KEY``).
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets / external services ---
    mistral_api_key: str  # required: fails fast if absent
    database_url: str | None = None  # Phase 3 (Supabase / PostgreSQL)
    logfire_token: str | None = None  # optional: local mode if absent

    # --- Mistral models (confirm exact ids against the Mistral docs) ---
    chat_model: str = "mistral-small-latest"
    embedding_model: str = "mistral-embed"

    # --- Chunking / embeddings ---
    chunk_size: int = 1500
    chunk_overlap: int = 150
    embedding_batch_size: int = 32

    # --- Embedding resilience (throttle + retry/backoff vs Mistral free rate-limits) ---
    embedding_max_retries: int = 5
    embedding_throttle_seconds: float = 0.5  # pause between batches

    # --- Retrieval ---
    search_k: int = 5  # default number of chunks retrieved

    # --- SQL / database (Phase 3) ---
    # Mistral's OpenAI-compatible endpoint, used by the Pydantic AI agent
    # (pydantic-ai-slim without the `mistral` extra, which would force mistralai>=2).
    mistral_openai_base_url: str = "https://api.mistral.ai/v1"
    sql_sample_rows_in_table_info: int = 3  # sample rows shown to the SQL LLM
    sql_default_limit: int = 20  # LIMIT hint for generated non-aggregate queries

    # --- Paths (anchored to the project root) ---
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    vector_db_dir: Path = PROJECT_ROOT / "vector_db"
    sqlite_db_file: Path = PROJECT_ROOT / "db" / "nba.sqlite"  # local fallback DB

    @property
    def faiss_index_file(self) -> Path:
        """Path to the persisted FAISS index."""
        return self.vector_db_dir / "faiss_index.idx"

    @property
    def document_chunks_file(self) -> Path:
        """Path to the persisted chunks (jsonl, not pickle — portable & safe)."""
        return self.vector_db_dir / "document_chunks.jsonl"

    @property
    def excel_file(self) -> Path:
        """Path to the NBA stats Excel workbook (source of the SQL database)."""
        return self.data_dir / "regular+NBA.xlsx"

    @property
    def sqlalchemy_url(self) -> str:
        """SQLAlchemy connection URL: Supabase/PostgreSQL if ``DATABASE_URL`` is
        set, otherwise a local SQLite file (offline fallback, used by tests).

        Supabase hands out ``postgresql://`` URLs; SQLAlchemy needs the driver
        spelled out to pick psycopg v3 instead of the absent psycopg2.
        """
        if self.database_url:
            url = self.database_url
            for prefix in ("postgresql://", "postgres://"):
                if url.startswith(prefix):
                    return "postgresql+psycopg://" + url[len(prefix):]
            return url
        return f"sqlite:///{self.sqlite_db_file.as_posix()}"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (instantiated on first call)."""
    return Settings()
