"""Load the NBA Excel workbook into the SQL database — single-run entrypoint.

Validates every row through Pydantic, then recreates and fills the ``teams``,
``players`` and ``stats`` tables. Targets Supabase/PostgreSQL when
``DATABASE_URL`` is set in ``.env``, otherwise a local SQLite file
(``db/nba.sqlite``).

Usage:
    uv run python scripts/load_excel_to_db.py
    uv run python scripts/load_excel_to_db.py --excel data/regular+NBA.xlsx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sqlalchemy import text

from sportsee_rag.config import get_settings
from sportsee_rag.observability import setup_observability
from sportsee_rag.sql.loader import create_db_engine, load_excel_to_db

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Load the NBA Excel workbook into SQL.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=settings.excel_file,
        help=f"Path to the workbook (default: {settings.excel_file})",
    )
    args = parser.parse_args()

    setup_observability()
    engine = create_db_engine(settings)
    logger.info("Target database: %s", engine.url.render_as_string(hide_password=True))

    report = load_excel_to_db(engine, args.excel, settings=settings)
    logger.info("--- Load complete: %s ---", report)

    # Sanity check: top scorer straight from the DB (expected: SGA, 2485 pts).
    with engine.connect() as conn:
        name, pts = conn.execute(
            text("SELECT name, pts FROM players ORDER BY pts DESC LIMIT 1")
        ).one()
    logger.info("Sanity check — top scorer: %s (%d pts)", name, pts)


if __name__ == "__main__":
    main()
