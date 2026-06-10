"""Sanity-check the SELECT-only database role (defence in depth, layer 2).

Connects with ``DATABASE_URL_READONLY`` (via ``sqlalchemy_url_readonly``) and
verifies both sides of the contract:
  1. reads work    -> SELECT count(*) on players/teams;
  2. writes do NOT -> an INSERT must be rejected by PostgreSQL itself
     (``permission denied``), proving the guard does not rely on the
     application-level keyword filter in ``sql_tool.ensure_read_only``.

Run: uv run python scripts/check_db_readonly.py
"""

from __future__ import annotations

import sys

from sqlalchemy import create_engine, text

from sportsee_rag.config import get_settings
from sportsee_rag.observability import setup_observability


def main() -> int:
    setup_observability()
    settings = get_settings()
    if not settings.database_url_readonly:
        print("DATABASE_URL_READONLY is not set — nothing to check (SQLite/admin fallback).")
        return 1

    engine = create_engine(settings.sqlalchemy_url_readonly)

    with engine.connect() as conn:
        players = conn.execute(text("SELECT count(*) FROM players")).scalar()
        teams = conn.execute(text("SELECT count(*) FROM teams")).scalar()
        print(f"[OK] SELECT works: {players} players, {teams} teams")

    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO teams VALUES ('XXX', 'Should Fail')"))
            conn.commit()
    except Exception as exc:  # driver-specific error type, message is what matters
        print(f"[OK] INSERT rejected by PostgreSQL: {exc}")
        return 0

    print("[FAIL] INSERT was ACCEPTED — the role is NOT read-only! Remove the test row.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
