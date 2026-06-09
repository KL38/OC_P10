"""Tests for the NL->SQL tool: output sanitisation, read-only guard, execution.

The LLM is the conftest ``FakeLLM`` (canned SQL string) and the database an
in-memory SQLite filled with two players — no network, no key.
"""

from __future__ import annotations

import pytest
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from sportsee_rag.sql.schema import Base, Player, Team
from sportsee_rag.sql.sql_tool import (
    SqlTool,
    SqlToolError,
    ensure_read_only,
    sanitize_query,
)
from sqlalchemy.orm import Session

from tests.conftest import FakeLLM

STATS_DEFAULTS = dict(
    gp=70, wins=40, losses=30, min_per_game=30.0,
    pts=0, fgm=0, fga=0, fg_pct=0.0, fg3m=0, fg3a=0, fg3_pct=0.0,
    ftm=0, fta=0, ft_pct=0.0, oreb=0, dreb=0, reb=0, ast=0, tov=0,
    stl=0, blk=0, pf=0, fantasy_points=0, dd2=0, td3=0, plus_minus=0.0,
    offrtg=0.0, defrtg=0.0, netrtg=0.0, ast_pct=0.0, ast_to=0.0,
    ast_ratio=0.0, oreb_pct=0.0, dreb_pct=0.0, reb_pct=0.0, to_ratio=0.0,
    efg_pct=0.0, ts_pct=0.0, usg_pct=0.0, pace=0.0, pie=0.0, poss=0,
)


@pytest.fixture
def db() -> SQLDatabase:
    """In-memory DB with one team and two players (deterministic answers)."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Team(code="OKC", name="Oklahoma City Thunder"))
        session.add(
            Player(name="Player One", team_code="OKC", age=26,
                   **{**STATS_DEFAULTS, "pts": 2485})
        )
        session.add(
            Player(name="Player Two", team_code="OKC", age=30,
                   **{**STATS_DEFAULTS, "pts": 1500})
        )
        session.commit()
    return SQLDatabase(engine)


class TestSanitizeQuery:
    def test_strips_markdown_fences_and_semicolon(self):
        raw = "```sql\nSELECT 1;\n```"
        assert sanitize_query(raw) == "SELECT 1"

    def test_fence_with_space_before_language_tag(self):
        assert sanitize_query("``` sql\nSELECT 1\n```") == "SELECT 1"

    def test_plain_query_untouched(self):
        assert sanitize_query("SELECT name FROM players") == "SELECT name FROM players"


class TestEnsureReadOnly:
    def test_select_accepted(self):
        ensure_read_only("SELECT * FROM players")

    def test_cte_accepted(self):
        ensure_read_only("WITH top AS (SELECT pts FROM stats) SELECT * FROM top")

    def test_keyword_inside_player_name_accepted(self):
        # 'Grant' is a real NBA surname and 'grant' a forbidden keyword: the
        # guard must scan SQL tokens, not literal contents.
        ensure_read_only("SELECT pts FROM players WHERE name = 'Jerami Grant'")

    def test_semicolon_inside_literal_accepted(self):
        ensure_read_only("SELECT * FROM players WHERE name = 'a;b'")

    @pytest.mark.parametrize(
        "query",
        [
            "INSERT INTO players VALUES (1)",
            "DROP TABLE players",
            "SELECT 1; DROP TABLE players",
            "SELECT * FROM players WHERE id IN (DELETE FROM stats)",
            # PostgreSQL writable CTE: caught by both `delete` and `returning`.
            "WITH x AS (DELETE FROM stats RETURNING player_id) SELECT * FROM x",
            # Comment must not hide the second statement's separator.
            "SELECT 1 -- comment\n; DROP TABLE players",
        ],
    )
    def test_writes_and_multistatements_rejected(self, query):
        with pytest.raises(SqlToolError):
            ensure_read_only(query)


class TestSqlToolRun:
    def test_runs_generated_select(self, settings, db):
        llm = FakeLLM(answer="SELECT pts FROM players ORDER BY pts DESC LIMIT 1;")
        tool = SqlTool(settings=settings, llm=llm, db=db)
        result = tool.run("Combien de points pour le meilleur scoreur ?")
        assert result.error is None
        assert "2485" in result.result
        assert result.query.startswith("SELECT")

    def test_fenced_output_is_sanitized_then_run(self, settings, db):
        llm = FakeLLM(answer="```sql\nSELECT COUNT(*) FROM players\n```")
        tool = SqlTool(settings=settings, llm=llm, db=db)
        result = tool.run("Combien de joueurs ?")
        assert result.error is None
        assert "2" in result.result

    def test_write_attempt_is_blocked(self, settings, db):
        llm = FakeLLM(answer="DROP TABLE players")
        tool = SqlTool(settings=settings, llm=llm, db=db)
        result = tool.run("Supprime tout")
        assert result.error is not None
        assert "SELECT" in result.error
        # The table is still there.
        assert "2" in db.run("SELECT COUNT(*) FROM players")

    def test_broken_sql_reported_not_raised(self, settings, db):
        llm = FakeLLM(answer="SELECT nope FROM nowhere")
        tool = SqlTool(settings=settings, llm=llm, db=db)
        result = tool.run("Question quelconque")
        assert result.error is not None
        assert "execution failed" in result.error
