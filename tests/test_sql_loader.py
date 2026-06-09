"""Tests for the Excel -> Pydantic -> SQL loader.

Two layers:
- unit: ``PlayerRow`` validation (constraints + cross-field consistency);
- integration: full load of the real workbook into a temp SQLite DB, checked
  against the "Analyse" sheet used as an *oracle* (its pre-computed aggregates
  must be reproducible by SQL — that is exactly why it is not loaded as a
  table, see plan). Stats are season totals, so equality is exact.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from sportsee_rag.config import PROJECT_ROOT
from sportsee_rag.sql.loader import PlayerRow, load_excel_to_db

EXCEL_FILE = PROJECT_ROOT / "data" / "regular+NBA.xlsx"

VALID_ROW = dict(
    name="Test Player",
    team_code="OKC",
    age=25,
    gp=70, wins=40, losses=30, min_per_game=30.5,
    pts=1500, fgm=550, fga=1100, fg_pct=50.0,
    fg3m=100, fg3a=280, fg3_pct=35.7,
    ftm=300, fta=350, ft_pct=85.7,
    oreb=60, dreb=240, reb=300, ast=350, tov=150,
    stl=80, blk=40, pf=140,
    fantasy_points=3000, dd2=5, td3=0, plus_minus=4.2,
    offrtg=115.0, defrtg=110.0, netrtg=5.0,
    ast_pct=20.0, ast_to=2.3, ast_ratio=15.0,
    oreb_pct=2.0, dreb_pct=10.0, reb_pct=6.0, to_ratio=8.0,
    efg_pct=54.5, ts_pct=60.0, usg_pct=28.0,
    pace=100.0, pie=12.0, poss=4500,
)


class TestPlayerRowValidation:
    def test_valid_row_accepted(self):
        row = PlayerRow(**VALID_ROW)
        assert row.name == "Test Player"

    def test_bad_team_code_rejected(self):
        with pytest.raises(ValueError):
            PlayerRow(**{**VALID_ROW, "team_code": "Okc"})

    def test_negative_points_rejected(self):
        with pytest.raises(ValueError):
            PlayerRow(**{**VALID_ROW, "pts": -1})

    def test_wins_losses_must_sum_to_gp(self):
        with pytest.raises(ValueError, match="wins"):
            PlayerRow(**{**VALID_ROW, "wins": 41})

    def test_made_shots_cannot_exceed_attempts(self):
        with pytest.raises(ValueError, match="attempts"):
            PlayerRow(**{**VALID_ROW, "fgm": 1101})


@pytest.mark.skipif(not EXCEL_FILE.exists(), reason="workbook not present (data/ is gitignored)")
class TestLoadExcelToDb:
    """Integration: real workbook -> temp SQLite, checked against the Analyse oracle."""

    @pytest.fixture(scope="class")
    def engine(self, tmp_path_factory):
        engine = create_engine(f"sqlite:///{tmp_path_factory.mktemp('db') / 'nba.sqlite'}")
        report = load_excel_to_db(engine, EXCEL_FILE)
        assert (report.teams, report.players) == (30, 569)
        return engine

    def _scalar(self, engine, query: str):
        with engine.connect() as conn:
            return conn.execute(text(query)).scalar_one()

    def test_team_total_points_matches_analyse_sheet(self, engine):
        # Analyse sheet, per-team block: OKC total points = 9880.
        total = self._scalar(
            engine, "SELECT SUM(pts) FROM players WHERE team_code = 'OKC'"
        )
        assert total == 9880

    def test_team_player_count_matches_analyse_sheet(self, engine):
        # Analyse sheet: Miami Heat has 19 players.
        count = self._scalar(
            engine,
            "SELECT COUNT(*) FROM players p JOIN teams t ON t.code = p.team_code "
            "WHERE t.name = 'Miami Heat'",
        )
        assert count == 19

    def test_top_scorer_matches_analyse_sheet(self, engine):
        # Analyse sheet, top-15 block: SGA leads with 2485 points.
        with engine.connect() as conn:
            name, pts = conn.execute(
                text("SELECT name, pts FROM players ORDER BY pts DESC LIMIT 1")
            ).one()
        assert (name, pts) == ("Shai Gilgeous-Alexander", 2485)

    def test_eval_reference_top_rebounds(self, engine):
        # Eval reference (questions.yaml): Zubac, 1008 total rebounds.
        with engine.connect() as conn:
            name, reb = conn.execute(
                text("SELECT name, reb FROM players ORDER BY reb DESC LIMIT 1")
            ).one()
        assert (name, reb) == ("Ivica Zubac", 1008)

    def test_eval_reference_count_over_2000_points(self, engine):
        # Eval reference: 4 players scored more than 2000 points.
        assert self._scalar(engine, "SELECT COUNT(*) FROM players WHERE pts > 2000") == 4
