"""SQLAlchemy schema for the NBA stats database (Phase 3).

Two tables, mirroring the two source sheets 1:1 (decided 2026-06-10 — academic
single-season scope, keep it simple):
    teams   <- sheet "Equipe"        (code -> full name, 30 rows)
    players <- sheet "Données NBA"   (one row per player: identity + season stats)

Design notes:
- ``name`` is the primary key: verified unique in the source, and no query
  ever needs a surrogate id — one less column for the LLM to misuse.
- ``team_code`` and ``age`` are season-dependent attributes: they live here
  only because the scope is a single season. A multi-season model would move
  them (with a ``season`` key) to a player-season fact table, keeping just the
  identity in ``players`` (documented limitation, see RAPPORT).
- The "Analyse" sheet is deliberately NOT loaded: everything in it derives
  from these tables with plain SQL, and pre-computed aggregates would let the
  system answer by lookup instead of demonstrating SQL generation. It is
  reused as a test oracle instead.
- Column names are snake_case and SQL-safe (``3P%`` -> ``fg3_pct``, ``+/-`` ->
  ``plus_minus``) because the LLM has to quote them in generated queries.

⚠️ Semantics (verified against the workbook, the data dictionary is wrong on
this): all counting stats are **season totals** (SGA pts=2485), only
``min_per_game`` is a per-game average. The SQL tool prompt restates this.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all SportSee SQL models."""


class Team(Base):
    """An NBA team (3-letter code + full name), from the "Equipe" sheet."""

    __tablename__ = "teams"

    code: Mapped[str] = mapped_column(String(3), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    players: Mapped[list[Player]] = relationship(back_populates="team")


class Player(Base):
    """One player's identity and season stat line, from the "Données NBA" sheet.

    Stats are season totals, except ``min_per_game`` and the ``*_pct``/ratio
    columns.
    """

    __tablename__ = "players"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    team_code: Mapped[str] = mapped_column(ForeignKey("teams.code"), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)

    # --- volume ---
    gp: Mapped[int] = mapped_column(Integer, nullable=False)  # games played
    wins: Mapped[int] = mapped_column(Integer, nullable=False)
    losses: Mapped[int] = mapped_column(Integer, nullable=False)
    min_per_game: Mapped[float] = mapped_column(Float, nullable=False)

    # --- scoring (season totals) ---
    pts: Mapped[int] = mapped_column(Integer, nullable=False)
    fgm: Mapped[int] = mapped_column(Integer, nullable=False)
    fga: Mapped[int] = mapped_column(Integer, nullable=False)
    fg_pct: Mapped[float] = mapped_column(Float, nullable=False)
    fg3m: Mapped[int] = mapped_column(Integer, nullable=False)  # Excel col "3PM"
    fg3a: Mapped[int] = mapped_column(Integer, nullable=False)
    fg3_pct: Mapped[float] = mapped_column(Float, nullable=False)
    ftm: Mapped[int] = mapped_column(Integer, nullable=False)
    fta: Mapped[int] = mapped_column(Integer, nullable=False)
    ft_pct: Mapped[float] = mapped_column(Float, nullable=False)

    # --- rebounds / playmaking / defense (season totals) ---
    oreb: Mapped[int] = mapped_column(Integer, nullable=False)
    dreb: Mapped[int] = mapped_column(Integer, nullable=False)
    reb: Mapped[int] = mapped_column(Integer, nullable=False)
    ast: Mapped[int] = mapped_column(Integer, nullable=False)
    tov: Mapped[int] = mapped_column(Integer, nullable=False)
    stl: Mapped[int] = mapped_column(Integer, nullable=False)
    blk: Mapped[int] = mapped_column(Integer, nullable=False)
    pf: Mapped[int] = mapped_column(Integer, nullable=False)

    # --- misc / advanced ---
    fantasy_points: Mapped[int] = mapped_column(Integer, nullable=False)
    dd2: Mapped[int] = mapped_column(Integer, nullable=False)  # double-doubles
    td3: Mapped[int] = mapped_column(Integer, nullable=False)  # triple-doubles
    plus_minus: Mapped[float] = mapped_column(Float, nullable=False)
    offrtg: Mapped[float] = mapped_column(Float, nullable=False)
    defrtg: Mapped[float] = mapped_column(Float, nullable=False)
    netrtg: Mapped[float] = mapped_column(Float, nullable=False)
    ast_pct: Mapped[float] = mapped_column(Float, nullable=False)
    ast_to: Mapped[float] = mapped_column(Float, nullable=False)
    ast_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    oreb_pct: Mapped[float] = mapped_column(Float, nullable=False)
    dreb_pct: Mapped[float] = mapped_column(Float, nullable=False)
    reb_pct: Mapped[float] = mapped_column(Float, nullable=False)
    to_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    efg_pct: Mapped[float] = mapped_column(Float, nullable=False)
    ts_pct: Mapped[float] = mapped_column(Float, nullable=False)
    usg_pct: Mapped[float] = mapped_column(Float, nullable=False)
    pace: Mapped[float] = mapped_column(Float, nullable=False)
    pie: Mapped[float] = mapped_column(Float, nullable=False)  # player impact estimate
    poss: Mapped[int] = mapped_column(Integer, nullable=False)

    team: Mapped[Team] = relationship(back_populates="players")
