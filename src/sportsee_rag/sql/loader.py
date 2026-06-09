"""Excel -> Pydantic -> SQL loader for the NBA stats workbook (Phase 3).

Reads the two source sheets ("Equipe", "Données NBA"), validates every row
through Pydantic *before* anything touches the database (fail fast, no partial
load), then inserts everything in a single transaction.

Workbook quirks handled here (see plan / inspection notes):
- header row of "Données NBA" is the 2nd line (``header=1``);
- the "3PM" column header was parsed by Excel as ``datetime.time(15:00)`` ->
  renamed positionally;
- trailing ``Unnamed: *`` columns are dropped;
- stats are season **totals** (only ``Min`` is per-game) — loaded as-is.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from .schema import Base, Player, Team

logger = logging.getLogger(__name__)

PLAYERS_SHEET = "Données NBA"
TEAMS_SHEET = "Equipe"

# Excel header -> DB column. "3PM" is restored positionally before this rename.
EXCEL_TO_DB: dict[str, str] = {
    "Player": "name",
    "Team": "team_code",
    "Age": "age",
    "GP": "gp",
    "W": "wins",
    "L": "losses",
    "Min": "min_per_game",
    "PTS": "pts",
    "FGM": "fgm",
    "FGA": "fga",
    "FG%": "fg_pct",
    "3PM": "fg3m",
    "3PA": "fg3a",
    "3P%": "fg3_pct",
    "FTM": "ftm",
    "FTA": "fta",
    "FT%": "ft_pct",
    "OREB": "oreb",
    "DREB": "dreb",
    "REB": "reb",
    "AST": "ast",
    "TOV": "tov",
    "STL": "stl",
    "BLK": "blk",
    "PF": "pf",
    "FP": "fantasy_points",
    "DD2": "dd2",
    "TD3": "td3",
    "+/-": "plus_minus",
    "OFFRTG": "offrtg",
    "DEFRTG": "defrtg",
    "NETRTG": "netrtg",
    "AST%": "ast_pct",
    "AST/TO": "ast_to",
    "AST RATIO": "ast_ratio",
    "OREB%": "oreb_pct",
    "DREB%": "dreb_pct",
    "REB%": "reb_pct",
    "TO RATIO": "to_ratio",
    "EFG%": "efg_pct",
    "TS%": "ts_pct",
    "USG%": "usg_pct",
    "PACE": "pace",
    "PIE": "pie",
    "POSS": "poss",
}


class TeamRow(BaseModel):
    """One validated row of the "Equipe" sheet."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(pattern=r"^[A-Z]{3}$")
    name: str = Field(min_length=3)


class PlayerRow(BaseModel):
    """One validated row of the "Données NBA" sheet (identity + season stats)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=2)
    team_code: str = Field(pattern=r"^[A-Z]{3}$")
    age: int = Field(ge=16, le=50)

    gp: int = Field(ge=1, le=82)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    min_per_game: float = Field(ge=0, le=48)

    pts: int = Field(ge=0)
    fgm: int = Field(ge=0)
    fga: int = Field(ge=0)
    fg_pct: float = Field(ge=0, le=100)
    fg3m: int = Field(ge=0)
    fg3a: int = Field(ge=0)
    fg3_pct: float = Field(ge=0, le=100)
    ftm: int = Field(ge=0)
    fta: int = Field(ge=0)
    ft_pct: float = Field(ge=0, le=100)

    oreb: int = Field(ge=0)
    dreb: int = Field(ge=0)
    reb: int = Field(ge=0)
    ast: int = Field(ge=0)
    tov: int = Field(ge=0)
    stl: int = Field(ge=0)
    blk: int = Field(ge=0)
    pf: int = Field(ge=0)

    fantasy_points: int = Field(ge=0)
    dd2: int = Field(ge=0)
    td3: int = Field(ge=0)
    plus_minus: float
    offrtg: float = Field(ge=0)
    defrtg: float = Field(ge=0)
    netrtg: float
    ast_pct: float = Field(ge=0, le=100)
    ast_to: float = Field(ge=0)
    ast_ratio: float = Field(ge=0)
    oreb_pct: float = Field(ge=0, le=100)
    dreb_pct: float = Field(ge=0, le=100)
    reb_pct: float = Field(ge=0, le=100)
    to_ratio: float = Field(ge=0)
    # EFG%/TS% can legitimately exceed 100 (3-point weighting / free throws):
    # e.g. a 1-game player shooting only made threes -> EFG% = 125.
    efg_pct: float = Field(ge=0)
    ts_pct: float = Field(ge=0)
    usg_pct: float = Field(ge=0, le=100)
    pace: float = Field(ge=0)
    pie: float
    poss: int = Field(ge=0)

    @model_validator(mode="after")
    def check_internal_consistency(self) -> PlayerRow:
        """Cross-field sanity checks (cheap referential integrity on the row).

        ``oreb + dreb == reb`` is deliberately NOT enforced: the source data
        violates it in 207/569 rows (e.g. Jokić 203+693=896 vs reb=889) — the
        workbook's totals are internally inconsistent (documented in RAPPORT),
        and the loader's job is to load the source faithfully, not to fix it.
        """
        if self.wins + self.losses != self.gp:
            raise ValueError(f"wins+losses != gp ({self.wins}+{self.losses} != {self.gp})")
        if self.fgm > self.fga or self.fg3m > self.fg3a or self.ftm > self.fta:
            raise ValueError("made shots exceed attempts (fgm/fg3m/ftm vs fga/fg3a/fta)")
        return self


@dataclass(frozen=True)
class LoadReport:
    """Counts of what was inserted, returned by ``load_excel_to_db``."""

    teams: int
    players: int


def create_db_engine(settings: Settings | None = None) -> Engine:
    """Create the SQLAlchemy engine (Supabase/PostgreSQL or local SQLite).

    For the SQLite fallback the parent directory is created on the fly.
    """
    settings = settings or get_settings()
    url = settings.sqlalchemy_url
    if url.startswith("sqlite:///"):
        settings.sqlite_db_file.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def read_team_rows(excel_path: Path) -> list[TeamRow]:
    """Read and validate the "Equipe" sheet."""
    df = pd.read_excel(excel_path, sheet_name=TEAMS_SHEET)
    df.columns = ["code", "name"]  # "Code", "Nom complet de l'équipe"
    return _validate_rows(df, TeamRow, sheet=TEAMS_SHEET)


def read_player_rows(excel_path: Path) -> list[PlayerRow]:
    """Read and validate the "Données NBA" sheet (with all workbook quirks)."""
    df = pd.read_excel(excel_path, sheet_name=PLAYERS_SHEET, header=1)
    # Drop trailing parasite columns, restore the time-typed "3PM" header.
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [c if isinstance(c, str) else "3PM" for c in df.columns]
    missing = set(EXCEL_TO_DB) - set(df.columns)
    if missing:
        raise ValueError(f"{PLAYERS_SHEET}: missing expected columns: {sorted(missing)}")
    df = df.rename(columns=EXCEL_TO_DB)[list(EXCEL_TO_DB.values())]
    return _validate_rows(df, PlayerRow, sheet=PLAYERS_SHEET)


def _validate_rows(df: pd.DataFrame, model: type[BaseModel], *, sheet: str) -> list:
    """Validate every row of ``df`` against ``model``; fail with a clear digest.

    All rows are checked before raising, so one bad cell doesn't hide others.
    """
    rows, errors = [], []
    for index, record in enumerate(df.to_dict(orient="records")):
        try:
            rows.append(model(**record))
        except ValidationError as exc:
            errors.append(f"row {index + 1}: {exc.errors()[0].get('msg', exc)}")
    if errors:
        digest = "; ".join(errors[:5])
        raise ValueError(
            f"{sheet}: {len(errors)} invalid row(s) — first errors: {digest}"
        )
    return rows


def load_excel_to_db(
    engine: Engine,
    excel_path: Path | None = None,
    *,
    settings: Settings | None = None,
) -> LoadReport:
    """Validate the workbook then (re)load it into the database.

    The schema is dropped and recreated on every run (single-run entrypoint,
    same philosophy as ``build_index.py``): the Excel file stays the single
    source of truth, the DB is a regenerable artifact.

    ⚠️ On PostgreSQL, ``drop_all`` can block or fail if another client (the
    Streamlit app, an idle pooler connection) holds the tables — stop the app
    before reloading. Acceptable for this single-operator project.
    """
    settings = settings or get_settings()
    excel_path = excel_path or settings.excel_file

    team_rows = read_team_rows(excel_path)
    player_rows = read_player_rows(excel_path)

    # Referential integrity before touching the DB.
    known_codes = {t.code for t in team_rows}
    orphans = sorted({p.team_code for p in player_rows} - known_codes)
    if orphans:
        raise ValueError(f"players reference unknown team codes: {orphans}")

    # players.name is the primary key: fail with a readable message, not an
    # IntegrityError halfway through the insert.
    duplicates = sorted(n for n, c in Counter(p.name for p in player_rows).items() if c > 1)
    if duplicates:
        raise ValueError(f"duplicate player names in workbook: {duplicates}")

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(Team(code=t.code, name=t.name) for t in team_rows)
        # PlayerRow fields map 1:1 onto the Player columns.
        session.add_all(Player(**row.model_dump()) for row in player_rows)
        session.commit()

    report = LoadReport(teams=len(team_rows), players=len(player_rows))
    logger.info("Database loaded: %d teams, %d players", report.teams, report.players)
    return report
