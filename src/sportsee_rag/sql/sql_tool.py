"""NL -> SQL tool over the NBA stats database (Phase 3, Brief Étape 2).

Pipeline: question -> LLM generates a SELECT (few-shot prompt built from the
live schema via LangChain ``SQLDatabase.get_table_info``) -> read-only guard ->
execution through ``SQLDatabase.run`` -> typed ``SqlResult``.

LangChain's ``SQLDatabase`` is imposed by the brief; generation goes through
our resilient ``MistralLLM`` wrapper (retry/backoff on the free-tier 429s)
rather than a second LangChain LLM client.

The few-shot examples deliberately do NOT reuse the eval questions
(``eval/questions.yaml``): seeding the prompt with the very questions RAGAS
scores would leak the benchmark into the system and inflate the after-side of
the before/after comparison. Same query *shapes*, different columns.
"""

from __future__ import annotations

import logging
import re

import logfire
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from ..config import Settings, get_settings
from ..llm.client import MistralError, MistralLLM
from ..models import SqlResult

logger = logging.getLogger(__name__)

# Standalone keywords that must never appear in a generated query (write or
# side-effecting statements). Word-boundary matched, case-insensitive, scanned
# AFTER string literals and comments are stripped (so a player named 'Jerami
# Grant' or a quoted ';' cannot trip the guard). ``returning`` blocks the
# PostgreSQL writable-CTE path (``WITH x AS (DELETE ... RETURNING) SELECT``).
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|attach"
    r"|pragma|copy|returning|merge|call|execute|do)\b",
    re.IGNORECASE,
)

SQL_PROMPT_TEMPLATE = """Tu es un expert SQL. Génère UNE requête {dialect} répondant à la question.

SCHÉMA DE LA BASE :
{table_info}

NOTES SUR LES DONNÉES :
- Les stats de `players` sont des TOTAUX sur la saison (pts, reb, ast, stl...),
  sauf `min_per_game` (moyenne par match) et les colonnes `*_pct`/ratios/ratings.
- `players.team_code` = code 3 lettres ; le nom complet de l'équipe est dans `teams.name`.
- Pour un "meilleur pourcentage", filtre sur un volume minimal de tentatives si la
  question en mentionne un (ex. `fg3a >= 200`).

RÈGLES :
- Uniquement du SELECT (jamais d'écriture).
- Join équipe : `players.team_code = teams.code`.
- Ajoute `LIMIT {default_limit}` aux requêtes non agrégées.
- Réponds UNIQUEMENT avec la requête SQL, sans explication ni balises markdown.

EXEMPLES :
Question : Quel joueur a le plus d'interceptions ?
SQL : SELECT name, stl FROM players ORDER BY stl DESC LIMIT 1

Question : Combien de joueurs ont disputé au moins 70 matchs ?
SQL : SELECT COUNT(*) FROM players WHERE gp >= 70

Question : Quel est le meilleur pourcentage aux lancers francs parmi les joueurs ayant tenté au moins 300 lancers ?
SQL : SELECT name, ft_pct FROM players WHERE fta >= 300 ORDER BY ft_pct DESC LIMIT 1

Question : Quel est le total de passes décisives des joueurs des Boston Celtics ?
SQL : SELECT SUM(p.ast) FROM players p JOIN teams t ON t.code = p.team_code WHERE t.name = 'Boston Celtics'

Question : Quel joueur des Denver Nuggets a le plus de contres ?
SQL : SELECT p.name, p.blk FROM players p JOIN teams t ON t.code = p.team_code WHERE t.name = 'Denver Nuggets' ORDER BY p.blk DESC LIMIT 1

Question : {question}
SQL :"""


class SqlToolError(RuntimeError):
    """Raised when a generated query is rejected by the read-only guard."""


def sanitize_query(raw: str) -> str:
    """Normalise the LLM output into a bare SQL statement.

    Strips markdown fences, a leading ``sql`` language tag (with or without a
    separating space), surrounding whitespace and the trailing semicolon.
    """
    query = raw.strip()
    query = re.sub(r"^```\s*(?:sql)?\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s*```$", "", query)
    return query.strip().rstrip(";").strip()


def _strip_literals_and_comments(query: str) -> str:
    """Return a scan-only copy with comments removed and string literals emptied.

    The guard must inspect what the SQL engine will *parse*, not raw text:
    comments are ignored by the engine, and literal contents are data — a
    player named 'Jerami Grant' or a quoted ';' must not trip the keyword scan.
    The executed query itself is left untouched.
    """
    query = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    query = re.sub(r"--[^\n]*", " ", query)
    query = re.sub(r"'(?:[^']|'')*'", "''", query)  # '' = escaped quote inside a literal
    return query


def ensure_read_only(query: str) -> None:
    """Reject anything that is not a single SELECT/CTE statement.

    Application-level guard only — defence in depth: the runtime DB role
    should additionally be SELECT-only (Supabase), which no string-level
    bypass can defeat.
    """
    if not re.match(r"^(select|with)\b", query, re.IGNORECASE):
        raise SqlToolError(f"Only SELECT queries are allowed, got: {query[:80]!r}")
    scannable = _strip_literals_and_comments(query)
    if ";" in scannable:
        raise SqlToolError("Multiple SQL statements are not allowed")
    forbidden = _FORBIDDEN.search(scannable)
    if forbidden:
        raise SqlToolError(f"Forbidden SQL keyword {forbidden.group(0)!r} in query")


class SqlTool:
    """Answer natural-language questions about NBA stats through SQL.

    Both the database handle and the LLM are injectable for tests
    (in-memory SQLite + canned generator, no network).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm: MistralLLM | None = None,
        db: SQLDatabase | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or MistralLLM(self._settings)
        if db is None:
            # Engine built explicitly (instead of SQLDatabase.from_uri) so it
            # can be instrumented: one Logfire span per executed query.
            # Connects with the SELECT-only role (real read-only enforcement;
            # ensure_read_only is only the first, string-level layer).
            engine = create_engine(self._settings.sqlalchemy_url_readonly)
            logfire.instrument_sqlalchemy(engine=engine)
            db = SQLDatabase(
                engine,
                sample_rows_in_table_info=self._settings.sql_sample_rows_in_table_info,
            )
        self._db = db

    def generate_query(self, question: str) -> str:
        """Ask the LLM for a SQL query answering ``question`` (not executed)."""
        prompt = SQL_PROMPT_TEMPLATE.format(
            dialect=self._db.dialect,
            table_info=self._db.get_table_info(),
            default_limit=self._settings.sql_default_limit,
            question=question,
        )
        raw = self._llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        return sanitize_query(raw)

    def run(self, question: str) -> SqlResult:
        """Full round-trip: generate, guard, execute.

        Failures land in ``SqlResult.error`` (not raised) so the agent can
        report them gracefully instead of aborting the whole run. The whole
        round-trip is wrapped in a span: the SQL-generation LLM call goes
        through our own ``MistralLLM`` wrapper, invisible to the Pydantic AI
        instrumentation — without this span it would be a black box in the
        trace. Errors are recorded as attributes (returned, never raised).
        """
        with logfire.span("SQL tool run", question=question) as span:
            try:
                query = self.generate_query(question)
            except MistralError as exc:
                logger.error("SQL generation failed: %s", exc)
                span.set_attribute("error", str(exc))
                return SqlResult(query="", error=f"SQL generation failed: {exc}")
            span.set_attribute("generated_sql", query)

            try:
                ensure_read_only(query)
                result = self._db.run(query)
            except SqlToolError as exc:
                logger.warning("Generated query rejected: %s", exc)
                span.set_attribute("error", str(exc))
                return SqlResult(query=query, error=str(exc))
            except Exception as exc:  # noqa: BLE001 - driver error types vary by backend
                logger.warning("SQL execution failed: %s", exc)
                span.set_attribute("error", str(exc))
                return SqlResult(query=query, error=f"SQL execution failed: {exc}")

            logger.info("SQL tool ran query: %s", query)
            return SqlResult(query=query, result=str(result))
