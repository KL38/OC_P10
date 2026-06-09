"""RAGAS evaluation harness — baseline & enriched runs share this script.

Pipeline:
  1. GENERATE : run each question from ``questions.yaml`` through our RAG pipeline,
     collecting (question, answer, retrieved_contexts, reference).
  2. SCORE    : score four metrics — faithfulness, answer relevancy, context
     precision (with reference), context recall.
  3. REPORT   : write a JSON report + markdown table (overall and per category) to
     ``eval/reports/``, plus the raw predictions for re-scoring / comparison.

Why the *modern* RAGAS stack (ported from project P9):
  The classic ``evaluate()`` + ``ragas.metrics`` path routes the judge through
  ``LangchainLLMWrapper``, whose multi-completion handling breaks with
  ``ChatMistralAI`` — ``answer_relevancy`` (strictness>1) raises
  ``TypeError: dict += dict`` and yields NaN. The modern path
  (``ragas.metrics.collections`` + ``InstructorLLM``) generates the N questions as
  N separate ``instructor`` calls, so **strictness=3 works**. We build the judge by
  patching the Mistral client ourselves (ragas' ``llm_factory(provider='mistral')``
  is itself broken), and score each metric per-sample via ``metric.ascore`` —
  sequentially, with 429 backoff, to stay under the free-tier rate limits.

Judge limitation (documented): the judge is ``mistral-small`` (free tier), not a
frontier model — scores are indicative; the *delta* baseline->enriched is the signal.

Usage:
  uv run python eval/evaluate_ragas.py --label baseline --limit 2   # cheap smoke
  uv run python eval/evaluate_ragas.py --label baseline             # full run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import instructor
import pandas as pd
import yaml
from mistralai import Mistral

from sportsee_rag.config import Settings, get_settings
from sportsee_rag.models import AskRequest
from sportsee_rag.observability import setup_observability
from sportsee_rag.rag.pipeline import answer_question
from sportsee_rag.retrieval.vector_store import VectorStoreManager

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.embeddings import LiteLLMEmbeddings
    from ragas.llms.base import InstructorLLM, InstructorModelArgs
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

logger = logging.getLogger(__name__)

QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"
REPORTS_DIR = Path(__file__).parent / "reports"
JUDGE_EMBED_MODEL = "mistral/mistral-embed"  # litellm id (answer_relevancy embeddings)
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
# Excluded from the headline average: RAGAS scores refusals/out-of-coverage poorly
# (noisy, even anti-correlated with correct behaviour) — reported on its own line.
AGGREGATE_EXCLUDE = {"hors_couverture"}


# --- 1. generation --------------------------------------------------------

def load_questions(limit: int | None = None) -> list[dict]:
    data = yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions = data["questions"]
    return questions[:limit] if limit else questions


def generate_predictions(
    questions: list[dict], *, manager: VectorStoreManager, settings: Settings, throttle: float
) -> list[dict]:
    """Run each question through our RAG pipeline and build RAGAS samples."""
    samples: list[dict] = []
    for i, q in enumerate(questions, start=1):
        logger.info("Generate %d/%d [%s] %s", i, len(questions), q["id"], q["question"])
        answer = answer_question(AskRequest(question=q["question"]), manager=manager, settings=settings)
        samples.append({
            "id": q["id"],
            "category": q["category"],
            "expected": q.get("expected"),
            "user_input": q["question"],
            "response": answer.answer,
            "retrieved_contexts": answer.contexts or ["(aucun contexte récupéré)"],
            "reference": q.get("reference", ""),
        })
        time.sleep(throttle)  # be gentle with the Mistral free tier
    return samples


# --- 2. scoring (modern RAGAS: instructor judge + collections, async) ------

def _is_rate_limit(exc: Exception) -> bool:
    """True if the exception is a Mistral 429 rate-limit error."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429
    return "429" in str(exc) or "rate limit" in str(exc).lower()


async def _ascore(metric, kwargs: dict, *, max_attempts: int = 5, base_wait: float = 4.0) -> float:
    """Run ``metric.ascore(**kwargs)``, retrying on failure; NaN only as last resort.

    Two failure modes are retried: a Mistral 429 (long exponential backoff) and the
    judge returning unparseable structured output (``InstructorRetryException`` —
    stochastic, so a short retry usually recovers it). One metric failing on one
    sample must not abort the run, so we yield NaN after exhausting the attempts.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            result = await metric.ascore(**kwargs)
            return float(result.value)
        except Exception as exc:  # noqa: BLE001
            if attempt == max_attempts:
                logger.warning("Metric %s gave up after %d attempts (%s) -> NaN",
                               metric.name, max_attempts, type(exc).__name__)
                return float("nan")
            # 429 -> long exponential backoff; schema/other errors -> short retry.
            wait = base_wait * (2 ** (attempt - 1)) if _is_rate_limit(exc) else 2.0
            logger.warning("Metric %s failed (%s), retry %d/%d in %.0fs",
                           metric.name, type(exc).__name__, attempt, max_attempts, wait)
            await asyncio.sleep(wait)
    return float("nan")


async def score_samples(samples: list[dict], metrics_spec: list[tuple]) -> list[dict]:
    """Score every sample on every metric, sequentially (free-tier friendly)."""
    rows: list[dict] = []
    for i, sample in enumerate(samples, start=1):
        row = {}
        for metric, keys in metrics_spec:
            row[metric.name] = await _ascore(metric, {k: sample[k] for k in keys})
        rows.append(row)
        logger.info("scored %d/%d", i, len(samples))
    return rows


def build_judge(settings: Settings, judge_model: str):
    """Build the modern RAGAS judge (instructor-patched Mistral) + embeddings.

    Bypasses ragas' broken ``llm_factory(provider='mistral')`` by patching the
    Mistral client directly with ``instructor.from_mistral`` (async).
    """
    os.environ["MISTRAL_API_KEY"] = settings.mistral_api_key  # litellm / instructor read it
    patched = instructor.from_mistral(Mistral(api_key=settings.mistral_api_key), use_async=True)
    # max_tokens=4096 (vs the 1024 default): the judge's structured JSON gets truncated on
    # long answers, the likely cause of the faithfulness InstructorRetryException -> NaN.
    judge_llm = InstructorLLM(
        client=patched, model=judge_model, provider="mistral",
        model_args=InstructorModelArgs(max_tokens=4096),
    )
    judge_emb = LiteLLMEmbeddings(model=JUDGE_EMBED_MODEL, api_key=settings.mistral_api_key)
    return judge_llm, judge_emb


def build_metrics_spec(judge_llm, judge_emb, *, strictness: int) -> list[tuple]:
    """The four metrics with the exact sample keys each one consumes."""
    return [
        (Faithfulness(llm=judge_llm, name="faithfulness"),
         ["user_input", "response", "retrieved_contexts"]),
        (AnswerRelevancy(llm=judge_llm, embeddings=judge_emb, strictness=strictness, name="answer_relevancy"),
         ["user_input", "response"]),
        (ContextPrecisionWithReference(llm=judge_llm, name="context_precision"),
         ["user_input", "reference", "retrieved_contexts"]),
        (ContextRecall(llm=judge_llm, name="context_recall"),
         ["user_input", "retrieved_contexts", "reference"]),
    ]


# --- 3. reporting ---------------------------------------------------------

def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join("---" for _ in headers) + " |\n"
    for row in rows:
        out += "| " + " | ".join(str(c) for c in row) + " |\n"
    return out


def _fmt(v: Any) -> str:
    """Format a metric value for markdown (NaN -> em dash)."""
    return "—" if pd.isna(v) else f"{float(v):.3f}"


def build_markdown(label: str, stamp: str, judge_model: str, df: pd.DataFrame, by_cat: pd.DataFrame) -> str:
    """Assemble the full markdown report (overall + per category + per question)."""
    in_scope = df[~df["category"].isin(AGGREGATE_EXCLUDE)]
    overall = {m: in_scope[m].mean() for m in METRIC_NAMES}
    n_nan = {m: int(df[m].isna().sum()) for m in METRIC_NAMES}

    md = f"# Rapport RAGAS — {label}\n\n"
    md += f"- Date : {stamp}\n- Juge : `{judge_model}`\n- Questions : {len(df)}\n"
    md += f"- NaN par métrique (échecs du juge, ignorés dans les moyennes) : "
    md += ", ".join(f"{m}={n_nan[m]}" for m in METRIC_NAMES) + "\n\n"

    md += "## Scores globaux (hors `hors_couverture`)\n\n"
    md += _md_table(["métrique", "score moyen"], [[m, _fmt(overall[m])] for m in METRIC_NAMES])
    md += (
        "\n> `hors_couverture` (questions sans réponse dans la base) est **exclu de la "
        "moyenne** : RAGAS n'évalue pas correctement les refus — ses scores y sont bruités, "
        "voire anti-corrélés au bon comportement. Voir sa ligne dédiée ci-dessous.\n"
    )

    md += "\n## Scores par catégorie\n\n"
    md += _md_table(
        ["catégorie"] + METRIC_NAMES,
        [[cat] + [_fmt(by_cat.loc[cat, m]) for m in METRIC_NAMES] for cat in by_cat.index],
    )

    md += "\n## Détail par question\n\n"
    md += _md_table(
        ["id", "catégorie", "attendu"] + METRIC_NAMES,
        [[r["id"], r["category"], r["expected"]] + [_fmt(r[m]) for m in METRIC_NAMES]
         for r in df.to_dict(orient="records")],
    )
    return md


def write_reports(samples: list[dict], score_rows: list[dict], *, label: str, judge_model: str) -> Path:
    """Write JSON + markdown reports and the raw predictions. Returns the md path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta = pd.DataFrame([{k: s[k] for k in ("id", "category", "expected")} for s in samples])
    df = pd.concat([meta, pd.DataFrame(score_rows)], axis=1)

    in_scope = df[~df["category"].isin(AGGREGATE_EXCLUDE)]
    overall = {m: round(float(in_scope[m].mean()), 3) for m in METRIC_NAMES}
    by_cat = df.groupby("category")[METRIC_NAMES].mean().round(3)

    report = {
        "label": label,
        "timestamp": stamp,
        "judge_model": judge_model,
        "n_questions": len(samples),
        "aggregate_excludes": sorted(AGGREGATE_EXCLUDE),
        "overall": overall,
        "by_category": {cat: row.to_dict() for cat, row in by_cat.iterrows()},
        "per_question": df.to_dict(orient="records"),
    }
    (REPORTS_DIR / f"{label}_{stamp}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (REPORTS_DIR / f"{label}_{stamp}_predictions.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = REPORTS_DIR / f"{label}_{stamp}.md"
    md_path.write_text(build_markdown(label, stamp, judge_model, df, by_cat), encoding="utf-8")
    return md_path


# --- entrypoint -----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAGAS evaluation harness (modern stack).")
    parser.add_argument("--label", default="baseline", help="Run label (baseline / enriched).")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    parser.add_argument("--judge-model", default="mistral-small-2506",
                        help="Judge model. small-2506 = best free-tier limits (RPS 5, TPM 2.25M); "
                             "with max_tokens=4096 the faithfulness truncation NaN should be largely fixed.")
    parser.add_argument("--strictness", type=int, default=3,
                        help="answer_relevancy: questions generated per answer (3-5 ideal).")
    parser.add_argument("--throttle", type=float, default=1.0, help="Seconds between generations.")
    args = parser.parse_args()

    setup_observability()
    settings = get_settings()

    questions = load_questions(args.limit)
    logger.info("Loaded %d questions (label=%s, judge=%s, strictness=%d)",
                len(questions), args.label, args.judge_model, args.strictness)

    manager = VectorStoreManager(settings=settings)
    if manager.index is None:
        raise SystemExit("Index introuvable — lance d'abord `uv run python scripts/build_index.py`.")

    logger.info("--- Generation ---")
    samples = generate_predictions(questions, manager=manager, settings=settings, throttle=args.throttle)

    logger.info("--- Scoring (RAGAS modern stack) ---")
    judge_llm, judge_emb = build_judge(settings, args.judge_model)
    metrics_spec = build_metrics_spec(judge_llm, judge_emb, strictness=args.strictness)
    score_rows = asyncio.run(score_samples(samples, metrics_spec))

    md_path = write_reports(samples, score_rows, label=args.label, judge_model=args.judge_model)
    logger.info("--- Done --- report: %s", md_path)
    df = pd.DataFrame(score_rows)
    print(f"\nRapport : {md_path}")
    print({m: round(float(df[m].mean()), 3) for m in METRIC_NAMES})


if __name__ == "__main__":
    main()
