"""Ask the RAG pipeline a single question from the command line.

A small manual tester for the RAG chain (retrieve -> prompt -> answer) and a
handy demo entrypoint. Requires the index to be built first
(``uv run python scripts/build_index.py``).

Usage:
    uv run python scripts/ask.py "Quelles équipes ont impressionné en playoffs ?"
    uv run python scripts/ask.py "Quel joueur a le meilleur % à 3 points ?" --k 8
"""

from __future__ import annotations

import argparse

from sportsee_rag.models import AskRequest
from sportsee_rag.observability import setup_observability
from sportsee_rag.rag.pipeline import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question.")
    parser.add_argument("question", type=str, help="The question to ask.")
    parser.add_argument("--k", type=int, default=None, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    setup_observability()
    answer = answer_question(AskRequest(question=args.question, k=args.k))

    print("\n" + "=" * 70)
    print(f"QUESTION : {args.question}")
    print("=" * 70)
    print(f"\nANSWER ({answer.used_tool}):\n{answer.answer}\n")
    print("-" * 70)
    print(f"SOURCES ({len(answer.sources)}):")
    for src in dict.fromkeys(answer.sources):  # de-duplicated, order preserved
        print(f"  - {src}")


if __name__ == "__main__":
    main()
