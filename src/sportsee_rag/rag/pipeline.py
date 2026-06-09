"""RAG pipeline: the question -> answer flow, extracted from Streamlit.

In the prototype this logic was trapped inside ``MistralChat.py`` (search ->
format context -> call the LLM -> write to the UI), with no return value, so it
could not be called from a script or evaluated. Here it becomes a plain,
testable function returning a typed ``RagAnswer`` — the contract RAGAS and the
Phase 3 agent both consume.

The retrieval, the system prompt and the single-user-message construction are
kept verbatim from the prototype: only the SDK and the I/O typing changed.
"""

from __future__ import annotations

import logging

from ..config import Settings, get_settings
from ..llm.client import MistralLLM
from ..models import AskRequest, RagAnswer
from ..retrieval.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# System prompt — kept verbatim from the prototype (MistralChat.py). Unchanged
# on purpose so the RAG behaviour stays recognisable for the before/after eval.
SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""

_NO_CONTEXT = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."


def answer_question(
    request: AskRequest,
    *,
    manager: VectorStoreManager | None = None,
    llm: MistralLLM | None = None,
    settings: Settings | None = None,
) -> RagAnswer:
    """Run the RAG flow for one question and return a typed answer.

    Dependencies are injectable so tests can pass mocks (no network, no key).
    """
    settings = settings or get_settings()
    manager = manager or VectorStoreManager(settings=settings)
    llm = llm or manager._llm  # reuse the manager's client by default

    # 1. Retrieve context.
    results = manager.search(request.question, k=request.k)

    # 2. Format the context block (verbatim prototype formatting).
    if results:
        context_str = "\n\n---\n\n".join(
            f"Source: {r.metadata.get('source', 'Inconnue')} (Score: {r.score:.1f}%)\n"
            f"Contenu: {r.text}"
            for r in results
        )
    else:
        context_str = _NO_CONTEXT
        logger.warning("No context found for query: %s", request.question)

    # 3. Build the single user message and call the LLM (prompt unchanged).
    prompt = SYSTEM_PROMPT.format(context_str=context_str, question=request.question)
    answer = llm.chat([{"role": "user", "content": prompt}], temperature=0.1)

    # 4. Return a typed answer (contexts feed RAGAS directly).
    return RagAnswer(
        answer=answer,
        contexts=[r.text for r in results],
        sources=[r.metadata.get("source", "Inconnue") for r in results],
        used_tool="rag",
    )
